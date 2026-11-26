/* pocl-air-gen.cc - lowering of PoCL's LLVM IR to Apple's AIR, and packaging
   of the result as a Metal library

   Copyright (c) 2026 PoCL developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

/* Apple doesn't document AIR (the LLVM-bitcode-based IR in Metal libraries)
   nor the metallib container. The conventions implemented here follow the
   ones reverse-engineered by Metal.jl / GPUCompiler.jl
   (https://github.com/JuliaGPU/GPUCompiler.jl/blob/master/src/metal.jl and
   https://github.com/JuliaGPU/Metal.jl/tree/main/src/metallib):

   - AIR uses address spaces 0 (thread), 1 (device), 2 (constant) and
     3 (threadgroup), the same numbering as the spir64 target PoCL compiles
     Metal kernels for. There is no generic address space.
   - Work-item coordinates are kernel arguments tagged with metadata
     (air.thread_position_in_grid etc.) rather than intrinsics.
   - Kernel arguments are buffers; values are passed by reference, through
     constant-address-space pointers.
   - Math, integer and atomic operations are calls to air.* functions.
   - Metal's library loader only reads LLVM <= 15 bitcode, so the module is
     converted with libllvm_downgrade (https://github.com/JuliaLLVM/llvm-downgrade).
*/

#include "config.h"

#include "pocl-air-gen.h"

#include "pocl_cl.h"
#include "pocl_debug.h"
#include "pocl_llvm_api.h"
#include "pocl_runtime_config.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/ReplaceConstant.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/StripDeadPrototypes.h>
#include <llvm/Transforms/Scalar/InferAddressSpaces.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/LowerAtomic.h>

#include <llvm-downgrade.h>

#include <CommonCrypto/CommonDigest.h>

#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace llvm;

namespace {

enum AirAddrSpace : unsigned {
  AS_THREAD = 0,
  AS_DEVICE = 1,
  AS_CONSTANT = 2,
  AS_THREADGROUP = 3,
};

/* MSL mem_flags */
enum : unsigned {
  MEM_NONE = 0,
  MEM_DEVICE = 1,
  MEM_THREADGROUP = 2,
  MEM_TEXTURE = 4,
};

/* MSL thread_scope */
enum : unsigned {
  SCOPE_THREAD = 0,
  SCOPE_THREADGROUP = 1,
  SCOPE_DEVICE = 2,
  SCOPE_SIMDGROUP = 4,
};

const char *AirDataLayout =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"
    "-f32:32:32-f64:64:64"
    "-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128"
    "-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
    "-n8:16:32";

const char *LocalRegionName = "__pocl_air_local_region";

/* The work-item query placeholder variables of the kernel library
   (lib/kernel/metal/air_workitem.h), and the AIR kernel input arguments
   (all uint3) that implement them. */
struct AirInput {
  const char *Placeholder;
  const char *AirName;
};
const char *LaunchParamsPlaceholder = "_pocl_air_launch_params";
const char *PrintfBufferPlaceholder = "_pocl_air_printf_buffer";
const AirInput AirInputs[] = {
    {"_pocl_air_thread_position_in_grid", "thread_position_in_grid"},
    {"_pocl_air_thread_position_in_threadgroup",
     "thread_position_in_threadgroup"},
    {"_pocl_air_threadgroup_position_in_grid", "threadgroup_position_in_grid"},
    {"_pocl_air_threads_per_threadgroup", "threads_per_threadgroup"},
    {"_pocl_air_threadgroups_per_grid", "threadgroups_per_grid"},
    {"_pocl_air_threads_per_grid", "threads_per_grid"},
};

enum class ArgKind { Buffer, Value, ByVal, Local };

struct KernelArg {
  ArgKind Kind;
  /* for Value / ByVal: the value type; for Buffer: unused */
  Type *ValTy = nullptr;
  unsigned AddrSpace = 0;
  std::string Name;
  /* MSL-ish type name for reflection; for buffers, of the pointee */
  std::string TypeName;
  uint64_t TypeSize = 1;
  uint64_t TypeAlign = 1;
};

struct KernelInfo {
  std::vector<KernelArg> Args;
  /* indices into AirInputs, in the order of the trailing parameters */
  std::vector<unsigned> Inputs;
};

class AirLowering {
public:
  AirLowering(Module &M, const pocl_metal_air_target_t &T)
      : M(M), Ctx(M.getContext()), T(T), I32(Type::getInt32Ty(Ctx)),
        I1(Type::getInt1Ty(Ctx)) {}

  bool run(unsigned &LocalRegionSize);
  const std::string &error() const { return Err; }
  /* program-scope variable storage */
  uint64_t GVarSize = 0;
  std::vector<uint8_t> GVarInit;
  std::vector<pocl_metal_gvar_reloc_t> GVarRelocs;

private:
  Module &M;
  LLVMContext &Ctx;
  const pocl_metal_air_target_t &T;
  Type *I32, *I1;
  std::string Err;
  std::vector<Function *> Kernels;
  std::map<Function *, KernelInfo> Infos;
  GlobalVariable *LocalRegion = nullptr;
  /* program-scope variables and their offsets in the storage buffer */
  std::vector<std::pair<GlobalVariable *, uint64_t>> GVars;

  bool fail(const Twine &Msg) {
    if (Err.empty())
      Err = Msg.str();
    return false;
  }

  bool collectKernels();
  void normalizeCallingConventions();
  bool inlineEverything();
  bool runPipeline(StringRef Pipeline);
  bool moveConstantGlobals();
  bool checkGlobals(uint64_t &StaticLocalSize);
  bool collectProgramScopeVariables();
  bool writeConstant(Constant *C, uint64_t Offset);
  bool rewriteKernel(Function *K, uint64_t RegionSize);
  KernelArg classifyArg(Function *K, Argument &A);
  bool lowerSyncPlaceholders();
  bool lowerAtomics();
  bool lowerIntrinsics();
  void lowerUnreachables();
  void stripMetadata();
  void addAirMetadata();
  FunctionCallee airFunction(StringRef Name, Type *Ret, ArrayRef<Type *> Args,
                             bool Convergent = false);

public:
  FunctionCallee airFunctionPublic(StringRef Name, Type *Ret,
                                   ArrayRef<Type *> Args) {
    return airFunction(Name, Ret, Args);
  }

private:
  void emitFence(IRBuilder<> &B, unsigned Flags, unsigned Scope);
};

std::string typeSuffix(Type *Ty) {
  if (auto *VT = dyn_cast<FixedVectorType>(Ty))
    return "v" + std::to_string(VT->getNumElements()) +
           typeSuffix(VT->getElementType());
  if (Ty->isIntegerTy())
    return "i" + std::to_string(Ty->getIntegerBitWidth());
  if (Ty->isHalfTy())
    return "f16";
  if (Ty->isBFloatTy())
    return "bf16";
  if (Ty->isFloatTy())
    return "f32";
  if (Ty->isDoubleTy())
    return "f64";
  return "unknown";
}

/* A reflection type name derived from an OpenCL kernel_arg_type string. MSL
   names for the common scalar and vector types, an identifier otherwise. */
std::string sanitizeTypeName(StringRef Name) {
  std::string S = Name.str();
  /* drop pointer and qualifiers */
  while (!S.empty() && (S.back() == '*' || S.back() == ' '))
    S.pop_back();
  for (const char *Q : {"const ", "volatile ", "restrict ", "struct ",
                        "union ", "enum "}) {
    size_t P;
    while ((P = S.find(Q)) != std::string::npos)
      S.erase(P, strlen(Q));
  }
  if (S.empty())
    return "uchar";
  std::string R;
  for (char C : S)
    R += (isalnum((unsigned char)C) || C == '_') ? C : '_';
  if (isdigit((unsigned char)R[0]))
    R = "_" + R;
  return R;
}

/* the size of an OpenCL C scalar or vector type, 0 if unknown */
uint64_t clTypeSize(StringRef Name) {
  static const std::pair<const char *, unsigned> Scalars[] = {
      {"char", 1},  {"uchar", 1}, {"bool", 1},  {"short", 2},
      {"ushort", 2}, {"half", 2},  {"int", 4},   {"uint", 4},
      {"float", 4}, {"long", 8},  {"ulong", 8}, {"double", 8}};
  for (auto &P : Scalars) {
    StringRef SN(P.first);
    if (!Name.starts_with(SN))
      continue;
    StringRef Rest = Name.drop_front(SN.size());
    if (Rest.empty())
      return P.second;
    unsigned N;
    if (!Rest.getAsInteger(10, N))
      return P.second * (N == 3 ? 4 : N);
  }
  return 0;
}

bool AirLowering::collectKernels() {
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    /* same criteria as pocl::isKernelToProcess(), which libpocl doesn't
       export */
    bool IsKernel = (F.getMetadata("kernel_arg_access_qual") != nullptr &&
                     F.getMetadata("pocl_generated") == nullptr) ||
                    F.getCallingConv() == CallingConv::SPIR_KERNEL;
    if (IsKernel)
      Kernels.push_back(&F);
  }
  if (Kernels.empty())
    return fail("the program contains no kernels");
  return true;
}

/* The spir64 target uses the spir_func/spir_kernel calling conventions; AIR
   uses the default one. */
void AirLowering::normalizeCallingConventions() {
  for (Function &F : M) {
    F.setCallingConv(CallingConv::C);
    for (Instruction &I : instructions(F))
      if (auto *CB = dyn_cast<CallBase>(&I))
        CB->setCallingConv(CallingConv::C);
  }
}

bool AirLowering::runPipeline(StringRef Pipeline) {
  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;
  if (auto E = PB.parsePassPipeline(MPM, Pipeline)) {
    return fail("could not parse pass pipeline '" + Pipeline +
                "': " + toString(std::move(E)));
  }
  MPM.run(M, MAM);
  return true;
}

/* AIR supports function calls in principle, but keep the kernels
   self-contained: this lets the work-item query placeholders be replaced by
   the kernel's own input arguments. */
bool AirLowering::inlineEverything() {
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    F.removeFnAttr(Attribute::NoInline);
    F.removeFnAttr(Attribute::OptimizeNone);
    if (std::find(Kernels.begin(), Kernels.end(), &F) == Kernels.end())
      F.addFnAttr(Attribute::AlwaysInline);
  }
  if (!runPipeline("always-inline"))
    return false;

  /* drop the now unused non-kernel functions */
  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (Function &F : make_early_inc_range(M)) {
      if (F.isDeclaration() ||
          std::find(Kernels.begin(), Kernels.end(), &F) != Kernels.end())
        continue;
      F.removeDeadConstantUsers();
      if (F.use_empty()) {
        F.eraseFromParent();
        Changed = true;
      }
    }
  }
  for (Function &F : M) {
    if (F.isDeclaration() ||
        std::find(Kernels.begin(), Kernels.end(), &F) != Kernels.end())
      continue;
    return fail("function '" + F.getName() +
                "' could not be inlined into the kernels (recursion and "
                "function pointers are not supported by the Metal driver)");
  }
  return true;
}

/* Constant data in the thread address space (e.g. lookup tables from C
   sources of the kernel library) has to live in the constant address
   space in AIR. */
bool AirLowering::moveConstantGlobals() {
  /* Constant data in the thread or device address spaces (e.g. lookup
     tables from C sources of the kernel library, or the "(null)" string of
     printf) has to live in the constant address space in AIR. */
  for (unsigned AS : {(unsigned)AS_THREAD, (unsigned)AS_DEVICE}) {
    bool Moved = false;
    for (GlobalVariable &G : make_early_inc_range(M.globals())) {
      if (G.getAddressSpace() != AS || !G.isConstant() || !G.hasInitializer())
        continue;
      auto *NG = new GlobalVariable(
          M, G.getValueType(), true, GlobalValue::InternalLinkage,
          G.getInitializer(), G.getName() + ".const", nullptr,
          GlobalValue::NotThreadLocal, AS_CONSTANT);
      NG->setAlignment(G.getAlign());
      G.replaceAllUsesWith(ConstantExpr::getAddrSpaceCast(NG, G.getType()));
      G.eraseFromParent();
      Moved = true;
    }
    if (!Moved)
      continue;
    /* Propagate the constant address space to the users, treating AS as
       the "flat" one for this purpose: only pointers derived from the
       address space casts are rewritten (not allocas or kernel arguments). */
    FunctionPassManager FPM;
    FPM.addPass(InferAddressSpacesPass(AS));
    ModulePassManager MPM;
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    PassBuilder PB;
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    MPM.run(M, MAM);
  }
  return true;
}

bool AirLowering::checkGlobals(uint64_t &StaticLocalSize) {
  const DataLayout &DL = M.getDataLayout();
  StaticLocalSize = 0;
  for (GlobalVariable &G : M.globals()) {
    G.removeDeadConstantUsers();
    if (G.use_empty() || G.getName().starts_with("_pocl_air_"))
      continue;
    bool IsGVar = false;
    for (auto &GV : GVars)
      IsGVar |= GV.first == &G;
    if (IsGVar)
      continue;
    switch (G.getAddressSpace()) {
    case AS_CONSTANT:
      break;
    case AS_THREADGROUP:
      StaticLocalSize = alignTo(StaticLocalSize, G.getAlign().valueOrOne());
      StaticLocalSize += DL.getTypeAllocSize(G.getValueType());
      break;
    default:
      return fail("program-scope variable '" + G.getName() +
                  "' in address space " + Twine(G.getAddressSpace()) +
                  " is not supported by the Metal driver");
    }
  }
  if (StaticLocalSize > T.local_mem_size)
    return fail("the kernels use " + Twine(StaticLocalSize) +
                " bytes of __local memory, more than the " +
                Twine(T.local_mem_size) + " bytes available");
  return true;
}

/* Serializes the initializer C of a program-scope variable to GVarInit at
   Offset. */
bool AirLowering::writeConstant(Constant *C, uint64_t Offset) {
  const DataLayout &DL = M.getDataLayout();
  Type *Ty = C->getType();
  auto WriteInt = [&](const APInt &V) {
    unsigned Bytes = DL.getTypeStoreSize(Ty);
    for (unsigned I = 0; I < Bytes; ++I)
      GVarInit[Offset + I] = V.extractBitsAsZExtValue(8, 8 * I);
  };
  if (isa<ConstantAggregateZero>(C) || isa<UndefValue>(C) ||
      isa<ConstantPointerNull>(C))
    return true; /* already zero */
  if (Ty->isPointerTy()) {
    /* a pointer to (an element of) a program-scope variable: relocated by
       the host */
    int64_t ConstOffset = 0;
    Value *Base = GetPointerBaseWithConstantOffset(C, ConstOffset, DL);
    for (auto &GV : GVars)
      if (GV.first == Base && DL.getPointerSize() == 8) {
        GVarRelocs.push_back(
            {(unsigned)Offset, (unsigned)(GV.second + ConstOffset)});
        return true;
      }
    return fail("unsupported initializer of a program-scope variable: a "
                "pointer to something else than a program-scope variable");
  }
  /* (vector splats can be ConstantInt / ConstantFP too: handled below) */
  if (auto *CI = dyn_cast<ConstantInt>(C); CI && !Ty->isVectorTy()) {
    if (CI->getBitWidth() % 8 != 0)
      return fail("unsupported initializer of a program-scope variable");
    WriteInt(CI->getValue());
    return true;
  }
  if (auto *CF = dyn_cast<ConstantFP>(C); CF && !Ty->isVectorTy()) {
    WriteInt(CF->getValueAPF().bitcastToAPInt());
    return true;
  }
  if (auto *ST = dyn_cast<StructType>(Ty)) {
    const StructLayout *SL = DL.getStructLayout(ST);
    for (unsigned I = 0; I < ST->getNumElements(); ++I)
      if (!writeConstant(C->getAggregateElement(I),
                         Offset + SL->getElementOffset(I)))
        return false;
    return true;
  }
  if (isa<ArrayType>(Ty) || isa<FixedVectorType>(Ty)) {
    Type *ElTy = isa<ArrayType>(Ty) ? Ty->getArrayElementType()
                                    : cast<FixedVectorType>(Ty)->getElementType();
    uint64_t N = isa<ArrayType>(Ty)
                     ? Ty->getArrayNumElements()
                     : cast<FixedVectorType>(Ty)->getNumElements();
    uint64_t Stride = DL.getTypeAllocSize(ElTy);
    for (uint64_t I = 0; I < N; ++I) {
      Constant *El = C->getAggregateElement(I);
      if (El == nullptr)
        return fail("unsupported initializer of a program-scope variable");
      if (!writeConstant(El, Offset + I * Stride))
        return false;
    }
    return true;
  }
  return fail("unsupported initializer of a program-scope variable");
}

/* Program-scope __global variables live in a per-program buffer that's
   passed to every kernel. */
bool AirLowering::collectProgramScopeVariables() {
  const DataLayout &DL = M.getDataLayout();
  SmallVector<Constant *, 8> Vars;
  for (GlobalVariable &G : M.globals()) {
    G.removeDeadConstantUsers();
    if (G.getAddressSpace() != AS_DEVICE || G.use_empty() ||
        G.getName().starts_with("_pocl_air_"))
      continue;
    Type *Ty = G.getValueType();
    uint64_t A = std::max<uint64_t>(G.getAlign().valueOrOne().value(),
                                    DL.getABITypeAlign(Ty).value());
    GVarSize = alignTo(GVarSize, A);
    GVars.push_back({&G, GVarSize});
    GVarSize += DL.getTypeAllocSize(Ty);
    Vars.push_back(&G);
  }
  if (GVars.empty())
    return true;
  GVarInit.assign(GVarSize, 0);
  for (auto &GV : GVars)
    if (GV.first->hasInitializer() &&
        !writeConstant(GV.first->getInitializer(), GV.second))
      return false;
  convertUsersOfConstantsToInstructions(Vars);
  return true;
}

static MDNode *getKernelMD(Function *K, StringRef Name) {
  return K->getMetadata(Name);
}

static std::string getMDString(MDNode *N, unsigned I) {
  if (N == nullptr || I >= N->getNumOperands())
    return "";
  if (auto *S = dyn_cast_or_null<MDString>(N->getOperand(I)))
    return S->getString().str();
  return "";
}

KernelArg AirLowering::classifyArg(Function *K, Argument &A) {
  const DataLayout &DL = M.getDataLayout();
  KernelArg KA;
  unsigned I = A.getArgNo();
  KA.Name = getMDString(getKernelMD(K, "kernel_arg_name"), I);
  if (KA.Name.empty())
    KA.Name = A.hasName() ? A.getName().str() : ("arg" + std::to_string(I));
  std::string CLType = getMDString(getKernelMD(K, "kernel_arg_type"), I);

  Type *Ty = A.getType();
  if (Ty->isPointerTy() && A.hasByValAttr()) {
    KA.Kind = ArgKind::ByVal;
    KA.ValTy = A.getParamByValType();
  } else if (Ty->isPointerTy() &&
             Ty->getPointerAddressSpace() == AS_THREADGROUP) {
    KA.Kind = ArgKind::Local;
    KA.ValTy = I32;
    KA.TypeName = "uint";
    KA.TypeSize = KA.TypeAlign = 4;
    return KA;
  } else if (Ty->isPointerTy()) {
    KA.Kind = ArgKind::Buffer;
    KA.AddrSpace = Ty->getPointerAddressSpace();
    StringRef Pointee = StringRef(CLType).rtrim("* ");
    KA.TypeName = sanitizeTypeName(Pointee);
    uint64_t S = clTypeSize(KA.TypeName);
    if (S == 0) {
      KA.TypeName = "uchar";
      S = 1;
    }
    KA.TypeSize = KA.TypeAlign = S;
    return KA;
  } else {
    KA.Kind = ArgKind::Value;
    KA.ValTy = Ty;
  }
  KA.TypeName = CLType.empty() ? std::string("uchar")
                               : sanitizeTypeName(CLType);
  KA.TypeSize = DL.getTypeAllocSize(KA.ValTy);
  KA.TypeAlign = DL.getABITypeAlign(KA.ValTy).value();
  return KA;
}

/* Rewrites a kernel into the AIR calling convention described in
   pocl-air-gen.h. */
bool AirLowering::rewriteKernel(Function *K, uint64_t RegionSize) {
  KernelInfo Info;
  for (Argument &A : K->args())
    Info.Args.push_back(classifyArg(K, A));

  /* the work-item inputs the kernel uses */
  auto UsedIn = [&](GlobalVariable *G) {
    if (G == nullptr)
      return false;
    for (User *U : G->users())
      if (auto *I = dyn_cast<Instruction>(U))
        if (I->getFunction() == K)
          return true;
    return false;
  };
  for (unsigned I = 0; I < sizeof(AirInputs) / sizeof(AirInputs[0]); ++I)
    if (UsedIn(M.getNamedGlobal(AirInputs[I].Placeholder)))
      Info.Inputs.push_back(I);
  GlobalVariable *LaunchPlaceholder = M.getNamedGlobal(LaunchParamsPlaceholder);
  bool UsesLaunchParams = UsedIn(LaunchPlaceholder);
  GlobalVariable *PrintfPlaceholder = M.getNamedGlobal(PrintfBufferPlaceholder);
  bool UsesPrintf = UsedIn(PrintfPlaceholder);

  Type *ConstPtr = PointerType::get(Ctx, AS_CONSTANT);
  Type *UInt3 = FixedVectorType::get(I32, 3);
  SmallVector<Type *, 16> Params;
  for (KernelArg &KA : Info.Args)
    Params.push_back(KA.Kind == ArgKind::Buffer
                         ? PointerType::get(Ctx, KA.AddrSpace)
                         : ConstPtr);
  Params.push_back(ConstPtr); /* launch parameters */
  Params.push_back(PointerType::get(Ctx, AS_DEVICE)); /* printf buffer */
  Params.push_back(PointerType::get(Ctx, AS_DEVICE)); /* program variables */
  for (unsigned I = 0; I < Info.Inputs.size(); ++I)
    Params.push_back(UInt3);

  /* Metal allows 31 buffer bindings */
  if (Info.Args.size() > POCL_METAL_MAX_KERNEL_ARGS)
    return fail("kernel '" + K->getName() + "' has " +
                Twine(Info.Args.size()) + " arguments; the Metal driver "
                "supports at most " + Twine(POCL_METAL_MAX_KERNEL_ARGS));

  bool HasLocalArgs = false;
  for (KernelArg &KA : Info.Args)
    HasLocalArgs |= KA.Kind == ArgKind::Local;
  if (HasLocalArgs && LocalRegion == nullptr) {
    if (RegionSize == 0)
      return fail("no __local memory left for __local kernel arguments");
    auto *ATy = ArrayType::get(Type::getInt8Ty(Ctx), RegionSize);
    LocalRegion = new GlobalVariable(
        M, ATy, false, GlobalValue::InternalLinkage, UndefValue::get(ATy),
        LocalRegionName, nullptr, GlobalValue::NotThreadLocal,
        AS_THREADGROUP);
    LocalRegion->setAlignment(Align(POCL_METAL_LOCAL_ARG_ALIGN));
  }

  auto *FTy = FunctionType::get(Type::getVoidTy(Ctx), Params, false);
  Function *NF = Function::Create(FTy, GlobalValue::ExternalLinkage,
                                  K->getAddressSpace(), "", &M);
  NF->takeName(K);
  NF->setAttributes(AttributeList::get(
      Ctx, K->getAttributes().getFnAttrs(), AttributeSet(), {}));
  NF->setCallingConv(CallingConv::C);
  NF->splice(NF->begin(), K);

  IRBuilder<> B(&*NF->getEntryBlock().getFirstInsertionPt());
  const DataLayout &DL = M.getDataLayout();
  for (unsigned I = 0; I < Info.Args.size(); ++I) {
    KernelArg &KA = Info.Args[I];
    Argument *Old = K->getArg(I);
    Argument *New = NF->getArg(I);
    New->setName(KA.Name);
    Value *Repl = nullptr;
    switch (KA.Kind) {
    case ArgKind::Buffer: {
      /* NULL buffer arguments are flagged in the launch parameters */
      Argument *LP = NF->getArg(Info.Args.size());
      Value *Mask = B.CreateAlignedLoad(
          I32, B.CreateConstInBoundsGEP1_32(I32, LP, 4), Align(4),
          "null_args");
      Value *IsNull = B.CreateICmpNE(
          B.CreateAnd(Mask, B.getInt32(1u << I)), B.getInt32(0));
      Repl = B.CreateSelect(
          IsNull, ConstantPointerNull::get(cast<PointerType>(New->getType())),
          New, KA.Name);
      break;
    }
    case ArgKind::Value:
      Repl = B.CreateAlignedLoad(KA.ValTy, New, DL.getABITypeAlign(KA.ValTy),
                                 KA.Name);
      break;
    case ArgKind::ByVal: {
      AllocaInst *Copy = B.CreateAlloca(KA.ValTy, nullptr, KA.Name + ".copy");
      Copy->setAlignment(DL.getPrefTypeAlign(KA.ValTy));
      B.CreateMemCpy(Copy, Copy->getAlign(), New,
                     DL.getABITypeAlign(KA.ValTy),
                     DL.getTypeAllocSize(KA.ValTy));
      Repl = Copy;
      break;
    }
    case ArgKind::Local: {
      Value *Offset = B.CreateAlignedLoad(I32, New, Align(4),
                                          KA.Name + ".offset");
      Repl = B.CreateInBoundsGEP(Type::getInt8Ty(Ctx), LocalRegion,
                                 B.CreateZExt(Offset, B.getInt64Ty()),
                                 KA.Name);
      break;
    }
    }
    Old->replaceAllUsesWith(Repl);
  }

  Argument *Launch = NF->getArg(Info.Args.size());
  Launch->setName("_pocl_launch_params");
  Argument *PrintfBuf = NF->getArg(Info.Args.size() + 1);
  PrintfBuf->setName("_pocl_printf_buffer");
  Argument *GVarBuf = NF->getArg(Info.Args.size() + 2);
  GVarBuf->setName("_pocl_gvar_buffer");
  for (unsigned I = 0; I < Info.Inputs.size(); ++I)
    NF->getArg(Info.Args.size() + 3 + I)
        ->setName(AirInputs[Info.Inputs[I]].AirName);

  /* Replace the placeholder variables with private copies of the kernel's
     input arguments / launch parameters; SROA turns these into plain
     values. */
  B.SetInsertPoint(&*NF->getEntryBlock().getFirstInsertionPt());
  auto Privatize = [&](GlobalVariable *G, Value *V) {
    AllocaInst *A = B.CreateAlloca(G->getValueType(), nullptr, G->getName());
    A->setAlignment(std::max(G->getAlign().valueOrOne(),
                             DL.getPrefTypeAlign(G->getValueType())));
    B.CreateAlignedStore(V, A, A->getAlign());
    G->replaceUsesWithIf(A, [&](Use &U) {
      auto *I = dyn_cast<Instruction>(U.getUser());
      return I && I->getFunction() == NF;
    });
  };
  for (unsigned I = 0; I < Info.Inputs.size(); ++I) {
    GlobalVariable *G = M.getNamedGlobal(AirInputs[Info.Inputs[I]].Placeholder);
    Value *V = NF->getArg(Info.Args.size() + 3 + I);
    if (G->getValueType() != V->getType())
      return fail("unexpected type of " + G->getName());
    Privatize(G, V);
  }
  if (UsesLaunchParams) {
    GlobalVariable *G = LaunchPlaceholder;
    /* { work_dim, global_offset[3] } */
    if (DL.getTypeAllocSize(G->getValueType()) != 4 * sizeof(unsigned))
      return fail("unexpected type of " + G->getName());
    Value *V = B.CreateAlignedLoad(FixedVectorType::get(I32, 4), Launch,
                                   Align(4));
    Privatize(G, V);
  }
  if (UsesPrintf) {
    if (!PrintfPlaceholder->getValueType()->isPointerTy())
      return fail("unexpected type of " + PrintfPlaceholder->getName());
    Privatize(PrintfPlaceholder, PrintfBuf);
  }

  /* program-scope variables: their place in the storage buffer */
  for (auto &GV : GVars) {
    GlobalVariable *G = GV.first;
    bool Used = false;
    for (User *U : G->users())
      if (auto *I = dyn_cast<Instruction>(U))
        Used |= I->getFunction() == NF;
    if (!Used)
      continue;
    Value *P = B.CreateConstInBoundsGEP1_64(Type::getInt8Ty(Ctx), GVarBuf,
                                            GV.second, G->getName());
    G->replaceUsesWithIf(P, [&](Use &U) {
      auto *I = dyn_cast<Instruction>(U.getUser());
      return I && I->getFunction() == NF;
    });
  }

  for (Argument &A : K->args())
    A.replaceAllUsesWith(PoisonValue::get(A.getType()));
  K->replaceAllUsesWith(PoisonValue::get(K->getType()));
  K->eraseFromParent();

  Infos[NF] = std::move(Info);
  return true;
}

FunctionCallee AirLowering::airFunction(StringRef Name, Type *Ret,
                                        ArrayRef<Type *> Args,
                                        bool Convergent) {
  FunctionType *FTy = FunctionType::get(Ret, Args, false);
  FunctionCallee FC = M.getOrInsertFunction(Name, FTy);
  if (auto *F = dyn_cast<Function>(FC.getCallee())) {
    F->addFnAttr(Attribute::NoUnwind);
    F->addFnAttr(Attribute::WillReturn);
    F->addFnAttr(Attribute::MustProgress);
    if (Convergent)
      F->addFnAttr(Attribute::Convergent);
  }
  return FC;
}

void AirLowering::emitFence(IRBuilder<> &B, unsigned Flags, unsigned Scope) {
  /* before MSL 4.1, fences are always sequentially consistent */
  FunctionCallee Fence =
      airFunction("air.atomic.fence", B.getVoidTy(), {I32, I32, I32});
  B.CreateCall(Fence, {B.getInt32(Flags), B.getInt32(5), B.getInt32(Scope)});
}

bool AirLowering::lowerSyncPlaceholders() {
  /* The kernel library's barriers call pocl.barrier(); order both device and
     threadgroup memory, as the OpenCL fence flags are not passed through. */
  if (Function *Barrier = M.getFunction("pocl.barrier")) {
    for (User *U : make_early_inc_range(Barrier->users())) {
      auto *CB = dyn_cast<CallBase>(U);
      if (CB == nullptr)
        return fail("unexpected use of pocl.barrier");
      IRBuilder<> B(CB);
      unsigned Flags = MEM_DEVICE | MEM_THREADGROUP;
      if (T.air_major > 2 || (T.air_major == 2 && T.air_minor >= 9)) {
        FunctionCallee F = airFunction("air.wg.barrier", B.getVoidTy(),
                                       {I32, I32, I32}, true);
        B.CreateCall(F, {B.getInt32(Flags), B.getInt32(5),
                         B.getInt32(SCOPE_THREADGROUP)});
      } else {
        FunctionCallee F =
            airFunction("air.wg.barrier", B.getVoidTy(), {I32, I32}, true);
        B.CreateCall(F, {B.getInt32(Flags), B.getInt32(SCOPE_THREADGROUP)});
      }
      CB->eraseFromParent();
    }
    Barrier->eraseFromParent();
  }

  for (GlobalVariable &G : make_early_inc_range(M.globals())) {
    if (!G.getName().starts_with("_pocl_air_"))
      continue;
    G.removeDeadConstantUsers();
    if (!G.use_empty())
      return fail("unexpected use of '" + G.getName() +
                  "' outside of a kernel");
    G.eraseFromParent();
  }
  return true;
}

/* LLVM sync scope -> MSL thread scope */
static unsigned metalScope(LLVMContext &Ctx, SyncScope::ID SSID,
                           unsigned AddrSpace) {
  unsigned Scope = SCOPE_DEVICE;
  if (SSID == SyncScope::SingleThread) {
    Scope = SCOPE_THREAD;
  } else if (SSID != SyncScope::System) {
    SmallVector<StringRef, 8> Names;
    Ctx.getSyncScopeNames(Names);
    StringRef Name = SSID < Names.size() ? Names[SSID] : "";
    if (Name == "workgroup")
      Scope = SCOPE_THREADGROUP;
    else if (Name == "subgroup" || Name == "wavefront")
      Scope = SCOPE_SIMDGROUP;
    else if (Name == "singlethread")
      Scope = SCOPE_THREAD;
  }
  /* threadgroup memory is never shared beyond the threadgroup */
  if (AddrSpace == AS_THREADGROUP && Scope == SCOPE_DEVICE)
    Scope = SCOPE_THREADGROUP;
  return Scope;
}

static bool isOrdered(AtomicOrdering O) {
  return O != AtomicOrdering::NotAtomic && O != AtomicOrdering::Unordered &&
         O != AtomicOrdering::Monotonic;
}

bool AirLowering::lowerAtomics() {
  bool NewAirAtomics = T.air_major > 2 || (T.air_major == 2 && T.air_minor >= 9);

  /* 1) expand the read-modify-write operations AIR has no function for to
        compare-exchange loops */
  SmallVector<AtomicRMWInst *, 8> ToExpand;
  for (Function &F : M)
    for (Instruction &I : instructions(F))
      if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
        Type *Ty = RMW->getValOperand()->getType();
        bool Supported = false;
        switch (RMW->getOperation()) {
        case AtomicRMWInst::Xchg:
          Supported = Ty->isIntegerTy(32) || Ty->isFloatTy();
          break;
        case AtomicRMWInst::Add:
        case AtomicRMWInst::Sub:
        case AtomicRMWInst::And:
        case AtomicRMWInst::Or:
        case AtomicRMWInst::Xor:
        case AtomicRMWInst::Max:
        case AtomicRMWInst::Min:
        case AtomicRMWInst::UMax:
        case AtomicRMWInst::UMin:
          Supported = Ty->isIntegerTy(32);
          break;
        case AtomicRMWInst::FAdd:
        case AtomicRMWInst::FSub:
          Supported = Ty->isFloatTy();
          break;
        default:
          break;
        }
        if (!Supported)
          ToExpand.push_back(RMW);
      }

  for (AtomicRMWInst *RMW : ToExpand) {
    Type *Ty = RMW->getValOperand()->getType();
    unsigned Bits = Ty->getPrimitiveSizeInBits();
    if (Bits != 32)
      return fail("unsupported " + Twine(Bits) +
                  "-bit atomic operation (only 32-bit atomics are "
                  "supported by the Metal driver)");
    Type *IntTy = Type::getIntNTy(Ctx, Bits);
    BasicBlock *Before = RMW->getParent();
    BasicBlock *After = Before->splitBasicBlock(RMW->getIterator(), "rmw.end");
    BasicBlock *Loop =
        BasicBlock::Create(Ctx, "rmw.loop", Before->getParent(), After);
    Before->getTerminator()->eraseFromParent();
    IRBuilder<> B(Before);
    LoadInst *Init = B.CreateAlignedLoad(Ty, RMW->getPointerOperand(),
                                         RMW->getAlign(), "rmw.init");
    Init->setVolatile(true);
    B.CreateBr(Loop);
    B.SetInsertPoint(Loop);
    PHINode *Loaded = B.CreatePHI(Ty, 2, "rmw.loaded");
    Loaded->addIncoming(Init, Before);
    Value *NewV =
        buildAtomicRMWValue(RMW->getOperation(), B, Loaded,
                            RMW->getValOperand());
    Value *Cmp = B.CreateBitCast(Loaded, IntTy);
    Value *New = B.CreateBitCast(NewV, IntTy);
    AtomicCmpXchgInst *CX = B.CreateAtomicCmpXchg(
        RMW->getPointerOperand(), Cmp, New, RMW->getAlign(),
        RMW->getOrdering(),
        AtomicCmpXchgInst::getStrongestFailureOrdering(RMW->getOrdering()),
        RMW->getSyncScopeID());
    Value *Success = B.CreateExtractValue(CX, 1);
    Value *OldV = B.CreateBitCast(B.CreateExtractValue(CX, 0), Ty);
    Loaded->addIncoming(OldV, Loop);
    B.CreateCondBr(Success, After, Loop);
    RMW->replaceAllUsesWith(OldV);
    RMW->eraseFromParent();
  }

  /* 2) select the air.atomic.* functions */
  SmallVector<Instruction *, 32> Atomics;
  for (Function &F : M)
    for (Instruction &I : instructions(F)) {
      if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I) ||
          isa<FenceInst>(I))
        Atomics.push_back(&I);
      else if (auto *L = dyn_cast<LoadInst>(&I); L && L->isAtomic())
        Atomics.push_back(&I);
      else if (auto *S = dyn_cast<StoreInst>(&I); S && S->isAtomic())
        Atomics.push_back(&I);
    }

  std::set<Function *> AtomicDecls;
  for (Instruction *I : Atomics) {
    IRBuilder<> B(I);
    if (auto *FI = dyn_cast<FenceInst>(I)) {
      emitFence(B, MEM_DEVICE | MEM_THREADGROUP,
                metalScope(Ctx, FI->getSyncScopeID(), AS_DEVICE));
      FI->eraseFromParent();
      continue;
    }

    Value *Ptr = getLoadStorePointerOperand(I);
    AtomicOrdering Order = AtomicOrdering::Monotonic;
    SyncScope::ID SSID = SyncScope::System;
    if (auto *RMW = dyn_cast<AtomicRMWInst>(I)) {
      Ptr = RMW->getPointerOperand();
      Order = RMW->getOrdering();
      SSID = RMW->getSyncScopeID();
    } else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(I)) {
      Ptr = CX->getPointerOperand();
      Order = CX->getSuccessOrdering();
      SSID = CX->getSyncScopeID();
    } else if (auto *L = dyn_cast<LoadInst>(I)) {
      Order = L->getOrdering();
      SSID = L->getSyncScopeID();
    } else if (auto *S = dyn_cast<StoreInst>(I)) {
      Order = S->getOrdering();
      SSID = S->getSyncScopeID();
    }

    unsigned AS = Ptr->getType()->getPointerAddressSpace();
    if (AS == AS_THREAD) {
      /* atomics on private memory: plain accesses */
      if (auto *L = dyn_cast<LoadInst>(I)) {
        L->setAtomic(AtomicOrdering::NotAtomic);
        continue;
      }
      if (auto *S = dyn_cast<StoreInst>(I)) {
        S->setAtomic(AtomicOrdering::NotAtomic);
        continue;
      }
      if (auto *RMW = dyn_cast<AtomicRMWInst>(I)) {
        Value *Old = B.CreateLoad(RMW->getType(), Ptr);
        B.CreateStore(buildAtomicRMWValue(RMW->getOperation(), B, Old,
                                          RMW->getValOperand()),
                      Ptr);
        RMW->replaceAllUsesWith(Old);
        RMW->eraseFromParent();
        continue;
      }
      auto *CX = cast<AtomicCmpXchgInst>(I);
      Value *Old = B.CreateLoad(CX->getNewValOperand()->getType(), Ptr);
      Value *Eq = B.CreateICmpEQ(Old, CX->getCompareOperand());
      B.CreateStore(B.CreateSelect(Eq, CX->getNewValOperand(), Old), Ptr);
      Value *R = B.CreateInsertValue(PoisonValue::get(CX->getType()), Old, 0);
      R = B.CreateInsertValue(R, Eq, 1);
      CX->replaceAllUsesWith(R);
      CX->eraseFromParent();
      continue;
    }
    if (AS != AS_DEVICE && AS != AS_THREADGROUP)
      return fail("atomic operation on unsupported address space " +
                  Twine(AS));

    const char *Mem = AS == AS_DEVICE ? "global" : "local";
    unsigned Scope = metalScope(Ctx, SSID, AS);
    unsigned FenceFlags = AS == AS_DEVICE ? MEM_DEVICE : MEM_THREADGROUP;
    bool Ordered = isOrdered(Order);
    /* targets before MSL 4.1 have relaxed atomics only: bracket ordered
       operations with fences */
    if (Ordered)
      emitFence(B, FenceFlags, Scope);

    SmallVector<Type *, 3> TrailTys;
    SmallVector<Value *, 3> Trail;
    if (NewAirAtomics) {
      TrailTys = {I32, I32, I1};
      Trail = {B.getInt32(Scope), B.getInt32(0), B.getInt1(true)};
    } else {
      TrailTys = {I32, I1};
      Trail = {B.getInt32(Scope), B.getInt1(true)};
    }
    Value *Relaxed = B.getInt32(0);
    Value *Result = nullptr;
    Type *ValTy = nullptr;
    Function *Decl = nullptr;

    auto declare = [&](const Twine &Name, Type *Ret,
                       SmallVector<Type *, 8> Args) {
      Args.append(TrailTys.begin(), TrailTys.end());
      FunctionCallee FC = airFunction(Name.str(), Ret, Args);
      Decl = cast<Function>(FC.getCallee());
      AtomicDecls.insert(Decl);
      return FC;
    };

    if (auto *L = dyn_cast<LoadInst>(I)) {
      ValTy = L->getType();
      Type *OpTy = ValTy->isFloatTy() ? I32 : ValTy;
      if (!OpTy->isIntegerTy(32))
        return fail("unsupported atomic load type");
      FunctionCallee FC =
          declare(Twine("air.atomic.") + Mem + ".load.i32", I32,
                  {Ptr->getType(), I32});
      SmallVector<Value *, 6> Args = {Ptr, Relaxed};
      Args.append(Trail.begin(), Trail.end());
      Result = B.CreateBitCast(B.CreateCall(FC, Args), ValTy);
    } else if (auto *S = dyn_cast<StoreInst>(I)) {
      Value *V = S->getValueOperand();
      if (V->getType()->isFloatTy())
        V = B.CreateBitCast(V, I32);
      if (!V->getType()->isIntegerTy(32))
        return fail("unsupported atomic store type");
      FunctionCallee FC =
          declare(Twine("air.atomic.") + Mem + ".store.i32", B.getVoidTy(),
                  {Ptr->getType(), I32, I32});
      SmallVector<Value *, 6> Args = {Ptr, V, Relaxed};
      Args.append(Trail.begin(), Trail.end());
      B.CreateCall(FC, Args);
    } else if (auto *RMW = dyn_cast<AtomicRMWInst>(I)) {
      ValTy = RMW->getType();
      Value *V = RMW->getValOperand();
      std::string Op;
      std::string Suffix = ValTy->isFloatTy() ? "f32" : "i32";
      switch (RMW->getOperation()) {
      case AtomicRMWInst::Xchg:
        Op = "xchg";
        if (ValTy->isFloatTy()) {
          V = B.CreateBitCast(V, I32);
          Suffix = "i32";
        }
        break;
      case AtomicRMWInst::Add: Op = "add.s"; break;
      case AtomicRMWInst::Sub: Op = "sub.s"; break;
      case AtomicRMWInst::And: Op = "and.s"; break;
      case AtomicRMWInst::Or: Op = "or.s"; break;
      case AtomicRMWInst::Xor: Op = "xor.s"; break;
      case AtomicRMWInst::Max: Op = "max.s"; break;
      case AtomicRMWInst::Min: Op = "min.s"; break;
      case AtomicRMWInst::UMax: Op = "max.u"; break;
      case AtomicRMWInst::UMin: Op = "min.u"; break;
      case AtomicRMWInst::FAdd: Op = "add"; break;
      case AtomicRMWInst::FSub: Op = "sub"; break;
      default:
        return fail("unexpected atomicrmw operation");
      }
      Type *OpTy = V->getType();
      FunctionCallee FC =
          declare(Twine("air.atomic.") + Mem + "." + Op + "." + Suffix, OpTy,
                  {Ptr->getType(), OpTy, I32});
      SmallVector<Value *, 6> Args = {Ptr, V, Relaxed};
      Args.append(Trail.begin(), Trail.end());
      Result = B.CreateBitCast(B.CreateCall(FC, Args), ValTy);
    } else {
      auto *CX = cast<AtomicCmpXchgInst>(I);
      Type *Ty = CX->getNewValOperand()->getType();
      if (!Ty->isIntegerTy(32))
        return fail("unsupported " + Twine(Ty->getPrimitiveSizeInBits()) +
                    "-bit atomic compare-exchange");
      /* AIR has only a weak compare-exchange, which takes the expected
         value by reference; like MSL, derive the success flag from the
         returned old value */
      IRBuilder<> EB(&*CX->getFunction()->getEntryBlock().getFirstInsertionPt());
      AllocaInst *Expected = EB.CreateAlloca(Ty, nullptr, "cmpxchg.expected");
      B.CreateStore(CX->getCompareOperand(), Expected);
      FunctionCallee FC =
          declare(Twine("air.atomic.") + Mem + ".cmpxchg.weak.i32", Ty,
                  {Ptr->getType(), Expected->getType(), Ty, I32, I32});
      SmallVector<Value *, 8> Args = {Ptr, Expected, CX->getNewValOperand(),
                                      Relaxed, Relaxed};
      Args.append(Trail.begin(), Trail.end());
      Value *Old = B.CreateCall(FC, Args);
      Value *Success = B.CreateICmpEQ(Old, CX->getCompareOperand());
      Value *R = B.CreateInsertValue(PoisonValue::get(CX->getType()), Old, 0);
      Result = B.CreateInsertValue(R, Success, 1);
    }

    if (Ordered) {
      B.SetInsertPoint(I->getNextNode());
      emitFence(B, FenceFlags, Scope);
    }
    if (Result)
      I->replaceAllUsesWith(Result);
    I->eraseFromParent();
  }

  /* tell the downgrader the element types of the pointer arguments of the
     atomic functions, which it can't infer from opaque pointers */
  for (Function *F : AtomicDecls) {
    Type *ElTy = F->getName().ends_with(".f32") ? Type::getFloatTy(Ctx) : I32;
    SmallVector<Metadata *, 4> MDs;
    for (unsigned I = 0; I < F->arg_size() && I < 2; ++I) {
      if (!F->getArg(I)->getType()->isPointerTy())
        continue;
      MDs.push_back(ConstantAsMetadata::get(ConstantInt::get(I32, I)));
      MDs.push_back(ConstantAsMetadata::get(Constant::getNullValue(ElTy)));
    }
    F->setMetadata("arg_eltypes", MDNode::get(Ctx, MDs));
  }
  return true;
}

/* Calls the scalar AIR function Name (e.g. "air.trunc") on each lane of
   the floating-point (vector) arguments. */
static Value *callAirScalarized(AirLowering &L, IRBuilder<> &B, StringRef Name,
                                ArrayRef<Value *> Args) {
  Type *Ty = Args[0]->getType();
  Type *ElTy = Ty->getScalarType();
  SmallVector<Type *, 3> ArgTys;
  for (Value *A : Args)
    ArgTys.push_back(A->getType()->getScalarType());
  FunctionCallee FC =
      L.airFunctionPublic((Name + "." + typeSuffix(ElTy)).str(), ElTy, ArgTys);
  auto *VT = dyn_cast<FixedVectorType>(Ty);
  if (VT == nullptr)
    return B.CreateCall(FC, SmallVector<Value *, 3>(Args));
  Value *V = PoisonValue::get(Ty);
  for (unsigned L2 = 0; L2 < VT->getNumElements(); ++L2) {
    SmallVector<Value *, 3> Lane;
    for (Value *A : Args)
      Lane.push_back(B.CreateExtractElement(A, L2));
    V = B.CreateInsertElement(V, B.CreateCall(FC, Lane), L2);
  }
  return V;
}

/* copysign(Mag, Sign) with bit operations */
static Value *copySignBits(IRBuilder<> &B, Value *Mag, Value *Sign) {
  Type *Ty = Mag->getType();
  unsigned BW = Ty->getScalarSizeInBits();
  Type *ITy = Ty->getWithNewType(B.getIntNTy(BW));
  APInt SignMask = APInt::getSignMask(BW);
  Value *M = B.CreateAnd(B.CreateBitCast(Mag, ITy),
                         ConstantInt::get(ITy, ~SignMask));
  Value *S = B.CreateAnd(B.CreateBitCast(Sign, ITy),
                         ConstantInt::get(ITy, SignMask));
  return B.CreateBitCast(B.CreateOr(M, S), Ty);
}

/* 2^K as a floating-point value of type Ty (K: matching integer (vector)
   type, within the normal exponent range) */
static Value *exp2Int(IRBuilder<> &B, Type *Ty, Value *K) {
  const fltSemantics &Sem = Ty->getScalarType()->getFltSemantics();
  unsigned P = APFloat::semanticsPrecision(Sem) - 1; /* mantissa bits */
  int Bias = -APFloat::semanticsMinExponent(Sem) + 1;
  Type *ITy = K->getType();
  Value *E = B.CreateShl(B.CreateAdd(K, ConstantInt::get(ITy, Bias)),
                         ConstantInt::get(ITy, P));
  return B.CreateBitCast(E, Ty);
}

/* ldexp(X, N) as multiplications by powers of two, in three steps so that
   each factor is a normal number */
static Value *expandLdexp(IRBuilder<> &B, Value *X, Value *N) {
  Type *Ty = X->getType();
  const fltSemantics &Sem = Ty->getScalarType()->getFltSemantics();
  unsigned BW = Ty->getScalarSizeInBits();
  int Bias = -APFloat::semanticsMinExponent(Sem) + 1;
  Type *ITy = Ty->getWithNewType(B.getIntNTy(BW));
  Type *NTy = N->getType();
  /* clamp N so that the three steps cover any meaningful exponent */
  int Lim = 3 * (Bias - 1);
  Value *Lo = ConstantInt::get(NTy, -Lim), *Hi = ConstantInt::get(NTy, Lim);
  N = B.CreateSelect(B.CreateICmpSLT(N, Lo), Lo, N);
  N = B.CreateSelect(B.CreateICmpSGT(N, Hi), Hi, N);
  N = B.CreateSExtOrTrunc(N, ITy);
  Value *Three = ConstantInt::get(ITy, 3), *Two = ConstantInt::get(ITy, 2);
  Value *K1 = B.CreateSDiv(N, Three);
  Value *R = B.CreateSub(N, K1);
  Value *K2 = B.CreateSDiv(R, Two);
  Value *K3 = B.CreateSub(R, K2);
  Value *V = B.CreateFMul(X, exp2Int(B, Ty, K1));
  V = B.CreateFMul(V, exp2Int(B, Ty, K2));
  return B.CreateFMul(V, exp2Int(B, Ty, K3));
}

/* frexp(X) -> { mantissa in [0.5, 1), exponent } */
static Value *expandFrexp(IRBuilder<> &B, CallInst *CI) {
  Value *X = CI->getArgOperand(0);
  Type *Ty = X->getType();
  auto *STy = cast<StructType>(CI->getType());
  Type *ExpTy = STy->getElementType(1);
  const fltSemantics &Sem = Ty->getScalarType()->getFltSemantics();
  unsigned BW = Ty->getScalarSizeInBits();
  unsigned P = APFloat::semanticsPrecision(Sem) - 1;
  int Bias = -APFloat::semanticsMinExponent(Sem) + 1;
  Type *ITy = Ty->getWithNewType(B.getIntNTy(BW));
  auto C = [&](uint64_t V) { return ConstantInt::get(ITy, V); };

  /* scale subnormals into the normal range first */
  Value *Bits = B.CreateBitCast(X, ITy);
  APInt SignMask = APInt::getSignMask(BW);
  Value *Mag = B.CreateAnd(Bits, ConstantInt::get(ITy, ~SignMask));
  APInt MinNorm = APFloat::getSmallestNormalized(Sem).bitcastToAPInt();
  APInt Inf = APFloat::getInf(Sem).bitcastToAPInt();
  Value *IsSub = B.CreateAnd(B.CreateICmpULT(Mag, ConstantInt::get(ITy, MinNorm)),
                             B.CreateICmpNE(Mag, C(0)));
  Value *Scaled = B.CreateFMul(X, exp2Int(B, Ty, C(P + 1)));
  Value *Xs = B.CreateSelect(IsSub, Scaled, X);
  Value *Adj = B.CreateSelect(IsSub, C(P + 1), C(0));

  Value *SBits = B.CreateBitCast(Xs, ITy);
  uint64_t ExpMask = ((1ull << (BW - 1 - P)) - 1) << P;
  Value *E = B.CreateLShr(B.CreateAnd(SBits, C(ExpMask)), C(P));
  Value *Exp = B.CreateSub(B.CreateSub(E, C(Bias - 1)), Adj);
  Value *MantBits = B.CreateOr(B.CreateAnd(SBits, C(~ExpMask)),
                               C((uint64_t)(Bias - 1) << P));
  Value *Mant = B.CreateBitCast(MantBits, Ty);

  /* zero, inf and nan are returned as is, with a zero exponent */
  Value *Special = B.CreateOr(B.CreateICmpEQ(Mag, C(0)),
                              B.CreateICmpUGE(Mag, ConstantInt::get(ITy, Inf)));
  Mant = B.CreateSelect(Special, X, Mant);
  Exp = B.CreateSelect(Special, C(0), Exp);
  Exp = B.CreateSExtOrTrunc(Exp, ExpTy);
  Value *R = B.CreateInsertValue(PoisonValue::get(STy), Mant, 0);
  return B.CreateInsertValue(R, Exp, 1);
}

/* Lowering of the LLVM intrinsics that AIR doesn't have. */
bool AirLowering::lowerIntrinsics() {
  /* intrinsic -> (precise AIR function, relaxed f32 variant or null) */
  static const std::map<Intrinsic::ID, std::pair<const char *, const char *>>
      MathFns = {
          {Intrinsic::sqrt, {"air.sqrt", "air.fast_sqrt"}},
          {Intrinsic::fma, {"air.fma", nullptr}},
          {Intrinsic::fmuladd, {"air.fma", nullptr}},
          {Intrinsic::floor, {"air.floor", "air.fast_floor"}},
          {Intrinsic::ceil, {"air.ceil", "air.fast_ceil"}},
          {Intrinsic::trunc, {"air.trunc", "air.fast_trunc"}},
          {Intrinsic::rint, {"air.rint", "air.fast_rint"}},
          {Intrinsic::nearbyint, {"air.rint", "air.fast_rint"}},
          {Intrinsic::roundeven, {"air.rint", "air.fast_rint"}},
          {Intrinsic::round, {"air.round", "air.fast_round"}},
          {Intrinsic::exp, {"air.exp", "air.fast_exp"}},
          {Intrinsic::exp2, {"air.exp2", "air.fast_exp2"}},
#if LLVM_MAJOR >= 18
          {Intrinsic::exp10, {"air.exp10", "air.fast_exp10"}},
#endif
          {Intrinsic::log, {"air.log", "air.fast_log"}},
          {Intrinsic::log2, {"air.log2", "air.fast_log2"}},
          {Intrinsic::log10, {"air.log10", "air.fast_log10"}},
          {Intrinsic::pow, {"air.pow", "air.fast_pow"}},
          {Intrinsic::sin, {"air.sin", "air.fast_sin"}},
          {Intrinsic::cos, {"air.cos", "air.fast_cos"}},
#if LLVM_MAJOR >= 19
          {Intrinsic::tan, {"air.tan", "air.fast_tan"}},
          {Intrinsic::asin, {"air.asin", "air.fast_asin"}},
          {Intrinsic::acos, {"air.acos", "air.fast_acos"}},
          {Intrinsic::atan, {"air.atan", "air.fast_atan"}},
          {Intrinsic::sinh, {"air.sinh", "air.fast_sinh"}},
          {Intrinsic::cosh, {"air.cosh", "air.fast_cosh"}},
          {Intrinsic::tanh, {"air.tanh", "air.fast_tanh"}},
#endif
#if LLVM_MAJOR >= 20
          {Intrinsic::atan2, {"air.atan2", "air.fast_atan2"}},
#endif
      };
  /* intrinsic -> (AIR function, signedness: -1 n/a, 0 unsigned, 1 signed);
     these take vectors as well */
  static const std::map<Intrinsic::ID, std::pair<const char *, int>>
      ValueFns = {
          {Intrinsic::fabs, {"air.fabs", -1}},
          {Intrinsic::minnum, {"air.fmin", -1}},
          {Intrinsic::maxnum, {"air.fmax", -1}},
          {Intrinsic::abs, {"air.abs", 1}},
          {Intrinsic::smin, {"air.min", 1}},
          {Intrinsic::smax, {"air.max", 1}},
          {Intrinsic::umin, {"air.min", 0}},
          {Intrinsic::umax, {"air.max", 0}},
      };
  /* plain renames, same signature */
  static const std::map<Intrinsic::ID, const char *> BitFns = {
      {Intrinsic::ctlz, "air.clz"},
      {Intrinsic::cttz, "air.ctz"},
      {Intrinsic::ctpop, "air.popcount"},
      {Intrinsic::bitreverse, "air.reverse_bits"},
  };

  SmallVector<CallInst *, 64> Calls;
  for (Function &F : M)
    for (Instruction &I : instructions(F))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (Function *Callee = CI->getCalledFunction())
          if (Callee->isIntrinsic())
            Calls.push_back(CI);

  std::set<std::string> Unsupported;
  for (CallInst *CI : Calls) {
    Intrinsic::ID ID = CI->getIntrinsicID();
    IRBuilder<> B(CI);
    Type *Ty = CI->getType();
    Value *Repl = nullptr;

    switch (ID) {
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::assume:
    case Intrinsic::experimental_noalias_scope_decl:
    case Intrinsic::trap:
    case Intrinsic::debugtrap:
    case Intrinsic::ubsantrap:
    case Intrinsic::donothing:
    case Intrinsic::sideeffect:
      CI->eraseFromParent();
      continue;
    case Intrinsic::canonicalize:
    case Intrinsic::arithmetic_fence:
    case Intrinsic::expect:
      Repl = CI->getArgOperand(0);
      break;
    case Intrinsic::memcpy:
    case Intrinsic::memmove:
    case Intrinsic::memset:
      /* emitted by Apple's frontend as well */
      continue;
    case Intrinsic::memcpy_inline: {
      auto *MI = cast<MemTransferInst>(CI);
      B.CreateMemCpy(MI->getDest(), MI->getDestAlign(), MI->getSource(),
                     MI->getSourceAlign(), MI->getLength(), MI->isVolatile());
      CI->eraseFromParent();
      continue;
    }
    case Intrinsic::ldexp:
      Repl = expandLdexp(B, CI->getArgOperand(0), CI->getArgOperand(1));
      break;
    case Intrinsic::frexp:
      Repl = expandFrexp(B, CI);
      break;
    case Intrinsic::modf: {
      /* { fractional part, integral part }, both with the sign of x */
      Value *X = CI->getArgOperand(0);
      Value *IntPart = callAirScalarized(*this, B, "air.trunc", {X});
      Value *Frac = B.CreateFSub(X, IntPart);
      Value *IsInf = B.CreateFCmpOEQ(
          callAirScalarized(*this, B, "air.fabs", {X}),
          ConstantFP::getInfinity(X->getType()));
      Frac = B.CreateSelect(IsInf, ConstantFP::get(X->getType(), 0.0), Frac);
      Frac = copySignBits(B, Frac, X);
      Value *R = B.CreateInsertValue(PoisonValue::get(Ty), Frac, 0);
      Repl = B.CreateInsertValue(R, IntPart, 1);
      break;
    }
    case Intrinsic::powi: {
      /* pow(|x|, n), negated for negative x and odd n */
      Value *X = CI->getArgOperand(0);
      Value *NI = CI->getArgOperand(1);
      Type *XTy = X->getType();
      if (auto *VT = dyn_cast<FixedVectorType>(XTy))
        if (!NI->getType()->isVectorTy())
          NI = B.CreateVectorSplat(VT->getNumElements(), NI);
      Value *N = B.CreateSIToFP(NI, XTy);
      Value *AbsX = callAirScalarized(*this, B, "air.fabs", {X});
      Value *P = callAirScalarized(*this, B, "air.pow", {AbsX, N});
      Value *Odd = B.CreateTrunc(NI, NI->getType()->getWithNewType(B.getInt1Ty()));
      Value *Neg = B.CreateFCmpOLT(X, ConstantFP::get(XTy, 0.0));
      Repl = B.CreateSelect(B.CreateAnd(Odd, Neg), B.CreateFNeg(P), P);
      break;
    }
    case Intrinsic::copysign: {
      Type *ITy =
          Ty->getWithNewType(Type::getIntNTy(Ctx, Ty->getScalarSizeInBits()));
      Value *X = B.CreateBitCast(CI->getArgOperand(0), ITy);
      Value *Y = B.CreateBitCast(CI->getArgOperand(1), ITy);
      APInt SignMask = APInt::getSignMask(Ty->getScalarSizeInBits());
      Value *Mag = B.CreateAnd(X, ConstantInt::get(ITy, ~SignMask));
      Value *Sign = B.CreateAnd(Y, ConstantInt::get(ITy, SignMask));
      Repl = B.CreateBitCast(B.CreateOr(Mag, Sign), Ty);
      break;
    }
    case Intrinsic::fshl:
    case Intrinsic::fshr: {
      Value *A = CI->getArgOperand(0), *Bv = CI->getArgOperand(1),
            *C = CI->getArgOperand(2);
      unsigned BW = Ty->getScalarSizeInBits();
      Value *BWV = ConstantInt::get(Ty, BW);
      Value *Sh = B.CreateURem(C, BWV);
      Value *Inv = B.CreateSub(BWV, Sh);
      Value *Zero = Constant::getNullValue(Ty);
      Value *IsZero = B.CreateICmpEQ(Sh, Zero);
      Value *R;
      if (ID == Intrinsic::fshl)
        R = B.CreateSelect(IsZero, A,
                           B.CreateOr(B.CreateShl(A, Sh), B.CreateLShr(Bv, Inv)));
      else
        R = B.CreateSelect(IsZero, Bv,
                           B.CreateOr(B.CreateShl(A, Inv), B.CreateLShr(Bv, Sh)));
      Repl = R;
      break;
    }
    case Intrinsic::uadd_sat:
    case Intrinsic::usub_sat:
    case Intrinsic::sadd_sat:
    case Intrinsic::ssub_sat: {
      Value *A = CI->getArgOperand(0), *Bv = CI->getArgOperand(1);
      unsigned BW = Ty->getScalarSizeInBits();
      if (ID == Intrinsic::uadd_sat) {
        Value *S = B.CreateAdd(A, Bv);
        Repl = B.CreateSelect(B.CreateICmpULT(S, A),
                              Constant::getAllOnesValue(Ty), S);
      } else if (ID == Intrinsic::usub_sat) {
        Repl = B.CreateSelect(B.CreateICmpUGT(A, Bv), B.CreateSub(A, Bv),
                              Constant::getNullValue(Ty));
      } else {
        Value *R = ID == Intrinsic::sadd_sat ? B.CreateAdd(A, Bv)
                                              : B.CreateSub(A, Bv);
        Value *Ovf =
            ID == Intrinsic::sadd_sat
                ? B.CreateAnd(B.CreateXor(A, R), B.CreateXor(Bv, R))
                : B.CreateAnd(B.CreateXor(A, Bv), B.CreateXor(A, R));
        Value *IsOvf = B.CreateICmpSLT(Ovf, Constant::getNullValue(Ty));
        Value *Sat = B.CreateSelect(
            B.CreateICmpSLT(A, Constant::getNullValue(Ty)),
            ConstantInt::get(Ty, APInt::getSignedMinValue(BW)),
            ConstantInt::get(Ty, APInt::getSignedMaxValue(BW)));
        Repl = B.CreateSelect(IsOvf, Sat, R);
      }
      break;
    }
    case Intrinsic::is_fpclass: {
      Value *X = CI->getArgOperand(0);
      uint64_t Mask = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      Type *XTy = X->getType();
      unsigned BW = XTy->getScalarSizeInBits();
      Type *ITy = XTy->getWithNewType(Type::getIntNTy(Ctx, BW));
      const fltSemantics &Sem = XTy->getScalarType()->getFltSemantics();
      APInt Inf = APFloat::getInf(Sem).bitcastToAPInt();
      APInt QNaN = APFloat::getQNaN(Sem).bitcastToAPInt();
      APInt MinNorm = APFloat::getSmallestNormalized(Sem).bitcastToAPInt();
      Value *Bits = B.CreateBitCast(X, ITy);
      Value *Mag =
          B.CreateAnd(Bits, ConstantInt::get(ITy, ~APInt::getSignMask(BW)));
      Value *Neg = B.CreateICmpSLT(Bits, Constant::getNullValue(ITy));
      Value *Pos = B.CreateNot(Neg);
      auto C = [&](const APInt &V) { return ConstantInt::get(ITy, V); };
      Value *IsNaN = B.CreateICmpUGT(Mag, C(Inf));
      Value *IsQNaN = B.CreateICmpUGE(Mag, C(QNaN));
      Value *IsSNaN = B.CreateAnd(IsNaN, B.CreateNot(IsQNaN));
      Value *IsInf = B.CreateICmpEQ(Mag, C(Inf));
      Value *IsZero = B.CreateICmpEQ(Mag, Constant::getNullValue(ITy));
      Value *IsSub = B.CreateICmpULT(B.CreateSub(Mag, C(APInt(BW, 1))),
                                     C(MinNorm - 1));
      Value *IsNorm =
          B.CreateICmpULT(B.CreateSub(Mag, C(MinNorm)), C(Inf - MinNorm));
      Value *R = Constant::getNullValue(Ty);
      auto Add = [&](unsigned Bit, Value *V) {
        if (Mask & (1u << Bit))
          R = B.CreateOr(R, V);
      };
      Add(0, IsSNaN);
      Add(1, IsQNaN);
      Add(2, B.CreateAnd(IsInf, Neg));
      Add(3, B.CreateAnd(IsNorm, Neg));
      Add(4, B.CreateAnd(IsSub, Neg));
      Add(5, B.CreateAnd(IsZero, Neg));
      Add(6, B.CreateAnd(IsZero, Pos));
      Add(7, B.CreateAnd(IsSub, Pos));
      Add(8, B.CreateAnd(IsNorm, Pos));
      Add(9, B.CreateAnd(IsInf, Pos));
      Repl = R;
      break;
    }
    case Intrinsic::uadd_with_overflow:
    case Intrinsic::usub_with_overflow:
    case Intrinsic::sadd_with_overflow:
    case Intrinsic::ssub_with_overflow: {
      Value *A = CI->getArgOperand(0), *Bv = CI->getArgOperand(1);
      Type *ETy = A->getType();
      Value *R, *Ovf;
      if (ID == Intrinsic::uadd_with_overflow) {
        R = B.CreateAdd(A, Bv);
        Ovf = B.CreateICmpULT(R, A);
      } else if (ID == Intrinsic::usub_with_overflow) {
        R = B.CreateSub(A, Bv);
        Ovf = B.CreateICmpULT(A, Bv);
      } else if (ID == Intrinsic::sadd_with_overflow) {
        R = B.CreateAdd(A, Bv);
        Ovf = B.CreateICmpSLT(B.CreateAnd(B.CreateXor(A, R), B.CreateXor(Bv, R)),
                              Constant::getNullValue(ETy));
      } else {
        R = B.CreateSub(A, Bv);
        Ovf = B.CreateICmpSLT(B.CreateAnd(B.CreateXor(A, Bv), B.CreateXor(A, R)),
                              Constant::getNullValue(ETy));
      }
      Value *S = B.CreateInsertValue(PoisonValue::get(Ty), R, 0);
      Repl = B.CreateInsertValue(S, Ovf, 1);
      break;
    }
    default:
      break;
    }

    if (Repl == nullptr) {
      auto MI = MathFns.find(ID);
      auto VI = ValueFns.find(ID);
      auto BI = BitFns.find(ID);
      Type *ElTy = Ty->getScalarType();
      if (MI != MathFns.end() && (ElTy->isFloatTy() || ElTy->isHalfTy())) {
        bool Fast = MI->second.second != nullptr && ElTy->isFloatTy() &&
                    isa<FPMathOperator>(CI) && CI->hasApproxFunc();
        std::string Name = std::string(Fast ? MI->second.second
                                            : MI->second.first) +
                           "." + typeSuffix(ElTy);
        unsigned NumArgs = CI->arg_size();
        SmallVector<Type *, 3> ArgTys(NumArgs, ElTy);
        FunctionCallee FC = airFunction(Name, ElTy, ArgTys);
        /* AIR math functions are scalar: scalarize vectors */
        if (auto *VT = dyn_cast<FixedVectorType>(Ty)) {
          Value *V = PoisonValue::get(Ty);
          for (unsigned L = 0; L < VT->getNumElements(); ++L) {
            SmallVector<Value *, 3> Args;
            for (unsigned A = 0; A < NumArgs; ++A)
              Args.push_back(B.CreateExtractElement(CI->getArgOperand(A), L));
            V = B.CreateInsertElement(V, B.CreateCall(FC, Args), L);
          }
          Repl = V;
        } else {
          SmallVector<Value *, 3> Args(CI->args());
          Repl = B.CreateCall(FC, Args);
        }
      } else if (VI != ValueFns.end()) {
        std::string Name = VI->second.first;
        if (VI->second.second >= 0)
          Name += VI->second.second ? ".s" : ".u";
        Name += "." + typeSuffix(Ty);
        /* llvm.abs has an extra i1 operand */
        SmallVector<Value *, 2> Args;
        SmallVector<Type *, 2> ArgTys;
        for (Value *A : CI->args())
          if (A->getType() == Ty) {
            Args.push_back(A);
            ArgTys.push_back(Ty);
          }
        Repl = B.CreateCall(airFunction(Name, Ty, ArgTys), Args);
      } else if (BI != BitFns.end()) {
        std::string Name = std::string(BI->second) + "." + typeSuffix(Ty);
        SmallVector<Value *, 2> Args(CI->args());
        SmallVector<Type *, 2> ArgTys;
        for (Value *A : Args)
          ArgTys.push_back(A->getType());
        Repl = B.CreateCall(airFunction(Name, Ty, ArgTys), Args);
      } else {
        Unsupported.insert(CI->getCalledFunction()->getName().str());
        continue;
      }
    }

    CI->replaceAllUsesWith(Repl);
    CI->eraseFromParent();
  }

  if (!Unsupported.empty()) {
    std::string List;
    for (const std::string &S : Unsupported)
      List += (List.empty() ? "" : ", ") + S;
    return fail("the Metal driver can't lower these LLVM intrinsics: " + List);
  }

  for (Function &F : make_early_inc_range(M))
    if (F.isIntrinsic() && F.use_empty())
      F.eraseFromParent();
  return true;
}

/* Apple's back-end doesn't handle unreachable code well (see Metal.jl
   #433 and #370). Everything is inlined into the (void) kernels by now, so
   simply return instead. */
void AirLowering::lowerUnreachables() {
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (BasicBlock &BB : F) {
      auto *U = dyn_cast<UnreachableInst>(BB.getTerminator());
      if (U == nullptr)
        continue;
      ReturnInst::Create(Ctx, nullptr, U->getIterator());
      U->eraseFromParent();
    }
    F.removeFnAttr(Attribute::NoReturn);
  }
}

void AirLowering::stripMetadata() {
  StripDebugInfo(M);
  SmallVector<NamedMDNode *, 8> Named;
  for (NamedMDNode &N : M.named_metadata())
    Named.push_back(&N);
  for (NamedMDNode *N : Named)
    N->eraseFromParent();
  for (Function &F : M) {
    F.clearMetadata();
    for (Instruction &I : instructions(F))
      I.dropUnknownNonDebugMetadata();
  }
  for (GlobalVariable &G : M.globals())
    G.clearMetadata();
}

void AirLowering::addAirMetadata() {
  auto Int = [&](uint64_t V) {
    return ConstantAsMetadata::get(ConstantInt::get(I32, V));
  };
  auto Str = [&](StringRef S) { return MDString::get(Ctx, S); };

  NamedMDNode *Kernels = M.getOrInsertNamedMetadata("air.kernel");
  for (auto &KV : Infos) {
    Function *F = KV.first;
    KernelInfo &Info = KV.second;
    SmallVector<Metadata *, 16> ArgMDs;
    unsigned Idx = 0;
    for (KernelArg &KA : Info.Args) {
      SmallVector<Metadata *, 20> MD = {Int(Idx), Str("air.buffer")};
      bool ByRef = KA.Kind != ArgKind::Buffer;
      if (ByRef) {
        MD.push_back(Str("air.buffer_size"));
        MD.push_back(Int(KA.TypeSize));
      }
      MD.push_back(Str("air.location_index"));
      MD.push_back(Int(Idx));
      MD.push_back(Int(1));
      bool ReadOnly = ByRef || KA.AddrSpace == AS_CONSTANT;
      MD.push_back(Str(ReadOnly ? "air.read" : "air.read_write"));
      MD.push_back(Str("air.address_space"));
      MD.push_back(Int(ByRef ? AS_CONSTANT : KA.AddrSpace));
      MD.push_back(Str("air.arg_type_size"));
      MD.push_back(Int(KA.TypeSize));
      MD.push_back(Str("air.arg_type_align_size"));
      MD.push_back(Int(KA.TypeAlign));
      MD.push_back(Str("air.arg_type_name"));
      MD.push_back(Str(KA.TypeName));
      MD.push_back(Str("air.arg_name"));
      MD.push_back(Str(KA.Name));
      ArgMDs.push_back(MDNode::get(Ctx, MD));
      ++Idx;
    }
    /* launch parameters */
    ArgMDs.push_back(MDNode::get(
        Ctx, {Int(Idx), Str("air.buffer"), Str("air.buffer_size"),
              Int(sizeof(pocl_metal_launch_params_t)),
              Str("air.location_index"), Int(Idx), Int(1), Str("air.read"),
              Str("air.address_space"), Int(AS_CONSTANT),
              Str("air.arg_type_size"),
              Int(sizeof(pocl_metal_launch_params_t)),
              Str("air.arg_type_align_size"), Int(4),
              Str("air.arg_type_name"), Str("pocl_launch_params"),
              Str("air.arg_name"), Str("_pocl_launch_params")}));
    ++Idx;
    /* printf buffer */
    ArgMDs.push_back(MDNode::get(
        Ctx, {Int(Idx), Str("air.buffer"), Str("air.location_index"),
              Int(Idx), Int(1), Str("air.read_write"),
              Str("air.address_space"), Int(AS_DEVICE),
              Str("air.arg_type_size"), Int(4),
              Str("air.arg_type_align_size"), Int(4),
              Str("air.arg_type_name"), Str("uint"), Str("air.arg_name"),
              Str("_pocl_printf_buffer")}));
    ++Idx;
    /* program-scope variables */
    ArgMDs.push_back(MDNode::get(
        Ctx, {Int(Idx), Str("air.buffer"), Str("air.location_index"),
              Int(Idx), Int(1), Str("air.read_write"),
              Str("air.address_space"), Int(AS_DEVICE),
              Str("air.arg_type_size"), Int(1),
              Str("air.arg_type_align_size"), Int(1),
              Str("air.arg_type_name"), Str("uchar"), Str("air.arg_name"),
              Str("_pocl_gvar_buffer")}));
    ++Idx;
    for (unsigned I : Info.Inputs) {
      std::string Name = AirInputs[I].AirName;
      ArgMDs.push_back(MDNode::get(
          Ctx, {Int(Idx), Str("air." + Name), Str("air.arg_type_name"),
                Str("uint3"), Str("air.arg_name"), Str(Name)}));
      ++Idx;
    }
    Kernels->addOperand(
        MDNode::get(Ctx, {ValueAsMetadata::get(F), MDNode::get(Ctx, {}),
                          MDNode::get(Ctx, ArgMDs)}));
  }

  NamedMDNode *Flags = M.getOrInsertNamedMetadata("llvm.module.flags");
  auto Flag = [&](unsigned Behavior, StringRef Name, uint64_t V) {
    Flags->addOperand(MDNode::get(Ctx, {Int(Behavior), Str(Name), Int(V)}));
  };
  Flag(1, "wchar_size", 4);
  Flag(7, "air.max_device_buffers", 31);
  Flag(7, "air.max_constant_buffers", 31);
  Flag(7, "air.max_threadgroup_buffers", 31);
  Flag(7, "air.max_textures", 128);
  Flag(7, "air.max_read_write_textures", 8);
  Flag(7, "air.max_samplers", 16);

  M.getOrInsertNamedMetadata("llvm.ident")
      ->addOperand(MDNode::get(Ctx, {Str("PoCL Metal driver")}));
  M.getOrInsertNamedMetadata("air.version")
      ->addOperand(MDNode::get(
          Ctx, {Int(T.air_major), Int(T.air_minor), Int(0)}));
  M.getOrInsertNamedMetadata("air.language_version")
      ->addOperand(MDNode::get(Ctx, {Str("Metal"), Int(T.metal_major),
                                     Int(T.metal_minor), Int(0)}));
  NamedMDNode *Opts = M.getOrInsertNamedMetadata("air.compile_options");
  for (const char *O : {"air.compile.denorms_disable",
                        "air.compile.fast_math_disable",
                        "air.compile.framebuffer_fetch_enable"})
    Opts->addOperand(MDNode::get(Ctx, {Str(O)}));
}

/* Null is 0 in every AIR address space, so address space casts of null
   pointers are null pointers. LLVM can't assume that and keeps e.g.
   addrspacecast (ptr addrspace(4) null to ptr) from "(void *)0", which
   then compares unequal to null. */
static Constant *foldNullCasts(Constant *C, DenseMap<Constant *, Constant *> &Memo) {
  auto It = Memo.find(C);
  if (It != Memo.end())
    return It->second;
  Constant *R = C;
  if (auto *CE = dyn_cast<ConstantExpr>(C)) {
    if (CE->getOpcode() == Instruction::AddrSpaceCast &&
        isa<ConstantPointerNull>(CE->getOperand(0))) {
      R = ConstantPointerNull::get(cast<PointerType>(CE->getType()));
    } else {
      SmallVector<Constant *, 4> Ops;
      bool Changed = false;
      for (Use &U : CE->operands()) {
        Constant *Op = cast<Constant>(U.get());
        Constant *NOp = foldNullCasts(Op, Memo);
        Changed |= NOp != Op;
        Ops.push_back(NOp);
      }
      if (Changed)
        R = CE->getWithOperands(Ops);
    }
  }
  Memo[C] = R;
  return R;
}

static void foldNullAddrSpaceCasts(Module &M) {
  DenseMap<Constant *, Constant *> Memo;
  for (Function &F : M)
    for (Instruction &I : instructions(F)) {
      if (auto *ASC = dyn_cast<AddrSpaceCastInst>(&I))
        if (isa<ConstantPointerNull>(ASC->getPointerOperand())) {
          ASC->replaceAllUsesWith(
              ConstantPointerNull::get(cast<PointerType>(ASC->getType())));
          continue;
        }
      for (Use &U : I.operands())
        if (auto *C = dyn_cast<ConstantExpr>(U.get())) {
          Constant *NC = foldNullCasts(C, Memo);
          if (NC != C)
            U.set(NC);
        }
    }
}

bool AirLowering::run(unsigned &LocalRegionSize) {
  LocalRegionSize = 0;
  if (!collectKernels())
    return false;
  normalizeCallingConventions();
  foldNullAddrSpaceCasts(M);
  if (!inlineEverything())
    return false;
  if (!moveConstantGlobals())
    return false;
  /* clean up after inlining (without introducing new intrinsics we can't
     handle: the intrinsics are lowered after this) */
  if (!runPipeline("function(sroa,early-cse,instcombine,simplifycfg,"
                   "expand-reductions),globaldce"))
    return false;

  if (!collectProgramScopeVariables())
    return false;
  uint64_t StaticLocal = 0;
  if (!checkGlobals(StaticLocal))
    return false;
  uint64_t RegionSize = alignDown(T.local_mem_size - StaticLocal,
                                  POCL_METAL_LOCAL_ARG_ALIGN);

  /* make all uses of the placeholder variables instructions, so that they
     can be replaced per kernel */
  SmallVector<Constant *, 8> Placeholders;
  for (GlobalVariable &G : M.globals())
    if (G.getName().starts_with("_pocl_air_"))
      Placeholders.push_back(&G);
  convertUsersOfConstantsToInstructions(Placeholders);

  std::vector<Function *> Ks = Kernels;
  for (Function *K : Ks)
    if (!rewriteKernel(K, RegionSize))
      return false;
  if (LocalRegion)
    LocalRegionSize = RegionSize;
  /* (the initializers, now in GVarInit, may refer to other variables) */
  for (auto &GV : GVars)
    if (GV.first->hasInitializer())
      GV.first->setInitializer(nullptr);
  for (auto &GV : GVars) {
    GV.first->removeDeadConstantUsers();
    if (!GV.first->use_empty())
      return fail("unexpected use of the program-scope variable '" +
                  GV.first->getName() + "' outside of a kernel");
    GV.first->eraseFromParent();
  }

  if (!lowerSyncPlaceholders())
    return false;
  /* promote the private copies of the work-item inputs (these passes don't
     introduce intrinsics) */
  if (!runPipeline("function(sroa,early-cse,instsimplify)"))
    return false;
  if (!lowerAtomics())
    return false;
  if (!lowerIntrinsics())
    return false;
  lowerUnreachables();

  if (!runPipeline("function(dce,simplifycfg),strip-dead-prototypes"))
    return false;

  /* LLVM's loop idiom recognition may turn byte loops (e.g. the format
     string length computation of printf) into strlen calls, but there's no
     C library on Metal: expand them back into loops */
  if (Function *Strlen = M.getFunction("strlen")) {
    for (User *U : make_early_inc_range(Strlen->users())) {
      auto *CI = dyn_cast<CallInst>(U);
      if (CI == nullptr || CI->getCalledFunction() != Strlen)
        continue;
      Value *Str = CI->getArgOperand(0);
      Type *SizeTy = CI->getType();
      BasicBlock *Pre = CI->getParent();
      BasicBlock *Exit = Pre->splitBasicBlock(CI->getIterator(), "strlen.end");
      BasicBlock *Loop = BasicBlock::Create(Ctx, "strlen.loop",
                                            Pre->getParent(), Exit);
      Pre->getTerminator()->eraseFromParent();
      IRBuilder<> B(Pre);
      B.CreateBr(Loop);
      B.SetInsertPoint(Loop);
      PHINode *Idx = B.CreatePHI(SizeTy, 2, "strlen.idx");
      Idx->addIncoming(ConstantInt::get(SizeTy, 0), Pre);
      Value *Ch = B.CreateLoad(B.getInt8Ty(),
                               B.CreateGEP(B.getInt8Ty(), Str, Idx));
      Value *Next = B.CreateAdd(Idx, ConstantInt::get(SizeTy, 1));
      Idx->addIncoming(Next, Loop);
      B.CreateCondBr(B.CreateICmpEQ(Ch, B.getInt8(0)), Exit, Loop);
      CI->replaceAllUsesWith(Idx);
      CI->eraseFromParent();
    }
    if (Strlen->use_empty())
      Strlen->eraseFromParent();
  }

  /* anything else would be an undefined symbol for Metal's compiler */
  for (Function &F : M) {
    if (!F.isDeclaration() || F.use_empty())
      continue;
    StringRef N = F.getName();
    if (N.starts_with("air.") || N.starts_with("llvm."))
      continue;
    return fail("call to the function '" + N +
                "', which the Metal driver doesn't support");
  }

  /* Keep only the attributes Apple's frontend puts on kernels; the newer
     LLVM ones (memory(...), nocallback, ...) have no meaning in AIR. */
  for (auto &KV : Infos) {
    Function *F = KV.first;
    bool Convergent = F->hasFnAttribute(Attribute::Convergent);
    F->setAttributes(AttributeList());
    F->addFnAttr(Attribute::NoUnwind);
    if (Convergent)
      F->addFnAttr(Attribute::Convergent);
    /* MSL compiles with fast math by default; state explicitly that the
       kernels don't allow fast-math assumptions (as the module's
       air.compile.fast_math_disable option also does). */
    for (const char *A : {"no-infs-fp-math", "no-nans-fp-math",
                          "no-signed-zeros-fp-math", "unsafe-fp-math",
                          "approx-func-fp-math", "less-precise-fpmad"})
      F->addFnAttr(A, "false");
  }

  /* Apple's back-end folds loads from constant tables after unrolling loops,
     and then gets comparisons on special values wrong (e.g. isfinite of a
     NaN loaded from a __constant array). LLVM has already done the folding
     it can, so keep the back-end from folding the table contents. */
  for (GlobalVariable &G : M.globals())
    if (G.getAddressSpace() == AS_CONSTANT && G.hasInitializer() &&
        G.getValueType()->isAggregateType())
      G.setExternallyInitialized(true);

  stripMetadata();
  M.setDataLayout(AirDataLayout);
  std::string Triple = "air64_v" + std::to_string(T.air_major) +
                       std::to_string(T.air_minor) + "-apple-macosx" +
                       std::to_string(T.macos_major) + "." +
                       std::to_string(T.macos_minor) + ".0";
#if LLVM_MAJOR >= 21
  M.setTargetTriple(llvm::Triple(Triple));
#else
  M.setTargetTriple(Triple);
#endif
  addAirMetadata();

  std::string VerifyErr;
  raw_string_ostream VS(VerifyErr);
  if (verifyModule(M, &VS))
    return fail("the lowered AIR module is invalid: " + VS.str());
  return true;
}

/****************************************************************************/
/* Metal library (metallib) container */

class ByteWriter {
public:
  std::vector<uint8_t> Data;
  void u8(uint8_t V) { Data.push_back(V); }
  void u16(uint16_t V) {
    for (int I = 0; I < 2; ++I)
      Data.push_back((V >> (8 * I)) & 0xff);
  }
  void u32(uint32_t V) {
    for (int I = 0; I < 4; ++I)
      Data.push_back((V >> (8 * I)) & 0xff);
  }
  void u64(uint64_t V) {
    for (int I = 0; I < 8; ++I)
      Data.push_back((V >> (8 * I)) & 0xff);
  }
  void bytes(const void *P, size_t N) {
    const uint8_t *B = (const uint8_t *)P;
    Data.insert(Data.end(), B, B + N);
  }
  void str(const char *S) { bytes(S, strlen(S)); }
  void patch64(size_t Pos, uint64_t V) {
    for (int I = 0; I < 8; ++I)
      Data[Pos + I] = (V >> (8 * I)) & 0xff;
  }
  size_t size() const { return Data.size(); }
};

/* A tag group: a sequence of (4-char tag, 16-bit size, value) entries ended
   by ENDT, optionally prefixed by its 32-bit total size (including the size
   field itself). */
class TagGroup {
public:
  ByteWriter W;
  void tag(const char *Tag, const ByteWriter &Value) {
    W.str(Tag);
    W.u16((uint16_t)Value.size());
    W.bytes(Value.Data.data(), Value.size());
  }
  std::vector<uint8_t> finish(bool WithSize) {
    W.str("ENDT");
    ByteWriter Out;
    if (WithSize)
      Out.u32((uint32_t)(W.size() + 4));
    Out.bytes(W.Data.data(), W.size());
    return Out.Data;
  }
};

std::vector<uint8_t> buildMetallib(const std::vector<std::string> &Kernels,
                                   const char *Module, size_t ModuleSize,
                                   const pocl_metal_air_target_t &T) {
  uint8_t Hash[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(Module, (CC_LONG)ModuleSize, Hash);

  /* function list: all kernels share the one module */
  ByteWriter FunctionList;
  FunctionList.u32((uint32_t)Kernels.size());
  for (const std::string &K : Kernels) {
    TagGroup G;
    ByteWriter V;
    V.str(K.c_str());
    V.u8(0);
    G.tag("NAME", V);
    V = ByteWriter();
    V.u8(2); /* compute function */
    G.tag("TYPE", V);
    V = ByteWriter();
    V.bytes(Hash, sizeof(Hash));
    G.tag("HASH", V);
    V = ByteWriter();
    V.u64(0); /* public metadata offset */
    V.u64(0); /* private metadata offset */
    V.u64(0); /* module offset */
    G.tag("OFFT", V);
    V = ByteWriter();
    V.u16(T.air_major);
    V.u16(T.air_minor);
    V.u16(T.metal_major);
    V.u16(T.metal_minor);
    G.tag("VERS", V);
    V = ByteWriter();
    V.u64(ModuleSize);
    G.tag("MDSZ", V);
    std::vector<uint8_t> B = G.finish(true);
    FunctionList.bytes(B.data(), B.size());
  }

  /* header extension: the library UUID, derived from the module hash like a
     version 4 UUID */
  uint8_t UUID[16];
  memcpy(UUID, Hash, 16);
  UUID[6] = (UUID[6] & 0x0f) | 0x40;
  UUID[8] = (UUID[8] & 0x3f) | 0x80;
  TagGroup HeaderExt;
  {
    ByteWriter V;
    V.bytes(UUID, 16);
    HeaderExt.tag("UUID", V);
  }
  std::vector<uint8_t> HeaderExtBytes = HeaderExt.finish(false);

  /* empty public and private metadata, shared by all functions */
  TagGroup Empty;
  std::vector<uint8_t> EmptyMD = Empty.finish(true);

  ByteWriter Out;
  Out.str("MTLB");
  Out.u16(1 | 0x8000); /* file version 1.2.8, macOS */
  Out.u16(2);
  Out.u16(8);
  Out.u8(0);           /* executable */
  Out.u8(1 | 0x80);    /* macOS, 64-bit */
  Out.u16(T.macos_major);
  Out.u8(T.macos_minor);
  Out.u8(0);
  size_t FileSizePos = Out.size();
  Out.u64(0);
  size_t FunctionListPos = Out.size();
  Out.u64(0);
  Out.u64(FunctionList.size() - 4);
  size_t PublicMDPos = Out.size();
  Out.u64(0);
  Out.u64(EmptyMD.size());
  size_t PrivateMDPos = Out.size();
  Out.u64(0);
  Out.u64(EmptyMD.size());
  size_t ModulePos = Out.size();
  Out.u64(0);
  Out.u64(ModuleSize);
  assert(Out.size() == 0x58);

  Out.patch64(FunctionListPos, Out.size());
  Out.bytes(FunctionList.Data.data(), FunctionList.size());
  Out.bytes(HeaderExtBytes.data(), HeaderExtBytes.size());
  Out.patch64(PublicMDPos, Out.size());
  Out.bytes(EmptyMD.data(), EmptyMD.size());
  Out.patch64(PrivateMDPos, Out.size());
  Out.bytes(EmptyMD.data(), EmptyMD.size());
  Out.patch64(ModulePos, Out.size());
  Out.bytes(Module, ModuleSize);
  Out.patch64(FileSizePos, Out.size());
  return Out.Data;
}

char *dupError(const std::string &S) { return strdup(S.c_str()); }

} // namespace

int pocl_metal_generate_metallib(void *ProgramPtr, unsigned DeviceI,
                                 const pocl_metal_air_target_t *Target,
                                 char **Metallib, size_t *MetallibSize,
                                 unsigned *LocalRegionSize,
                                 unsigned *GVarSize, char **GVarInit,
                                 pocl_metal_gvar_reloc_t **GVarRelocs,
                                 unsigned *NumGVarRelocs, char **ErrorLog) {
  cl_program Program = (cl_program)ProgramPtr;
  *Metallib = nullptr;
  *MetallibSize = 0;
  *GVarSize = 0;
  *GVarInit = nullptr;
  *GVarRelocs = nullptr;
  *NumGVarRelocs = 0;
  if (ErrorLog)
    *ErrorLog = nullptr;

  PoclLLVMContextData *LLVMCtx =
      (PoclLLVMContextData *)Program->context->llvm_context_data;
  std::vector<std::string> KernelNames;
  SmallVector<char, 0> Bitcode;
  {
    struct LockGuard {
      pocl_lock_t *L;
      LockGuard(pocl_lock_t *L) : L(L) { POCL_LOCK(*L); }
      ~LockGuard() { POCL_UNLOCK(*L); }
    } LockHolder(&LLVMCtx->Lock);
    llvm::Module *Orig = (llvm::Module *)Program->llvm_irs[DeviceI];
    if (Orig == nullptr) {
      if (ErrorLog)
        *ErrorLog = dupError("no LLVM IR for the program");
      return -1;
    }
    std::unique_ptr<llvm::Module> M = CloneModule(*Orig);
    AirLowering L(*M, *Target);
    if (!L.run(*LocalRegionSize)) {
      if (ErrorLog)
        *ErrorLog = dupError(L.error());
      return -1;
    }
    if (L.GVarSize > 0) {
      *GVarSize = (unsigned)L.GVarSize;
      *GVarInit = (char *)malloc(L.GVarSize);
      memcpy(*GVarInit, L.GVarInit.data(), L.GVarSize);
    }
    if (!L.GVarRelocs.empty()) {
      size_t Bytes = L.GVarRelocs.size() * sizeof(pocl_metal_gvar_reloc_t);
      *GVarRelocs = (pocl_metal_gvar_reloc_t *)malloc(Bytes);
      memcpy(*GVarRelocs, L.GVarRelocs.data(), Bytes);
      *NumGVarRelocs = L.GVarRelocs.size();
    }
    if (pocl_get_bool_option("POCL_METAL_DUMP_AIR", 0)) {
      std::string S;
      raw_string_ostream OS(S);
      M->print(OS, nullptr);
      POCL_MSG_PRINT_METAL("AIR module:\n%s\n", S.c_str());
    }
    if (NamedMDNode *K = M->getNamedMetadata("air.kernel"))
      for (MDNode *N : K->operands())
        if (auto *VM = dyn_cast<ValueAsMetadata>(N->getOperand(0)))
          KernelNames.push_back(VM->getValue()->getName().str());

    raw_svector_ostream OS(Bitcode);
    WriteBitcodeToFile(*M, OS);
  }

  /* Metal's library loader reads LLVM <= 15 bitcode */
  LLVMDGMemoryBufferRef Downgraded = nullptr;
  char *Message = nullptr;
  if (LLVMDGDowngrade(Bitcode.data(), Bitcode.size(), 14, 0, &Downgraded,
                      &Message) != 0) {
    if (ErrorLog)
      *ErrorLog = dupError(std::string("bitcode downgrade failed: ") +
                           (Message ? Message : "(no message)"));
    LLVMDGDisposeMessage(Message);
    return -1;
  }
  LLVMDGDisposeMessage(Message);

  std::vector<uint8_t> Lib = buildMetallib(
      KernelNames, LLVMDGGetBufferStart(Downgraded),
      LLVMDGGetBufferSize(Downgraded), *Target);
  LLVMDGDisposeMemoryBuffer(Downgraded);

  *Metallib = (char *)malloc(Lib.size());
  memcpy(*Metallib, Lib.data(), Lib.size());
  *MetallibSize = Lib.size();
  return 0;
}
