// LLVM module pass to create the single function (fully inlined)
// and parallelized kernel for an OpenCL workgroup.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos
//               2012-2022 Pekka Jääskeläinen / Parform Oy
//               2023-2025 Pekka Jääskeläinen / Intel Finland Oy
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm-c/Core.h>
#include <llvm-c/Target.h>
POP_COMPILER_DIAGS

#include "Barrier.h"
#include "BarrierTailReplication.h"
#include "CanonicalizeBarriers.h"
#include "DebugHelpers.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "ProgramScopeVariables.h"
#include "VariableUniformityAnalysis.h"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"

#include "pocl_file_util.h"
#include "pocl_llvm_api.h"

#include <iostream>
#include <map>
#include <sstream>
#include <string>

#if _WIN32
#  include "vccompat.hpp"
#endif

#define STRING_LENGTH 32

#define PASS_NAME "workgroup"
#define PASS_CLASS pocl::Workgroup
#define PASS_DESC "Workgroup creation pass"

namespace pocl {

using namespace llvm;

enum PoclContextStructFields {
  PC_NUM_GROUPS,
  PC_GLOBAL_OFFSET,
  PC_LOCAL_SIZE,
  PC_PRINTF_BUFFER,
  PC_PRINTF_BUFFER_POSITION,
  PC_PRINTF_BUFFER_CAPACITY,
  PC_GLOBAL_VAR_BUFFER,
  PC_WORK_DIM,
  PC_EXECUTION_FAILED
};

using FunctionVec = std::vector<llvm::Function *>;

using PtrAndType = std::pair<llvm::Value *, llvm::Type *>;

class WorkgroupImpl {
public:
  bool runOnModule(Module &M, llvm::FunctionAnalysisManager &FAM);

private:
  llvm::Function *createWrapper(llvm::Function *F,
                                FunctionMapping &PrintfCache);

  void createGridLauncher(llvm::Function *KernFunc, llvm::Function *WGFunc,
                          std::string KernName);

  llvm::Function *createArgBufferWorkgroupLauncher(llvm::Function *Func,
                                                   std::string KernName);

  void createDefaultWorkgroupLauncher(llvm::Function *F);

  std::vector<llvm::Value *> globalHandlesToContextStructLoads(
      llvm::IRBuilder<> &Builder,
      const std::vector<std::string> &&GlobalHandleNames, int StructFieldIndex);

  void addPlaceHolder(llvm::IRBuilder<> &Builder, llvm::Value *Value,
                      const std::string TypeStr);

  void privatizeGlobalLoads(llvm::Function *F, llvm::IRBuilder<> &Builder,
                            const std::vector<std::string> &&GlobalHandleNames,
                            std::vector<llvm::Value *> PrivateValues);

  void privatizeGlobalStores(llvm::Function *F, llvm::IRBuilder<> &Builder,
                             const std::vector<std::string> &&GlobalHandleNames,
                             std::vector<PtrAndType> PrivatePointers);

  void privatizeContext(llvm::Function *F);

  llvm::Value *createLoadFromContext(llvm::IRBuilder<> &Builder,
                                     int StructFieldIndex, int FieldIndex,
                                     std::string Name);

  PtrAndType getPtrAndTypeForContextVar(IRBuilder<> &Builder,
                                        int StructFieldIndex, int FieldIndex);

  void addGEPs(llvm::IRBuilder<> &Builder, int StructFieldIndex,
               const char *FormatStr);

  void addRangeMetadataForPCField(llvm::Instruction *Instr,
                                  int StructFieldIndex, int FieldIndex = -1);

  LLVMValueRef createAllocaMemcpyForStruct(LLVMModuleRef M,
                                           LLVMBuilderRef Builder,
                                           llvm::Argument &Arg,
                                           LLVMValueRef ArgByteOffset);

  LLVMValueRef createArgBufferLoad(LLVMBuilderRef Builder,
                                   LLVMValueRef ArgBufferPtr,
                                   uint64_t *ArgBufferOffsets,
                                   LLVMContextRef Ctx, LLVMValueRef F,
                                   unsigned ParamIndex, std::string Name);

  llvm::Value *getRequiredSubgroupSize(llvm::Function &F);

  llvm::Module *M;
  llvm::LLVMContext *C;

  // Set to the hidden context argument.
  llvm::Argument *ContextArg;

  // Set to the hidden group_id_* kernel args.
  std::vector<llvm::Value *> GroupIdArgs;

  // Number of hidden args added to the work-group function.
  unsigned HiddenArgs = 0;

  // The width of the size_t data type in the current target.
  int SizeTWidth = 64;
  llvm::Type *SizeT = nullptr;
  llvm::Type *PoclContextT = nullptr;
  llvm::FunctionType *LauncherFuncT = nullptr;

  // Copies of compilation parameters
  std::string KernelName;
  unsigned long AddressBits;
  bool WGAssumeZeroGlobalOffset;
  bool WGDynamicLocalSize;
  bool DeviceUsingArgBufferLauncher;
  bool DeviceUsingGridLauncher;
  bool DeviceIsSPMD;
  unsigned long WGLocalSizeX;
  unsigned long WGLocalSizeY;
  unsigned long WGLocalSizeZ;
  unsigned long WGMaxGridDimWidth;

  unsigned long DeviceGlobalASid;
  unsigned long DeviceLocalASid;
  unsigned long DeviceConstantASid;
  unsigned long DeviceContextASid;
  unsigned long DeviceArgsASid;
  bool DeviceSidePrintf;
  bool DeviceAllocaLocals;
  unsigned long DeviceMaxWItemDim;
  unsigned long DeviceMaxWItemSizes[3];
};

/// Convert address space casts through integer casts to proper address space
/// cast instructions.
///
/// At least the HIP/CUDA frontend since LLVM 19 seems to produce AS casts
/// between generic and global ASs through ptrtoint - inttoptr conversion chains
/// when targeting SPIR-V. It confuses the alias analyzer and leads to lack of
/// utilization of "restrict" pointer information. It should be fixed upstream,
/// but while that happens, let's just peephole convert it.
static bool cleanupAddressSpaceCasts(Function &F) {

  bool Changed = false;
  llvm::IRBuilder<> Builder(F.getContext());
  for (auto &BB : F) {
    for (BasicBlock::iterator BI = BB.begin(), BE = BB.end(); BI != BE;) {

      // The casts look like this:
      // %55 = ptrtoint ptr addrspace(1) %7 to i64
      // %56 = inttoptr i64 %55 to ptr addrspace(4)

      PtrToIntInst *PtrToInt = dyn_cast<PtrToIntInst>(&*BI);
      if (PtrToInt != nullptr &&
          (PtrToInt->getPointerAddressSpace() == SPIR_ADDRESS_SPACE_GLOBAL ||
           PtrToInt->getPointerAddressSpace() == SPIR_ADDRESS_SPACE_GENERIC) &&
          PtrToInt->hasOneUse()) {
        IntToPtrInst *User =
            dyn_cast<IntToPtrInst>(*PtrToInt->users().begin());
        if (User == nullptr ||
            (User->getAddressSpace() != SPIR_ADDRESS_SPACE_GLOBAL &&
             User->getAddressSpace() != SPIR_ADDRESS_SPACE_GENERIC))
          break;
        Builder.SetInsertPoint(PtrToInt);
        AddrSpaceCastInst *AddrSpaceCast =
            cast<AddrSpaceCastInst>(Builder.CreatePointerBitCastOrAddrSpaceCast(
                PtrToInt->getPointerOperand(), User->getType()));
        User->replaceAllUsesWith(AddrSpaceCast);
        User->eraseFromParent();
        PtrToInt->eraseFromParent();
        BI = AddrSpaceCast->getIterator();
        BE = BB.end();
        Changed = true;
      }
      ++BI;
    }
  }
  return Changed;
}

bool WorkgroupImpl::runOnModule(Module &M, llvm::FunctionAnalysisManager &FAM) {

  this->M = &M;
  this->C = &M.getContext();

  getModuleIntMetadata(M, "device_address_bits", AddressBits);
  getModuleBoolMetadata(M, "device_arg_buffer_launcher",
                        DeviceUsingArgBufferLauncher);
  getModuleBoolMetadata(M, "device_grid_launcher",
                        DeviceUsingGridLauncher);
  getModuleBoolMetadata(M, "device_is_spmd", DeviceIsSPMD);

  getModuleStringMetadata(M, "KernelName", KernelName);
  getModuleIntMetadata(M, "WGMaxGridDimWidth", WGMaxGridDimWidth);
  getModuleIntMetadata(M, "WGLocalSizeX", WGLocalSizeX);
  getModuleIntMetadata(M, "WGLocalSizeY", WGLocalSizeY);
  getModuleIntMetadata(M, "WGLocalSizeZ", WGLocalSizeZ);
  getModuleBoolMetadata(M, "WGDynamicLocalSize", WGDynamicLocalSize);
  getModuleBoolMetadata(M, "WGAssumeZeroGlobalOffset",
                        WGAssumeZeroGlobalOffset);

  getModuleIntMetadata(M, "device_global_as_id", DeviceGlobalASid);
  getModuleIntMetadata(M, "device_local_as_id", DeviceLocalASid);
  getModuleIntMetadata(M, "device_constant_as_id", DeviceConstantASid);
  getModuleIntMetadata(M, "device_args_as_id", DeviceArgsASid);
  getModuleIntMetadata(M, "device_context_as_id", DeviceContextASid);

  getModuleBoolMetadata(M, "device_side_printf", DeviceSidePrintf);
  getModuleBoolMetadata(M, "device_alloca_locals", DeviceAllocaLocals);

  getModuleIntMetadata(M, "device_max_witem_dim", DeviceMaxWItemDim);
  getModuleIntMetadata(M, "device_max_witem_sizes_0", DeviceMaxWItemSizes[0]);
  getModuleIntMetadata(M, "device_max_witem_sizes_1", DeviceMaxWItemSizes[1]);
  getModuleIntMetadata(M, "device_max_witem_sizes_2", DeviceMaxWItemSizes[2]);

  HiddenArgs = 0;
  SizeTWidth = AddressBits;
  SizeT = pocl::SizeT(&M);

  llvm::Type *Int32T = Type::getInt32Ty(*C);
  llvm::Type *Int8T = Type::getInt8Ty(*C);
  PoclContextT = StructType::get(
      ArrayType::get(SizeT, 3),                   // NUM_GROUPS
      ArrayType::get(SizeT, 3),                   // GLOBAL_OFFSET
      ArrayType::get(SizeT, 3),                   // LOCAL_SIZE
      PointerType::get(Int8T, DeviceGlobalASid),  // PRINTF_BUFFER
      PointerType::get(Int32T, DeviceGlobalASid), // PRINTF_BUFFER_POSITION
      Int32T,                                     // PRINTF_BUFFER_CAPACITY
      PointerType::get(Int8T, DeviceGlobalASid),  // GLOBAL_VAR_BUFFER
      Int32T,                                     // WORK_DIM
      Int32T);                                    // EXECUTION_FAILED

  LauncherFuncT = FunctionType::get(
      Type::getVoidTy(*C),
      {PointerType::get(PointerType::get(Type::getInt8Ty(*C), 0),
                        DeviceArgsASid),
       PointerType::get(PoclContextT, DeviceContextASid), SizeT, SizeT, SizeT},
      false);

  assert ((SizeTWidth == 64 || SizeTWidth == 32) &&
          "Target has an unsupported pointer width.");

  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    // Don't internalize functions starting with "__wrap_" for the use of GNU
    // linker's switch --wrap=symbol, where calls to the "symbol" are replaced
    // with "__wrap_symbol" at link time.  These functions may not be referenced
    // until final link and being deleted by LLVM optimizations before it.
    // Also, if there's a main aux function, we want to keep it as it will be
    // likely used in the kernel command functionality for the device.
    if (!i->isDeclaration() && !i->getName().starts_with("__wrap_") &&
        i->getName() != "main")
      i->setLinkage(Function::InternalLinkage);
  }

  for (Function &F : M.functions()) {
    if (F.isDeclaration())
      continue;
    cleanupAddressSpaceCasts(F);
  }

  // Store the new and old kernel pairs in order to regenerate
  // all the metadata that used to point to the unmodified
  // kernels.
  FunctionMapping KernelsMap;

  // Mapping of all functions which have been transformed to take
  // extra printf arguments.
  FunctionMapping PrintfCache;

  // TODO perhaps pass "-cl-opt-disable" build opt as bool module metadata
  bool IsDebugEnabled = false;
  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    Function &OrigKernel = *i;
    if (!isKernelToProcess(OrigKernel)) continue;
    if (OrigKernel.hasMetadata("dbg"))
      IsDebugEnabled = true;
    Function *L = createWrapper(&OrigKernel, PrintfCache);
    KernelsMap[&OrigKernel] = L;

    privatizeContext(L);

    if (DeviceUsingArgBufferLauncher) {
      Function *WGLauncher =
        createArgBufferWorkgroupLauncher(L, OrigKernel.getName().str());

      WGLauncher->addFnAttr(Attribute::AlwaysInline);
      WGLauncher->removeFnAttr(Attribute::OptimizeNone);
      if (DeviceUsingGridLauncher)
        createGridLauncher(L, WGLauncher, OrigKernel.getName().str());
    } else if (DeviceIsSPMD) {
      // For SPMD machines there is no need for a WG launcher, the device will
      // call/handle the single-WI kernel function directly.
    } else {
      createDefaultWorkgroupLauncher(L);
    }
  }

  if (!DeviceUsingArgBufferLauncher && DeviceIsSPMD) {
    regenerate_kernel_metadata(M, KernelsMap);
  }

  // Delete the old kernels. They are inlined into the wrapper, and
  // they contain references to global variables (_local_id_x etc)
  for (FunctionMapping::const_iterator i = KernelsMap.begin(), e = KernelsMap.end();
       i != e; ++i) {
    Function *OldKernel = i->first;
    Function *NewKernel = i->second;
    // this should not happen
    assert(OldKernel != NewKernel);
    FAM.clear(*OldKernel, "parallel.bc");
    OldKernel->eraseFromParent();
  }

  // remove all functions that call the old printf. They should be
  // cloned with the new printf calls at this point, and they refer
  // to global variables
  Function *NewPrintfAlloc = M.getFunction("pocl_printf_alloc");
  Function *OldPrintfAlloc = M.getFunction("pocl_printf_alloc_stub");

  if (NewPrintfAlloc && OldPrintfAlloc) {
    // add functions that call OldPrintfAlloc but are not used anywhere
    // to PrintfCache; this can happen e.g. b/c of inlining
    std::set<llvm::Function *> Removed;
    for (auto U : OldPrintfAlloc->users()) {
      if (Instruction *I = dyn_cast<Instruction>(U)) {
        auto OldF = I->getParent()->getParent();
        if (OldF == nullptr)
          continue;
        if (OldF->getNumUses() != 0)
          continue;
        if (PrintfCache.find(OldF) == PrintfCache.end()) {
          PrintfCache.insert(std::make_pair(OldF, OldF));
        }
      }
    }

    // remove the old functions that were cloned
    bool AtLeastOneRemoved;
    do {
      AtLeastOneRemoved = false;
      for (auto [OldF, NewF] : PrintfCache) {
        if (Removed.count(OldF))
          continue;
        if (OldF->getNumUses() == 0) {
          FAM.clear(*OldF, "parallel.bc");
          OldF->eraseFromParent();
          Removed.insert(OldF);
          AtLeastOneRemoved = true;
        }
      }
    } while (AtLeastOneRemoved);

    if (OldPrintfAlloc->getNumUses() == 0) {
      OldPrintfAlloc->eraseFromParent();
    } else {
      // if we still have users, at least replace the body
      // with ret nullptr. This allows us to erase
      // the global variables

      // drop all BBs
      while (!OldPrintfAlloc->empty())
        OldPrintfAlloc->back().eraseFromParent();

      // create BB with "ret nullptr"
      BasicBlock *BB =
          BasicBlock::Create(M.getContext(), "entry", OldPrintfAlloc);
      auto PtrTy = cast<PointerType>(OldPrintfAlloc->getReturnType());
      ConstantPointerNull *NullPtr = ConstantPointerNull::get(PtrTy);

      llvm::IRBuilder<> Builder(M.getContext());
      Builder.SetInsertPoint(BB);
      Builder.CreateRet(NullPtr);
    }
    // remove the global variables
    GlobalVariable *GV;
    GV = M.getGlobalVariable("_printf_buffer");
    if (GV && GV->getNumUses() == 0)
      GV->eraseFromParent();
    GV = M.getGlobalVariable("_printf_buffer_position");
    if (GV && GV->getNumUses() == 0)
      GV->eraseFromParent();
    GV = M.getGlobalVariable("_printf_buffer_capacity");
    if (GV && GV->getNumUses() == 0)
      GV->eraseFromParent();
  }

  // Remove all WI-related functions and global variables;
  // all of these should be privatized now. Functions first because
  // they still refer to the global variables
  for (auto Name : WIFuncNameVec) {
    Function *F = M.getFunction(Name);
    if (F && F->getNumUses() == 0) {
      FAM.clear(*F, "parallel.bc");
      F->eraseFromParent();
    }
  }

  for (auto Name : WorkgroupVariablesVector) {
    GlobalVariable *GV = M.getGlobalVariable(Name);
    if (GV && GV->getNumUses() == 0)
      GV->eraseFromParent();
  }

  // remove the declaration of the pocl.barrier placeholder
  if (auto *F = M.getFunction(BARRIER_FUNCTION_NAME)) {
    FAM.clear(*F, "parallel.bc");
    F->eraseFromParent();
  }

  // Unify some attributes among all functions; this is necessary because
  // -O0 compiled builtin library has many attributes setup differently than
  // -O2 compiled user code, and the inliner will check&refuse to inline.
  // See hasCompatibleFnAttrs in llvm/include/llvm/IR/Attributes.inc
  // and getAttributeBasedInliningDecision in llvm/lib/Analysis/InlineCost.cpp
  std::string TargetCPU;
  std::string TargetFeatures;
  for (Function &F : M) {
    if (TargetCPU.empty() && F.hasFnAttribute("target-cpu")) {
      TargetCPU = F.getFnAttribute("target-cpu").getValueAsString().str();
    }
    if (TargetFeatures.empty() && F.hasFnAttribute("target-features")) {
      TargetFeatures =
          F.getFnAttribute("target-features").getValueAsString().str();
    }
  }

  for (Function &F : M.functions()) {
    if (F.isDeclaration())
      continue;

    // removes optnone/noinline/alwaysinline from F & its callsites
    markFunctionAlwaysInline(&F);
    F.removeFnAttr(Attribute::AlwaysInline);

    // these are checked in LLVM's areInlineCompatible -> hasCompatibleFnArgs
    F.removeFnAttr(Attribute::SafeStack);
    F.removeFnAttr(Attribute::ShadowCallStack);
    F.removeFnAttr(Attribute::NoProfile);
    F.removeFnAttr(Attribute::StackProtect);
    F.removeFnAttr(Attribute::StackProtectReq);
    F.removeFnAttr(Attribute::StackProtectStrong);
    // denorm & strictFP must be compatible
    // TODO strictfp should depend on program's build opts
    // denormals are handled by the driver / execution environment
    F.removeFnAttr(Attribute::StrictFP);
    F.removeFnAttr("denormal-fp-math");
    F.removeFnAttr("denormal-fp-math-f32");
    // prevents vectorizing/inlining of functions with builtins (llvm.sin.f32 etc)
    F.removeFnAttr("no-builtins");
    F.addFnAttr("no-trapping-math", "true");
    // required because the InlineCost -> areInlineCompatible also calls
    // TTI->areInlineCompatible; X86TTIImpl::areInlineCompatible checks CPU
    // features; this might require even more attrs (target-abi?) for RISC-V
    if (!TargetCPU.empty())
      F.addFnAttr("target-cpu", TargetCPU);
    if (!TargetFeatures.empty())
      F.addFnAttr("target-features", TargetFeatures);

    // the following settings are not necessary for inlining,
    // but should improve optimization
    F.removeFnAttr("use-sample-profile");
    F.removeFnAttr("stack-protector-buffer-size");
    if (IsDebugEnabled)
      F.addFnAttr("frame-pointer", "all");
    else
      F.removeFnAttr("frame-pointer");

    // these should be safe to enable
    F.addFnAttr(Attribute::NoFree);
    // recursion forbidden by OpenCL
    F.addFnAttr(Attribute::NoRecurse);
    // no exceptions or return by callback
    F.addFnAttr(Attribute::NoCallback);
    F.addFnAttr(Attribute::NoUnwind);
#if 0
    // It appears OpenCL follows C99 in which they are NOT UB.
    // This breaks infinite for loops when enabled.
    F.addFnAttr(Attribute::WillReturn);
#endif

    // Override the preferred vector width on x86 targets.
    // By default, clang uses 256-bit even if a processor supports 512-bit SIMD.
    if (int VecWidth =
            pocl_get_int_option("POCL_VECTORIZER_PREFER_VECTOR_WIDTH", 0)) {
      F.addFnAttr("prefer-vector-width", std::to_string(VecWidth));
    }
  }

  return true;
}

// Ensures the given value is not optimized away even if it's not used
// by LLVM IR.
void WorkgroupImpl::addPlaceHolder(llvm::IRBuilder<> &Builder, llvm::Value *Val,
                                   const std::string TypeStr = "r") {

  // For the lack of a better holder, add a dummy inline asm that reads the
  // arg arguments.
  FunctionType *DummyIAType =
    FunctionType::get(Type::getVoidTy(M->getContext()), Val->getType(),
                      false);

  llvm::InlineAsm *DummyIA =
    llvm::InlineAsm::get(DummyIAType, "", TypeStr, false, false);
  Builder.CreateCall(DummyIA, Val);
}

// Adds Range metadata with range [Min, Max] to the given instruction.
static void addRangeMetadata(llvm::Instruction *Instr, size_t Min, size_t Max) {
  MDBuilder MDB(Instr->getContext());
  size_t BitWidth = Instr->getType()->getIntegerBitWidth();
  MDNode *Range =
      MDB.createRange(APInt(BitWidth, Min), APInt(BitWidth, Max + 1));
  Instr->setMetadata(LLVMContext::MD_range, Range);
}

void WorkgroupImpl::addRangeMetadataForPCField(llvm::Instruction *Instr,
                                               int StructFieldIndex,
                                               int FieldIndex) {
  uint64_t Min = 0;
  uint64_t Max = 0;
  uint64_t LocalSizes[] = {WGLocalSizeX, WGLocalSizeY, WGLocalSizeZ};
  switch (StructFieldIndex) {
  case PC_WORK_DIM:
    Min = 1;
    Max = DeviceMaxWItemDim;
    break;
  case PC_NUM_GROUPS:
    Min = 1;
    switch (FieldIndex) {
    case 0:
    case 1:
    case 2: {
      Max = WGMaxGridDimWidth > 0 ? WGMaxGridDimWidth : 0;
      if (!WGDynamicLocalSize) {
        // If we know also the local size, we can minimize the known max group
        // count by dividing by it. Upwards rounding due to 2.0 partial WGs.
        Max = (Max + LocalSizes[FieldIndex] - 1) / LocalSizes[FieldIndex];
      }
      break;
    }
    default:
      llvm_unreachable("More than 3 grid dimensions unsupported.");
    }
    break;
  case PC_GLOBAL_OFFSET:
    switch (FieldIndex) {
    case 0:
    case 1:
    case 2:
      // WGAssumeZeroGlobalOffset will be used to convert to a constant 0, so
      // here we just speculate on the range in case of non-zero offset.
      Max = WGMaxGridDimWidth;
      break;
    default:
      llvm_unreachable("More than 3 grid dimensions unsupported.");
    }
    break;
  case PC_LOCAL_SIZE:
    Min = 1;
    switch (FieldIndex) {
    case 0:
    case 1:
    case 2:
      if (WGDynamicLocalSize) {
        Max = (WGMaxGridDimWidth > 0 ? WGMaxGridDimWidth
                                     : std::min(DeviceMaxWItemSizes[FieldIndex],
                                                WGMaxGridDimWidth));
      } else {
        // The local size is converted to constant with static WGs, so this is
        // actually useless.
        Max = LocalSizes[FieldIndex];
      }
      break;
    default:
      llvm_unreachable("More than 3 grid dimensions unsupported.");
    }
    break;
  default:
    break;
  }
  if (Max > 0) {
    addRangeMetadata(Instr, Min, Max);
#if 0
    std::cerr << "Added range [" << Min << ", " << Max << "] " << std::endl;
    std::cerr << StructFieldIndex << " " << FieldIndex << std::endl;
#endif
  }
  return;
}

// Computes the Pointer and the Type of a PoclContext variable
PtrAndType WorkgroupImpl::getPtrAndTypeForContextVar(IRBuilder<> &Builder,
                                                     int StructFieldIndex,
                                                     int FieldIndex = -1) {
  Value *GEP, *Ptr;
  GEP = Builder.CreateStructGEP(PoclContextT, ContextArg, StructFieldIndex);
  Type *GEPType = PoclContextT->getStructElementType(StructFieldIndex);

  if (SizeTWidth == 64) {
    if (FieldIndex == -1)
      Ptr = Builder.CreateConstGEP1_64(GEPType, GEP, 0);
    else
      Ptr = Builder.CreateConstGEP2_64(GEPType, GEP, 0, FieldIndex);
  } else {
    if (FieldIndex == -1)
      Ptr = Builder.CreateConstGEP1_32(GEPType, GEP, 0);
    else
      Ptr = Builder.CreateConstGEP2_32(GEPType, GEP, 0, FieldIndex);
  }

  Type *FinalType = GEPType;
  if (FieldIndex >= 0) {
    ArrayType *AT = nullptr;
    AT = dyn_cast<ArrayType>(GEPType);
    assert(AT);
    FinalType = AT->getArrayElementType();
  }

  return PtrAndType{Ptr, FinalType};
}

// Creates a load from the hidden context structure argument for
// the given element.
llvm::Value *WorkgroupImpl::createLoadFromContext(IRBuilder<> &Builder,
                                                  int StructFieldIndex,
                                                  int FieldIndex = -1,
                                                  std::string Name = "") {

  llvm::LoadInst *Load = nullptr;
  PtrAndType PT =
      getPtrAndTypeForContextVar(Builder, StructFieldIndex, FieldIndex);
  Load = Builder.CreateLoad(PT.second, PT.first, Name.c_str());
  addRangeMetadataForPCField(Load, StructFieldIndex, FieldIndex);
  return Load;
}

// TODO we should use printf users instead of searching the call tree
static bool callsPrintf(Function *F) {
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {
      Instruction *Instr = dyn_cast<Instruction>(BI);
      if (!llvm::isa<CallInst>(Instr))
        continue;
      CallInst *CallInstr = dyn_cast<CallInst>(Instr);
      if (CallInstr->isInlineAsm())
        continue;
      Function *callee = CallInstr->getCalledFunction();

      if (callee->getName() == "llvm.")
        continue;

      if (callee->getName() == "pocl_printf_alloc_stub")
        return true;
      if (callee->getName() == "pocl_printf_alloc")
        return true;
      if (callee->getName() == "pocl_flush_printf_buffer")
        return true;
      if (callsPrintf(callee))
        return true;
    }
  }
  return false;
}

// Clones a function while adding 3 new arguments for printf calls.
static Function *cloneFunctionWithPrintfArgs(Value *pb, Value *pbp, Value *pbc,
                                             Function *F, Module *M) {

  SmallVector<Type *, 8> Parameters;

  Parameters.push_back(pb->getType());
  Parameters.push_back(pbp->getType());
  Parameters.push_back(pbc->getType());

  for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
       i != e; ++i)
    Parameters.push_back(i->getType());

  // Create the new function.
  FunctionType *FT =
      FunctionType::get(F->getReturnType(), Parameters, F->isVarArg());
  Function *NewF = Function::Create(FT, F->getLinkage(), "", M);
  NewF->setName(F->getName() + ".with_printf");

  ValueToValueMapTy VV;
  Function::arg_iterator j = NewF->arg_begin();
  j->setName("print_buffer");
  ++j;
  j->setName("print_buffer_position");
  ++j;
  j->setName("print_buffer_capacity");
  ++j;
  for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
       i != e; ++i) {
    j->setName(i->getName());
    VV[&*i] = &*j;
    ++j;
  }

  SmallVector<ReturnInst *, 1> RI;

  // As of LLVM 5.0 we need to let CFI to make module level changes,
  // otherwise there will be an assertion. The changes are likely
  // additional debug info nodes added when cloning the function into
  // the other.  For some reason it doesn't want to reuse the old ones.
  CloneFunctionIntoAbs(NewF, F, VV, RI);

  return NewF;
}

// Recursively replace pocl_printf_alloc_stub calls with pocl_printf_alloc calls,
// while propagating the required pocl_context->printf_buffer arguments.
static void replacePrintfCalls(Value *pb, Value *pbp, Value *pbc, bool isKernel,
                               Function *NewPrintfAlloc,
                               Function *ReplacedPrintfAlloc, Module &M,
                               Function *L, FunctionMapping &printfCache) {

  // If none of the kernels use printf(), it will not be linked into the
  // module.
  if (NewPrintfAlloc == nullptr || ReplacedPrintfAlloc == nullptr) {
    return;
  }

  // For kernel function, we are provided with proper printf arguments;
  // for non-kernel functions, we assume the function was replaced with
  // cloneFunctionWithPrintfArgs() and use the first three arguments.
  if (!isKernel) {
    auto i = L->arg_begin();
    pb = &*i;
    ++i;
    pbp = &*i;
    ++i;
    pbc = &*i;
  }

  SmallDenseMap<CallInst *, CallInst *> replaceCIMap(16);
  SmallVector<Value *, 8> ops;
  SmallVector<CallInst *, 32> callsToCheck;

  // First, replace printf calls in body of L.
  for (Function::iterator I = L->begin(), E = L->end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {
      Instruction *Instr = dyn_cast<Instruction>(BI);
      if (!llvm::isa<CallInst>(Instr))
        continue;
      CallInst *CallInstr = dyn_cast<CallInst>(Instr);
      if (CallInstr->isInlineAsm())
        continue;
      Function *oldF = CallInstr->getCalledFunction();

      // Skip inline asm blocks.
      if (oldF == nullptr)
        continue;

      if (oldF == ReplacedPrintfAlloc) {
        ops.clear();
        ops.push_back(pb);
        ops.push_back(pbp);
        ops.push_back(pbc);

        unsigned j = CallInstr->getNumOperands() - 1;
        for (unsigned i = 0; i < j; ++i) {
          auto *Operand = CallInstr->getOperand(i);
          ops.push_back(Operand);
        }

        CallInst *NewCI = CallInst::Create(NewPrintfAlloc, ops);
        NewCI->setCallingConv(NewPrintfAlloc->getCallingConv());
        auto *CB = dyn_cast<CallBase>(CallInstr);
        NewCI->setTailCall(CB->isTailCall());

        replaceCIMap.insert(
            std::pair<CallInst *, CallInst *>(CallInstr, NewCI));
      } else {
        if (!oldF->getName().starts_with("llvm."))
          callsToCheck.push_back(CallInstr);
      }
    }
  }

  // replace printf calls
  for (auto it : replaceCIMap) {
    CallInst *CI = it.first;
    CallInst *newCI = it.second;

    // LLVM may modify the result type of the called function to void.
    if (CI->getType()->isVoidTy()) {
#if LLVM_MAJOR < 20
      newCI->insertBefore(CI);
#else
      newCI->insertBefore(CI->getIterator());
#endif
      CI->eraseFromParent();
    } else {
      CI->replaceAllUsesWith(newCI);
      ReplaceInstWithInst(CI, newCI);
    }
  }

  replaceCIMap.clear();

  // check each called function recursively
  for (auto it : callsToCheck) {
    CallInst *CI = it;
    CallInst *NewCI = nullptr;
    Function *oldF = CI->getCalledFunction();
    Function *newF = nullptr;
    bool needsPrintf = false;
    auto i = printfCache.find(oldF);
    if (i != printfCache.end()) {
      // function was already cloned
      needsPrintf = true;
      newF = i->second;
    } else {
      // create new clone
      needsPrintf = callsPrintf(oldF);
      if (needsPrintf) {
        newF = cloneFunctionWithPrintfArgs(pb, pbp, pbc, oldF, &M);
        replacePrintfCalls(nullptr, nullptr, nullptr, false, NewPrintfAlloc,
                           ReplacedPrintfAlloc, M, newF, printfCache);

        printfCache.insert(
            std::pair<llvm::Function *, llvm::Function *>(oldF, newF));
      }
    }

    // if the called function calls Printf, replace with newF and add arguments
    if (needsPrintf) {
      ops.clear();
      ops.push_back(pb);
      ops.push_back(pbp);
      ops.push_back(pbc);
      unsigned j = CI->getNumOperands() - 1;
      for (unsigned i = 0; i < j; ++i)
        ops.push_back(CI->getOperand(i));

      NewCI = CallInst::Create(newF, ops);
      replaceCIMap.insert(std::pair<CallInst *, CallInst *>(CI, NewCI));
      NewCI->setCallingConv(CI->getCallingConv());
    }
  }

  for (auto it : replaceCIMap) {
    CallInst *CI = it.first;
    CallInst *newCI = it.second;

    CI->replaceAllUsesWith(newCI);
    ReplaceInstWithInst(CI, newCI);
  }
}

// Create a wrapper named "_pocl_kernel_<original-kernel-name>"
// for the kernel and add pocl-specific hidden arguments.
// Also inlines the wrapped function to the wrapper.
Function *WorkgroupImpl::createWrapper(Function *F,
                                       FunctionMapping &PrintfCache) {

  SmallVector<Type *, 8> FuncParams;
  LLVMContext &C = M->getContext();
  for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
       i != e; ++i)
    FuncParams.push_back(i->getType());

  if (!DeviceUsingArgBufferLauncher && DeviceIsSPMD) {
    FuncParams.push_back(PointerType::get(PoclContextT, DeviceContextASid));
    HiddenArgs = 1;
  } else {
    // pocl_context
    FuncParams.push_back(PointerType::get(PoclContextT, DeviceContextASid));
    // group_x
    FuncParams.push_back(SizeT);
    // group_y
    FuncParams.push_back(SizeT);
    // group_z
    FuncParams.push_back(SizeT);

    // we might not have all of the globals anymore in the module in case the
    // kernel does not refer to them and they are optimized away
    HiddenArgs = 4;
  }

  FunctionType *FuncT =
      FunctionType::get(Type::getVoidTy(C), ArrayRef<Type *>(FuncParams), false);

  std::string FuncName = F->getName().str();
  Function *L = NULL;
  if (!DeviceUsingArgBufferLauncher && DeviceIsSPMD) {
    Function *F = M->getFunction(FuncName);
    F->setName(FuncName + "_original");
    L = Function::Create(FuncT, Function::ExternalLinkage, FuncName, M);
  } else {
    L = Function::Create(FuncT, Function::ExternalLinkage,
                         "_pocl_kernel_" + FuncName, M);
  }

  SmallVector<Value *, 8> FuncArgs;
  Function::arg_iterator ai = L->arg_begin();
  for (unsigned i = 0, e = F->arg_size(); i != e; ++i) {
    FuncArgs.push_back(&*ai);
    ++ai;
  }

  ContextArg = &*(ai++);
  GroupIdArgs.resize(3);
  GroupIdArgs[0] = &*(ai++);
  GroupIdArgs[1] = &*(ai++);
  GroupIdArgs[2] = &*(ai++);

  // Copy the function attributes to transfer noalias etc. from the
  // original kernel which will be inlined into the launcher.
  L->setAttributes(F->getAttributes());

  // At least the argument address space metadata is useful. The argument
  // indices should still hold even though we appended the hidden args.
  // Note: This also copies the DISubprogram !dbg, if any. We have to retain
  // a valid DISubprogram for correctness for enabling debug output.
  L->copyMetadata(F, 0);

  if (F->getSubprogram() != nullptr) {
    L->setSubprogram(
        pocl::mimicDISubprogram(F->getSubprogram(), L->getName(), nullptr));
  }

  // We need to mark the generated function to avoid it being considered a
  // new kernel to process (which results in infinite recursion). This is
  // because kernels are detected by the presense of the argument metadata
  // we just copied from the original kernel function.
  L->setMetadata("pocl_generated", MDNode::get(C, {createConstantIntMD(C, 1)}));

  IRBuilder<> Builder(BasicBlock::Create(C, "", L));

  Value *PrintfBuf, *PrintfBufPos, *PrintfBufCapa;
  PrintfBuf = PrintfBufPos = PrintfBufCapa = nullptr;
  if (DeviceSidePrintf) {
    PrintfBuf =
        createLoadFromContext(Builder, PC_PRINTF_BUFFER, -1, "printf_buffer");
    PrintfBufPos = createLoadFromContext(Builder, PC_PRINTF_BUFFER_POSITION, -1,
                                         "printf_buffer_write_pos");
    PrintfBufCapa = createLoadFromContext(Builder, PC_PRINTF_BUFFER_CAPACITY,
                                          -1, "printf_buffer_capacity");
  }

  CallInst *CI = Builder.CreateCall(F, ArrayRef<Value *>(FuncArgs));
  Builder.CreateRetVoid();

  if (L->getSubprogram() != nullptr && F->getSubprogram() != nullptr) {
    CI->setDebugLoc(llvm::DILocation::get(CI->getContext(),
                                          F->getSubprogram()->getLine(), 0,
                                          L->getSubprogram(), nullptr, true));
  }

  // SPMD machines might need a special calling convention to mark the
  // kernels that should be executed in SPMD fashion. For MIMD/CPU,
  // we want to use the default calling convention for the work group
  // function.
  if (DeviceIsSPMD)
    L->setCallingConv(F->getCallingConv());

  // needed for printf
  InlineFunctionInfo IFI;
  InlineFunction(*CI, IFI);

  if (DeviceSidePrintf) {
    Function *NewPrintfAlloc = M->getFunction("pocl_printf_alloc");
    Function *OldPrintfAlloc = M->getFunction("pocl_printf_alloc_stub");
    if (NewPrintfAlloc && OldPrintfAlloc) {
      replacePrintfCalls(PrintfBuf, PrintfBufPos, PrintfBufCapa, true,
                         NewPrintfAlloc, OldPrintfAlloc, *M, L, PrintfCache);
      markFunctionAlwaysInline(NewPrintfAlloc);
    }
  }

  return L;
}

// Converts the given global context variable handles to loads from the
// hidden context struct argument. If there is no reference to the global,
// the corresponding entry in the returned vector will contain a nullptr.
std::vector<llvm::Value *> WorkgroupImpl::globalHandlesToContextStructLoads(
    IRBuilder<> &Builder, const std::vector<std::string> &&GlobalHandleNames,
    int StructFieldIndex) {

  std::vector<Value*> StructLoads(GlobalHandleNames.size());
  for (size_t i = 0; i < GlobalHandleNames.size(); ++i) {
    if (M->getGlobalVariable(GlobalHandleNames.at(i)) == nullptr) {
      StructLoads[i] = nullptr;
      continue;
    }
    StructLoads[i] = createLoadFromContext(
        Builder, StructFieldIndex, GlobalHandleNames.size() == 1 ? -1 : i);
  }
  return StructLoads;
}

// Converts Load uses of the given pseudo variable handles (magic external
// global variables) to Load from the given function-private values instead.
void WorkgroupImpl::privatizeGlobalLoads(
    llvm::Function *F, llvm::IRBuilder<> &Builder,
    const std::vector<std::string> &&GlobalHandleNames,
    std::vector<llvm::Value *> PrivateValues) {

  for (Function::iterator i = F->begin(), e = F->end();
       i != e; ++i) {
    for (BasicBlock::iterator ii = i->begin(), ee = i->end(),
           Next = std::next(ii); ii != ee; ii = Next) {
      Next = std::next(ii);
      for (size_t j = 0; j < GlobalHandleNames.size(); ++j) {
        if (PrivateValues[j] == nullptr) {
          continue;
        }
        if (!isa<llvm::LoadInst>(ii)) {
          continue;
        }
        llvm::GlobalValue *GlobalHandle =
            M->getGlobalVariable(GlobalHandleNames.at(j));
        if (GlobalHandle == nullptr)
          continue;

        llvm::LoadInst *L = cast<llvm::LoadInst>(ii);
        if (L->getPointerOperand()->stripPointerCasts() != GlobalHandle)
          continue;
        llvm::Value *Cast = PrivateValues[j];
        if (L->getType() != PrivateValues[j]->getType())
          Cast = Builder.CreateTruncOrBitCast(PrivateValues[j], L->getType());
        ii->replaceAllUsesWith(Cast);
        ii->eraseFromParent();
        break;
      }
    }
  }
}

// Converts Store uses of the given pseudo variable handles (magic external
// global variables) to Store into the given function-private pointers instead.
void WorkgroupImpl::privatizeGlobalStores(
    llvm::Function *F, llvm::IRBuilder<> &Builder,
    const std::vector<std::string> &&GlobalHandleNames,
    std::vector<PtrAndType> PrivatePointers) {

  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
    for (BasicBlock::iterator ii = i->begin(), ee = i->end(), Next = ii;
         ii != ee; ii = Next) {
      Next = std::next(ii);
      for (size_t j = 0; j < GlobalHandleNames.size(); ++j) {
        if (PrivatePointers[j].first == nullptr) {
          continue;
        }
        if (!isa<llvm::StoreInst>(ii)) {
          continue;
        }
        llvm::GlobalValue *GlobalHandle =
            M->getGlobalVariable(GlobalHandleNames.at(j));
        if (GlobalHandle == nullptr)
          continue;

        llvm::StoreInst *S = cast<llvm::StoreInst>(ii);
        if (S->getPointerOperand()->stripPointerCasts() != GlobalHandle)
          continue;
        llvm::Value *StoredVal = S->getValueOperand();
        if (S->getValueOperand()->getType() != PrivatePointers[j].second)
          StoredVal = Builder.CreateTruncOrBitCast(StoredVal,
                                                   PrivatePointers[j].second);

        Builder.SetInsertPoint(ii->getParent(), ii->getIterator());
        Builder.CreateStore(StoredVal, PrivatePointers[j].first);
        ii->eraseFromParent();

        break;
      }
    }
  }
  // restore the insertion point for Loads
  Builder.SetInsertPoint(&F->front(), F->front().getFirstInsertionPt());
}

/**
 * Makes the work-item context data function private.
 *
 * Until this point all the work-group generation passes have referred to
 * magic global variables to access the work-item identifiers. These are
 * converted to kernel-local allocas by this function.
 */
void WorkgroupImpl::privatizeContext(Function *F) {

  // Privatize _global_id_* to private allocas.
  // They are referred to by WorkItemLoops to fetch the global id directly.

  CreateBuilder(Builder, F->getEntryBlock());

  // For replace the global_ids with local allocas for easier
  // data flow analysis.
  std::vector<Value *> GlobalIdAllocas(3);
  for (int i = 0; i < 3; ++i) {
    if (M->getGlobalVariable(GID_G_NAME(i)) == nullptr)
      continue;
    GlobalIdAllocas[i] = Builder.CreateAlloca(SizeT, 0, GID_G_NAME(i));
  }

  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
    for (BasicBlock::iterator ii = i->begin(), ee = i->end(); ii != ee; ++ii) {
      for (int j = 0; j < 3; ++j) {
        if (M->getGlobalVariable(GID_G_NAME(j)) == nullptr)
          continue;
        ii->replaceUsesOfWith(M->getGlobalVariable(GID_G_NAME(j)),
                              GlobalIdAllocas[j]);
      }
    }
  }

  char TempStr[STRING_LENGTH];

  std::vector<GlobalVariable*> LocalIdGlobals(3);
  std::vector<AllocaInst*> LocalIdAllocas(3);
  // Privatize _local_id to allocas. They are used as iteration variables in
  // WorkItemLoops, thus referred to later on.
  for (int i = 0; i < 3; ++i) {
    snprintf(TempStr, STRING_LENGTH, "_local_id_%c", 'x' + i);
    LocalIdGlobals[i] = M->getGlobalVariable(TempStr);
    if (LocalIdGlobals[i] != NULL) {
      LocalIdAllocas[i] =
        Builder.CreateAlloca(LocalIdGlobals[i]->getValueType(), 0,
                             TempStr);
      if (LocalIdGlobals[i]->hasInitializer()) {
        Constant *C = LocalIdGlobals[i]->getInitializer();
        Builder.CreateStore(C, LocalIdAllocas[i]);
      }
    }
  }
  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
    for (BasicBlock::iterator ii = i->begin(), ee = i->end();
         ii != ee; ++ii) {
      for (int j = 0; j < 3; ++j)
        ii->replaceUsesOfWith(LocalIdGlobals[j], LocalIdAllocas[j]);
    }
  }

  std::vector<GlobalVariable*> LocalSizeGlobals(3, nullptr);
  std::vector<AllocaInst*> LocalSizeAllocas(3, nullptr);
  // Privatize _local_size* to private allocas.
  // They are referred to by WorkItemLoops to fetch the WI loop bounds.
  for (int i = 0; i < 3; ++i) {
    snprintf(TempStr, STRING_LENGTH, "_local_size_%c", 'x' + i);
    LocalSizeGlobals[i] = M->getGlobalVariable(TempStr);
    if (LocalSizeGlobals[i] != NULL) {
      LocalSizeAllocas[i] =
        Builder.CreateAlloca(LocalSizeGlobals[i]->getValueType(),
                             0, TempStr);
      if (LocalSizeGlobals[i]->hasInitializer()) {
        Constant *C = LocalSizeGlobals[i]->getInitializer();
        Builder.CreateStore(C, LocalSizeAllocas[i]);
      }
    }
  }
  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
    for (BasicBlock::iterator ii = i->begin(), ee = i->end();
         ii != ee; ++ii) {
      for (int j = 0; j < 3; ++j)
        ii->replaceUsesOfWith(LocalSizeGlobals[j], LocalSizeAllocas[j]);
    }
  }

  StoreInst *LocalSizeXStore = nullptr;
  if (WGDynamicLocalSize) {
    if (LocalSizeAllocas[0] != nullptr)
      LocalSizeXStore =
          Builder.CreateStore(createLoadFromContext(Builder, PC_LOCAL_SIZE, 0),
                              LocalSizeAllocas[0]);

    if (LocalSizeAllocas[1] != nullptr)
      Builder.CreateStore(
        createLoadFromContext(Builder, PC_LOCAL_SIZE, 1),
        LocalSizeAllocas[1]);

    if (LocalSizeAllocas[2] != nullptr)
      Builder.CreateStore(
        createLoadFromContext(Builder, PC_LOCAL_SIZE, 2),
      LocalSizeAllocas[2]);
  } else {
    if (LocalSizeAllocas[0] != nullptr)
      LocalSizeXStore = Builder.CreateStore(
          ConstantInt::get(LocalSizeAllocas[0]->getAllocatedType(),
                           WGLocalSizeX, 0),
          LocalSizeAllocas[0]);

    if (LocalSizeAllocas[1] != nullptr)
      Builder.CreateStore(
        ConstantInt::get(
          LocalSizeAllocas[1]->getAllocatedType(), WGLocalSizeY, 1),
        LocalSizeAllocas[1]);

    if (LocalSizeAllocas[2] != nullptr)
      Builder.CreateStore(
        ConstantInt::get(
          LocalSizeAllocas[2]->getAllocatedType(), WGLocalSizeZ, 2),
        LocalSizeAllocas[2]);
  }

  privatizeGlobalLoads(
      F, Builder, {"_group_id_x", "_group_id_y", "_group_id_z"}, GroupIdArgs);

  if (WGAssumeZeroGlobalOffset) {
    privatizeGlobalLoads(
        F, Builder,
        {"_global_offset_x", "_global_offset_y", "_global_offset_z"},
        {ConstantInt::get(SizeT, 0), ConstantInt::get(SizeT, 0),
         ConstantInt::get(SizeT, 0)});
  } else {
    privatizeGlobalLoads(
        F, Builder,
        {"_global_offset_x", "_global_offset_y", "_global_offset_z"},
        globalHandlesToContextStructLoads(
            Builder,
            {"_global_offset_x", "_global_offset_y", "_global_offset_z"},
            PC_GLOBAL_OFFSET));
  }

  privatizeGlobalLoads(
      F, Builder, {"_work_dim"},
      {createLoadFromContext(Builder, PC_WORK_DIM, -1, "_work_dim")});

  privatizeGlobalStores(
      F, Builder, {"__pocl_context_unreachable"},
      {getPtrAndTypeForContextVar(Builder, PC_EXECUTION_FAILED)});

  privatizeGlobalLoads(F, Builder, {PoclGVarBufferName},
                       globalHandlesToContextStructLoads(Builder,
                                                         {PoclGVarBufferName},
                                                         PC_GLOBAL_VAR_BUFFER));

  privatizeGlobalLoads(
      F, Builder, {"_num_groups_x", "_num_groups_y", "_num_groups_z"},
      globalHandlesToContextStructLoads(
          Builder, {"_num_groups_x", "_num_groups_y", "_num_groups_z"},
          PC_NUM_GROUPS));

  // Initialize the SG size global and privatize it.
  if (M->getGlobalVariable("_pocl_sub_group_size") != nullptr) {
    Value *SGSize = getRequiredSubgroupSize(*F);
    if (SGSize == nullptr) {
      Builder.SetInsertPoint(LocalSizeXStore->getNextNode());
      SGSize = Builder.CreateLoad(LocalSizeAllocas[0]->getAllocatedType(),
                                  LocalSizeAllocas[0]);
    }
    assert(SGSize != nullptr);
    privatizeGlobalLoads(F, Builder, {"_pocl_sub_group_size"}, {SGSize});
  }

  if (DeviceSidePrintf) {
    // Privatize _printf_buffer
    privatizeGlobalLoads(F, Builder, {"_printf_buffer"},
                         {createLoadFromContext(Builder, PC_PRINTF_BUFFER, -1,
                                                "printf_buffer")});

    privatizeGlobalLoads(
        F, Builder, {"_printf_buffer_position"},
        {createLoadFromContext(Builder, PC_PRINTF_BUFFER_POSITION, -1,
                               "printf_buffer_pos")});

    privatizeGlobalLoads(
        F, Builder, {"_printf_buffer_capacity"},
        {createLoadFromContext(Builder, PC_PRINTF_BUFFER_CAPACITY, -1,
                               "printf_buffer_capacity")});
  }
}

// Creates a work group launcher function (called KERNELNAME_workgroup)
// that assumes kernel pointer arguments are stored as pointers to the
// actual buffers and that scalar data is loaded from the default memory.
void WorkgroupImpl::createDefaultWorkgroupLauncher(llvm::Function *F) {

  LLVMContext &C = F->getContext();
  IRBuilder<> Builder(C);

  std::string FuncName = "";
  FuncName = F->getName().str();

  FunctionCallee fc =
      M->getOrInsertFunction(FuncName + "_workgroup", LauncherFuncT);
  Function *WorkGroup = dyn_cast<Function>(fc.getCallee());
  // copy only the function attributes (not the param/ret attributes)
  AttributeSet KernelFnAttrSet = F->getAttributes().getFnAttrs();
  AttributeList WorkgAttrL = WorkGroup->getAttributes();
  AttributeList WorkTempL = WorkgAttrL.removeAttributesAtIndex(
      C, AttributeList::AttrIndex::FunctionIndex);
  AttrBuilder AtB(C, KernelFnAttrSet);
  AttributeList NewWorkL = WorkTempL.addAttributesAtIndex(
      C, AttributeList::AttrIndex::FunctionIndex, AtB);
  WorkGroup->setAttributes(NewWorkL);

  WorkGroup->setLinkage(Function::ExternalLinkage);
  Triple TT(F->getParent()->getTargetTriple());
  if (TT.getEnvironment() == llvm::Triple::MSVC) {
    // dllexport is needed for exposing the symbol in the kernel module
    // when MSVC-based toolchain is used.
    WorkGroup->setDLLStorageClass(GlobalValue::DLLExportStorageClass);
  }

  // Propagate the DISubprogram to the launcher so we get debug data emitted
  // in case the kernel is inlined to it.
  if (auto *KernelSp = F->getSubprogram()) {
    WorkGroup->setSubprogram(
        pocl::mimicDISubprogram(KernelSp, WorkGroup->getName(), nullptr));
  }

  assert(WorkGroup != nullptr);
  BasicBlock *Block = BasicBlock::Create(M->getContext(), "", WorkGroup);
  Builder.SetInsertPoint(Block);

  Function::arg_iterator ai = WorkGroup->arg_begin();
  Argument *AI = &*ai;

  SmallVector<Value *, 8> Arguments;
  size_t i = 0;
  const DataLayout &DL = M->getDataLayout();
  for (Function::const_arg_iterator ii = F->arg_begin(), ee = F->arg_end();
       ii != ee; ++ii) {

    if (i == F->arg_size() - 4)
      break;

    Type *ArgType = ii->getType();
    Type* I32Ty = Type::getInt32Ty(M->getContext());

    Type *I8Ty = Type::getInt8Ty(M->getContext());
    Type *I8PtrTy = I8Ty->getPointerTo(0);
    Value *GEP = Builder.CreateGEP(I8PtrTy, AI, ConstantInt::get(I32Ty, i));
    Value *Pointer = Builder.CreateLoad(I8PtrTy, GEP);

    Value *Arg;
    if (DeviceAllocaLocals && isLocalMemFunctionArg(F, i)) {
      // Generate allocas for the local buffer arguments.
      // The size is passed directly instead of the pointer.
      PointerType *ParamType = dyn_cast<PointerType>(ArgType);
      assert(ParamType != nullptr);

      uint64_t ParamByteSize = DL.getTypeStoreSize(ParamType);
      Type *SizeIntType = IntegerType::get(C, ParamByteSize * 8);
      Value *LocalArgByteSize = Builder.CreatePointerCast(Pointer, SizeIntType);

      Type *ArgElementType = I8Ty;
      Value *ElementCount = LocalArgByteSize;

      Arg = new llvm::AllocaInst(ArgElementType, ParamType->getAddressSpace(),
                                 ElementCount,
                                 llvm::Align(
                                 MAX_EXTENDED_ALIGNMENT),
                                 "local_arg", Block);
    } else {
      if (ii->hasByValAttr()) {

        // If it's a pass-by-value pointer argument, we can pass the pointer
        // as is to the function, but only if the alignment of the arg is
        // <= than preferred alignment; otherwise it could crash
        // (chipStar test tests/runtime/TestAlignAttrRuntime)
        //
        // this can also be solved by inlining the WG func into launcher
        auto ArgAlign = ii->getParamAlign().valueOrOne();
        Type *BVType = ii->getParamByValType();
        auto PrefAlign = DL.getPrefTypeAlign(BVType);

        if (ArgAlign <= PrefAlign) {
          Arg = Builder.CreatePointerCast(Pointer, ArgType);
        } else {
#ifdef DEBUG_WORK_GROUP_GEN
          std::cerr << "WORKGROUP: arg alignment is larger, creating load\n";
#endif
          uint64_t ArgTypeSize = DL.getTypeStoreSize(BVType);
          Value *Src = Builder.CreatePointerCast(Pointer, ArgType);
          unsigned AddrSp = DL.getAllocaAddrSpace();
          AllocaInst *AI = new AllocaInst(BVType, AddrSp, nullptr, ArgAlign);
          Builder.Insert(AI);
          Builder.CreateMemCpy(AI, Align(1), Src, Align(1), ArgTypeSize);
          Arg = AI;
        }
      } else {
        Arg = Builder.CreateAlignedLoad(ArgType, Pointer,
                                        DL.getPrefTypeAlign(ArgType));
      }
    }

    Arguments.push_back(Arg);
    ++i;
  }

  ++ai;
  Arguments.push_back(&*ai);
  ++ai;
  Arguments.push_back(&*ai);
  ++ai;
  Arguments.push_back(&*ai);
  ++ai;
  Arguments.push_back(&*ai);

  llvm::CallInst *CI = Builder.CreateCall(F, ArrayRef<Value *>(Arguments));
  if (WorkGroup->getSubprogram() != nullptr && F->getSubprogram() != nullptr) {
    CI->setDebugLoc(
        llvm::DILocation::get(CI->getContext(), F->getSubprogram()->getLine(),
                              0, WorkGroup->getSubprogram(), nullptr, true));
  }

  Builder.CreateRetVoid();

  Function *Callee = CI->getCalledFunction();

  InlineFunctionInfo IFI;
  InlineFunction(*CI, IFI);

  if (Callee->getNumUses() == 0)
    Callee->eraseFromParent();
}

static inline uint64_t
align64(uint64_t value, unsigned alignment)
{
  return (value + alignment - 1) & ~((uint64_t)alignment - 1);
}

static bool isByValPtrArgument(llvm::Argument &Arg) {
  return Arg.getType()->isPointerTy() && Arg.hasByValAttr();
}

static size_t getArgumentSize(llvm::Argument &Arg) {
  llvm::Type *TypeInBuf = nullptr;
  if (Arg.getType()->isPointerTy()) {
    if (Arg.hasByValAttr()) {
      TypeInBuf = Arg.getParamByValType();
    } else {
      TypeInBuf = Arg.getType();
    }
  } else {
    // Scalar argument.
    TypeInBuf = Arg.getType();
  }

  const DataLayout &DL = Arg.getParent()->getParent()->getDataLayout();
  return DL.getTypeStoreSize(TypeInBuf);
}

static void computeArgBufferOffsets(LLVMValueRef F,
                                    uint64_t *ArgBufferOffsets) {

  uint64_t Offset = 0;
  uint64_t ArgCount = LLVMCountParams(F);

  // Compute the byte offsets of arguments in the arg buffer.
  for (size_t i = 0; i < ArgCount; i++) {
    LLVMValueRef Param = LLVMGetParam(F, i);
    // TODO: This is a target specific type? We would like to get the
    // natural size or the "packed size" instead...
    uint64_t ByteSize = getArgumentSize(cast<Argument>(*unwrap(Param)));

    // Always align each argument by the max extended alignment so we can push
    // structs and vectors to the buffer without needing to inspect the
    // content/alignment preferences of the structs. This naturally is a bit
    // wasteful, but should not matter in the big picture.
    uint64_t Alignment = MAX_EXTENDED_ALIGNMENT;

    assert(ByteSize > 0 && "Arg type size is zero?");
    Offset = align64(Offset, Alignment);

    ArgBufferOffsets[i] = Offset;
    Offset += ByteSize;
  }
}

LLVMValueRef WorkgroupImpl::createAllocaMemcpyForStruct(
    LLVMModuleRef M, LLVMBuilderRef Builder, llvm::Argument &Arg,
    LLVMValueRef ArgByteOffset) {

  LLVMContextRef LLVMContext = LLVMGetModuleContext(M);
  LLVMValueRef MemCpy1 = LLVMGetNamedFunction(M, "_pocl_memcpy_1");
  LLVMValueRef MemCpy4 = LLVMGetNamedFunction(M, "_pocl_memcpy_4");
  LLVMTypeRef Int8Type = LLVMInt8TypeInContext(LLVMContext);
  LLVMTypeRef Int32Type = LLVMInt32TypeInContext(LLVMContext);

  assert(isByValPtrArgument(Arg));
  llvm::Type *TypeInArg = Arg.getParamByValType();
  const DataLayout &DL = Arg.getParent()->getParent()->getDataLayout();
  Align alignment = DL.getABITypeAlign(TypeInArg);
  uint64_t StoreSize = DL.getTypeStoreSize(TypeInArg);
  LLVMValueRef Size =
      LLVMConstInt(LLVMInt32TypeInContext(LLVMContext), StoreSize, 0);

  LLVMValueRef LocalArgAlloca =
      LLVMBuildAlloca(Builder, wrap(TypeInArg), "struct_arg");

  if ((alignment >= 4) && (StoreSize % 4 == 0)) {
    LLVMTypeRef i32PtrAS0 = LLVMPointerType(Int32Type, 0);
    LLVMTypeRef i32PtrAS1 = LLVMPointerType(Int32Type, DeviceArgsASid);
    LLVMValueRef CARG0 =
        LLVMBuildPointerCast(Builder, LocalArgAlloca, i32PtrAS0, "cargDst");
    LLVMValueRef CARG1 =
        LLVMBuildPointerCast(Builder, ArgByteOffset, i32PtrAS1, "cargSrc");

    LLVMValueRef args[3];
    args[0] = CARG0;
    args[1] = CARG1;
    args[2] = Size;

    LLVMTypeRef FnTy = LLVMGetCalledFunctionType(MemCpy4);
    LLVMBuildCall2(Builder, FnTy, MemCpy4, args, 3, "");
  } else {
    LLVMTypeRef i8PtrAS0 = LLVMPointerType(Int8Type, 0);
    LLVMTypeRef i8PtrAS1 = LLVMPointerType(Int8Type, DeviceArgsASid);
    LLVMValueRef CARG0 =
        LLVMBuildPointerCast(Builder, LocalArgAlloca, i8PtrAS0, "cargDst");
    LLVMValueRef CARG1 =
        LLVMBuildPointerCast(Builder, ArgByteOffset, i8PtrAS1, "cargSrc");

    LLVMValueRef args[3];
    args[0] = CARG0;
    args[1] = CARG1;
    args[2] = Size;

    LLVMTypeRef FnTy = LLVMGetCalledFunctionType(MemCpy1);
    LLVMBuildCall2(Builder, FnTy, MemCpy1, args, 3, "");
  }

  return LocalArgAlloca;
}

/// Creates a load to get an argument from an argument buffer.
///
/// \param Builder The LLVM IR builder to use.
/// \param ArgBufferPtr The LLVM IR Value pointing to the arg buffer.
/// \param ArgBufferOffsets The offsets of arguments in the buffer.
/// \param Ctx LLVM Context to use.
/// \param F The function with the arguments.
/// \param ParamIndex The index of the argument.
/// \param Name The name to give for the load (for IR readability).
LLVMValueRef WorkgroupImpl::createArgBufferLoad(
    LLVMBuilderRef Builder, LLVMValueRef ArgBufferPtr,
    uint64_t *ArgBufferOffsets, LLVMContextRef Ctx, LLVMValueRef F,
    unsigned ParamIndex, std::string Name) {

  LLVMValueRef Param = LLVMGetParam(F, ParamIndex);
  LLVMTypeRef ParamType = LLVMTypeOf(Param);

  LLVMModuleRef M = LLVMGetGlobalParent(F);
  LLVMContextRef LLVMContext = LLVMGetModuleContext(M);

  uint64_t ArgPos = ArgBufferOffsets[ParamIndex];
  LLVMValueRef Offs =
      LLVMConstInt(LLVMInt32TypeInContext(LLVMContext), ArgPos, 0);
  LLVMTypeRef Int8Type = LLVMInt8TypeInContext(Ctx);

  LLVMValueRef ArgByteOffset =
      LLVMBuildGEP2(Builder, Int8Type, ArgBufferPtr, &Offs, 1, "arg_byte_offset");

  llvm::Argument &Arg = cast<Argument>(*unwrap(Param));

  // byval arguments (private structs), passed via pointer
  if (isByValPtrArgument(Arg)) {

    // the kernel AS for private structs is always zero (private).
    // if the arg address space is also zero, nothing to do here just cast...
    if (DeviceArgsASid == 0)
      return LLVMBuildPointerCast(Builder, ArgByteOffset, ParamType,
                                  "inval_arg_ptr");

    // ... otherwise the arg AS is different, and we need an alloca+memcpy.
    else
      return createAllocaMemcpyForStruct(M, Builder, Arg, ArgByteOffset);

    // not by-val argument
  } else {
    LLVMTypeRef DestTy = LLVMPointerType(ParamType, DeviceArgsASid);
    LLVMValueRef ArgOffsetBitcast =
        LLVMBuildPointerCast(Builder, ArgByteOffset, DestTy, "arg_ptr");
    LLVMTypeRef LoadTy = ParamType;
    return LLVMBuildLoad2(Builder, LoadTy, ArgOffsetBitcast, Name.c_str());
  }
}

/// Creates a work group launcher with all the argument data passed
/// in a single argument buffer.
///
/// All argument values, including pointers are stored directly in the
/// argument buffer with natural alignment. The rules for populating the
/// buffer are those of the HSA kernel calling convention. The name of
/// the generated function is KERNELNAME_workgroup_argbuffer.
///
/// \param Func The kernel to generate the launcher for.
/// \param KernName The prefix for the launcher function's name, which will
/// be called 'KernName_workgroup_argbuffer'.
Function *
WorkgroupImpl::createArgBufferWorkgroupLauncher(Function *Func,
                                                std::string KernName) {

  LLVMValueRef F = wrap(Func);
  uint64_t ArgCount = LLVMCountParams(F);
  std::vector<uint64_t> ArgBufferOffsets(ArgCount);
  LLVMModuleRef M = wrap(this->M);

  computeArgBufferOffsets(F, ArgBufferOffsets.data());

  LLVMContextRef LLVMContext = LLVMGetModuleContext(M);

  LLVMTypeRef Int8Type = LLVMInt8TypeInContext(LLVMContext);
  LLVMTypeRef Int32Type = LLVMInt32TypeInContext(LLVMContext);
  LLVMTypeRef Int64Type = LLVMInt64TypeInContext(LLVMContext);

  LLVMTypeRef ArgsPtrType = LLVMPointerType(Int8Type, DeviceArgsASid);

  LLVMTypeRef CtxPtrType = LLVMPointerType(Int8Type, DeviceContextASid);

  std::ostringstream StrStr;
  StrStr << KernName;
  StrStr << "_workgroup_argbuffer";

  std::string FName = StrStr.str();
  const char *FunctionName = FName.c_str();

  LLVMTypeRef LauncherArgTypes[] = {
    ArgsPtrType, // args
    CtxPtrType, // pocl_ctx
    wrap(SizeT), // group_x
    wrap(SizeT), // group_y
    wrap(SizeT), // group_z
  };

  LLVMTypeRef VoidType = LLVMVoidTypeInContext(LLVMContext);
  LLVMTypeRef LauncherFuncType =
    LLVMFunctionType(VoidType, LauncherArgTypes, 1 + HiddenArgs, 0);

  LLVMValueRef WrapperKernel =
    LLVMAddFunction(M, FunctionName, LauncherFuncType);

  LLVMBasicBlockRef Block =
    LLVMAppendBasicBlockInContext(LLVMContext, WrapperKernel, "entry");

  LLVMBuilderRef Builder = LLVMCreateBuilderInContext(LLVMContext);
  assert(Builder);

  LLVMPositionBuilderAtEnd(Builder, Block);

  std::vector<LLVMValueRef> Args(ArgCount);
  LLVMValueRef ArgBuffer = LLVMGetParam(WrapperKernel, 0);
  size_t i = 0;
  for (; i < ArgCount - HiddenArgs; ++i) {
    if (DeviceAllocaLocals && isLocalMemFunctionArg(Func, i)) {

      // Generate allocas for the local buffer arguments.
      // The size is passed directly instead of the pointer.
      LLVMValueRef Param = LLVMGetParam(F, i);
      LLVMTypeRef ParamType = LLVMTypeOf(Param);
      assert(ParamType != nullptr);
      LLVMTargetDataRef DataLayout = LLVMGetModuleDataLayout(M);

      uint64_t ParamByteSize = LLVMStoreSizeOfType(DataLayout, ParamType);
      LLVMTypeRef SizeIntType = (ParamByteSize == 4) ? Int32Type : Int64Type;

      uint64_t ArgPos = ArgBufferOffsets[i];
      LLVMValueRef Offs = LLVMConstInt(Int32Type, ArgPos, 0);

      LLVMValueRef SizeByteOffset = LLVMBuildGEP2(Builder, Int8Type, ArgBuffer,
                                                  &Offs, 1, "size_byte_offset");
      LLVMTypeRef DestTy = LLVMPointerType(SizeIntType, 0);
      LLVMValueRef SizeOffsetBitcast =
          LLVMBuildPointerCast(Builder, SizeByteOffset, DestTy, "size_ptr");

      LLVMTypeRef AllocaType = Int8Type;

      LLVMTypeRef LoadTy = SizeIntType;
      LLVMValueRef LocalArgByteSize =
          LLVMBuildLoad2(Builder, LoadTy, SizeOffsetBitcast, "byte_size");
      LLVMValueRef ElementCount = LocalArgByteSize;

      LLVMValueRef LocalArgAlloca = wrap(new llvm::AllocaInst(
          unwrap(AllocaType), LLVMGetPointerAddressSpace(ParamType),
          unwrap(ElementCount),
          llvm::Align(
              MAX_EXTENDED_ALIGNMENT),
          "local_arg", unwrap(Block)));
      Args[i] = LocalArgAlloca;
    } else {
      Args[i] = createArgBufferLoad(
          Builder, ArgBuffer, ArgBufferOffsets.data(), LLVMContext, F, i,
          std::string("kernel_arg_") + std::to_string(i));
    }
  }

  size_t Arg = 1;
  // Pass the context object
  LLVMValueRef CtxParam = LLVMGetParam(WrapperKernel, Arg++);
  LLVMTypeRef CtxT = wrap(PoclContextT);
  LLVMTypeRef CtxPtrTypeActual = LLVMPointerType(CtxT, DeviceContextASid);
  LLVMValueRef CastContext =
      LLVMBuildPointerCast(Builder, CtxParam, CtxPtrTypeActual, "ctx_ptr");
  Args[i++] = CastContext;
  // Pass the group ids.
  Args[i++] = LLVMGetParam(WrapperKernel, Arg++);
  Args[i++] = LLVMGetParam(WrapperKernel, Arg++);
  Args[i++] = LLVMGetParam(WrapperKernel, Arg++);

  assert (i == ArgCount);

  LLVMTypeRef FnTy = wrap(Func->getFunctionType());
  LLVMValueRef Call =
      LLVMBuildCall2(Builder, FnTy, F, Args.data(), ArgCount, "");
  LLVMBuildRetVoid(Builder);

  llvm::CallInst *CallI = llvm::dyn_cast<llvm::CallInst>(llvm::unwrap(Call));
  CallI->setCallingConv(Func->getCallingConv());

  InlineFunctionInfo IFI;
  InlineFunction(*dyn_cast<CallInst>(llvm::unwrap(Call)), IFI);

  LLVMDisposeBuilder(Builder);
  return llvm::dyn_cast<llvm::Function>(llvm::unwrap(WrapperKernel));
}

/// Creates a launcher function that executes all work-items in the grid by
/// launching a given work-group function for all work-group ids.
///
/// The function adheres to the PHSA calling convention where the first two
/// arguments are for PHSA's context data, and the third one is the argument
/// buffer. The name will be phsa_kernel.KERNELNAME_grid_launcher.
///
/// \param KernFunc The kernel function to generate the launcher for.
/// \param WGFunc The work-group function.
/// \param KernName The (original) name of the kernel.
void WorkgroupImpl::createGridLauncher(Function *KernFunc, Function *WGFunc,
                                       std::string KernName) {

  LLVMValueRef Kernel = llvm::wrap(KernFunc);
  LLVMValueRef WGF = llvm::wrap(WGFunc);
  LLVMModuleRef M = llvm::wrap(this->M);
  LLVMContextRef LLVMContext = LLVMGetModuleContext(M);

  LLVMTypeRef Int8Type = LLVMInt8TypeInContext(LLVMContext);
  LLVMTypeRef Int8PtrType = LLVMPointerType(Int8Type, 0);
  LLVMTypeRef ArgsPtrType = LLVMPointerType(Int8Type, DeviceArgsASid);

  std::ostringstream StrStr("phsa_kernel.", std::ios::ate);
  StrStr << KernName;
  StrStr << "_grid_launcher";

  std::string FName = StrStr.str();
  const char *FunctionName = FName.c_str();

  LLVMTypeRef LauncherArgTypes[] = {
      Int8PtrType /*phsactx0*/, Int8PtrType /*phsactx1*/, ArgsPtrType /*args*/};

  LLVMTypeRef VoidType = LLVMVoidTypeInContext(LLVMContext);
  LLVMTypeRef LauncherFuncType =
    LLVMFunctionType(VoidType, LauncherArgTypes, 3, 0);

  LLVMValueRef Launcher =
    LLVMAddFunction(M, FunctionName, LauncherFuncType);

  LLVMBasicBlockRef Block =
    LLVMAppendBasicBlockInContext(LLVMContext, Launcher, "entry");

  LLVMBuilderRef Builder = LLVMCreateBuilderInContext(LLVMContext);
  assert(Builder);

  LLVMPositionBuilderAtEnd(Builder, Block);

  LLVMValueRef RunnerFunc = LLVMGetNamedFunction(M, "_pocl_run_all_wgs");
  assert (RunnerFunc != nullptr);

  LLVMTypeRef RunnerArgTypes[] = {LLVMTypeOf(LLVMGetParam(RunnerFunc, 0)),
                                  LLVMTypeOf(LLVMGetParam(RunnerFunc, 1)),
                                  LLVMTypeOf(LLVMGetParam(RunnerFunc, 2)),
                                  LLVMTypeOf(LLVMGetParam(RunnerFunc, 3))};

  uint64_t KernArgCount = LLVMCountParams(Kernel);
  std::vector<uint64_t> KernArgBufferOffsets(KernArgCount);
  computeArgBufferOffsets(Kernel, KernArgBufferOffsets.data());

  // The second argument in the native phsa interface is auxiliary
  // driver-specific data that is passed as the last argument to the grid
  // launcher.
  LLVMValueRef AuxParam = LLVMGetParam(Launcher, 1);
  LLVMValueRef ArgBuffer = LLVMGetParam(Launcher, 2);

  // Load the pointer to the pocl context (in global memory), assuming it is
  // stored as the 4th last argument in the kernel.
  LLVMValueRef PoclCtx = createArgBufferLoad(
      Builder, ArgBuffer, KernArgBufferOffsets.data(), LLVMContext, Kernel,
      KernArgCount - HiddenArgs, "pocl_context");

  LLVMValueRef Args[4] = {
      LLVMBuildPointerCast(Builder, WGF, RunnerArgTypes[0], "wg_func"),
      LLVMBuildPointerCast(Builder, ArgBuffer, RunnerArgTypes[1], "args"),
      LLVMBuildPointerCast(Builder, PoclCtx, RunnerArgTypes[2], "ctx"),
      LLVMBuildPointerCast(Builder, AuxParam, RunnerArgTypes[3], "aux")};

  LLVMTypeRef RunnerFnTy = LLVMFunctionType(VoidType, RunnerArgTypes, 4, 0);
  LLVMValueRef Call =
      LLVMBuildCall2(Builder, RunnerFnTy, RunnerFunc, Args, 4, "");
  LLVMBuildRetVoid(Builder);

  InlineFunctionInfo IFI;
  InlineFunction(*dyn_cast<CallInst>(llvm::unwrap(Call)), IFI);

  // Add a fixed name global variable which points to the generated grid
  // launcher, if there is a declaration by that name. If there is, we
  // have a device main that refers to it.
  LLVMValueRef GridLauncherGlobal = LLVMGetNamedGlobal(M, "pocl_grid_launcher");
  if (GridLauncherGlobal != nullptr) {
    LLVMSetExternallyInitialized(GridLauncherGlobal, false);
    LLVMSetInitializer(GridLauncherGlobal, Launcher);
  }
}

// The subgroup size is currently defined for the CPU implementations
// via the intel_reqd_subgroup_size metadata or the local dimension
// x size (the default).
llvm::Value *WorkgroupImpl::getRequiredSubgroupSize(llvm::Function &F) {

  if (MDNode *SGSizeMD = F.getMetadata("intel_reqd_sub_group_size")) {
    // Use the constant from the metadata.
    ConstantAsMetadata *ConstMD =
        cast<ConstantAsMetadata>(SGSizeMD->getOperand(0));
    ConstantInt *Const = cast<ConstantInt>(ConstMD->getValue());
    return Const;
  }
  return nullptr;
}

llvm::PreservedAnalyses Workgroup::run(llvm::Module &M,
                                       llvm::ModuleAnalysisManager &AM) {
  WorkgroupImpl WGI;
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  PAChanged.preserve<VariableUniformityAnalysis>();

  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  bool Ret = WGI.runOnModule(M, FAM);

#ifdef DEBUG_WORK_GROUP_GEN
  std::cerr << "### After Workgroup:\n";
  M.dump();
#endif

  return Ret ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_MPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
