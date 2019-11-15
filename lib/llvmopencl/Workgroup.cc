// LLVM module pass to create the single function (fully inlined)
// and parallelized kernel for an OpenCL workgroup.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos
//               2012-2019 Pekka Jääskeläinen
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

#include <cstdio>
#include <map>
#include <iostream>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "config.h"
#include "pocl.h"
#include "pocl_cl.h"

#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CallSite.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#ifdef LLVM_OLDER_THAN_7_0
#include <llvm/IR/TypeBuilder.h>
#endif
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#ifdef LLVM_OLDER_THAN_7_0
#include <llvm/Transforms/Utils/Local.h>
#endif

#include <llvm-c/Core.h>
#include <llvm-c/Target.h>

#include "CanonicalizeBarriers.h"
#include "BarrierTailReplication.h"
#include "WorkitemReplication.h"
#include "Barrier.h"
#include "Workgroup.h"

#include <cstdio>
#include <map>
#include <iostream>
#include <sstream>

#if _MSC_VER
#  include "vccompat.hpp"
#endif

#define STRING_LENGTH 32

POP_COMPILER_DIAGS

extern cl_device_id currentPoclDevice;

using namespace std;
using namespace llvm;
using namespace pocl;

/* The kernel to process in this kernel compiler launch. */
cl::opt<string> KernelName("kernel", cl::desc("Kernel function name"),
                           cl::value_desc("kernel"), cl::init(""));

#ifdef LLVM_OLDER_THAN_7_0
namespace llvm {

  typedef struct _pocl_context PoclContext;

  template<bool xcompile> class TypeBuilder<PoclContext, xcompile> {
  public:
    static StructType *get(LLVMContext &Context) {
      if (size_t_width == 64)
        {
        SmallVector<Type *, 10> Elements;
        Elements.push_back(
            TypeBuilder<types::i<64>[3], xcompile>::get(Context));
        Elements.push_back(
            TypeBuilder<types::i<64>[3], xcompile>::get(Context));
        Elements.push_back(
            TypeBuilder<types::i<64>[3], xcompile>::get(Context));
        Elements.push_back(TypeBuilder<types::i<8> *, xcompile>::get(Context));
        Elements.push_back(TypeBuilder<types::i<32> *, xcompile>::get(Context));
        Elements.push_back(TypeBuilder<types::i<32>, xcompile>::get(Context));
        Elements.push_back(TypeBuilder<types::i<32>, xcompile>::get(Context));
        return StructType::get(Context, Elements);
        }
      else if (size_t_width == 32)
        {
          SmallVector<Type *, 10> Elements;
          Elements.push_back(
            TypeBuilder<types::i<32>[3], xcompile>::get(Context));
          Elements.push_back(
            TypeBuilder<types::i<32>[3], xcompile>::get(Context));
          Elements.push_back(
            TypeBuilder<types::i<32>[3], xcompile>::get(Context));
          Elements.push_back(
              TypeBuilder<types::i<8> *, xcompile>::get(Context));
          Elements.push_back(
              TypeBuilder<types::i<32> *, xcompile>::get(Context));
          Elements.push_back(TypeBuilder<types::i<32>, xcompile>::get(Context));
          Elements.push_back(
            TypeBuilder<types::i<32>, xcompile>::get(Context));

          return StructType::get(Context, Elements);
        }
      else
        {
          assert (false && "Unsupported size_t width.");
          return NULL;
        }
    }

    static void setSizeTWidth(int width) {
      size_t_width = width;
    }

  private:
    static int size_t_width;
  };

  template<bool xcompile>
  int TypeBuilder<PoclContext, xcompile>::size_t_width = 0;
}  // namespace llvm

#endif

enum PoclContextStructFields {
  PC_NUM_GROUPS,
  PC_GLOBAL_OFFSET,
  PC_LOCAL_SIZE,
  PC_PRINTF_BUFFER,
  PC_PRINTF_BUFFER_POSITION,
  PC_PRINTF_BUFFER_CAPACITY,
  PC_WORK_DIM
};

char Workgroup::ID = 0;
static RegisterPass<Workgroup> X("workgroup", "Workgroup creation pass");

bool
Workgroup::runOnModule(Module &M) {

  this->M = &M;
  this->C = &M.getContext();

  HiddenArgs = 0;
  SizeTWidth = currentPoclDevice->address_bits;
  SizeT = IntegerType::get(*C, SizeTWidth);

#ifdef LLVM_OLDER_THAN_7_0
  TypeBuilder<PoclContext, true>::setSizeTWidth(SizeTWidth);
  PoclContextT = TypeBuilder<PoclContext, true>::get(*C);
  LauncherFuncT =
      SizeTWidth == 32
          ? TypeBuilder<void(types::i<8> *[], PoclContext *, types::i<32>,
                             types::i<32>, types::i<32>),
                        true>::get(M.getContext())
          : TypeBuilder<void(types::i<8> *[], PoclContext *, types::i<64>,
                             types::i<64>, types::i<64>),
                        true>::get(M.getContext());
#else
  // LLVM 8.0 dropped the TypeBuilder API. This is a cleaner version
  // anyways as it builds the context type using the SizeT directly.
  llvm::Type *Int32T = Type::getInt32Ty(*C);
  llvm::Type *Int8T = Type::getInt8Ty(*C);
  PoclContextT =
    StructType::get(
      ArrayType::get(SizeT, 3), // NUM_GROUPS
      ArrayType::get(SizeT, 3), // GLOBAL_OFFSET
      ArrayType::get(SizeT, 3), // LOCAL_SIZE
      PointerType::get(Int8T, 0), // PRINTF_BUFFER
      PointerType::get(Int32T, 0), // PRINTF_BUFFER_POSITION
      Int32T, // PRINTF_BUFFER_CAPACITY
      Int32T); // WORK_DIM

  LauncherFuncT = FunctionType::get(
      Type::getVoidTy(*C),
      {PointerType::get(PointerType::get(Type::getInt8Ty(*C), 0),
                        currentPoclDevice->args_as_id),
       PointerType::get(PoclContextT, currentPoclDevice->context_as_id), SizeT,
       SizeT, SizeT},
      false);
#endif

  assert ((SizeTWidth == 64 || SizeTWidth == 32) &&
          "Target has an unsupported pointer width.");

  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    // Don't internalize functions starting with "__wrap_" for the use of GNU
    // linker's switch --wrap=symbol, where calls to the "symbol" are replaced
    // with "__wrap_symbol" at link time.  These functions may not be referenced
    // until final link and being deleted by LLVM optimizations before it.
    if (!i->isDeclaration() && !i->getName().startswith("__wrap_"))
      i->setLinkage(Function::InternalLinkage);
  }

  // Store the new and old kernel pairs in order to regenerate
  // all the metadata that used to point to the unmodified
  // kernels.
  FunctionMapping kernels;

  // Mapping of all functions which have been transformed to take
  // extra printf arguments.
  FunctionMapping printfCache;

  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    Function &OrigKernel = *i;
    if (!isKernelToProcess(OrigKernel)) continue;
    Function *L = createWrapper(&OrigKernel, printfCache);

    privatizeContext(L);

    if (currentPoclDevice->arg_buffer_launcher) {
      Function *WGLauncher =
        createArgBufferWorkgroupLauncher(L, OrigKernel.getName().str());
      L->addFnAttr(Attribute::NoInline);
      L->removeFnAttr(Attribute::AlwaysInline);
      WGLauncher->addFnAttr(Attribute::AlwaysInline);
      createGridLauncher(L, WGLauncher, OrigKernel.getName().str());
    } else if (currentPoclDevice->spmd) {
      // For SPMD machines there is no need for a WG launcher, the device will
      // call/handle the single-WI kernel function directly.
      kernels[&OrigKernel] = L;
    } else {
      createDefaultWorkgroupLauncher(L);
      // This is used only by TCE anymore. TODO: Replace all with the
      // ArgBuffer one.
      createFastWorkgroupLauncher(L);
    }
  }

  if (!currentPoclDevice->arg_buffer_launcher && currentPoclDevice->spmd) {
    regenerate_kernel_metadata(M, kernels);

    // Delete the old kernels.
    for (FunctionMapping::const_iterator i = kernels.begin(),
           e = kernels.end(); i != e; ++i) {
        Function *old_kernel = (*i).first;
        Function *new_kernel = (*i).second;
        if (old_kernel == new_kernel) continue;
        old_kernel->eraseFromParent();
    }
  }
  return true;
}

// Ensures the given value is not optimized away even if it's not used
// by LLVM IR.
void Workgroup::addPlaceHolder(llvm::IRBuilder<> &Builder,
                               llvm::Value *Val,
                               const std::string TypeStr="r") {

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

static void addRangeMetadataForPCField(llvm::Instruction *Instr,
                                       int StructFieldIndex,
                                       int FieldIndex = -1) {
  uint64_t Min = 0;
  uint64_t Max = 0;
  uint64_t LocalSizes[] = {WGLocalSizeX, WGLocalSizeY, WGLocalSizeZ};
  switch (StructFieldIndex) {
  case PC_WORK_DIM:
    Min = 1;
    Max = currentPoclDevice->max_work_item_dimensions;
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
        Max = (WGMaxGridDimWidth > 0
                   ? WGMaxGridDimWidth
                   : min(currentPoclDevice->max_work_item_sizes[FieldIndex],
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

// Creates a load from the hidden context structure argument for
// the given element.
llvm::Value *
Workgroup::createLoadFromContext(
  IRBuilder<> &Builder, int StructFieldIndex, int FieldIndex=-1) {

  Value *GEP;
  GEP = Builder.CreateStructGEP(ContextArg->getType()->getPointerElementType(),
                                ContextArg, StructFieldIndex);

  llvm::LoadInst *Load = nullptr;
  if (SizeTWidth == 64) {
    if (FieldIndex == -1)
      Load = Builder.CreateLoad(Builder.CreateConstGEP1_64(GEP, 0));
    else
      Load = Builder.CreateLoad(Builder.CreateConstGEP2_64(GEP, 0, FieldIndex));
  } else {
    if (FieldIndex == -1)
      Load = Builder.CreateLoad(Builder.CreateConstGEP1_32(GEP, 0));
    else
      Load = Builder.CreateLoad(Builder.CreateConstGEP2_32(
          GEP->getType()->getPointerElementType(), GEP, 0, FieldIndex));
  }
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
      Function *callee = CallInstr->getCalledFunction();

      if (callee->getName().startswith("llvm."))
        continue;
      if (callee->getName().equals("printf"))
        return true;
      if (callee->getName().equals("__pocl_printf"))
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
  NewF->takeName(F);

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
  CloneFunctionInto(NewF, F, VV, true, RI);

  return NewF;
}

// Recursively replace _cl_printf calls with _pocl_printf calls, while
// propagating the required pocl_context->printf_buffer arguments.
static void replacePrintfCalls(Value *pb, Value *pbp, Value *pbc, bool isKernel,
                               Function *poclPrintf, Module &M, Function *L,
                               FunctionMapping &printfCache) {

  // If none of the kernels use printf(), it will not be linked into the
  // module.
  if (poclPrintf == nullptr)
    return;

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
      Function *oldF = CallInstr->getCalledFunction();

      // Skip inline asm blocks.
      if (oldF == nullptr)
        continue;

      if (oldF->getName().equals("printf")) {
        ops.clear();
        ops.push_back(pb);
        ops.push_back(pbp);
        ops.push_back(pbc);

        unsigned j = CallInstr->getNumOperands() - 1;
        for (unsigned i = 0; i < j; ++i)
          ops.push_back(CallInstr->getOperand(i));

        CallSite CS(CallInstr);
        CallInst *NewCI = CallInst::Create(poclPrintf, ops);
        NewCI->setCallingConv(poclPrintf->getCallingConv());
        NewCI->setTailCall(CS.isTailCall());

        replaceCIMap.insert(
            std::pair<CallInst *, CallInst *>(CallInstr, NewCI));
      } else {
        if (!oldF->getName().startswith("llvm."))
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
      newCI->insertBefore(CI);
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
        replacePrintfCalls(nullptr, nullptr, nullptr, false, poclPrintf, M,
                           newF, printfCache);

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
    }
  }

  for (auto it : replaceCIMap) {
    CallInst *CI = it.first;
    CallInst *newCI = it.second;

    CI->replaceAllUsesWith(newCI);
    ReplaceInstWithInst(CI, newCI);
  }
}

// Create a wrapper for the kernel and add pocl-specific hidden arguments.
// Also inlines the wrapped function to the wrapper.
Function *
Workgroup::createWrapper(Function *F, FunctionMapping &printfCache) {

  SmallVector<Type *, 8> sv;
  LLVMContext &C = M->getContext();
  for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
       i != e; ++i)
    sv.push_back(i->getType());

  if (!currentPoclDevice->arg_buffer_launcher && currentPoclDevice->spmd) {
    sv.push_back(
      PointerType::get(PoclContextT, currentPoclDevice->context_as_id));
    HiddenArgs = 1;
  } else {
    // pocl_context
    sv.push_back(
      PointerType::get(PoclContextT, currentPoclDevice->context_as_id));
    // group_x
    sv.push_back(SizeT);
    // group_y
    sv.push_back(SizeT);
    // group_z
    sv.push_back(SizeT);

    // we might not have all of the globals anymore in the module in case the
    // kernel does not refer to them and they are optimized away
    HiddenArgs = 4;

  }

  FunctionType *ft =
      FunctionType::get(Type::getVoidTy(C), ArrayRef<Type *>(sv), false);

  std::string funcName = "";
  funcName = F->getName().str();
  Function *L = NULL;
  if (!currentPoclDevice->arg_buffer_launcher && currentPoclDevice->spmd) {
    Function *F = M->getFunction(funcName);
    F->setName(funcName + "_original");
    L = Function::Create(ft, Function::ExternalLinkage, funcName, M);
  } else
    L = Function::Create(ft, Function::ExternalLinkage,
                         "_pocl_kernel_" + funcName, M);

  SmallVector<Value *, 8> arguments;
  Function::arg_iterator ai = L->arg_begin();
  for (unsigned i = 0, e = F->arg_size(); i != e; ++i) {
    arguments.push_back(&*ai);
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
  L->copyMetadata(F, 0);
  // We need to mark the generated function to avoid it being considered a
  // new kernel to process (which results in infinite recursion). This is
  // because kernels are detected by the presense of the argument metadata
  // we just copied from the original kernel function.
  L->setMetadata("pocl_generated", MDNode::get(C, {createConstantIntMD(C, 1)}));

  IRBuilder<> Builder(BasicBlock::Create(C, "", L));

  Value *pb, *pbp, *pbc;
  if (currentPoclDevice->device_side_printf) {
    pb = createLoadFromContext(Builder, PC_PRINTF_BUFFER);
    pbp = createLoadFromContext(Builder, PC_PRINTF_BUFFER_POSITION);
    pbc = createLoadFromContext(Builder, PC_PRINTF_BUFFER_CAPACITY);
  } else {
    pb = pbp = pbc = nullptr;
  }

  CallInst *c = Builder.CreateCall(F, ArrayRef<Value*>(arguments));
  Builder.CreateRetVoid();

  std::set<CallInst *> CallsToRemove;

  // At least with LLVM 4.0, the runtime of AddAliasScopeMetadata of
  // llvm::InlineFunction explodes in case of kernels with restrict
  // metadata and a lot of lifetime markers. The issue produces at
  // least with EinsteinToolkit which has a lot of restrict kernel
  // args). Remove them here before inlining to speed it up.
  // TODO: Investigate the root cause.

  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {
      Instruction *Instr = dyn_cast<Instruction>(BI);
      if (!llvm::isa<CallInst>(Instr)) continue;
      CallInst *CallInstr = dyn_cast<CallInst>(Instr);
      Function *Callee = CallInstr->getCalledFunction();
      // At least with LLVM 4.0, the runtime of AddAliasScopeMetadata of
      // llvm::InlineFunction explodes in case of kernels with restrict
      // metadata and a lot of lifetime markers. The issue produces at
      // least with EinsteinToolkit which has a lot of restrict kernel
      // args). Remove them here before inlining to speed it up.
      // TODO: Investigate the root cause.
      if (Callee != nullptr &&
          (Callee->getName().startswith("llvm.lifetime.end") ||
           Callee->getName().startswith("llvm.lifetime.start"))) {
        CallsToRemove.insert(CallInstr);
        continue;
      }
    }
  }

  for (auto C : CallsToRemove) {
    C->eraseFromParent();
  }

  // needed for printf
  InlineFunctionInfo IFI;
  InlineFunction(c, IFI);

  if (currentPoclDevice->device_side_printf) {
    Function *poclPrintf = M->getFunction("__pocl_printf");
    replacePrintfCalls(pb, pbp, pbc, true, poclPrintf, *M, L, printfCache);
  }

  L->setSubprogram(F->getSubprogram());

  // SPMD machines might need a special calling convention to mark the
  // kernels that should be executed in SPMD fashion. For MIMD/CPU,
  // we want to use the default calling convention for the work group
  // function.
  if (currentPoclDevice->spmd)
    L->setCallingConv(F->getCallingConv());

  return L;
}

// Converts the given global context variable handles to loads from the
// hidden context struct argument. If there is no reference to the global,
// the corresponding entry in the returned vector will contain a nullptr.
std::vector<llvm::Value*>
Workgroup::globalHandlesToContextStructLoads(
  IRBuilder<> &Builder,
  const std::vector<std::string> &&GlobalHandleNames,
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

// Converts uses of the given variable handles (external global variables) to
// use the given function-private values instead.
void
Workgroup::privatizeGlobals(llvm::Function *F, llvm::IRBuilder<> &Builder,
                            const std::vector<std::string> &&GlobalHandleNames,
                            std::vector<llvm::Value*> PrivateValues) {

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
        llvm::LoadInst *L = cast<llvm::LoadInst>(ii);
        llvm::GlobalValue *GlobalHandle =
          M->getGlobalVariable(GlobalHandleNames.at(j));

        if (GlobalHandle == nullptr)
          continue;

        if (L->getPointerOperand()->stripPointerCasts() != GlobalHandle)
          continue;

        llvm::Value *Cast =
          Builder.CreateTruncOrBitCast(PrivateValues[j], L->getType());
        ii->replaceAllUsesWith(Cast);
        ii->eraseFromParent();
        break;
      }
    }
  }
}

void
Workgroup::privatizeContext(Function *F)
{
  char TempStr[STRING_LENGTH];
  IRBuilder<> Builder(F->getEntryBlock().getFirstNonPHI());

  std::vector<GlobalVariable*> LocalIdGlobals(3);
  std::vector<AllocaInst*> LocalIdAllocas(3);
  // Privatize _local_id to allocas. They are used as iteration variables in
  // WorkItemLoops, thus referred to later on.
  for (int i = 0; i < 3; ++i) {
    snprintf(TempStr, STRING_LENGTH, "_local_id_%c", 'x' + i);
    LocalIdGlobals[i] = M->getGlobalVariable(TempStr);
    if (LocalIdGlobals[i] != NULL) {
      LocalIdAllocas[i] =
        Builder.CreateAlloca(LocalIdGlobals[i]->getType()->getElementType(), 0,
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
        Builder.CreateAlloca(LocalSizeGlobals[i]->getType()->getElementType(),
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

  if (WGDynamicLocalSize) {
    if (LocalSizeAllocas[0] != nullptr)
      Builder.CreateStore(
        createLoadFromContext(Builder, PC_LOCAL_SIZE, 0),
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
      Builder.CreateStore(
        ConstantInt::get(
          LocalSizeAllocas[0]->getAllocatedType(), WGLocalSizeX, 0),
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

  privatizeGlobals(
    F, Builder, {"_group_id_x", "_group_id_y", "_group_id_z"}, GroupIdArgs);

  if (WGAssumeZeroGlobalOffset) {
    privatizeGlobals(
        F, Builder,
        {"_global_offset_x", "_global_offset_y", "_global_offset_z"},
        {ConstantInt::get(SizeT, 0), ConstantInt::get(SizeT, 0),
         ConstantInt::get(SizeT, 0)});
  } else {
    privatizeGlobals(
        F, Builder,
        {"_global_offset_x", "_global_offset_y", "_global_offset_z"},
        globalHandlesToContextStructLoads(
            Builder,
            {"_global_offset_x", "_global_offset_y", "_global_offset_z"},
            PC_GLOBAL_OFFSET));
  }

  privatizeGlobals(
    F, Builder, {"_work_dim"},
    globalHandlesToContextStructLoads(Builder, {"_work_dim"}, PC_WORK_DIM));

  privatizeGlobals(
    F, Builder, {"_num_groups_x", "_num_groups_y", "_num_groups_z"},
    globalHandlesToContextStructLoads(
      Builder, {"_num_groups_x", "_num_groups_y", "_num_groups_z"},
      PC_NUM_GROUPS));

  if (currentPoclDevice->device_side_printf) {
    // Privatize _printf_buffer
    privatizeGlobals(
      F, Builder, {"_printf_buffer"}, {
        createLoadFromContext(
          Builder, PC_PRINTF_BUFFER)});

    privatizeGlobals(
      F, Builder, {"_printf_buffer_position"}, {
        createLoadFromContext(
          Builder, PC_PRINTF_BUFFER_POSITION)});

    privatizeGlobals(
      F, Builder, {"_printf_buffer_capacity"}, {
        createLoadFromContext(
          Builder, PC_PRINTF_BUFFER_CAPACITY)});
  }
}

// Creates a work group launcher function (called KERNELNAME_workgroup)
// that assumes kernel pointer arguments are stored as pointers to the
// actual buffers and that scalar data is loaded from the default memory.
void
Workgroup::createDefaultWorkgroupLauncher(llvm::Function *F) {

  IRBuilder<> Builder(M->getContext());

  std::string FuncName = "";
  FuncName = F->getName().str();

#ifdef LLVM_OLDER_THAN_9_0
  Function *WorkGroup =
    dyn_cast<Function>(M->getOrInsertFunction(
                         FuncName + "_workgroup", LauncherFuncT));
#else
  FunctionCallee fc = M->getOrInsertFunction(FuncName + "_workgroup", LauncherFuncT);
  Function *WorkGroup = dyn_cast<Function>(fc.getCallee());
#endif

  assert(WorkGroup != nullptr);
  BasicBlock *Block = BasicBlock::Create(M->getContext(), "", WorkGroup);
  Builder.SetInsertPoint(Block);

  Function::arg_iterator ai = WorkGroup->arg_begin();

  SmallVector<Value *, 8> Arguments;
  size_t i = 0;
  for (Function::const_arg_iterator ii = F->arg_begin(), ee = F->arg_end();
       ii != ee; ++ii) {

    if (i == F->arg_size() - 4)
      break;

    Type *ArgType = ii->getType();

    Value *GEP = Builder.CreateGEP(
        &*ai, ConstantInt::get(IntegerType::get(M->getContext(), 32), i));
    Value *Pointer = Builder.CreateLoad(GEP);

    Value *Arg;
    if (currentPoclDevice->device_alloca_locals &&
        isLocalMemFunctionArg(F, i)) {
      // Generate allocas for the local buffer arguments.
      PointerType *ParamType = dyn_cast<PointerType>(ArgType);
      Type *ArgElementType = ParamType->getElementType();
      if (ArgElementType->isArrayTy()) {
        // Known static local size (converted automatic local).
        Arg =
            new llvm::AllocaInst(ArgElementType, ParamType->getAddressSpace(),
                                 ConstantInt::get(IntegerType::get(*C, 32), 1),
                                 MaybeAlign(MAX_EXTENDED_ALIGNMENT), "local_auto", Block);
      } else {
        // Dynamic (runtime-set) size local argument.

        const DataLayout &DL = M->getDataLayout();
        // The size is passed directly instead of the pointer.
        uint64_t ParamByteSize = DL.getTypeStoreSize(ParamType);
        uint64_t ElementSize = DL.getTypeStoreSize(ArgElementType);
        Type *SizeIntType = IntegerType::get(*C, ParamByteSize * 8);
        Value *LocalArgByteSize =
            Builder.CreatePointerCast(Pointer, SizeIntType);
        Value *ElementCount = Builder.CreateUDiv(
            LocalArgByteSize, ConstantInt::get(SizeIntType, ElementSize));
        Arg = new llvm::AllocaInst(ArgElementType, ParamType->getAddressSpace(),
                                   ElementCount, MaybeAlign(MAX_EXTENDED_ALIGNMENT),
                                   "local_arg", Block);
      }
    } else {
      // If it's a pass by value pointer argument, we just pass the pointer
      // as is to the function, no need to load from it first.
      if (ii->hasByValAttr()) {
        Arg = Builder.CreatePointerCast(Pointer, ArgType);
      } else {
        Arg = Builder.CreatePointerCast(Pointer, ArgType->getPointerTo());
        Arg = Builder.CreateLoad(Arg);
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

  Builder.CreateCall(F, ArrayRef<Value *>(Arguments));
  Builder.CreateRetVoid();
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
      TypeInBuf = Arg.getType()->getPointerElementType();
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
    uint64_t Alignment = ByteSize;

    assert(ByteSize > 0 && "Arg type size is zero?");
    Offset = align64(Offset, Alignment);
    ArgBufferOffsets[i] = Offset;
    Offset += ByteSize;
  }
}

static LLVMValueRef
createArgBufferLoad(LLVMBuilderRef Builder, LLVMValueRef ArgBufferPtr,
                    uint64_t *ArgBufferOffsets, LLVMValueRef F,
                    unsigned ParamIndex) {

  LLVMValueRef Param = LLVMGetParam(F, ParamIndex);
  LLVMTypeRef ParamType = LLVMTypeOf(Param);

  LLVMModuleRef M = LLVMGetGlobalParent(F);
  LLVMContextRef LLVMContext = LLVMGetModuleContext(M);

  uint64_t ArgPos = ArgBufferOffsets[ParamIndex];
  LLVMValueRef Offs =
      LLVMConstInt(LLVMInt32TypeInContext(LLVMContext), ArgPos, 0);
  LLVMValueRef ArgByteOffset =
      LLVMBuildGEP(Builder, ArgBufferPtr, &Offs, 1, "arg_byte_offset");

  if (isByValPtrArgument(cast<Argument>(*unwrap(Param)))) {
    // In case of byval arguments (private structs), the struct
    // is in the arg buffer directly. Just refer to its address.
    return LLVMBuildPointerCast(Builder, ArgByteOffset, ParamType,
                                "inval_arg_ptr");
  } else {
    LLVMValueRef ArgOffsetBitcast = LLVMBuildPointerCast(
        Builder, ArgByteOffset, LLVMPointerType(ParamType, 0), "arg_ptr");
    return LLVMBuildLoad(Builder, ArgOffsetBitcast, "");
  }
}

/**
 * Creates a work group launcher with all the argument data passed
 * in a single argument buffer.
 *
 * All argument values, including pointers are stored directly in the
 * argument buffer with natural alignment. The rules for populating the
 * buffer are those of the HSA kernel calling convention. The name of
 * the generated function is KERNELNAME_workgroup_argbuffer.
 */
Function*
Workgroup::createArgBufferWorkgroupLauncher(Function *Func,
                                            std::string KernName) {

  LLVMValueRef F = wrap(Func);
  uint64_t ArgCount = LLVMCountParams(F);
  uint64_t ArgBufferOffsets[ArgCount];
  LLVMModuleRef M = wrap(this->M);

  computeArgBufferOffsets(F, ArgBufferOffsets);

  LLVMContextRef LLVMContext = LLVMGetModuleContext(M);

  LLVMTypeRef Int8Type = LLVMInt8TypeInContext(LLVMContext);
  LLVMTypeRef Int32Type = LLVMInt32TypeInContext(LLVMContext);
  LLVMTypeRef Int64Type = LLVMInt64TypeInContext(LLVMContext);

  LLVMTypeRef ArgsPtrType =
    LLVMPointerType(Int8Type, currentPoclDevice->args_as_id);

  LLVMTypeRef CtxPtrType =
    LLVMPointerType(Int8Type, currentPoclDevice->context_as_id);

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

  LLVMValueRef Args[ArgCount];
  LLVMValueRef ArgBuffer = LLVMGetParam(WrapperKernel, 0);
  size_t i = 0;
  for (; i < ArgCount - HiddenArgs + 1; ++i) {

    if (currentPoclDevice->device_alloca_locals &&
        isLocalMemFunctionArg(Func, i)) {

      // Generate allocas for the local buffer arguments.

      LLVMValueRef Param = LLVMGetParam(F, i);
      LLVMTypeRef ParamType = LLVMTypeOf(Param);

      LLVMTargetDataRef DataLayout = LLVMGetModuleDataLayout(M);

      LLVMTypeRef ArgElementType = LLVMGetElementType(ParamType);
      LLVMValueRef LocalArgAlloca = nullptr;

      if (LLVMGetTypeKind(ArgElementType) == LLVMArrayTypeKind) {

        // Known static local size (converted automatic local).
        LocalArgAlloca = wrap(new llvm::AllocaInst(
            unwrap(ArgElementType), LLVMGetPointerAddressSpace(ParamType),
            unwrap(LLVMConstInt(Int32Type, 1, 0)), MaybeAlign(MAX_EXTENDED_ALIGNMENT),
            "local_auto", unwrap(Block)));
      } else {

        // Dynamic (runtime-set) size local argument.

        uint64_t ParamByteSize = LLVMStoreSizeOfType(DataLayout, ParamType);
        LLVMTypeRef ParamIntType = ParamByteSize == 4 ? Int32Type : Int64Type;

        uint64_t ArgPos = ArgBufferOffsets[i];
        LLVMValueRef Offs = LLVMConstInt(Int32Type, ArgPos, 0);
        LLVMValueRef SizeByteOffset =
            LLVMBuildGEP(Builder, ArgBuffer, &Offs, 1, "size_byte_offset");
        LLVMValueRef SizeOffsetBitcast =
            LLVMBuildPointerCast(Builder, SizeByteOffset,
                                 LLVMPointerType(ParamIntType, 0), "size_ptr");

        // The buffer size passed from the runtime is a byte size, we
        // need to convert it to an element count for the alloca.
        LLVMValueRef LocalArgByteSize =
            LLVMBuildLoad(Builder, SizeOffsetBitcast, "byte_size");
        uint64_t ElementSize = LLVMStoreSizeOfType(DataLayout, ArgElementType);
        LLVMValueRef ElementCount =
            LLVMBuildUDiv(Builder, LocalArgByteSize,
                          LLVMConstInt(ParamIntType, ElementSize, 0), "");
        LocalArgAlloca = wrap(new llvm::AllocaInst(
            unwrap(LLVMGetElementType(ParamType)),
            LLVMGetPointerAddressSpace(ParamType), unwrap(ElementCount),
            MaybeAlign(MAX_EXTENDED_ALIGNMENT), "local_arg", unwrap(Block)));
      }
      Args[i] = LocalArgAlloca;
    } else {
      Args[i] = createArgBufferLoad(Builder, ArgBuffer, ArgBufferOffsets, F, i);
    }
  }

  size_t Arg = 2;
  // Pass the group ids.
  Args[i++] = LLVMGetParam(WrapperKernel, Arg++);
  Args[i++] = LLVMGetParam(WrapperKernel, Arg++);
  Args[i++] = LLVMGetParam(WrapperKernel, Arg++);

  assert (i == ArgCount);

  // Pass the context object.
  LLVMValueRef Call = LLVMBuildCall(Builder, F, Args, ArgCount, "");
  LLVMBuildRetVoid(Builder);

  llvm::CallInst *CallI = llvm::dyn_cast<llvm::CallInst>(llvm::unwrap(Call));
  CallI->setCallingConv(Func->getCallingConv());

  return llvm::dyn_cast<llvm::Function>(llvm::unwrap(WrapperKernel));
}

/**
 * Creates a launcher function that executes all work-items in the grid by
 * launching a given work-group function for all work-group ids.
 *
 * The function adheres to the PHSA calling convention where the first two
 * arguments are for PHSA's context data, and the third one is the argument
 * buffer. The name will be phsa_kernel.KERNELNAME_grid_launcher.
 */
void
Workgroup::createGridLauncher(Function *KernFunc, Function *WGFunc,
                              std::string KernName) {

  LLVMValueRef Kernel = llvm::wrap(KernFunc);
  LLVMValueRef WGF = llvm::wrap(WGFunc);
  LLVMModuleRef M = llvm::wrap(this->M);
  LLVMContextRef LLVMContext = LLVMGetModuleContext(M);

  LLVMTypeRef Int8Type = LLVMInt8TypeInContext(LLVMContext);
  LLVMTypeRef Int8PtrType = LLVMPointerType(Int8Type, 0);
  LLVMTypeRef ArgsPtrType =
      LLVMPointerType(Int8Type, currentPoclDevice->args_as_id);

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

  LLVMTypeRef ArgTypes[] = {
    LLVMTypeOf(LLVMGetParam(RunnerFunc, 0)),
    LLVMTypeOf(LLVMGetParam(RunnerFunc, 1)),
    LLVMTypeOf(LLVMGetParam(RunnerFunc, 2))};

  uint64_t KernArgCount = LLVMCountParams(Kernel);
  uint64_t KernArgBufferOffsets[KernArgCount];
  computeArgBufferOffsets(Kernel, KernArgBufferOffsets);

  // The second argument in the native phsa interface is auxiliary
  // driver-specific data that is passed as the last argument to
  // the grid launcher.
  LLVMValueRef AuxParam = LLVMGetParam(Launcher, 1);
  LLVMValueRef ArgBuffer = LLVMGetParam(Launcher, 2);

  // Load the pointer to the pocl context (in global memory),
  // assuming it is stored as the 4th last argument in the kernel.
  LLVMValueRef PoclCtx =
    createArgBufferLoad(Builder, ArgBuffer, KernArgBufferOffsets, Kernel,
                        KernArgCount - HiddenArgs);

  LLVMValueRef Args[4] = {
      LLVMBuildPointerCast(Builder, WGF, ArgTypes[0], "wg_func"),
      LLVMBuildPointerCast(Builder, ArgBuffer, ArgTypes[1], "args"),
      LLVMBuildPointerCast(Builder, PoclCtx, ArgTypes[2], "ctx"),
      LLVMBuildPointerCast(Builder, AuxParam, ArgTypes[1], "aux")};

  LLVMValueRef Call = LLVMBuildCall(Builder, RunnerFunc, Args, 4, "");
  LLVMBuildRetVoid(Builder);

  InlineFunctionInfo IFI;
  InlineFunction(dyn_cast<CallInst>(llvm::unwrap(Call)), IFI);
}

/**
 * Creates a work group launcher more suitable for the heterogeneous
 * host-device setup  (called KERNELNAME_workgroup_fast).
 *
 * 1) Pointer arguments are stored directly as pointers to the
 *    buffers in the argument buffer.
 *
 * 2) Scalar values are loaded from the global memory address
 *    space.
 *
 * This should minimize copying of data and memory allocation
 * at the device.
 */
void
Workgroup::createFastWorkgroupLauncher(llvm::Function *F) {

  IRBuilder<> builder(M->getContext());

  std::string funcName = "";
  funcName = F->getName().str();

#ifdef LLVM_OLDER_THAN_9_0
  Function *WorkGroup =
    dyn_cast<Function>(M->getOrInsertFunction(
                         funcName + "_workgroup_fast", LauncherFuncT));
#else
  FunctionCallee fc = M->getOrInsertFunction(
                         funcName + "_workgroup_fast", LauncherFuncT);
  Function *WorkGroup = dyn_cast<Function>(fc.getCallee());
#endif
  assert(WorkGroup != NULL);

  builder.SetInsertPoint(BasicBlock::Create(M->getContext(), "", WorkGroup));

  Function::arg_iterator ai = WorkGroup->arg_begin();

  SmallVector<Value*, 8> arguments;
  size_t i = 0;
  for (Function::const_arg_iterator ii = F->arg_begin(), ee = F->arg_end();
       ii != ee; ++ii, ++i) {

    if (i == F->arg_size() - 4)
      break;

    Type *t = ii->getType();
    Value *gep = builder.CreateGEP(&*ai,
            ConstantInt::get(IntegerType::get(M->getContext(), 32), i));
    Value *pointer = builder.CreateLoad(gep);

    if (t->isPointerTy()) {
      if (!ii->hasByValAttr()) {
        // Assume the pointer is directly in the arg array.
        arguments.push_back(builder.CreatePointerCast(pointer, t));
        continue;
      }

      // It's a pass by value pointer argument, use the underlying
      // element type in subsequent load.
      t = t->getPointerElementType();
    }

    // If it's a pass by value pointer argument, we just pass the pointer
    // as is to the function, no need to load from it first.
    Value *value;

    if (!ii->hasByValAttr() || ((PointerType*)t)->getAddressSpace() == 1)
      value = builder.CreatePointerCast
        (pointer, t->getPointerTo(currentPoclDevice->global_as_id));
    else
      value = builder.CreatePointerCast(pointer, t->getPointerTo());

    if (!ii->hasByValAttr()) {
      value = builder.CreateLoad(value);
    }

    arguments.push_back(value);
  }

  ++ai;
  arguments.push_back(&*ai);
  ++ai;
  arguments.push_back(&*ai);
  ++ai;
  arguments.push_back(&*ai);
  ++ai;
  arguments.push_back(&*ai);

  builder.CreateCall(F, ArrayRef<Value*>(arguments));
  builder.CreateRetVoid();
}

// Returns true in case the given function is a kernel that
// should be processed by the kernel compiler.
bool Workgroup::isKernelToProcess(const Function &F) {

  const Module *m = F.getParent();

  if (F.getMetadata("kernel_arg_access_qual") &&
      F.getMetadata("pocl_generated") == nullptr)
    return true;

  NamedMDNode *kernels = m->getNamedMetadata("opencl.kernels");
  if (kernels == NULL) {
    if (KernelName == "")
      return true;
    if (F.getName() == KernelName)
      return true;

    return false;
  }

  for (unsigned i = 0, e = kernels->getNumOperands(); i != e; ++i) {
    if (kernels->getOperand(i)->getOperand(0) == NULL)
      continue; // globaldce might have removed uncalled kernels
    Function *k =
      cast<Function>(
        dyn_cast<ValueAsMetadata>(kernels->getOperand(i)->getOperand(0))
        ->getValue());
    if (&F == k)
      return true;
  }

  return false;
}

// Returns true in case the given function is a kernel with work-group
// barriers inside it.
bool
Workgroup::hasWorkgroupBarriers(const Function &F)
{
  for (llvm::Function::const_iterator i = F.begin(), e = F.end();
       i != e; ++i) {
    const llvm::BasicBlock* bb = &*i;
    if (Barrier::hasBarrier(bb)) {

      // Ignore the implicit entry and exit barriers.
      if (Barrier::hasOnlyBarrier(bb) && bb == &F.getEntryBlock())
        continue;

      if (Barrier::hasOnlyBarrier(bb) &&
          bb->getTerminator()->getNumSuccessors() == 0)
        continue;

      return true;
    }
  }
  return false;
}
