// LLVM module pass to create the single function (fully inlined)
// and parallelized kernel for an OpenCL workgroup.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos
//               2012-2018 Pekka Jääskeläinen
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
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#ifdef LLVM_OLDER_THAN_7
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

#include "TargetAddressSpaces.h"

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
cl::opt<string>
KernelName("kernel",
       cl::desc("Kernel function name"),
       cl::value_desc("kernel"),
       cl::init(""));

#ifdef LLVM_OLDER_THAN_7_0
namespace llvm {

  typedef struct _pocl_context PoclContext;

  template<bool xcompile> class TypeBuilder<PoclContext, xcompile> {
  public:
    static StructType *get(LLVMContext &Context) {
      if (size_t_width == 64)
        {
#ifdef LLVM_OLDER_THAN_5_0
        return StructType::get(
            TypeBuilder<types::i<32>, xcompile>::get(Context),
            TypeBuilder<types::i<64>[3], xcompile>::get(Context),
            TypeBuilder<types::i<64>[3], xcompile>::get(Context),
            TypeBuilder<types::i<64>[3], xcompile>::get(Context),
            TypeBuilder<types::i<8> *, xcompile>::get(Context),
            TypeBuilder<types::i<32> *, xcompile>::get(Context),
            TypeBuilder<types::i<32>, xcompile>::get(Context), NULL);
#else
        SmallVector<Type *, 10> Elements;
        Elements.push_back(TypeBuilder<types::i<32>, xcompile>::get(Context));
        Elements.push_back(
            TypeBuilder<types::i<64>[3], xcompile>::get(Context));
        Elements.push_back(
            TypeBuilder<types::i<64>[3], xcompile>::get(Context));
        Elements.push_back(
            TypeBuilder<types::i<64>[3], xcompile>::get(Context));
        Elements.push_back(TypeBuilder<types::i<8> *, xcompile>::get(Context));
        Elements.push_back(TypeBuilder<types::i<32> *, xcompile>::get(Context));
        Elements.push_back(TypeBuilder<types::i<32>, xcompile>::get(Context));
        return StructType::get(Context, Elements);
#endif
        }
      else if (size_t_width == 32)
        {
#ifdef LLVM_OLDER_THAN_5_0
          return StructType::get(
              TypeBuilder<types::i<32>, xcompile>::get(Context),
              TypeBuilder<types::i<32>[3], xcompile>::get(Context),
              TypeBuilder<types::i<32>[3], xcompile>::get(Context),
              TypeBuilder<types::i<32>[3], xcompile>::get(Context),
              TypeBuilder<types::i<8> *, xcompile>::get(Context),
              TypeBuilder<types::i<32> *, xcompile>::get(Context),
              TypeBuilder<types::i<32>, xcompile>::get(Context), NULL);
#else
          SmallVector<Type *, 10> Elements;
          Elements.push_back(
            TypeBuilder<types::i<32>, xcompile>::get(Context));
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

          return StructType::get(Context, Elements);
#endif
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
  PC_WORK_DIM,
  PC_NUM_GROUPS,
  PC_GLOBAL_OFFSET,
  PC_LOCAL_SIZE,
  PC_PRINTF_BUFFER,
  PC_PRINTF_BUFFER_POSITION,
  PC_PRINTF_BUFFER_CAPACITY
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
      Int32T, // WORK_DIM
      ArrayType::get(SizeT, 3), // NUM_GROUPS
      ArrayType::get(SizeT, 3), // GLOBAL_OFFSET
      ArrayType::get(SizeT, 3), // LOCAL_SIZE
      PointerType::get(Int8T, 0), // PRINTF_BUFFER
      PointerType::get(Int32T, 0), // PRINTF_BUFFER_POSITION
      Int32T); // PRINTF_BUFFER_CAPACITY

  LauncherFuncT =
    FunctionType::get(
      Type::getVoidTy(*C),
      {PointerType::get(
          PointerType::get(Type::getInt8Ty(*C), 0), 0),
          PointerType::get(PoclContextT, 0),
          SizeT, SizeT, SizeT}, false);
#endif

  assert ((SizeTWidth == 64 || SizeTWidth == 32) &&
          "Target has an unsupported pointer width.");

  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    if (!i->isDeclaration())
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

#if LLVM_OLDER_THAN_5_0
  Function *barrier = cast<Function>
    (M.getOrInsertFunction(BARRIER_FUNCTION_NAME,
                           Type::getVoidTy(M.getContext()), NULL));
#else
  Function *barrier = cast<Function>
    (M.getOrInsertFunction(BARRIER_FUNCTION_NAME,
                           Type::getVoidTy(M.getContext())));
#endif
  BasicBlock *bb = BasicBlock::Create(M.getContext(), "", barrier);
  ReturnInst::Create(M.getContext(), 0, bb);

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

// Creates a load from the hidden context structure argument for
// the given element.
llvm::Value *
Workgroup::createLoadFromContext(
  IRBuilder<> &Builder, int StructFieldIndex, int FieldIndex=-1) {

  Value *GEP;
#ifdef LLVM_OLDER_THAN_3_7
  GEP = Builder.CreateStructGEP(ContextArg, StructFieldIndex);
#else
  GEP = Builder.CreateStructGEP(ContextArg->getType()->getPointerElementType(),
                                ContextArg, StructFieldIndex);
#endif
  if (SizeTWidth == 64) {
    if (FieldIndex == -1)
      return Builder.CreateLoad(Builder.CreateConstGEP1_64(GEP, 0));
    else
      return Builder.CreateLoad(Builder.CreateConstGEP2_64(GEP, 0, FieldIndex));
  } else {
#ifdef LLVM_OLDER_THAN_3_7
    if (FieldIndex == -1)
      return Builder.CreateLoad(Builder.CreateConstGEP1_32(GEP, 0));
    else
      return Builder.CreateLoad(Builder.CreateConstGEP2_32(GEP, 0, FieldIndex));
#else
    if (FieldIndex == -1)
      return Builder.CreateLoad(Builder.CreateConstGEP1_32(GEP, 0));
    else
      return Builder.CreateLoad(
        Builder.CreateConstGEP2_32(
          GEP->getType()->getPointerElementType(), GEP, 0, FieldIndex));
#endif
  }
}

// TODO we should use __cl_printf users instead of searching the call tree
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
      if (callee->getName().equals("__cl_printf"))
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

      if (oldF->getName().equals("__cl_printf")) {
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

    CI->replaceAllUsesWith(newCI);
    ReplaceInstWithInst(CI, newCI);
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

  for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
       i != e; ++i)
    sv.push_back(i->getType());

  if (!currentPoclDevice->arg_buffer_launcher && currentPoclDevice->spmd) {
    sv.push_back(
      PointerType::get(PoclContextT, currentPoclDevice->global_as_id));
    HiddenArgs = 1;
  } else {
    // pocl_context
    sv.push_back(PointerType::get(PoclContextT, 0));
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

  FunctionType *ft = FunctionType::get(Type::getVoidTy(M->getContext()),
                                       ArrayRef<Type *> (sv),
                                       false);

  std::string funcName = "";
  funcName = F->getName().str();
  Function *L = NULL;
  if (!currentPoclDevice->arg_buffer_launcher && currentPoclDevice->spmd) {
    Function *F = M->getFunction(funcName);
    F->setName(funcName + "_original");
    L = Function::Create(ft,
                         Function::ExternalLinkage,
                         funcName,
                         M);
  } else
    L = Function::Create(ft,
                         Function::ExternalLinkage,
                         "_pocl_kernel_" + funcName,
                         M);

  SmallVector<Value *, 8> arguments;
  Function::arg_iterator ai = L->arg_begin();
  for (unsigned i = 0, e = F->arg_size(); i != e; ++i) {
    arguments.push_back(&*ai);
    ++ai;
  }

#ifdef LLVM_OLDER_THAN_3_7
  ContextArg = ai++;
  GroupIdArgs.resize(3);
  GroupIdArgs[0] = ai++;
  GroupIdArgs[1] = ai++;
  GroupIdArgs[2] = ai++;

#else

  ContextArg = &*(ai++);
  GroupIdArgs.resize(3);
  GroupIdArgs[0] = &*(ai++);
  GroupIdArgs[1] = &*(ai++);
  GroupIdArgs[2] = &*(ai++);

#endif

  // Copy the function attributes to transfer noalias etc. from the
  // original kernel which will be inlined into the launcher.
  L->setAttributes(F->getAttributes());

  IRBuilder<> Builder(BasicBlock::Create(M->getContext(), "", L));

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

#ifndef LLVM_OLDER_THAN_4_0
  // At least with LLVM 4.0, the runtime of AddAliasScopeMetadata of
  // llvm::InlineFunction explodes in case of kernels with restrict
  // metadata and a lot of lifetime markers. The issue produces at
  // least with EinsteinToolkit which has a lot of restrict kernel
  // args). Remove them here before inlining to speed it up.
  // TODO: Investigate the root cause.

  std::set<CallInst*> Calls;

  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {
      Instruction *Instr = dyn_cast<Instruction>(BI);
      if (!llvm::isa<CallInst>(Instr)) continue;
      CallInst *CallInstr = dyn_cast<CallInst>(Instr);
      Function *oldF = CallInstr->getCalledFunction();
      if (oldF != nullptr &&
          (oldF->getName().startswith("llvm.lifetime.end") ||
           oldF->getName().startswith("llvm.lifetime.start"))) {
        Calls.insert(CallInstr);
      }
    }
  }

  for (auto C : Calls) {
    C->eraseFromParent();
  }

#endif

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
    StructLoads[i] =
      createLoadFromContext(
			    Builder, StructFieldIndex,
          GlobalHandleNames.size() == 1 ? -1 : i);
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

  IRBuilder<> builder(M->getContext());

  std::string FuncName = "";
  FuncName = F->getName().str();

  Function *workgroup =
    dyn_cast<Function>(M->getOrInsertFunction(FuncName + "_workgroup",
                                              LauncherFuncT));
  assert(workgroup != nullptr);

  builder.SetInsertPoint(BasicBlock::Create(M->getContext(), "", workgroup));

  Function::arg_iterator ai = workgroup->arg_begin();

  SmallVector<Value*, 8> arguments;
  size_t i = 0;
  for (Function::const_arg_iterator ii = F->arg_begin(), ee = F->arg_end();
       ii != ee; ++ii) {

    if (i == F->arg_size() - 4)
      break;

    Type *t = ii->getType();

    Value *gep = builder.CreateGEP(&*ai,
            ConstantInt::get(IntegerType::get(M->getContext(), 32), i));
    Value *pointer = builder.CreateLoad(gep);

    // If it's a pass by value pointer argument, we just pass the pointer
    // as is to the function, no need to load form it first.
    Value *value;
    if (ii->hasByValAttr()) {
        value = builder.CreatePointerCast(pointer, t);
    } else {
        value = builder.CreatePointerCast(pointer, t->getPointerTo());
        value = builder.CreateLoad(value);
    }

    arguments.push_back(value);
    ++i;
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
  LLVMModuleRef M = LLVMGetGlobalParent(F);

#ifdef LLVM_OLDER_THAN_3_9
  const char *Str = LLVMGetDataLayout(M);
  LLVMTargetDataRef DataLayout = LLVMCreateTargetData(Str);
#else
  LLVMTargetDataRef DataLayout = LLVMGetModuleDataLayout(M);
#endif

  // Compute the byte offsets of arguments in the arg buffer.
  for (size_t i = 0; i < ArgCount; i++) {
    LLVMValueRef Param = LLVMGetParam(F, i);
    LLVMTypeRef ParamType = LLVMTypeOf(Param);
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
  LLVMTypeRef Int8PtrType = LLVMPointerType(Int8Type, 0);

  std::ostringstream StrStr;
  StrStr << KernName;
  StrStr << "_workgroup_argbuffer";

  std::string FName = StrStr.str();
  const char *FunctionName = FName.c_str();

  LLVMTypeRef LauncherArgTypes[] = {
    Int8PtrType, // args
    Int8PtrType, // pocl_ctx
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
  for (; i < ArgCount - HiddenArgs + 1; ++i)
    Args[i] = createArgBufferLoad(Builder, ArgBuffer, ArgBufferOffsets, F, i);

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

  std::ostringstream StrStr("phsa_kernel.", std::ios::ate);
  StrStr << KernName;
  StrStr << "_grid_launcher";

  std::string FName = StrStr.str();
  const char *FunctionName = FName.c_str();

  LLVMTypeRef LauncherArgTypes[] =
    {Int8PtrType /*phsactx0*/,
     Int8PtrType /*phsactx1*/,
     Int8PtrType /*args*/};

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

  LLVMValueRef ArgBuffer = LLVMGetParam(Launcher, 2);
  // Load the pointer to the pocl context (in global memory),
  // assuming it is stored as the 4th last argument in the kernel.
  LLVMValueRef PoclCtx =
    createArgBufferLoad(Builder, ArgBuffer, KernArgBufferOffsets, Kernel,
                        KernArgCount - HiddenArgs);

  LLVMValueRef Args[3] = {
    LLVMBuildPointerCast(Builder, WGF, ArgTypes[0], "wg_func"),
    LLVMBuildPointerCast(Builder, ArgBuffer, ArgTypes[1], "args"),
    LLVMBuildPointerCast(Builder, PoclCtx, ArgTypes[2], "ctx")};
  LLVMValueRef Call = LLVMBuildCall(Builder, RunnerFunc, Args, 3, "");
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
  Function *workgroup =
    dyn_cast<Function>(M->getOrInsertFunction(
                         funcName + "_workgroup_fast", LauncherFuncT));
  assert(workgroup != NULL);

  builder.SetInsertPoint(BasicBlock::Create(M->getContext(), "", workgroup));

  Function::arg_iterator ai = workgroup->arg_begin();

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
#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
      value = builder.CreatePointerCast
        (pointer, t->getPointerTo(POCL_FAKE_AS_GLOBAL));
#else
      value = builder.CreatePointerCast
        (pointer, t->getPointerTo(currentPoclDevice->global_as_id));
#endif
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
bool
Workgroup::isKernelToProcess(const Function &F)
{
  const Module *m = F.getParent();

  if (F.getMetadata("kernel_arg_access_qual"))
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
