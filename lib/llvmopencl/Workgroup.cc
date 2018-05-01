// LLVM module pass to create the single function (fully inlined)
// and parallelized kernel for an OpenCL workgroup.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos
//               2012-2015 Pekka Jääskeläinen
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

#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

#include "CanonicalizeBarriers.h"
#include "BarrierTailReplication.h"
#include "WorkitemReplication.h"
#include "Barrier.h"
#include "Workgroup.h"

#include "LLVMUtils.h"
#include <cstdio>
#include <map>
#include <iostream>

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

static Function *createLauncher(Module &M, Function *F,
                                FunctionMapping &printfCache);
static void privatizeContext(Module &M, Function *F);
static void createWorkgroup(Module &M, Function *F);
static void createWorkgroupFast(Module &M, Function *F);

/* The kernel to process in this kernel compiler launch. */
cl::opt<string>
KernelName("kernel",
       cl::desc("Kernel function name"),
       cl::value_desc("kernel"),
       cl::init(""));

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
            TypeBuilder<types::i<64>[3], xcompile>::get(Context),
            TypeBuilder<types::i<8> *, xcompile>::get(Context),
            TypeBuilder<types::i<64> *, xcompile>::get(Context),
            TypeBuilder<types::i<64>, xcompile>::get(Context), NULL);
#else
        SmallVector<Type *, 10> Elements;
        Elements.push_back(TypeBuilder<types::i<32>, xcompile>::get(Context));
        Elements.push_back(
            TypeBuilder<types::i<64>[3], xcompile>::get(Context));
        Elements.push_back(
            TypeBuilder<types::i<64>[3], xcompile>::get(Context));
        Elements.push_back(
            TypeBuilder<types::i<64>[3], xcompile>::get(Context));
        Elements.push_back(
            TypeBuilder<types::i<64>[3], xcompile>::get(Context));
        Elements.push_back(TypeBuilder<types::i<8> *, xcompile>::get(Context));
        Elements.push_back(TypeBuilder<types::i<64> *, xcompile>::get(Context));
        Elements.push_back(TypeBuilder<types::i<64>, xcompile>::get(Context));

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

    /** 
     * We compile for various targets with various widths for the size_t
     * type that depends on the pointer type. 
     *
     * This should be set when the correct type is known. This is a hack
     * until a better way is found. It's not thread safe, e.g. if one
     * compiles multiple Modules for multiple different pointer widths in
     * a same process with multiple threads. */
    static void setSizeTWidth(int width) {
      size_t_width = width;
    }

    enum Fields {
      WORK_DIM,
      NUM_GROUPS,
      GROUP_ID,
      GLOBAL_OFFSET,
      LOCAL_SIZE,
      PRINTF_BUFFER,
      PRINTF_BUFFER_POSITION,
      PRINTF_BUFFER_CAPACITY
    };

  private:
    static int size_t_width;
  };

  template<bool xcompile>
  int TypeBuilder<PoclContext, xcompile>::size_t_width = 0;

}  // namespace llvm

char Workgroup::ID = 0;
static RegisterPass<Workgroup> X("workgroup", "Workgroup creation pass");


bool
Workgroup::runOnModule(Module &M)
{
  switch (currentPoclDevice->address_bits) {
  case 64:
    TypeBuilder<PoclContext, true>::setSizeTWidth(64);
    break;
  case 32:
    TypeBuilder<PoclContext, true>::setSizeTWidth(32);
    break;
  default:
    assert (false && "Target has an unsupported pointer width.");
    break;
  }

  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    if (!i->isDeclaration())
      i->setLinkage(Function::InternalLinkage);
  }

  // store the new and old kernel pairs in order to regenerate
  // all the metadata that used to point to the unmodified
  // kernels
  FunctionMapping kernels;

  // mapping of all functions which have been transformed to take
  // extra printf arguments.
  FunctionMapping printfCache;

  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    if (!isKernelToProcess(*i)) continue;
    Function *L = createLauncher(M, &*i, printfCache);

    privatizeContext(M, L);

    if (!currentPoclDevice->spmd) {
      createWorkgroup(M, L);
      createWorkgroupFast(M, L);
    }
    else
      kernels[&*i] = L;
  }

  if (currentPoclDevice->spmd) {
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

static void addGEPs(llvm::Module &M,
                    IRBuilder<> &builder,
                    llvm::Argument *a,
                    int size_t_width,
                    int llvmtype,
                    const char* format_str) {

  Value *ptr, *v;
  char s[STRING_LENGTH];
  GlobalVariable *gv;

#ifdef LLVM_OLDER_THAN_3_7
    ptr = builder.CreateStructGEP(a, llvmtype);
#else
    ptr = builder.CreateStructGEP(a->getType()->getPointerElementType(), a, llvmtype );
#endif

    for (unsigned i = 0; i < 3; ++i) {
      snprintf(s, STRING_LENGTH, format_str, 'x' + i);
      gv = M.getGlobalVariable(s);
      if (gv != NULL) {
        if (size_t_width == 64) {
            v = builder.CreateLoad(builder.CreateConstGEP2_64(ptr, 0, i));
        } else {
#ifdef LLVM_OLDER_THAN_3_7
            v = builder.CreateLoad(
                  builder.CreateConstGEP2_32(ptr, 0, i));
#else
            v = builder.CreateLoad(
              builder.CreateConstGEP2_32(
                ptr->getType()->getPointerElementType(), ptr, 0, i));
#endif
        }
        builder.CreateStore(v, gv);
      }
    }
}

static Value *addGEP1(llvm::Module &M, IRBuilder<> &builder, llvm::Argument *a,
                      int size_t_width, int llvmtype, const char *name) {

  Value *ptr, *v = nullptr;
  GlobalVariable *gv;

  gv = M.getGlobalVariable(name);

  if (gv != NULL) {
#ifdef LLVM_OLDER_THAN_3_7
    ptr = builder.CreateStructGEP(a, llvmtype);
#else
    ptr = builder.CreateStructGEP(a->getType()->getPointerElementType(), a,
                                  llvmtype);
#endif
    if (size_t_width == 64) {
      v = builder.CreateLoad(builder.CreateConstGEP1_64(ptr, 0));
    } else {
#ifdef LLVM_OLDER_THAN_3_7
      v = builder.CreateLoad(builder.CreateConstGEP1_32(ptr, 0));
#else
      v = builder.CreateLoad(builder.CreateConstGEP1_32(
          ptr->getType()->getPointerElementType(), ptr, 0));
#endif
    }
    builder.CreateStore(v, gv);
  }
  return v;
}

/* TODO we should use __cl_printf users instead of searching the call tree */
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

/* clones a function while adding 3 new arguments for printf calls. */
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

/* recursively replace _cl_printf calls with _pocl_printf calls, while
 * propagating the required pocl_context->printf_buffer arguments. */
static void replacePrintfCalls(Value *pb, Value *pbp, Value *pbc, bool isKernel,
                               Function *poclPrintf, Module &M, Function *L,
                               FunctionMapping &printfCache) {

  // if none of the kernels use printf(), it will not be linked into the module
  if (poclPrintf == nullptr)
    return;

  /* for kernel function, we are provided with proper printf arguments;
   * for non-kernel functions, we assume the function was replaced with
   * cloneFunctionWithPrintfArgs() and use the first three arguments. */
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

  // first, replace printf calls in body of L
  for (Function::iterator I = L->begin(), E = L->end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {
      Instruction *Instr = dyn_cast<Instruction>(BI);
      if (!llvm::isa<CallInst>(Instr))
        continue;
      CallInst *CallInstr = dyn_cast<CallInst>(Instr);
      Function *oldF = CallInstr->getCalledFunction();

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

//########################################################################

static Function *createLauncher(Module &M, Function *F,
                                FunctionMapping &printfCache) {

  SmallVector<Type *, 8> sv;

  for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
       i != e; ++i)
    sv.push_back (i->getType());
  if (currentPoclDevice->spmd) {
    PointerType* g_pc_ptr =
      PointerType::get(TypeBuilder<PoclContext, true>::get(M.getContext()), 1);
    sv.push_back(g_pc_ptr);
  } else
    sv.push_back(TypeBuilder<PoclContext*, true>::get(M.getContext()));

  FunctionType *ft = FunctionType::get(Type::getVoidTy(M.getContext()),
                                       ArrayRef<Type *> (sv),
                                       false);

  std::string funcName = "";
  funcName = F->getName().str();
  Function *L = NULL;
  if (currentPoclDevice->spmd) {
    Function *F = M.getFunction(funcName);
    F->setName(funcName + "_original");
    L = Function::Create(ft,
                         Function::ExternalLinkage,
                         funcName,
                         &M);
  } else
    L = Function::Create(ft,
                         Function::ExternalLinkage,
                         "_pocl_launcher_" + funcName,
                         &M);

  SmallVector<Value *, 8> arguments;
  Function::arg_iterator ai = L->arg_begin();
  for (unsigned i = 0, e = F->arg_size(); i != e; ++i) {
    arguments.push_back(&*ai);
    ++ai;
  }

  // Copy the function attributes to transfer noalias etc. from the
  // original kernel which will be inlined into the launcher.
  L->setAttributes(F->getAttributes());

  IRBuilder<> builder(BasicBlock::Create(M.getContext(), "", L));

  GlobalVariable *gv = M.getGlobalVariable("_work_dim");
  if (gv != NULL) {
    Value *ptr;
#ifdef LLVM_OLDER_THAN_3_7
    ptr =
      builder.CreateStructGEP(ai, TypeBuilder<PoclContext, true>::WORK_DIM);
#else
    ptr =
      builder.CreateStructGEP(ai->getType()->getPointerElementType(), &*ai,
                              TypeBuilder<PoclContext, true>::WORK_DIM);
#endif
    Value *v = builder.CreateLoad(builder.CreateConstGEP1_32(ptr, 0));
    builder.CreateStore(v, gv);
  }

  int size_t_width = 32;
  if (currentPoclDevice->address_bits == 64)
    size_t_width = 64;

#ifdef LLVM_OLDER_THAN_3_7
  llvm::Argument *a = ai;
#else
  llvm::Argument *a = &*ai;
#endif

  addGEPs(M, builder, a, size_t_width, TypeBuilder<PoclContext, true>::GROUP_ID,
          "_group_id_%c");

  if (WGDynamicLocalSize)
    addGEPs(M, builder, a, size_t_width, TypeBuilder<PoclContext, true>::LOCAL_SIZE,
            "_local_size_%c");

  addGEPs(M, builder, a, size_t_width, TypeBuilder<PoclContext, true>::NUM_GROUPS,
          "_num_groups_%c");

  addGEPs(M, builder, a, size_t_width, TypeBuilder<PoclContext, true>::GLOBAL_OFFSET,
          "_global_offset_%c");

  Value *pb, *pbp, *pbc;
  if (currentPoclDevice->device_side_printf) {
    pb = addGEP1(M, builder, a, size_t_width,
                 TypeBuilder<PoclContext, true>::PRINTF_BUFFER,
                 "_printf_buffer");

    pbp = addGEP1(M, builder, a, size_t_width,
                  TypeBuilder<PoclContext, true>::PRINTF_BUFFER_POSITION,
                  "_printf_buffer_position");

    pbc = addGEP1(M, builder, a, size_t_width,
                  TypeBuilder<PoclContext, true>::PRINTF_BUFFER_CAPACITY,
                  "_printf_buffer_capacity");
  } else {
    pb = pbp = pbc = nullptr;
  }


  CallInst *c = builder.CreateCall(F, ArrayRef<Value*>(arguments));
  builder.CreateRetVoid();

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
    Function *poclPrintf = M.getFunction("__pocl_printf");
    replacePrintfCalls(pb, pbp, pbc, true, poclPrintf, M, L, printfCache);
  }

  return L;
}

static void
privatizeContext(Module &M, Function *F)
{
  char s[STRING_LENGTH];
  GlobalVariable *gv[3];
  AllocaInst *ai[3] = {NULL, NULL, NULL};

  IRBuilder<> builder(F->getEntryBlock().getFirstNonPHI());

  // Privatize _local_id  
  for (int i = 0; i < 3; ++i) {
    snprintf(s, STRING_LENGTH, "_local_id_%c", 'x' + i);
    gv[i] = M.getGlobalVariable(s);
    if (gv[i] != NULL) {
      ai[i] = builder.CreateAlloca(gv[i]->getType()->getElementType(),
				   0, s);
      if(gv[i]->hasInitializer()) {
        Constant *c = gv[i]->getInitializer();
        builder.CreateStore(c, ai[i]);
      }
    }
  }
  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
    for (BasicBlock::iterator ii = i->begin(), ee = i->end();
         ii != ee; ++ii) {
      for (int j = 0; j < 3; ++j)
        ii->replaceUsesOfWith(gv[j], ai[j]);
    }
  }
  
  // Privatize _local_size
  for (int i = 0; i < 3; ++i) {
    snprintf(s, STRING_LENGTH, "_local_size_%c", 'x' + i);
    gv[i] = M.getGlobalVariable(s);
    if (gv[i] != NULL) {
      ai[i] = builder.CreateAlloca(gv[i]->getType()->getElementType(),
                                   0, s);
      if(gv[i]->hasInitializer()) {
        Constant *c = gv[i]->getInitializer();
        builder.CreateStore(c, ai[i]);
      }
    }
  }
  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
    for (BasicBlock::iterator ii = i->begin(), ee = i->end();
         ii != ee; ++ii) {
      for (int j = 0; j < 3; ++j)
        ii->replaceUsesOfWith(gv[j], ai[j]);
    }
  }

  // Privatize _work_dim
  gv[0] = M.getGlobalVariable("_work_dim");
  if (gv[0] != NULL) {
    ai[0] = builder.CreateAlloca(gv[0]->getType()->getElementType(),
                                 0, "_work_dim");
    if(gv[0]->hasInitializer()) {
      Constant *c = gv[0]->getInitializer();
      builder.CreateStore(c, ai[0]);
    }
  }
  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
    for (BasicBlock::iterator ii = i->begin(), ee = i->end();
	 ii != ee; ++ii) {
      ii->replaceUsesOfWith(gv[0], ai[0]);
    }
  }
  
  // Privatize _num_groups
  for (int i = 0; i < 3; ++i) {
    snprintf(s, STRING_LENGTH, "_num_groups_%c", 'x' + i);
    gv[i] = M.getGlobalVariable(s);
    if (gv[i] != NULL) {
      ai[i] = builder.CreateAlloca(gv[i]->getType()->getElementType(),
                                   0, s);
      if(gv[i]->hasInitializer()) {
        Constant *c = gv[i]->getInitializer();
        builder.CreateStore(c, ai[i]);
      }
    }
  }
  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
    for (BasicBlock::iterator ii = i->begin(), ee = i->end();
         ii != ee; ++ii) {
      for (int j = 0; j < 3; ++j)
        ii->replaceUsesOfWith(gv[j], ai[j]);
    }
  }

  // Privatize _group_id
  for (int i = 0; i < 3; ++i) {
    snprintf(s, STRING_LENGTH, "_group_id_%c", 'x' + i);
    gv[i] = M.getGlobalVariable(s);
    if (gv[i] != NULL) {
      ai[i] = builder.CreateAlloca(gv[i]->getType()->getElementType(),
                                   0, s);
      if(gv[i]->hasInitializer()) {
        Constant *c = gv[i]->getInitializer();
        builder.CreateStore(c, ai[i]);
      }
    }
  }
  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
    for (BasicBlock::iterator ii = i->begin(), ee = i->end();
         ii != ee; ++ii) {
      for (int j = 0; j < 3; ++j)
        ii->replaceUsesOfWith(gv[j], ai[j]);
    }
  }
  
  // Privatize _global_offset
  for (int i = 0; i < 3; ++i) {
    snprintf(s, STRING_LENGTH, "_global_offset_%c", 'x' + i);
    gv[i] = M.getGlobalVariable(s);
    if (gv[i] != NULL) {
      ai[i] = builder.CreateAlloca(gv[i]->getType()->getElementType(),
                                   0, s);
      if(gv[i]->hasInitializer()) {
        Constant *c = gv[i]->getInitializer();
        builder.CreateStore(c, ai[i]);
      }
    }
  }
  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
    for (BasicBlock::iterator ii = i->begin(), ee = i->end();
         ii != ee; ++ii) {
      for (int j = 0; j < 3; ++j)
        ii->replaceUsesOfWith(gv[j], ai[j]);
    }
  }

  if (currentPoclDevice->device_side_printf) {
    // Privatize _printf_buffer
    gv[0] = M.getGlobalVariable("_printf_buffer");
    if (gv[0] != NULL) {
      ai[0] = builder.CreateAlloca(gv[0]->getType()->getElementType(), 0,
                                   "_printf_buffer");
      if (gv[0]->hasInitializer()) {
        Constant *c = gv[0]->getInitializer();
        builder.CreateStore(c, ai[0]);
      }
    }

    // Privatize _printf_buffer_position
    gv[1] = M.getGlobalVariable("_printf_buffer_position");
    if (gv[1] != NULL) {
      ai[1] = builder.CreateAlloca(gv[1]->getType()->getElementType(), 0,
                                   "_printf_buffer_position");
      if (gv[1]->hasInitializer()) {
        Constant *c = gv[1]->getInitializer();
        builder.CreateStore(c, ai[1]);
      }
    }

    // Privatize _printf_buffer_capacity
    gv[2] = M.getGlobalVariable("_printf_buffer_capacity");
    if (gv[2] != NULL) {
      ai[2] = builder.CreateAlloca(gv[2]->getType()->getElementType(), 0,
                                   "_printf_buffer_capacity");
      if (gv[2]->hasInitializer()) {
        Constant *c = gv[2]->getInitializer();
        builder.CreateStore(c, ai[2]);
      }
    }

    for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
      for (BasicBlock::iterator ii = i->begin(), ee = i->end(); ii != ee; ++ii) {
        for (int j = 0; j < 3; ++j)
          ii->replaceUsesOfWith(gv[j], ai[j]);
      }
    }
  }
}

/**
 * Creates a work group launcher function (called KERNELNAME_workgroup)
 * that assumes kernel pointer arguments are stored as pointers to the
 * actual buffers and that scalar data is loaded from the default memory.
 */
static void
createWorkgroup(Module &M, Function *F)
{
  IRBuilder<> builder(M.getContext());

  FunctionType *ft =
    TypeBuilder<void(types::i<8>*[],
		     PoclContext*), true>::get(M.getContext());

  std::string funcName = "";
  funcName = F->getName().str();

  Function *workgroup =
    dyn_cast<Function>(M.getOrInsertFunction(funcName + "_workgroup", ft));
  assert(workgroup != NULL);

  builder.SetInsertPoint(BasicBlock::Create(M.getContext(), "", workgroup));

  Function::arg_iterator ai = workgroup->arg_begin();

  SmallVector<Value*, 8> arguments;
  int i = 0;
  for (Function::const_arg_iterator ii = F->arg_begin(), ee = F->arg_end();
       ii != ee; ++ii) {
    Type *t = ii->getType();

    Value *gep = builder.CreateGEP(&*ai,
            ConstantInt::get(IntegerType::get(M.getContext(), 32), i));
    Value *pointer = builder.CreateLoad(gep);

    /* If it's a pass by value pointer argument, we just pass the pointer
     * as is to the function, no need to load form it first. */
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

  arguments.back() = &*(++ai);

  builder.CreateCall(F, ArrayRef<Value*>(arguments));
  builder.CreateRetVoid();
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
static void
createWorkgroupFast(Module &M, Function *F)
{
  IRBuilder<> builder(M.getContext());

  FunctionType *ft =
    TypeBuilder<void(types::i<8>*[],
		     PoclContext*), true>::get(M.getContext());

  std::string funcName = "";
  funcName = F->getName().str();
  Function *workgroup =
    dyn_cast<Function>(M.getOrInsertFunction(funcName + "_workgroup_fast", ft));
  assert(workgroup != NULL);

  builder.SetInsertPoint(BasicBlock::Create(M.getContext(), "", workgroup));

  Function::arg_iterator ai = workgroup->arg_begin();

  SmallVector<Value*, 8> arguments;
  int i = 0;
  for (Function::const_arg_iterator ii = F->arg_begin(), ee = F->arg_end();
       ii != ee; ++i, ++ii) {
    Type *t = ii->getType();
    Value *gep = builder.CreateGEP(&*ai,
            ConstantInt::get(IntegerType::get(M.getContext(), 32), i));
    Value *pointer = builder.CreateLoad(gep);
     
    if (t->isPointerTy()) {
      if (!ii->hasByValAttr()) {
        /* Assume the pointer is directly in the arg array. */
        arguments.push_back(builder.CreatePointerCast(pointer, t));
        continue;
      }

      /* It's a pass by value pointer argument, use the underlying
       * element type in subsequent load. */
      t = t->getPointerElementType();
    }

    /* If it's a pass by value pointer argument, we just pass the pointer
     * as is to the function, no need to load from it first. */
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

  arguments.back() = &*(++ai);
  
  builder.CreateCall(F, ArrayRef<Value*>(arguments));
  builder.CreateRetVoid();
}


/**
 * Returns true in case the given function is a kernel that
 * should be processed by the kernel compiler.
 */
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

/**
 * Returns true in case the given function is a kernel 
 * with work-group barriers inside it.
 */
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
