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
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/TypeBuilder.h>
#include <llvm/Pass.h>
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

#include "LLVMUtils.h"
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

static Function *createWrapper(Module &M, Function *F,
                               FunctionMapping &printfCache);
static void privatizeContext(Module &M, Function *F);
static void createDefaultWorkgroupLauncher(Module &M, Function *F);
static Function *createArgBufferWorkgroupLauncher(Module &M, Function *F,
                                             std::string KernName);
static void createGridLauncher(Module &Mod, Function *KernFunc,
                               Function *WGFunc, std::string KernName);

static void createFastWorkgroupLauncher(Module &M, Function *F);

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
    Function &OrigKernel = *i;
    if (!isKernelToProcess(OrigKernel)) continue;
    Function *L = createWrapper(M, &OrigKernel, printfCache);

    privatizeContext(M, L);

    if (currentPoclDevice->spmd) {
      // For SPMD machines there is no need for a WG launcher, the device will
      // call/handle the single-WI kernel function directly.
      kernels[&OrigKernel] = L;
    } else if (currentPoclDevice->arg_buffer_launcher) {
      Function *WGLauncher =
        createArgBufferWorkgroupLauncher(M, L, OrigKernel.getName().str());
      L->addFnAttr(Attribute::NoInline);
      L->removeFnAttr(Attribute::AlwaysInline);
      WGLauncher->addFnAttr(Attribute::AlwaysInline);
      createGridLauncher(M, L, WGLauncher, OrigKernel.getName().str());
    } else {
      createDefaultWorkgroupLauncher(M, L);
      // This is used only by TCE anymore. TODO: Replace all with the
      // ArgBuffer one.
      createFastWorkgroupLauncher(M, L);
    }
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
                      unsigned size_t_width, unsigned llvmtype, const char *name) {

  Value *ptr, *v = nullptr;
  GlobalVariable *gv;

  gv = M.getGlobalVariable(name);

  if (gv != nullptr) {
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

/**
 * Create a wrapper for the kernel and add pocl-specific hidden arguments.
 *
 * Also inlines the wrapped function to the wrapper.
 */
static Function *createWrapper(Module &M, Function *F,
                               FunctionMapping &printfCache) {

  SmallVector<Type *, 8> sv;

  for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
       i != e; ++i)
    sv.push_back (i->getType());
  if (currentPoclDevice->spmd) {
    PointerType* g_pc_ptr =
      PointerType::get(TypeBuilder<PoclContext, true>::get(M.getContext()), 1);
    sv.push_back(g_pc_ptr);
  } else {
    // pocl_context
    sv.push_back(TypeBuilder<PoclContext*, true>::get(M.getContext()));
    // group_x
    sv.push_back(TypeBuilder<types::i<32>, true>::get(M.getContext()));
    // group_y
    sv.push_back(TypeBuilder<types::i<32>, true>::get(M.getContext()));
    // group_z
    sv.push_back(TypeBuilder<types::i<32>, true>::get(M.getContext()));
  }

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
                         "_pocl_kernel_" + funcName,
                         &M);

  SmallVector<Value *, 8> arguments;
  Function::arg_iterator ai = L->arg_begin();
  for (unsigned i = 0, e = F->arg_size(); i != e; ++i) {
    arguments.push_back(&*ai);
    ++ai;
  }

#ifdef LLVM_OLDER_THAN_3_7
  llvm::Argument *ContextArg = ai++;
  llvm::Argument *GroupIdArgs[] = {ai++, ai++, ai};
#else
  llvm::Argument *ContextArg = &*(ai++);
  llvm::Argument *GroupIdArgs[] = {&*(ai++), &*(ai++), &*ai};
#endif

  // Copy the function attributes to transfer noalias etc. from the
  // original kernel which will be inlined into the launcher.
  L->setAttributes(F->getAttributes());

  IRBuilder<> builder(BasicBlock::Create(M.getContext(), "", L));

  GlobalVariable *gv = M.getGlobalVariable("_work_dim");
  if (gv != NULL) {
    Value *ptr;
#ifdef LLVM_OLDER_THAN_3_7
    ptr =
      builder.CreateStructGEP(ContextArg,
                              TypeBuilder<PoclContext, true>::WORK_DIM);
#else
    ptr =
      builder.CreateStructGEP(ContextArg->getType()->getPointerElementType(),
                              ContextArg,
                              TypeBuilder<PoclContext, true>::WORK_DIM);
#endif
    Value *v = builder.CreateLoad(builder.CreateConstGEP1_32(ptr, 0));
    builder.CreateStore(v, gv);
  }

  int size_t_width = 32;
  if (currentPoclDevice->address_bits == 64)
    size_t_width = 64;


  // Group ids are passed as hidden function arguments.
  for (int Dim = 0; Dim < 3; ++Dim) {
    std::ostringstream NameStrStr("_group_id_", std::ios::ate);
    NameStrStr << (char)('x' + Dim);
    std::string VarName = NameStrStr.str();
    GlobalVariable *gv = M.getGlobalVariable(VarName);
    if (gv != NULL) {
      builder.CreateStore(
        builder.CreateZExt(
          GroupIdArgs[Dim], gv->getType()->getPointerElementType()), gv);
    }

  }

  // The rest of the execution context / id data is passed in the pocl_context
  // struct.
  if (WGDynamicLocalSize)
    addGEPs(M, builder, ContextArg, size_t_width,
            TypeBuilder<PoclContext, true>::LOCAL_SIZE,
            "_local_size_%c");

  addGEPs(M, builder, ContextArg, size_t_width,
          TypeBuilder<PoclContext, true>::NUM_GROUPS,
          "_num_groups_%c");

  addGEPs(M, builder, ContextArg, size_t_width,
          TypeBuilder<PoclContext, true>::GLOBAL_OFFSET,
          "_global_offset_%c");

  Value *pb, *pbp, *pbc;
  if (currentPoclDevice->device_side_printf) {
    pb = addGEP1(M, builder, ContextArg, size_t_width,
                 TypeBuilder<PoclContext, true>::PRINTF_BUFFER,
                 "_printf_buffer");

    pbp = addGEP1(M, builder, ContextArg, size_t_width,
                  TypeBuilder<PoclContext, true>::PRINTF_BUFFER_POSITION,
                  "_printf_buffer_position");

    pbc = addGEP1(M, builder, ContextArg, size_t_width,
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

  L->setSubprogram(F->getSubprogram());

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
createDefaultWorkgroupLauncher(Module &M, Function *F) {

  IRBuilder<> builder(M.getContext());

  FunctionType *ft =
    TypeBuilder<void(types::i<8>*[],
		     PoclContext*,
		     types::i<32>,
		     types::i<32>,
		     types::i<32>), true>::get(M.getContext());

  std::string funcName = "";
  funcName = F->getName().str();

  Function *workgroup =
    dyn_cast<Function>(M.getOrInsertFunction(funcName + "_workgroup", ft));
  assert(workgroup != nullptr);

  builder.SetInsertPoint(BasicBlock::Create(M.getContext(), "", workgroup));

  Function::arg_iterator ai = workgroup->arg_begin();

  SmallVector<Value*, 8> arguments;
  size_t i = 0;
  for (Function::const_arg_iterator ii = F->arg_begin(), ee = F->arg_end();
       ii != ee; ++ii) {

    if (i == F->arg_size() - 4)
      break;

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

static void
computeArgBufferOffsets(LLVMValueRef F, uint64_t *ArgBufferOffsets) {

  uint64_t Offset = 0;
  uint64_t ArgCount = LLVMCountParams(F);
  LLVMModuleRef M = LLVMGetGlobalParent(F);

#ifdef LLVM_OLDER_THAN_3_9
  const char *str = LLVMGetDataLayout(M);
  LLVMTargetDataRef DataLayout = LLVMCreateTargetData(str);
#else
  LLVMTargetDataRef DataLayout = LLVMGetModuleDataLayout(M);
#endif

  // Compute the byte offsets of arguments in the arg buffer.
  for (size_t i = 0; i < ArgCount; i++) {
    LLVMValueRef Param = LLVMGetParam(F, i);
    LLVMTypeRef ParamType = LLVMTypeOf(Param);
    // TODO: This is a target specific type? We would like to get the
    // natural size or the "packed size" instead...
    uint64_t ByteSize = LLVMStoreSizeOfType(DataLayout, ParamType);
    uint64_t Alignment = ByteSize;

    assert (ByteSize && "Arg type size is zero?");
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
  LLVMValueRef ArgOffsetBitcast =
    LLVMBuildPointerCast(Builder, ArgByteOffset,
                         LLVMPointerType(ParamType, 0), "arg_ptr");
  return LLVMBuildLoad(Builder, ArgOffsetBitcast, "");
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
static Function*
createArgBufferWorkgroupLauncher(Module &Mod, Function *Func,
                                 std::string KernName) {

  LLVMValueRef F = wrap(Func);
  uint64_t ArgCount = LLVMCountParams(F);
  uint64_t ArgBufferOffsets[ArgCount];
  LLVMModuleRef M = wrap(&Mod);

  computeArgBufferOffsets(F, ArgBufferOffsets);

  LLVMContextRef LLVMContext = LLVMGetModuleContext(M);

  LLVMTypeRef Int8Type = LLVMInt8TypeInContext(LLVMContext);
  LLVMTypeRef Int32Type = LLVMInt32TypeInContext(LLVMContext);

  LLVMTypeRef Int8PtrType = LLVMPointerType(Int8Type, 0);

  std::ostringstream StrStr;
  StrStr << KernName;
  StrStr << "_workgroup_argbuffer";

  std::string FName = StrStr.str();
  const char *FunctionName = FName.c_str();

  LLVMTypeRef LauncherArgTypes[] = {
    Int8PtrType /* args */,
    Int8PtrType /* pocl_ctx */,
    Int32Type /* group_x */,
    Int32Type /* group_y */,
    Int32Type /* group_z */};

  LLVMTypeRef VoidType = LLVMVoidTypeInContext(LLVMContext);
  LLVMTypeRef LauncherFuncType =
    LLVMFunctionType(VoidType, LauncherArgTypes, 5, 0);

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
  for (; i < ArgCount - 3; ++i)
    Args[i] = createArgBufferLoad(Builder, ArgBuffer, ArgBufferOffsets, F, i);

  // Pass the group ids.
  Args[i++] = LLVMGetParam(WrapperKernel, 2);
  Args[i++] = LLVMGetParam(WrapperKernel, 3);
  Args[i++] = LLVMGetParam(WrapperKernel, 4);

  assert (i == ArgCount);

  // Pass the context object.
  LLVMBuildCall(Builder, F, Args, ArgCount, "");
  LLVMBuildRetVoid(Builder);

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
static void
createGridLauncher(Module &Mod, Function *KernFunc, Function *WGFunc,
                   std::string KernName) {

  LLVMValueRef Kernel = llvm::wrap(KernFunc);
  LLVMValueRef WGF = llvm::wrap(WGFunc);
  LLVMModuleRef M = llvm::wrap(&Mod);
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
  LLVMTypeRef LauncherFuncType = LLVMFunctionType(VoidType, LauncherArgTypes,
                                                  3, 0);

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
                        KernArgCount - 4);

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
static void
createFastWorkgroupLauncher(Module &M, Function *F)
{
  IRBuilder<> builder(M.getContext());

  FunctionType *ft =
    TypeBuilder<void(types::i<8>*[],
                     PoclContext*,
                     types::i<32>,
                     types::i<32>,
                     types::i<32>), true>::get(M.getContext());

  std::string funcName = "";
  funcName = F->getName().str();
  Function *workgroup =
    dyn_cast<Function>(M.getOrInsertFunction(funcName + "_workgroup_fast", ft));
  assert(workgroup != NULL);

  builder.SetInsertPoint(BasicBlock::Create(M.getContext(), "", workgroup));

  Function::arg_iterator ai = workgroup->arg_begin();

  SmallVector<Value*, 8> arguments;
  size_t i = 0;
  for (Function::const_arg_iterator ii = F->arg_begin(), ee = F->arg_end();
       ii != ee; ++ii, ++i) {

    if (i == F->arg_size() - 4)
      break;

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
