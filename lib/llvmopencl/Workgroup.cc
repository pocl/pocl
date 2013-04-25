// LLVM module pass to create the single function (fully inlined)
// and parallelized kernel for an OpenCL workgroup.
// 
// Copyright (c) 2011 Universidad Rey Juan Carlos
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

#include "Workgroup.h"

#include "CanonicalizeBarriers.h"
#include "BarrierTailReplication.h"
#include "WorkitemReplication.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "config.h"
#ifdef LLVM_3_1
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InstrTypes.h"
#include "llvm/Module.h"
#elif defined LLVM_3_2
#include "llvm/IRBuilder.h"
#include "llvm/TypeBuilder.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InstrTypes.h"
#include "llvm/Module.h"
#else
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"
#endif
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cstdio>
#include <map>
#include <iostream>

#include "pocl.h"

#define STRING_LENGTH 32

using namespace std;
using namespace llvm;
using namespace pocl;

static void noaliasArguments(Function *F);
static Function *createLauncher(Module &M, Function *F);
static void privatizeContext(Module &M, Function *F);
static void createWorkgroup(Module &M, Function *F);
static void createWorkgroupFast(Module &M, Function *F);

// extern cl::opt<string> Header;
// extern cl::list<int> LocalSize;

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
          return StructType::get
            (TypeBuilder<types::i<32>, xcompile>::get(Context),
             TypeBuilder<types::i<64>[3], xcompile>::get(Context),
             TypeBuilder<types::i<64>[3], xcompile>::get(Context),
             TypeBuilder<types::i<64>[3], xcompile>::get(Context),
             NULL);
        }
      else if (size_t_width == 32)
        {
          return StructType::get
            (TypeBuilder<types::i<32>, xcompile>::get(Context),
             TypeBuilder<types::i<32>[3], xcompile>::get(Context),
             TypeBuilder<types::i<32>[3], xcompile>::get(Context),
             TypeBuilder<types::i<32>[3], xcompile>::get(Context),
             NULL);
        }
      else
        {
          assert (false && "Unsupported size_t width.");
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
      GLOBAL_OFFSET
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
  if (M.getPointerSize() == llvm::Module::Pointer64)
    {
      TypeBuilder<PoclContext, true>::setSizeTWidth(64);
    }
  else if (M.getPointerSize() == llvm::Module::Pointer32) 
    {
      TypeBuilder<PoclContext, true>::setSizeTWidth(32);
    }
  else 
    {
      assert (false && "Target has an unsupported pointer width.");
    }  

  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    if (!i->isDeclaration())
      i->setLinkage(Function::InternalLinkage);
  }

  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    if (!isKernelToProcess(*i)) continue;
    Function *L = createLauncher(M, i);
      
#if defined LLVM_3_2
    L->addFnAttr(Attributes::NoInline);
#else
    L->addFnAttr(Attribute::NoInline);
#endif

    privatizeContext(M, L);

    createWorkgroup(M, L);
    createWorkgroupFast(M, L);
  }

  Function *barrier = cast<Function> 
    (M.getOrInsertFunction("pocl.barrier",
                           Type::getVoidTy(M.getContext()),
                           NULL));
  BasicBlock *bb = BasicBlock::Create(M.getContext(), "", barrier);
  ReturnInst::Create(M.getContext(), 0, bb);
  
  return true;
}

/**
 * Marks the pointer arguments to the kernel functions as noalias.
 */
static void
noaliasArguments(Function *F)
{
  for (unsigned i = 0, e = F->getFunctionType()->getNumParams(); i < e; ++i)
    if (isa<PointerType> (F->getFunctionType()->getParamType(i)))
      F->setDoesNotAlias(i + 1); // arg 0 is return type
}

static Function *
createLauncher(Module &M, Function *F)
{
  SmallVector<Type *, 8> sv;

  for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
       i != e; ++i)
    sv.push_back (i->getType());
  sv.push_back(TypeBuilder<PoclContext*, true>::get(M.getContext()));

  FunctionType *ft = FunctionType::get(Type::getVoidTy(M.getContext()),
				       ArrayRef<Type *> (sv),
				       false);

  std::string funcName = "";
  funcName = F->getName().str();

  Function *L = Function::Create(ft,
				 Function::ExternalLinkage,
				 "_" + funcName,
				 &M);

  SmallVector<Value *, 8> arguments;
  Function::arg_iterator ai = L->arg_begin();
  for (unsigned i = 0, e = F->getArgumentList().size(); i != e; ++i)  {
    arguments.push_back(ai);
    ++ai;
  }  

  /* Copy the function attributes to transfer noalias etc. from the
     original kernel which will be inlined into the launcher. */
  L->setAttributes(F->getAttributes());

  Value *ptr, *v;
  char s[STRING_LENGTH];
  GlobalVariable *gv;

  IRBuilder<> builder(BasicBlock::Create(M.getContext(), "", L));

  ptr = builder.CreateStructGEP(ai,
				TypeBuilder<PoclContext, true>::WORK_DIM);
  gv = M.getGlobalVariable("_work_dim");
  if (gv != NULL) {
    v = builder.CreateLoad(builder.CreateConstGEP1_32(ptr, 0));
    builder.CreateStore(v, gv);
  }


  int size_t_width = 32;
  if (M.getPointerSize() == llvm::Module::Pointer64)
    size_t_width = 64;

  ptr = builder.CreateStructGEP(ai,
				TypeBuilder<PoclContext, true>::GROUP_ID);
  for (int i = 0; i < 3; ++i) {
    snprintf(s, STRING_LENGTH, "_group_id_%c", 'x' + i);
    gv = M.getGlobalVariable(s);
    if (gv != NULL) {
      if (size_t_width == 64)
        {
          v = builder.CreateLoad(builder.CreateConstGEP2_64(ptr, 0, i));
        }
      else
        {
          v = builder.CreateLoad(builder.CreateConstGEP2_32(ptr, 0, i));
        }
      builder.CreateStore(v, gv);
    }
  }

  ptr = builder.CreateStructGEP(ai,
				TypeBuilder<PoclContext, true>::NUM_GROUPS);
  for (int i = 0; i < 3; ++i) {
    snprintf(s, STRING_LENGTH, "_num_groups_%c", 'x' + i);
    gv = M.getGlobalVariable(s);
    if (gv != NULL) {
      if (size_t_width == 64)
        {
          v = builder.CreateLoad(builder.CreateConstGEP2_64(ptr, 0, i));
        }
      else 
        {
          v = builder.CreateLoad(builder.CreateConstGEP2_32(ptr, 0, i));
        }
      builder.CreateStore(v, gv);
    }
  }

  ptr = builder.CreateStructGEP(ai,
				TypeBuilder<PoclContext, true>::GLOBAL_OFFSET);
  for (int i = 0; i < 3; ++i) {
    snprintf(s, STRING_LENGTH, "_global_offset_%c", 'x' + i);
    gv = M.getGlobalVariable(s);
    if (gv != NULL) {
      if (size_t_width == 64)
        {
          v = builder.CreateLoad(builder.CreateConstGEP2_64(ptr, 0, i));
        }
      else
        {
          v = builder.CreateLoad(builder.CreateConstGEP2_32(ptr, 0, i));
        }
      builder.CreateStore(v, gv);
    }
  }

  CallInst *c = builder.CreateCall(F, ArrayRef<Value*>(arguments));
  builder.CreateRetVoid();

  InlineFunctionInfo IFI;
  InlineFunction(c, IFI);
  
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

    Value *gep = builder.CreateGEP(ai,
            ConstantInt::get(IntegerType::get(M.getContext(), 32), i));
    Value *pointer = builder.CreateLoad(gep);

    /* If it's a pass by value pointer argument, we just pass the pointer
     * as is to the function, no need to load form it first. */
    Value *value;
    if (ii->hasByValAttr()) {
        value = builder.CreateBitCast(pointer, t);
    } else {
        value = builder.CreateBitCast(pointer, t->getPointerTo());
        value = builder.CreateLoad(value);
    }

    arguments.push_back(value);
    ++i;
  }

  arguments.back() = ++ai;
  
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
    Value *gep = builder.CreateGEP(ai, 
            ConstantInt::get(IntegerType::get(M.getContext(), 32), i));
    Value *pointer = builder.CreateLoad(gep);
    Value *bc = NULL;
     
    if (t->isPointerTy()) {
      if (!ii->hasByValAttr()) {
        /* Assume the pointer is directly in the arg array. */
        arguments.push_back(builder.CreateBitCast(pointer, t));
        continue;
      }

      /* It's a pass by value pointer argument, use the underlying
       * element type in subsequent load. */
      t = t->getPointerElementType();
    }

    /* Assume the pointer points to data in the global memory space. */
    bc = builder.CreateBitCast(pointer,
            t->getPointerTo(POCL_ADDRESS_SPACE_GLOBAL));

    /* If it's a pass by value pointer argument, we just pass the pointer
     * as is to the function, no need to load from it first. */
    Value *value = builder.CreateBitCast(
        pointer, t->getPointerTo(POCL_ADDRESS_SPACE_GLOBAL));
    if (!ii->hasByValAttr()) {
        value = builder.CreateLoad(value);
    }
    
    arguments.push_back(value);
  }

  arguments.back() = ++ai;
  
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
    Function *k = cast<Function>(kernels->getOperand(i)->getOperand(0));
    if (&F == k)
      return true;
  }

  return false;
}
