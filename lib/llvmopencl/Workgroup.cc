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
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InstrTypes.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicInliner.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cstdio>
#include <map>
#include <iostream>

#define STRING_LENGTH 32

using namespace std;
using namespace llvm;
using namespace pocl;

static void noaliasArguments(Function *F);
static Function *createLauncher(Module &M, Function *F);
static void privatizeContext(Module &M, Function *F);
static void createWorkgroup(Module &M, Function *F);

// extern cl::opt<string> Header;
// extern cl::list<int> LocalSize;

cl::opt<string>
Kernel("kernel",
       cl::desc("Kernel function name"),
       cl::value_desc("kernel"),
       cl::init(""));

namespace llvm {

  typedef struct _pocl_context PoclContext;
  
  template<bool xcompile> class TypeBuilder<PoclContext, xcompile> {
  public:
    static StructType *get(LLVMContext &Context) {
        return StructType::get(
        TypeBuilder<types::i<32>, xcompile>::get(Context),
	TypeBuilder<types::i<32>[3], xcompile>::get(Context),
	TypeBuilder<types::i<32>[3], xcompile>::get(Context),
	TypeBuilder<types::i<32>[3], xcompile>::get(Context),
        NULL);
    }
  
    enum Fields {
      WORK_DIM,
      NUM_GROUPS,
      GROUP_ID,
      GLOBAL_OFFSET
    };
  };  

}  // namespace llvm
  
char Workgroup::ID = 0;
static RegisterPass<Workgroup> X("workgroup", "Workgroup creation pass");

bool
Workgroup::runOnModule(Module &M)
{
  // BasicInliner BI;

  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    if (!i->isDeclaration())
      i->setLinkage(Function::InternalLinkage);
  }
  //   BI.addFunction(i);
  // }

  // BI.inlineFunctions();

  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    if (isKernelToProcess(*i)) {
      Function *L = createLauncher(M, i);
      
      L->addFnAttr(Attribute::NoInline);
      noaliasArguments(L);

      privatizeContext(M, L);

      createWorkgroup(M, L);
    }
  }

  Function *barrier = cast<Function> 
    (M.getOrInsertFunction("pocl.barrier",
                           Type::getVoidTy(M.getContext()),
                           NULL));
  BasicBlock *bb = BasicBlock::Create(M.getContext(), "", barrier);
  ReturnInst::Create(M.getContext(), 0, bb);
  
  return true;
}

bool
Workgroup::isKernelToProcess(const Function &F)
{
  const Module *m = F.getParent();

  NamedMDNode *kernels = m->getNamedMetadata("opencl.kernels");
  if (kernels == NULL) {
    if (Kernel == "")
      return true;
    if (F.getName() == Kernel)
      return true;

    return false;
  }

  for (unsigned i = 0, e = kernels->getNumOperands(); i != e; ++i) {
    Function *k = cast<Function>(kernels->getOperand(i)->getOperand(0));
    if (&F == k)
      return true;
  }

  return false;
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
#ifdef LLVM_3_0
  funcName = F->getNameStr();
#else
  funcName = F->getName().str();
#endif

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

  Value *ptr, *v;
  char s[STRING_LENGTH];
  GlobalVariable *gv;

  IRBuilder<> builder(BasicBlock::Create(M.getContext(), "", L));

  // TODO: _num_groups_%c and friends should probably have type size_t
  // instead of unsigned int, because this may avoid integer
  // conversions when accessing these variables

  // TODO: _num_groups_%c and friends should probably be stored as
  // arrays instead of as 3 independent variables, because this may
  // lead to better code when the respective get_* functions are
  // called in a loop (array access instead of switch statement)

  ptr = builder.CreateStructGEP(ai,
				TypeBuilder<PoclContext, true>::WORK_DIM);
  gv = M.getGlobalVariable("_work_dim");
  if (gv != NULL) {
    v = builder.CreateLoad(builder.CreateConstGEP1_32(ptr, 0));
    builder.CreateStore(v, gv);
  }

  ptr = builder.CreateStructGEP(ai,
				TypeBuilder<PoclContext, true>::GROUP_ID);
  for (int i = 0; i < 3; ++i) {
    snprintf(s, STRING_LENGTH, "_group_id_%c", 'x' + i);
    gv = M.getGlobalVariable(s);
    if (gv != NULL) {
      v = builder.CreateLoad(builder.CreateConstGEP2_32(ptr, 0, i));
      builder.CreateStore(v, gv);
    }
  }

  ptr = builder.CreateStructGEP(ai,
				TypeBuilder<PoclContext, true>::NUM_GROUPS);
  for (int i = 0; i < 3; ++i) {
    snprintf(s, STRING_LENGTH, "_num_groups_%c", 'x' + i);
    gv = M.getGlobalVariable(s);
    if (gv != NULL) {
      v = builder.CreateLoad(builder.CreateConstGEP2_32(ptr, 0, i));
      builder.CreateStore(v, gv);
    }
  }

  ptr = builder.CreateStructGEP(ai,
				TypeBuilder<PoclContext, true>::GLOBAL_OFFSET);
  for (int i = 0; i < 3; ++i) {
    snprintf(s, STRING_LENGTH, "_global_offset_%c", 'x' + i);
    gv = M.getGlobalVariable(s);
    if (gv != NULL) {
      v = builder.CreateLoad(builder.CreateConstGEP2_32(ptr, 0, i));
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

static void
createWorkgroup(Module &M, Function *F)
{
  IRBuilder<> builder(M.getContext());

  FunctionType *ft =
    TypeBuilder<void(types::i<8>*[],
		     PoclContext*), true>::get(M.getContext());

  std::string funcName = "";
#ifdef LLVM_3_0
  funcName = F->getNameStr();
#else
  funcName = F->getName().str();
#endif
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
    Value *bc = builder.CreateBitCast(pointer, t->getPointerTo());
    arguments.push_back(builder.CreateLoad(bc));
    ++i;
  }

  arguments.back() = ++ai;
  
  builder.CreateCall(F, ArrayRef<Value*>(arguments));
  builder.CreateRetVoid();
}
