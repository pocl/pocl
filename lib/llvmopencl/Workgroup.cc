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
#include "llvm/Transforms/Utils/BasicInliner.h"
#include "llvm/Transforms/Utils/Local.h"
#include <map>

using namespace std;
using namespace llvm;
using namespace locl;

static void createSizeGlobals(Module &M);
static void createTrampoline(Module &M, Function *F);

extern cl::opt<string> Kernel;
extern cl::list<int> LocalSize;

namespace {
  class Workgroup : public ModulePass {
  
  public:
    static char ID;
    Workgroup() : ModulePass(ID) {}

    virtual bool runOnModule(Module &M);
  };
}
  
char Workgroup::ID = 0;
static RegisterPass<Workgroup> X("workgroup", "Workgroup creation pass");

bool
Workgroup::runOnModule(Module &M)
{
  createSizeGlobals(M);

  BasicInliner BI;

  Function *F = NULL;
  
  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    if (!i->isDeclaration())
      i->setLinkage(Function::InternalLinkage);
    BI.addFunction(i);

    if (i->getName() == Kernel)
      F = i;
  }

  BI.inlineFunctions();

  BarrierTailReplication BTR;
  BTR.runOnFunction(*F);

  WorkitemReplication WR;
  WR.doInitialization(M);
  WR.runOnFunction(*F);
  WR.doFinalization(M);

  createTrampoline(M, F);

  F->removeFnAttr(Attribute::NoInline);
  F->addFnAttr(Attribute::AlwaysInline);
  BI.addFunction(F);
  BI.inlineFunctions();

  return true;
}

static void
createSizeGlobals(Module &M)
{
  GlobalVariable *x = M.getGlobalVariable("_size_x");
  if (x != NULL)
    x->setInitializer(ConstantInt::get(IntegerType::get(M.getContext(), 32),
				       LocalSize[0]));
  
  GlobalVariable *y = M.getGlobalVariable("_size_y");
  if (y != NULL)
    y->setInitializer(ConstantInt::get(IntegerType::get(M.getContext(), 32),
				       LocalSize[1]));
  
  GlobalVariable *z = M.getGlobalVariable("_size_z");
  if (z != NULL)
    z->setInitializer(ConstantInt::get(IntegerType::get(M.getContext(), 32),
				       LocalSize[2]));
}

static void
createTrampoline(Module &M, Function *F)
{
  IRBuilder<> builder(M.getContext());

  FunctionType *ft =
    TypeBuilder<void(types::i<32>,
		     types::i<32>,
		     types::i<32>), true>::get(M.getContext());

  Function *workgroup =
    dyn_cast<Function>(M.getOrInsertFunction("_workgroup", ft));
  assert(workgroup != NULL);

  builder.SetInsertPoint(BasicBlock::Create(M.getContext(), "", workgroup));

  Function::arg_iterator ai = workgroup->arg_begin();

  GlobalVariable *x = M.getGlobalVariable("_group_x");
  if (x != NULL)
    builder.CreateStore(ai, x);

  ++ai;

  GlobalVariable *y = M.getGlobalVariable("_group_y");
  if (y != NULL)
    builder.CreateStore(ai, y);

  ++ai;

  GlobalVariable *z = M.getGlobalVariable("_group_z");
  if (z != NULL)
    builder.CreateStore(ai, z);

  SmallVector<Value*, 8> arguments;
  int i = 0;
  for (Function::const_arg_iterator ii = F->arg_begin(), ee = F->arg_end();
       ii != ee; ++ii) {
    Type *t = ii->getType();
    
    GlobalVariable *gv =
      new GlobalVariable(M, t, false, GlobalVariable::ExternalLinkage,
			 UndefValue::get(t), "_arg" + Twine(i));
    arguments.push_back(builder.CreateLoad(gv));
    ++i;
  }
  
  builder.CreateCall(F, ArrayRef<Value*>(arguments));
  builder.CreateRetVoid();
}
