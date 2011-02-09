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

#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/instrTypes.h"
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

typedef vector<BasicBlock*> BasicBlockVector;
typedef map<Value*, Value*> Value2ValueMap;

extern cl::opt<string> Kernel;

cl::list<int>
Size("size",
     cl::desc("Local size (x y z)"),
     cl::multi_val(3));

namespace {
  struct Workgroup : public ModulePass {
    
    static char ID;
    Workgroup() : ModulePass(ID) {}

    virtual bool runOnModule(Module &M);
//     virtual void getAnalisysUsage(AnalisysUsage &Info) const;
  };
  
  char Workgroup::ID = 0;
  INITIALIZE_PASS(Workgroup, "workgroup", "Workgroup creation pass", false, false);
}

bool
Workgroup::runOnModule(Module &M)
{
  IRBuilder<> builder(M.getContext());

  GlobalVariable *gv_sx = M.getGlobalVariable("_size_x");
  if (gv_sx != NULL)
    gv_sx->setInitializer(ConstantInt::get(IntegerType::get(M.getContext(), 32), Size[0]));

  GlobalVariable *gv_sy = M.getGlobalVariable("_size_y");
  if (gv_sy != NULL)
    gv_sy->setInitializer(ConstantInt::get(IntegerType::get(M.getContext(), 32), Size[1]));

  GlobalVariable *gv_sz = M.getGlobalVariable("_size_z");
  if (gv_sz != NULL)
    gv_sz->setInitializer(ConstantInt::get(IntegerType::get(M.getContext(), 32), Size[2]));

  const FunctionType *ft = TypeBuilder<void(types::i<32>, types::i<32>, types::i<32>), true>::get(M.getContext());
  Function *workgroup = dyn_cast<Function>(M.getOrInsertFunction("_workgroup", ft));
  assert(workgroup != NULL);

  builder.SetInsertPoint(BasicBlock::Create(M.getContext(), "", workgroup));

  Function::arg_iterator ai = workgroup->arg_begin();

  GlobalVariable *gv_gx = M.getGlobalVariable("_group_x");
  if (gv_gx != NULL) {
    gv_gx->setInitializer(UndefValue::get(IntegerType::get(M.getContext(), 32)));
    builder.CreateStore(ai, gv_gx);
  }

  ++ai;

  GlobalVariable *gv_gy = M.getGlobalVariable("_group_y");
  if (gv_gy != NULL) {
    gv_gy->setInitializer(UndefValue::get(IntegerType::get(M.getContext(), 32)));
    builder.CreateStore(ai, gv_gy);
  }

  ++ai;

  GlobalVariable *gv_gz = M.getGlobalVariable("_group_z");
  if (gv_gz != NULL) {
    gv_gz->setInitializer(UndefValue::get(IntegerType::get(M.getContext(), 32)));
    builder.CreateStore(ai, gv_gz);
  }

  GlobalVariable *gv_x = M.getGlobalVariable("_local_x");
  if (gv_x != NULL)
    gv_x->setInitializer(UndefValue::get(IntegerType::get(M.getContext(), 32)));

  GlobalVariable *gv_y = M.getGlobalVariable("_local_y");
  if (gv_y != NULL)
    gv_y->setInitializer(UndefValue::get(IntegerType::get(M.getContext(), 32)));

  GlobalVariable *gv_z = M.getGlobalVariable("_local_z");
  if (gv_z != NULL)
    gv_z->setInitializer(UndefValue::get(IntegerType::get(M.getContext(), 32)));

  SmallVector<Value*, 8> arguments;

  BasicInliner BI;

  Function *F;
  
  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    BI.addFunction(i);

    if (i->getName() == Kernel) {
      F = i;
      
      int j = 0;
      for (Function::const_arg_iterator ii = i->arg_begin(),
	     ee = i->arg_end();
	   ii != ee; ++ii) {
	const Type *t = ii->getType();
	
	GlobalVariable *gv = new GlobalVariable(M, t, false, GlobalVariable::ExternalLinkage,
						UndefValue::get(t), "_arg" + Twine(j));
	arguments.push_back(builder.CreateLoad(gv));
	++j;
      }
    }
  }

  builder.CreateCall(F, arguments.begin(), arguments.end());
  builder.CreateRetVoid();

  BI.inlineFunctions();

  BasicBlockVector original;
  BasicBlockVector replicated;

  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++e)
    original.push_back(i);

  BasicBlock *bb_insertion_point = original.front();

  for (int z = Size[2] - 1; z >= 0; --z) {
    for (int y = Size[1] - 1; y >= 0; --y) {
      for (int x = Size[0] - 1; x >= 0; --x) {

	if (x == 0 && y == 0 && z == 0)
	  continue;

	Value2ValueMap vm;

	for (BasicBlockVector::const_iterator i = original.begin(),
	       e = original.end();
	     i != e; ++i) {
	  BasicBlock *bb = BasicBlock::Create((*i)->getContext(),
					      "",
					      (*i)->getParent(),
					      bb_insertion_point);

	  replicated.push_back(bb);
	  vm.insert(make_pair((*i), bb));

	  for (BasicBlock::iterator ii = (*i)->begin(), ee = (*i)->end();
	       ii != ee; ++ii) {
	    Instruction *c = ii->clone();
	    vm.insert(make_pair(ii, c));
	    bb->getInstList().push_back(c);
	  }
	}

	vm.erase(original.back()->getTerminator());
	replicated.back()->getTerminator()->eraseFromParent();

	for (BasicBlockVector::iterator i = replicated.begin(), e = replicated.end();
	     i != e; ++i) {
	  for (BasicBlock::iterator ii = (*i)->begin(), ee = (*i)->end();
	       ii != ee; ++ii) {
	    for (Value2ValueMap::const_iterator vi = vm.begin(), ve = vm.end();
		 vi != ve; ++vi) {
	      assert(vi->first != NULL && vi->second != NULL);
	      ii->replaceUsesOfWith(vi->first, vi->second);
	    }
	  }
	}

	builder.SetInsertPoint(replicated.back());
	builder.CreateBr(bb_insertion_point);

	if (gv_x != NULL) {
	  builder.SetInsertPoint(bb_insertion_point, bb_insertion_point->front());
	  builder.CreateStore(ConstantInt::get(IntegerType::get(M.getContext(), 32), x),
			      gv_x);
	}

	bb_insertion_point = replicated.front();

	original = replicated;
	replicated.clear();
      }

      builder.SetInsertPoint(bb_insertion_point, bb_insertion_point->front());
      if (gv_y != NULL)
	builder.CreateStore(ConstantInt::get(IntegerType::get(M.getContext(), 32), y),
			    gv_y);
    }
    if (gv_z != NULL)
      builder.CreateStore(ConstantInt::get(IntegerType::get(M.getContext(), 32), z),
			  gv_z);
  }

  builder.SetInsertPoint(bb_insertion_point, bb_insertion_point->front());
  if (gv_x != NULL)
    builder.CreateStore(ConstantInt::get(IntegerType::get(M.getContext(), 32), 0),
			gv_x);

  F->removeFnAttr(Attribute::NoInline);
  F->addFnAttr(Attribute::AlwaysInline);
  BI.addFunction(F);
  BI.inlineFunctions();

//   // Changing the local_id variables from global to local to BasicBlock to allow
//   // DSE to work properly.
//   builder.SetInsertPoint(bb_insertion_point, bb_insertion_point->front());
//   AllocaInst *local_x = builder.CreateAlloca(IntegerType::get(M.getContext(), 32));
//   AllocaInst *local_y = builder.CreateAlloca(IntegerType::get(M.getContext(), 32));
//   AllocaInst *local_z = builder.CreateAlloca(IntegerType::get(M.getContext(), 32));
//   for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
//     for (BasicBlock::iterator ii = i->begin(), ee = i->end(); ii != ee; ++ii) {
//       ii->replaceUsesOfWith(gv_z, local_x);
//       ii->replaceUsesOfWith(gv_y, local_y);
//       ii->replaceUsesOfWith(gv_x, local_z);
//     }
//   }
  
//   createCFGSimplificationPass()->runOnFunction(*F);
//   for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
//     if (SimplifyCFG(i)) {
//       i = F->begin();
//       e = F->end();
//     }
//   }

//   Constant *c;
//   for (GlobalValue::use_iterator i = gv_x->use_begin(), e = gv_x->use_end();
//        i != e; ++i) {
//     assert(isa<LoadInst>(*i) || isa<StoreInst>(*i));
//     if (StoreInst *si = dyn_cast<StoreInst>(*i))
//       c = dyn_cast<Constant> (si->getValueOperand());
//     if (LoadInst *li = dyn_cast<LoadInst>(*i)) {
//       assert(c != NULL);
//       li->replaceAllUsesWith(c);
//     }
//   }

  return true;
}

// void
// Workgroup::getAnalisysUsage(AnalisysUsage &AU) const
// {
//   AU.addRequiredTransitive<
// }
