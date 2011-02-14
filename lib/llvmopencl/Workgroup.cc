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

static void createSizeGlobals(Module &M);
static void createTrampoline(Module &M, Function *F);
static void replicateRegion(BasicBlockVector bbv,
			    unsigned size_x, unsigned size_y, unsigned size_z,
			    GlobalVariable *gv_x,
			    GlobalVariable *gv_y,
			    GlobalVariable *gv_z,
			    Value2ValueMap &vm);

extern cl::opt<string> Kernel;

cl::list<int>
Size("size",
     cl::desc("Local size (x y z)"),
     cl::multi_val(3));

namespace {
  class Workgroup : public ModulePass {
  
  public:
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

  createSizeGlobals(M);

  BasicInliner BI;

  Function *F;
  
  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    BI.addFunction(i);

    if (i->getName() == Kernel)
      F = i;
  }

  createTrampoline(M, F);

  BI.inlineFunctions();

  GlobalVariable *gv_x = M.getGlobalVariable("_local_x");
  if (gv_x != NULL)
    gv_x->setInitializer(UndefValue::get(IntegerType::get(M.getContext(), 32)));

  GlobalVariable *gv_y = M.getGlobalVariable("_local_y");
  if (gv_y != NULL)
    gv_y->setInitializer(UndefValue::get(IntegerType::get(M.getContext(), 32)));

  GlobalVariable *gv_z = M.getGlobalVariable("_local_z");
  if (gv_z != NULL)
    gv_z->setInitializer(UndefValue::get(IntegerType::get(M.getContext(), 32)));

  BasicBlockVector original;

  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++e)
    original.push_back(i);

  Value2ValueMap vm;

  replicateRegion(original, Size[0], Size[1], Size[2],
		  gv_x, gv_y, gv_z, vm);

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

static void
createSizeGlobals(Module &M)
{
  GlobalVariable *x = M.getGlobalVariable("_size_x");
  if (x != NULL)
    x->setInitializer(ConstantInt::get(IntegerType::get(M.getContext(), 32),
				       Size[0]));
  
  GlobalVariable *y = M.getGlobalVariable("_size_y");
  if (y != NULL)
    y->setInitializer(ConstantInt::get(IntegerType::get(M.getContext(), 32),
				       Size[1]));
  
  GlobalVariable *z = M.getGlobalVariable("_size_z");
  if (z != NULL)
    z->setInitializer(ConstantInt::get(IntegerType::get(M.getContext(), 32),
				       Size[2]));
}

static void
createTrampoline(Module &M, Function *F)
{
  IRBuilder<> builder(M.getContext());

  const FunctionType *ft =
    TypeBuilder<void(types::i<32>,
		     types::i<32>,
		     types::i<32>), true>::get(M.getContext());

  Function *workgroup =
    dyn_cast<Function>(M.getOrInsertFunction("_workgroup", ft));
  assert(workgroup != NULL);

  builder.SetInsertPoint(BasicBlock::Create(M.getContext(), "", workgroup));

  Function::arg_iterator ai = workgroup->arg_begin();

  GlobalVariable *x = M.getGlobalVariable("_group_x");
  if (x != NULL) {
    x->setInitializer(UndefValue::get(IntegerType::get(M.getContext(), 32)));
    builder.CreateStore(ai, x);
  }

  ++ai;

  GlobalVariable *y = M.getGlobalVariable("_group_y");
  if (y != NULL) {
    y->setInitializer(UndefValue::get(IntegerType::get(M.getContext(), 32)));
    builder.CreateStore(ai, y);
  }

  ++ai;

  GlobalVariable *z = M.getGlobalVariable("_group_z");
  if (z != NULL) {
    z->setInitializer(UndefValue::get(IntegerType::get(M.getContext(), 32)));
    builder.CreateStore(ai, z);
  }

  SmallVector<Value*, 8> arguments;
  int i = 0;
  for (Function::const_arg_iterator ii = F->arg_begin(), ee = F->arg_end();
       ii != ee; ++ii) {
	const Type *t = ii->getType();
	
	GlobalVariable *gv =
	  new GlobalVariable(M, t, false, GlobalVariable::ExternalLinkage,
			     UndefValue::get(t), "_arg" + Twine(i));
	arguments.push_back(builder.CreateLoad(gv));
	++i;
  }
  
  builder.CreateCall(F, arguments.begin(), arguments.end());
  builder.CreateRetVoid();
}

// Replicate region formed by BasicBlocks in bbv, by size_{x,y,z}
// times in each corresponding dimension. Local IDs are stored
// in gv{x, y, z} at the start of code correponding to each
// workitem. vm is updated to contain a map from old value
// references to newly created ones. This function must be
// called for every region in dominance order, so vm already
// contains use update values for every previous BasicBlock.
static void
replicateRegion(BasicBlockVector bbv,
		unsigned size_x, unsigned size_y, unsigned size_z,
		GlobalVariable *gv_x,
		GlobalVariable *gv_y,
		GlobalVariable *gv_z,
		Value2ValueMap &vm)
{
  Module &M = *bbv.front()->getParent()->getParent();

  IRBuilder<> builder(M.getContext());

  BasicBlockVector new_bbv;
  BasicBlock *bb_insertion_point = bbv.front();

  for (int z = size_z - 1; z >= 0; --z) {
    for (int y = size_y - 1; y >= 0; --y) {
      for (int x = size_x - 1; x >= 0; --x) {
	
	if (x == 0 && y == 0 && z == 0)
	  continue;

	for (BasicBlockVector::const_iterator i = bbv.begin(),
	       e = bbv.end();
	     i != e; ++i) {
	  BasicBlock *bb = BasicBlock::Create((*i)->getContext(),
					      "",
					      (*i)->getParent(),
					      bb_insertion_point);
	  
	  new_bbv.push_back(bb);
	  vm.insert(make_pair((*i), bb));

	  for (BasicBlock::iterator ii = (*i)->begin(), ee = (*i)->end();
	       ii != ee; ++ii) {
	    Instruction *c = ii->clone();
	    vm.insert(make_pair(ii, c));
	    bb->getInstList().push_back(c);
	  }
	}

	vm.erase(bbv.back()->getTerminator());
	new_bbv.back()->getTerminator()->eraseFromParent();

	for (BasicBlockVector::iterator i = new_bbv.begin(), e = new_bbv.end();
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
	
	builder.SetInsertPoint(new_bbv.back());
	builder.CreateBr(bb_insertion_point);

	if (gv_x != NULL) {
	  builder.SetInsertPoint(bb_insertion_point, bb_insertion_point->front());
	  builder.CreateStore(ConstantInt::get(IntegerType::get(M.getContext(), 32), x),
			      gv_x);
	}
	
	bb_insertion_point = new_bbv.front();

	bbv = new_bbv;
	new_bbv.clear();
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
}

