// LLVM function pass to create loops that run all the work items
// in a work group while respecting barrier synchronization points.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen / Tampere University
//               2022-2023 Pekka Jääskeläinen / Intel Finland Oy
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

#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include "pocl.h"

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "workitem-loops"

#include "WorkitemLoops.h"
#include "Workgroup.h"
#include "Barrier.h"
#include "Kernel.h"
#include "WorkitemHandlerChooser.h"

//#define DUMP_CFGS

#include "DebugHelpers.h"

//#define DEBUG_WORK_ITEM_LOOPS

#include "VariableUniformityAnalysis.h"

#define CONTEXT_ARRAY_ALIGN 64

using namespace llvm;
using namespace pocl;

namespace {
static RegisterPass<WorkitemLoops> X("workitemloops",
                                     "Workitem loop generation pass");
}

char WorkitemLoops::ID = 0;

// Magic function used to allocate "local memory" dynamically. Used with SG/WG
// shuffles as temporary storage.
const char *POCL_LOCAL_MEM_ALLOCA_FUNC_NAME = "__pocl_local_mem_alloca";

// Another, which multiplies the size by the number of WIs in the WG.
const char *POCL_WORK_GROUP_ALLOCA_FUNC_NAME = "__pocl_work_group_alloca";

void
WorkitemLoops::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.addRequired<PostDominatorTreeWrapperPass>();

  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();

  AU.addRequired<VariableUniformityAnalysis>();
  AU.addPreserved<pocl::VariableUniformityAnalysis>();

  AU.addRequired<pocl::WorkitemHandlerChooser>();
  AU.addPreserved<pocl::WorkitemHandlerChooser>();

}

bool WorkitemLoops::runOnFunction(Function &F) {

  if (!Workgroup::isKernelToProcess(F))
    return false;

  if (getAnalysis<pocl::WorkitemHandlerChooser>().chosenHandler() !=
      pocl::WorkitemHandlerChooser::POCL_WIH_LOOPS)
    return false;

  DTP = &getAnalysis<DominatorTreeWrapperPass>();
  DT = &DTP->getDomTree();
  LI = &getAnalysis<LoopInfoWrapperPass>();

  PDT = &getAnalysis<PostDominatorTreeWrapperPass>();

  WGSizeInstr = nullptr;

  tempInstructionIndex = 0;

  LocalMemAllocaFuncDecl =
      F.getParent()->getFunction(POCL_LOCAL_MEM_ALLOCA_FUNC_NAME);

  WorkGroupAllocaFuncDecl =
      F.getParent()->getFunction(POCL_WORK_GROUP_ALLOCA_FUNC_NAME);

  bool Changed = ProcessFunction(F);

#if LLVM_MAJOR > 13
  Changed |= handleLocalMemAllocas(cast<Kernel>(F));
#endif

#ifdef DUMP_CFGS
  dumpCFG(F, F.getName().str() + "_after_wiloops.dot",
          original_parallel_regions);
#endif

  Changed |= fixUndominatedVariableUses(DTP, F);

#if 0
  /* Split large BBs so we can print the Dot without it crashing. */
  Changed |= chopBBs(F, *this);
  F.viewCFG();
#endif
  contextArrays.clear();
  tempInstructionIds.clear();

  releaseParallelRegions();
  return Changed;
}

std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
WorkitemLoops::CreateLoopAround
(ParallelRegion &region,
 llvm::BasicBlock *entryBB, llvm::BasicBlock *exitBB,
 bool peeledFirst, llvm::Value *localIdVar, size_t LocalSizeForDim,
 bool addIncBlock, llvm::Value *DynamicLocalSize)
{
  assert (localIdVar != NULL);

  /*

    Generate a structure like this for each loop level (x,y,z):

    for.init:

    ; if peeledFirst is false:
    store i32 0, i32* %_local_id_x, align 4

    ; if peeledFirst is true (assume the 0,0,0 iteration has been executed earlier)
    ; assume _local_id_x_first is is initialized to 1 in the peeled pregion copy
    store _local_id_x_first, i32* %_local_id_x, align 4
    store i32 0, %_local_id_x_first

    br label %for.body

    for.body: 

    ; the parallel region code here

    br label %for.inc

    for.inc:

    ; Separated inc and cond check blocks for easier loop unrolling later on.
    ; Can then chain N times for.body+for.inc to unroll.

    %2 = load i32* %_local_id_x, align 4
    %inc = add nsw i32 %2, 1

    store i32 %inc, i32* %_local_id_x, align 4
    br label %for.cond

    for.cond:

    ; loop header, compare the id to the local size
    %0 = load i32* %_local_id_x, align 4
    %cmp = icmp ult i32 %0, i32 123
    br i1 %cmp, label %for.body, label %for.end

    for.end:

    OPTIMIZE: Use a separate iteration variable across all the loops to iterate the context 
    data arrays to avoid needing multiplications to find the correct location, and to 
    enable easy vectorization of loading the context data when there are parallel iterations.
  */     

  llvm::BasicBlock *loopBodyEntryBB = entryBB;
  llvm::LLVMContext &C = loopBodyEntryBB->getContext();
  llvm::Function *F = loopBodyEntryBB->getParent();
  loopBodyEntryBB->setName(std::string("pregion_for_entry.") + entryBB->getName().str());

  assert (exitBB->getTerminator()->getNumSuccessors() == 1);

  llvm::BasicBlock *oldExit = exitBB->getTerminator()->getSuccessor(0);

  llvm::BasicBlock *forInitBB = 
    BasicBlock::Create(C, "pregion_for_init", F, loopBodyEntryBB);

  llvm::BasicBlock *loopEndBB = 
    BasicBlock::Create(C, "pregion_for_end", F, exitBB);

  llvm::BasicBlock *forCondBB = 
    BasicBlock::Create(C, "pregion_for_cond", F, exitBB);


  DTP->runOnFunction(*F);

  /* Collect the basic blocks in the parallel region that dominate the
     exit. These are used in determining whether load instructions may
     be executed unconditionally in the parallel loop (see below). */
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> dominatesExitBB;
  for (auto bb: region) {
    if (DT->dominates(bb, exitBB)) {
      dominatesExitBB.insert(bb);
    }
  }

  //  F->viewCFG();
  /* Fix the old edges jumping to the region to jump to the basic block
     that starts the created loop. Back edges should still point to the
     old basic block so we preserve the old loops. */
  BasicBlockVector preds;
  llvm::pred_iterator PI = 
    llvm::pred_begin(entryBB), 
    E = llvm::pred_end(entryBB);

  for (; PI != E; ++PI)
    {
      llvm::BasicBlock *bb = *PI;
      preds.push_back(bb);
    }    

  for (BasicBlockVector::iterator i = preds.begin();
       i != preds.end(); ++i)
    {
      llvm::BasicBlock *bb = *i;
      /* Do not fix loop edges inside the region. The loop
         is replicated as a whole to the body of the wi-loop.*/
      if (DT->dominates(loopBodyEntryBB, bb))
        continue;
      bb->getTerminator()->replaceUsesOfWith(loopBodyEntryBB, forInitBB);
    }

  IRBuilder<> builder(forInitBB);

  if (peeledFirst) {
    builder.CreateStore(builder.CreateLoad(SizeT, localIdXFirstVar), localIdVar);
    builder.CreateStore(ConstantInt::get(SizeT, 0), localIdXFirstVar);

    if (WGDynamicLocalSize) {
      llvm::Value *cmpResult;
      cmpResult = builder.CreateICmpULT(builder.CreateLoad(SizeT, localIdVar),
                                        builder.CreateLoad(SizeT, DynamicLocalSize));

      builder.CreateCondBr(cmpResult, loopBodyEntryBB, loopEndBB);
    } else {
      builder.CreateBr(loopBodyEntryBB);
    }
  } else {
    builder.CreateStore(ConstantInt::get(SizeT, 0), localIdVar);

    builder.CreateBr(loopBodyEntryBB);
  }

  exitBB->getTerminator()->replaceUsesOfWith(oldExit, forCondBB);
  if (addIncBlock)
    {
      AppendIncBlock(exitBB, localIdVar);
    }

  builder.SetInsertPoint(forCondBB);

  llvm::Value *cmpResult;
  if (!WGDynamicLocalSize)
    cmpResult = builder.CreateICmpULT(
                  builder.CreateLoad(SizeT, localIdVar),
                    ConstantInt::get(SizeT, LocalSizeForDim));
  else
    cmpResult = builder.CreateICmpULT(
                  builder.CreateLoad(SizeT, localIdVar),
                    builder.CreateLoad(SizeT, DynamicLocalSize));
  
  Instruction *loopBranch =
      builder.CreateCondBr(cmpResult, loopBodyEntryBB, loopEndBB);

  /* Add the metadata to mark a parallel loop. The metadata 
     refer to a loop-unique dummy metadata that is not merged
     automatically. */

  /* This creation of the identifier metadata is copied from
     LLVM's MDBuilder::createAnonymousTBAARoot(). */

  MDNode *Dummy = MDNode::getTemporary(C, ArrayRef<Metadata*>()).release();
  MDNode *AccessGroupMD = MDNode::getDistinct(C, {});
  MDNode *ParallelAccessMD = MDNode::get(
      C, {MDString::get(C, "llvm.loop.parallel_accesses"), AccessGroupMD});

  MDNode *Root = MDNode::get(C, {Dummy, ParallelAccessMD});

  // At this point we have
  //   !0 = metadata !{}            <- dummy
  //   !1 = metadata !{metadata !0} <- root
  // Replace the dummy operand with the root node itself and delete the dummy.
  Root->replaceOperandWith(0, Root);
  MDNode::deleteTemporary(Dummy);
  // We now have
  //   !1 = metadata !{metadata !1} <- self-referential root
  loopBranch->setMetadata("llvm.loop", Root);

  auto IsLoadUnconditionallySafe =
    [&dominatesExitBB](llvm::Instruction *insn) -> bool {
      assert(insn->mayReadFromMemory());
      // Checks that the instruction isn't in a conditional block.
      return dominatesExitBB.count(insn->getParent());
    };

  region.AddParallelLoopMetadata(AccessGroupMD, IsLoadUnconditionallySafe);

  builder.SetInsertPoint(loopEndBB);
  builder.CreateBr(oldExit);

  return std::make_pair(forInitBB, loopEndBB);
}

ParallelRegion*
WorkitemLoops::RegionOfBlock(llvm::BasicBlock *bb)
{
  for (ParallelRegion::ParallelRegionVector::iterator
           i = original_parallel_regions->begin(), 
           e = original_parallel_regions->end();
       i != e; ++i) 
  {
    ParallelRegion *region = (*i);
    if (region->HasBlock(bb)) return region;
  } 
  return NULL;
}

void WorkitemLoops::releaseParallelRegions() {
  if (original_parallel_regions) {
    for (auto i = original_parallel_regions->begin(),
              e = original_parallel_regions->end();
              i != e; ++i) {
      ParallelRegion *p = *i;
      delete p;
    }

    delete original_parallel_regions;
    original_parallel_regions = nullptr;
  }
}

bool
WorkitemLoops::ProcessFunction(Function &F)
{
  Kernel *K = cast<Kernel> (&F);

  llvm::Module *M = K->getParent();

  Initialize(K);
  unsigned workItemCount = WGLocalSizeX*WGLocalSizeY*WGLocalSizeZ;

  if (workItemCount == 1 && !WGDynamicLocalSize)
    {
      K->addLocalSizeInitCode(WGLocalSizeX, WGLocalSizeY, WGLocalSizeZ);
      ParallelRegion::insertLocalIdInit(&F.getEntryBlock(), 0, 0, 0);
      return true;
    }

  releaseParallelRegions();

  original_parallel_regions = K->getParallelRegions(&LI->getLoopInfo());

#ifdef DUMP_CFGS
  F.dump();
  dumpCFG(F, F.getName().str() + "_before_wiloops.dot",
          original_parallel_regions);
#endif

  IRBuilder<> builder(&*(F.getEntryBlock().getFirstInsertionPt()));
  localIdXFirstVar = builder.CreateAlloca(SizeT, 0, ".pocl.local_id_x_init");

  //  F.viewCFGOnly();

#if 0
  std::cerr << "### Original" << std::endl;
  F.viewCFGOnly();
#endif

#if 0
  for (ParallelRegion::ParallelRegionVector::iterator
           i = original_parallel_regions->begin(),
           e = original_parallel_regions->end();
       i != e; ++i) 
  {
    ParallelRegion *region = (*i);
    region->InjectRegionPrintF();
    region->InjectVariablePrintouts();
  }
#endif

  /* Count how many parallel regions share each entry node to
     detect diverging regions that need to be peeled. */
  std::map<llvm::BasicBlock*, int> entryCounts;

  for (ParallelRegion::ParallelRegionVector::iterator
           i = original_parallel_regions->begin(), 
           e = original_parallel_regions->end();
       i != e; ++i) 
  {
    ParallelRegion *region = (*i);
#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### Adding context save/restore for PR: ";
    region->dumpNames();
#endif
    fixMultiRegionVariables(region);
    entryCounts[region->entryBB()]++;
  }

#if 0
  std::cerr << "### After context code addition:" << std::endl;
  F.viewCFG();
#endif
  std::map<ParallelRegion*, bool> peeledRegion;
  for (ParallelRegion::ParallelRegionVector::iterator
           i = original_parallel_regions->begin(), 
           e = original_parallel_regions->end();
       i != e;  ++i) 
  {

    llvm::ValueToValueMapTy reference_map;
    ParallelRegion *original = (*i);

#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### handling region:" << std::endl;
    original->dumpNames();    
    //F.viewCFGOnly();
#endif

    /* In case of conditional barriers, the first iteration
       has to be peeled so we know which branch to execute
       with the work item loop. In case there are more than one
       parallel region sharing an entry BB, it's a diverging
       region.

       Post dominance of entry by exit does not work in case the
       region is inside a loop and the exit block is in the path
       towards the loop exit (and the function exit).
    */
    bool peelFirst =  entryCounts[original->entryBB()] > 1;
    
    peeledRegion[original] = peelFirst;

    std::pair<llvm::BasicBlock *, llvm::BasicBlock *> l;
    // the original predecessor nodes of which successor
    // should be fixed if not peeling
    BasicBlockVector preds;

    bool unrolled = false;
    if (peelFirst) 
      {
#ifdef DEBUG_WORK_ITEM_LOOPS
        std::cerr << "### conditional region, peeling the first iteration" << std::endl;
#endif
        ParallelRegion *replica = 
          original->replicate(reference_map, ".peeled_wi");
        replica->chainAfter(original);    
        replica->purge();
        
        l = std::make_pair(replica->entryBB(), replica->exitBB());
      }
    else
      {
        llvm::pred_iterator PI = 
          llvm::pred_begin(original->entryBB()), 
          E = llvm::pred_end(original->entryBB());

        for (; PI != E; ++PI)
          {
            llvm::BasicBlock *bb = *PI;
            if (DT->dominates(original->entryBB(), bb) &&
                (RegionOfBlock(original->entryBB()) == 
                 RegionOfBlock(bb)))
              continue;
            preds.push_back(bb);
          }

        unsigned unrollCount;
        if (getenv("POCL_WILOOPS_MAX_UNROLL_COUNT") != NULL)
            unrollCount = atoi(getenv("POCL_WILOOPS_MAX_UNROLL_COUNT"));
        else
            unrollCount = 1;
        /* Find a two's exponent unroll count, if available. */
        while (unrollCount >= 1)
          {
            if (WGLocalSizeX % unrollCount == 0 &&
                unrollCount <= WGLocalSizeX)
              {
                break;
              }
            unrollCount /= 2;
          }

        if (unrollCount > 1) {
            ParallelRegion *prev = original;
            llvm::BasicBlock *lastBB =
                AppendIncBlock(original->exitBB(), LocalIdXGlobal);
            original->AddBlockAfter(lastBB, original->exitBB());
            original->SetExitBB(lastBB);

            if (AddWIMetadata)
                original->AddIDMetadata(F.getContext(), 0);

            for (unsigned c = 1; c < unrollCount; ++c)
            {
                ParallelRegion *unrolled =
                    original->replicate(reference_map, ".unrolled_wi");
                unrolled->chainAfter(prev);
                prev = unrolled;
                lastBB = unrolled->exitBB();
                if (AddWIMetadata)
                    unrolled->AddIDMetadata(F.getContext(), c);
            }
            unrolled = true;
            l = std::make_pair(original->entryBB(), lastBB);
        } else {
            l = std::make_pair(original->entryBB(), original->exitBB());
        }
      }

    if (WGDynamicLocalSize) {
      GlobalVariable *gv;
      gv = M->getGlobalVariable("_local_size_x");
      if (gv == NULL)
        gv = new GlobalVariable(*M, SizeT, true, GlobalValue::CommonLinkage,
                                NULL, "_local_size_x", NULL,
                                GlobalValue::ThreadLocalMode::NotThreadLocal,
                                0, true);

      l = CreateLoopAround(*original, l.first, l.second, peelFirst,
                           LocalIdXGlobal, WGLocalSizeX, !unrolled, gv);

      gv = M->getGlobalVariable("_local_size_y");
      if (gv == NULL)
        gv = new GlobalVariable(*M, SizeT, false, GlobalValue::CommonLinkage,
                                NULL, "_local_size_y");

      l = CreateLoopAround(*original, l.first, l.second,
                           false, LocalIdYGlobal, WGLocalSizeY, !unrolled, gv);

      gv = M->getGlobalVariable("_local_size_z");
      if (gv == NULL)
        gv = new GlobalVariable(*M, SizeT, true, GlobalValue::CommonLinkage,
                                NULL, "_local_size_z", NULL,
                                GlobalValue::ThreadLocalMode::NotThreadLocal,
                                0, true);

      l = CreateLoopAround(*original, l.first, l.second,
                           false, LocalIdZGlobal, WGLocalSizeZ, !unrolled, gv);

    } else {
      if (WGLocalSizeX > 1) {
        l = CreateLoopAround(*original, l.first, l.second, peelFirst,
                             LocalIdXGlobal, WGLocalSizeX, !unrolled);
      }

      if (WGLocalSizeY > 1) {
        l = CreateLoopAround(*original, l.first, l.second, false,
                             LocalIdYGlobal, WGLocalSizeY);
      }

      if (WGLocalSizeZ > 1) {
        l = CreateLoopAround(*original, l.first, l.second, false,
                             LocalIdZGlobal, WGLocalSizeZ);
      }
    }

    /* Loop edges coming from another region mean B-loops which means 
       we have to fix the loop edge to jump to the beginning of the wi-loop 
       structure, not its body. This has to be done only for non-peeled
       blocks as the semantics is correct in the other case (the jump is
       to the beginning of the peeled iteration). */
    if (!peelFirst)
      {
        for (BasicBlockVector::iterator i = preds.begin();
             i != preds.end(); ++i)
          {
            llvm::BasicBlock *bb = *i;
            bb->getTerminator()->replaceUsesOfWith
              (original->entryBB(), l.first);
          }
      }
  }

  // for the peeled regions we need to add a prologue
  // that initializes the local ids and the first iteration
  // counter
  for (ParallelRegion::ParallelRegionVector::iterator
           i = original_parallel_regions->begin(), 
           e = original_parallel_regions->end();
       i != e; ++i)
  {
    ParallelRegion *pr = (*i);

    if (!peeledRegion[pr]) continue;
    pr->insertPrologue(0, 0, 0);
    builder.SetInsertPoint(&*(pr->entryBB()->getFirstInsertionPt()));
    builder.CreateStore(ConstantInt::get(SizeT, 1), localIdXFirstVar);
  }

  if (!WGDynamicLocalSize)
    K->addLocalSizeInitCode(WGLocalSizeX, WGLocalSizeY, WGLocalSizeZ);

  ParallelRegion::insertLocalIdInit(&F.getEntryBlock(), 0, 0, 0);

  return true;
}

// Add context save/restore code to variables that are defined in
// the given region and are used outside the region.
//
// Each such variable gets a slot in the stack frame. The variable
// is restored from the stack whenever it's used.
void WorkitemLoops::fixMultiRegionVariables(ParallelRegion *Region) {

  InstructionIndex InstructionsInRegion;
  InstructionVec InstructionsToFix;

  // Construct an index of the region's instructions so it's fast to figure
  // out if the variable uses are all in the region.
  for (BasicBlockVector::iterator I = Region->begin(); I != Region->end();
       ++I) {
    for (llvm::BasicBlock::iterator Instr = (*I)->begin(); Instr != (*I)->end();
         ++Instr) {
      InstructionsInRegion.insert(&*Instr);
    }
  }

  // Find all the instructions that define new values and check if they need
  // to be context saved.
  for (BasicBlockVector::iterator R = Region->begin(); R != Region->end();
       ++R) {
    for (llvm::BasicBlock::iterator I = (*R)->begin(); I != (*R)->end(); ++I) {

      llvm::Instruction *Instr = &*I;

      if (shouldNotBeContextSaved(&*Instr)) continue;

      for (Instruction::use_iterator UI = Instr->use_begin(),
             UE = Instr->use_end();
           UI != UE; ++UI) {
        llvm::Instruction *User = dyn_cast<Instruction>(UI->getUser());

        if (User == NULL)
          continue;

        // Allocas (originating from OpenCL C private arrays) should be
        // privatized always. Otherwise we end up reading the same array,
        // but replicating only the GEP pointing to it.
        if (isa<AllocaInst>(Instr) ||
            // If the instruction is used also inside another region (not
            // in a regionless BB like the B-loop construct BBs), we need
            // to context save it to pass the private data over.
            (InstructionsInRegion.find(User) ==
             InstructionsInRegion.end() &&
             RegionOfBlock(User->getParent()) != NULL)) {
          InstructionsToFix.push_back(Instr);
          break;
        }
      }
    }
  }

  // Finally generate the context save/restore code for the instructions
  // requiring it.
  for (InstructionVec::iterator I = InstructionsToFix.begin();
       I != InstructionsToFix.end(); ++I) {
#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### adding context/save restore for" << std::endl;
    (*I)->dump();
#endif
    addContextSaveRestore(*I);
  }
}

#if LLVM_MAJOR > 13
// Convert calls to the __pocl_{work_group,local_mem}_alloca() pseudo function
// to allocas.
bool WorkitemLoops::handleLocalMemAllocas(Kernel &K) {

  std::vector<CallInst *> InstructionsToFix;

  for (BasicBlock &BB : K) {
    for (Instruction &I : BB) {

      if (!isa<CallInst>(I)) continue;
      CallInst &Call = cast<CallInst>(I);

      if (Call.getCalledFunction() != LocalMemAllocaFuncDecl &&
          Call.getCalledFunction() != WorkGroupAllocaFuncDecl) continue;
      InstructionsToFix.push_back(&Call);
    }
  }

  bool Changed = false;
  for (CallInst *Call : InstructionsToFix) {
    Value *Size = Call->getArgOperand(0);
    Align Alignment =
      cast<ConstantInt>(Call->getArgOperand(1))->getAlignValue();

    IRBuilder<> Builder(K.getEntryBlock().getTerminator());

    if (Call->getCalledFunction() == WorkGroupAllocaFuncDecl) {
          Instruction *WGSize = getWorkGroupSizeInstr(K);
          Size = Builder.CreateBinOp(Instruction::Mul, WGSize, Size);
    }
    AllocaInst *Alloca = new AllocaInst(
        llvm::Type::getInt8Ty(Call->getContext()), 0, Size, Alignment,
        "__pocl_wg_alloca", K.getEntryBlock().getTerminator());
#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### fixing..." << std::endl;
    Call->dump();
    std::cerr << "### to..." << std::endl;
    Alloca->dump();
#endif
    Call->replaceAllUsesWith(Alloca);
    Call->eraseFromParent();
    Changed = true;
  }
  return Changed;
}
#endif

llvm::Value *
WorkitemLoops::GetLinearWiIndex(llvm::IRBuilder<> &builder, llvm::Module *M,
                               ParallelRegion *region)
{
  GlobalVariable *LocalSizeXPtr =
    cast<GlobalVariable>(M->getOrInsertGlobal("_local_size_x", SizeT));
  GlobalVariable *LocalSizeYPtr =
    cast<GlobalVariable>(M->getOrInsertGlobal("_local_size_y", SizeT));

  assert(LocalSizeXPtr != NULL && LocalSizeYPtr != NULL);

  LoadInst* LoadX = builder.CreateLoad(SizeT, LocalSizeXPtr, "ls_x");
  LoadInst* LoadY = builder.CreateLoad(SizeT, LocalSizeYPtr, "ls_y");

  /* Form linear index from xyz coordinates:
       local_size_x * local_size_y * local_id_z  (z dimension)
     + local_size_x * local_id_y                 (y dimension)
     + local_id_x                                (x dimension)
  */
  Value* LocalSizeXTimesY =
    builder.CreateBinOp(Instruction::Mul, LoadX, LoadY, "ls_xy");

  Value* ZPart =
    builder.CreateBinOp(Instruction::Mul, LocalSizeXTimesY,
                        region->LocalIDZLoad(),
                        "tmp");

  Value* YPart =
    builder.CreateBinOp(Instruction::Mul, LoadX, region->LocalIDYLoad(),
                        "ls_x_y");

  Value* ZYSum =
    builder.CreateBinOp(Instruction::Add, ZPart, YPart,
                        "zy_sum");

  return builder.CreateBinOp(Instruction::Add, ZYSum, region->LocalIDXLoad(),
                             "linear_xyz_idx");
}

llvm::Instruction *WorkitemLoops::AddContextSave(llvm::Instruction *instruction,
                                                 llvm::AllocaInst *alloca) {

  if (isa<AllocaInst>(instruction)) {
    // If the variable to be context saved is itself an alloca, we have created
    // one big alloca that stores the data of all the work-items and return
    // pointers to that array. Thus, we need no initialization code other than
    // the context data alloca itself.
    return NULL;
  }

  /* Save the produced variable to the array. */
  BasicBlock::iterator definition = (dyn_cast<Instruction>(instruction))->getIterator();
  ++definition;
  while (isa<PHINode>(definition)) ++definition;

  IRBuilder<> builder(&*definition);
  std::vector<llvm::Value *> gepArgs;

  /* Reuse the id loads earlier in the region, if possible, to
     avoid messy output with lots of redundant loads. */
  ParallelRegion *region = RegionOfBlock(instruction->getParent());
  assert ("Adding context save outside any region produces illegal code." && 
          region != NULL);

  if (WGDynamicLocalSize)
    {
      Module *M = alloca->getParent()->getParent()->getParent();
      gepArgs.push_back(GetLinearWiIndex(builder, M, region));
    }
  else
    {
      gepArgs.push_back(ConstantInt::get(SizeT, 0));
      gepArgs.push_back(region->LocalIDZLoad());
      gepArgs.push_back(region->LocalIDYLoad());
      gepArgs.push_back(region->LocalIDXLoad());
    }

    return builder.CreateStore(
        instruction,
#ifdef LLVM_OLDER_THAN_15_0
        builder.CreateGEP(alloca->getType()->getPointerElementType(), alloca,
                          gepArgs));
#else
        builder.CreateGEP(alloca->getAllocatedType(), alloca, gepArgs));
#endif
}

llvm::Instruction *WorkitemLoops::AddContextRestore(
    llvm::Value *val, llvm::AllocaInst *alloca, llvm::Type *InstType,
    bool PoclWrapperStructAdded, llvm::Instruction *before, bool isAlloca) {

  assert(val != NULL);
  assert(alloca != NULL);
  IRBuilder<> builder(alloca);
  if (before != NULL) {
    builder.SetInsertPoint(before);
  } else if (isa<Instruction>(val)) {
    builder.SetInsertPoint(dyn_cast<Instruction>(val));
    before = dyn_cast<Instruction>(val);
  } else {
    assert(false && "Unknown context restore location!");
  }

  std::vector<llvm::Value *> gepArgs;

  /* Reuse the id loads earlier in the region, if possible, to
     avoid messy output with lots of redundant loads. */
  ParallelRegion *region = RegionOfBlock(before->getParent());
  assert ("Adding context save outside any region produces illegal code." && 
          region != NULL);

  if (WGDynamicLocalSize)
    {
      Module *M = alloca->getParent()->getParent()->getParent();
      gepArgs.push_back(GetLinearWiIndex(builder, M, region));
    }
  else
    {
      gepArgs.push_back(ConstantInt::get(SizeT, 0));
      gepArgs.push_back(region->LocalIDZLoad());
      gepArgs.push_back(region->LocalIDYLoad());
      gepArgs.push_back(region->LocalIDXLoad());
    }

  if (PoclWrapperStructAdded)
    gepArgs.push_back(
      ConstantInt::get(Type::getInt32Ty(alloca->getContext()), 0));

#ifdef LLVM_OLDER_THAN_15_0
  llvm::Instruction *gep = dyn_cast<Instruction>(
    builder.CreateGEP(
      alloca->getType()->getPointerElementType(), alloca, gepArgs));
#else
  llvm::Instruction *gep = dyn_cast<Instruction>(
    builder.CreateGEP(
      alloca->getAllocatedType(), alloca, gepArgs));
#endif


  if (isAlloca) {
    /* In case the context saved instruction was an alloca, we created a
       context array with pointed-to elements, and now want to return a
       pointer to the elements to emulate the original alloca. */
    return gep;
  }
  return builder.CreateLoad(InstType, gep);
}

// Returns the context array (alloca) for the given Value, creates it if not
// found.
//
// PoCLWrapperStructAdded will be set to true in case a wrapper struct was
// added to enforce proper alignment to the elements of the array.
llvm::AllocaInst *WorkitemLoops::getContextArray(llvm::Instruction *instruction,
                                                 bool &PoclWrapperStructAdded) {
  PoclWrapperStructAdded = false;
  /*
   * Unnamed temp instructions need a generated name for the
   * context array. Create one using a running integer.
   */
  std::ostringstream var;
  var << ".";

  if (std::string(instruction->getName().str()) != "")
    {
      var << instruction->getName().str();
    }
  else if (tempInstructionIds.find(instruction) != tempInstructionIds.end())
    {
      var << tempInstructionIds[instruction];
    }
  else
    {
      tempInstructionIds[instruction] = tempInstructionIndex++;
      var << tempInstructionIds[instruction];
    }

  var << ".pocl_context";
  std::string varName = var.str();

  if (contextArrays.find(varName) != contextArrays.end())
    return contextArrays[varName];

  BasicBlock &bb = instruction->getParent()->getParent()->getEntryBlock();
  IRBuilder<> builder(&*(bb.getFirstInsertionPt()));
  Function *FF = instruction->getParent()->getParent();
  Module *M = instruction->getParent()->getParent()->getParent();
  const llvm::DataLayout &Layout = M->getDataLayout();
  DICompileUnit *CU = nullptr;
  std::unique_ptr<DIBuilder> DB;
  if (M->debug_compile_units_begin() != M->debug_compile_units_end()) {
    CU = *M->debug_compile_units_begin();
    DB = std::unique_ptr<DIBuilder>{new DIBuilder(*M, true, CU)};
  }

  // find the debug metadata corresponding to this variable
  Value *DebugVal = nullptr;
  IntrinsicInst *DebugCall = nullptr;

  if (CU) {
    for (BasicBlock &BB : (*FF)) {
      for (Instruction &I : BB) {
        IntrinsicInst *CI = dyn_cast<IntrinsicInst>(&I);
        if (CI && (CI->getIntrinsicID() == llvm::Intrinsic::dbg_declare)) {
          Metadata *Meta =
              cast<MetadataAsValue>(CI->getOperand(0))->getMetadata();
          if (isa<ValueAsMetadata>(Meta)) {
            Value *V = cast<ValueAsMetadata>(Meta)->getValue();
            if (instruction == V) {
              DebugVal = V;
              DebugCall = CI;
              break;
            }
          }
        }
      }
    }
  }

#ifdef DEBUG_WORK_ITEM_LOOPS
  if (DebugVal && DebugCall) {
    std::cerr << "### DI INTRIN: \n";
    DebugCall->dump();
    std::cerr << "### DI VALUE:  \n";
    DebugVal->dump();
  }
#endif

  llvm::Type *elementType;
  if (isa<AllocaInst>(instruction))
    {
      /* If the variable to be context saved was itself an alloca,
         create one big alloca that stores the data of all the 
         work-items and directly return pointers to that array.
         This enables moving all the allocas to the entry node without
         breaking the parallel loop.
         Otherwise we would rely on a dynamic alloca to allocate 
         unique stack space to all the work-items when its wiloop
         iteration is executed. */
      elementType = 
        dyn_cast<AllocaInst>(instruction)->getAllocatedType();
    } 
  else 
    {
      elementType = instruction->getType();
    }

  /* 3D context array. In case the elementType itself is an array or struct,
   * we must take into account it could be alloca-ed with alignment and loads
   * or stores might use vectorized instructions expecting proper alignment.
   * Because of that, we cannot simply allocate x*y*z*(size), we must
   * enlarge the type to fit the alignment. */
  Type *AllocType = elementType;
  AllocaInst *InstCast = dyn_cast<AllocaInst>(instruction);
  if (InstCast) {
#ifdef LLVM_OLDER_THAN_15_0
    unsigned Alignment = InstCast->getAlignment();
#else
    unsigned Alignment = InstCast->getAlign().value();
#endif
    uint64_t StoreSize =
        Layout.getTypeStoreSize(InstCast->getAllocatedType());

    if ((Alignment > 1) && (StoreSize & (Alignment - 1))) {
      uint64_t AlignedSize = (StoreSize & (~(Alignment - 1))) + Alignment;
#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "### unaligned type found: aligning " << StoreSize << " to "
                << AlignedSize << "\n";
#endif
      assert(AlignedSize > StoreSize);
      uint64_t RequiredExtraBytes = AlignedSize - StoreSize;

      if (isa<ArrayType>(elementType)) {

        ArrayType *StructPadding = ArrayType::get(
            Type::getInt8Ty(M->getContext()), RequiredExtraBytes);

        std::vector<Type *> PaddedStructElements;
        PaddedStructElements.push_back(elementType);
        PaddedStructElements.push_back(StructPadding);
        const ArrayRef<Type *> NewStructElements(PaddedStructElements);
        AllocType = StructType::get(M->getContext(), NewStructElements, true);
        PoclWrapperStructAdded = true;
        uint64_t NewStoreSize = Layout.getTypeStoreSize(AllocType);
        assert(NewStoreSize == AlignedSize);

      } else if (isa<StructType>(elementType)) {
        StructType *OldStruct = dyn_cast<StructType>(elementType);

        ArrayType *StructPadding =
            ArrayType::get(Type::getInt8Ty(M->getContext()), RequiredExtraBytes);
        std::vector<Type *> PaddedStructElements;
        for (unsigned j = 0; j < OldStruct->getNumElements(); j++)
          PaddedStructElements.push_back(OldStruct->getElementType(j));
        PaddedStructElements.push_back(StructPadding);
        const ArrayRef<Type *> NewStructElements(PaddedStructElements);
        AllocType = StructType::get(OldStruct->getContext(), NewStructElements,
                                    OldStruct->isPacked());
        uint64_t NewStoreSize = Layout.getTypeStoreSize(AllocType);
        assert(NewStoreSize == AlignedSize);
      }
    }
  }

  llvm::AllocaInst *Alloca = nullptr;
  if (WGDynamicLocalSize)
    {
      char GlobalName[32];
      GlobalVariable* LocalSize;
      LoadInst* LocalSizeLoad[3];
      for (int i = 0; i < 3; ++i) {
        snprintf(GlobalName, 32, "_local_size_%c", 'x' + i);
        LocalSize =
          cast<GlobalVariable>(M->getOrInsertGlobal(GlobalName, SizeT));
        LocalSizeLoad[i] = builder.CreateLoad(SizeT, LocalSize);
      }

      Value* LocalXTimesY =
        builder.CreateBinOp(Instruction::Mul, LocalSizeLoad[0],
                            LocalSizeLoad[1], "tmp");
      Value* NumberOfWorkItems =
        builder.CreateBinOp(Instruction::Mul, LocalXTimesY,
                            LocalSizeLoad[2], "num_wi");

      Alloca = builder.CreateAlloca(AllocType, NumberOfWorkItems, varName);
    }
  else
    {
      llvm::Type *contextArrayType = ArrayType::get(
          ArrayType::get(ArrayType::get(AllocType, WGLocalSizeX), WGLocalSizeY),
          WGLocalSizeZ);

      /* Allocate the context data array for the variable. */
      Alloca = builder.CreateAlloca(contextArrayType, nullptr, varName);
    }

  /* Align the context arrays to stack to enable wide vectors
     accesses to them. Also, LLVM 3.3 seems to produce illegal
     code at least with Core i5 when aligned only at the element
     size. */
#ifndef LLVM_OLDER_THAN_11_0
  Alloca->setAlignment(llvm::Align(CONTEXT_ARRAY_ALIGN));
#else
  Alloca->setAlignment(llvm::MaybeAlign(CONTEXT_ARRAY_ALIGN));
#endif

    if (DebugVal && DebugCall && !WGDynamicLocalSize) {

      llvm::SmallVector<llvm::Metadata *, 4> Subscripts;
      Subscripts.push_back(DB->getOrCreateSubrange(0, WGLocalSizeZ));
      Subscripts.push_back(DB->getOrCreateSubrange(0, WGLocalSizeY));
      Subscripts.push_back(DB->getOrCreateSubrange(0, WGLocalSizeX));
      llvm::DINodeArray SubscriptArray = DB->getOrCreateArray(Subscripts);

      size_t sizeBits;
      sizeBits = Alloca
                     ->getAllocationSizeInBits(M->getDataLayout())
#if !defined(LLVM_OLDER_THAN_15_0)
                     .value_or(TypeSize(0, false))
                     .getFixedValue();
#elif !defined(LLVM_OLDER_THAN_12_0)
                     .getValueOr(TypeSize(0, false))
                     .getFixedValue();
#else
                     .getValueOr(0);
#endif

      assert(sizeBits != 0);

      // if (size == 0) WGLocalSizeX * WGLocalSizeY * WGLocalSizeZ * 8 *
      // Alloca->getAllocatedType()->getScalarSizeInBits();
#ifdef LLVM_OLDER_THAN_15_0
      size_t alignBits = Alloca->getAlignment() * 8;
#else
      size_t alignBits = Alloca->getAlign().value() * 8;
#endif

      Metadata *VariableDebugMeta =
          cast<MetadataAsValue>(DebugCall->getOperand(1))->getMetadata();
#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "### VariableDebugMeta :  ";
      VariableDebugMeta->dump();
      std::cerr << "### sizeBits :  " << sizeBits
                << "  alignBits: " << alignBits << "\n";
#endif

      DILocalVariable *LocalVar = dyn_cast<DILocalVariable>(VariableDebugMeta);
      assert(LocalVar);
      if (LocalVar) {

        DICompositeType *CT = DB->createArrayType(
            sizeBits, alignBits, LocalVar->getType(), SubscriptArray);

#ifdef DEBUG_WORK_ITEM_LOOPS
        std::cerr << "### DICompositeType:\n";
        CT->dump();
#endif
        DILocalVariable *NewLocalVar = DB->createAutoVariable(
            LocalVar->getScope(), LocalVar->getName(), LocalVar->getFile(),
            LocalVar->getLine(), CT, false, LocalVar->getFlags());

        Metadata *NewMeta = ValueAsMetadata::get(Alloca);
        DebugCall->setOperand(0,
                              MetadataAsValue::get(M->getContext(), NewMeta));

        MetadataAsValue *NewLV =
            MetadataAsValue::get(M->getContext(), NewLocalVar);
        DebugCall->setOperand(1, NewLV);

        DebugCall->removeFromParent();
        DebugCall->insertAfter(Alloca);
      }
    }

    contextArrays[varName] = Alloca;
    return Alloca;
}

// Adds context save/restore code for the value produced by the
// given instruction.
//
// TODO: add only one restore per variable per region.
// TODO: add only one load of the id variables per region.
// Could be done by having a context restore BB in the beginning of the
// region and a context save BB at the end.
// TODO: ignore work group variables completely (the iteration variables)
// The LLVM should optimize these away but it would improve
// the readability of the output during debugging.
// TODO: rematerialize some values such as extended values of global
// variables (especially global id which is computed from local id) or kernel
// argument values instead of allocating stack space for them
void WorkitemLoops::addContextSaveRestore(llvm::Instruction *Instr) {

  //

  // Allocate the context data array for the variable.
  bool PoclWrapperStructAdded = false;
  llvm::AllocaInst *Alloca = getContextArray(Instr, PoclWrapperStructAdded);
  llvm::Instruction *TheStore = AddContextSave(Instr, Alloca);

  InstructionVec Uses;
  // Restore the produced variable before each use to ensure the correct
  // context copy is used.

  // We could add the restore only to other regions outside the variable
  // defining region and use the original variable in the defining region due
  // to the SSA virtual registers being unique. However, alloca variables can
  // be redefined also in the same region, thus we need to ensure the correct
  // alloca context position is written, not the original unreplicated one.
  // These variables can be generated by volatile variables, private arrays,
  // and due to the PHIs to allocas pass.

  // Find out the uses to fix first as fixing them invalidates the iterator.
  for (Instruction::use_iterator UI = Instr->use_begin(),
         UE = Instr->use_end(); UI != UE; ++UI) {
    llvm::Instruction *User = cast<Instruction>(UI->getUser());
    if (User == NULL || User == TheStore) continue;
    Uses.push_back(User);
  }

  for (InstructionVec::iterator I = Uses.begin(); I != Uses.end(); ++I) {
    Instruction *UserI = *I;
    Instruction *ContextRestoreLocation = UserI;
    // If the user is in a block that doesn't belong to a region, the variable
    // itself must be a "work group variable", that is, not dependent on the
    // work item. Most likely an iteration variable of a for loop with a
    // barrier.
    if (RegionOfBlock(UserI->getParent()) == NULL) continue;

    PHINode* Phi = dyn_cast<PHINode>(UserI);
    if (Phi != NULL) {
      // In case of PHI nodes, we cannot just insert the context restore code
      // before it in the same basic block because it is assumed there are no
      // non-phi Instructions before PHIs which the context restore code
      // constitutes to. Add the context restore to the incomingBB instead.

      // There can be values in the PHINode that are incoming from another
      // region even though the decision BB is within the region. For those
      // values we need to add the context restore code in the incoming BB
      // (which is known to be inside the region due to the assumption of not
      // having to touch PHI nodes in PRentry BBs).

      // PHINodes at region entries are broken down earlier.
      assert ("Cannot add context restore for a PHI node at the region entry!"
              && RegionOfBlock(
                Phi->getParent())->entryBB() != Phi->getParent());
#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "### adding context restore code before PHI" << std::endl;
      UserI->dump();
      std::cerr << "### in BB:" << std::endl;
      UserI->getParent()->dump();
#endif
      BasicBlock *IncomingBB = NULL;
      for (unsigned Incoming = 0; Incoming < Phi->getNumIncomingValues();
           ++Incoming) {
        Value *Val = Phi->getIncomingValue(Incoming);
        BasicBlock *BB = Phi->getIncomingBlock(Incoming);
        if (Val == Instr) IncomingBB = BB;
      }
      assert(IncomingBB != NULL);
      ContextRestoreLocation = IncomingBB->getTerminator();
    }
    llvm::Value *LoadedValue = AddContextRestore(
      UserI, Alloca, Instr->getType(),
      PoclWrapperStructAdded, ContextRestoreLocation,
      isa<AllocaInst>(Instr));
    UserI->replaceUsesOfWith(Instr, LoadedValue);

#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### done, the user was converted to:" << std::endl;
    UserI->dump();
#endif
  }
}

bool WorkitemLoops::shouldNotBeContextSaved(llvm::Instruction *Instr) {

  if (isa<BranchInst>(Instr)) return true;

  // The local memory allocation call is uniform, the same pointer to the
  // work-group shared memory area is returned to all work-items. It must
  // not be replicated.
  if (isa<CallInst>(Instr) &&
      cast<CallInst>(Instr)->getCalledFunction() == LocalMemAllocaFuncDecl)
    return true;

  // _local_id loads should not be replicated as it leads to/ problems in
  // conditional branch case where the header node of the region is shared
  // across the branches and thus the header node's ID loads might get context
  // saved which leads to egg-chicken problems.

  llvm::LoadInst *Load = dyn_cast<llvm::LoadInst>(Instr);
  if (Load != NULL &&
      (Load->getPointerOperand() == LocalIdZGlobal ||
       Load->getPointerOperand() == LocalIdYGlobal ||
       Load->getPointerOperand() == LocalIdXGlobal))
    return true;

  VariableUniformityAnalysis &VUA =
    getAnalysis<VariableUniformityAnalysis>();

  // In case of uniform variables (same value for all work-items), there is no
  // point to create a context array slot for them, but just use the original
  // value everywhere.

  // Allocas are problematic since they include the de-phi induction variables
  // of the b-loops. In those case each work item has a separate loop iteration
  // variable in LLVM IR but which is really a parallel region loop invariant.
  // But because we cannot separate such loop invariant variables at this point
  // sensibly, let's just replicate the iteration variable to each work item
  // and hope the latter optimizations reduce them back to a single induction
  // variable outside the parallel loop.
  if (!VUA.shouldBePrivatized(Instr->getParent()->getParent(), Instr)) {
#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### based on VUA, not context saving:";
    Instr->dump();
#endif
    return true;
  }

  return false;
}

llvm::BasicBlock *
WorkitemLoops::AppendIncBlock
(llvm::BasicBlock* after, llvm::Value *localIdVar)
{
  llvm::LLVMContext &C = after->getContext();

  llvm::BasicBlock *oldExit = after->getTerminator()->getSuccessor(0);
  assert (oldExit != NULL);

  llvm::BasicBlock *forIncBB =
    BasicBlock::Create(C, "pregion_for_inc", after->getParent());

  after->getTerminator()->replaceUsesOfWith(oldExit, forIncBB);

  IRBuilder<> builder(oldExit);

  builder.SetInsertPoint(forIncBB);
  /* Create the iteration variable increment */
  builder.CreateStore(builder.CreateAdd(
                        builder.CreateLoad(SizeT, localIdVar),
                        ConstantInt::get(SizeT, 1)),
                      localIdVar);

  builder.CreateBr(oldExit);

  return forIncBB;
}

llvm::Instruction *WorkitemLoops::getWorkGroupSizeInstr(llvm::Function &F) {

  if (WGSizeInstr != nullptr)
      return WGSizeInstr;

  IRBuilder<> Builder(F.getEntryBlock().getTerminator());

  llvm::Module *M = F.getParent();
  GlobalVariable *GV = M->getGlobalVariable("_local_size_x");
  if (GV != NULL) {
    WGSizeInstr = Builder.CreateLoad(SizeT, GV);
  }

  GV = M->getGlobalVariable("_local_size_y");
  if (GV != NULL) {
    WGSizeInstr =
      cast<llvm::Instruction>(
        Builder.CreateBinOp(
          Instruction::Mul, Builder.CreateLoad(SizeT, GV), WGSizeInstr));
  }

  GV = M->getGlobalVariable("_local_size_z");
  if (GV != NULL) {
    WGSizeInstr =
      cast<llvm::Instruction>(
        Builder.CreateBinOp(
          Instruction::Mul, Builder.CreateLoad(SizeT, GV),
          WGSizeInstr));
  }

  return WGSizeInstr;
}
