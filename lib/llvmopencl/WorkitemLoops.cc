// LLVM function pass to create a loop that runs all the work items 
// in a work group.
// 
// Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
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

#define DEBUG_TYPE "workitem-loops"

#include "WorkitemLoops.h"
#include "Workgroup.h"
#include "Barrier.h"
#include "Kernel.h"
#include "config.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CommandLine.h"
#ifdef LLVM_3_1
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/ValueSymbolTable.h"
#elif defined LLVM_3_2
#include "llvm/IRBuilder.h"
#include "llvm/TypeBuilder.h"
#include "llvm/DataLayout.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/ValueSymbolTable.h"
#else
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueSymbolTable.h"
#endif
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "WorkitemHandlerChooser.h"

#include <iostream>
#include <map>
#include <sstream>
#include <vector>

//#define DUMP_RESULT_CFG

#ifdef DUMP_RESULT_CFG
#include "llvm/Analysis/CFGPrinter.h"
#endif

//#define DEBUG_WORK_ITEM_LOOPS

using namespace llvm;
using namespace pocl;

namespace {
  static
  RegisterPass<WorkitemLoops> X("workitemloops", 
                                "Workitem loop generation pass");
}

char WorkitemLoops::ID = 0;

void
WorkitemLoops::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.addRequired<DominatorTree>();
  AU.addRequired<PostDominatorTree>();
  AU.addRequired<LoopInfo>();
#ifdef LLVM_3_1
  AU.addRequired<TargetData>();
#else
  AU.addRequired<DataLayout>();
#endif
  AU.addRequired<pocl::WorkitemHandlerChooser>();
}

bool
WorkitemLoops::runOnFunction(Function &F)
{
  if (!Workgroup::isKernelToProcess(F))
    return false;

  if (getAnalysis<pocl::WorkitemHandlerChooser>().chosenHandler() != 
      pocl::WorkitemHandlerChooser::POCL_WIH_LOOPS)
    return false;

  DT = &getAnalysis<DominatorTree>();
  LI = &getAnalysis<LoopInfo>();
  PDT = &getAnalysis<PostDominatorTree>();

  tempInstructionIndex = 0;

  llvm::Module *M = F.getParent();

#if 0
  std::cerr << "### original:" << std::endl;
  F.viewCFG();
#endif

  bool changed = ProcessFunction(F);
#ifdef DUMP_RESULT_CFG
  FunctionPass* cfgPrinter = createCFGOnlyPrinterPass();
  cfgPrinter->runOnFunction(F);
#endif  

#if 0
  std::cerr << "### after:" << std::endl;
  F.viewCFG();
#endif

  changed |= fixUndominatedVariableUses(DT, F);

#ifdef DEBUG_WORK_ITEM_LOOPS
  //std::cerr << "### processed:" << std::endl;
  //F.viewCFGOnly();
  TargetData &TD = getAnalysis<TargetData>();

  size_t totalContext = 0;
  for (StrInstructionMap::iterator i = contextArrays.begin();
       i != contextArrays.end(); ++i)
  {
      llvm::AllocaInst *a = dyn_cast<llvm::AllocaInst>((*i).second);
      llvm::Value *s = a->getArraySize();
      assert (a != NULL);
      assert (isa<ConstantInt>(s));
      llvm::ConstantInt *size = dyn_cast<llvm::ConstantInt>(s);
      totalContext += size->getZExtValue() * TD.getTypeAllocSize(a->getAllocatedType());
  }
  std::cerr << "### total context data needed " << totalContext << " bytes" 
            << std::endl;
#endif

#if 0
  /* Split large BBs so we can print the Dot without it crashing. */
  bool fchanged = false;
  const int MAX_INSTRUCTIONS_PER_BB = 70;
  do {
    fchanged = false;
    for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
      BasicBlock *b = i;
      
      if (b->size() > MAX_INSTRUCTIONS_PER_BB + 1)
        {
          int count = 0;
          BasicBlock::iterator splitPoint = b->begin();
          while (count < MAX_INSTRUCTIONS_PER_BB || isa<PHINode>(splitPoint))
            {
              ++splitPoint;
              ++count;
            }
          SplitBlock(b, splitPoint, this);
          fchanged = true;
          break;
        }
    }  

  } while (fchanged);

  //F.viewCFG();
#endif

  // this breaks cutcp/Parboil
  //changed |= pocl::WorkitemHandler::runOnFunction(F);
  return changed;
}

std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
WorkitemLoops::CreateLoopAround
(llvm::BasicBlock *entryBB, llvm::BasicBlock *exitBB, 
 bool peeledFirst, llvm::Value *localIdVar, size_t LocalSizeForDim,
 bool addIncBlock) 
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
  loopBodyEntryBB->setName("pregion.for.body");

  assert (exitBB->getTerminator()->getNumSuccessors() == 1);

  llvm::BasicBlock *oldExit = exitBB->getTerminator()->getSuccessor(0);

  llvm::BasicBlock *forInitBB = 
    BasicBlock::Create(C, "pregion.for.init", F, loopBodyEntryBB);

  llvm::BasicBlock *loopEndBB = 
    BasicBlock::Create(C, "pregion.for.end", F, exitBB);

  llvm::BasicBlock *forCondBB = 
    BasicBlock::Create(C, "pregion.for.cond", F, exitBB);

  DT->runOnFunction(*F);

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

  if (peeledFirst)
    {
      builder.CreateStore(builder.CreateLoad(localIdXFirstVar), localIdVar);
      builder.CreateStore
        (ConstantInt::get(IntegerType::get(C, size_t_width), 0), localIdXFirstVar);
    }
  else
    {
      builder.CreateStore
        (ConstantInt::get(IntegerType::get(C, size_t_width), 0), localIdVar);
    }

  builder.CreateBr(loopBodyEntryBB);

  exitBB->getTerminator()->replaceUsesOfWith(oldExit, forCondBB);
  if (addIncBlock)
    {
      AppendIncBlock(exitBB, localIdVar);
    }

  builder.SetInsertPoint(forCondBB);
  llvm::Value *cmpResult = 
    builder.CreateICmpULT
    (builder.CreateLoad(localIdVar),
     (ConstantInt::get
      (IntegerType::get(C, size_t_width), 
       LocalSizeForDim)));
      
  Instruction *loopBranch =
      builder.CreateCondBr(cmpResult, loopBodyEntryBB, loopEndBB);

  /* Add the metadata to mark a parallel loop. */
  loopBranch->setMetadata
      ("parallel_loop", 
       MDNode::get(C, ConstantInt::get(Type::getInt32Ty(C), 1)));

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

bool
WorkitemLoops::ProcessFunction(Function &F)
{
  Kernel *K = cast<Kernel> (&F);
  Initialize(K);
  unsigned workItemCount = LocalSizeX*LocalSizeY*LocalSizeZ;

  if (workItemCount == 1)
    {
      K->addLocalSizeInitCode(LocalSizeX, LocalSizeY, LocalSizeZ);
      ParallelRegion::insertLocalIdInit(&F.getEntryBlock(), 0, 0, 0);
      return true;
    }

  original_parallel_regions =
    K->getParallelRegions(LI);

  IRBuilder<> builder(F.getEntryBlock().getFirstInsertionPt());
  localIdXFirstVar = 
    builder.CreateAlloca
    (IntegerType::get(F.getContext(), size_t_width), 0, ".pocl.local_id_x_init");

  //  F.viewCFGOnly();

#if 0
  std::cerr << "### Original" << std::endl;
  F.viewCFG();
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
    FixMultiRegionVariables(region);
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
       (with the loop). We know it's such a region in case
       the region exit does not post dominate the entry.
       That is, there will be two branches from the same
       entry. */
    bool peelFirst = !PDT->dominates(original->exitBB(), original->entryBB());
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

        int unrollCount;
        if (getenv("POCL_WILOOPS_MAX_UNROLL_COUNT") != NULL)
            unrollCount = atoi(getenv("POCL_WILOOPS_MAX_UNROLL_COUNT"));
        else
            unrollCount = 1;
        /* Find a two's exponent unroll count, if available. */
        while (unrollCount >= 1)
          {
            if (LocalSizeX % unrollCount == 0 &&
                unrollCount <= LocalSizeX)
              {
                break;
              }
            unrollCount /= 2;
          }

        ParallelRegion *prev = original;
        llvm::BasicBlock *lastBB = 
          AppendIncBlock(original->exitBB(), localIdX);
        original->AddBlockAfter(lastBB, original->exitBB());
        original->SetExitBB(lastBB);

        if (AddWIMetadata)
          original->AddIDMetadata(F.getContext(), 0);

        for (int c = 1; c < unrollCount; ++c) 
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
      }

    if (LocalSizeX > 1)
      l = CreateLoopAround(l.first, l.second, peelFirst, localIdX, LocalSizeX, !unrolled);

    if (LocalSizeY > 1)
      l = CreateLoopAround(l.first, l.second, false, localIdY, LocalSizeY);

    if (LocalSizeZ > 1)
      l = CreateLoopAround(l.first, l.second, false, localIdZ, LocalSizeZ);

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

    //F.viewCFGOnly();
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
    builder.SetInsertPoint(pr->entryBB()->getFirstInsertionPt());
    builder.CreateStore
      (ConstantInt::get(IntegerType::get(F.getContext(), size_t_width), 1), 
       localIdXFirstVar);       
  }

  K->addLocalSizeInitCode(LocalSizeX, LocalSizeY, LocalSizeZ);
  ParallelRegion::insertLocalIdInit(&F.getEntryBlock(), 0, 0, 0);

#if 0
  F.viewCFG();
#endif

  return true;
}

/*
 * Add context save/restore code to variables that are defined in 
 * the given region and are used outside the region.
 *
 * Each such variable gets a slot in the stack frame. The variable
 * is restored from the stack whenever its used.
 *
 */
void
WorkitemLoops::FixMultiRegionVariables(ParallelRegion *region)
{
  InstructionIndex instructionsInRegion;
  InstructionVec instructionsToFix;

  /* Construct an index of the region's instructions so it's
     fast to figure out if the variable uses are all
     in the region. */
  for (BasicBlockVector::iterator i = region->begin();
       i != region->end(); ++i)
    {
      llvm::BasicBlock *bb = *i;
      for (llvm::BasicBlock::iterator instr = bb->begin();
           instr != bb->end(); ++instr) 
        {
          llvm::Instruction *instruction = instr;
          instructionsInRegion.insert(instruction);
        }
    }

  /* Find all the instructions that define new values and
     check if they need to be context saved. */
  for (BasicBlockVector::iterator i = region->begin();
       i != region->end(); ++i)
    {
      llvm::BasicBlock *bb = *i;
      for (llvm::BasicBlock::iterator instr = bb->begin();
           instr != bb->end(); ++instr) 
        {
          llvm::Instruction *instruction = instr;
          if (!ShouldBeContextSaved(instruction)) 
            {
#ifdef DEBUG_WORK_ITEM_LOOPS
              std::cerr << "### can be rematerialized, not context saving:" << std::endl;
              instruction->dump();
#endif
              continue;
            }

          for (Instruction::use_iterator ui = instruction->use_begin(),
                 ue = instruction->use_end();
               ui != ue; ++ui) 
            {
              Instruction *user;
              if ((user = dyn_cast<Instruction> (*ui)) == NULL) continue;
              // if the instruction is used outside this region inside another
              // region (not in a regionless BB like the B-loop construct BBs),
              // need to context save it.
              if (instructionsInRegion.find(user) == instructionsInRegion.end() &&
                  RegionOfBlock(user->getParent()) != NULL)
                {
#ifdef DEBUG_WORK_ITEM_LOOPS
                  std::cerr << "### instr used in another region" << std::endl;
                  instr->dump();
                  ParallelRegion *region = RegionOfBlock(user->getParent());
                  if (region != NULL) region->dumpNames();
#endif                  
                  instructionsToFix.push_back(instruction);
                  break;
                }
            }
        }
    }  

  /* Finally, fix the instructions. */
  for (InstructionVec::iterator i = instructionsToFix.begin();
       i != instructionsToFix.end(); ++i)
    {
#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "### adding context/save restore for" << std::endl;
      (*i)->dump();
#endif 
      llvm::Instruction *instructionToFix = *i;
      AddContextSaveRestore(instructionToFix, instructionsInRegion);
    }
}

llvm::Instruction *
WorkitemLoops::AddContextSave
(llvm::Instruction *instruction, llvm::Instruction *alloca)
{
  /* Save the produced variable to the array. */
  BasicBlock::iterator definition = dyn_cast<Instruction>(instruction);

  ++definition;
  while (isa<PHINode>(definition)) ++definition;

  IRBuilder<> builder(definition); 
  std::vector<llvm::Value *> gepArgs;
  gepArgs.push_back(ConstantInt::get(IntegerType::get(instruction->getContext(), size_t_width), 0));

  /* Reuse the id loads earlier in the region, if possible, to
     avoid messy output with lots of redundant loads. */
  ParallelRegion *region = RegionOfBlock(instruction->getParent());
  assert ("Adding context save outside any region produces illegal code." && 
          region != NULL);

  gepArgs.push_back(region->LocalIDZLoad());
  gepArgs.push_back(region->LocalIDYLoad());
  gepArgs.push_back(region->LocalIDXLoad());

  return builder.CreateStore(instruction, builder.CreateGEP(alloca, gepArgs));
}

llvm::Instruction *
WorkitemLoops::AddContextRestore
(llvm::Value *val, llvm::Instruction *alloca, llvm::Instruction *before)
{
  assert (val != NULL);
  IRBuilder<> builder(alloca);
  if (before != NULL) 
    {
      builder.SetInsertPoint(before);
    }
  else if (isa<Instruction>(val))
    {
      builder.SetInsertPoint(dyn_cast<Instruction>(val));
      before = dyn_cast<Instruction>(val);
    }
  else 
    {
      assert (false && "Unknown context restore location!");
    }

  
  std::vector<llvm::Value *> gepArgs;
  gepArgs.push_back(ConstantInt::get(IntegerType::get(val->getContext(), size_t_width), 0));

  /* Reuse the id loads earlier in the region, if possible, to
     avoid messy output with lots of redundant loads. */
  ParallelRegion *region = RegionOfBlock(before->getParent());
  assert ("Adding context save outside any region produces illegal code." && 
          region != NULL);

  gepArgs.push_back(region->LocalIDZLoad());
  gepArgs.push_back(region->LocalIDYLoad());
  gepArgs.push_back(region->LocalIDXLoad());
          
  return builder.CreateLoad(builder.CreateGEP(alloca, gepArgs));
}

/**
 * Returns the context array (alloca) for the given Value, creates it if not
 * found.
 */
llvm::Instruction *
WorkitemLoops::GetContextArray(llvm::Instruction *instruction)
{
  
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

  IRBuilder<> builder(instruction->getParent()->getParent()->getEntryBlock().getFirstInsertionPt());

  llvm::Type *scalart = instruction->getType();
  llvm::Type *contextArrayType = 
    ArrayType::get(ArrayType::get(ArrayType::get(scalart, LocalSizeX), LocalSizeY), LocalSizeZ);

  /* Allocate the context data array for the variable. */
  llvm::Instruction *alloca = 
    builder.CreateAlloca(contextArrayType, 0, varName);

  contextArrays[varName] = alloca;
  return alloca;
}


/**
 * Adds context save/restore code for the value produced by the
 * given instruction.
 *
 * TODO: add only one restore per variable per region.
 * TODO: add only one load of the id variables per region. 
 * Could be done by having a context restore BB in the beginning of the
 * region and a context save BB at the end.
 * TODO: ignore work group variables completely (the iteration variables)
 * The LLVM should optimize these away but it would improve
 * the readability of the output during debugging.
 * TODO: rematerialize some values such as extended values of global 
 * variables or kernel argument values instead of allocating stack
 * space for them
 */
void
WorkitemLoops::AddContextSaveRestore
(llvm::Instruction *instruction, const InstructionIndex& instructionsInRegion)
{
  /* Allocate the context data array for the variable. */
  llvm::Instruction *alloca = GetContextArray(instruction);
  llvm::Instruction *theStore = AddContextSave(instruction, alloca);

  InstructionVec uses;
  /* Restore the produced variable before each use outside the region. */

  /* Find out the uses to fix first as fixing them invalidates
     the iterator. */
  for (Instruction::use_iterator ui = instruction->use_begin(),
         ue = instruction->use_end();
       ui != ue; ++ui) 
    {
      Instruction *user;
      if ((user = dyn_cast<Instruction> (*ui)) == NULL) continue;
      if (user == theStore) continue;
      if (instructionsInRegion.find(user) != instructionsInRegion.end()) continue;
      uses.push_back(user);
    }

  for (InstructionVec::iterator i = uses.begin(); i != uses.end(); ++i)
    {
      Instruction *user = *i;
      Instruction *contextRestoreLocation = user;
      /* If the user is in a block that doesn't belong to a region,
         the variable itself must be a "work group variable", that is,
         not dependent on the work item. Most likely an iteration
         variable of a for loop with a barrier. */
      if (RegionOfBlock(user->getParent()) == NULL) continue;

      PHINode* phi = dyn_cast<PHINode>(user);
      if (phi != NULL)
        {
          /* In case of PHI nodes, we cannot just insert the context 
             restore code before it in the same basic block because it is 
             assumed there are no non-phi Instructions before PHIs which 
             the context restore code constitutes to. Add the context
             restore to the incomingBB instead.

             There can be values in the PHINode that are incoming
             from another region even though the decision BB is within the region. 
             For those values we need to add the context restore code in the 
             incoming BB (which is known to be inside the region due to the
             assumption of not having to touch PHI nodes in PRentry BBs).
          */          

          /* PHINodes at region entries are broken down earlier. */
          assert ("Cannot add context restore for a PHI node at the region entry!" &&
                  RegionOfBlock(phi->getParent())->entryBB() != phi->getParent());
#ifdef DEBUG_WORK_ITEM_LOOPS
          std::cerr << "### adding context restore code before PHI" << std::endl;
          user->dump();
          std::cerr << "### in BB:" << std::endl;
          user->getParent()->dump();
#endif
          BasicBlock *incomingBB = NULL;
          for (unsigned incoming = 0; incoming < phi->getNumIncomingValues(); 
               ++incoming)
            {
              Value *val = phi->getIncomingValue(incoming);
              BasicBlock *bb = phi->getIncomingBlock(incoming);
              if (val == instruction) incomingBB = bb;
            }
          assert (incomingBB != NULL);
          contextRestoreLocation = incomingBB->getTerminator();
        }
      llvm::Value *loadedValue = AddContextRestore(user, alloca, contextRestoreLocation);
      user->replaceUsesOfWith(instruction, loadedValue);
#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "### done, the user was converted to:" << std::endl;
      user->dump();
#endif
    }
}

bool
WorkitemLoops::ShouldBeContextSaved(llvm::Instruction *instr)
{
    /* TODO: rematerialization:

       If the instruction is a load from a scalar global, we need
       not context save it but should just (re)load. This
       covers loads from the temporary _local_id_x etc. variables. 

       Also rematerialize "cheap" computations.
    */

    /*
      Loads from non-indexed scalar global addresses are known not to be 
      ID-dependent.

      Especially _local_id_x loads should not be replicated as it leads to
      problems in conditional branch case where the header node
      of the region is shared across the branches and thus the
      header node's ID loads might get context saved which leads
      to egg-chicken problems. 
    */
    llvm::LoadInst *load = dyn_cast<llvm::LoadInst>(instr);
    if (load != NULL &&
        (load->getPointerOperand() == localIdZ ||
         load->getPointerOperand() == localIdY ||
         load->getPointerOperand() == localIdX))
      return false;
    return true;
}

llvm::BasicBlock *
WorkitemLoops::AppendIncBlock
(llvm::BasicBlock* after, llvm::Value *localIdVar)
{
  llvm::LLVMContext &C = after->getContext();

  llvm::BasicBlock *oldExit = after->getTerminator()->getSuccessor(0);
  assert (oldExit != NULL);

  llvm::BasicBlock *forIncBB = 
    BasicBlock::Create(C, "pregion.for.inc", after->getParent());

  after->getTerminator()->replaceUsesOfWith(oldExit, forIncBB);

  IRBuilder<> builder(oldExit);

  builder.SetInsertPoint(forIncBB);
  /* Create the iteration variable increment */
  builder.CreateStore
    (builder.CreateAdd
     (builder.CreateLoad(localIdVar),
      ConstantInt::get(IntegerType::get(C, size_t_width), 1)),
     localIdVar);

  builder.CreateBr(oldExit);

  return forIncBB;
}
