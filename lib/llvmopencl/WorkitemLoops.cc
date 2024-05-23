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


#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
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

#include "Barrier.h"
#include "DebugHelpers.h"
#include "Kernel.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
#include "WorkitemLoops.h"
POP_COMPILER_DIAGS

#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#define DEBUG_TYPE "workitem-loops"
//#define DUMP_CFGS
//#define DEBUG_WORK_ITEM_LOOPS

// this must be at least the alignment of largest OpenCL type (= 128 bytes)
#define CONTEXT_ARRAY_ALIGN MAX_EXTENDED_ALIGNMENT

#define PASS_NAME "workitemloops"
#define PASS_CLASS pocl::WorkitemLoops
#define PASS_DESC "Workitem loop generation pass"

namespace pocl {

using namespace llvm;

// Magic function used to allocate "local memory" dynamically. Used with SG/WG
// shuffles as temporary storage.
static const char *POCL_LOCAL_MEM_ALLOCA_FUNC_NAME = "__pocl_local_mem_alloca";

// Another, which multiplies the size by the number of WIs in the WG.
static const char *POCL_WORK_GROUP_ALLOCA_FUNC_NAME =
    "__pocl_work_group_alloca";

class WorkitemLoopsImpl : public pocl::WorkitemHandler {
public:
  WorkitemLoopsImpl(llvm::DominatorTree &DT, llvm::LoopInfo &LI,
                    llvm::PostDominatorTree &PDT,
                    VariableUniformityAnalysisResult &VUA)
      : WorkitemHandler(), DT(DT), LI(LI), PDT(PDT), VUA(VUA) {}
  virtual bool runOnFunction(llvm::Function &F);

private:
  using BasicBlockVector = std::vector<llvm::BasicBlock *>;
  using InstructionIndex = std::set<llvm::Instruction *>;
  using InstructionVec = std::vector<llvm::Instruction *>;
  using StrInstructionMap = std::map<std::string, llvm::AllocaInst *>;

  llvm::DominatorTree &DT;
  llvm::LoopInfo &LI;
  llvm::PostDominatorTree &PDT;
  VariableUniformityAnalysisResult &VUA;

  ParallelRegion::ParallelRegionVector OriginalParallelRegions;

  StrInstructionMap ContextArrays;

  // Points to the __pocl_local_mem_alloca pseudo function declaration, if
  // it's been referred to in the processed module.
  llvm::Function *LocalMemAllocaFuncDecl;

  // Points to the __pocl_work_group_alloca pseudo function declaration, if
  // it's been referred to in the processed module.
  llvm::Function *WorkGroupAllocaFuncDecl;

  // Points to the work-group size computation instruction in the entry
  // block of the currently handled function.
  llvm::Instruction *WGSizeInstr;

  bool processFunction(llvm::Function &F);

  void fixMultiRegionVariables(ParallelRegion *region);
  bool handleLocalMemAllocas(Kernel &K);
  void addContextSaveRestore(llvm::Instruction *instruction);
  void releaseParallelRegions();

  // Returns an instruction in the entry block which computes the
  // total size of work-items in the work-group. If it doesn't
  // exist, creates it to the end of the entry block.
  llvm::Instruction *getWorkGroupSizeInstr(llvm::Function &F);

  llvm::Value *getLinearWiIndex(llvm::IRBuilder<> &Builder, llvm::Module *M,
                                ParallelRegion *Region);
  llvm::Instruction *addContextSave(llvm::Instruction *Instruction,
                                    llvm::AllocaInst *AllocaI);
  llvm::Instruction *
  addContextRestore(llvm::Value *Val, llvm::AllocaInst *AllocaI,
                    llvm::Type *InstType, bool PoclWrapperStructAdded,
                    llvm::Instruction *Before = nullptr, bool isAlloca = false);
  llvm::AllocaInst *getContextArray(llvm::Instruction *Inst,
                                    bool &PoclWrapperStructAdded);

  std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
  createLoopAround(ParallelRegion &Region, llvm::BasicBlock *EntryBB,
                   llvm::BasicBlock *ExitBB, bool PeeledFirst,
                   llvm::Value *LocalIdVar, size_t LocalSizeForDim,
                   bool AddIncBlock = true,
                   llvm::Value *DynamicLocalSize = nullptr);

  llvm::BasicBlock *appendIncBlock(llvm::BasicBlock *After,
                                   llvm::Value *LocalIdVar);

  ParallelRegion *regionOfBlock(llvm::BasicBlock *BB);

  bool shouldNotBeContextSaved(llvm::Instruction *Instr);

  llvm::Type *recursivelyAlignArrayType(llvm::Type *ArrayType,
                                        llvm::Type *ElementType,
                                        size_t Alignment,
                                        const llvm::DataLayout &Layout);

  std::map<llvm::Instruction *, unsigned> TempInstructionIds;
  size_t TempInstructionIndex;
  // An alloca in the kernel which stores the first iteration to execute
  // in the inner (dimension 0) loop. This is set to 1 in an peeled iteration
  // to skip the 0, 0, 0 iteration in the loops.
  llvm::Value *LocalIdXFirstVar;
};

bool WorkitemLoopsImpl::runOnFunction(Function &F) {
  WGSizeInstr = nullptr;

  TempInstructionIndex = 0;

  LocalMemAllocaFuncDecl =
      F.getParent()->getFunction(POCL_LOCAL_MEM_ALLOCA_FUNC_NAME);

  WorkGroupAllocaFuncDecl =
      F.getParent()->getFunction(POCL_WORK_GROUP_ALLOCA_FUNC_NAME);

  bool Changed = processFunction(F);

  Changed |= handleLocalMemAllocas(cast<Kernel>(F));

#ifdef DUMP_CFGS
  dumpCFG(F, F.getName().str() + "_after_wiloops.dot", nullptr,
          &OriginalParallelRegions);
#endif

  Changed |= fixUndominatedVariableUses(DT, F);

#if 0
  /* Split large BBs so we can print the Dot without it crashing. */
  Changed |= chopBBs(F, *this);
  F.viewCFG();
#endif
  ContextArrays.clear();
  TempInstructionIds.clear();

  releaseParallelRegions();
  return Changed;
}

std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
WorkitemLoopsImpl::createLoopAround(ParallelRegion &Region,
                                    llvm::BasicBlock *EntryBB,
                                    llvm::BasicBlock *ExitBB,
                                    bool PeeledFirst,
                                    llvm::Value *LocalIdVar,
                                    size_t LocalSizeForDim,
                                    bool AddIncBlock,
                                    llvm::Value *DynamicLocalSize) {
  assert (LocalIdVar != NULL);

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

  llvm::BasicBlock *LoopBodyEntryBB = EntryBB;
  llvm::LLVMContext &C = LoopBodyEntryBB->getContext();
  llvm::Function *F = LoopBodyEntryBB->getParent();
  LoopBodyEntryBB->setName(std::string("pregion_for_entry.") + EntryBB->getName().str());

  assert (ExitBB->getTerminator()->getNumSuccessors() == 1);

  llvm::BasicBlock *oldExit = ExitBB->getTerminator()->getSuccessor(0);

  llvm::BasicBlock *forInitBB = 
    BasicBlock::Create(C, "pregion_for_init", F, LoopBodyEntryBB);

  llvm::BasicBlock *loopEndBB = 
    BasicBlock::Create(C, "pregion_for_end", F, ExitBB);

  llvm::BasicBlock *forCondBB = 
    BasicBlock::Create(C, "pregion_for_cond", F, ExitBB);

  DT.reset();
  DT.recalculate(*F);

  /* Collect the basic blocks in the parallel region that dominate the
     exit. These are used in determining whether load instructions may
     be executed unconditionally in the parallel loop (see below). */
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> dominatesExitBB;
  for (auto BB: Region) {
    if (DT.dominates(BB, ExitBB)) {
      dominatesExitBB.insert(BB);
    }
  }

  //  F->viewCFG();
  /* Fix the old edges jumping to the region to jump to the basic block
     that starts the created loop. Back edges should still point to the
     old basic block so we preserve the old loops. */
  BasicBlockVector preds;
  llvm::pred_iterator PI = 
    llvm::pred_begin(EntryBB),
    E = llvm::pred_end(EntryBB);

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
      if (DT.dominates(LoopBodyEntryBB, bb))
        continue;
      bb->getTerminator()->replaceUsesOfWith(LoopBodyEntryBB, forInitBB);
    }

  IRBuilder<> builder(forInitBB);

  if (PeeledFirst) {
      builder.CreateStore(builder.CreateLoad(SizeT, LocalIdXFirstVar), LocalIdVar);
      builder.CreateStore(ConstantInt::get(SizeT, 0), LocalIdXFirstVar);

    if (WGDynamicLocalSize) {
      llvm::Value *cmpResult;
      cmpResult = builder.CreateICmpULT(builder.CreateLoad(SizeT, LocalIdVar),
                                        builder.CreateLoad(SizeT, DynamicLocalSize));

      builder.CreateCondBr(cmpResult, LoopBodyEntryBB, loopEndBB);
    } else {
      builder.CreateBr(LoopBodyEntryBB);
    }
  } else {
    builder.CreateStore(ConstantInt::get(SizeT, 0), LocalIdVar);

    builder.CreateBr(LoopBodyEntryBB);
  }

  ExitBB->getTerminator()->replaceUsesOfWith(oldExit, forCondBB);
  if (AddIncBlock)
    {
    appendIncBlock(ExitBB, LocalIdVar);
    }

  builder.SetInsertPoint(forCondBB);

  llvm::Value *cmpResult;
  if (!WGDynamicLocalSize)
    cmpResult = builder.CreateICmpULT(
                  builder.CreateLoad(SizeT, LocalIdVar),
                    ConstantInt::get(SizeT, LocalSizeForDim));
  else
    cmpResult = builder.CreateICmpULT(
                  builder.CreateLoad(SizeT, LocalIdVar),
                    builder.CreateLoad(SizeT, DynamicLocalSize));
  
  Instruction *loopBranch =
      builder.CreateCondBr(cmpResult, LoopBodyEntryBB, loopEndBB);

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

  Region.AddParallelLoopMetadata(AccessGroupMD, IsLoadUnconditionallySafe);

  builder.SetInsertPoint(loopEndBB);
  builder.CreateBr(oldExit);

  return std::make_pair(forInitBB, loopEndBB);
}

ParallelRegion *WorkitemLoopsImpl::regionOfBlock(llvm::BasicBlock *BB) {
  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
           PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {
    ParallelRegion *region = (*PRI);
    if (region->HasBlock(BB)) return region;
  }
  return nullptr;
}

void WorkitemLoopsImpl::releaseParallelRegions() {
  for (auto PRI = OriginalParallelRegions.begin(),
            PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {
    ParallelRegion *P = *PRI;
    delete P;
  }
}

bool WorkitemLoopsImpl::processFunction(Function &F) {
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

  K->getParallelRegions(LI, &OriginalParallelRegions);

#ifdef DUMP_CFGS
  F.dump();
  dumpCFG(F, F.getName().str() + "_before_wiloops.dot",
          &OriginalParallelRegions);
#endif

  IRBuilder<> builder(&*(F.getEntryBlock().getFirstInsertionPt()));
  LocalIdXFirstVar = builder.CreateAlloca(SizeT, 0, ".pocl.local_id_x_init");

  //  F.viewCFGOnly();

#if 0
  std::cerr << "### Original" << std::endl;
  F.viewCFGOnly();
#endif

#if 0
  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
           PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI)
  {
    ParallelRegion *region = (*PRI);
    region->InjectRegionPrintF();
    region->InjectVariablePrintouts();
  }
#endif

  /* Count how many parallel regions share each entry node to
     detect diverging regions that need to be peeled. */
  std::map<llvm::BasicBlock*, int> entryCounts;

  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
           PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {
    ParallelRegion *Region = (*PRI);
#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### Adding context save/restore for PR: ";
    Region->dumpNames();
#endif
    fixMultiRegionVariables(Region);
    entryCounts[Region->entryBB()]++;
  }

#if 0
  std::cerr << "### After context code addition:" << std::endl;
  F.viewCFG();
#endif
  std::map<ParallelRegion*, bool> peeledRegion;
  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
           PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {

    llvm::ValueToValueMapTy reference_map;
    ParallelRegion *original = (*PRI);

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
            llvm::BasicBlock *BB = *PI;
            if (DT.dominates(original->entryBB(), BB) &&
                (regionOfBlock(original->entryBB()) == regionOfBlock(BB)))
              continue;
            preds.push_back(BB);
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
                appendIncBlock(original->exitBB(), LocalIdXGlobal);
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

      l = createLoopAround(*original, l.first, l.second, peelFirst,
                           LocalIdXGlobal, WGLocalSizeX, !unrolled, gv);

      gv = M->getGlobalVariable("_local_size_y");
      if (gv == NULL)
        gv = new GlobalVariable(*M, SizeT, false, GlobalValue::CommonLinkage,
                                NULL, "_local_size_y");

      l = createLoopAround(*original, l.first, l.second,
                           false, LocalIdYGlobal, WGLocalSizeY, !unrolled, gv);

      gv = M->getGlobalVariable("_local_size_z");
      if (gv == NULL)
        gv = new GlobalVariable(*M, SizeT, true, GlobalValue::CommonLinkage,
                                NULL, "_local_size_z", NULL,
                                GlobalValue::ThreadLocalMode::NotThreadLocal,
                                0, true);

      l = createLoopAround(*original, l.first, l.second,
                           false, LocalIdZGlobal, WGLocalSizeZ, !unrolled, gv);

    } else {
      if (WGLocalSizeX > 1) {
        l = createLoopAround(*original, l.first, l.second, peelFirst,
                             LocalIdXGlobal, WGLocalSizeX, !unrolled);
      }

      if (WGLocalSizeY > 1) {
        l = createLoopAround(*original, l.first, l.second, false,
                             LocalIdYGlobal, WGLocalSizeY);
      }

      if (WGLocalSizeZ > 1) {
        l = createLoopAround(*original, l.first, l.second, false,
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
           PRI = OriginalParallelRegions.begin(),
       PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {
    ParallelRegion *PR = (*PRI);

    if (!peeledRegion[PR]) continue;
    PR->insertPrologue(0, 0, 0);
    builder.SetInsertPoint(&*(PR->entryBB()->getFirstInsertionPt()));
    builder.CreateStore(ConstantInt::get(SizeT, 1), LocalIdXFirstVar);
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
void WorkitemLoopsImpl::fixMultiRegionVariables(ParallelRegion *Region) {

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
             regionOfBlock(User->getParent()) != NULL)) {
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

// Convert calls to the __pocl_{work_group,local_mem}_alloca() pseudo function
// to allocas.
bool WorkitemLoopsImpl::handleLocalMemAllocas(Kernel &K) {

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
    Value *ExtraSize = Call->getArgOperand(2);

    IRBuilder<> Builder(K.getEntryBlock().getTerminator());

    if (Call->getCalledFunction() == WorkGroupAllocaFuncDecl) {
          Instruction *WGSize = getWorkGroupSizeInstr(K);
          Size = Builder.CreateBinOp(Instruction::Mul, WGSize, Size);
          Size = Builder.CreateBinOp(Instruction::Add, Size, ExtraSize);
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

llvm::Value *WorkitemLoopsImpl::getLinearWiIndex(llvm::IRBuilder<> &Builder,
                                                 llvm::Module *M,
                                                 ParallelRegion *Region) {
  GlobalVariable *LocalSizeXPtr =
    cast<GlobalVariable>(M->getOrInsertGlobal("_local_size_x", SizeT));
  GlobalVariable *LocalSizeYPtr =
    cast<GlobalVariable>(M->getOrInsertGlobal("_local_size_y", SizeT));

  assert(LocalSizeXPtr != NULL && LocalSizeYPtr != NULL);

  LoadInst* LoadX = Builder.CreateLoad(SizeT, LocalSizeXPtr, "ls_x");
  LoadInst* LoadY = Builder.CreateLoad(SizeT, LocalSizeYPtr, "ls_y");

  /* Form linear index from xyz coordinates:
       local_size_x * local_size_y * local_id_z  (z dimension)
     + local_size_x * local_id_y                 (y dimension)
     + local_id_x                                (x dimension)
  */
  Value* LocalSizeXTimesY =
    Builder.CreateBinOp(Instruction::Mul, LoadX, LoadY, "ls_xy");

  Value* ZPart =
    Builder.CreateBinOp(Instruction::Mul, LocalSizeXTimesY,
                        Region->LocalIDZLoad(),
                        "tmp");

  Value* YPart =
    Builder.CreateBinOp(Instruction::Mul, LoadX,
                        Region->LocalIDYLoad(),
                        "ls_x_y");

  Value* ZYSum =
    Builder.CreateBinOp(Instruction::Add, ZPart, YPart,
                        "zy_sum");

  return Builder.CreateBinOp(Instruction::Add, ZYSum, Region->LocalIDXLoad(),
                             "linear_xyz_idx");
}

llvm::Instruction *
WorkitemLoopsImpl::addContextSave(llvm::Instruction *Inst,
                                  llvm::AllocaInst *AllocaI) {

  if (isa<AllocaInst>(Inst)) {
    // If the variable to be context saved is itself an alloca, we have created
    // one big alloca that stores the data of all the work-items and return
    // pointers to that array. Thus, we need no initialization code other than
    // the context data alloca itself.
    return NULL;
  }

  /* Save the produced variable to the array. */
  BasicBlock::iterator definition = (dyn_cast<Instruction>(Inst))->getIterator();
  ++definition;
  while (isa<PHINode>(definition)) ++definition;

  IRBuilder<> builder(&*definition);
  std::vector<llvm::Value *> gepArgs;

  /* Reuse the id loads earlier in the region, if possible, to
     avoid messy output with lots of redundant loads. */
  ParallelRegion *region = regionOfBlock(Inst->getParent());
  assert ("Adding context save outside any region produces illegal code." && 
          region != NULL);

  if (WGDynamicLocalSize)
    {
      Module *M = AllocaI->getParent()->getParent()->getParent();
      gepArgs.push_back(getLinearWiIndex(builder, M, region));
    }
  else
    {
      gepArgs.push_back(ConstantInt::get(SizeT, 0));
      gepArgs.push_back(region->LocalIDZLoad());
      gepArgs.push_back(region->LocalIDYLoad());
      gepArgs.push_back(region->LocalIDXLoad());
    }

    return builder.CreateStore(
        Inst,
#if LLVM_MAJOR < 15
        builder.CreateGEP(AllocaI->getType()->getPointerElementType(), AllocaI,
                          gepArgs));
#else
        builder.CreateGEP(AllocaI->getAllocatedType(), AllocaI, gepArgs));
#endif
}

llvm::Instruction *WorkitemLoopsImpl::addContextRestore(llvm::Value *Val,
    llvm::AllocaInst *AllocaI, llvm::Type *InstType,
    bool PoclWrapperStructAdded, llvm::Instruction *Before, bool isAlloca) {

  assert(Val != NULL);
  assert(AllocaI != NULL);
  IRBuilder<> builder(AllocaI);
  if (Before != NULL) {
    builder.SetInsertPoint(Before);
  } else if (isa<Instruction>(Val)) {
    builder.SetInsertPoint(dyn_cast<Instruction>(Val));
    Before = dyn_cast<Instruction>(Val);
  } else {
    assert(false && "Unknown context restore location!");
  }

  std::vector<llvm::Value *> gepArgs;

  /* Reuse the id loads earlier in the region, if possible, to
     avoid messy output with lots of redundant loads. */
  ParallelRegion *region = regionOfBlock(Before->getParent());
  assert ("Adding context save outside any region produces illegal code." && 
          region != NULL);

  if (WGDynamicLocalSize)
    {
      Module *M = AllocaI->getParent()->getParent()->getParent();
      gepArgs.push_back(getLinearWiIndex(builder, M, region));
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
      ConstantInt::get(Type::getInt32Ty(AllocaI->getContext()), 0));

#if LLVM_MAJOR < 15
  llvm::Instruction *gep = dyn_cast<Instruction>(
    builder.CreateGEP(
      AllocaI->getType()->getPointerElementType(), AllocaI, gepArgs));
#else
  llvm::Instruction *gep = dyn_cast<Instruction>(
    builder.CreateGEP(
      AllocaI->getAllocatedType(), AllocaI, gepArgs));
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
llvm::AllocaInst *
WorkitemLoopsImpl::getContextArray(llvm::Instruction *Inst,
                                   bool &PoclWrapperStructAdded) {
  PoclWrapperStructAdded = false;
  /*
   * Unnamed temp instructions need a generated name for the
   * context array. Create one using a running integer.
   */
  std::ostringstream var;
  var << ".";

  if (std::string(Inst->getName().str()) != "")
    {
      var << Inst->getName().str();
    }
  else if (TempInstructionIds.find(Inst) != TempInstructionIds.end())
    {
      var << TempInstructionIds[Inst];
    }
  else
    {
      TempInstructionIds[Inst] = TempInstructionIndex++;
      var << TempInstructionIds[Inst];
    }

  var << ".pocl_context";
  std::string varName = var.str();

  if (ContextArrays.find(varName) != ContextArrays.end())
    return ContextArrays[varName];

  BasicBlock &bb = Inst->getParent()->getParent()->getEntryBlock();
  IRBuilder<> builder(&*(bb.getFirstInsertionPt()));
  Function *FF = Inst->getParent()->getParent();
  Module *M = Inst->getParent()->getParent()->getParent();
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
            if (Inst == V) {
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
  if (isa<AllocaInst>(Inst))
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
        dyn_cast<AllocaInst>(Inst)->getAllocatedType();
    } 
  else 
    {
      elementType = Inst->getType();
    }

  /* 3D context array. In case the elementType itself is an array or struct,
   * we must take into account it could be alloca-ed with alignment and loads
   * or stores might use vectorized instructions expecting proper alignment.
   * Because of that, we cannot simply allocate x*y*z*(size), we must
   * enlarge the type to fit the alignment. */
  Type *AllocType = elementType;
  AllocaInst *InstCast = dyn_cast<AllocaInst>(Inst);
  if (InstCast) {
#if LLVM_MAJOR < 15
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
  Alloca->setAlignment(llvm::Align(CONTEXT_ARRAY_ALIGN));

    if (DebugVal && DebugCall && !WGDynamicLocalSize) {

      llvm::SmallVector<llvm::Metadata *, 4> Subscripts;
      Subscripts.push_back(DB->getOrCreateSubrange(0, WGLocalSizeZ));
      Subscripts.push_back(DB->getOrCreateSubrange(0, WGLocalSizeY));
      Subscripts.push_back(DB->getOrCreateSubrange(0, WGLocalSizeX));
      llvm::DINodeArray SubscriptArray = DB->getOrCreateArray(Subscripts);

      size_t sizeBits;
      sizeBits = Alloca
                     ->getAllocationSizeInBits(M->getDataLayout())
#if LLVM_MAJOR > 14
                     .value_or(TypeSize(0, false))
                     .getFixedValue();
#else
                     .getValueOr(TypeSize(0, false))
                     .getFixedValue();
#endif

      assert(sizeBits != 0);

      // if (size == 0) WGLocalSizeX * WGLocalSizeY * WGLocalSizeZ * 8 *
      // Alloca->getAllocatedType()->getScalarSizeInBits();
#if LLVM_MAJOR < 15
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

    ContextArrays[varName] = Alloca;
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
void WorkitemLoopsImpl::addContextSaveRestore(llvm::Instruction *Instr) {

  //

  // Allocate the context data array for the variable.
  bool PoclWrapperStructAdded = false;
  llvm::AllocaInst *Alloca = getContextArray(Instr, PoclWrapperStructAdded);
  llvm::Instruction *TheStore = addContextSave(Instr, Alloca);

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
    if (regionOfBlock(UserI->getParent()) == NULL) continue;

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
               && regionOfBlock(
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
    llvm::Value *LoadedValue = addContextRestore(
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

bool WorkitemLoopsImpl::shouldNotBeContextSaved(llvm::Instruction *Instr) {

  if (isa<BranchInst>(Instr)) return true;

  // The local memory allocation call is uniform, the same pointer to the
  // work-group shared memory area is returned to all work-items. It must
  // not be replicated.
  if (isa<CallInst>(Instr)) {
    Function *F = cast<CallInst>(Instr)->getCalledFunction();
    if (F && (F == LocalMemAllocaFuncDecl || F == WorkGroupAllocaFuncDecl))
      return true;
  }

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

llvm::BasicBlock *WorkitemLoopsImpl::appendIncBlock(llvm::BasicBlock *After,
                                                    llvm::Value *LocalIdVar) {
  llvm::LLVMContext &C = After->getContext();

  llvm::BasicBlock *oldExit = After->getTerminator()->getSuccessor(0);
  assert (oldExit != NULL);

  llvm::BasicBlock *forIncBB =
    BasicBlock::Create(C, "pregion_for_inc", After->getParent());

  After->getTerminator()->replaceUsesOfWith(oldExit, forIncBB);

  IRBuilder<> builder(oldExit);

  builder.SetInsertPoint(forIncBB);
  /* Create the iteration variable increment */
  builder.CreateStore(builder.CreateAdd(
                        builder.CreateLoad(SizeT, LocalIdVar),
                        ConstantInt::get(SizeT, 1)),
                      LocalIdVar);

  builder.CreateBr(oldExit);

  return forIncBB;
}

llvm::Instruction *WorkitemLoopsImpl::getWorkGroupSizeInstr(llvm::Function &F) {

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

// enable new pass manager infrastructure
llvm::PreservedAnalyses WorkitemLoops::run(llvm::Function &F,
                                           llvm::FunctionAnalysisManager &AM) {
  if (!isKernelToProcess(F))
    return llvm::PreservedAnalyses::all();

  WorkitemHandlerType WIH = AM.getResult<WorkitemHandlerChooser>(F).WIH;
  if (WIH != WorkitemHandlerType::LOOPS &&
      !(WIH == WorkitemHandlerType::CBS && !hasWorkgroupBarriers(F)))
    return llvm::PreservedAnalyses::all();

  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
  auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);
  auto &VUA = AM.getResult<VariableUniformityAnalysis>(F);

  llvm::PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<VariableUniformityAnalysis>();
  PAChanged.preserve<WorkitemHandlerChooser>();

  WorkitemLoopsImpl WIL(DT, LI, PDT, VUA);
  return WIL.runOnFunction(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
