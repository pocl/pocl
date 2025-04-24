// LLVM function pass to create loops that run all the work items
// in a work group while respecting barrier synchronization points.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen / Tampere University
//               2022-2025 Pekka Jääskeläinen / Intel Finland Oy
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
#include <llvm/ADT/Statistic.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ValueSymbolTable.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "Barrier.h"
#include "DebugHelpers.h"
#include "Kernel.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
#include "WorkitemLoops.h"

POP_COMPILER_DIAGS

#include <array>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#define DEBUG_TYPE "WIL"

#define PASS_NAME "workitemloops"
#define PASS_CLASS pocl::WorkitemLoops
#define PASS_DESC "Workitem loop generation pass"

//#define DEBUG_WORK_ITEM_LOOPS
//#define POCL_KERNEL_COMPILER_DUMP_CFGS

// Use the LLVM_DEBUG-style macros to gradually convert to LLVM-upstreamable
// code.
#ifdef LLVM_DEBUG
#undef LLVM_DEBUG
#endif

#ifdef DEBUG_WORK_ITEM_LOOPS
#define LLVM_DEBUG(X) X
#define dbgs() std::cerr << DEBUG_TYPE << ": "
#else
#define LLVM_DEBUG(X)
#endif

namespace pocl {

using namespace llvm;

class WorkitemLoopsImpl : public pocl::WorkitemHandler {
public:
  WorkitemLoopsImpl(llvm::DominatorTree &DT, llvm::LoopInfo &LI,
                    llvm::PostDominatorTree &PDT,
                    VariableUniformityAnalysisResult &VUA)
      : WorkitemHandler(), DT(DT), LI(LI), PDT(PDT), VUA(VUA) {}
  virtual bool runOnFunction(llvm::Function &F);

protected:
  llvm::Value *getLinearWIIndexInRegion(llvm::Instruction *Instr) override;
  llvm::Instruction *getLocalIdInRegion(llvm::Instruction *Instr,
                                        size_t Dim) override;

private:
  using BasicBlockVector = std::vector<llvm::BasicBlock *>;
  using InstructionIndex = std::set<llvm::Instruction *>;
  using InstructionVec = std::vector<llvm::Instruction *>;
  using StrInstructionMap = std::map<std::string, llvm::AllocaInst *>;

  llvm::DominatorTree &DT;
  llvm::LoopInfo &LI;
  llvm::PostDominatorTree &PDT;
  llvm::Module *M;
  llvm::Function *F;

  // Count of times the basic block is a region entry. Used to detect
  // diverging barrier regions which should be peeled to figure out
  // the control flow.
  std::map<llvm::BasicBlock*, size_t> RegionEntryCounts;

  VariableUniformityAnalysisResult &VUA;

  ParallelRegion::ParallelRegionVector OriginalParallelRegions;

  StrInstructionMap ContextArrays;

  // Temporary global_id_* iteration variables updated by the work-item
  // loops.
  std::array<llvm::GlobalVariable *, 3> GlobalIdIterators;

  bool processFunction(llvm::Function &F);

  void fixMultiRegionVariables(ParallelRegion *Region);
  void addContextSaveRestore(llvm::Instruction *instruction);
  void releaseParallelRegions();

  // Returns an instruction in the entry block which computes the
  // total size of work-items in the work-group. If it doesn't
  // exist, creates it to the end of the entry block.
  llvm::Instruction *getWorkGroupSizeInstr(llvm::Function &F);

  llvm::Value *getLinearWiIndex(llvm::IRBuilder<> &Builder, llvm::Module *M,
                                ParallelRegion *Region);
  llvm::Instruction *addContextSave(llvm::Instruction *Def,
                                    llvm::AllocaInst *AllocaI);
  llvm::Value *tryToRematerialize(llvm::Instruction *Before, llvm::Value *Def,
                                  const std::string &NamePrefix,
                                  bool *CanDoIt = nullptr, int *Depth = 0);

  llvm::Instruction *
  addContextRestore(llvm::Value *Val, llvm::AllocaInst *AllocaI,
                    llvm::Type *LoadInstType, bool PaddingWasAdded,
                    llvm::Instruction *Before = nullptr, bool isAlloca = false);
  llvm::AllocaInst *getContextArray(llvm::Instruction *Inst,
                                    bool &PoclWrapperStructAdded);

  std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
  createLoopAround(ParallelRegion &Region, llvm::BasicBlock *EntryBB,
                   llvm::BasicBlock *ExitBB, bool PeeledFirst, int Dim,
                   bool AddIncBlock = true,
                   llvm::Value *DynamicLocalSize = nullptr);

  llvm::BasicBlock *appendIncBlock(llvm::BasicBlock *After, int Dim,
                                   llvm::BasicBlock *Before = nullptr,
                                   const std::string &BBName = "");

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

bool WorkitemLoopsImpl::runOnFunction(Function &Func) {

  M = Func.getParent();
  F = &Func;
  Initialize(cast<Kernel>(&Func));

  LLVM_DEBUG(dbgs() << "Before WILoops:\n");
  LLVM_DEBUG(Func.dump());

  GlobalIdIterators = {
      cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(0), ST)),
      cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(1), ST)),
      cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(2), ST))};

  TempInstructionIndex = 0;

  bool Changed = processFunction(Func);

  Changed |= handleLocalMemAllocas();

#ifdef DUMP_CFGS
  dumpCFG(*F, F->getName().str() + "_after_wiloops.dot", nullptr,
          &OriginalParallelRegions);
#endif

  Changed |= fixUndominatedVariableUses(DT, Func);

  ContextArrays.clear();
  TempInstructionIds.clear();

  releaseParallelRegions();
  LLVM_DEBUG(dbgs() << "After WILoops:\n");
  LLVM_DEBUG(Func.dump());
  return Changed;
}

std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
WorkitemLoopsImpl::createLoopAround(ParallelRegion &Region,
                                    llvm::BasicBlock *EntryBB,
                                    llvm::BasicBlock *ExitBB, bool PeeledFirst,
                                    int Dim, bool AddIncBlock,
                                    llvm::Value *DynamicLocalSize) {
  Value *LocalIdVar = LocalIdGlobals[Dim];

  size_t LocalSizes[] = {WGLocalSizeX, WGLocalSizeY, WGLocalSizeZ};
  size_t LocalSizeForDim = LocalSizes[Dim];
  Instruction *GlobalIdOrigin = getGlobalIdOrigin(Dim);

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

  // If the first iteration is peeled to figure out the barrier condition, we
  // need to skip the execution of the first iteration in the very first
  // iteration of the innermost loop.
  if (PeeledFirst) {
    Instruction *LocalIdXFirstVal = builder.CreateLoad(ST, LocalIdXFirstVar);
    builder.CreateStore(LocalIdXFirstVal, LocalIdVar);

    // Initialize the global id counter with skipped WI as well.
    GlobalVariable *GlobalId = GlobalIdIterators[Dim];
    builder.CreateStore(builder.CreateAdd(GlobalIdOrigin, LocalIdXFirstVal),
                        GlobalId);

    // Then reset the initializer to 0 since we want to execute the first WI
    // from the X dimension for the next Y and Z iterations.
    builder.CreateStore(ConstantInt::get(ST, 0), LocalIdXFirstVar);

    if (WGDynamicLocalSize) {
      llvm::Value *cmpResult;
      cmpResult =
          builder.CreateICmpULT(builder.CreateLoad(ST, LocalIdVar),
                                builder.CreateLoad(ST, DynamicLocalSize));

      builder.CreateCondBr(cmpResult, LoopBodyEntryBB, loopEndBB);
    } else {
      builder.CreateBr(LoopBodyEntryBB);
    }
  } else {
    builder.CreateStore(ConstantInt::get(ST, 0), LocalIdVar);

    // Initialize the global id counter with the base.
    GlobalVariable *GlobalId = GlobalIdIterators[Dim];
    builder.CreateStore(GlobalIdOrigin, GlobalId);

    builder.CreateBr(LoopBodyEntryBB);
  }

  ExitBB->getTerminator()->replaceUsesOfWith(oldExit, forCondBB);
  if (AddIncBlock) {
    appendIncBlock(ExitBB, Dim);
  }

  builder.SetInsertPoint(forCondBB);

  llvm::Value *cmpResult;
  if (!WGDynamicLocalSize)
    cmpResult = builder.CreateICmpULT(builder.CreateLoad(ST, LocalIdVar),
                                      ConstantInt::get(ST, LocalSizeForDim));
  else
    cmpResult = builder.CreateICmpULT(builder.CreateLoad(ST, LocalIdVar),
                                      builder.CreateLoad(ST, DynamicLocalSize));

  Instruction *LoopBranch =
      builder.CreateCondBr(cmpResult, LoopBodyEntryBB, loopEndBB);

  if (canAnnotateParallelLoops()) {
    // Add the metadata to mark a parallel loop. The metadata refers to
    // a loop-unique dummy metadata that is not merged automatically.
    // TODO: Merge with the similar code in SubCFGFormation.

    // This creation of the identifier metadata is copied from
    // LLVM's MDBuilder::createAnonymousTBAARoot().

    MDNode *Dummy = MDNode::getTemporary(C, ArrayRef<Metadata *>()).release();
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
    LoopBranch->setMetadata("llvm.loop", Root);

    auto IsLoadUnconditionallySafe =
        [&dominatesExitBB](llvm::Instruction *Insn) -> bool {
      assert(Insn->mayReadFromMemory());
      // Checks that the instruction isn't in a conditional block.
      return dominatesExitBB.count(Insn->getParent());
    };

    Region.addParallelLoopMetadata(AccessGroupMD, IsLoadUnconditionallySafe);
  }

  builder.SetInsertPoint(loopEndBB);
  builder.CreateBr(oldExit);

  return std::make_pair(forInitBB, loopEndBB);
}

ParallelRegion *WorkitemLoopsImpl::regionOfBlock(llvm::BasicBlock *BB) {
  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
           PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {
    ParallelRegion *PRegion = (*PRI);
    if (PRegion->hasBlock(BB))
      return PRegion;
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

  releaseParallelRegions();

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  // Append 'dyn' or 'static' to the dot files to differentiate between the
  // dynamic WG one (produced for the binaries) and the specialized static one.
  std::string DotSuffix = WGDynamicLocalSize ? "_dyn" : "_static";
  dumpCFG(F, F.getName().str() + "_before_pregions" + DotSuffix + ".dot", nullptr, nullptr);
#endif

  K->getParallelRegions(LI, &OriginalParallelRegions);
  handleWorkitemFunctions();

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  dumpCFG(F, F.getName().str() + "_before_wiloops" + DotSuffix + ".dot", nullptr,
          &OriginalParallelRegions);
#endif

  IRBuilder<> builder(&*(F.getEntryBlock().getFirstInsertionPt()));
  LocalIdXFirstVar = builder.CreateAlloca(ST, 0, ".pocl.local_id_x_init");

#if 0
  for (ParallelRegion::ParallelRegionVector::iterator
         PRI = OriginalParallelRegions.begin(),
         PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {
    ParallelRegion *region = (*PRI);
    region->InjectRegionPrintF();
    region->InjectVariablePrintouts();
  }
#endif

  /* Count how many parallel regions share each entry node to
     detect diverging regions that need to be peeled. */
  RegionEntryCounts.clear();
  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
           PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {
    ParallelRegion *Region = (*PRI);
    RegionEntryCounts[Region->entryBB()]++;
  }

  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
           PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {
    ParallelRegion *Region = (*PRI);
    LLVM_DEBUG(dbgs() << "#### Adding context save/restore for PR:\n");
    LLVM_DEBUG(Region->dumpNames());
    fixMultiRegionVariables(Region);
  }

#if 0
  std::cerr << "### After context code addition:" << std::endl;
  F.viewCFG();
#endif
  std::map<ParallelRegion*, bool> PeeledRegion;
  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
           PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {

    llvm::ValueToValueMapTy reference_map;
    ParallelRegion *PRegion = (*PRI);

    LLVM_DEBUG(dbgs() << "Handling region:\n");
    LLVM_DEBUG(PRegion->dumpNames());
    //F.viewCFGOnly();

    /* In case of conditional barriers, the first iteration
       has to be peeled so we know which branch to execute
       with the work item loop. In case there are more than one
       parallel region sharing an entry BB, it's a diverging
       region.

       Post dominance of entry by exit does not work in case the
       region is inside a loop and the exit block is in the path
       towards the loop exit (and the function exit).
    */
    bool PeelFirst = RegionEntryCounts[PRegion->entryBB()] > 1;
    PeeledRegion[PRegion] = PeelFirst;

    std::pair<llvm::BasicBlock *, llvm::BasicBlock *> l;
    // the original predecessor nodes of which successor
    // should be fixed if not peeling
    BasicBlockVector preds;

    bool Unrolled = false;
    if (PeelFirst) {
        ParallelRegion *replica =
            PRegion->replicate(reference_map, ".peeled_wi");
        replica->chainAfter(PRegion);
        replica->purge();
      LLVM_DEBUG(dbgs() << "Conditional region, peeling the first iteration\n");

        l = std::make_pair(replica->entryBB(), replica->exitBB());
    } else {
      llvm::pred_iterator PI = llvm::pred_begin(PRegion->entryBB()),
                          E = llvm::pred_end(PRegion->entryBB());

      for (; PI != E; ++PI) {
        llvm::BasicBlock *BB = *PI;
        if (DT.dominates(PRegion->entryBB(), BB) &&
            (regionOfBlock(PRegion->entryBB()) == regionOfBlock(BB)))
          continue;
        preds.push_back(BB);
      }

      unsigned UnrollCount;
      if (getenv("POCL_WILOOPS_MAX_UNROLL_COUNT") != NULL)
        UnrollCount = atoi(getenv("POCL_WILOOPS_MAX_UNROLL_COUNT"));
      else
        UnrollCount = 1;
      /* Find a two's exponent unroll count, if available. */
      while (UnrollCount >= 1) {
        if (WGLocalSizeX % UnrollCount == 0 && UnrollCount <= WGLocalSizeX) {
          break;
        }
        UnrollCount /= 2;
      }

      if (UnrollCount > 1) {
        ParallelRegion *prev = PRegion;
        llvm::BasicBlock *lastBB = appendIncBlock(
            PRegion->exitBB(), 0, PRegion->exitBB(),
            std::string("pregion.") + std::to_string(PRegion->getID()) +
                ".dim0.for_inc");
        PRegion->AddBlockAfter(lastBB, PRegion->exitBB());
        PRegion->SetExitBB(lastBB);

        for (unsigned c = 1; c < UnrollCount; ++c) {
          ParallelRegion *UnrolledPR =
              PRegion->replicate(reference_map, ".unrolled_wi");
          UnrolledPR->chainAfter(prev);
          prev = UnrolledPR;
          lastBB = UnrolledPR->exitBB();
        }
        Unrolled = true;
        l = std::make_pair(PRegion->entryBB(), lastBB);
      } else {
        l = std::make_pair(PRegion->entryBB(), PRegion->exitBB());
      }
    }

    if (WGDynamicLocalSize) {
      GlobalVariable *gv;
      gv = M->getGlobalVariable("_local_size_x");
      if (gv == NULL)
        gv = new GlobalVariable(
            *M, ST, true, GlobalValue::CommonLinkage, NULL, "_local_size_x",
            NULL, GlobalValue::ThreadLocalMode::NotThreadLocal, 0, true);

      l = createLoopAround(*PRegion, l.first, l.second, PeelFirst, 0, !Unrolled,
                           gv);

      gv = M->getGlobalVariable("_local_size_y");
      if (gv == NULL)
        gv = new GlobalVariable(*M, ST, false, GlobalValue::CommonLinkage, NULL,
                                "_local_size_y");

      l = createLoopAround(*PRegion, l.first, l.second, false, 1, !Unrolled,
                           gv);

      gv = M->getGlobalVariable("_local_size_z");
      if (gv == NULL)
        gv = new GlobalVariable(
            *M, ST, true, GlobalValue::CommonLinkage, NULL, "_local_size_z",
            NULL, GlobalValue::ThreadLocalMode::NotThreadLocal, 0, true);

      l = createLoopAround(*PRegion, l.first, l.second, false, 2, !Unrolled,
                           gv);

    } else {
      if (WGLocalSizeX > 1) {
        l = createLoopAround(*PRegion, l.first, l.second, PeelFirst, 0,
                             !Unrolled);
      } else {
        // Ensure the global id for a 1-size dimension is initialized.
        getGlobalIdOrigin(0);
      }

      if (WGLocalSizeY > 1) {
        l = createLoopAround(*PRegion, l.first, l.second, false, 1);
      } else {
        getGlobalIdOrigin(1);
      }

      if (WGLocalSizeZ > 1) {
        l = createLoopAround(*PRegion, l.first, l.second, false, 2);
      } else {
        getGlobalIdOrigin(2);
      }
    }

    // Loop edges coming from another region mean B-loops which means
    // we have to fix the loop edge to jump to the beginning of the wi-loop
    // structure, not its body. This has to be done only for non-peeled
    // blocks as the semantics is correct in the other case (the jump is
    // to the beginning of the peeled iteration).
    if (!PeelFirst) {
      for (BasicBlockVector::iterator i = preds.begin(); i != preds.end();
           ++i) {
        llvm::BasicBlock *BB = *i;
        BB->getTerminator()->replaceUsesOfWith(PRegion->entryBB(), l.first);
      }
    }
  }

  // For the peeled regions we need to add a prologue that initializes the local
  // ids and the first iteration counter.
  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
       PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {
    ParallelRegion *PR = (*PRI);

    if (!PeeledRegion[PR]) continue;
    PR->insertPrologue(0, 0, 0);
    builder.SetInsertPoint(&*(PR->entryBB()->getFirstInsertionPt()));
    builder.CreateStore(ConstantInt::get(ST, 1), LocalIdXFirstVar);
  }

  if (!WGDynamicLocalSize)
    K->addLocalSizeInitCode(WGLocalSizeX, WGLocalSizeY, WGLocalSizeZ);

  ParallelRegion::insertLocalIdInit(&F.getEntryBlock(), 0, 0, 0);

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  dumpCFG(*K, K->getName().str() + "_after_wiloops" + DotSuffix + ".dot", nullptr,
          &OriginalParallelRegions);
#endif

  return true;
}

/// Add context save/restore code to variables that are defined in
/// the given region and are used outside the region.
void WorkitemLoopsImpl::fixMultiRegionVariables(ParallelRegion *Region) {

  InstructionIndex InstructionsInRegion;
  InstructionVec ValuesToContextSave;

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
          ValuesToContextSave.push_back(Instr);
          break;
        }
      }
    }
  }
  // Finally generate the context save/restore (or rematerialization) code for
  // the instructions requiring it.
  for (auto &I : ValuesToContextSave) {
    LLVM_DEBUG(dbgs() << "\nAdding context/save restore for\n");
    LLVM_DEBUG(I->dump());
    addContextSaveRestore(I);
  }
}

// TO CLEAN: Refactor into getLinearWIIndexInRegion.
llvm::Value *WorkitemLoopsImpl::getLinearWiIndex(llvm::IRBuilder<> &Builder,
                                                 llvm::Module *M,
                                                 ParallelRegion *Region) {
  GlobalVariable *LocalSizeXPtr =
      cast<GlobalVariable>(M->getOrInsertGlobal("_local_size_x", ST));
  GlobalVariable *LocalSizeYPtr =
      cast<GlobalVariable>(M->getOrInsertGlobal("_local_size_y", ST));

  assert(LocalSizeXPtr != NULL && LocalSizeYPtr != NULL);

  LoadInst *LoadX = Builder.CreateLoad(ST, LocalSizeXPtr, "ls_x");
  LoadInst *LoadY = Builder.CreateLoad(ST, LocalSizeYPtr, "ls_y");

  /* Form linear index from xyz coordinates:
       local_size_x * local_size_y * local_id_z  (z dimension)
     + local_size_x * local_id_y                 (y dimension)
     + local_id_x                                (x dimension)
  */
  Value* LocalSizeXTimesY =
    Builder.CreateBinOp(Instruction::Mul, LoadX, LoadY, "ls_xy");

  Value *ZPart =
      Builder.CreateBinOp(Instruction::Mul, LocalSizeXTimesY,
                          Region->getOrCreateIDLoad(LID_G_NAME(2)), "tmp");

  Value *YPart =
      Builder.CreateBinOp(Instruction::Mul, LoadX,
                          Region->getOrCreateIDLoad(LID_G_NAME(1)), "ls_x_y");

  Value* ZYSum =
    Builder.CreateBinOp(Instruction::Add, ZPart, YPart,
                        "zy_sum");

  return Builder.CreateBinOp(Instruction::Add, ZYSum,
                             Region->getOrCreateIDLoad(LID_G_NAME(0)),
                             "linear_xyz_idx");
}

llvm::Value *
WorkitemLoopsImpl::getLinearWIIndexInRegion(llvm::Instruction *Instr) {
  ParallelRegion *ParRegion = regionOfBlock(Instr->getParent());
  assert(ParRegion != nullptr);
  IRBuilder<> Builder(Instr);
  return getLinearWiIndex(Builder, M, ParRegion);
}

llvm::Instruction *
WorkitemLoopsImpl::getLocalIdInRegion(llvm::Instruction *Instr, size_t Dim) {
  ParallelRegion *ParRegion = regionOfBlock(Instr->getParent());
  if (ParRegion != nullptr) {
    return ParRegion->getOrCreateIDLoad(LID_G_NAME(Dim));
  }
  IRBuilder<> Builder(Instr);
  return Builder.CreateLoad(ST, LocalIdGlobals[Dim]);
}

/// Adds a value store to the context array after the given defining
/// instruction.
///
/// \param Def The instruction that defines the original value.
/// \param AllocaI The alloca created for for the context array.
llvm::Instruction *
WorkitemLoopsImpl::addContextSave(llvm::Instruction *Def,
                                  llvm::AllocaInst *AllocaI) {

  if (isa<AllocaInst>(Def)) {
    // If the variable to be context saved is itself an alloca, we have created
    // one big alloca that stores the data of all the work-items and return
    // pointers to that array. Thus, we need no initialization code other than
    // the context data alloca itself.
    return NULL;
  }

  /* Save the produced variable to the array. */
  BasicBlock::iterator definition = (dyn_cast<Instruction>(Def))->getIterator();
  ++definition;
  while (isa<PHINode>(definition)) ++definition;

  // TO CLEAN: Refactor by calling CreateContextArrayGEP.
  IRBuilder<> builder(&*definition);
  std::vector<llvm::Value *> gepArgs;

  /* Reuse the id loads earlier in the region, if possible, to
     avoid messy output with lots of redundant loads. */
  ParallelRegion *region = regionOfBlock(Def->getParent());
  assert ("Adding context save outside any region produces illegal code." && 
          region != NULL);

  if (WGDynamicLocalSize) {
    Module *M = AllocaI->getParent()->getParent()->getParent();
    gepArgs.push_back(getLinearWiIndex(builder, M, region));
  } else {
    gepArgs.push_back(ConstantInt::get(ST, 0));
    gepArgs.push_back(region->getOrCreateIDLoad(LID_G_NAME(2)));
    gepArgs.push_back(region->getOrCreateIDLoad(LID_G_NAME(1)));
    gepArgs.push_back(region->getOrCreateIDLoad(LID_G_NAME(0)));
  }

  return builder.CreateStore(
      Def,
#if LLVM_MAJOR < 15
      builder.CreateGEP(AllocaI->getType()->getPointerElementType(), AllocaI,
                        gepArgs));
#else
      builder.CreateGEP(AllocaI->getAllocatedType(), AllocaI, gepArgs));
#endif
}

llvm::Instruction *WorkitemLoopsImpl::addContextRestore(
    llvm::Value *Val, llvm::AllocaInst *AllocaI, llvm::Type *LoadInstType,
    bool PaddingWasAdded, llvm::Instruction *Before, bool isAlloca) {

  assert(Before != nullptr);

  llvm::Instruction *GEP =
      createContextArrayGEP(AllocaI, Before, PaddingWasAdded);
  if (isAlloca) {
    // In case the context saved instruction was an alloca, we created a
    // context array with pointed-to elements, and now want to return a
    // pointer to the elements to emulate the original alloca.
    return GEP;
  }
  IRBuilder<> Builder(Before);
  return Builder.CreateLoad(LoadInstType, GEP);
}

/// Returns the context array (alloca) for the given \param Inst, creates it if
/// not found.
///
/// \param PaddingAdded will be set to true in case a wrapper struct was
/// added for padding in order to enforce proper alignment to the elements of
/// the array. Such padding might be needed to ensure aligned accessed from
/// single work-items accessing aggregates in the context data.
llvm::AllocaInst *WorkitemLoopsImpl::getContextArray(llvm::Instruction *Inst,
                                                     bool &PaddingAdded) {
  PaddingAdded = false;

  std::ostringstream Var;
  Var << ".";

  if (std::string(Inst->getName().str()) != "") {
    Var << Inst->getName().str();
  } else if (TempInstructionIds.find(Inst) != TempInstructionIds.end()) {
    Var << TempInstructionIds[Inst];
  } else {
    // Unnamed temp instructions need a name generated for the context array.
    // Create one using a running integer.
    TempInstructionIds[Inst] = TempInstructionIndex++;
    Var << TempInstructionIds[Inst];
  }

  Var << ".pocl_context";
  std::string CArrayName = Var.str();

  if (ContextArrays.find(CArrayName) != ContextArrays.end())
    return ContextArrays[CArrayName];

  BasicBlock &Entry = K->getEntryBlock();
  return ContextArrays[CArrayName] = createAlignedAndPaddedContextAlloca(
             Inst, &*(Entry.getFirstInsertionPt()), CArrayName, PaddingAdded);
}

/// Tries to rematerialize the given value-defining instruction.
///
/// Rematerialization in this context means recomputing the value produced
/// in the use site instead of storing and loading a once-computed variable
/// from the context.
///
/// \param Before the instruction before which the cloned instructions should
/// be added.
/// \param Def is the produced value to attempt to clone recursively.
/// \param NamePrefix a prefix string to add to the name of the cloned
/// instructions.
/// \param CanDoIt can be set to a true-initialized boolean in which case the
/// cloning is not actually done, but only its possibility is investigated.
/// \param Depth the recursion depth. Used to limit rematerialization size.
/// \return The rematerialized instruction if possible and beneficial.
llvm::Value *WorkitemLoopsImpl::tryToRematerialize(
    llvm::Instruction *Before, llvm::Value *Def, const std::string &NamePrefix,
    bool *CanDoIt, int *Depth) {

  auto DbgRemat = [=](const std::string &Reason) {
    LLVM_DEBUG(dbgs() << "##### " << Reason << "\n");
    LLVM_DEBUG(Def->dump());
  };

#define UNABLE_TO_REMAT(REASON)                                                \
  do {                                                                         \
    DbgRemat("cannot remat: " REASON);                                         \
    if (CanDoIt != nullptr)                                                    \
      *CanDoIt = false;                                                        \
    return nullptr;                                                            \
  } while (0)

#define ABLE_TO_REMAT()                                                        \
  do {                                                                         \
    if (CanDoIt != nullptr)                                                    \
      return nullptr;                                                          \
  } while (0)

  // A call without arguments: Setup a pre-check before cloning to see if we
  // can succeed.
  if (CanDoIt == nullptr && Depth == nullptr) {
    bool Able = true;
    int Depth = 0;
    tryToRematerialize(Before, Def, NamePrefix, &Able, &Depth);
    if (!Able)
      return nullptr;
    Depth = 0;
    return tryToRematerialize(Before, Def, NamePrefix, nullptr, &Depth);
  }

  // Limit the height of the cloned instruction tree to avoid counter-
  // productive rematerialization.
  if (Depth != nullptr && *Depth > 10)
    UNABLE_TO_REMAT("too deep");

  if (llvm::CallInst *Call = dyn_cast<CallInst>(Def)) {
    auto *Callee = Call->getCalledFunction();
    if (Callee == nullptr || (Callee->getName() != GID_BUILTIN_NAME &&
                              Callee->getName() != GS_BUILTIN_NAME &&
                              Callee->getName() != GROUP_ID_BUILTIN_NAME &&
                              Callee->getName() != LID_BUILTIN_NAME &&
                              Callee->getName() != LS_BUILTIN_NAME)) {
      UNABLE_TO_REMAT("called an unsupported function");
    }
  } else if (isa<Constant>(Def) || isa<Argument>(Def)) {
    ABLE_TO_REMAT();
    // No need to clone a constant or function argument, we can refer to the
    // original directly.
    return Def;
  } else if (isa<AllocaInst>(Def) &&
             dyn_cast<AllocaInst>(Def)->getParent() != &K->getEntryBlock()) {
    // The allocas in the pure uniform entry block can be referred to without
    // rematerialization. But other than that we do not yet handle recursive
    // alloca references. Should be an easy and valuable low hanging fruit.
    UNABLE_TO_REMAT("accesses another alloca that we cannot remat");
  }

  llvm::Instruction *Inst = dyn_cast<Instruction>(Def);
  if (Inst == nullptr)
    UNABLE_TO_REMAT("unsupported value type");

  if (Inst->mayWriteToMemory() || Inst->mayHaveSideEffects())
    UNABLE_TO_REMAT("has side-effects");

  if (Depth != nullptr)
    (*Depth)++;

  // If we end up referring to instructions in pure uniform blocks (at
  // least work group allocas are such), let's stop the cloning there
  // and refer to the original.
  if (Inst->getParent() == &K->getEntryBlock())
    return Inst;

  llvm::Instruction *Copy = CanDoIt == nullptr ? Inst->clone() : nullptr;
  if (Copy != nullptr) {
    Copy->setName(NamePrefix + ".remat");
    Copy->insertBefore(Inst2InsertPt(Before));
  }
  for (unsigned I = 0; I < Inst->getNumOperands(); ++I) {
    llvm::Value *ClonedArg = tryToRematerialize(Copy, Inst->getOperand(I),
                                                NamePrefix, CanDoIt, Depth);
    if (CanDoIt == nullptr)
      Copy->setOperand(I, ClonedArg);
    else if (!CanDoIt)
      return nullptr;
  }
  return Copy;
}

/// Adds context save/restore code for the value produced by the given
/// instruction.
///
/// First attemps to rematerialize the value instead of storing it to memory.
void WorkitemLoopsImpl::addContextSaveRestore(llvm::Instruction *Def) {

  InstructionVec Uses;
  // Restore the produced variable before each use to ensure the correct
  // context copy is used.

  bool RematCandidate = true;

  // In case of a rematerialized alloca with only a single store, this will have
  // the store that initializes it.
  StoreInst *InitializerStore = nullptr;
  size_t Stores = 0;
  ParallelRegion *PrevStoreRegion = nullptr;

  // Find out the uses to fix first as fixing them invalidates the iterator.
  for (Instruction::use_iterator UI = Def->use_begin(), UE = Def->use_end();
       UI != UE; ++UI) {

    llvm::Instruction *User = cast<Instruction>(UI->getUser());
    if (User == NULL)
      continue;

    ParallelRegion *PRegion = regionOfBlock(User->getParent());
    // If this region is peeled to figure out the barrier entry,
    // we should disable rematerialization as it doesn't data flow
    // analyze the local_id_x access, which is set to 1 after the
    // peeled work-item.
    if (PRegion != nullptr && RegionEntryCounts[PRegion->entryBB()] > 1)
      RematCandidate = false;

    if (StoreInst *ST = dyn_cast<StoreInst>(User)) {
      if (!isa<UndefValue>(ST->getValueOperand())) {
        Stores++;
        if (Stores == 1) {
          InitializerStore = ST;
        } else {
          InitializerStore = nullptr;
          RematCandidate = false;
#ifdef DEBUG_WORK_ITEM_LOOPS
          std::cerr << "#### Multiple stores\n";
          User->dump();
#endif
        }
        if (PrevStoreRegion == nullptr) {
          PrevStoreRegion = PRegion;
        } else if (PrevStoreRegion != PRegion) {
          RematCandidate = false;
#ifdef DEBUG_WORK_ITEM_LOOPS
          std::cerr << "#### Stores from multiple regions\n";
          User->dump();
#endif
        }
        if (LI.getLoopFor(ST->getParent()) != nullptr) {
          RematCandidate = false;
#ifdef DEBUG_WORK_ITEM_LOOPS
          std::cerr << "#### Stores from inside a loop\n";
          User->dump();
#endif
        }
      }
    }

    // If the user is in a block that doesn't belong to a region, the variable
    // itself must be a "work group variable", that is, not dependent on the
    // work item. Most likely an iteration variable of a for loop with a
    // barrier.
    if (PRegion == nullptr) {
      LLVM_DEBUG(dbgs() << "User in a pure uniform block?\n");
      LLVM_DEBUG(User->dump());
      continue;
    }

    if (isa<CallInst>(User)) {
      if (!User->isLifetimeStartOrEnd()) {
        RematCandidate = false;
        LLVM_DEBUG(dbgs() << "Using in an unknown call\n");
        LLVM_DEBUG(User->dump());
      }
    } else if (llvm::AllocaInst *Alloca = dyn_cast_or_null<AllocaInst>(Def)) {
      if (!isa<StoreInst>(User) && !isa<LoadInst>(User)) {
        RematCandidate = false;
        LLVM_DEBUG(dbgs() << "Taking address of the alloca?\n");
        LLVM_DEBUG(User->dump());
      } else {
        // If we perform reinterpret casts, let's not rematerialize as it might
        // require to store the value temporarily to stack.
        if ((isa<LoadInst>(User) &&
             User->getType() != Alloca->getAllocatedType()) ||
            (isa<StoreInst>(User) &&
             User->getOperand(0)->getType() != Alloca->getAllocatedType())) {
          LLVM_DEBUG(dbgs() << "Found a user with a different pointee type\n");
          LLVM_DEBUG(User->dump());
          LLVM_DEBUG(Def->dump());
          RematCandidate = false;
        }
      }
    }

    Uses.push_back(User);
  }

  llvm::AllocaInst *ContextArrayAlloca = nullptr;
  bool PaddingAdded = false;

  for (Instruction *UserI : Uses) {
    Instruction *ContextRestoreLocation = UserI;

    PHINode* Phi = dyn_cast<PHINode>(UserI);
    if (Phi != NULL) {
      // TODO: This is now obsolete. For source input we work on unoptimized
      // clang output and for SPIR-V we break down the PHIs.

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
      LLVM_DEBUG(dbgs() << "Adding context restore code before PHI\n");
      LLVM_DEBUG(UserI->dump());
      LLVM_DEBUG(dbgs() << "In BB:\n");
      LLVM_DEBUG(UserI->getParent()->dump());

      BasicBlock *IncomingBB = NULL;
      for (unsigned Incoming = 0; Incoming < Phi->getNumIncomingValues();
           ++Incoming) {
        Value *Val = Phi->getIncomingValue(Incoming);
        BasicBlock *BB = Phi->getIncomingBlock(Incoming);
        if (Val == Def)
          IncomingBB = BB;
      }
      assert(IncomingBB != NULL);
      ContextRestoreLocation = IncomingBB->getTerminator();
    }

    llvm::Value *RematerializedValue = nullptr;
    if (RematCandidate) {
      if (isa<AllocaInst>(Def))
        RematerializedValue = tryToRematerialize(
            ContextRestoreLocation, InitializerStore->getValueOperand(),
            Def->getName().str());
      else
        RematerializedValue = tryToRematerialize(ContextRestoreLocation, Def,
                                                 Def->getName().str());
    }

    if (RematerializedValue != nullptr) {
      LLVM_DEBUG(dbgs() << "Successful rematerialization:\n");
      LLVM_DEBUG(RematerializedValue->dump());

      if (isa<AllocaInst>(Def)) {
        if (StoreInst *Store = dyn_cast<StoreInst>(UserI)) {
          // The original store could be left intact, but then we'd need to
          // figure out the materialization-ability beforehand.
          Store->setOperand(0, RematerializedValue);
        } else if (LoadInst *Load = dyn_cast<LoadInst>(UserI)) {
          // We can get rid of the alloca load altogether and use the
          // rematerialized value directly.
          UserI->replaceAllUsesWith(RematerializedValue);
          /// Kuten ongelmatapauksessa, allocan loadin tulosta käytetään
          /// toisessa parallel regionissa. Tässä oletetaan, että loadin
          /// tulokset myös samassa PR:ssä.
          LLVM_DEBUG(dbgs() << "Alloca load was converted to a remat value:\n");
          LLVM_DEBUG(UserI->dump());
          LLVM_DEBUG(RematerializedValue->dump());
        } else if (UserI->isLifetimeStartOrEnd()) {
          // We can leave the original lifetime marker for the alloca as is.
        } else {
          llvm_unreachable("Unexpected alloca usage.");
        }
      } else {
        UserI->replaceUsesOfWith(Def, RematerializedValue);
        LLVM_DEBUG(dbgs() << "The user was converted to a remat value:\n");
        LLVM_DEBUG(UserI->dump());
      }
    } else {
      // Unable to rematerialize the value.
      // Allocate a context data array for the variable.
      if (ContextArrayAlloca == nullptr) {
        ContextArrayAlloca = getContextArray(Def, PaddingAdded);
        addContextSave(Def, ContextArrayAlloca);
      }

      llvm::Value *ContextArrayLoad = addContextRestore(
          UserI, ContextArrayAlloca, Def->getType(), PaddingAdded,
          ContextRestoreLocation, isa<AllocaInst>(Def));

      UserI->replaceUsesOfWith(Def, ContextArrayLoad);

      LLVM_DEBUG(dbgs() << "the user was converted to a context load:\n");
      LLVM_DEBUG(UserI->dump());
    }
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

  // Generated id loads should not be replicated as it leads to problems in
  // conditional branch case where the header node of the region is shared
  // across the peeled branches and thus the header node's ID loads might get
  // context saved which leads to egg-chicken problems.
  llvm::LoadInst *Load = dyn_cast<llvm::LoadInst>(Instr);
  if (Load != NULL && (Load->getPointerOperand() == LocalIdGlobals[0] ||
                       Load->getPointerOperand() == LocalIdGlobals[1] ||
                       Load->getPointerOperand() == LocalIdGlobals[2] ||
                       Load->getPointerOperand() == GlobalIdGlobals[0] ||
                       Load->getPointerOperand() == GlobalIdGlobals[1] ||
                       Load->getPointerOperand() == GlobalIdGlobals[2]))
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
    LLVM_DEBUG(dbgs() << "Based on VUA, not context saving:\n");
    LLVM_DEBUG(Instr->dump());
    return true;
  }

  return false;
}

/// Appends a local id loop incrementing basic block.
///
/// \param After the basic block which flows to the increment block.
/// \param Dim the local id dimension to increment.
/// \param Before the basic block before which to add the new one.
/// \param BBName name to give to the basic block.
llvm::BasicBlock *WorkitemLoopsImpl::appendIncBlock(llvm::BasicBlock *After,
                                                    int Dim,
                                                    llvm::BasicBlock *Before,
                                                    const std::string &BBName) {

  llvm::Value *LocalIdVar = LocalIdGlobals[Dim];
  llvm::GlobalVariable *GlobalIdVar = GlobalIdIterators[Dim];

  llvm::LLVMContext &C = After->getContext();

  llvm::BasicBlock *oldExit = After->getTerminator()->getSuccessor(0);
  assert (oldExit != NULL);

  llvm::BasicBlock *forIncBB =
    BasicBlock::Create(C, "pregion_for_inc", After->getParent());

  After->getTerminator()->replaceUsesOfWith(oldExit, forIncBB);

  IRBuilder<> builder(oldExit);

  builder.SetInsertPoint(forIncBB);
  // Create the iteration variable increment for both the local and global ids.
  builder.CreateStore(builder.CreateAdd(builder.CreateLoad(ST, LocalIdVar),
                                        ConstantInt::get(ST, 1)),
                      LocalIdVar);

  builder.CreateStore(builder.CreateAdd(builder.CreateLoad(ST, GlobalIdVar),
                                        ConstantInt::get(ST, 1)),
                      GlobalIdVar);

  builder.CreateBr(oldExit);

  return forIncBB;
}

// enable new pass manager infrastructure
llvm::PreservedAnalyses WorkitemLoops::run(llvm::Function &F,
                                           llvm::FunctionAnalysisManager &AM) {
  if (!isKernelToProcess(F))
    return llvm::PreservedAnalyses::all();

  WorkitemHandlerType WIH = AM.getResult<WorkitemHandlerChooser>(F).WIH;
  if (WIH != WorkitemHandlerType::LOOPS)
    return llvm::PreservedAnalyses::all();

  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
  auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);
  auto &VUA = AM.getResult<VariableUniformityAnalysis>(F);

  llvm::PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<VariableUniformityAnalysis>();
  PAChanged.preserve<WorkitemHandlerChooser>();

  WorkitemLoopsImpl WIL(DT, LI, PDT, VUA);
  // llvm::verifyFunction(F);

  return WIL.runOnFunction(F) ? PAChanged : PreservedAnalyses::all();
}

bool WorkitemLoops::canHandleKernel(llvm::Function &K,
                                    llvm::FunctionAnalysisManager &AM) {

  // Do not handle kernels with barriers inside loops which have early exits
  // or continues.
  // It would require additional complexity that is unlikely worth it since
  // the vectorizer won't produce efficient code for such loops anyhow.
  // Tested by tricky_for.cl.
  LoopInfo &LI = AM.getResult<llvm::LoopAnalysis>(K);
  for (auto L : LI) {
    if (!Barrier::isLoopWithBarrier(*L))
      continue;
    // More than one 'break' point. It would lead to a complex control flow
    // structure which likely ruins loopvec efficiency anyhow.
    if (L->getExitingBlock() == nullptr) {
      LLVM_DEBUG(
          dbgs() << "Multiple breaks inside a barrier loop, won't handle.\n");
      return false;
    }
  }

  return true;
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
