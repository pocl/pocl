// LLVM function pass to create loops that run all the work items
// in a work group while respecting barrier synchronization points.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen / Tampere University
//               2022-2024 Pekka Jääskeläinen / Intel Finland Oy
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

#define DEBUG_TYPE "workitem-loops"

#define PASS_NAME "workitemloops"
#define PASS_CLASS pocl::WorkitemLoops
#define PASS_DESC "Workitem loop generation pass"

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
  llvm::Value *getLocalIdInRegion(llvm::Instruction *Instr,
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

  VariableUniformityAnalysisResult &VUA;

  ParallelRegion::ParallelRegionVector OriginalParallelRegions;

  StrInstructionMap ContextArrays;

  // Temporary global_id_* iteration variables updated by the work-item
  // loops.
  std::array<llvm::GlobalVariable *, 3> GlobalIdIterators;

  bool processFunction(llvm::Function &F);

  void fixMultiRegionVariables(ParallelRegion *region);
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
  llvm::Instruction *
  addContextRestore(llvm::Value *Val, llvm::AllocaInst *AllocaI,
                    llvm::Type *LoadInstType, bool PaddingWasAdded,
                    llvm::Instruction *Before = nullptr, bool isAlloca = false);
  llvm::AllocaInst *getContextArray(llvm::Instruction *Inst,
                                    bool &PoclWrapperStructAdded);

  std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
  createLoopAround(ParallelRegion &Region, llvm::BasicBlock *EntryBB,
                   llvm::BasicBlock *ExitBB, int Dim,
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
};

bool WorkitemLoopsImpl::runOnFunction(Function &Func) {

  M = Func.getParent();
  F = &Func;
  Initialize(cast<Kernel>(&Func));

#ifdef DEBUG_WORK_ITEM_LOOPS
  std::cerr << "### Before WILoops:\n";
  Func.dump();
#endif

  GlobalIdIterators = {
      cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(0), ST)),
      cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(1), ST)),
      cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(2), ST))};

  TempInstructionIndex = 0;

  bool Changed = processFunction(Func);

  Changed |= handleLocalMemAllocas();

  Changed |= fixUndominatedVariableUses(DT, Func);

  ContextArrays.clear();
  TempInstructionIds.clear();

  releaseParallelRegions();
#ifdef DEBUG_WORK_ITEM_LOOPS
  std::cerr << "### After WILoops:\n";
  Func.dump();
#endif
  return Changed;
}

std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
WorkitemLoopsImpl::createLoopAround(ParallelRegion &Region,
                                    llvm::BasicBlock *EntryBB,
                                    llvm::BasicBlock *ExitBB, int Dim,
                                    llvm::Value *DynamicLocalSize) {
  Value *LocalIdVars[] = {LocalIdXGlobal, LocalIdYGlobal, LocalIdZGlobal};
  Value *LocalIdVar = LocalIdVars[Dim];

  size_t LocalSizes[] = {WGLocalSizeX, WGLocalSizeY, WGLocalSizeZ};
  size_t LocalSizeForDim = LocalSizes[Dim];
  Instruction *GlobalIdOrigin = getGlobalIdOrigin(Dim);

  llvm::BasicBlock *LoopBodyEntryBB = EntryBB;
  llvm::LLVMContext &C = LoopBodyEntryBB->getContext();
  llvm::Function *F = LoopBodyEntryBB->getParent();
  LoopBodyEntryBB->setName(std::string("pregion_for_entry.") + EntryBB->getName().str());

  assert (ExitBB->getTerminator()->getNumSuccessors() == 1);

  llvm::BasicBlock *OldExit = ExitBB->getTerminator()->getSuccessor(0);

  llvm::BasicBlock *ForInitBB =
      BasicBlock::Create(C, "pregion_for_init", F, LoopBodyEntryBB);

  llvm::BasicBlock *LoopEndBB =
      BasicBlock::Create(C, "pregion_for_end", F, ExitBB);

  llvm::BasicBlock *ForCondBB =
      BasicBlock::Create(C, "pregion_for_cond", F, ExitBB);

  DT.reset();
  DT.recalculate(*F);

  // Collect the basic blocks in the parallel region that dominate the
  // exit. These are used in determining whether load instructions may
  // be executed unconditionally in the parallel loop (see below).
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> DominatesExitBB;
  for (auto BB: Region) {
    if (DT.dominates(BB, ExitBB)) {
      DominatesExitBB.insert(BB);
    }
  }

  // For fixing the old edges jumping to the region to jump to the basic block
  // that starts the created loop. Back edges should still point to the old
  // basic block so we preserve the old loops. TODO: is this still needed with
  // the forced PR entry block?
  BasicBlockVector Preds;
  llvm::pred_iterator PI = llvm::pred_begin(EntryBB),
                      E = llvm::pred_end(EntryBB);

  for (; PI != E; ++PI)
    Preds.push_back(*PI);

  for (BasicBlockVector::iterator i = Preds.begin(); i != Preds.end(); ++i) {
    llvm::BasicBlock *BB = *i;
    // Do not fix loop edges inside the region. The loop is replicated as
    // a whole to the body of the WI-loop.
    if (DT.dominates(LoopBodyEntryBB, BB))
      continue;
    BB->getTerminator()->replaceUsesOfWith(LoopBodyEntryBB, ForInitBB);
  }

  IRBuilder<> Builder(ForInitBB);

  Builder.CreateStore(ConstantInt::get(ST, 0), LocalIdVar);

  // Initialize the global id counter with the base.
  GlobalVariable *GlobalId = GlobalIdIterators[Dim];
  Builder.CreateStore(GlobalIdOrigin, GlobalId);

  Builder.CreateBr(LoopBodyEntryBB);

  ExitBB->getTerminator()->replaceUsesOfWith(OldExit, ForCondBB);
  appendIncBlock(ExitBB, Dim);

  Builder.SetInsertPoint(ForCondBB);

  llvm::Value *CmpResult;
  if (!WGDynamicLocalSize) {
    CmpResult = Builder.CreateICmpULT(Builder.CreateLoad(ST, LocalIdVar),
                                      ConstantInt::get(ST, LocalSizeForDim));
  } else {
    GlobalVariable *LocalSizeGlobal = M->getGlobalVariable(LSIZE_G_NAME(Dim));
    if (LocalSizeGlobal == NULL)
      LocalSizeGlobal = new GlobalVariable(
          *M, ST, true, GlobalValue::CommonLinkage, NULL, LSIZE_G_NAME(Dim),
          NULL, GlobalValue::ThreadLocalMode::NotThreadLocal, 0, true);
    CmpResult = Builder.CreateICmpULT(Builder.CreateLoad(ST, LocalIdVar),
                                      Builder.CreateLoad(ST, LocalSizeGlobal));
  }

  Instruction *LoopBranch =
      Builder.CreateCondBr(CmpResult, LoopBodyEntryBB, LoopEndBB);

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
        [&DominatesExitBB](llvm::Instruction *Insn) -> bool {
      assert(Insn->mayReadFromMemory());
      // Checks that the instruction isn't in a conditional block.
      return DominatesExitBB.count(Insn->getParent());
    };

    Region.addParallelLoopMetadata(AccessGroupMD, IsLoadUnconditionallySafe);
  }

  Builder.SetInsertPoint(LoopEndBB);
  Builder.CreateBr(OldExit);

  return std::make_pair(ForInitBB, LoopEndBB);
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
  OriginalParallelRegions.clear();
}

bool WorkitemLoopsImpl::processFunction(Function &F) {

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  // Append 'dyn' or 'static' to the dot files to differentiate between the
  // dynamic WG one (produced for the binaries) and the specialized static one.
  std::string DotSuffix = WGDynamicLocalSize ? "_dyn" : "_static";
  dumpCFG(F, F.getName().str() + "_before_pregions" + DotSuffix + ".dot", nullptr, nullptr);
#endif

  K->getParallelRegions(LI, &OriginalParallelRegions);

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  dumpCFG(F, F.getName().str() + "_before_wiloops" + DotSuffix + ".dot", nullptr,
#endif

  IRBuilder<> builder(&*(F.getEntryBlock().getFirstInsertionPt()));
  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
           PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI)
    fixMultiRegionVariables(*PRI);

  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
           PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {

    ParallelRegion *PRegion = (*PRI);

    // The original predecessor nodes of which branches should be fixed
    // later on to jump to the looped region's start.
    BasicBlockVector Preds;
    llvm::pred_iterator PI = llvm::pred_begin(PRegion->entryBB()),
                        E = llvm::pred_end(PRegion->entryBB());
    for (; PI != E; ++PI) {
      llvm::BasicBlock *BB = *PI;
      if (DT.dominates(PRegion->entryBB(), BB) &&
          (regionOfBlock(PRegion->entryBB()) == regionOfBlock(BB)))
        continue;
      Preds.push_back(BB);
    }

    // The parallel WI-loop being constructed.
    std::pair<llvm::BasicBlock *, llvm::BasicBlock *> WILoop =
        std::make_pair(PRegion->entryBB(), PRegion->exitBB());

    for (size_t Dim = 0; Dim < 3; ++Dim) {
      WILoop = createLoopAround(*PRegion, WILoop.first, WILoop.second, Dim);
      // Ensure the global id for is initialized even for a 1-size dimension.
      getGlobalIdOrigin(Dim);
    }

    // Fix the predecessors to jump to the beginning of the new WI loop.
    for (BasicBlockVector::iterator i = Preds.begin(); i != Preds.end(); ++i) {
      llvm::BasicBlock *BB = *i;
      BB->getTerminator()->replaceUsesOfWith(PRegion->entryBB(), WILoop.first);
    }
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

llvm::Value *WorkitemLoopsImpl::getLocalIdInRegion(llvm::Instruction *Instr,
                                                   size_t Dim) {
  ParallelRegion *ParRegion = regionOfBlock(Instr->getParent());
  assert(ParRegion != nullptr);
  return ParRegion->getOrCreateIDLoad(LID_G_NAME(Dim));
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
    /* In case the context saved instruction was an alloca, we created a
       context array with pointed-to elements, and now want to return a
       pointer to the elements to emulate the original alloca. */
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

/// Adds context save/restore code for the value produced by the
/// given instruction.
///
/// \todo add only one restore per variable per region.
/// \todo add only one load of the id variables per region.
/// Could be done by having a context restore BB in the beginning of the
/// region and a context save BB at the end.
/// \todo ignore work group variables completely (the iteration variables)
/// The LLVM should optimize these away but it would improve
/// the readability of the output during debugging.
/// \todo rematerialize some values such as extended values of global
/// variables (especially global id which is computed from local id) or kernel
/// argument values instead of allocating stack space for them.
void WorkitemLoopsImpl::addContextSaveRestore(llvm::Instruction *Def) {

  // Allocate the context data array for the variable.
  bool PaddingAdded = false;
  llvm::AllocaInst *Alloca = getContextArray(Def, PaddingAdded);
  llvm::Instruction *TheStore = addContextSave(Def, Alloca);

  InstructionVec Uses;
  // Restore the produced variable before each use to ensure the correct
  // context copy is used.

#if 0
  // TODO: This is now obsolete:
  // We could add the restore only to other regions outside the variable
  // defining region and use the original variable in the defining region due
  // to the SSA virtual registers being unique. However, alloca variables can
  // be redefined also in the same region, thus we need to ensure the correct
  // alloca context position is written, not the original unreplicated one.
  // These variables can be generated by volatile variables, private arrays,
  // and due to the PHIs to allocas pass.
#endif

  // Find out the uses to fix first as fixing them invalidates the iterator.
  for (Instruction::use_iterator UI = Def->use_begin(), UE = Def->use_end();
       UI != UE; ++UI) {
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
        if (Val == Def)
          IncomingBB = BB;
      }
      assert(IncomingBB != NULL);
      ContextRestoreLocation = IncomingBB->getTerminator();
    }
    llvm::Value *LoadedValue =
        addContextRestore(UserI, Alloca, Def->getType(), PaddingAdded,
                          ContextRestoreLocation, isa<AllocaInst>(Def));
    UserI->replaceUsesOfWith(Def, LoadedValue);
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

  // _local_id or _global_id loads should not be replicated as it leads to
  // problems in conditional branch case where the header node of the region is
  // shared across the branches and thus the header node's ID loads might get
  // context saved which leads to egg-chicken problems.

  llvm::LoadInst *Load = dyn_cast<llvm::LoadInst>(Instr);
  if (Load != NULL &&
      (Load->getPointerOperand() == LocalIdZGlobal ||
       Load->getPointerOperand() == LocalIdYGlobal ||
       Load->getPointerOperand() == LocalIdXGlobal ||
       Load->getPointerOperand()->getName().starts_with("_global_id")))
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

  llvm::Value *LocalIdVars[] = {LocalIdXGlobal, LocalIdYGlobal, LocalIdZGlobal};
  llvm::Value *LocalIdVar = LocalIdVars[Dim];
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
#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "### multiple breaks inside a barrier loop, won't handle.\n";
#endif
      return false;
    }
  }

  // Do not handle kernels with non-uniform predicates. These used to be
  // handled by "peeling" the first iteration of the work-item loop to
  // check if the barrier is taken or not, but it complicates the parallel
  // region formation and control flow construction quite a bit without
  // known benchmark benefits. The produced peeled loops are also not easily
  // vectorizable, thus the performance won't be good anyhow for SIMD. For
  // VLIW/ILP it might be useful though, but it still doesn't justify the
  // complexity add.

  // The tricky part here is detecting cases where we have barriers inside
  // uniform ifs inside for-loops of which iteration counts are not known.
  // For the purpose of this check, we treat them as always taken for-loops,
  // relying on the "all or none" barrier semantics. The current checks
  // is "robustness first": It includes all barriers which are made
  // conditional with something else than the loop condition. An optimization
  // would be to allow detected uniform conditions and isolate the if part
  // to a uniform region with a separate parallel region in both branches.

  llvm::PostDominatorTree &PDT = AM.getResult<PostDominatorTreeAnalysis>(K);
  for (Function::iterator FI = K.begin(), FE = K.end(); FI != FE; ++FI) {
    BasicBlock *BB = &*FI;
    if (!Barrier::hasBarrier(BB)) continue;

    // Unconditional barrier for this purpose postdominates the entry node or
    // the loop header that it's in.
    Loop *L = LI.getLoopFor(BB);;
    BasicBlock *PostDomBlock =
      L == nullptr ? &K.getEntryBlock() : L->getHeader();
    if (PDT.dominates(BB, PostDomBlock)) continue;

#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### Detected a conditional barrier not currently supported by WILoops:\n";
    BB->dump();
#endif
    return false;
  }

  return true;
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
