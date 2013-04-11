//===- WIVectorize.cpp - A Work Item Vectorizer -------------------------===//
//
// This code has been adapted from BBVectorize of the LLVM project.
// The original file comment:
//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
// This file implements a basic-block vectorization pass. The algorithm was
// inspired by that used by the Vienna MAP Vectorizor by Franchetti and Kral,
// et al. It works by looking for chains of pairable operations and then
// pairing them.
//
//===----------------------------------------------------------------------===//
// 
// WIVectorize:
// 
// Additional options are provided to vectorize only candidate from differnt
// work items according to metadata provided by 'pocl' frontend 
// (launchpad.net/pocl). 
//
// Additional option is also available to vectorize loads and stores only.
// Still work in progress by vladimir guzma [at] tut fi.
//
//===----------------------------------------------------------------------===//

#define WIV_NAME "wi-vectorize"
#define DEBUG_TYPE WIV_NAME
#include "config.h"
#ifdef LLVM_3_1
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/TypeBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/Type.h"
#include "llvm/Metadata.h"
#elif defined LLVM_3_2
#include "llvm/IRBuilder.h"
#include "llvm/TypeBuilder.h"
#include "llvm/DataLayout.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/Type.h"
#include "llvm/Metadata.h"
#include "llvm/TargetTransformInfo.h"
#else
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#endif
#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Transforms/Vectorize.h"
#include <algorithm>
#include <map>
#include <iostream>
using namespace llvm;

static cl::opt<bool>
IgnoreTargetInfo("wi-vectorize-ignore-target-info",  cl::init(true),
  cl::Hidden, cl::desc("Ignore target information"));

static cl::opt<unsigned>
ReqChainDepth("wi-vectorize-req-chain-depth", cl::init(3), cl::Hidden,
  cl::desc("The required chain depth for vectorization"));

static cl::opt<unsigned>
VectorWidth("wi-vectorize-vector-width", cl::init(8), cl::Hidden,
  cl::desc("The width of the machine vector in words."));

static cl::opt<bool>
NoMath("wi-vectorize-no-math", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize floating-point math intrinsics"));

static cl::opt<bool>
NoFMA("wi-vectorize-no-fma", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize the fused-multiply-add intrinsic"));

static cl::opt<bool>
NoMemOps("wi-vectorize-no-mem-ops", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize loads and stores"));

static cl::opt<bool>
AlignedOnly("wi-vectorize-aligned-only", cl::init(false), cl::Hidden,
  cl::desc("Only generate aligned loads and stores"));

static cl::opt<bool>
MemOpsOnly("wi-vectorize-mem-ops-only", cl::init(false), cl::Hidden,
  cl::desc("Try to vectorize loads and stores only"));

static cl::opt<bool>
NoFP("wi-vectorize-no-fp", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize floating-point operations"));

static cl::opt<bool>
NoCMP("wi-vectorize-no-cmp", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize comparison operations"));

static cl::opt<bool>
NoCount("wi-vectorize-no-counters", cl::init(false), cl::Hidden,
  cl::desc("Forbid vectorization based no loop counter "
          "arithmetic"));
static cl::opt<bool>
NoGEP("wi-vectorize-no-GEP", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize getelementpointer operations"));

#ifndef NDEBUG
static cl::opt<bool>
DebugInstructionExamination("wi-vectorize-debug-instruction-examination",
  cl::init(false), cl::Hidden,
  cl::desc("When debugging is enabled, output information on the"
           " instruction-examination process"));
static cl::opt<bool>
DebugCandidateSelection("wi-vectorize-debug-candidate-selection",
  cl::init(false), cl::Hidden,
  cl::desc("When debugging is enabled, output information on the"
           " candidate-selection process"));
static cl::opt<bool>
DebugPairSelection("wi-vectorize-debug-pair-selection",
  cl::init(false), cl::Hidden,
  cl::desc("When debugging is enabled, output information on the"
           " pair-selection process"));
static cl::opt<bool>
DebugCycleCheck("wi-vectorize-debug-cycle-check",
  cl::init(false), cl::Hidden,
  cl::desc("When debugging is enabled, output information on the"
           " cycle-checking process"));
#endif

STATISTIC(NumFusedOps, "Number of operations fused by wi-vectorize");

namespace llvm {
    FunctionPass* createWIVectorizePass();    
}
namespace {
  struct WIVectorize : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    WIVectorize() : FunctionPass(ID) {}

    typedef std::pair<Value *, Value *> ValuePair;
    typedef std::pair<ValuePair, size_t> ValuePairWithDepth;
    typedef std::pair<ValuePair, ValuePair> VPPair; // A ValuePair pair
    typedef std::pair<std::multimap<Value *, Value *>::iterator,
              std::multimap<Value *, Value *>::iterator> VPIteratorPair;
    typedef std::pair<std::multimap<ValuePair, ValuePair>::iterator,
              std::multimap<ValuePair, ValuePair>::iterator>
                VPPIteratorPair;
    typedef std::vector<Value *> ValueVector;
    typedef DenseMap<Value*, ValueVector*> ValueVectorMap;

    AliasAnalysis *AA;
    ScalarEvolution *SE;
#ifdef LLVM_3_1
    TargetData *TD;
#elif defined LLVM_3_2
    DataLayout *TD;
    TargetTransformInfo *TTI;
    const VectorTargetTransformInfo *VTTI;    
#else
    DataLayout *TD;
    TargetTransformInfo *TTI;
    const TargetTransformInfo *VTTI;
#endif
    DenseMap<Value*, Value*> storedSources;
    DenseMap<std::pair<int,int>, ValueVector*> stridedOps;    
    std::multimap<Value*, Value*> flippedStoredSources;
    // FIXME: const correct?

    bool vectorizePairs(BasicBlock &BB);
    
    bool vectorizePhiNodes(BasicBlock &BB);
    
    bool vectorizeAllocas(BasicBlock& BB);
    
    void replaceUses(BasicBlock& BB,
                     AllocaInst& oldAlloca, 
                     AllocaInst& newAlloca, 
                     int indx);
    
    Type* newAllocaType(Type* start, unsigned int width);
    
    bool removeDuplicates(BasicBlock &BB);

    void dropUnused(BasicBlock& BB);
    
    bool getCandidatePairs(BasicBlock &BB,
                       BasicBlock::iterator &Start,
                       std::multimap<Value *, Value *> &CandidatePairs,
                       std::vector<Value *> &PairableInsts);
    
    bool getCandidateAllocas(BasicBlock &BB,
                       std::multimap<int, ValueVector *>& candidateAllocas);

    void computeConnectedPairs(std::multimap<Value *, Value *> &CandidatePairs,
                       std::vector<Value *> &PairableInsts,
                       std::multimap<ValuePair, ValuePair> &ConnectedPairs);

    void buildDepMap(BasicBlock &BB,
                       std::multimap<Value *, Value *> &CandidatePairs,
                       std::vector<Value *> &PairableInsts,
                       DenseSet<ValuePair> &PairableInstUsers);

    void choosePairs(std::multimap<Value *, Value *> &CandidatePairs,
                        std::vector<Value *> &PairableInsts,
                        std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                        DenseSet<ValuePair> &PairableInstUsers,
                        DenseMap<Value *, Value *>& ChosenPairs);

    void fuseChosenPairs(BasicBlock &BB,
                     std::vector<Value *> &PairableInsts,
                     DenseMap<Value *, Value *>& ChosenPairs);
    
    bool isInstVectorizable(Instruction *I, bool &IsSimpleLoadStore);

    bool areInstsCompatible(Instruction *I, Instruction *J,
                       bool IsSimpleLoadStore);

    bool areInstsCompatibleFromDifferentWi(Instruction *I, Instruction *J);

    bool trackUsesOfI(DenseSet<Value *> &Users,
                      AliasSetTracker &WriteSet, Instruction *I,
                      Instruction *J, bool UpdateUsers = true,
                      std::multimap<Value *, Value *> *LoadMoveSet = 0);

    void computePairsConnectedTo(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      ValuePair P);

    bool pairsConflict(ValuePair P, ValuePair Q,
                 DenseSet<ValuePair> &PairableInstUsers,
                 std::multimap<ValuePair, ValuePair> *PairableInstUserMap = 0);

    bool pairWillFormCycle(ValuePair P,
                       std::multimap<ValuePair, ValuePair> &PairableInstUsers,
                       DenseSet<ValuePair> &CurrentPairs);

    void pruneTreeFor(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      DenseSet<ValuePair> &PairableInstUsers,
                      std::multimap<ValuePair, ValuePair> &PairableInstUserMap,
                      DenseMap<Value *, Value *> &ChosenPairs,
                      DenseMap<ValuePair, size_t> &Tree,
                      DenseSet<ValuePair> &PrunedTree, ValuePair J,
                      bool UseCycleCheck);

    void buildInitialTreeFor(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      DenseSet<ValuePair> &PairableInstUsers,
                      DenseMap<Value *, Value *> &ChosenPairs,
                      DenseMap<ValuePair, size_t> &Tree, ValuePair J);

    void findBestTreeFor(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      DenseSet<ValuePair> &PairableInstUsers,
                      std::multimap<ValuePair, ValuePair> &PairableInstUserMap,
                      DenseMap<Value *, Value *> &ChosenPairs,
                      DenseSet<ValuePair> &BestTree, size_t &BestMaxDepth,
                      size_t &BestEffSize, VPIteratorPair ChoiceRange,
                      bool UseCycleCheck);

    Value *getReplacementPointerInput(LLVMContext& Context, Instruction *I,
                     Instruction *J, unsigned o, bool FlipMemInputs);

    void fillNewShuffleMask(LLVMContext& Context, Instruction *J,
                     unsigned NumElem, unsigned MaskOffset, unsigned NumInElem,
                     unsigned IdxOffset, std::vector<Constant*> &Mask);

    Value *getReplacementShuffleMask(LLVMContext& Context, Instruction *I,
                     Instruction *J);

    Value *getReplacementInput(LLVMContext& Context, Instruction *I,
                     Instruction *J, unsigned o, bool FlipMemInputs);

    Value* CommonShuffleSource(Instruction *I, Instruction *J);

    void getReplacementInputsForPair(LLVMContext& Context, Instruction *I,
                     Instruction *J, SmallVector<Value *, 3> &ReplacedOperands,
                     bool FlipMemInputs);
    
    void replaceOutputsOfPair(LLVMContext& Context, Instruction *I,
                     Instruction *J, Instruction *K,
                     Instruction *&InsertionPt, Instruction *&K1,
                     Instruction *&K2, bool FlipMemInputs);

    void collectPairLoadMoveSet(BasicBlock &BB,
                     DenseMap<Value *, Value *> &ChosenPairs,
                     std::multimap<Value *, Value *> &LoadMoveSet,
                     Instruction *I);

    void collectLoadMoveSet(BasicBlock &BB,
                     std::vector<Value *> &PairableInsts,
                     DenseMap<Value *, Value *> &ChosenPairs,
                     std::multimap<Value *, Value *> &LoadMoveSet);

    void moveUsesOfIAfterJ(BasicBlock &BB,
                     std::multimap<Value *, Value *> &LoadMoveSet,
                     Instruction *&InsertionPt,
                     Instruction *I, Instruction *J);
    
    void collectPtrInfo(std::vector<Value *> &PairableInsts,
                        DenseMap<Value *, Value *> &ChosenPairs,
                        DenseSet<Value *> &LowPtrInsts);    
    
    bool doInitialization(Module& /*m*/) {
      return false;
    }
    bool doFinalization(Module& /*m*/) {
      return false;
    }
    virtual bool runOnFunction(Function &Func) {
        
      AA = &getAnalysis<AliasAnalysis>();
      SE = &getAnalysis<ScalarEvolution>();
#ifdef LLVM_3_1
      TD = getAnalysisIfAvailable<TargetData>();
#elif defined LLVM_3_2
      TD = getAnalysisIfAvailable<DataLayout>();
      TTI = IgnoreTargetInfo ? 0 :
        getAnalysisIfAvailable<TargetTransformInfo>();
      VTTI = TTI ? TTI->getVectorTargetTransformInfo() : 0;        
#else
      TD = getAnalysisIfAvailable<DataLayout>();
      TTI = IgnoreTargetInfo ? 0 :
        getAnalysisIfAvailable<TargetTransformInfo>();
      VTTI = TTI;
#endif
      
      bool changed = false;      
      for (Function::iterator i = Func.begin();
         i != Func.end(); i++) {
        changed |=runOnBasicBlock(*i);
      }
      return changed;
    }
    
    virtual bool runOnBasicBlock(BasicBlock &BB) {

      bool changed = false;
      
      // First try to create vectors of all allocas, if there are any
      changed |= vectorizeAllocas(BB);      
      // Iterate a sufficient number of times to merge types of size 1 bit,
      // then 2 bits, then 4, etc. up to half of the target vector width of the
      // target vector register.
      bool vectorizeTwice = false;
      

      // There are 3 possible cases of vectorization in regards to memory
      // operations:
      // 1: Explicitly forbid vectorization of mem ops (NoMemOps)
      // 2: Allow only vectorization of mem ops (MemOpsOnly)
      // 3: Vectorize mem ops as well as everything else
      // In cases 1 and 2, following test makes sure vectorization is
      // run only once.
      // For case 3, we first run vectorization of memory operations only
      // and then we run vectorization of everything else. In between
      // we remove unused operations, which are typicaly memory
      // access computations that are not needed anymore and their vectorization
      // is waste of resources. Instruction combiner is not able to get rid
      // of those on it's own once they are in vectors.
      
      // Store original values of two variables. They can be changed bellow
      // but have to be restored before calling this for next BB.
      bool originalMemOpsOnly = MemOpsOnly;
      bool originalNoMemOps = NoMemOps;
      if (!MemOpsOnly && !NoMemOps) {
          MemOpsOnly = true;
          vectorizeTwice = true;
      }
#if 0      
#ifdef LLVM_3_3
      if (TTI) {
          std::cerr << " settign new vector width" << std::endl;
          unsigned WidestRegister = TTI->getRegisterBitWidth(true);      
          VectorWidth = WidestRegister/32;
          std::cerr << VectorWidth << std::endl;
      }
#endif      
#endif

      for (unsigned v = 2, n = 1; v <= VectorWidth;
          v *= 2, ++n) {
          DEBUG(dbgs() << "WIV: fusing memm only in loop #" << n << 
              " for " << BB.getName() << " in " <<
              BB.getParent()->getName() << "...\n");
          if (vectorizePairs(BB)) {
            dropUnused(BB);
            changed = true;
          }
          else
            break;
      }
      if (vectorizeTwice) {
          MemOpsOnly = false;
          NoMemOps = true;
          for (unsigned v = 2, n = 1; v <= VectorWidth;
               v *= 2, ++n) {
              DEBUG(dbgs() << "WIV: fusing loop #" << n <<
                    " for " << BB.getName() << " in " <<
                    BB.getParent()->getName() << "...\n");
              if (vectorizePairs(BB)) {
                  dropUnused(BB);                  
                  changed = true;
              }
              else
                  break;
          }
      } 

      if (changed) {
        vectorizePhiNodes(BB);
        removeDuplicates(BB);      
      }

      DEBUG(dbgs() << "WIV: done!\n");
      MemOpsOnly = originalMemOpsOnly;
      NoMemOps = originalNoMemOps;
      return changed;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      FunctionPass::getAnalysisUsage(AU);
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<ScalarEvolution>();
      AU.addPreserved<AliasAnalysis>();
      AU.addPreserved<ScalarEvolution>();
      AU.setPreservesCFG();
    }
    // This returns the vector type that holds a pair of the provided type.
    // If the provided type is already a vector, then its length is doubled.
    static inline VectorType *getVecTypeForVector(Type *ElemTy) {
      if (VectorType *VTy = dyn_cast<VectorType>(ElemTy)) {
        unsigned numElem = VTy->getNumElements();
        return VectorType::get(ElemTy->getScalarType(), numElem*VectorWidth);
      } else {
        return VectorType::get(ElemTy->getScalarType(), VectorWidth);
          
      }

      return VectorType::get(ElemTy, 2);
    }
    // This returns the vector type that holds a pair of the provided type.
    // If the provided type is already a vector, then its length is doubled.
    static inline VectorType *getVecTypeForPair(Type *ElemTy, Type *Elem2Ty) {
      assert(ElemTy->getScalarType() == Elem2Ty->getScalarType() &&
             "Cannot form vector from incompatible scalar types");
      Type *STy = ElemTy->getScalarType();

      unsigned numElem;
      if (VectorType *VTy = dyn_cast<VectorType>(ElemTy)) {
        numElem = VTy->getNumElements();
      } else {
        numElem = 1;
      }

      if (VectorType *VTy = dyn_cast<VectorType>(Elem2Ty)) {
        numElem += VTy->getNumElements();
      } else {
        numElem += 1;
      }

      return VectorType::get(STy, numElem);
    }
    
    std::string getReplacementName(Instruction *I, bool IsInput, unsigned o,
                        unsigned n = 0) {
        if (!I->hasName())
        return "";

        return (I->getName() + (IsInput ? ".v.i" : ".v.r") + utostr(o) +
                (n > 0 ? "." + utostr(n) : "")).str();
    }

    // Returns the weight associated with the provided value. A chain of
    // candidate pairs has a length given by the sum of the weights of its
    // members (one weight per pair; the weight of each member of the pair
    // is assumed to be the same). This length is then compared to the
    // chain-length threshold to determine if a given chain is significant
    // enough to be vectorized. The length is also used in comparing
    // candidate chains where longer chains are considered to be better.
    // Note: when this function returns 0, the resulting instructions are
    // not actually fused.
    static inline size_t getDepthFactor(Value *V) {
      // InsertElement and ExtractElement have a depth factor of zero. This is
      // for two reasons: First, they cannot be usefully fused. Second, because
      // the pass generates a lot of these, they can confuse the simple metric
      // used to compare the trees in the next iteration. Thus, giving them a
      // weight of zero allows the pass to essentially ignore them in
      // subsequent iterations when looking for vectorization opportunities
      // while still tracking dependency chains that flow through those
      // instructions.
      if (isa<InsertElementInst>(V) || isa<ExtractElementInst>(V))
        return 0;

      // Give a load or store half of the required depth so that load/store
      // pairs will vectorize.
      if ((isa<LoadInst>(V) || isa<StoreInst>(V)))
        return ReqChainDepth;
        
      return 1;
    }
    // Returns the cost of the provided instruction using VTTI.
    // This does not handle loads and stores.
    unsigned getInstrCost(unsigned Opcode, Type *T1, Type *T2) {
#ifdef LLVM_3_1
        return 1;
#else
      switch (Opcode) {
      default: break;
      case Instruction::GetElementPtr:
        // We mark this instruction as zero-cost because scalar GEPs are usually
        // lowered to the intruction addressing mode. At the moment we don't
        // generate vector GEPs.
        return 0;
      case Instruction::Br:
        return VTTI->getCFInstrCost(Opcode);
      case Instruction::PHI:
        return 0;
      case Instruction::Add:
      case Instruction::FAdd:
      case Instruction::Sub:
      case Instruction::FSub:
      case Instruction::Mul:
      case Instruction::FMul:
      case Instruction::UDiv:
      case Instruction::SDiv:
      case Instruction::FDiv:
      case Instruction::URem:
      case Instruction::SRem:
      case Instruction::FRem:
      case Instruction::Shl:
      case Instruction::LShr:
      case Instruction::AShr:
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor:
        return VTTI->getArithmeticInstrCost(Opcode, T1);
      case Instruction::Select:
      case Instruction::ICmp:
      case Instruction::FCmp:
        return VTTI->getCmpSelInstrCost(Opcode, T1, T2);
      case Instruction::ZExt:
      case Instruction::SExt:
      case Instruction::FPToUI:
      case Instruction::FPToSI:
      case Instruction::FPExt:
      case Instruction::PtrToInt:
      case Instruction::IntToPtr:
      case Instruction::SIToFP:
      case Instruction::UIToFP:
      case Instruction::Trunc:
      case Instruction::FPTrunc:
      case Instruction::BitCast:
      case Instruction::ShuffleVector:
        return VTTI->getCastInstrCost(Opcode, T1, T2);
      }
      return 1;
#endif      
    }     
    // This determines the relative offset of two loads or stores, returning
    // true if the offset could be determined to be some constant value.
    // For example, if OffsetInElmts == 1, then J accesses the memory directly
    // after I; if OffsetInElmts == -1 then I accesses the memory
    // directly after J. This function assumes that both instructions
    // have the same type.
    bool getPairPtrInfo(Instruction *I, Instruction *J,
        Value *&IPtr, Value *&JPtr, unsigned &IAlignment, unsigned &JAlignment,
        unsigned &IAddressSpace, unsigned &JAddressSpace,
        int64_t &OffsetInElmts) {
      OffsetInElmts = 0;
      if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
        LoadInst *LJ = cast<LoadInst>(J);
        IPtr = LI->getPointerOperand();
        JPtr = LJ->getPointerOperand();
        IAlignment = LI->getAlignment();
        JAlignment = LJ->getAlignment();
        IAddressSpace = LI->getPointerAddressSpace();
        JAddressSpace = LJ->getPointerAddressSpace();        
      } else if (isa<GetElementPtrInst>(I)) {
        Instruction::op_iterator it = cast<GetElementPtrInst>(I)->idx_begin();
        IPtr = *it;
        Instruction::op_iterator jt = cast<GetElementPtrInst>(J)->idx_begin();
        JPtr = *jt;
        if (!IPtr || !JPtr)
            return false;
        IAlignment = 0;
        JAlignment = 0;        
      } else {
        StoreInst *SI = cast<StoreInst>(I), *SJ = cast<StoreInst>(J);
        IPtr = SI->getPointerOperand();
        JPtr = SJ->getPointerOperand();
        IAlignment = SI->getAlignment();
        JAlignment = SJ->getAlignment();
        IAddressSpace = SI->getPointerAddressSpace();
        JAddressSpace = SJ->getPointerAddressSpace();     
      }
      if ((isa<GetElementPtrInst>(I) && !SE->isSCEVable(IPtr->getType())) 
          || (isa<GetElementPtrInst>(J) && !SE->isSCEVable(JPtr->getType()))) {
          // Asume, the getelementpointer is already vector, so the pointer
          // operand is also the vector and LLVM scalar evaluation can
          // not understand it.
          OffsetInElmts = 2;
          return true;
      }
      const SCEV *IPtrSCEV = SE->getSCEV(IPtr);
      const SCEV *JPtrSCEV = SE->getSCEV(JPtr);

      // If this is a trivial offset, then we'll get something like
      // 1*sizeof(type). With target data, which we need anyway, this will get
      // constant folded into a number.
      const SCEV *OffsetSCEV = SE->getMinusSCEV(JPtrSCEV, IPtrSCEV);
      if (const SCEVConstant *ConstOffSCEV =
            dyn_cast<SCEVConstant>(OffsetSCEV)) {
        ConstantInt *IntOff = ConstOffSCEV->getValue();
        int64_t Offset = IntOff->getSExtValue();
        if (isa<GetElementPtrInst>(I)) {
            OffsetInElmts = Offset;
            return (abs64(Offset)) > 1;
        }
        Type *VTy = cast<PointerType>(IPtr->getType())->getElementType();
        int64_t VTyTSS = (int64_t) TD->getTypeStoreSize(VTy);

        Type *VTy2 = cast<PointerType>(JPtr->getType())->getElementType();
        if (VTy != VTy2 && Offset < 0) {
          int64_t VTy2TSS = (int64_t) TD->getTypeStoreSize(VTy2);
          OffsetInElmts = Offset/VTy2TSS;
          return (abs64(Offset) % VTy2TSS) == 0;
        }
        OffsetInElmts = Offset/VTyTSS;
        
        return (abs64(Offset) % VTyTSS) == 0;
      }
      return false;
    }

    // Returns true if the provided CallInst represents an intrinsic that can
    // be vectorized.
    bool isVectorizableIntrinsic(CallInst* I) {
      Function *F = I->getCalledFunction();
      if (!F) return false;

      unsigned IID = F->getIntrinsicID();
      if (!IID) return false;

      switch(IID) {
      default:
        return false;
      case Intrinsic::sqrt:
      case Intrinsic::powi:
      case Intrinsic::sin:
      case Intrinsic::cos:
      case Intrinsic::log:
      case Intrinsic::log2:
      case Intrinsic::log10:
      case Intrinsic::exp:
      case Intrinsic::exp2:
      case Intrinsic::pow:
        return !NoMath;
      case Intrinsic::fma:
        return !NoFMA;
      }
    }

    // Returns true if J is the second element in some pair referenced by
    // some multimap pair iterator pair.
    template <typename V>
    bool isSecondInIteratorPair(V J, std::pair<
           typename std::multimap<V, V>::iterator,
           typename std::multimap<V, V>::iterator> PairRange) {
      for (typename std::multimap<V, V>::iterator K = PairRange.first;
           K != PairRange.second; ++K)
        if (K->second == J) return true;

      return false;
    }
  };
  // In some cases, instructions did not get combined correctly by previous passes.
  // For example with large number of replicated work items, scalar load of constant
  // happened for first work item and then exactly same load in 15 and 30th work item. 
  // The work items in between reused the previous value.
  // Also, the vectorization vectorization leads to situations where scalar value
  // needs to be replicated to create vector, however, separate vectors were
  // created each time the value was to be used.
  // This fixes that by search for exactly same Instructions, with same type
  // and exactly same parameters and removing later one of them, replacing
  // all uses with former.
    bool WIVectorize::removeDuplicates(BasicBlock &BB) {
        BasicBlock::iterator Start = BB.getFirstInsertionPt();
        BasicBlock::iterator End = BB.end();
        for (BasicBlock::iterator I = Start; I != End; ++I) {
            BasicBlock::iterator J = llvm::next(I);
            
            for ( ; J != End; ) {
                
                if (isa<AllocaInst>(I) || !I->isIdenticalTo(J)) {
                    J = llvm::next(J);
                    continue;
                } else {
                    J->replaceAllUsesWith(I);
                    AA->replaceWithNewValue(J, I);  
                    SE->forgetValue(J);
                    BasicBlock::iterator K = llvm::next(J);
                    J->eraseFromParent();
                    J = K;
                }
            }
        }

        return false;
    }
    // Replace phi nodes of individual valiables with vector they originated 
    // from.
    bool WIVectorize::vectorizePhiNodes(BasicBlock &BB) {
        BasicBlock::iterator Start = BB.begin();
        BasicBlock::iterator End = BB.getFirstInsertionPt();

        ValueVectorMap valueMap;
        for (BasicBlock::iterator I = Start; I != End; ++I) {
            PHINode* node = dyn_cast<PHINode>(I);
            if (node) {
                ValueVector* candidateVector = new ValueVector;
                for (BasicBlock::iterator J = llvm::next(I);
                    J != End; ++J) {
                    PHINode* node2 = dyn_cast<PHINode>(J);
                    if (node2) {
                        bool match = true;
                        if (node->getNumIncomingValues() != 
                            node2->getNumIncomingValues())
                            continue;
                        
                        for (unsigned int i = 0; 
                             i < node->getNumIncomingValues(); i++) {
                            Value* v1 = node->getIncomingValue(i);
                            Value* v2 = node2->getIncomingValue(i);
                            if (node->getIncomingBlock(i) != 
                                node2->getIncomingBlock(i)) {
                                match = false;
                            }
                            // Stored sources contain original value from
                            // which one in phi node was extracted from
                            DenseMap<Value*, Value*>::iterator vi = 
                                storedSources.find(v1);
                            if (vi != storedSources.end()) {
                                DenseMap<Value*, Value*>::iterator ji =
                                    storedSources.find(v2);
                                if (ji != storedSources.end() &&
                                    (*vi).second == (*ji).second) {
                                } else {
                                    match = false;
                                }
                            } else {
                                // Incaming value can be also constant, they 
                                // have to match.                                
                                Constant* const1 = dyn_cast<Constant>(v1);
                                Constant* const2 = dyn_cast<Constant>(v2);
                                if (!(const1 && const2)) /* && 
                                    const1->getValue() == const2->getValue())) */{
                                    match = false;
                                }
                            }
                        }
                        if (match)
                            candidateVector->push_back(node2);
                    }
                }
                if (candidateVector->size() == VectorWidth -1) {
                    Value* newV = cast<Value>(node);
                    valueMap[newV] = candidateVector;
                }
            }
        }
        // Actually create new phi node
        for (DenseMap<Value*, ValueVector*>::iterator i =
            valueMap.begin(); i != valueMap.end(); i++) {
            ValueVector& v = *(*i).second;
            PHINode* orig = cast<PHINode>((*i).first);          
            Type *IType = orig->getType();
            Type *VType = getVecTypeForVector(IType);          
            PHINode* phi = PHINode::Create(VType, orig->getNumIncomingValues(),
                    getReplacementName(orig, false,0), orig);
            // Add incoming pairs to the phi node.
            for (unsigned int i = 0; i < orig->getNumIncomingValues(); i++) {
                Value* inc = orig->getIncomingValue(i);
                BasicBlock* BB = orig->getIncomingBlock(i);
                DenseMap<Value*, Value*>::iterator iter = 
                    storedSources.find(inc);
                if (iter != storedSources.end()) {
                    phi->addIncoming((*iter).second, BB);
                } else {
                    Constant* origConst = cast<Constant>(inc);
                    Constant* cons = ConstantVector::getSplat(                      
                        VectorWidth, origConst);
                    phi->addIncoming(cons, BB);
                }
            }
            // Extract scalar values from phi node to be used in the body 
            // of basic block. Replacing their uses cause instruction combiner
            // to find extractlement -> insertelement pairs and drop them
            // leaving direct use of vector.
            LLVMContext& Context = BB.getContext();
            BasicBlock::iterator toFill = BB.getFirstInsertionPt();
            int index = 0;
            
            // Find from the user of original phi node in which position it
            // is inserted to the vector before being used by vector instruction.
            // We have to extract it from same position of the vector phi node.
            Instruction::use_iterator useiter = orig->use_begin();            
            while (useiter != orig->use_end()) {
                llvm::User* tmp = *useiter;
                if (isa<InsertElementInst>(tmp)) {
                    Value* in = tmp->getOperand(2);
                    if (isa<ConstantInt>(in)) {
                        index =
                            cast<ConstantInt>(in)->getZExtValue();      
                            break;
                    }
                }
                useiter++;                
            }
                    
            //}
            Value *X = ConstantInt::get(Type::getInt32Ty(Context), index);       
            Instruction* other = ExtractElementInst::Create(phi, X,
                                            getReplacementName(phi, false, 0));
            other->insertAfter(toFill);
            orig->replaceAllUsesWith(other);
            AA->replaceWithNewValue(orig, other);
            SE->forgetValue(orig);
            orig->eraseFromParent();
            Instruction* ins = other;
            for (unsigned int i = 0; i < v.size(); i++) {
                Instruction* tmp = cast<Instruction>(v[i]);            
                // Find from the user of original phi node in which position it
                // is inserted to the vector before being used by vector instruction.
                // We have to extract it from same position of the vector phi node.                
                Instruction::use_iterator ui = tmp->use_begin();                
                while (ui != tmp->use_end()) {
                    llvm::User* user = *ui;
                    if (isa<InsertElementInst>(user)) {
                        Value* in = user->getOperand(2);
                        if (isa<ConstantInt>(in)) {
                            index =
                                cast<ConstantInt>(in)->getZExtValue();  
                                break;
                        }
                    }
                    ui++;                
                }
                X = ConstantInt::get(Type::getInt32Ty(Context), index);            
                Instruction* other = ExtractElementInst::Create(phi, X,
                                            getReplacementName(phi, false, index));
                other->insertAfter(ins);

                tmp->replaceAllUsesWith(other);
                AA->replaceWithNewValue(tmp, other);  
                SE->forgetValue(tmp);
                tmp->eraseFromParent();
                ins = other;
            }          
            
        }
        return true;      
    }
  // This function implements one vectorization iteration on the provided
  // basic block. It returns true if the block is changed.
  bool WIVectorize::vectorizePairs(BasicBlock &BB) {
    bool ShouldContinue;
    BasicBlock::iterator Start = BB.getFirstInsertionPt();

    std::vector<Value *> AllPairableInsts;
    DenseMap<Value *, Value *> AllChosenPairs;
    
      std::vector<Value *> PairableInsts;
      std::multimap<Value *, Value *> CandidatePairs;
      ShouldContinue = getCandidatePairs(BB, Start, CandidatePairs,
                                         PairableInsts);
      if (PairableInsts.empty()) return false;
      // Now we have a map of all of the pairable instructions and we need to
      // select the best possible pairing. A good pairing is one such that the
      // users of the pair are also paired. This defines a (directed) forest
      // over the pairs such that two pairs are connected iff the second pair
      // uses the first.

      // Note that it only matters that both members of the second pair use some
      // element of the first pair (to allow for splatting).

      std::multimap<ValuePair, ValuePair> ConnectedPairs;
      computeConnectedPairs(CandidatePairs, PairableInsts, ConnectedPairs);
      
      // Build the pairable-instruction dependency map
      DenseSet<ValuePair> PairableInstUsers;
      buildDepMap(BB, CandidatePairs, PairableInsts, PairableInstUsers);
      
      // There is now a graph of the connected pairs. For each variable, pick
      // the pairing with the largest tree meeting the depth requirement on at
      // least one branch. Then select all pairings that are part of that tree
      // and remove them from the list of available pairings and pairable
      // variables.

      DenseMap<Value *, Value *> ChosenPairs;
      choosePairs(CandidatePairs, PairableInsts, ConnectedPairs,
        PairableInstUsers, ChosenPairs);
      
      if (ChosenPairs.empty())
          return false;
      
      AllPairableInsts.insert(AllPairableInsts.end(), PairableInsts.begin(),
                              PairableInsts.end());
      AllChosenPairs.insert(ChosenPairs.begin(), ChosenPairs.end());

    if (AllChosenPairs.empty()) return false;
    NumFusedOps += AllChosenPairs.size();

    // A set of pairs has now been selected. It is now necessary to replace the
    // paired instructions with vector instructions. For this procedure each
    // operand must be replaced with a vector operand. This vector is formed
    // by using build_vector on the old operands. The replaced values are then
    // replaced with a vector_extract on the result.  Subsequent optimization
    // passes should coalesce the build/extract combinations.

    fuseChosenPairs(BB, AllPairableInsts, AllChosenPairs);
    
    return true;
  }
  
  // This function returns true if the provided instruction is capable of being
  // fused into a vector instruction. This determination is based only on the
  // type and other attributes of the instruction.
  bool WIVectorize::isInstVectorizable(Instruction *I,
                                         bool &IsSimpleLoadStore) {
    IsSimpleLoadStore = false;

    if (MemOpsOnly && 
        !(isa<LoadInst>(I) || isa<StoreInst>(I) || isa<GetElementPtrInst>(I)))
        return false;
    
    if (CallInst *C = dyn_cast<CallInst>(I)) {
      if (!isVectorizableIntrinsic(C)) {
        return false;

      }
    } else if (LoadInst *L = dyn_cast<LoadInst>(I)) {
      // Vectorize simple loads if possbile:
      IsSimpleLoadStore = L->isSimple();
      if (!IsSimpleLoadStore || NoMemOps) {
        return false;
      }
    } else if (StoreInst *S = dyn_cast<StoreInst>(I)) {
      // Vectorize simple stores if possbile:
      IsSimpleLoadStore = S->isSimple();
      if (!IsSimpleLoadStore || NoMemOps) {
        return false;
      }
    } else if (CastInst *C = dyn_cast<CastInst>(I)) {
      // We can vectorize casts, but not casts of pointer types, etc.

      Type *SrcTy = C->getSrcTy();
      if (!SrcTy->isSingleValueType() || SrcTy->isPointerTy()) {
        return false;
      }
      Type *DestTy = C->getDestTy();
      if (!DestTy->isSingleValueType() || DestTy->isPointerTy()) {
        return false;
      }
    } else if (GetElementPtrInst *G = dyn_cast<GetElementPtrInst>(I)) {
      // Currently, vector GEPs exist only with one index.
      if (G->getNumIndices() != 1 || NoMemOps || NoGEP)
        return false;         
    } else if (isa<CmpInst>(I)) {
        if (NoCMP)
            return false;
    } else if (!(I->isBinaryOp())){ /*|| isa<ShuffleVectorInst>(I) ||
        isa<ExtractElementInst>(I) || isa<InsertElementInst>(I))) {*/
        return false;
    } 
    // We can't vectorize memory operations without target data
    if (TD == 0 && IsSimpleLoadStore)
      return false;

    Type *T1, *T2;
    if (isa<StoreInst>(I)) {
      // For stores, it is the value type, not the pointer type that matters
      // because the value is what will come from a vector register.

      Value *IVal = cast<StoreInst>(I)->getValueOperand();
      T1 = IVal->getType();
    } else {
      T1 = I->getType();
    }

    if (I->isCast())
      T2 = cast<CastInst>(I)->getSrcTy();
    else
      T2 = T1;

    // Not every type can be vectorized...
    if (!(VectorType::isValidElementType(T1) || T1->isVectorTy()) ||
        !(VectorType::isValidElementType(T2) || T2->isVectorTy())) {
      return false;
    }
    if ((T1->getPrimitiveSizeInBits() > (VectorWidth*32)/2 ||
        T2->getPrimitiveSizeInBits() > (VectorWidth*32)/2)) {
      return false;
    }
    
    // Floating point vectorization can be dissabled
    if (I->getType()->isFloatingPointTy() && NoFP)
        return false;
    
     // Do not vectorizer pointer types. Currently do not work with LLVM 3.1.
    if (!isa<GetElementPtrInst>(I) && 
         (T1->getScalarType()->isPointerTy() ||
         T2->getScalarType()->isPointerTy()))
       return false;  
    // Check if the instruction can be loop counter, we do not vectorize those
    // since they have to be same for all work items we are vectorizing
    // and computations of load/store indexes usually depenends on them.
    // Instruction combiner pass will remove duplicates.
    if (SE->isSCEVable(I->getType())) {
        const SCEV* sc = SE->getSCEV(I);
        if (const SCEVAddRecExpr* S = dyn_cast<SCEVAddRecExpr>(sc)) {
            if (I->hasNUses(2)) {
                // Loop counter instruction is used in the comparison
                // operation before branch and with the phi node.
                // Any more uses indicates that the instruction is also
                // used as part of some computation and possibly needs
                // to get vectorize.
                bool compare = false;
                bool phi = false;
                for (Value::use_iterator it = I->use_begin();
                     it != I->use_end();
                     it++) {
                    if (isa<CmpInst>(*it))
                        compare = true;
                    if (isa<PHINode>(*it))                    
                        phi = true;
                }   
                if (compare && phi)
                    return false;
            }
        }
    } 
    return true;
  }
    // This function returns true if the two provided instructions are compatible
    // (meaning that they can be fused into a vector instruction). This assumes
    // that I has already been determined to be vectorizable and that J is not
    // in the use tree of I.
    bool WIVectorize::areInstsCompatibleFromDifferentWi(Instruction *I, 
                                                        Instruction *J) {
        
        if (I->getMetadata("wi") == NULL || J->getMetadata("wi") == NULL) {
          return false;
        }
        if (MemOpsOnly && 
            !((isa<LoadInst>(I) && isa<LoadInst>(J)) ||
              (isa<StoreInst>(I) && isa<StoreInst>(J)) ||
              (isa<GetElementPtrInst>(I) && isa<GetElementPtrInst>(J)))) {
            return false;
        }
        MDNode* mi = I->getMetadata("wi");
        MDNode* mj = J->getMetadata("wi");
        assert(mi->getNumOperands() == 3);
        assert(mj->getNumOperands() == 3);

        // Second operand of MDNode contains MDNode with XYZ tripplet.
        MDNode* iXYZ= dyn_cast<MDNode>(mi->getOperand(2));
        MDNode* jXYZ= dyn_cast<MDNode>(mj->getOperand(2));
        assert(iXYZ->getNumOperands() == 4);
        assert(jXYZ->getNumOperands() == 4);
        
        ConstantInt *CIX = dyn_cast<ConstantInt>(iXYZ->getOperand(1));
        ConstantInt *CJX = dyn_cast<ConstantInt>(jXYZ->getOperand(1));
        
        ConstantInt *CIY = dyn_cast<ConstantInt>(iXYZ->getOperand(2));
        ConstantInt *CJY = dyn_cast<ConstantInt>(jXYZ->getOperand(2));
        
        ConstantInt *CIZ = dyn_cast<ConstantInt>(iXYZ->getOperand(3));
        ConstantInt *CJZ = dyn_cast<ConstantInt>(jXYZ->getOperand(3));
        
        if ( CIX->getValue() == CJX->getValue()
            && CIY->getValue() == CJY->getValue()
            && CIZ->getValue() == CJZ->getValue()) {
            // Same work item, no vectorizing
            return false;
        }
        mi = I->getMetadata("wi_counter");
        mj = J->getMetadata("wi_counter");
                
        ConstantInt *CI = dyn_cast<ConstantInt>(mi->getOperand(1));
        ConstantInt *CJ = dyn_cast<ConstantInt>(mj->getOperand(1));
        if (CI->getValue() != CJ->getValue()) {
          // different line in the original work item
          // we do not want to vectorize operations that do not match
          return false;
        }
        return true;
    }
    static inline void getInstructionTypes(Instruction *I,
                                           Type *&T1, Type *&T2) {
      if (isa<StoreInst>(I)) {
        // For stores, it is the value type, not the pointer type that matters
        // because the value is what will come from a vector register.
  
        Value *IVal = cast<StoreInst>(I)->getValueOperand();
        T1 = IVal->getType();
      } else {
        T1 = I->getType();
      }
  
      if (I->isCast())
        T2 = cast<CastInst>(I)->getSrcTy();
      else
        T2 = T1;

      if (SelectInst *SI = dyn_cast<SelectInst>(I)) {
        T2 = SI->getCondition()->getType();
      } else if (ShuffleVectorInst *SI = dyn_cast<ShuffleVectorInst>(I)) {
        T2 = SI->getOperand(0)->getType();
      } else if (CmpInst *CI = dyn_cast<CmpInst>(I)) {
        T2 = CI->getOperand(0)->getType();
      }
    }
    
  // This function returns true if the two provided instructions are compatible
  // (meaning that they can be fused into a vector instruction). This assumes
  // that I has already been determined to be vectorizable and that J is not
  // in the use tree of I.
  bool WIVectorize::areInstsCompatible(Instruction *I, Instruction *J,
                       bool IsSimpleLoadStore) {
    DEBUG( if (DebugInstructionExamination) dbgs() << "WIV: looking at " << *I <<
                     " <-> " << *J << "\n");

    // Loads and stores can be merged if they have different alignments,
    // but are otherwise the same.
    LoadInst *LI, *LJ;
    StoreInst *SI, *SJ;
    if (!J->isSameOperationAs(I)) {
      return false;
    }
    Type *IT1, *IT2, *JT1, *JT2;
    getInstructionTypes(I, IT1, IT2);
    getInstructionTypes(J, JT1, JT2);

    if (IsSimpleLoadStore || isa<GetElementPtrInst>(I)) {
      Value *IPtr, *JPtr;
      unsigned IAlignment, JAlignment, IAddressSpace, JAddressSpace;
      int64_t OffsetInElmts = 0;
      bool foundPointer = 
          getPairPtrInfo(I, J, IPtr, JPtr, IAlignment, JAlignment, 
                 IAddressSpace, JAddressSpace, OffsetInElmts);
      if ( foundPointer && abs64(OffsetInElmts) == 1) {         
            Type *aTypeI = isa<StoreInst>(I) ?
              cast<StoreInst>(I)->getValueOperand()->getType() : I->getType();
            Type *aTypeJ = isa<StoreInst>(J) ?
              cast<StoreInst>(J)->getValueOperand()->getType() : J->getType();
            Type *VType = getVecTypeForPair(aTypeI, aTypeJ);                
            // An aligned load or store is possible only if the instruction
            // with the lower offset has an alignment suitable for the
            // vector type.

            unsigned BottomAlignment = IAlignment;
            if (OffsetInElmts < 0) BottomAlignment = JAlignment;

            unsigned VecAlignment = TD->getPrefTypeAlignment(VType);
            if (AlignedOnly) {            
                if (BottomAlignment < VecAlignment) {
                    return false;
                }
            }
#ifndef LLVM_3_1            
            if (VTTI) {
              unsigned ICost = VTTI->getMemoryOpCost(I->getOpcode(), I->getType(),
                                                     IAlignment, IAddressSpace);
              unsigned JCost = VTTI->getMemoryOpCost(J->getOpcode(), J->getType(),
                                                     JAlignment, JAddressSpace);
              unsigned VCost = VTTI->getMemoryOpCost(I->getOpcode(), VType,
                                                     BottomAlignment,
                                                     IAddressSpace);
              if (VCost > ICost + JCost)
                return false;

              // We don't want to fuse to a type that will be split, even
              // if the two input types will also be split and there is no other
              // associated cost.
              unsigned VParts = VTTI->getNumberOfParts(VType);
              if (VParts > 1)
                return false;
              else if (!VParts && VCost == ICost + JCost)
                return false;

            }   
#endif                     
      } else if(foundPointer && abs64(OffsetInElmts)>1){
          if (isa<GetElementPtrInst>(I)) {
              return true;
          }            
          // Collect information on memory accesses with stride.
          // This is not usefull for anything, just to analyze code a bit.
          if (I->getMetadata("wi") != NULL) {
              MDNode* md = I->getMetadata("wi");
              MDNode* mdCounter = I->getMetadata("wi_counter");
              MDNode* mdRegion = dyn_cast<MDNode>(md->getOperand(1));
      
              unsigned CI = 
                cast<ConstantInt>(mdCounter->getOperand(1))->getZExtValue();
              unsigned RI = 
                cast<ConstantInt>(mdRegion->getOperand(1))->getZExtValue();
              std::pair<int, int> index = std::pair<int,int>(RI,CI);
              DenseMap<std::pair<int,int>, ValueVector*>::iterator it = 
                stridedOps.find(index);
              ValueVector* v = NULL;
              if (it != stridedOps.end()) {
                  v = (*it).second;
              } else {
                  v = new ValueVector;
              }
              v->push_back(I);
              v->push_back(J);
              stridedOps.insert(
                  std::pair< std::pair<int, int>, ValueVector*>(index, v));
          }
          return false;
      } else {
        return false;
      }
    } else if (isa<ShuffleVectorInst>(I)) {
      // Only merge two shuffles if they're both constant
      return isa<Constant>(I->getOperand(2)) &&
             isa<Constant>(J->getOperand(2));
      // FIXME: We may want to vectorize non-constant shuffles also.
#ifdef LLVM_3_1             
    }
#else    
    }  else if (VTTI) {
      unsigned ICost = getInstrCost(I->getOpcode(), IT1, IT2);
      unsigned JCost = getInstrCost(J->getOpcode(), JT1, JT2);
      Type *VT1 = getVecTypeForPair(IT1, JT1),
           *VT2 = getVecTypeForPair(IT2, JT2);
      unsigned VCost = getInstrCost(I->getOpcode(), VT1, VT2);

      if (VCost > ICost + JCost) {
        return false;
      }
      // We don't want to fuse to a type that will be split, even
      // if the two input types will also be split and there is no other
      // associated cost.
      unsigned VParts1 = VTTI->getNumberOfParts(VT1),
               VParts2 = VTTI->getNumberOfParts(VT2);
      if (VParts1 > 1 || VParts2 > 1)
        return false;
      else if ((!VParts1 || !VParts2) && VCost == ICost + JCost)
        return false;

      //CostSavings = ICost + JCost - VCost;
    }
#endif    
    // The powi intrinsic is special because only the first argument is
    // vectorized, the second arguments must be equal.
    CallInst *CI = dyn_cast<CallInst>(I);
    Function *FI;
    if (CI && (FI = CI->getCalledFunction()) &&
        FI->getIntrinsicID() == Intrinsic::powi) {

      Value *A1I = CI->getArgOperand(1),
            *A1J = cast<CallInst>(J)->getArgOperand(1);
      const SCEV *A1ISCEV = SE->getSCEV(A1I),
                 *A1JSCEV = SE->getSCEV(A1J);
      return (A1ISCEV == A1JSCEV);
    }    
    return true;
  }

  // Figure out whether or not J uses I and update the users and write-set
  // structures associated with I. Specifically, Users represents the set of
  // instructions that depend on I. WriteSet represents the set
  // of memory locations that are dependent on I. If UpdateUsers is true,
  // and J uses I, then Users is updated to contain J and WriteSet is updated
  // to contain any memory locations to which J writes. The function returns
  // true if J uses I. By default, alias analysis is used to determine
  // whether J reads from memory that overlaps with a location in WriteSet.
  // If LoadMoveSet is not null, then it is a previously-computed multimap
  // where the key is the memory-based user instruction and the value is
  // the instruction to be compared with I. So, if LoadMoveSet is provided,
  // then the alias analysis is not used. This is necessary because this
  // function is called during the process of moving instructions during
  // vectorization and the results of the alias analysis are not stable during
  // that process.
  bool WIVectorize::trackUsesOfI(DenseSet<Value *> &Users,
                       AliasSetTracker &WriteSet, Instruction *I,
                       Instruction *J, bool UpdateUsers,
                       std::multimap<Value *, Value *> *LoadMoveSet) {
    bool UsesI = false;

    // This instruction may already be marked as a user due, for example, to
    // being a member of a selected pair.
    if (Users.count(J))
      UsesI = true;

    if (!UsesI)
      for (User::op_iterator JU = J->op_begin(), JE = J->op_end();
           JU != JE; ++JU) {
        Value *V = *JU;
        if (I == V || Users.count(V)) {
          UsesI = true;
          break;
        }
      }
    if (!UsesI && J->mayReadFromMemory()) {
      if (LoadMoveSet) {
        VPIteratorPair JPairRange = LoadMoveSet->equal_range(J);
        UsesI = isSecondInIteratorPair<Value*>(I, JPairRange);
      }
    }

    if (UsesI && UpdateUsers) {
      if (J->mayWriteToMemory()) WriteSet.add(J);
      Users.insert(J);
    }
    
    return UsesI;
  }
  
  // This function iterates over all instruction pairs in the provided
  // basic block and collects all candidate pairs for vectorization.
  bool WIVectorize::getCandidatePairs(BasicBlock &BB,
                       BasicBlock::iterator &Start,
                       std::multimap<Value *, Value *> &CandidatePairs,
                       std::vector<Value *> &PairableInsts) {
    BasicBlock::iterator E = BB.end();
    LLVMContext& context = BB.getContext();
    
    if (Start == E) return false;

    std::multimap<int, ValueVector*> temporary;
    for (BasicBlock::iterator I = Start++; I != E; ++I) {

        if (I->getMetadata("wi") == NULL)
            continue;
        bool IsSimpleLoadStore;
        if (!isInstVectorizable(I, IsSimpleLoadStore)) {          
            continue;          
        }
        
        MDNode* md = I->getMetadata("wi");
        MDNode* mdCounter = I->getMetadata("wi_counter");
        MDNode* mdRegion = dyn_cast<MDNode>(md->getOperand(1));
        
        unsigned CI = cast<ConstantInt>(mdCounter->getOperand(1))->getZExtValue();
        unsigned RI = cast<ConstantInt>(mdRegion->getOperand(1))->getZExtValue();
        
        std::multimap<int,ValueVector*>::iterator itb = temporary.lower_bound(CI);
        std::multimap<int,ValueVector*>::iterator ite = temporary.upper_bound(CI);
        ValueVector* tmpVec = NULL;   
        while(itb != ite) {
            if (I->isSameOperationAs(cast<Instruction>((*(*itb).second)[0]))) {
                // Test also if instructions are from same region.
                MDNode* tmpMD = 
                    cast<Instruction>((*(*itb).second)[0])->getMetadata("wi");
                MDNode* tmpRINode = dyn_cast<MDNode>(tmpMD->getOperand(1));
                unsigned tmpRI = 
                    cast<ConstantInt>(tmpRINode->getOperand(1))->getZExtValue();                
                if (RI == tmpRI)
                    tmpVec = (*itb).second;
            }
            itb++;
        }
        if (tmpVec == NULL) {
            tmpVec = new ValueVector;
            temporary.insert(std::pair<int, ValueVector*>(CI, tmpVec));          
        }
        tmpVec->push_back(I);
    }
    DenseSet<Value *> Users;
    AliasSetTracker WriteSet(*AA);    
    for (std::multimap<int, ValueVector*>::iterator insIt = temporary.begin();
         insIt != temporary.end(); insIt++) {
        ValueVector* tmpVec = (*insIt).second;
        // Prevent creation of vectors shorter then the vector width in case
        // vectorization of asymetric counters is disabled.
        if (tmpVec->size() % 2 != 0 && NoCount) {
            continue;
        }
            
        if (tmpVec->size() % 2 != 0 && !MemOpsOnly) {

            // Ok, this is extremely ugly, however this code is specific for
            // for situation where the base address of some array is computed
            // one way and the addresses for the rest of the work items are
            // computed other way. E.g.
            // id_0 = x*y*z
            // id_1 = id_0 + const
            // id_2 = id_0 + const + const
            // ...
            // Therefore only applicable to add operation.
            // It should bring some performance improvements when targetting TTA.
            
            // NOTE: results are opposide of what is expected.
            // With NoCount set to true, the vectorization of loop counter arithmetic
            // operations is actually prevented. The ProgramPartitioner is assigning
            // them to the lanes. This seems to provide better performance.
            // With NoCount set to false, the vectorization of loop counter
            // arithmetic is allowed, creating better bitcode, but when mapped
            // to TTA, performance is much worse.

            Instruction* tmp = cast<Instruction>((*tmpVec)[0]);                        
            if ( !(tmpVec->size() == 1 || 
                tmp->getType()->isVectorTy() ||
                tmp->getOpcode() != Instruction::Add)) { 
                
                bool identity = false;
                bool argumentOperand = false;
                // If none of the arguments to add is constant
                // we do not replace it with identity, neither if operand
                // is function argument since that can be used in different
                // blocks.
                for (unsigned o = 0; o < tmp->getNumOperands(); ++o) {
                    if (isa<ConstantInt>(tmp->getOperand(o))) {
                        identity = true;
                    }
                    if (isa<Argument>(tmp->getOperand(o))) {
                        argumentOperand = true;
                    }                    
                }
                if (!identity || argumentOperand) 
                    continue;
                
                Instruction* K = tmp->clone();
                if ((*tmpVec)[0]->hasName()) {
                    std::string name = (*tmpVec)[0]->getName().str() + "_temp_0";
                    K->setName(name);
                }
                  
                if (tmp->getMetadata("wi") != NULL) {                                                
                    MDNode* md = tmp->getMetadata("wi");
                    MDNode* xyz = dyn_cast<MDNode>(md->getOperand(2));
                    MDNode* region = dyn_cast<MDNode>(md->getOperand(1));
                    ConstantInt *CIX = 
                        dyn_cast<ConstantInt>(xyz->getOperand(1));    
                    ConstantInt *CIY = 
                        dyn_cast<ConstantInt>(xyz->getOperand(2));        
                    ConstantInt *CIZ = 
                        dyn_cast<ConstantInt>(xyz->getOperand(3));
                    if (CIX->getValue() == 1) {
                        Value *v2[] = {
                            MDString::get(context, "WI_xyz"),      
                            ConstantInt::get(Type::getInt32Ty(context), 0),
                            CIY,      
                            CIZ};                 
                        MDNode* newXYZ = MDNode::get(context, v2);
                        Value *v[] = {
                            MDString::get(context, "WI_data"),      
                            region,
                            newXYZ};
                        MDNode* mdNew = MDNode::get(context, v);              
                        K->setMetadata("wi", mdNew);
                        K->setMetadata("wi_counter", tmp->getMetadata("wi_counter"));
                    }
                }
                for (unsigned o = 0; o < K->getNumOperands(); ++o) {
                    if (isa<ConstantInt>(K->getOperand(o))) {
                        K->setOperand(o, 
                              ConstantInt::get(K->getOperand(o)->getType(), 0));
                    }
                }
                
                Value* original = NULL;
                for (unsigned o = 0; o < K->getNumOperands(); ++o) {
                    if (!isa<PHINode>(K->getOperand(o)) &&
                        isa<Instruction>(K->getOperand(o))) {
                        original = K->getOperand(o);
                    }
                }
                if (original != NULL) {
                    K->insertAfter(cast<Instruction>(original));
                    std::vector<User*> usesToReplace;
                    for (Value::use_iterator it = original->use_begin();
                         it != original->use_end();
                         it++) {
                        bool usedInVec = false;                            
                        if (*it != K) {
                            if (!NoCount) {
                                for (unsigned int j = 0; j < tmpVec->size(); j++) {
                                    if ((*it) == (*tmpVec)[j]) {
                                        usedInVec = true;
                                        break;
                                    }
                                }
                            }
                            if (!usedInVec) {
                                usesToReplace.push_back(*it);
                            }
                        }
                    }                    
                    for (unsigned int j = 0; j < usesToReplace.size(); j++) {
                       usesToReplace[j]->replaceUsesOfWith(original, K);
                    }
                } else {
                    K->insertBefore(tmp);
                }
                tmpVec->insert(tmpVec->begin(), K);
            }
        }
        
        // Create actual candidate pairs
        for (unsigned j = 0; j < tmpVec->size()/2; j++) {
            Instruction* I = cast<Instruction>((*tmpVec)[2*j]);
            Instruction* J = cast<Instruction>((*tmpVec)[2*j+1]);
            if (!areInstsCompatibleFromDifferentWi(I,J)) continue;
            bool IsSimpleLoadStore;

            if (!isInstVectorizable(I, IsSimpleLoadStore)) {
                break;            
            }            

            if (!areInstsCompatible(I, J, IsSimpleLoadStore)) { 
                break;
            }            
            
            // Determine if J uses I, if so, exit the loop.
            bool UsesI = trackUsesOfI(Users, WriteSet, I, J, true);            
            if (UsesI) {
                break;
            }

            if (!PairableInsts.size() ||
                PairableInsts[PairableInsts.size()-1] != I) {
                PairableInsts.push_back(I);
            }
            CandidatePairs.insert(ValuePair(I, J));            
        }
    }
    return false;
  }
  
  // Finds candidate pairs connected to the pair P = <PI, PJ>. This means that
  // it looks for pairs such that both members have an input which is an
  // output of PI or PJ.
  void WIVectorize::computePairsConnectedTo(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *>& /*PairableInsts*/,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      ValuePair P) {
    StoreInst *SI, *SJ;
    // For each possible pairing for this variable, look at the uses of
    // the first value...
    for (Value::use_iterator I = P.first->use_begin(),
         E = P.first->use_end(); I != E; ++I) {
      if (isa<LoadInst>(*I)) {
        // A pair cannot be connected to a load because the load only takes one
        // operand (the address) and it is a scalar even after vectorization.
        continue;
      } else if ((SI = dyn_cast<StoreInst>(*I)) &&
                 P.first == SI->getPointerOperand()) {
        // Similarly, a pair cannot be connected to a store through its
        // pointer operand.
        continue;
      }        
      VPIteratorPair IPairRange = CandidatePairs.equal_range(*I);

      // For each use of the first variable, look for uses of the second
      // variable...
      for (Value::use_iterator J = P.second->use_begin(),
           E2 = P.second->use_end(); J != E2; ++J) {
        
        if ((SJ = dyn_cast<StoreInst>(*J)) &&
            P.second == SJ->getPointerOperand())
          continue;          
      
        VPIteratorPair JPairRange = CandidatePairs.equal_range(*J);

        // Look for <I, J>:
        if (isSecondInIteratorPair<Value*>(*J, IPairRange))
          ConnectedPairs.insert(VPPair(P, ValuePair(*I, *J)));

        // Look for <J, I>:
        if (isSecondInIteratorPair<Value*>(*I, JPairRange))
          ConnectedPairs.insert(VPPair(P, ValuePair(*J, *I)));
      }
      // Look for cases where just the first value in the pair is used by
      // both members of another pair (splatting).
      for (Value::use_iterator J = P.first->use_begin(); J != E; ++J) {
        if ((SJ = dyn_cast<StoreInst>(*J)) &&
            P.first == SJ->getPointerOperand())
          continue;          
        
        if (isSecondInIteratorPair<Value*>(*J, IPairRange))
          ConnectedPairs.insert(VPPair(P, ValuePair(*I, *J)));
      }
    }
    // Look for cases where just the second value in the pair is used by
    // both members of another pair (splatting).
    for (Value::use_iterator I = P.second->use_begin(),
         E = P.second->use_end(); I != E; ++I) {
      if (isa<LoadInst>(*I)) {
        continue;
      } else if ((SI = dyn_cast<StoreInst>(*I)) &&
               P.second == SI->getPointerOperand()) {
        continue;        
      }
      VPIteratorPair IPairRange = CandidatePairs.equal_range(*I);

      for (Value::use_iterator J = P.second->use_begin(); J != E; ++J) {
        if ((SJ = dyn_cast<StoreInst>(*J)) &&
            P.second == SJ->getPointerOperand())
          continue;
          
        if (isSecondInIteratorPair<Value*>(*J, IPairRange))
          ConnectedPairs.insert(VPPair(P, ValuePair(*I, *J)));
      }
    }
  }

  // This function figures out which pairs are connected.  Two pairs are
  // connected if some output of the first pair forms an input to both members
  // of the second pair.
  void WIVectorize::computeConnectedPairs(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs) {

    for (std::vector<Value *>::iterator PI = PairableInsts.begin(),
         PE = PairableInsts.end(); PI != PE; ++PI) {
      VPIteratorPair choiceRange = CandidatePairs.equal_range(*PI);

      for (std::multimap<Value *, Value *>::iterator P = choiceRange.first;
           P != choiceRange.second; ++P)
        computePairsConnectedTo(CandidatePairs, PairableInsts,
                                ConnectedPairs, *P);
    }

    DEBUG(dbgs() << "WIV: found " << ConnectedPairs.size()
                 << " pair connections.\n");
  }

  // This function builds a set of use tuples such that <A, B> is in the set
  // if B is in the use tree of A. If B is in the use tree of A, then B
  // depends on the output of A.
  void WIVectorize::buildDepMap(
                      BasicBlock &BB,
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *>& /*PairableInsts*/,
                      DenseSet<ValuePair> &PairableInstUsers) {
    DenseSet<Value *> IsInPair;
    for (std::multimap<Value *, Value *>::iterator C = CandidatePairs.begin(),
         E = CandidatePairs.end(); C != E; ++C) {
      IsInPair.insert(C->first);
      IsInPair.insert(C->second);
    }

    // Iterate through the basic block, recording all Users of each
    // pairable instruction.

    BasicBlock::iterator E = BB.end();
    for (BasicBlock::iterator I = BB.getFirstInsertionPt(); I != E; ++I) {
      if (IsInPair.find(I) == IsInPair.end()) continue;

      DenseSet<Value *> Users;
      AliasSetTracker WriteSet(*AA);
      for (BasicBlock::iterator J = llvm::next(I); J != E; ++J)
        (void) trackUsesOfI(Users, WriteSet, I, J);

      for (DenseSet<Value *>::iterator U = Users.begin(), E = Users.end();
           U != E; ++U)
        PairableInstUsers.insert(ValuePair(I, *U));
    }
  }

  // Returns true if an input to pair P is an output of pair Q and also an
  // input of pair Q is an output of pair P. If this is the case, then these
  // two pairs cannot be simultaneously fused.
  bool WIVectorize::pairsConflict(ValuePair P, ValuePair Q,
                     DenseSet<ValuePair> &PairableInstUsers,
                     std::multimap<ValuePair, ValuePair> *PairableInstUserMap) {
      
    // Two pairs are in conflict if they are mutual Users of eachother.
    bool QUsesP = PairableInstUsers.count(ValuePair(P.first,  Q.first))  ||
                  PairableInstUsers.count(ValuePair(P.first,  Q.second)) ||
                  PairableInstUsers.count(ValuePair(P.second, Q.first))  ||
                  PairableInstUsers.count(ValuePair(P.second, Q.second));
    bool PUsesQ = PairableInstUsers.count(ValuePair(Q.first,  P.first))  ||
                  PairableInstUsers.count(ValuePair(Q.first,  P.second)) ||
                  PairableInstUsers.count(ValuePair(Q.second, P.first))  ||
                  PairableInstUsers.count(ValuePair(Q.second, P.second));
    if (PairableInstUserMap) {
      // FIXME: The expensive part of the cycle check is not so much the cycle
      // check itself but this edge insertion procedure. This needs some
      // profiling and probably a different data structure (same is true of
      // most uses of std::multimap).
      if (PUsesQ) {
        VPPIteratorPair QPairRange = PairableInstUserMap->equal_range(Q);
        if (!isSecondInIteratorPair(P, QPairRange))
          PairableInstUserMap->insert(VPPair(Q, P));
      }
      if (QUsesP) {
        VPPIteratorPair PPairRange = PairableInstUserMap->equal_range(P);
        if (!isSecondInIteratorPair(Q, PPairRange))
          PairableInstUserMap->insert(VPPair(P, Q));
      }
    }

    return (QUsesP && PUsesQ);
  }

  // This function walks the use graph of current pairs to see if, starting
  // from P, the walk returns to P.
  bool WIVectorize::pairWillFormCycle(ValuePair P,
                       std::multimap<ValuePair, ValuePair> &PairableInstUserMap,
                       DenseSet<ValuePair> &CurrentPairs) {
      
    DEBUG(if (DebugCycleCheck)
            dbgs() << "WIV: starting cycle check for : " << *P.first << " <-> "
                   << *P.second << "\n");
    // A lookup table of visisted pairs is kept because the PairableInstUserMap
    // contains non-direct associations.
    DenseSet<ValuePair> Visited;
    SmallVector<ValuePair, 32> Q;
    // General depth-first post-order traversal:
    Q.push_back(P);
    do {
      ValuePair QTop = Q.pop_back_val();
      Visited.insert(QTop);

      DEBUG(if (DebugCycleCheck)
              dbgs() << "WIV: cycle check visiting: " << *QTop.first << " <-> "
                     << *QTop.second << "\n");
      VPPIteratorPair QPairRange = PairableInstUserMap.equal_range(QTop);
      for (std::multimap<ValuePair, ValuePair>::iterator C = QPairRange.first;
           C != QPairRange.second; ++C) {
        if (C->second == P) {
          DEBUG(dbgs()
                 << "WIV: rejected to prevent non-trivial cycle formation: "
                 << *C->first.first << " <-> " << *C->first.second << "\n");
          return true;
        }

        if (CurrentPairs.count(C->second) && !Visited.count(C->second))
          Q.push_back(C->second);
      }
    } while (!Q.empty());

    return false;
  }

  // This function builds the initial tree of connected pairs with the
  // pair J at the root.
  void WIVectorize::buildInitialTreeFor(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *>& /*PairableInsts*/,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      DenseSet<ValuePair>& /*PairableInstUsers*/,
                      DenseMap<Value *, Value *>& /*ChosenPairs*/,
                      DenseMap<ValuePair, size_t> &Tree, ValuePair J) {
    // Each of these pairs is viewed as the root node of a Tree. The Tree
    // is then walked (depth-first). As this happens, we keep track of
    // the pairs that compose the Tree and the maximum depth of the Tree.
    SmallVector<ValuePairWithDepth, 32> Q;
    // General depth-first post-order traversal:
    Q.push_back(ValuePairWithDepth(J, getDepthFactor(J.first)));
    do {
      ValuePairWithDepth QTop = Q.back();

      // Push each child onto the queue:
      bool MoreChildren = false;
      size_t MaxChildDepth = QTop.second;
      VPPIteratorPair qtRange = ConnectedPairs.equal_range(QTop.first);
      for (std::multimap<ValuePair, ValuePair>::iterator k = qtRange.first;
           k != qtRange.second; ++k) {
        // Make sure that this child pair is still a candidate:
        bool IsStillCand = false;
        VPIteratorPair checkRange =
          CandidatePairs.equal_range(k->second.first);
        for (std::multimap<Value *, Value *>::iterator m = checkRange.first;
             m != checkRange.second; ++m) {
          if (m->second == k->second.second) {
            IsStillCand = true;
            break;
          }
        }

        if (IsStillCand) {
          DenseMap<ValuePair, size_t>::iterator C = Tree.find(k->second);
          if (C == Tree.end()) {
            size_t d = getDepthFactor(k->second.first);
            Q.push_back(ValuePairWithDepth(k->second, QTop.second+d));
            MoreChildren = true;
          } else {
            MaxChildDepth = std::max(MaxChildDepth, C->second);
          }
        }
      }

      if (!MoreChildren) {
        // Record the current pair as part of the Tree:
        Tree.insert(ValuePairWithDepth(QTop.first, MaxChildDepth));
        Q.pop_back();
      }
    } while (!Q.empty());
  }

  // Given some initial tree, prune it by removing conflicting pairs (pairs
  // that cannot be simultaneously chosen for vectorization).
  void WIVectorize::pruneTreeFor(
                      std::multimap<Value *, Value *> &/*CandidatePairs*/,
                      std::vector<Value *> &/*PairableInsts*/,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      DenseSet<ValuePair> &PairableInstUsers,
                      std::multimap<ValuePair, ValuePair> &PairableInstUserMap,
                      DenseMap<Value *, Value *> &ChosenPairs,
                      DenseMap<ValuePair, size_t> &Tree,
                      DenseSet<ValuePair> &PrunedTree, ValuePair J,
                      bool UseCycleCheck) {
    SmallVector<ValuePairWithDepth, 32> Q;
    // General depth-first post-order traversal:
    Q.push_back(ValuePairWithDepth(J, getDepthFactor(J.first)));
    do {
      ValuePairWithDepth QTop = Q.pop_back_val();
      PrunedTree.insert(QTop.first);

      // Visit each child, pruning as necessary...
      DenseMap<ValuePair, size_t> BestChildren;
      VPPIteratorPair QTopRange = ConnectedPairs.equal_range(QTop.first);
      for (std::multimap<ValuePair, ValuePair>::iterator K = QTopRange.first;
           K != QTopRange.second; ++K) {
        DenseMap<ValuePair, size_t>::iterator C = Tree.find(K->second);
        if (C == Tree.end()) continue;

        // This child is in the Tree, now we need to make sure it is the
        // best of any conflicting children. There could be multiple
        // conflicting children, so first, determine if we're keeping
        // this child, then delete conflicting children as necessary.

        // It is also necessary to guard against pairing-induced
        // dependencies. Consider instructions a .. x .. y .. b
        // such that (a,b) are to be fused and (x,y) are to be fused
        // but a is an input to x and b is an output from y. This
        // means that y cannot be moved after b but x must be moved
        // after b for (a,b) to be fused. In other words, after
        // fusing (a,b) we have y .. a/b .. x where y is an input
        // to a/b and x is an output to a/b: x and y can no longer
        // be legally fused. To prevent this condition, we must
        // make sure that a child pair added to the Tree is not
        // both an input and output of an already-selected pair.

        // Pairing-induced dependencies can also form from more complicated
        // cycles. The pair vs. pair conflicts are easy to check, and so
        // that is done explicitly for "fast rejection", and because for
        // child vs. child conflicts, we may prefer to keep the current
        // pair in preference to the already-selected child.
        DenseSet<ValuePair> CurrentPairs;

        bool CanAdd = true;
        for (DenseMap<ValuePair, size_t>::iterator C2
              = BestChildren.begin(), E2 = BestChildren.end();
             C2 != E2; ++C2) {
          if (C2->first.first == C->first.first ||
              C2->first.first == C->first.second ||
              C2->first.second == C->first.first ||
              C2->first.second == C->first.second ||
              pairsConflict(C2->first, C->first, PairableInstUsers,
                            UseCycleCheck ? &PairableInstUserMap : 0)) {
            if (C2->second >= C->second) {
              CanAdd = false;
              break;
            }

            CurrentPairs.insert(C2->first);
          }
        }
        if (!CanAdd) continue;

        // Even worse, this child could conflict with another node already
        // selected for the Tree. If that is the case, ignore this child.
        for (DenseSet<ValuePair>::iterator T = PrunedTree.begin(),
             E2 = PrunedTree.end(); T != E2; ++T) {
          if (T->first == C->first.first ||
              T->first == C->first.second ||
              T->second == C->first.first ||
              T->second == C->first.second ||
              pairsConflict(*T, C->first, PairableInstUsers,
                            UseCycleCheck ? &PairableInstUserMap : 0)) {
            CanAdd = false;
            break;
          }

          CurrentPairs.insert(*T);
        }
        if (!CanAdd) continue;

        // And check the queue too...
        for (SmallVector<ValuePairWithDepth, 32>::iterator C2 = Q.begin(),
             E2 = Q.end(); C2 != E2; ++C2) {
          if (C2->first.first == C->first.first ||
              C2->first.first == C->first.second ||
              C2->first.second == C->first.first ||
              C2->first.second == C->first.second ||
              pairsConflict(C2->first, C->first, PairableInstUsers,
                            UseCycleCheck ? &PairableInstUserMap : 0)) {
            CanAdd = false;
            break;
          }

          CurrentPairs.insert(C2->first);
        }
        if (!CanAdd) continue;

        // Last but not least, check for a conflict with any of the
        // already-chosen pairs.
        for (DenseMap<Value *, Value *>::iterator C2 =
              ChosenPairs.begin(), E2 = ChosenPairs.end();
             C2 != E2; ++C2) {
          if (pairsConflict(*C2, C->first, PairableInstUsers,
                            UseCycleCheck ? &PairableInstUserMap : 0)) {
            CanAdd = false;
            break;
          }

          CurrentPairs.insert(*C2);
        }
        if (!CanAdd) continue;

        // To check for non-trivial cycles formed by the addition of the
        // current pair we've formed a list of all relevant pairs, now use a
        // graph walk to check for a cycle. We start from the current pair and
        // walk the use tree to see if we again reach the current pair. If we
        // do, then the current pair is rejected.

        // FIXME: It may be more efficient to use a topological-ordering
        // algorithm to improve the cycle check. This should be investigated.
        if (UseCycleCheck &&
            pairWillFormCycle(C->first, PairableInstUserMap, CurrentPairs))
          continue;

        // This child can be added, but we may have chosen it in preference
        // to an already-selected child. Check for this here, and if a
        // conflict is found, then remove the previously-selected child
        // before adding this one in its place.
        for (DenseMap<ValuePair, size_t>::iterator C2
              = BestChildren.begin(); C2 != BestChildren.end();) {
          if (C2->first.first == C->first.first ||
              C2->first.first == C->first.second ||
              C2->first.second == C->first.first ||
              C2->first.second == C->first.second ||
              pairsConflict(C2->first, C->first, PairableInstUsers))
            BestChildren.erase(C2++);
          else
            ++C2;
        }

        BestChildren.insert(ValuePairWithDepth(C->first, C->second));
      }

      for (DenseMap<ValuePair, size_t>::iterator C
            = BestChildren.begin(), E2 = BestChildren.end();
           C != E2; ++C) {
        size_t DepthF = getDepthFactor(C->first.first);
        Q.push_back(ValuePairWithDepth(C->first, QTop.second+DepthF));
      }
    } while (!Q.empty());
  }

  // This function finds the best tree of mututally-compatible connected
  // pairs, given the choice of root pairs as an iterator range.
  void WIVectorize::findBestTreeFor(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      DenseSet<ValuePair> &PairableInstUsers,
                      std::multimap<ValuePair, ValuePair> &PairableInstUserMap,
                      DenseMap<Value *, Value *> &ChosenPairs,
                      DenseSet<ValuePair> &BestTree, size_t &BestMaxDepth,
                      size_t &BestEffSize, VPIteratorPair ChoiceRange,
                      bool UseCycleCheck) {
    for (std::multimap<Value *, Value *>::iterator J = ChoiceRange.first;
         J != ChoiceRange.second; ++J) {

      // Before going any further, make sure that this pair does not
      // conflict with any already-selected pairs (see comment below
      // near the Tree pruning for more details).
      DenseSet<ValuePair> ChosenPairSet;
      bool DoesConflict = false;
      for (DenseMap<Value *, Value *>::iterator C = ChosenPairs.begin(),
           E = ChosenPairs.end(); C != E; ++C) {
        if (pairsConflict(*C, *J, PairableInstUsers,
                          UseCycleCheck ? &PairableInstUserMap : 0)) {
          DoesConflict = true;
          break;
        }

        ChosenPairSet.insert(*C);
      }
      if (DoesConflict) continue;

      if (UseCycleCheck &&
          pairWillFormCycle(*J, PairableInstUserMap, ChosenPairSet))
        continue;

      DenseMap<ValuePair, size_t> Tree;
      buildInitialTreeFor(CandidatePairs, PairableInsts, ConnectedPairs,
                          PairableInstUsers, ChosenPairs, Tree, *J);

      // Because we'll keep the child with the largest depth, the largest
      // depth is still the same in the unpruned Tree.
      size_t MaxDepth = Tree.lookup(*J);

      DEBUG(if (DebugPairSelection) dbgs() << "WIV: found Tree for pair {"
                   << *J->first << " <-> " << *J->second << "} of depth " <<
                   MaxDepth << " and size " << Tree.size() << "\n");

      // At this point the Tree has been constructed, but, may contain
      // contradictory children (meaning that different children of
      // some tree node may be attempting to fuse the same instruction).
      // So now we walk the tree again, in the case of a conflict,
      // keep only the child with the largest depth. To break a tie,
      // favor the first child.

      DenseSet<ValuePair> PrunedTree;
      pruneTreeFor(CandidatePairs, PairableInsts, ConnectedPairs,
                   PairableInstUsers, PairableInstUserMap, ChosenPairs, Tree,
                   PrunedTree, *J, UseCycleCheck);

      size_t EffSize = 0;
      for (DenseSet<ValuePair>::iterator S = PrunedTree.begin(),
           E = PrunedTree.end(); S != E; ++S)
        EffSize += getDepthFactor(S->first);

      DEBUG(if (DebugPairSelection)
             dbgs() << "WIV: found pruned Tree for pair {"
             << *J->first << " <-> " << *J->second << "} of depth " <<
             MaxDepth << " and size " << PrunedTree.size() <<
            " (effective size: " << EffSize << ")\n");
#if defined LLVM_3_1      
      if (MaxDepth >= ReqChainDepth && EffSize > BestEffSize) {
#else          
      if ((VTTI || MaxDepth >= ReqChainDepth) && EffSize > BestEffSize) {          
#endif          
        BestMaxDepth = MaxDepth;
        BestEffSize = EffSize;
        BestTree = PrunedTree;
      }
    }
  }

  // Given the list of candidate pairs, this function selects those
  // that will be fused into vector instructions.
  void WIVectorize::choosePairs(
                      std::multimap<Value *, Value *> &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      std::multimap<ValuePair, ValuePair> &ConnectedPairs,
                      DenseSet<ValuePair> &PairableInstUsers,
                      DenseMap<Value *, Value *>& ChosenPairs) {
    bool UseCycleCheck = true;
    std::multimap<ValuePair, ValuePair> PairableInstUserMap;
    for (std::vector<Value *>::iterator I = PairableInsts.begin(),
         E = PairableInsts.end(); I != E; ++I) {
      // The number of possible pairings for this variable:
      size_t NumChoices = CandidatePairs.count(*I);
      if (!NumChoices) continue;

      VPIteratorPair ChoiceRange = CandidatePairs.equal_range(*I);

      // The best pair to choose and its tree:
      size_t BestMaxDepth = 0, BestEffSize = 0;
      DenseSet<ValuePair> BestTree;
      findBestTreeFor(CandidatePairs, PairableInsts, ConnectedPairs,
                      PairableInstUsers, PairableInstUserMap, ChosenPairs,
                      BestTree, BestMaxDepth, BestEffSize, ChoiceRange,
                      UseCycleCheck);

      // A tree has been chosen (or not) at this point. If no tree was
      // chosen, then this instruction, I, cannot be paired (and is no longer
      // considered).

      DEBUG(if (BestTree.size() > 0)
              dbgs() << "WIV: selected pairs in the best tree for: "
                     << *cast<Instruction>(*I) << "\n");

      for (DenseSet<ValuePair>::iterator S = BestTree.begin(),
           SE2 = BestTree.end(); S != SE2; ++S) {
        // Insert the members of this tree into the list of chosen pairs.
        ChosenPairs.insert(ValuePair(S->first, S->second));
        DEBUG(dbgs() << "WIV: selected pair: " << *S->first << " <-> " <<
               *S->second << "\n");

        // Remove all candidate pairs that have values in the chosen tree.
        for (std::multimap<Value *, Value *>::iterator K =
               CandidatePairs.begin(); K != CandidatePairs.end();) {
          if (K->first == S->first || K->second == S->first ||
              K->second == S->second || K->first == S->second) {
            // Don't remove the actual pair chosen so that it can be used
            // in subsequent tree selections.
            if (!(K->first == S->first && K->second == S->second))
              CandidatePairs.erase(K++);
            else
              ++K;
          } else {
            ++K;
          }
        }
      }
    }

    DEBUG(dbgs() << "WIV: selected " << ChosenPairs.size() << " pairs.\n");
  }

  // Returns the value that is to be used as the pointer input to the vector
  // instruction that fuses I with J.
  Value *WIVectorize::getReplacementPointerInput(LLVMContext& /*Context*/,
                     Instruction *I, Instruction *J, unsigned o,
                     bool FlipMemInputs) {
    Value *IPtr, *JPtr;
    unsigned IAlignment, JAlignment, IAddressSpace, JAddressSpace;
    int64_t OffsetInElmts;

    // Note: the analysis might fail here, that is why the pair order has
    // been precomputed (OffsetInElmts must be unused here).
    (void) getPairPtrInfo(I, J, IPtr, JPtr, IAlignment, JAlignment,
                          IAddressSpace, JAddressSpace,
                          OffsetInElmts);

    // The pointer value is taken to be the one with the lowest offset.
    Value *VPtr;
    if (!FlipMemInputs) {
      VPtr = IPtr;
    } else {
      FlipMemInputs = true;
      VPtr = JPtr;
    }
    
    // If pointer source is another bitcast, go directly to original
    // instruction.
    if (isa<BitCastInst>(VPtr)) {
        VPtr = cast<BitCastInst>(VPtr)->getOperand(0);
    }
    Type *ArgTypeI = cast<PointerType>(IPtr->getType())->getElementType();
    Type *ArgTypeJ = cast<PointerType>(JPtr->getType())->getElementType();
    Type *VArgType = getVecTypeForPair(ArgTypeI, ArgTypeJ);
    Type *VArgPtrType = PointerType::get(VArgType,
      cast<PointerType>(IPtr->getType())->getAddressSpace());
    BitCastInst* b =  new BitCastInst(VPtr, VArgPtrType, getReplacementName(I, true, o),
                        /* insert before */ FlipMemInputs ? J : I);
    
    if (I->getMetadata("wi") != NULL) {
      b->setMetadata("wi", I->getMetadata("wi"));
      b->setMetadata("wi_counter", I->getMetadata("wi_counter"));
    }
    return b;
  }

  void WIVectorize::fillNewShuffleMask(LLVMContext& Context, Instruction *J,
                     unsigned NumElem, unsigned MaskOffset, unsigned NumInElem,
                     unsigned IdxOffset, std::vector<Constant*> &Mask) {
    for (unsigned v = 0; v < NumElem/2; ++v) {
      int m = cast<ShuffleVectorInst>(J)->getMaskValue(v);
      if (m < 0) {
        Mask[v+MaskOffset] = UndefValue::get(Type::getInt32Ty(Context));
      } else {
        unsigned mm = m + (int) IdxOffset;
        if (m >= (int) NumInElem)
          mm += (int) NumInElem;

        Mask[v+MaskOffset] =
          ConstantInt::get(Type::getInt32Ty(Context), mm);
      }
    }
  }

  // Returns the value that is to be used as the vector-shuffle mask to the
  // vector instruction that fuses I with J.
  Value *WIVectorize::getReplacementShuffleMask(LLVMContext& Context,
                     Instruction *I, Instruction *J) {
    // This is the shuffle mask. We need to append the second
    // mask to the first, and the numbers need to be adjusted.

    Type *ArgTypeI = I->getType();
    Type *ArgTypeJ = J->getType();
    Type *VArgType = getVecTypeForPair(ArgTypeI, ArgTypeJ);
    // Get the total number of elements in the fused vector type.
    // By definition, this must equal the number of elements in
    // the final mask.
    unsigned NumElem = cast<VectorType>(VArgType)->getNumElements();
    std::vector<Constant*> Mask(NumElem);

    Type *OpType = I->getOperand(0)->getType();
    unsigned NumInElem = cast<VectorType>(OpType)->getNumElements();

    // For the mask from the first pair...
    fillNewShuffleMask(Context, I, NumElem, 0, NumInElem, 0, Mask);

    // For the mask from the second pair...
    fillNewShuffleMask(Context, J, NumElem, NumElem/2, NumInElem, NumInElem,
                       Mask);

    return ConstantVector::get(Mask);
  }

  Value *WIVectorize::CommonShuffleSource(Instruction *I, Instruction *J) {
      DenseMap<Value*, Value*>::iterator vi = storedSources.find(I);
      DenseMap<Value*, Value*>::iterator vj = storedSources.find(J);
      if (vi != storedSources.end() 
          && vj != storedSources.end()) {
          if ((*vi).second == (*vj).second) {
            return (*vi).second;
          }
      }
      return NULL;
  }
  // Returns the value to be used as the specified operand of the vector
  // instruction that fuses I with J.
  Value *WIVectorize::getReplacementInput(LLVMContext& Context, Instruction *I,
                     Instruction *J, unsigned o, bool FlipMemInputs) {
    Value *CV0 = ConstantInt::get(Type::getInt32Ty(Context), 0);
    Value *CV1 = ConstantInt::get(Type::getInt32Ty(Context), 1);

      // Compute the fused vector type for this operand
    Type *ArgType = I->getOperand(o)->getType();
    Type *ArgTypeJ = J->getOperand(o)->getType();
    VectorType *VArgType = getVecTypeForPair(ArgType, ArgTypeJ);
    Instruction *L = I, *H = J;
    if (FlipMemInputs) {
      L = J;
      H = I;
    }

    if (ArgType->isVectorTy()) {      
      ShuffleVectorInst *LSV
        = dyn_cast<ShuffleVectorInst>(L->getOperand(o));
      ShuffleVectorInst *HSV
        = dyn_cast<ShuffleVectorInst>(H->getOperand(o));        
      if (LSV && HSV &&
          LSV->getOperand(0)->getType() == HSV->getOperand(0)->getType() &&
          LSV->getOperand(1)->getType() == HSV->getOperand(1)->getType() &&
          LSV->getOperand(2)->getType() == HSV->getOperand(2)->getType()) {
          if (LSV->getOperand(0) == HSV->getOperand(0) &&
              LSV->getOperand(1) == HSV->getOperand(1)) {
              if (LSV->getOperand(2)->getType()->getVectorNumElements() ==
                  HSV->getOperand(2)->getType()->getVectorNumElements()) {
                unsigned elems = 
                    LSV->getOperand(2)->getType()->getVectorNumElements();
                bool continous = true;    
                bool identical = true;
                unsigned start = cast<ShuffleVectorInst>(LSV)->getMaskValue(0);
                for (unsigned i = 0; i < elems; i++) {
                    unsigned m = cast<ShuffleVectorInst>(LSV)->getMaskValue(i);
                    if (m != i) 
                        continous = false;
                    if (m != start)
                        identical = false;
                    unsigned n = cast<ShuffleVectorInst>(HSV)->getMaskValue(i);
                    if (n != i + elems)
                        continous = false;
                    if (n != start)
                        identical = false;
                }
                // This is the case where both sources come from same value and
                // are in order. e.g. 0,1,2,3,4,5,6,7, as produced when
                // replacing outputs of vector operation.
                if (continous && VArgType->getVectorNumElements() == elems*2) {
                    return LSV->getOperand(0);
                }
                // This is case where single value of input vector is replicated
                // to whole output. Eventually should turn to buildvector MI.
                if (identical) {
                    unsigned numElem = 
                        cast<VectorType>(VArgType)->getNumElements();
                    std::vector<Constant*> Mask(numElem);      
                    for (unsigned v = 0; v < numElem; ++v)
                        Mask[v] = 
                            ConstantInt::get(Type::getInt32Ty(Context), start);   
                            
                    Instruction *BV = new ShuffleVectorInst(
                            (start < numElem/2) ? 
                                LSV->getOperand(0): 
                                LSV->getOperand(1),
                            UndefValue::get(LSV->getOperand(0)->getType()),
                            ConstantVector::get(Mask),
                            getReplacementName(I, true, o));                    
                    if (LSV->getMetadata("wi") != NULL) {
                        BV->setMetadata("wi", LSV->getMetadata("wi"));
                        BV->setMetadata("wi_counter", LSV->getMetadata("wi_counter"));
                    }
                    BV->insertBefore(J);
                    return BV;                
                }
              }
          }
#if 0 
    // This was made obsolete by test for continuity of shuffle indexes above
    // and should be removed after futher tests for performance degradation.
          Value* res = CommonShuffleSource(LSV, HSV);
          if (res && 
              res->getType()->getVectorNumElements() == 
                VArgType->getVectorNumElements()) {
              return res;         
          }
#endif          
      }
      InsertElementInst *LIN
        = dyn_cast<InsertElementInst>(L->getOperand(o));
      InsertElementInst *HIN
        = dyn_cast<InsertElementInst>(H->getOperand(o));
      
      unsigned numElem = cast<VectorType>(VArgType)->getNumElements();
      if (LIN && HIN) {
          Instruction *newIn = InsertElementInst::Create(
                                          UndefValue::get(VArgType),
                                          LIN->getOperand(1), 
                                          LIN->getOperand(2),
                                          getReplacementName(I, true, o, 1));     
          if (I->getMetadata("wi")) {
            newIn->setMetadata("wi", I->getMetadata("wi"));
            newIn->setMetadata("wi_counter", I->getMetadata("wi_counter"));
          }
          newIn->insertBefore(J);
          
          LIN = dyn_cast<InsertElementInst>(LIN->getOperand(0));
          int counter = 2;
          int rounds = 0;
          while (rounds < 2) {
            while(LIN) {      
              unsigned Indx = cast<ConstantInt>(LIN->getOperand(2))->getZExtValue();
              Indx += rounds * (numElem/2);
              Value *newIndx = ConstantInt::get(Type::getInt32Ty(Context), Indx);             
              newIn = InsertElementInst::Create(
                                        newIn,
                                        LIN->getOperand(1),
                                        newIndx,
                                        getReplacementName(I, true, o ,counter));
              counter++;
              if (I->getMetadata("wi")) {
                newIn->setMetadata("wi", I->getMetadata("wi"));
                newIn->setMetadata("wi_counter", I->getMetadata("wi_counter"));
              }
              newIn->insertBefore(J);       
              LIN = dyn_cast<InsertElementInst>(LIN->getOperand(0));        
            }
            rounds ++;
            LIN = HIN;
          }       
          return newIn;
              
      }
      std::vector<Constant*> Mask(numElem);      
      for (unsigned v = 0; v < numElem; ++v)
          Mask[v] = ConstantInt::get(Type::getInt32Ty(Context), v);

      Instruction *BV = new ShuffleVectorInst(L->getOperand(o),
                                              H->getOperand(o),
                                              ConstantVector::get(Mask),
                                              getReplacementName(I, true, o));      
      if (L->getMetadata("wi") != NULL) {
        BV->setMetadata("wi", L->getMetadata("wi"));
        BV->setMetadata("wi_counter", L->getMetadata("wi_counter"));
      }
      BV->insertBefore(J);
      return BV;
    }

    // If these two inputs are the output of another vector instruction,
    // then we should use that output directly. It might be necessary to
    // permute it first. [When pairings are fused recursively, you can
    // end up with cases where a large vector is decomposed into scalars
    // using extractelement instructions, then built into size-2
    // vectors using insertelement and the into larger vectors using
    // shuffles. InstCombine does not simplify all of these cases well,
    // and so we make sure that shuffles are generated here when possible.
    ExtractElementInst *LEE
      = dyn_cast<ExtractElementInst>(L->getOperand(o));
    ExtractElementInst *HEE
      = dyn_cast<ExtractElementInst>(H->getOperand(o));

    if (LEE && HEE &&
        LEE->getOperand(0)->getType() == HEE->getOperand(0)->getType()) {
      VectorType *EEType = cast<VectorType>(LEE->getOperand(0)->getType());
      unsigned LowIndx = cast<ConstantInt>(LEE->getOperand(1))->getZExtValue();
      unsigned HighIndx = cast<ConstantInt>(HEE->getOperand(1))->getZExtValue();
      if (LEE->getOperand(0) == HEE->getOperand(0)) {
        if (LowIndx == 0 && HighIndx == 1)
          return LEE->getOperand(0);

        std::vector<Constant*> Mask(2);
        Mask[0] = ConstantInt::get(Type::getInt32Ty(Context), LowIndx);
        Mask[1] = ConstantInt::get(Type::getInt32Ty(Context), HighIndx);

        Instruction *BV = new ShuffleVectorInst(LEE->getOperand(0),
                                          UndefValue::get(EEType),
                                          ConstantVector::get(Mask),
                                          getReplacementName(I, true, o));
        if (I->getMetadata("wi") != NULL) {
          BV->setMetadata("wi", I->getMetadata("wi"));
          BV->setMetadata("wi_counter", I->getMetadata("wi_counter"));
        }       
        BV->insertBefore(J);
        return BV;
      }

      std::vector<Constant*> Mask(2);
      HighIndx += EEType->getNumElements();
      Mask[0] = ConstantInt::get(Type::getInt32Ty(Context), LowIndx);
      Mask[1] = ConstantInt::get(Type::getInt32Ty(Context), HighIndx);

      Instruction *BV = new ShuffleVectorInst(LEE->getOperand(0),
                                          HEE->getOperand(0),
                                          ConstantVector::get(Mask),
                                          getReplacementName(I, true, o));
      if (I->getMetadata("wi") != NULL) {
        BV->setMetadata("wi", I->getMetadata("wi"));
        BV->setMetadata("wi_counter", I->getMetadata("wi_counter"));
      }      
      BV->insertBefore(J);
      return BV;
    }

    Instruction *BV1 = InsertElementInst::Create(
                                          UndefValue::get(VArgType),
                                          L->getOperand(o), CV0,
                                          getReplacementName(I, true, o, 1));
    if (I->getMetadata("wi") != NULL) {
      BV1->setMetadata("wi", I->getMetadata("wi"));
      BV1->setMetadata("wi_counter", I->getMetadata("wi_counter"));
    }
    
    BV1->insertBefore(I);
    
    Instruction *BV2 = InsertElementInst::Create(BV1, H->getOperand(o),
                                          CV1,
                                          getReplacementName(I, true, o, 2));
    if (J->getMetadata("wi") != NULL) {
      BV2->setMetadata("wi",J->getMetadata("wi"));
      BV2->setMetadata("wi_counter",J->getMetadata("wi_counter"));
    }
    BV2->insertBefore(J);
    return BV2;
  }

  // This function creates an array of values that will be used as the inputs
  // to the vector instruction that fuses I with J.
  void WIVectorize::getReplacementInputsForPair(LLVMContext& Context,
                     Instruction *I, Instruction *J,
                     SmallVector<Value *, 3> &ReplacedOperands,
                     bool FlipMemInputs) {
    unsigned NumOperands = I->getNumOperands();

    for (unsigned p = 0, o = NumOperands-1; p < NumOperands; ++p, --o) {
      // Iterate backward so that we look at the store pointer
      // first and know whether or not we need to flip the inputs.

      if (isa<LoadInst>(I) || (o == 1 && isa<StoreInst>(I))) {
        // This is the pointer for a load/store instruction.
        ReplacedOperands[o] = getReplacementPointerInput(Context, I, J, o,
                                FlipMemInputs);
        continue;
      } else if (isa<CallInst>(I)) {
        Function *F = cast<CallInst>(I)->getCalledFunction();
        unsigned IID = F->getIntrinsicID();
        if (o == NumOperands-1) {
          BasicBlock &BB = *I->getParent();

          Module *M = BB.getParent()->getParent();
          Type *ArgTypeI = I->getType();
          Type *ArgTypeJ = J->getType();
          Type *VArgType = getVecTypeForPair(ArgTypeI, ArgTypeJ);

          // FIXME: is it safe to do this here?
          ReplacedOperands[o] = Intrinsic::getDeclaration(M,
            (Intrinsic::ID) IID, VArgType);
          continue;
        } else if (IID == Intrinsic::powi && o == 1) {
          // The second argument of powi is a single integer and we've already
          // checked that both arguments are equal. As a result, we just keep
          // I's second argument.
          ReplacedOperands[o] = I->getOperand(o);
          continue;
        }
      } else if (isa<ShuffleVectorInst>(I) && o == NumOperands-1) {
        ReplacedOperands[o] = getReplacementShuffleMask(Context, I, J);
        continue;
      }

      ReplacedOperands[o] =
        getReplacementInput(Context, I, J, o, FlipMemInputs);
    }
  }
  // As with the aliasing information, SCEV can also change because of
  // vectorization. This information is used to compute relative pointer
  // offsets; the necessary information will be cached here prior to
  // fusion.
  void WIVectorize::collectPtrInfo(std::vector<Value *> &PairableInsts,
                                   DenseMap<Value *, Value *> &ChosenPairs,
                                   DenseSet<Value *> &LowPtrInsts) {
    for (std::vector<Value *>::iterator PI = PairableInsts.begin(),
      PIE = PairableInsts.end(); PI != PIE; ++PI) {
      DenseMap<Value *, Value *>::iterator P = ChosenPairs.find(*PI);
      if (P == ChosenPairs.end()) continue;

      Instruction *I = cast<Instruction>(P->first);
      Instruction *J = cast<Instruction>(P->second);

      if (!isa<LoadInst>(I) && !isa<StoreInst>(I) && !isa<GetElementPtrInst>(I))
        continue;

      Value *IPtr, *JPtr;
      unsigned IAlignment, JAlignment, IAddressSpace, JAddressSpace;
      int64_t OffsetInElmts;
      if (!getPairPtrInfo(
              I, J, IPtr, JPtr, IAlignment, JAlignment, IAddressSpace, 
              JAddressSpace, OffsetInElmts) || abs64(OffsetInElmts) != 1) {
          if (!isa<GetElementPtrInst>(I))
            llvm_unreachable("Pre-fusion pointer analysis failed");
      }
      Value *LowPI = (OffsetInElmts > 0) ? I : J;
      LowPtrInsts.insert(LowPI);
    }
  }

  // This function creates two values that represent the outputs of the
  // original I and J instructions. These are generally vector shuffles
  // or extracts. In many cases, these will end up being unused and, thus,
  // eliminated by later passes.
  void WIVectorize::replaceOutputsOfPair(LLVMContext& Context, Instruction *I,
                     Instruction *J, Instruction *K,
                     Instruction *&InsertionPt,
                     Instruction *&K1, Instruction *&K2,
                     bool FlipMemInputs) {
    Value *CV0 = ConstantInt::get(Type::getInt32Ty(Context), 0);
    Value *CV1 = ConstantInt::get(Type::getInt32Ty(Context), 1);

    if (isa<StoreInst>(I)) {
      AA->replaceWithNewValue(I, K);
      AA->replaceWithNewValue(J, K);
    } else {
      Type *IType = I->getType();
      Type *JType = J->getType();

      VectorType *VType = getVecTypeForPair(IType, JType);

      if (IType->isVectorTy()) {
          unsigned numElem = cast<VectorType>(IType)->getNumElements();
          std::vector<Constant*> Mask1(numElem), Mask2(numElem);
          for (unsigned v = 0; v < numElem; ++v) {
            Mask1[v] = ConstantInt::get(Type::getInt32Ty(Context), v);
            Mask2[v] = ConstantInt::get(Type::getInt32Ty(Context), numElem+v);
          }

          K1 = new ShuffleVectorInst(K, UndefValue::get(VType),
                                       ConstantVector::get(
                                         FlipMemInputs ? Mask2 : Mask1),
                                       getReplacementName(K, false, 1));
          K2 = new ShuffleVectorInst(K, UndefValue::get(VType),
                                       ConstantVector::get(
                                         FlipMemInputs ? Mask1 : Mask2),
                                       getReplacementName(K, false, 2));
            storedSources.insert(ValuePair(FlipMemInputs ? K1 : K2, K));
            storedSources.insert(ValuePair(FlipMemInputs ? K2 : K1, K)); 
            flippedStoredSources.insert(ValuePair(K, FlipMemInputs ? K1 : K2));
            flippedStoredSources.insert(ValuePair(K, FlipMemInputs ? K2 : K1));
            Instruction* L = I;
            Instruction* H = J;
            if (FlipMemInputs) {
                L = J;
                H = I;
            }
            VPIteratorPair v1 = 
                flippedStoredSources.equal_range(L);
            for (std::multimap<Value*, Value*>::iterator ii = v1.first;
                 ii != v1.second; ii++) {        
                storedSources.erase((*ii).second);            
                storedSources.insert(ValuePair((*ii).second,K));
                flippedStoredSources.insert(ValuePair(K, (*ii).second));
                storedSources.erase(L);
            }
            flippedStoredSources.erase(L);              
            VPIteratorPair v2 = flippedStoredSources.equal_range(H);
            for (std::multimap<Value*, Value*>::iterator ji = v2.first;
                 ji != v2.second; ji++) {        
                storedSources.erase((*ji).second);
                storedSources.insert(ValuePair((*ji).second,K));
                flippedStoredSources.insert(ValuePair(K, (*ji).second));            
                storedSources.erase(H);
            }
            flippedStoredSources.erase(H);                        
      } else {
        K1 = ExtractElementInst::Create(K, FlipMemInputs ? CV1 : CV0,
                                          getReplacementName(K, false, 1));
        K2 = ExtractElementInst::Create(K, FlipMemInputs ? CV0 : CV1,
                                          getReplacementName(K, false, 2));
        storedSources.insert(ValuePair(K1,K));
        storedSources.insert(ValuePair(K2,K));    
        flippedStoredSources.insert(ValuePair(K, K1));
        flippedStoredSources.insert(ValuePair(K, K2));
      }
      if (I->getMetadata("wi") != NULL) {
        K1->setMetadata("wi", I->getMetadata("wi"));
        K1->setMetadata("wi_counter", I->getMetadata("wi_counter"));
      }
      if (J->getMetadata("wi") != NULL) {
        K2->setMetadata("wi", J->getMetadata("wi"));          
        K2->setMetadata("wi_counter", J->getMetadata("wi_counter"));
      }
      
      K1->insertAfter(K);
      K2->insertAfter(K1);
      InsertionPt = K2;
    }
  }

  // Move all uses of the function I (including pairing-induced uses) after J.
  void WIVectorize::moveUsesOfIAfterJ(BasicBlock &/*BB*/,
                     std::multimap<Value *, Value *> &LoadMoveSet,
                     Instruction *&InsertionPt,
                     Instruction *I, Instruction *J) {
    // Skip to the first instruction past I.
    BasicBlock::iterator L = llvm::next(BasicBlock::iterator(I));

    DenseSet<Value *> Users;
    AliasSetTracker WriteSet(*AA);
    for (; cast<Instruction>(L) != J;) {
      if (trackUsesOfI(Users, WriteSet, I, L, true, &LoadMoveSet)) {
        // Move this instruction
        Instruction *InstToMove = L; ++L;

        InstToMove->removeFromParent();
        InstToMove->insertAfter(InsertionPt);
        InsertionPt = InstToMove;
      } else {
        ++L;
      }
    }
  }


  // Collect all load instruction that are in the move set of a given first
  // pair member.  These loads depend on the first instruction, I, and so need
  // to be moved after J (the second instruction) when the pair is fused.
  void WIVectorize::collectPairLoadMoveSet(BasicBlock &BB,
                     DenseMap<Value *, Value *> &/*ChosenPairs*/,
                     std::multimap<Value *, Value *> &LoadMoveSet,
                     Instruction *I) {
    // Skip to the first instruction past I.
    BasicBlock::iterator L = llvm::next(BasicBlock::iterator(I));

    DenseSet<Value *> Users;
    AliasSetTracker WriteSet(*AA);

    // Note: We cannot end the loop when we reach J because J could be moved
    // farther down the use chain by another instruction pairing. Also, J
    // could be before I if this is an inverted input.
    for (BasicBlock::iterator E = BB.end(); cast<Instruction>(L) != E; ++L) {
      if (trackUsesOfI(Users, WriteSet, I, L)) {
        if (L->mayReadFromMemory())
          LoadMoveSet.insert(ValuePair(L, I));
      }
    }
  }

  // In cases where both load/stores and the computation of their pointers
  // are chosen for vectorization, we can end up in a situation where the
  // aliasing analysis starts returning different query results as the
  // process of fusing instruction pairs continues. Because the algorithm
  // relies on finding the same use trees here as were found earlier, we'll
  // need to precompute the necessary aliasing information here and then
  // manually update it during the fusion process.
  void WIVectorize::collectLoadMoveSet(BasicBlock &BB,
                     std::vector<Value *> &PairableInsts,
                     DenseMap<Value *, Value *> &ChosenPairs,
                     std::multimap<Value *, Value *> &LoadMoveSet) {
    for (std::vector<Value *>::iterator PI = PairableInsts.begin(),
         PIE = PairableInsts.end(); PI != PIE; ++PI) {
      DenseMap<Value *, Value *>::iterator P = ChosenPairs.find(*PI);
      if (P == ChosenPairs.end()) continue;

      Instruction *I = cast<Instruction>(P->first);
      collectPairLoadMoveSet(BB, ChosenPairs, LoadMoveSet, I);    
    }
  }

  // This function fuses the chosen instruction pairs into vector instructions,
  // taking care preserve any needed scalar outputs and, then, it reorders the
  // remaining instructions as needed (users of the first member of the pair
  // need to be moved to after the location of the second member of the pair
  // because the vector instruction is inserted in the location of the pair's
  // second member).
  void WIVectorize::fuseChosenPairs(BasicBlock &BB,
                     std::vector<Value *> &PairableInsts,
                     DenseMap<Value *, Value *> &ChosenPairs) {
    LLVMContext& Context = BB.getContext();

    // During the vectorization process, the order of the pairs to be fused
    // could be flipped. So we'll add each pair, flipped, into the ChosenPairs
    // list. After a pair is fused, the flipped pair is removed from the list.
    std::vector<ValuePair> FlippedPairs;
    FlippedPairs.reserve(ChosenPairs.size());
    for (DenseMap<Value *, Value *>::iterator P = ChosenPairs.begin(),
         E = ChosenPairs.end(); P != E; ++P)
      FlippedPairs.push_back(ValuePair(P->second, P->first));
    for (std::vector<ValuePair>::iterator P = FlippedPairs.begin(),
         E = FlippedPairs.end(); P != E; ++P)
      ChosenPairs.insert(*P);

    std::multimap<Value *, Value *> LoadMoveSet;
    collectLoadMoveSet(BB, PairableInsts, ChosenPairs, LoadMoveSet);
    DenseSet<Value *> LowPtrInsts;
    collectPtrInfo(PairableInsts, ChosenPairs, LowPtrInsts);
    
    DEBUG(dbgs() << "WIV: initial: \n" << BB << "\n");

    for (BasicBlock::iterator PI = BB.getFirstInsertionPt(); PI != BB.end();) {
      DenseMap<Value *, Value *>::iterator P = ChosenPairs.find(PI);
      if (P == ChosenPairs.end()) {
        ++PI;
        continue;
      }

      if (getDepthFactor(P->first) == 0) {
        // These instructions are not really fused, but are tracked as though
        // they are. Any case in which it would be interesting to fuse them
        // will be taken care of by InstCombine.
        --NumFusedOps;
        ++PI;
        continue;
      }

      Instruction *I = cast<Instruction>(P->first),
        *J = cast<Instruction>(P->second);

      DEBUG(dbgs() << "WIV: fusing: " << *I <<
             " <-> " << *J << "\n");

      // Remove the pair and flipped pair from the list.
      DenseMap<Value *, Value *>::iterator FP = ChosenPairs.find(P->second);
      assert(FP != ChosenPairs.end() && "Flipped pair not found in list");
      ChosenPairs.erase(FP);
      ChosenPairs.erase(P);

      bool FlipMemInputs = false;
      if (isa<LoadInst>(I) || isa<StoreInst>(I) || isa<GetElementPtrInst>(I))
        FlipMemInputs = (LowPtrInsts.find(I) == LowPtrInsts.end());
      unsigned NumOperands = I->getNumOperands();
      SmallVector<Value *, 3> ReplacedOperands(NumOperands);
      getReplacementInputsForPair(Context, I, J, ReplacedOperands,
        FlipMemInputs);

      // Make a copy of the original operation, change its type to the vector
      // type and replace its operands with the vector operands.      
      Instruction *K = I->clone();
      if (I->hasName()) K->takeName(I);
      
      if (I->getMetadata("wi") != NULL) {
          K->setMetadata("wi", I->getMetadata("wi"));
          K->setMetadata("wi_counter", I->getMetadata("wi_counter"));
      }
      if (!isa<StoreInst>(K))
        K->mutateType(getVecTypeForPair(I->getType(), J->getType()));

      for (unsigned o = 0; o < NumOperands; ++o)
        K->setOperand(o, ReplacedOperands[o]);

      // If we've flipped the memory inputs, make sure that we take the correct
      // alignment.
      if (FlipMemInputs) {
        if (isa<StoreInst>(K))
          cast<StoreInst>(K)->setAlignment(cast<StoreInst>(J)->getAlignment());
        else
          cast<LoadInst>(K)->setAlignment(cast<LoadInst>(J)->getAlignment());
      }

      K->insertAfter(J);

      // Instruction insertion point:
      Instruction *InsertionPt = K;
      Instruction *K1 = 0, *K2 = 0;
      replaceOutputsOfPair(Context, I, J, K, InsertionPt, K1, K2,
        FlipMemInputs);

      // The use tree of the first original instruction must be moved to after
      // the location of the second instruction. The entire use tree of the
      // first instruction is disjoint from the input tree of the second
      // (by definition), and so commutes with it.

      moveUsesOfIAfterJ(BB, LoadMoveSet, InsertionPt, I, J);

      if (!isa<StoreInst>(I)) {
        I->replaceAllUsesWith(K1);
        J->replaceAllUsesWith(K2);
        AA->replaceWithNewValue(I, K1);
        AA->replaceWithNewValue(J, K2);
      }

      // Instructions that may read from memory may be in the load move set.
      // Once an instruction is fused, we no longer need its move set, and so
      // the values of the map never need to be updated. However, when a load
      // is fused, we need to merge the entries from both instructions in the
      // pair in case those instructions were in the move set of some other
      // yet-to-be-fused pair. The loads in question are the keys of the map.
      if (I->mayReadFromMemory()) {
        std::vector<ValuePair> NewSetMembers;
        VPIteratorPair IPairRange = LoadMoveSet.equal_range(I);
        VPIteratorPair JPairRange = LoadMoveSet.equal_range(J);
        for (std::multimap<Value *, Value *>::iterator N = IPairRange.first;
             N != IPairRange.second; ++N)
          NewSetMembers.push_back(ValuePair(K, N->second));
        for (std::multimap<Value *, Value *>::iterator N = JPairRange.first;
             N != JPairRange.second; ++N)
          NewSetMembers.push_back(ValuePair(K, N->second));
        for (std::vector<ValuePair>::iterator A = NewSetMembers.begin(),
             AE = NewSetMembers.end(); A != AE; ++A)
          LoadMoveSet.insert(*A);
      }

      // Before removing I, set the iterator to the next instruction.
      PI = llvm::next(BasicBlock::iterator(I));
      if (cast<Instruction>(PI) == J)
        ++PI;

      SE->forgetValue(I);
      SE->forgetValue(J);
      I->eraseFromParent();
      J->eraseFromParent();
    }

    DEBUG(dbgs() << "WIV: final: \n" << BB << "\n");
  }
  void WIVectorize::dropUnused(BasicBlock& BB) {
    bool changed;
    do{
        BasicBlock::iterator J = BB.end();        
        BasicBlock::iterator I = llvm::prior(J);
        changed = false;
        while (I != BB.begin()) {
        
        if (isa<ShuffleVectorInst>(*I) ||
            isa<ExtractElementInst>(*I) ||
            isa<InsertElementInst>(*I) ||
            isa<BitCastInst>(*I)) {
            
            Value* V = dyn_cast<Value>(&(*I));
            
            if (V && V->use_empty()) {
                SE->forgetValue(&(*I));
                (*I).eraseFromParent();
                // removed instruction could have messed up things
                // start again from the end
                I = BB.end();
                J = llvm::prior(I);
                changed = true;
            } else {
                J = llvm::prior(I);      		
            }	  
        } else {
            J = llvm::prior(I);      		
        }
        I = J;      
        }
    } while (changed);
  }
  
  // Replace uses of alloca with new alloca.
  // This includes getelementpointer, bitcast, load and store only
  // atm.
  // In case original alloca was array, the getelementpointer and bitcast apply.
  void WIVectorize::replaceUses(BasicBlock& BB,
                                AllocaInst& oldAlloca, 
                                AllocaInst& newAlloca, 
                                int indx) {
      
    LLVMContext& Context = BB.getContext();          
    Instruction::use_iterator useiter = oldAlloca.use_begin();                

    while (useiter != oldAlloca.use_end()) {
        llvm::User* tmp = *useiter;
        
        if (isa<BitCastInst>(tmp)) {
            // Create new bitcast from new alloca to same type
            // as old bitcast had. This is situation where the 
            // alloca is casted to i8* followed by
            //  call void @llvm.lifetime.start(i64 -1, i8* %XYZ) nounwind
            BitCastInst* bitCast = cast<BitCastInst>(tmp);
            IRBuilder<> builder(bitCast);               
            BitCastInst* newBitcast = 
                cast<BitCastInst>(builder.CreateBitCast(
                    &newAlloca, bitCast->getDestTy(), bitCast->getName()));
                
            if (bitCast->getMetadata("wi") != NULL) {
                newBitcast->setMetadata("wi", bitCast->getMetadata("wi"));
                newBitcast->setMetadata("wi_counter", bitCast->getMetadata("wi_counter"));
            }
                
            bitCast->replaceAllUsesWith(newBitcast);
            AA->replaceWithNewValue(bitCast, newBitcast);      
            SE->forgetValue(bitCast);
            bitCast->eraseFromParent();                            
            
            useiter = oldAlloca.use_begin();
            continue;
        }
        
        if (isa<GetElementPtrInst>(tmp)) {
            // Original getelementpointer contains number of indexes
            // that indicate how to access element of allocated
            // memory. Since we changed the most inner type to
            // array, we add index to that array such as:
            // Original alloca:
            // %A = alloca [20 x [8 x i32]], align 4
            // Original getelementpointer:
            // %68 = getelementptr inbounds [20 x [8 x i32]]]* %A, i32 0, i32 %X, i32 0
            // New alloca:
            // %A = alloca [20 x [8 x [2 x i32]]], align 4
            // new getelementpointer:
            // %68 = getelementptr inbounds [20 x [8 x [2 x i32]]]* %A, i32 0, i32 %X, i32 0, i32 0
            
            GetElementPtrInst* gep = cast<GetElementPtrInst>(tmp);
            std::vector<llvm::Value *> gepArgs;            
            // Collect original indexes of getelementpointer
            for (unsigned int i = 1; i <= gep->getNumIndices(); i++) {
                gepArgs.push_back(gep->getOperand(i));
            }
            // Add index to the newly created array
            Value *V = ConstantInt::get(Type::getInt32Ty(Context), indx);
            gepArgs.push_back(V);
            IRBuilder<> builder(gep);   
            GetElementPtrInst* newGep = 
                cast<GetElementPtrInst>(
                    builder.CreateGEP(&newAlloca, gepArgs, gep->getName()));
            newGep->setIsInBounds(gep->isInBounds());
            
            if (gep->getMetadata("wi") != NULL) {
                newGep->setMetadata("wi", gep->getMetadata("wi"));
                newGep->setMetadata("wi_counter", gep->getMetadata("wi_counter"));
            }
            
            gep->replaceAllUsesWith(newGep);
            AA->replaceWithNewValue(gep, newGep);      
            SE->forgetValue(gep);
            gep->eraseFromParent();            
            useiter = oldAlloca.use_begin();
            continue;
        }
        if (isa<StoreInst>(tmp)) {
            // This is tricky, original alloca was for base type such 
            // as i32 or float so the variable was used directly.
            // Now this is array so we have to add getelementpointer.
            StoreInst* store = cast<StoreInst>(tmp);
            std::vector<llvm::Value *> gepArgs;            
            Value *V = ConstantInt::get(Type::getInt32Ty(Context), indx);
            gepArgs.push_back(V);
            IRBuilder<> builder(store);   
            GetElementPtrInst* newGep = 
                cast<GetElementPtrInst>(builder.CreateGEP(&newAlloca, gepArgs));
            if (store->getMetadata("wi") != NULL) {
                newGep->setMetadata("wi", store->getMetadata("wi"));
                newGep->setMetadata("wi_counter", store->getMetadata("wi_counter"));
            }

            for (unsigned int i = 0; i < store->getNumOperands(); i++) {
                // Either of store operands could be alloca, we either
                // store to allocated memory, or we are storing the pointer 
                // of the memory (this is rather dumb thing to do).
                if (store->getOperand(i) == &oldAlloca) {
                    IRBuilder<> builder(store);               
                    BitCastInst* newBitcast = 
                        cast<BitCastInst>(builder.CreateBitCast(
                            newGep, store->getOperand(i)->getType()));                    
                    if (store->getMetadata("wi") != NULL) {
                        newBitcast->setMetadata("wi", store->getMetadata("wi"));
                        newBitcast->setMetadata("wi_counter", store->getMetadata("wi_counter"));
                    }                    
                    store->setOperand(i, newBitcast);
                }
            }
            useiter = oldAlloca.use_begin();
            continue;            
        }
        if (isa<LoadInst>(tmp)) {
            // This is tricky, original alloca was for base type such 
            // as i32 or float so the variable was used directly.
            // Now this is array so we have to add getelementpointer.

            LoadInst* load = cast<LoadInst>(tmp);
            std::vector<llvm::Value *> gepArgs;            
            Value *V = ConstantInt::get(Type::getInt32Ty(Context), indx);
            gepArgs.push_back(V);
            IRBuilder<> builder(load);   
            GetElementPtrInst* newGep = 
                cast<GetElementPtrInst>(builder.CreateGEP(&newAlloca, gepArgs));
            if (load->getMetadata("wi") != NULL) {
                newGep->setMetadata("wi", load->getMetadata("wi"));
                newGep->setMetadata("wi_counter", load->getMetadata("wi_counter"));
            }

            for (unsigned int i = 0; i < load->getNumOperands(); i++) {
                // Find operand of load that was old alloca and 
                // use bitcast to point to to getelementpointer result.
                // There must be better way how to do this.
                if (load->getOperand(i) == &oldAlloca) {
                    IRBuilder<> builder(load);               
                    BitCastInst* newBitcast = 
                    cast<BitCastInst>(builder.CreateBitCast(
                        newGep, load->getOperand(i)->getType()));                    
                    if (load->getMetadata("wi") != NULL) {
                        newBitcast->setMetadata("wi", load->getMetadata("wi"));
                        newBitcast->setMetadata("wi_counter", load->getMetadata("wi_counter"));
                    }                    
                    load->setOperand(i, newBitcast);
                }
            }
            useiter = oldAlloca.use_begin();
            continue;            
        }        
        useiter++;
    }      
  }
  
  // Find new type for the vector alloca instruction
  Type* WIVectorize::newAllocaType(Type* start, unsigned int width) {
      
      if (start->isArrayTy()) {
          // If type is still array check what is allocated type
          int numElm = cast<ArrayType>(start)->getNumElements();
          return ArrayType::get(
                    newAllocaType(
                        cast<SequentialType>(start)->getElementType(),
                        width)
                    , numElm);
      } else if (start->isFirstClassType() && !start->isPointerTy()) {
          // Recursion stopping point
          // This should convert i32 to [width x i32] as base type of 
          // array
          return ArrayType::get(start, width);
      } else {
          // Not recognized type, just return it, alloca won't be replaced
          return start;
      }
  }
  
  // In case there is private variable in the kernel that does not fit into
  // register (multidimensional array for example), there are alloca 
  // defined to create necessary memory space for variable.
  // Those are defined then for each of the work items replicated.
  // This pass attempts to combine those allocas to create 'interleaved'
  // memory allocation that then can be accessed by vector loads and stores
  // as described bellow:
  //
  // __kernel xyz() {
  //
  // int A[100][100][100][100];
  // ...
  //}
  // Will become after replication with 2 work items:
  //
  // %A = alloca  [100 x [100 x [100 x i32]]], align 4
  // %A_wi_1_0_0 = alloca  [100 x [100 x [100 x i32]]], align 4  
  //
  // This in will be converted here to :
  // %A = alloca  [100 x [100 x [100 x [2 x i32]]]], align 4
  // And respective getelementpointer instruction will
  // be added additional paramter to select correct member from the pair.
  //
  // NOTE: This does work only for arrays ATM, the scalar type allocas
  // as produced by phistoallocas pass required for the work loops
  // are skipped for now.
  
  bool WIVectorize::vectorizeAllocas(BasicBlock& BB) {

    std::multimap<int, ValueVector*> allocas;
    getCandidateAllocas(BB, allocas);
    bool changed = false;
    
    for (std::multimap<int, ValueVector*>::iterator insIt = allocas.begin();
         insIt != allocas.end(); insIt++) {
        IRBuilder<> builder(
            BB.getParent()->getEntryBlock().getFirstInsertionPt());    
        
        ValueVector* tmpVec = (*insIt).second;
        // Create as 'wide' alloca as number of elements found,
        // could be smaller then vector width or larger.
        // Should be same as work group dimensions for work item replicas or
        // same as number of unrolled loops with work item loops.
        unsigned int allocaWidth = tmpVec->size();
        // No point vectorizing one alloca only
        if (allocaWidth <= 1)
            continue;
        
        AllocaInst* I = cast<AllocaInst>((*tmpVec)[0]);
        Type* startType = I->getAllocatedType();
        if (!startType->isArrayTy())
            continue;
        // Find new type for alloca by recursively searching through multiple
        // dimensions of array
        Type* newType = newAllocaType(startType, allocaWidth);

        // No new type was found, alloca type not supported.
        if (newType == startType)
            continue;
        
        changed = true;
        llvm::AllocaInst *alloca = 
            builder.CreateAlloca(newType, 0, I->getName().str() + "_allocamix");
        alloca->setAlignment(I->getAlignment());
        
        if (I->getMetadata("wi") != NULL) {
            alloca->setMetadata("wi", I->getMetadata("wi"));
            alloca->setMetadata("wi_counter", I->getMetadata("wi_counter"));
        }
        
        // Replace uses of first alloca with newly created one
        MDNode* mi = I->getMetadata("wi");
        assert(mi->getNumOperands() == 3);
        // Second operand of MDNode contains MDNode with XYZ tripplet.
        MDNode* iXYZ= dyn_cast<MDNode>(mi->getOperand(2));
        assert(iXYZ->getNumOperands() == 4);
        
        int index = dyn_cast<ConstantInt>(iXYZ->getOperand(1))->getZExtValue();        
        
        replaceUses(BB, *I, *alloca, index);
        SE->forgetValue(I);
        I->eraseFromParent();
        
        // Replaces uses of other allocas with newly created one
        for (unsigned int i = 1; i < allocaWidth; i++) {
            AllocaInst* J = cast<AllocaInst>((*tmpVec)[i]);
            MDNode* mj = J->getMetadata("wi");
            assert(mj->getNumOperands() == 3);
            MDNode* jXYZ= dyn_cast<MDNode>(mj->getOperand(2));
            assert(jXYZ->getNumOperands() == 4);            
            int index = 
                dyn_cast<ConstantInt>(jXYZ->getOperand(1))->getZExtValue();        
            
            replaceUses(BB, *J, *alloca, index);
            SE->forgetValue(J);
            J->eraseFromParent();            
        }
    }
    return changed;
  } 
  
  // Pass closely repated to getCandidatePairs, except this one only
  // picks AllocaInst and makes sure they are from different work items.
  // It also returns all instances of AllocaInst at the same time.
  bool WIVectorize::getCandidateAllocas(BasicBlock &BB,
                std::multimap<int, ValueVector*>& temporary) {
      
    BasicBlock::iterator Start = BB.getFirstInsertionPt();      
    BasicBlock::iterator E = BB.end();
    for (BasicBlock::iterator I = Start++; I != E; ++I) {

        if (!isa<AllocaInst>(I)) 
        continue;
        // TODO: This is bit tricky, should it be possible
        // to create vector of allocas that do not have metadata?
        if (I->getMetadata("wi") == NULL)
            continue;
        
        MDNode* md = I->getMetadata("wi");
        MDNode* mdCounter = I->getMetadata("wi_counter");
        MDNode* mdRegion = dyn_cast<MDNode>(md->getOperand(1));
        
        unsigned CI = cast<ConstantInt>(mdCounter->getOperand(1))->getZExtValue();
        unsigned RI = cast<ConstantInt>(mdRegion->getOperand(1))->getZExtValue();
        
        std::multimap<int,ValueVector*>::iterator itb = temporary.lower_bound(CI);
        std::multimap<int,ValueVector*>::iterator ite = temporary.upper_bound(CI);
        ValueVector* tmpVec = NULL;   
        while(itb != ite) {
            if (I->isSameOperationAs(cast<Instruction>((*(*itb).second)[0]))) {
                // Test also if instructions are from same region.
                MDNode* tmpMD = 
                    cast<Instruction>((*(*itb).second)[0])->getMetadata("wi");
                MDNode* tmpRINode = dyn_cast<MDNode>(tmpMD->getOperand(1));
                unsigned tmpRI = 
                    cast<ConstantInt>(tmpRINode->getOperand(1))->getZExtValue();                
                if (RI == tmpRI)
                    tmpVec = (*itb).second;
            }
            itb++;
        }
        if (tmpVec == NULL) {
            tmpVec = new ValueVector;
            temporary.insert(std::pair<int, ValueVector*>(CI, tmpVec));          
        }
        tmpVec->push_back(I);
    }
    for (std::multimap<int, ValueVector*>::iterator insIt = temporary.begin();
         insIt != temporary.end(); insIt++) {
        ValueVector* tmpVec = (*insIt).second;
        for (unsigned j = 0; j < tmpVec->size()/2; j++) {
            Instruction* I = cast<Instruction>((*tmpVec)[2*j]);
            Instruction* J = cast<Instruction>((*tmpVec)[2*j+1]);
            if (!areInstsCompatibleFromDifferentWi(I,J))
                continue;
        }
    }
    return true;
  }
  
}
char WIVectorize::ID = 0;
RegisterPass<WIVectorize>
  X("wi-vectorize", "Work item vectorization.");

FunctionPass *createWIVectorizePass() {
  return new WIVectorize();
}

