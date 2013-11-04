//===--- DecomposeVectors.cpp - Decompose vector operations ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "decompose-vectors"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/InstVisitor.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace llvm {
void initializeDecomposeVectorsPass(PassRegistry&);
};

namespace {
FunctionPass *createDecomposeVectorsPass();

// Used to store the scattered form of a vector.
typedef SmallVector<Value *, 8> ValueVector;

// Used to map a vector Value to its scattered form.  We use std::map
// because we want iterators to persist across insertion and because the
// values are relatively large.
typedef std::map<Value *, ValueVector> ScatterMap;

// Lists Instructions that have been replaced with scalar implementations,
// along with a pointer to their scattered forms.
typedef SmallVector<std::pair<Instruction *, ValueVector *>, 16> GatherList;

// Provides a very limited vector-like interface for lazily accessing one
// component of a scattered vector or vector pointer.
class Scatterer {
public:
  // Scatter V into Size components.  If new instructions are needed,
  // insert them before BBI in BB.  If Cache is nonnull, use it to cache
  // the results.
  Scatterer(BasicBlock *bb, BasicBlock::iterator bbi, Value *v,
            ValueVector *cachePtr = 0);

  // Return component I, creating a new Value for it if necessary.
  Value *operator[](unsigned I);

  // Return the number of components.
  unsigned size() const { return Size; }

private:
  BasicBlock *BB;
  BasicBlock::iterator BBI;
  Value *V;
  ValueVector *CachePtr;
  PointerType *PtrTy;
  ValueVector Tmp;
  unsigned Size;
};

// FCmpSpliiter(FCI)(Builder, X, Y, Name) uses Builder to create an FCmp
// called Name that compares X and Y in the same way as FCI.
struct FCmpSplitter {
  FCmpSplitter(FCmpInst &fci) : FCI(fci) {}
  Value *operator()(IRBuilder<> &Builder, Value *Op0, Value *Op1,
                    const Twine &Name) const {
    return Builder.CreateFCmp(FCI.getPredicate(), Op0, Op1, Name);
  }
  FCmpInst &FCI;
};

// FCmpSpliiter(FCI)(Builder, X, Y, Name) uses Builder to create an ICmp
// called Name that compares X and Y in the same way as ICI.
struct ICmpSplitter {
  ICmpSplitter(ICmpInst &ici) : ICI(ici) {}
  Value *operator()(IRBuilder<> &Builder, Value *Op0, Value *Op1,
                    const Twine &Name) const {
    return Builder.CreateICmp(ICI.getPredicate(), Op0, Op1, Name);
  }
  ICmpInst &ICI;
};

// BinarySpliiter(BO)(Builder, X, Y, Name) uses Builder to create
// a binary operator like BO called Name with operands X and Y.
struct BinarySplitter {
  BinarySplitter(BinaryOperator &bo) : BO(bo) {}
  Value *operator()(IRBuilder<> &Builder, Value *Op0, Value *Op1,
                    const Twine &Name) const {
    return Builder.CreateBinOp(BO.getOpcode(), Op0, Op1, Name);
  }
  BinaryOperator &BO;
};

class DecomposeVectors : public FunctionPass,
                         public InstVisitor<DecomposeVectors, bool> {
public:
  static char ID;

  DecomposeVectors() :
    FunctionPass(ID) {
#if 0
    initializeDecomposeVectorsPass(*PassRegistry::getPassRegistry());
#endif
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual bool runOnFunction(Function &F);

  // InstVisitor methods.  They return true if the instruction was decomposed,
  // false if nothing changed.
  bool visitInstruction(Instruction &) { return false; }
  bool visitSelectInst(SelectInst &SI);
  bool visitICmpInst(ICmpInst &);
  bool visitFCmpInst(FCmpInst &);
  bool visitBinaryOperator(BinaryOperator &);
  bool visitCastInst(CastInst &);
  bool visitBitCastInst(BitCastInst &);
  bool visitShuffleVectorInst(ShuffleVectorInst &);
  bool visitPHINode(PHINode &);
  bool visitLoadInst(LoadInst &);
  bool visitStoreInst(StoreInst &);

private:
  Scatterer scatter(Instruction *, Value *);
  void gather(Instruction *, const ValueVector &);
  bool finish();

  template<typename T> bool splitBinary(Instruction &, const T &);

  ScatterMap Scattered;
  GatherList Gathered;
};

char DecomposeVectors::ID = 0;
} // end anonymous namespace

static cl::opt<bool> DecomposeVectorLoadStore
  ("decompose-vector-load-store", cl::Hidden, cl::init(false),
   cl::desc("Allow the decompose-vectors pass to decompose loads and store"));

static RegisterPass<DecomposeVectors> X("decompose-vectors", 
                                        "Decompose vector operations into smaller pieces",
                                        false, false);
#if 0
INITIALIZE_PASS(DecomposeVectors, "decompose-vectors",
                "Decompose vector operations into smaller pieces",
                false, false)
#endif

Scatterer::Scatterer(BasicBlock *bb, BasicBlock::iterator bbi, Value *v,
                     ValueVector *cachePtr)
  : BB(bb), BBI(bbi), V(v), CachePtr(cachePtr) {
  Type *Ty = V->getType();
  PtrTy = dyn_cast<PointerType>(Ty);
  if (PtrTy)
    Ty = PtrTy->getElementType();
  Size = Ty->getVectorNumElements();
  if (!CachePtr)
    Tmp.resize(Size, 0);
  else if (CachePtr->empty())
    CachePtr->resize(Size, 0);
  else
    assert(Size == CachePtr->size() && "Inconsistent vector sizes");
}

// Return component I, creating a new Value for it if necessary.
Value *Scatterer::operator[](unsigned I) {
  ValueVector &CV = (CachePtr ? *CachePtr : Tmp);
  // Try to reuse a previous value.
  if (CV[I])
    return CV[I];
  IRBuilder<> Builder(BB, BBI);
  if (PtrTy) {
    if (!CV[0]) {
      Type *Ty =
        PointerType::get(PtrTy->getElementType()->getVectorElementType(),
                         PtrTy->getAddressSpace());
      CV[0] = Builder.CreateBitCast(V, Ty, V->getName() + ".i0");
    }
    if (I != 0)
      CV[I] = Builder.CreateConstGEP1_32(CV[0], I,
                                         V->getName() + ".i" + Twine(I));
  } else {
    // Search through a chain of InsertElementInsts looking for element I.
    // Record other elements in the cache.  The new V is still suitable
    // for all uncached indices.
    for (;;) {
      InsertElementInst *Insert = dyn_cast<InsertElementInst>(V);
      if (!Insert)
        break;
      ConstantInt *Idx = dyn_cast<ConstantInt>(Insert->getOperand(2));
      if (!Idx)
        break;
      unsigned J = Idx->getZExtValue();
      CV[J] = Insert->getOperand(1);
      V = Insert->getOperand(0);
      if (I == J)
        return CV[J];
    }
    CV[I] = Builder.CreateExtractElement(V, Builder.getInt32(I),
                                         V->getName() + ".i" + Twine(I));
  }
  return CV[I];
}

void DecomposeVectors::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetTransformInfo>();
  FunctionPass::getAnalysisUsage(AU);
}

bool DecomposeVectors::runOnFunction(Function &F) {
  //  const TargetTransformInfo *TTI = &getAnalysis<TargetTransformInfo>();
  for (Function::iterator BBI = F.begin(), BBE = F.end(); BBI != BBE; ++BBI) {
    BasicBlock *BB = BBI;
    for (BasicBlock::iterator II = BB->begin(), IE = BB->end(); II != IE;) {
      Instruction *I = II;
      bool Done = visit(I);
      ++II;
      if (Done && I->getType()->isVoidTy())
        I->eraseFromParent();
    }
  }
  return finish();
}

// Return a scattered form of V that can be accessed by Point.  V must be a
// vector or a pointer to a vector.
Scatterer DecomposeVectors::scatter(Instruction *Point, Value *V) {
  if (Argument *VArg = dyn_cast<Argument>(V)) {
    // Put the scattered form of arguments in the entry block,
    // so that it can be used everywhere.
    Function *F = VArg->getParent();
    BasicBlock *BB = &F->getEntryBlock();
    return Scatterer(BB, BB->begin(), V, &Scattered[V]);
  }
  if (Instruction *VOp = dyn_cast<Instruction>(V)) {
    // Put the scattered form of an instruction directly after the
    // instruction.
    BasicBlock *BB = VOp->getParent();
    return Scatterer(BB, llvm::next(BasicBlock::iterator(VOp)),
                     V, &Scattered[V]);
  }
  // In the fallback case, just put the scattered before Point and
  // keep the result local to Point.
  return Scatterer(Point->getParent(), Point, V);
}

// Replace Op with the gathered form of the components in CV.  Defer the
// deletion of Op and creation of the gathered form to the end of the pass,
// so that we can avoid creating the gathered form if all uses of Op are
// replaced with uses of CV.
void DecomposeVectors::gather(Instruction *Op, const ValueVector &CV) {
  // Since we're not deleting Op yet, stub out its operands, so that it
  // doesn't make anything live unnecessarily.
  for (unsigned I = 0, E = Op->getNumOperands(); I != E; ++I)
    Op->setOperand(I, UndefValue::get(Op->getOperand(I)->getType()));

  // If we already have a scattered form of Op (created from ExtractElements
  // of Op itself), replace them with the new form.
  ValueVector &SV = Scattered[Op];
  if (!SV.empty()) {
    for (unsigned I = 0, E = SV.size(); I != E; ++I) {
      Instruction *Old = cast<Instruction>(SV[I]);
      CV[I]->takeName(Old);
      Old->replaceAllUsesWith(CV[I]);
      Old->eraseFromParent();
    }
  }
  SV = CV;
  Gathered.push_back(GatherList::value_type(Op, &SV));
}

// Decompose two-operand instruction I, using Split(Builder, X, Y, Name)
// to create an instruction like I with operands X and Y and name Name.
template<typename Splitter>
bool DecomposeVectors::splitBinary(Instruction &I, const Splitter &Split) {
  VectorType *VT = dyn_cast<VectorType>(I.getType());
  if (!VT)
    return false;

  unsigned NumElems = VT->getNumElements();
  IRBuilder<> Builder(I.getParent(), &I);
  Scatterer Op0 = scatter(&I, I.getOperand(0));
  Scatterer Op1 = scatter(&I, I.getOperand(1));
  assert(Op0.size() == NumElems && "Mismatched binary operation");
  assert(Op1.size() == NumElems && "Mismatched binary operation");
  ValueVector Res;
  Res.resize(NumElems);
  for (unsigned Elem = 0; Elem < NumElems; ++Elem)
    Res[Elem] = Split(Builder, Op0[Elem], Op1[Elem],
                      I.getName() + ".i" + Twine(Elem));
  gather(&I, Res);
  return true;
}

bool DecomposeVectors::visitSelectInst(SelectInst &SI) {
  VectorType *VT = dyn_cast<VectorType>(SI.getType());
  if (!VT)
    return false;

  unsigned NumElems = VT->getNumElements();
  IRBuilder<> Builder(SI.getParent(), &SI);
  Scatterer Op1 = scatter(&SI, SI.getOperand(1));
  Scatterer Op2 = scatter(&SI, SI.getOperand(2));
  assert(Op1.size() == NumElems && "Mismatched select");
  assert(Op2.size() == NumElems && "Mismatched select");
  ValueVector Res;
  Res.resize(NumElems);

  if (SI.getOperand(0)->getType()->isVectorTy()) {
    Scatterer Op0 = scatter(&SI, SI.getOperand(0));
    assert(Op0.size() == NumElems && "Mismatched select");
    for (unsigned I = 0; I < NumElems; ++I)
      Res[I] = Builder.CreateSelect(Op0[I], Op1[I], Op2[I],
                                    SI.getName() + ".i" + Twine(I));
  } else {
    Value *Op0 = SI.getOperand(0);
    for (unsigned I = 0; I < NumElems; ++I)
      Res[I] = Builder.CreateSelect(Op0, Op1[I], Op2[I],
                                    SI.getName() + ".i" + Twine(I));
  }
  gather(&SI, Res);
  return true;
}

bool DecomposeVectors::visitICmpInst(ICmpInst &ICI) {
  return splitBinary(ICI, ICmpSplitter(ICI));
}

bool DecomposeVectors::visitFCmpInst(FCmpInst &FCI) {
  return splitBinary(FCI, FCmpSplitter(FCI));
}

bool DecomposeVectors::visitBinaryOperator(BinaryOperator &BO) {
  return splitBinary(BO, BinarySplitter(BO));
}

bool DecomposeVectors::visitCastInst(CastInst &CI) {
  VectorType *VT = dyn_cast<VectorType>(CI.getDestTy());
  if (!VT)
    return false;

  unsigned NumElems = VT->getNumElements();
  IRBuilder<> Builder(CI.getParent(), &CI);
  Scatterer Op0 = scatter(&CI, CI.getOperand(0));
  assert(Op0.size() == NumElems && "Mismatched cast");
  ValueVector Res;
  Res.resize(NumElems);
  for (unsigned I = 0; I < NumElems; ++I)
    Res[I] = Builder.CreateCast(CI.getOpcode(), Op0[I], VT->getElementType(),
                                CI.getName() + ".i" + Twine(I));
  gather(&CI, Res);
  return true;
}

bool DecomposeVectors::visitBitCastInst(BitCastInst &BCI) {
  VectorType *DstVT = dyn_cast<VectorType>(BCI.getDestTy());
  VectorType *SrcVT = dyn_cast<VectorType>(BCI.getSrcTy());
  if (!DstVT || !SrcVT)
    return false;

  unsigned DstNumElems = DstVT->getNumElements();
  unsigned SrcNumElems = SrcVT->getNumElements();
  IRBuilder<> Builder(BCI.getParent(), &BCI);
  Scatterer Op0 = scatter(&BCI, BCI.getOperand(0));
  ValueVector Res;
  Res.resize(DstNumElems);

  if (DstNumElems == SrcNumElems) {
    for (unsigned I = 0; I < DstNumElems; ++I)
      Res[I] = Builder.CreateBitCast(Op0[I], DstVT->getElementType(),
                                     BCI.getName() + ".i" + Twine(I));
  } else if (DstNumElems > SrcNumElems) {
    // <M x t1> -> <N*M x t2>.  Convert each t1 to <N x t2> and copy the
    // individual elements to the destination.
    unsigned FanOut = DstNumElems / SrcNumElems;
    Type *MidTy = VectorType::get(DstVT->getElementType(), FanOut);
    unsigned ResI = 0;
    for (unsigned Op0I = 0; Op0I < SrcNumElems; ++Op0I) {
      Value *V = Op0[Op0I];
      Instruction *VI;
      // Look through any existing bitcasts before converting to <N x t2>.
      // In the best case, the resulting conversion might be a no-op.
      while ((VI = dyn_cast<Instruction>(V)) &&
             VI->getOpcode() == Instruction::BitCast)
        V = VI->getOperand(0);
      V = Builder.CreateBitCast(V, MidTy, V->getName() + ".cast");
      Scatterer Mid = scatter(&BCI, V);
      for (unsigned MidI = 0; MidI < FanOut; ++MidI)
        Res[ResI++] = Mid[MidI];
    }
  } else {
    // <N*M x t1> -> <M x t2>.  Convert each group of <N x t1> into a t2.
    unsigned FanIn = SrcNumElems / DstNumElems;
    Type *MidTy = VectorType::get(SrcVT->getElementType(), FanIn);
    unsigned Op0I = 0;
    for (unsigned ResI = 0; ResI < DstNumElems; ++ResI) {
      Value *V = UndefValue::get(MidTy);
      for (unsigned MidI = 0; MidI < FanIn; ++MidI)
        V = Builder.CreateInsertElement(V, Op0[Op0I++], Builder.getInt32(MidI),
                                        BCI.getName() + ".i" + Twine(ResI)
                                        + ".upto" + Twine(MidI));
      Res[ResI] = Builder.CreateBitCast(V, DstVT->getElementType(),
                                        BCI.getName() + ".i" + Twine(ResI));
    }
  }
  gather(&BCI, Res);
  return true;
}

bool DecomposeVectors::visitShuffleVectorInst(ShuffleVectorInst &SVI) {
  VectorType *VT = dyn_cast<VectorType>(SVI.getType());
  if (!VT)
    return false;

  unsigned NumElems = VT->getNumElements();
  Scatterer Op0 = scatter(&SVI, SVI.getOperand(0));
  Scatterer Op1 = scatter(&SVI, SVI.getOperand(1));
  ValueVector Res;
  Res.resize(NumElems);

  for (unsigned I = 0; I < NumElems; ++I) {
    int Selector = SVI.getMaskValue(I);
    if (Selector < 0)
      Res[I] = UndefValue::get(VT->getElementType());
    else if (unsigned(Selector) < Op0.size())
      Res[I] = Op0[Selector];
    else
      Res[I] = Op1[Selector - Op0.size()];
  }
  gather(&SVI, Res);
  return true;
}

bool DecomposeVectors::visitPHINode(PHINode &PHI) {
  VectorType *VT = dyn_cast<VectorType>(PHI.getType());
  if (!VT)
    return false;

  unsigned NumElems = VT->getNumElements();
  IRBuilder<> Builder(PHI.getParent(), &PHI);
  ValueVector Res;
  Res.resize(NumElems);

  unsigned NumOps = PHI.getNumOperands();
  for (unsigned I = 0; I < NumElems; ++I)
    Res[I] = Builder.CreatePHI(VT->getElementType(), NumOps,
                               PHI.getName() + ".i" + Twine(I));

  for (unsigned I = 0; I < NumOps; ++I) {
    Scatterer Op = scatter(&PHI, PHI.getIncomingValue(I));
    BasicBlock *IncomingBlock = PHI.getIncomingBlock(I);
    for (unsigned J = 0; J < NumElems; ++J)
      cast<PHINode>(Res[J])->addIncoming(Op[J], IncomingBlock);
  }
  gather(&PHI, Res);
  return true;
}

bool DecomposeVectors::visitLoadInst(LoadInst &LI) {
  if (!DecomposeVectorLoadStore)
    return false;

  if (!LI.isSimple())
    return false;

  VectorType *VT = dyn_cast<VectorType>(LI.getType());
  if (!VT)
    return false;

  unsigned NumElems = VT->getNumElements();
  IRBuilder<> Builder(LI.getParent(), &LI);
  Scatterer Ptr = scatter(&LI, LI.getPointerOperand());
  ValueVector Res;
  Res.resize(NumElems);
  
  MDNode *Tag = LI.getMetadata(LLVMContext::MD_tbaa);
  for (unsigned I = 0; I < NumElems; ++I) {
    LoadInst *New = Builder.CreateLoad(Ptr[I], LI.getName() + ".i" + Twine(I));
    if (Tag)
      New->setMetadata(LLVMContext::MD_tbaa, Tag);
    Res[I] = New;    
  }
  gather(&LI, Res);
  return true;
}

bool DecomposeVectors::visitStoreInst(StoreInst &SI) {
  if (!DecomposeVectorLoadStore)
    return false;

  if (!SI.isSimple())
    return false;

  Value *FullValue = SI.getValueOperand();
  VectorType *VT = dyn_cast<VectorType>(FullValue->getType());
  if (!VT)
    return false;

  unsigned NumElems = VT->getNumElements();
  IRBuilder<> Builder(SI.getParent(), &SI);
  Scatterer Ptr = scatter(&SI, SI.getPointerOperand());
  Scatterer Val = scatter(&SI, FullValue);

  MDNode *Tag = SI.getMetadata(LLVMContext::MD_tbaa);
  for (unsigned I = 0; I < NumElems; ++I) {
    StoreInst *New = Builder.CreateStore(Val[I], Ptr[I]);
    if (Tag)
      New->setMetadata(LLVMContext::MD_tbaa, Tag);
  }
  return true;
}

// Delete the instructions that we decomposed.  If a full vector result
// is still needed, recreate it using InsertElements.
bool DecomposeVectors::finish() {
  if (Gathered.empty())
    return false;
  for (GatherList::iterator GMI = Gathered.begin(), GME = Gathered.end();
       GMI != GME; ++GMI) {
    Instruction *Op = GMI->first;
    ValueVector &CV = *GMI->second;
    if (!Op->use_empty()) {
      // The value is still needed, so recreate it using a series of
      // InsertElements.
      Type *Ty = Op->getType();
      Value *Res = UndefValue::get(Ty);
      unsigned Count = Ty->getVectorNumElements();
      IRBuilder<> Builder(Op->getParent(), Op);
      for (unsigned I = 0; I < Count; ++I)
        Res = Builder.CreateInsertElement(Res, CV[I], Builder.getInt32(I),
                                          Op->getName() + ".upto" + Twine(I));
      Res->takeName(Op);
      Op->replaceAllUsesWith(Res);
    }
    Op->eraseFromParent();
  }
  Gathered.clear();
  Scattered.clear();
  return true;
}

namespace llvm {
FunctionPass *createDecomposeVectorsPass() {
  return new DecomposeVectors();
}
}
