// TargetAddressSpaces.cc - map the fixed "logical" address-space ids to,
//                          the target-specific ones, if needed
// 
// Copyright (c) 2013-2015 Pekka Jääskeläinen / TUT
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
#include <string>
#include <set>

#include "pocl.h"

# include <llvm/IR/Instructions.h>
# include <llvm/IR/Module.h>
# include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "TargetAddressSpaces.h"
#include "Workgroup.h"
#include "LLVMUtils.h"

#define DEBUG_TARGET_ADDRESS_SPACES

namespace pocl {

using namespace llvm;

namespace {
  static
  RegisterPass<pocl::TargetAddressSpaces> X
  ("target-address-spaces", 
   "Convert the logical address space ids to the target specific ones.");
}

char TargetAddressSpaces::ID = 0;

TargetAddressSpaces::TargetAddressSpaces() : ModulePass(ID) {
}

static Type *
ConvertedType(llvm::Type *type, std::map<unsigned, unsigned> &addrSpaceMap,
              std::map<llvm::Type*, llvm::StructType*> &convertedStructsCache) {

  if (type->isPointerTy()) {
    unsigned AS = type->getPointerAddressSpace();
    unsigned newAS = addrSpaceMap[AS];
    return PointerType::get(
          ConvertedType(type->getPointerElementType(),
                        addrSpaceMap, convertedStructsCache),
          newAS);
  } else if (type->isArrayTy()) {
    return ArrayType::get(
          ConvertedType(type->getArrayElementType(),
                        addrSpaceMap, convertedStructsCache),
          type->getArrayNumElements());
#ifndef TCE_AVAILABLE
  } else if (type->isStructTy()) {
    if (convertedStructsCache[type])
      return convertedStructsCache[type];

    llvm::StructType* t = dyn_cast<llvm::StructType>(type);
    llvm::StructType* tn;
    if (!t->isLiteral()) {
      std::string s = t->getName().str();
      s += "_tas_struct";
      tn = StructType::create(t->getContext(), s);
      convertedStructsCache[type] = tn;
    }
    std::vector<llvm::Type*> newtypes;
    for (llvm::StructType::element_iterator i = t->element_begin(),
         e = t->element_end(); i < e; ++i) {
      newtypes.push_back(ConvertedType(*i, addrSpaceMap, convertedStructsCache));
    }
    ArrayRef<Type*> a(newtypes);
    if (t->isLiteral()) {
      tn = StructType::get(t->getContext(), a, t->isPacked());
      convertedStructsCache[type] = tn;
    } else {
      tn->setBody(a, t->isPacked());
    }
    return tn;
#endif
  } else {
    return type;
  }
}

static bool
UpdateAddressSpace(llvm::Value& val, std::map<unsigned, unsigned> &addrSpaceMap,
                   std::map<llvm::Type*, llvm::StructType*> &convertedStructsCache) {
  Type *type = val.getType();
  if (!type->isPointerTy()) return false;

  Type *newType = ConvertedType(type, addrSpaceMap, convertedStructsCache);
  if (newType == type) return false;

  val.mutateType(newType);
  return true;
}

/* Removes AddrSpaceCastInst either as Inst or ConstantExpr, if they cast
   to generic addrspace, or if they point to the same AS
   ConstExpr removing is 2 step: CE -> convert to ASCI -> remove ASCI.

   \param [in] v the ASCI to remove
   \param [in] beforeinst in case of a ConstantExpr, after converting it to Instr
               we need to insert it into BB; this is an Instr before
               which we insert it (it's the CE itself)
   \returns true if replacement took place (-> BB iterator needs to restart)
*/
static bool removeASCI(llvm::Value *v, llvm::Instruction *beforeinst,
                     std::map<unsigned, unsigned> &addrSpaceMap,
                     std::map<llvm::Type*, llvm::StructType*> &convertedStructsCache) {
  if (isa<ConstantExpr>(v)) {
      ConstantExpr *ce = dyn_cast<ConstantExpr>(v);
      Value *in = ce->getAsInstruction();
      AddrSpaceCastInst *asci = dyn_cast<AddrSpaceCastInst>(in);
      assert(asci);
      if (asci->getDestTy()->getPointerAddressSpace() == POCL_ADDRESS_SPACE_GENERIC) {
        asci->insertBefore(beforeinst);
        v->replaceAllUsesWith(in);
        in->takeName(v);
        return true;
      } else
        return false;
  }
  if (isa<AddrSpaceCastInst>(v)) {
      AddrSpaceCastInst *as = dyn_cast<AddrSpaceCastInst>(v);
      Type* SrcTy = as->getSrcTy();
      Type* DstTy = as->getDestTy();
      if (isa<PointerType>(SrcTy) && isa<PointerType>(DstTy)) {
        if ((DstTy->getPointerAddressSpace() == SrcTy->getPointerAddressSpace())
            || (DstTy->getPointerAddressSpace() == POCL_ADDRESS_SPACE_GENERIC))
          {
            if (DstTy->getPointerAddressSpace() == POCL_ADDRESS_SPACE_GENERIC)
              UpdateAddressSpace(*as, addrSpaceMap, convertedStructsCache);
            Value* srcVal = as->getOperand(0);
            as->replaceAllUsesWith(srcVal);
            as->eraseFromParent();
            return true;
          }
      }
  }

  return false;

}




/**
 * After converting the pointer address spaces, there
 * might be llvm.memcpy.* or llvm.memset.* calls to wrong
 * intrinsics with wrong address spaces which this function
 * fixes.
 */
static void
FixMemIntrinsics(llvm::Function& F) {

  // Collect the intrinsics first to avoid breaking the
  // iterators.
  std::vector<llvm::MemIntrinsic*> intrinsics;
  for (llvm::Function::iterator bbi = F.begin(), bbe = F.end(); bbi != bbe;
       ++bbi) {
    llvm::BasicBlock* bb = &*bbi;
    for (llvm::BasicBlock::iterator ii = bb->begin(), ie = bb->end();
         ii != ie; ++ii) {
      llvm::Instruction *instr = &*ii;
      if (!isa<llvm::MemIntrinsic>(instr)) continue;
      intrinsics.push_back(dyn_cast<llvm::MemIntrinsic>(instr));
    }
  }

  for (llvm::MemIntrinsic* i : intrinsics) {

    llvm::IRBuilder<> Builder(i);
    // The pointer arguments to the intrinsics are already converted
    // to the correct address spaces. All we need to do here is to
    // "refresh" the intrinsics so it gets correct address spaces
    // in the name (e.g. llvm.memcpy.p0i8.p0i8).
    if (llvm::MemCpyInst* old = dyn_cast<llvm::MemCpyInst>(i)) {
      Builder.CreateMemCpy(
        old->getRawDest(), old->getRawSource(), old->getLength(),
        old->getAlignment(), old->isVolatile());
      old->eraseFromParent();
    } else if (llvm::MemSetInst* old = dyn_cast<llvm::MemSetInst>(i)) {
      Builder.CreateMemSet(
        old->getRawDest(), old->getValue(), old->getLength(),
        old->getAlignment(), old->isVolatile());
      old->eraseFromParent();
    } else if (llvm::MemMoveInst* old = dyn_cast<llvm::MemMoveInst>(i)) {
      Builder.CreateMemMove(
        old->getRawDest(), old->getRawSource(), old->getLength(),
        old->getAlignment(), old->isVolatile());
      old->eraseFromParent();
    } else {
        assert (false && "Unknown MemIntrinsic.");
    }
  }
}



bool
TargetAddressSpaces::runOnModule(llvm::Module &M) {

  llvm::StringRef arch(M.getTargetTriple());

  std::map<llvm::Type*, llvm::StructType*> convertedStructsCache;

  std::map<unsigned, unsigned> addrSpaceMap;

  if (arch.startswith("x86_64")) {
    /* x86_64 supports flattening the address spaces at the backend, but
       we still flatten them in pocl due to a couple of reasons.

       At least LLVM 3.5 exposes an issue with pocl's printf or another LLVM pass:
       After the code emission optimizations there appears a
       PHI node where the two alternative pointer assignments have different
       address spaces:
       %format.addr.2347 =
          phi i8 addrspace(3)* [ %incdec.ptr58, %if.end56 ],
                               [ %format.addr.1, %while.body45.preheader ]

       This leads to an LLVM crash when it tries to generate a no-op bitcast
       while it won't be such due to the address space difference (I assume).
       Workaround this by flattening the address spaces to 0 here also for
       x86_64 until the real culprit is found.

       Another reason is that LoopVectorizer of LLVM 3.7 crashes when it
       tries to create a masked store intrinsics with the fake address space
       ids, so we need to flatten them out before vectorizing.
    */
    addrSpaceMap[POCL_ADDRESS_SPACE_GLOBAL] =
        addrSpaceMap[POCL_ADDRESS_SPACE_LOCAL] =
        addrSpaceMap[POCL_ADDRESS_SPACE_GENERIC] =
        addrSpaceMap[POCL_ADDRESS_SPACE_CONSTANT] = 0;

  } else if (arch.startswith("arm")) {
    /* Same thing happens here as with x86_64 above.
     */
    addrSpaceMap[POCL_ADDRESS_SPACE_GLOBAL] =
        addrSpaceMap[POCL_ADDRESS_SPACE_LOCAL] =
        addrSpaceMap[POCL_ADDRESS_SPACE_GENERIC] =
        addrSpaceMap[POCL_ADDRESS_SPACE_CONSTANT] = 0;
  } else if (arch.startswith("tce")) {
    /* TCE requires the remapping. */
    addrSpaceMap[POCL_ADDRESS_SPACE_GENERIC] = 0;
    addrSpaceMap[POCL_ADDRESS_SPACE_GLOBAL] = 3;
    addrSpaceMap[POCL_ADDRESS_SPACE_LOCAL] = 4;
    addrSpaceMap[POCL_ADDRESS_SPACE_CONSTANT] = 5;
  } else if (arch.startswith("mips")) {
    addrSpaceMap[POCL_ADDRESS_SPACE_GLOBAL] =
    addrSpaceMap[POCL_ADDRESS_SPACE_LOCAL] =
    addrSpaceMap[POCL_ADDRESS_SPACE_GENERIC] =
    addrSpaceMap[POCL_ADDRESS_SPACE_CONSTANT] = 0;
  } else if (arch.startswith("amdgcn") || arch.startswith("hsail")) {
    addrSpaceMap[POCL_ADDRESS_SPACE_GENERIC] = 0;
    addrSpaceMap[POCL_ADDRESS_SPACE_GLOBAL] = 1;
    addrSpaceMap[POCL_ADDRESS_SPACE_LOCAL] = 3;
    addrSpaceMap[POCL_ADDRESS_SPACE_CONSTANT] = 2;
  } else {
    /* Assume the fake address space map works directly in case not
       overridden here.  */
    return false;
  }

  bool changed = false;

  /* Handle global variables. These should be fixed *after*
     fixing the instruction referring to them.  If we fix
     the address spaces before, there might be possible
     illegal bitcasts casting the LLVM's global pointer to
     another one, causing the CloneFunctionInto to crash when
     it encounters such.

     Update: ^this seems not to be an issue anymore and this commit
     seems to cause the problems it is trying to fix on hsa and
     amd scanlargearrays....  Original commit:
     dcbcd39811638bcb953afbbfdd2620fb8ab45af4
  */
  llvm::Module::global_iterator globalI = M.global_begin();
  llvm::Module::global_iterator globalE = M.global_end();
  for (; globalI != globalE; ++globalI) {
    llvm::Value &global = *globalI;
    changed |= UpdateAddressSpace(global, addrSpaceMap, convertedStructsCache);
  }

  FunctionMapping funcReplacements;
  std::vector<llvm::Function*> unhandledFuncs;

  /* Collect the functions to process first because we add
     a new function per modified function which invalidates
     the Module's function iterator. */
  for (llvm::Module::iterator functionI = M.begin(), functionE = M.end(); 
       functionI != functionE; ++functionI) {
    if (functionI->empty() || functionI->getName().startswith("_GLOBAL")) 
      continue;
    unhandledFuncs.push_back(&*functionI);
  }

  for (std::vector<llvm::Function*>::iterator i = unhandledFuncs.begin(), 
         e = unhandledFuncs.end(); i != e; ++i) {
    llvm::Function &F = **i;
   
    /* Convert the FunctionType. Because there is no mutator API in
       LLVM for this, we need to recreate the whole darn function :( */
    SmallVector<Type *, 8> parameters;
    for (Function::const_arg_iterator i = F.arg_begin(),
           e = F.arg_end();
         i != e; ++i)
      parameters.push_back(
            ConvertedType(i->getType(), addrSpaceMap, convertedStructsCache));

    llvm::FunctionType *ft = FunctionType::get
      (ConvertedType(F.getReturnType(), addrSpaceMap, convertedStructsCache),
       parameters, F.isVarArg());

    llvm::Function *newFunc = Function::Create(ft, F.getLinkage(), "", &M);
    newFunc->takeName(&F);

    ValueToValueMapTy vv;
    Function::arg_iterator j = newFunc->arg_begin();
    for (Function::const_arg_iterator i = F.arg_begin(),
           e = F.arg_end();
         i != e; ++i) {
      j->setName(i->getName());
      vv[&*i] = &*j;
      ++j;
    }

    SmallVector<ReturnInst *, 1> ri;

    /* Remove generic address space casts. Converts types with generic AS to
     * private AS and then removes redundant AS casting instructions */
    for (llvm::Function::iterator bbi = F.begin(), bbe = F.end(); bbi != bbe;
         ++bbi)
      for (llvm::BasicBlock::iterator ii = bbi->begin(), ie = bbi->end(); ii != ie;
           ++ii) {

        llvm::Instruction *instr = &*ii;

        if (isa<AddrSpaceCastInst>(instr)) {
            if (removeASCI(instr, nullptr, addrSpaceMap, convertedStructsCache))
              { ii = bbi->begin(); continue; }
          }
        if (isa<StoreInst>(instr)) {
            StoreInst *st = dyn_cast<StoreInst>(instr);
            Value *pt = st->getPointerOperand();
            if (Operator::getOpcode(pt) == Instruction::AddrSpaceCast) {
              if (removeASCI(pt, instr, addrSpaceMap, convertedStructsCache))
                { ii = bbi->begin(); continue; }
            } else
              if (st->getPointerAddressSpace() == POCL_ADDRESS_SPACE_GENERIC)
                UpdateAddressSpace(*pt, addrSpaceMap, convertedStructsCache);
        }
        if (isa<LoadInst>(instr)) {
            LoadInst *ld = dyn_cast<LoadInst>(instr);
            Value *pt = ld->getPointerOperand();
            if (Operator::getOpcode(pt) == Instruction::AddrSpaceCast) {
              if (removeASCI(pt, instr, addrSpaceMap, convertedStructsCache))
                { ii = bbi->begin(); continue; }
            } else
              if (ld->getPointerAddressSpace() == POCL_ADDRESS_SPACE_GENERIC)
                UpdateAddressSpace(*pt, addrSpaceMap, convertedStructsCache);
        }
        if (isa<GetElementPtrInst>(instr)) {
            GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(instr);
            Value *pt = gep->getPointerOperand();
            if (Operator::getOpcode(pt) == Instruction::AddrSpaceCast) {
                if (removeASCI(pt, instr, addrSpaceMap, convertedStructsCache))
                  { ii = bbi->begin(); continue; }
              } else {
                if (gep->getPointerAddressSpace() == POCL_ADDRESS_SPACE_GENERIC)
                  UpdateAddressSpace(*pt, addrSpaceMap, convertedStructsCache);
              }
        }

      }

    class AddressSpaceReMapper : public ValueMapTypeRemapper {
    public:
      AddressSpaceReMapper(std::map<unsigned, unsigned> &addrSpaceMap,
                           std::map<llvm::Type*, llvm::StructType*> *c) :
        cStructCache_(c), addrSpaceMap_(addrSpaceMap) {}

      Type* remapType(Type *type) {
        Type *newType = ConvertedType(type, addrSpaceMap_, *cStructCache_);
        if (newType == type) return type;
        return newType;
      }
    private:
      std::map<llvm::Type*, llvm::StructType*> *cStructCache_;
      std::map<unsigned, unsigned>& addrSpaceMap_;
    } asvtm(addrSpaceMap, &convertedStructsCache);

    CloneFunctionInto(newFunc, &F, vv, true, ri, "", NULL, &asvtm);
    FixMemIntrinsics(*newFunc);
    funcReplacements[&F] = newFunc;
  }

  /* Replace all references to the old function to the new one.
     Also, for LLVM 3.4, replace the pointercasts to bitcasts in
     case the new address spaces are the same in both sides. */
  llvm::Module::iterator fI = M.begin();
  llvm::Module::iterator fE = M.end();
  for (; fI != fE; ++fI) {
    llvm::Function &F = *fI;
    for (llvm::Function::iterator bbi = F.begin(), bbe = F.end(); bbi != bbe;
         ++bbi) 
      for (llvm::BasicBlock::iterator ii = bbi->begin(), ie = bbi->end(); ii != ie;
           ++ii) {
        llvm::Instruction *instr = &*ii;

        if (isa<AddrSpaceCastInst>(instr)) {
          // Convert (now illegal) addresspacecasts to bitcasts.

          // The old unconverted functions are still there, skip them.
          if (instr->getOperand(0)->getType()->getPointerAddressSpace() !=
              dyn_cast<CastInst>(instr)->getDestTy()->getPointerAddressSpace())
            continue;

          llvm::ReplaceInstWithInst
          (instr, 
           CastInst::CreatePointerCast
           (instr->getOperand(0), dyn_cast<CastInst>(instr)->getDestTy()));

          // Start from the beginning just in case the iterators have
          // been invalidated.
          ii = bbi->begin();
          continue;
        }
        
        if (!isa<CallInst>(instr)) continue;

        llvm::CallInst *call = dyn_cast<CallInst>(instr);
        llvm::Function *calledF = call->getCalledFunction();
        if (funcReplacements.find(calledF) == funcReplacements.end()) continue;
         
        call->setCalledFunction(funcReplacements[calledF]);
      }
  }

  FunctionMapping::iterator i = funcReplacements.begin();
  /* Delete the old functions. */
  while (funcReplacements.size() > 0) {

    if (Workgroup::isKernelToProcess(*i->first)) {
      FunctionMapping repl;
      repl[i->first] = i->second;
      regenerate_kernel_metadata(M, repl);
    }

    if (i->first->getNumUses() > 0) {
      for (Value::use_iterator ui = i->first->use_begin(), 
             ue = i->first->use_end(); ui != ue; ++ui) {
        User* user = (*ui).getUser();
        user->dump();
      }
      assert ("All users of the function were not fixed?" &&
              i->first->getNumUses() == 0);
      break;
    }
    i->first->eraseFromParent();
    funcReplacements.erase(i);
    i = funcReplacements.begin();
  }

  return true;
}

}
