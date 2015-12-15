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

#include "config.h"
#include <iostream>
#include <string>
#include <set>

#ifdef LLVM_3_2
# include <llvm/Instructions.h>
# include <llvm/IntrinsicInst.h>
#else
# include <llvm/IR/Instructions.h>
# include <llvm/IR/Module.h>
# include <llvm/IR/IntrinsicInst.h>
#endif

#include <llvm/IR/IRBuilder.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "TargetAddressSpaces.h"
#include "Workgroup.h"
#include "LLVMUtils.h"
#include "pocl.h"

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
              std::map<llvm::Type*, llvm::StructType*> &converted_struct_cache) {

  if (type->isPointerTy()) {
    unsigned AS = type->getPointerAddressSpace();
    unsigned newAS = addrSpaceMap[AS];
    return PointerType::get(ConvertedType(type->getPointerElementType(), addrSpaceMap, converted_struct_cache), newAS);
  } else if (type->isArrayTy()) {
    return ArrayType::get
      (ConvertedType(type->getArrayElementType(), addrSpaceMap, converted_struct_cache), type->getArrayNumElements());
#ifndef TCE_AVAILABLE
  } else if (type->isStructTy()) {
    if (converted_struct_cache[type])
      return converted_struct_cache[type];

    llvm::StructType* t = dyn_cast<llvm::StructType>(type);
    llvm::StructType* tn;
    if (!t->isLiteral())
      {
        std::string s = t->getName().str();
        s += "_tas_struct";
        tn = StructType::create(t->getContext(), s);
        converted_struct_cache[type] = tn;
      }
    std::vector<llvm::Type*> newtypes;
    for (llvm::StructType::element_iterator i = t->element_begin(),
         e = t->element_end(); i < e; ++i)
      {
        newtypes.push_back(ConvertedType(*i, addrSpaceMap, converted_struct_cache));
      }
    ArrayRef<Type*> a(newtypes);
    if (t->isLiteral())
      {
        tn = StructType::get(t->getContext(), a, t->isPacked());
        converted_struct_cache[type] = tn;
      }
    else
      tn->setBody(a, t->isPacked());

    return tn;
#endif
  } else {
      return type;
  }
}

static bool
UpdateAddressSpace(llvm::Value& val, std::map<unsigned, unsigned> &addrSpaceMap,
                   std::map<llvm::Type*, llvm::StructType*> &converted_struct_cache) {
  Type *type = val.getType();
  if (!type->isPointerTy()) return false;

  Type *newType = ConvertedType(type, addrSpaceMap, converted_struct_cache);
  if (newType == type) return false;

  val.mutateType(newType);
  return true;
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
    llvm::BasicBlock* bb = bbi;
    for (llvm::BasicBlock::iterator ii = bb->begin(), ie = bb->end();
         ii != ie; ++ii) {
      llvm::Instruction *instr = ii;
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

  std::map<llvm::Type*, llvm::StructType*> converted_struct_cache;

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
        addrSpaceMap[POCL_ADDRESS_SPACE_CONSTANT] = 0;

  } else if (arch.startswith("arm")) {
    /* Same thing happens here as with x86_64 above.
     * NB: LLVM 3.5 on ARM did not need this yet, for some reason
     */
#if defined LLVM_3_2 || defined LLVM_3_3 || defined LLVM_3_4 || defined_LLVM_3_5
    return false;
#else
    addrSpaceMap[POCL_ADDRESS_SPACE_GLOBAL] =
        addrSpaceMap[POCL_ADDRESS_SPACE_LOCAL] =
        addrSpaceMap[POCL_ADDRESS_SPACE_CONSTANT] = 0;
#endif
  } else if (arch.startswith("tce")) {
    /* TCE requires the remapping. */
    addrSpaceMap[POCL_ADDRESS_SPACE_GLOBAL] = 3;
    addrSpaceMap[POCL_ADDRESS_SPACE_LOCAL] = 4;
    /* LLVM 3.2 detects 'constant' as cuda_constant (5) in the fake
       address space map. Add it for compatibility. */
    addrSpaceMap[5] = addrSpaceMap[POCL_ADDRESS_SPACE_CONSTANT] = 5;     
  } else if (arch.startswith("mips")) {
    addrSpaceMap[POCL_ADDRESS_SPACE_GLOBAL] =
        addrSpaceMap[POCL_ADDRESS_SPACE_LOCAL] =
        addrSpaceMap[POCL_ADDRESS_SPACE_CONSTANT] = 0;
  } else if (arch.startswith("amdgcn") || arch.startswith("hsail")) {
    addrSpaceMap[POCL_ADDRESS_SPACE_GLOBAL] = 1;
    addrSpaceMap[POCL_ADDRESS_SPACE_LOCAL] = 3;
    addrSpaceMap[POCL_ADDRESS_SPACE_CONSTANT] = 2;
  } else {
    /* Assume the fake address space map works directly in case not
       overridden here.  */
    return false;
  }

  bool changed = false;

  FunctionMapping funcReplacements;
  std::vector<llvm::Function*> unhandledFuncs;

  /* Collect the functions to process first because we add
     a new function per modified function which invalidates
     the Module's function iterator. */
  for (llvm::Module::iterator functionI = M.begin(), functionE = M.end(); 
       functionI != functionE; ++functionI) {
    if (functionI->empty() || functionI->getName().startswith("_GLOBAL")) 
      continue;
    unhandledFuncs.push_back(functionI);
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
      parameters.push_back(ConvertedType(i->getType(), addrSpaceMap, converted_struct_cache));

    llvm::FunctionType *ft = FunctionType::get
      (ConvertedType(F.getReturnType(), addrSpaceMap, converted_struct_cache),
       parameters, F.isVarArg());

    llvm::Function *newFunc = Function::Create(ft, F.getLinkage(), "", &M);
    newFunc->takeName(&F);

    ValueToValueMapTy vv;
    Function::arg_iterator j = newFunc->arg_begin();
    for (Function::const_arg_iterator i = F.arg_begin(),
           e = F.arg_end();
         i != e; ++i) {
      j->setName(i->getName());
      vv[i] = j;
      ++j;
    }

    SmallVector<ReturnInst *, 1> ri;

    class AddressSpaceReMapper : public ValueMapTypeRemapper {
    public:
      AddressSpaceReMapper(std::map<unsigned, unsigned> &addrSpaceMap,
                           std::map<llvm::Type*, llvm::StructType*> *c) :
        converted_struct_cache_(c), addrSpaceMap_(addrSpaceMap) {}

      Type* remapType(Type *type) {
        Type *newType = ConvertedType(type, addrSpaceMap_, *converted_struct_cache_);
        if (newType == type) return type;
        return newType;
      }
    private:
      std::map<llvm::Type*, llvm::StructType*> *converted_struct_cache_;
      std::map<unsigned, unsigned>& addrSpaceMap_;
    } asvtm(addrSpaceMap, &converted_struct_cache);

    CloneFunctionInto(newFunc, &F, vv, true, ri, "", NULL, &asvtm);
    FixMemIntrinsics(*newFunc);
    funcReplacements[&F] = newFunc;
  }

  /* Handle global variables. These should be fixed *after*
     fixing the instruction referring to them.  If we fix
     the address spaces before, there might be possible
     illegal bitcasts casting the LLVM's global pointer to
     another one, causing the CloneFunctionInto to crash when
     it encounters such.
   */
  llvm::Module::global_iterator globalI = M.global_begin();
  llvm::Module::global_iterator globalE = M.global_end();
  for (; globalI != globalE; ++globalI) {
    llvm::Value &global = *globalI;
    changed |= UpdateAddressSpace(global, addrSpaceMap, converted_struct_cache);
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
        llvm::Instruction *instr = ii;

#if !(defined(LLVM_3_2) || defined(LLVM_3_3))
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
#endif
        
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
#if (defined LLVM_3_2 || defined LLVM_3_3 || defined LLVM_3_4)
        User* user = *ui;
#else
        User* user = (*ui).getUser();
#endif
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
