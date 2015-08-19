// TargetAddressSpaces.cc - map the fixed "logical" address-space ids to
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

#ifdef LLVM_3_2
# include <llvm/Instructions.h>
#else
# include <llvm/IR/Instructions.h>
# include <llvm/IR/Module.h>

#endif
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
ConvertedType(llvm::Type *type, std::map<unsigned, unsigned> &addrSpaceMap) {

  if (type->isPointerTy()) {
    unsigned AS = type->getPointerAddressSpace();
    unsigned newAS = addrSpaceMap[AS];
    return PointerType::get(ConvertedType(type->getPointerElementType(), addrSpaceMap), newAS);
  } else if (type->isArrayTy()) {
    return ArrayType::get
      (ConvertedType(type->getArrayElementType(), addrSpaceMap), type->getArrayNumElements());
  } else { /* TODO: pointers inside structs */
    return type;
  }
}

static bool
UpdateAddressSpace(llvm::Value& val, std::map<unsigned, unsigned> &addrSpaceMap) {
  Type *type = val.getType();
  if (!type->isPointerTy()) return false;

  Type *newType = ConvertedType(type, addrSpaceMap);
  if (newType == type) return false;

  val.mutateType(newType);
  return true;
}


bool
TargetAddressSpaces::runOnModule(llvm::Module &M) {

  llvm::StringRef arch(M.getTargetTriple());

  std::map<unsigned, unsigned> addrSpaceMap;

  if (arch.startswith("x86_64")) {
#ifndef LLVM_OLDER_THAN_3_7
    /* For x86_64 the default isel seems to work with the
       fake address spaces. Skip the processing as it causes
       an overhead and is not fully implemented.
    */
    return false;
#else
    /* At least LLVM 3.5 exposes an issue with pocl's printf or another LLVM pass:
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
    */
#endif
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
  } else if (arch.startswith("amdgcn")) {
    addrSpaceMap[POCL_ADDRESS_SPACE_GLOBAL] = 1;
    addrSpaceMap[POCL_ADDRESS_SPACE_LOCAL] = 3;
    addrSpaceMap[POCL_ADDRESS_SPACE_CONSTANT] = 2;
  } else {
    /* Assume the fake address space map works directly in case not
       overridden here.  */
    return false;
  }

  bool changed = false;
  /* Handle global variables. */
  llvm::Module::global_iterator globalI = M.global_begin();
  llvm::Module::global_iterator globalE = M.global_end();
  for (; globalI != globalE; ++globalI) {
    llvm::Value &global = *globalI;
    changed |= UpdateAddressSpace(global, addrSpaceMap);
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
      parameters.push_back(ConvertedType(i->getType(), addrSpaceMap));

    llvm::FunctionType *ft = FunctionType::get
      (ConvertedType(F.getReturnType(), addrSpaceMap),
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
      AddressSpaceReMapper(std::map<unsigned, unsigned> &addrSpaceMap) :
        addrSpaceMap_(addrSpaceMap) {}      
      Type* remapType(Type *type) {
        Type *newType = ConvertedType(type, addrSpaceMap_);
        if (newType == type) return type;
        return newType;
      }
    private:
      std::map<unsigned, unsigned>& addrSpaceMap_;
    } asvtm(addrSpaceMap);

    CloneFunctionInto(newFunc, &F, vv, true, ri, "", NULL, &asvtm);
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
