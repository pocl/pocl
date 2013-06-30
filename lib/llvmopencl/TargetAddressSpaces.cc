// Header for TargetAddressSpaces
// 
// Copyright (c) 2013 Pekka Jääskeläinen / TUT
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
#include "llvm/Transforms/Utils/Cloning.h"

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
   "Convert the 'fake' address space ids to the target specific ones.");
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

  std::string triple = M.getTargetTriple();
  std::string arch = triple;
  size_t dash = triple.find("-");
  if (dash != std::string::npos) {
    arch = triple.substr(0, dash);
  }

  std::map<unsigned, unsigned> addrSpaceMap;

  if (arch == "x86_64") {
    /* For x86_64 the default isel seems to work with the
       fake address spaces. Skip the processing as it causes 
       an overhead and is not fully implemented.
    */
    return false; 
  } else if (arch == "tce") {
    /* TCE requires the remapping. */
    addrSpaceMap[POCL_ADDRESS_SPACE_GLOBAL] = 3;
    addrSpaceMap[POCL_ADDRESS_SPACE_LOCAL] = 4;
    /* LLVM 3.2 detects 'constant' as cuda_constant (5) in the fake
       address space map. Add it for compatibility. */
    addrSpaceMap[5] = addrSpaceMap[POCL_ADDRESS_SPACE_CONSTANT] = 5;     

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
  
  /* Replace all references to the old function to the new one. */
  llvm::Module::iterator fI = M.begin();
  llvm::Module::iterator fE = M.end();
  for (; fI != fE; ++fI) {
    llvm::Function &F = *fI;
    for (llvm::Function::iterator bbi = F.begin(), bbe = F.end(); bbi != bbe;
         ++bbi) 
      for (llvm::BasicBlock::iterator ii = bbi->begin(), ie = bbi->end(); ii != ie;
           ++ii) {
        llvm::Instruction *instr = ii;
        if (!isa<CallInst>(instr)) continue;
        llvm::CallInst *call = dyn_cast<CallInst>(instr);
        llvm::Function *calledF = call->getCalledFunction();
        if (funcReplacements.find(calledF) == funcReplacements.end()) continue;
        
        call->setCalledFunction(funcReplacements[calledF]);
      }
  }

  regenerate_kernel_metadata(M, funcReplacements);

  /* Delete the old functions. */
  for (FunctionMapping::iterator i = funcReplacements.begin(), 
         e = funcReplacements.end(); i != e; ++i) {
    i->first->eraseFromParent();
  }

  return true;
}

}
