// LLVM function pass to convert the sampler initializer calls to a target
// desired format
//
// Copyright (c) 2014 Kalle Raiskila
//               2016-2017 Pekka Jääskeläinen / Tampere University of Technology
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

/*
   Lightweight bitcode linker to replace
   llvm::Linker. Unlike the LLVM default linker, this
   does not link in the entire given module, only
   the called functions are cloned from the input.
   This is to speed up the linking of the kernel lib
   which is so big, that it takes seconds to clone it,
   even on top-of-the line current processors
*/

#include <list>
#include <iostream>

#include "config.h"
#include "pocl.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "pocl_cl.h"

#include "linker.h"

#include "TargetAddressSpaces.h"

using namespace llvm;

//#include <cstdio>
//#define DB_PRINT(...) printf("linker:" __VA_ARGS__)
#define DB_PRINT(...)

extern cl_device_id currentPoclDevice;

/*
 * Find needle in haystack. O(n) implementation, but n should be
 * small in our usecases.
 * Return true if found
 */
static bool
find_from_list(llvm::StringRef             needle,
               std::list<llvm::StringRef> &haystack)
{
    std::list<llvm::StringRef>::iterator li,le;
    for (li=haystack.begin(), le=haystack.end();
         li != le;
         li++) {
        if (needle == *li) {
            return true;
        }
    }
    return false;
}

/* Fixes address space on opencl.imageX_t arguments to be global.
 * Note this does not change the types in Function->FunctionType
 * so it's only used inside CopyFunc on kernel library functions */
static void fixOpenCLimageArguments(llvm::Function *Func) {
    Function::arg_iterator b = Func->arg_begin();
    Function::arg_iterator e = Func->arg_end();
    for (; b != e; b++)  {
        Argument *j = &*b;
        Type *t = j->getType();
        if (t->isPointerTy() && t->getPointerElementType()->isStructTy()) {
            Type *pe_type = t->getPointerElementType();
            if (pe_type->getStructName().startswith("opencl.image"))  {
#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
              Type *new_t =
                PointerType::get(pe_type, POCL_FAKE_AS_GLOBAL);
#else
              Type *new_t =
                PointerType::get(pe_type, currentPoclDevice->global_as_id);
#endif
              j->mutateType(new_t);
            }
        }
    }
}

/* Fixes opencl.imageX_t type arguments which miss address space global
 * returns F if no changes are required, or a new cloned F if the arguments
 * require a fix. To be used on user's kernel code itself, not on kernel library.
 */
static llvm::Function *
CloneFuncFixOpenCLImageT(llvm::Module *Mod, llvm::Function *F)
{
    assert(F && "No function to copy");
    assert(!F->isDeclaration());

    int changed = 0;
    ValueToValueMapTy VVMap;
    SmallVector<Type *, 8> sv;
    for (Function::arg_iterator i = F->arg_begin(), e = F->arg_end();
          i != e; ++i) {
        Argument *j = &*i;
        Type *t = j->getType();
        Type *new_t = t;
        if (t->isPointerTy() && t->getPointerElementType()->isStructTy()) {
          Type *pe_type = t->getPointerElementType();
          if (pe_type->getStructName().startswith("opencl.image")) {

#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
            if (t->getPointerAddressSpace() != POCL_FAKE_AS_GLOBAL) {
              new_t = PointerType::get(pe_type, POCL_FAKE_AS_GLOBAL);
              changed = 1;
            }
#else
            if (t->getPointerAddressSpace() != currentPoclDevice->global_as_id) {
              new_t = PointerType::get(pe_type, currentPoclDevice->global_as_id);
              changed = 1;
            }
#endif
          }
        }
        sv.push_back(new_t);
    }

    if (!changed)
      return F;

    F->removeFromParent();

    FunctionType *NewFT = FunctionType::get(F->getReturnType(),
                                         ArrayRef<Type *> (sv),
                                         false);
    assert(NewFT);
    llvm::Function *DstFunc = nullptr;

    DstFunc = Function::Create(NewFT, F->getLinkage(), F->getName(), Mod);

    Function::arg_iterator j = DstFunc->arg_begin();
    for (Function::const_arg_iterator i = F->arg_begin(),
         e = F->arg_end();
         i != e; ++i) {
        j->setName(i->getName());
        VVMap[&*i] = &*j;
        ++j;
    }

    DstFunc->copyAttributesFrom(F);

    SmallVector<ReturnInst*, 8> RI;          // Ignore returns cloned.
    CloneFunctionInto(DstFunc, F, VVMap, true, RI);
    delete F;

    return DstFunc;
}

// Find all functions in the calltree of F, append their
// name to list.
static inline void
find_called_functions(llvm::Function *F,
                      std::list<llvm::StringRef> &list)
{
  if (F->isDeclaration()) {
    DB_PRINT("it's a declaration.\n");
    return;
  }
  llvm::Function::iterator fi,fe;
  for (fi = F->begin(), fe = F->end();
       fi != fe; fi++) {
    llvm::BasicBlock::iterator bi, be;
    for (bi = fi->begin(), be = fi->end();
         bi != be;
         bi++) {
      CallInst *CI = dyn_cast<CallInst>(bi);
      if (CI == NULL)
        continue;
      llvm::Function *callee = CI->getCalledFunction();
      // this happens with e.g. inline asm calls
      if (callee == NULL) {
        DB_PRINT("search: %s callee NULL\n", F->getName().str().c_str());
        continue;
      }
      DB_PRINT("search: %s calls %s\n",
               F->getName().data(), callee->getName().data());
      if (find_from_list(callee->getName(), list))
        continue;
      else {
        list.push_back(callee->getName());
        DB_PRINT("search: recursing into %s\n",
                 callee->getName().data());
        find_called_functions(callee, list);
      }
    }
  }
}

// Copies one function from one module to another
// does not inspect it for callgraphs
static void
CopyFunc(const llvm::StringRef Name,
         const llvm::Module *  From,
         llvm::Module *        To,
         ValueToValueMapTy &   VVMap) {

    llvm::Function *SrcFunc = From->getFunction(Name);
    // TODO: is this the linker error "not found", and not an assert?
    assert(SrcFunc && "Did not find function to copy in kernel library");
    llvm::Function *DstFunc = To->getFunction(Name);

    if (DstFunc == NULL) {
        DB_PRINT("   %s not found in destination module, creating\n",
                 Name.data());
        DstFunc =
          Function::Create(cast<FunctionType>(
                             SrcFunc->getType()->getElementType()),
                           SrcFunc->getLinkage(),
                           SrcFunc->getName(),
                           To);
        DstFunc->copyAttributesFrom(SrcFunc);
    } else if (DstFunc->size() > 0) {
      // We have already encountered and copied this function.
      return;
    }
    VVMap[SrcFunc] = DstFunc;

    Function::arg_iterator j = DstFunc->arg_begin();
    for (Function::const_arg_iterator i = SrcFunc->arg_begin(),
         e = SrcFunc->arg_end();
         i != e; ++i) {
        j->setName(i->getName());
        VVMap[&*i] = &*j;
        ++j;
    }
    if (!SrcFunc->isDeclaration()) {
        SmallVector<ReturnInst*, 8> RI;          // Ignore returns cloned.
        DB_PRINT("  cloning %s\n", Name.data());
        CloneFunctionInto(DstFunc, SrcFunc, VVMap, true, RI);
        fixOpenCLimageArguments(DstFunc);
    } else {
        DB_PRINT("  found %s, but its a declaration, do nothing\n",
                 Name.data());
    }
}

/* Copy function F and all the functions in its call graph
 * that are defined in 'from', into 'to', adding the mappings to
 * 'vvm'.
 */
static int
copy_func_callgraph(const llvm::StringRef func_name,
                    const llvm::Module *  from,
                    llvm::Module *        to,
                    ValueToValueMapTy &   vvm) {
    std::list<llvm::StringRef> callees;
    llvm::Function *RootFunc = from->getFunction(func_name);
    if (RootFunc == NULL)
      return -1;
    DB_PRINT("copying function %s with callgraph\n", RootFunc.data());

    find_called_functions(RootFunc, callees);

    // First copy the callees of func, then the function itself.
    // Recurse into callees to handle the case where kernel library
    // functions call other kernel library functions.
    std::list<llvm::StringRef>::iterator ci,ce;
    for (ci = callees.begin(), ce = callees.end(); ci != ce; ci++) {
      llvm::Function *SrcFunc = from->getFunction(*ci);
      if (!SrcFunc->isDeclaration()) {
        copy_func_callgraph(*ci, from, to, vvm);
      } else {
        DB_PRINT("%s is declaration, not recursing into it!\n",
		 SrcFunc->getName().str().c_str());
      }
      CopyFunc(*ci, from, to, vvm);
    }
    CopyFunc(func_name, from, to, vvm);
    return 0;
}

static inline bool
stringref_equal(llvm::StringRef a, llvm::StringRef b)
{
    return a.equals(b);
}
static inline bool
stringref_cmp(llvm::StringRef a, llvm::StringRef b)
{
    return a.str() < b.str();
}

static void shared_copy(llvm::Module *program, const llvm::Module *lib,
                        std::string &log, ValueToValueMapTy &vvm) {

  llvm::Module::const_global_iterator gi,ge;

  // copy any aliases to program
  DB_PRINT("cloning the aliases:\n");
  llvm::Module::const_alias_iterator ai, ae;
  for (ai = lib->alias_begin(), ae = lib->alias_end(); ai != ae; ai++) {
    DB_PRINT(" %s\n", ai->getName().data());
    GlobalAlias *GA =
#ifndef LLVM_3_7
      GlobalAlias::create(
        ai->getType(), ai->getType()->getAddressSpace(), ai->getLinkage(),
        ai->getName(), NULL, program);
#else
    GlobalAlias::create(
        ai->getType(), ai->getLinkage(), ai->getName(), NULL, program);
#endif

    GA->copyAttributesFrom(&*ai);
    vvm[&*ai]=GA;
  }

  // initialize the globals that were copied
  for (gi=lib->global_begin(), ge=lib->global_end();
       gi != ge;
       gi++) {
      GlobalVariable *GV=cast<GlobalVariable>(vvm[&*gi]);
      if (gi->hasInitializer())
        GV->setInitializer(MapValue(gi->getInitializer(), vvm));
  }

  // copy metadata
  DB_PRINT("cloning metadata:\n");
  llvm::Module::const_named_metadata_iterator mi,me;
  for (mi=lib->named_metadata_begin(), me=lib->named_metadata_end();
       mi != me; mi++) {
      const NamedMDNode &NMD=*mi;
      /* This causes problems, because multiple wchar_size */
      if (NMD.getName() == StringRef("llvm.module.flags"))
        continue;
      /* This causes problems with NVidia,
       * and is regenerated by pocl-ptx-gen anyway */
      if (NMD.getName() == StringRef("nvvm.annotations"))
        continue;
      DB_PRINT(" %s:\n", NMD.getName().data());
      NamedMDNode *NewNMD=program->getOrInsertNamedMetadata(NMD.getName());
      for (unsigned i=0, e=NMD.getNumOperands(); i != e; ++i)
        NewNMD->addOperand(MapMetadata(NMD.getOperand(i), vvm));
  }

}

int link(llvm::Module *program, const llvm::Module *lib, std::string &log) {
  assert(program);
  assert(lib);
  ValueToValueMapTy vvm;
  std::list<llvm::StringRef> declared;

  llvm::Module::iterator fi, fe;

  // Find and fix opencl.imageX_t arguments
  for (fi = program->begin(), fe = program->end(); fi != fe; fi++) {
    llvm::Function *f = &*fi;
    if (f->isDeclaration())
      continue;
    // need to restart iteration if we replace a function
    if (CloneFuncFixOpenCLImageT(program, f) != f) {
      fi = program->begin();
    }
  }

  // Inspect the program, find undefined functions
  for (fi = program->begin(), fe = program->end(); fi != fe; fi++) {
    if ((*fi).isDeclaration()) {
      DB_PRINT("%s is not defined\n", fi->getName().data());
      declared.push_back(fi->getName());
      continue;
    }

    // Find all functions the program source calls
    // TODO: is there no direct way?
    find_called_functions(&*fi, declared);
  }
  declared.sort(stringref_cmp);
  declared.unique(stringref_equal);

  // Copy all the globals from lib to program.
  // It probably is faster to just copy them all, than to inspect
  // both program and lib to find which actually are used.
  DB_PRINT("cloning the global variables:\n");
  llvm::Module::const_global_iterator gi,ge;
  for (gi=lib->global_begin(), ge=lib->global_end(); gi != ge; gi++) {
    DB_PRINT(" %s\n", gi->getName().data());
    GlobalVariable *GV = new GlobalVariable(
      *program, gi->getType()->getElementType(), gi->isConstant(),
      gi->getLinkage(), (Constant*)0, gi->getName(), (GlobalVariable*)0,
      gi->getThreadLocalMode(), gi->getType()->getAddressSpace());
    GV->copyAttributesFrom(&*gi);
    vvm[&*gi]=GV;
  }

  // For each undefined function in program,
  // clone it from the lib to the program module,
  // if found in lib
  bool found_all_undefined = true;

  // this one is a handled with a special pocl LLVM pass
  StringRef pocl_sampler_handler("__translate_sampler_initializer");
  // ignore undefined llvm intrinsics
  StringRef llvm_intrins("llvm.");
  std::list<llvm::StringRef>::iterator di,de;
  for (di = declared.begin(), de = declared.end();
       di != de; di++) {
      llvm::StringRef r = *di;
      if (copy_func_callgraph(r, lib, program, vvm)) {
        Function *f = program->getFunction(r);
        if ((f == NULL) ||
            (f->isDeclaration() &&
             !f->getName().equals(pocl_sampler_handler) &&
             !f->getName().startswith(llvm_intrins))
           ) {
          log.append("Cannot find symbol ");
          log.append(r.str());
          log.append(" in kernel library\n");
          found_all_undefined = false;
        }
      }
  }
  if (!found_all_undefined)
    return 1;

  shared_copy(program, lib, log, vvm);

  return 0;
}

int copyKernelFromBitcode(const char* name, llvm::Module *parallel_bc,
                          const llvm::Module *program) {
  ValueToValueMapTy vvm;

  // Copy all the globals from lib to program.
  // It probably is faster to just copy them all, than to inspect
  // both program and lib to find which actually are used.
  DB_PRINT("cloning the global variables:\n");
  llvm::Module::const_global_iterator gi,ge;
  for (gi=program->global_begin(), ge=program->global_end(); gi != ge; gi++) {
    DB_PRINT(" %s\n", gi->getName().data());
    GlobalVariable *GV = new GlobalVariable(
      *parallel_bc, gi->getType()->getElementType(), gi->isConstant(),
      gi->getLinkage(), (Constant*)0, gi->getName(), (GlobalVariable*)0,
      gi->getThreadLocalMode(), gi->getType()->getAddressSpace());
    GV->copyAttributesFrom(&*gi);
    vvm[&*gi]=GV;
  }

  const StringRef kernel_name(name);
  copy_func_callgraph(kernel_name, program, parallel_bc, vvm);

  std::string log;
  shared_copy(parallel_bc, program, log, vvm);

  return 0;
}

/* vim: set expandtab ts=4 : */

