/*
   Lightweight linker to replace
   llvm::Linker. Unlike the LLVM default linker, this
   does not link in the entire given module, only
   the called functions are cloned from the input.
   This is to speed up the linking of the kernel lib
   which is so big, that it takes seconds to clone it,
   even on top-of-the line current processors

   Copyright 2014 Kalle Raiskila.
   This file is a part of pocl, distributed under the MIT
   licence. See file COPYING.
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

#include "linker.h"

using namespace llvm;

//#include <cstdio>
//#define DB_PRINT(...) printf("linker:" __VA_ARGS__)
#define DB_PRINT(...)

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
                Type *new_t = PointerType::get(pe_type, POCL_ADDRESS_SPACE_GLOBAL);
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
                if (t->getPointerAddressSpace() != POCL_ADDRESS_SPACE_GLOBAL) {
                    new_t = PointerType::get(pe_type, POCL_ADDRESS_SPACE_GLOBAL);
                    changed = 1;
                }
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

/* Find all functions in the calltree of F, append their
 * name to list.
 */
static inline void
find_called_functions(llvm::Function *            F,
                      std::list<llvm::StringRef> &list)
{
    llvm::Function::iterator fi,fe;
    for (fi=F->begin(), fe=F->end();
         fi != fe;
         fi++) {
        llvm::BasicBlock::iterator bi,be;
        for (bi=fi->begin(), be=fi->end();
             bi != be;
             bi++) {
            CallInst *CI=dyn_cast<CallInst>(bi);
            if (CI == NULL)
                continue;
            llvm::Function *callee=CI->getCalledFunction();
            // this happens with e.g. inline asm calls
            if (callee == NULL)
                continue;
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
CopyFunc( const llvm::StringRef Name,
          const llvm::Module *  From,
          llvm::Module *        To,
          ValueToValueMapTy &   VVMap)
{
    llvm::Function *SrcFunc=From->getFunction(Name);
    // TODO: is this the linker error "not found", and not an assert?
    assert(SrcFunc && "Did not find function to copy in kernel library");
    llvm::Function *DstFunc=To->getFunction(Name);

    if (DstFunc == NULL) {
        DB_PRINT("   %s not found in destination module, creating\n",
                 Name.data());
        DstFunc=
            Function::Create(cast<FunctionType>(
                                 SrcFunc->getType()->getElementType()),
                             SrcFunc->getLinkage(),
                             SrcFunc->getName(),
                             To);
        DstFunc->copyAttributesFrom(SrcFunc);
    }
    VVMap[SrcFunc]=DstFunc;

    Function::arg_iterator j=DstFunc->arg_begin();
    for (Function::const_arg_iterator i=SrcFunc->arg_begin(),
         e=SrcFunc->arg_end();
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
static void
copy_func_callgraph( const llvm::StringRef func_name,
                     const llvm::Module *  from,
                     llvm::Module *        to,
                     ValueToValueMapTy &   vvm)
{
    std::list<llvm::StringRef> callees;
    llvm::Function *rootfunc=from->getFunction(func_name);
    if (rootfunc == NULL)
        return;
    DB_PRINT("copying function %s with callgraph\n", func_name.data());

    find_called_functions(rootfunc, callees);

    // Fisrt copy the callees of func, then the function itself.
    // Recurse into callees to handle the case where kernel library
    // functions call other kernel library functions.
    std::list<llvm::StringRef>::iterator ci,ce;
    for (ci=callees.begin(), ce=callees.end();
         ci != ce;
         ci++) {
        llvm::Function *SrcFunc=from->getFunction(*ci);
        if (!SrcFunc->isDeclaration()) {
            copy_func_callgraph(*ci,from, to, vvm);
        }
        CopyFunc(*ci, from, to, vvm);
    }

    CopyFunc(func_name, from, to, vvm);
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

void
link(llvm::Module *krn, const llvm::Module *lib)
{
    assert(krn);
    assert(lib);
    ValueToValueMapTy vvm;
    std::list<llvm::StringRef> declared;

    llvm::Module::iterator fi,fe;

    // Find and fix opencl.imageX_t arguments
    for (fi=krn->begin(), fe=krn->end();
         fi != fe;
         fi++) {
        llvm::Function *f = &*fi;
        if (f->isDeclaration())
            continue;
        // need to restart iteration if we replace a function
        if (CloneFuncFixOpenCLImageT(krn, f) != f) {
            fi = krn->begin();
        }
    }

    // Inspect the kernel, find undefined functions
    for (fi=krn->begin(), fe=krn->end();
         fi != fe;
         fi++) {
        if ((*fi).isDeclaration()) {
            DB_PRINT("%s is not defined\n", fi->getName().data());
            declared.push_back(fi->getName());
            continue;
        }

        // Find all functions the kernel source calls
        // TODO: is there no direct way?
        find_called_functions(&*fi, declared);
    }
    declared.sort(stringref_cmp);
    declared.unique(stringref_equal);

    // copy all the globals from lib to krn.
    // it probably is faster to just copy them all, than to inspect
    // both krn and lib to find which actually are used.
    DB_PRINT("cloning the global variables:\n");
    llvm::Module::const_global_iterator gi,ge;
    for (gi=lib->global_begin(), ge=lib->global_end();
         gi != ge;
         gi++) {
        DB_PRINT(" %s\n", gi->getName().data());
        GlobalVariable *GV=new GlobalVariable(*krn,
                                              gi->getType()->getElementType(),
                                              gi->isConstant(),
                                              gi->getLinkage(),
                                              (Constant*) 0, gi->getName(),
                                              (GlobalVariable*) 0,
                                              gi->getThreadLocalMode(),
                                              gi->getType()->getAddressSpace());
        GV->copyAttributesFrom(&*gi);
        vvm[&*gi]=GV;
    }

    // For each undefined function in krn, clone it from the lib to the krn module,
    // if found in lib
    std::list<llvm::StringRef>::iterator di,de;
    for (di=declared.begin(), de=declared.end();
         di != de;
         di++) {
        copy_func_callgraph( *di, lib, krn, vvm);
    }

    // copy any aliases to krn
    DB_PRINT("cloning the aliases:\n");
    llvm::Module::const_alias_iterator ai, ae;
    for (ai = lib->alias_begin(), ae = lib->alias_end();
         ai != ae;
         ai++) {
        DB_PRINT(" %s\n", ai->getName().data());
        GlobalAlias *GA =
#ifndef LLVM_3_7
            GlobalAlias::create(ai->getType(),
                                ai->getType()->getAddressSpace(),
                                ai->getLinkage(), ai->getName(), NULL, krn);
#else
            GlobalAlias::create(ai->getType(),
                                ai->getLinkage(), ai->getName(), NULL, krn);
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
         mi != me;
         mi++) {
        const NamedMDNode &NMD=*mi;
        DB_PRINT(" %s:\n", NMD.getName().data());
        NamedMDNode *NewNMD=krn->getOrInsertNamedMetadata(NMD.getName());
        for (unsigned i=0, e=NMD.getNumOperands(); i != e; ++i)
            NewNMD->addOperand(MapMetadata(NMD.getOperand(i), vvm));
    }
}

/* vim: set expandtab ts=4 : */

