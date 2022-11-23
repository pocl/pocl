// Lightweight bitcode linker to replace llvm::Linker.
//
// Copyright (c) 2014 Kalle Raiskila
//               2016-2022 Pekka Jääskeläinen
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

/* Lightweight bitcode linker to replace llvm::Linker. Unlike the LLVM default
   linker, this does not link in the entire given module, only the called
   functions are cloned from the input. This is to speed up the linking of the
   kernel lib which is so big, that it takes seconds to clone it,  even on
   top-of-the line current processors. */

#include <iostream>
#include <set>

#include "config.h"
#include "pocl.h"
#include "pocl_llvm_api.h"

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/ADT/StringSet.h"
#include "pocl_cl.h"

#include "LLVMUtils.h"
#include "linker.h"


using namespace llvm;

// #include <cstdio>
// #define DB_PRINT(...) printf("linker:" __VA_ARGS__)
#define DB_PRINT(...)

// Find all functions in the calltree of F, append their
// name to function name set.
static inline void
find_called_functions(llvm::Function *F,
                      llvm::StringSet<> &FNameSet)
{
  if (F->isDeclaration()) {
    DB_PRINT("it's a declaration.\n");
    return;
  }
  llvm::Function::iterator FI;
  for (FI = F->begin(); FI != F->end(); FI++) {
    llvm::BasicBlock::iterator BI;
    for (BI = FI->begin(); BI != FI->end(); BI++) {
      CallInst *CI = dyn_cast<CallInst>(BI);
      if (CI == NULL)
        continue;
      llvm::Function *Callee = CI->getCalledFunction();
      // this happens with e.g. inline asm calls
      if (Callee == NULL) {
        DB_PRINT("search: %s callee NULL\n", F->getName().str().c_str());
        continue;
      }
      if (Callee->getCallingConv() == llvm::CallingConv::SPIR_FUNC ||
          CI->getCallingConv() == llvm::CallingConv::SPIR_FUNC) {
        // Loosen the CC to the default one. It should be always the
        // preferred one to SPIR_FUNC at this stage.
        Callee->setCallingConv(llvm::CallingConv::C);
        CI->setCallingConv(llvm::CallingConv::C);
      }
      if (!Callee->hasName())
        Callee->setName("__anonymous_internal_func__");
      const char* Name = Callee->getName().data();
      DB_PRINT("search: %s calls %s\n",
               F->getName().data(), Name);
      if (FNameSet.count(Callee->getName()) > 0)
        continue;
      else {
        DB_PRINT("inserting %s\n", Name);
        FNameSet.insert(Callee->getName());
        DB_PRINT("search: recursing into %s\n", Name);
        find_called_functions(Callee, FNameSet);
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
         ValueToValueMapTy &   VVMap,
         unsigned AS) {

    llvm::Function *SrcFunc = From->getFunction(Name);
    // TODO: is this the linker error "not found", and not an assert?
    assert(SrcFunc && "Did not find function to copy in kernel library");
    llvm::Function *DstFunc = To->getFunction(Name);

    if (DstFunc == NULL) {
        DB_PRINT("   %s not found in destination module, creating\n",
                 Name.data());
        DstFunc =
        Function::Create(cast<FunctionType>(SrcFunc->getValueType()),
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

        CloneFunctionIntoAbs(DstFunc, SrcFunc, VVMap, RI, false);
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
                    ValueToValueMapTy &   vvm,
                    unsigned AS) {
    llvm::StringSet<> callees;
    llvm::Function *RootFunc = from->getFunction(func_name);
    if (RootFunc == NULL)
      return -1;
    DB_PRINT("copying function %s with callgraph\n", RootFunc->getName().data());

    find_called_functions(RootFunc, callees);

    // First copy the callees of func, then the function itself.
    // Recurse into callees to handle the case where kernel library
    // functions call other kernel library functions.
    llvm::StringSet<>::iterator ci,ce;
    for (ci = callees.begin(), ce = callees.end(); ci != ce; ci++) {
      llvm::Function *SrcFunc = from->getFunction(ci->getKey());
      if (!SrcFunc->isDeclaration()) {
        copy_func_callgraph(ci->getKey(), from, to, vvm, AS);
      } else {
        DB_PRINT("%s is declaration, not recursing into it!\n",
		 SrcFunc->getName().str().c_str());
      }
      CopyFunc(ci->getKey(), from, to, vvm, AS);
    }
    CopyFunc(func_name, from, to, vvm, AS);
    return 0;
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
      GlobalAlias::create(
        ai->getType(), ai->getType()->getAddressSpace(), ai->getLinkage(),
        ai->getName(), NULL, program);

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
    const NamedMDNode &NMD = *mi;
    // This causes problems with NVidia, and is regenerated by pocl-ptx-gen
    // anyway.
    if (NMD.getName() == StringRef("nvvm.annotations"))
      continue;
    DB_PRINT(" %s:\n", NMD.getName().data());
    if (program->getNamedMetadata(NMD.getName())) {
      // Let's not overwrite existing metadata such as llvm.module.flags and
      // opencl.ocl.version.
      continue;
    }
    NamedMDNode *NewNMD = program->getOrInsertNamedMetadata(NMD.getName());
    for (unsigned i = 0, e = NMD.getNumOperands(); i != e; ++i)
      NewNMD->addOperand(MapMetadata(NMD.getOperand(i), vvm));
  }
}


// Printf requires special treatment at bitcode link time for seamless SPIR-V
// import support: The printf we link in might be SPIR-V compliant with the format
// string address space in the constant space or the other way around. The calls to
// the printf, however, depend whether we are importing the kernel from SPIR or
// through OpenCL C native compilation path. In the former, the kernel calls refer
// to constant address space in the format string, and in the latter, when compiling
// natively to CPUs and other flat address space targets, the calls see an AS0 format
// string address space due to Clang's printf declaration adhering to target address
// spaces.
//
// In this function we fix calls to the printf to refer to one in the bitcode
// library's printf, with the correct AS for the format string. Other considered
// options include building two different bitcode libraries: One with SPIR-V
// address spaces, another with the target's (flat) AS. This would be
// problematic in other ways and redundant.
void unifyPrintfFingerPrint(llvm::Module *Program, const llvm::Module *Lib) {

  llvm::Function *CalledPrintf = Program->getFunction("printf");
  llvm::Function *LibPrintf = Lib->getFunction("printf");
  llvm::Function *NewPrintf = nullptr;

  assert(LibPrintf != nullptr);
  if (CalledPrintf != nullptr && CalledPrintf->getArg(0)->getType() !=
      LibPrintf->getArg(0)->getType()) {
    CalledPrintf->setName("_old_printf");
    // Create a declaration with a fingerprint with the correct format argument
    // type which we will import from the BC library.
    NewPrintf =
        Function::Create(
            LibPrintf->getFunctionType(), LibPrintf->getLinkage(), "printf",
            Program);
  } else {
    // No printf fingerprint mismatch detected in this module.
    return;
  }

  // Fix the printf calls to point to the library imported declaration.
  while (CalledPrintf->getNumUses() > 0) {
    auto U = CalledPrintf->user_begin();
    llvm::CallInst *Call = dyn_cast<llvm::CallInst>(*U);
    if (Call == nullptr)
      continue;
    auto Cast =
        llvm::CastInst::CreatePointerBitCastOrAddrSpaceCast(
            Call->getArgOperand(0), NewPrintf->getArg(0)->getType(),
            "fmt_str_cast", Call);
    Call->setCalledFunction(NewPrintf);
    Call->setArgOperand(0, Cast);
  }
  CalledPrintf->eraseFromParent();
}

int link(llvm::Module *Program, const llvm::Module *Lib, std::string &log,
         unsigned global_AS, const char **DevAuxFuncs) {

  assert(Program);
  assert(Lib);
  ValueToValueMapTy vvm;
  llvm::StringSet<> DeclaredFunctions;

  unifyPrintfFingerPrint(Program, Lib);

  // Include auxiliary functions required by the device at hand.
  if (DevAuxFuncs) {
    const char **Func = DevAuxFuncs;
    while (*Func != nullptr) {
      DeclaredFunctions.insert(*Func++);
    }
  }

  llvm::Module::iterator fi, fe;

  // Inspect the program, find undefined functions
  for (fi = Program->begin(), fe = Program->end(); fi != fe; fi++) {
    if (fi->isDeclaration()) {
      DB_PRINT("%s is not defined\n", fi->getName().data());
      DeclaredFunctions.insert(fi->getName());
      continue;
    }

    // anonymous functions have no name, which breaks the algorithm later
    // when it searches for undefined functions in the kernel library.
    // assign a name here, this should be made unique by setName()
    if (!fi->hasName()) {
      fi->setName("__anonymous_internal_func__");
    }
    DB_PRINT("Function '%s' is defined\n", fi->getName().data());
    // Find all functions the program source calls
    // TODO: is there no direct way?
    find_called_functions(&*fi, DeclaredFunctions);
  }

  // some global variables can have external linkage. Set it to private
  // otherwise these end up in ELF relocation tables and cause link
  // failures when linking with -fPIC/PIE
  llvm::Module::global_iterator gi1, ge1;
  for (gi1 = Program->global_begin(), ge1 = Program->global_end(); gi1 != ge1;
       gi1++) {
    GlobalValue::LinkageTypes linkage = gi1->getLinkage();
    if (linkage == GlobalValue::LinkageTypes::ExternalLinkage)
      gi1->setLinkage(GlobalValue::LinkageTypes::PrivateLinkage);
  }

  // Copy all the globals from lib to program.
  // It probably is faster to just copy them all, than to inspect
  // both program and lib to find which actually are used.
  DB_PRINT("cloning the global variables:\n");
  llvm::Module::const_global_iterator gi,ge;
  for (gi = Lib->global_begin(), ge = Lib->global_end(); gi != ge; gi++) {
    DB_PRINT(" %s\n", gi->getName().data());
    GlobalVariable *GV = new GlobalVariable(
      *Program, gi->getValueType(), gi->isConstant(),
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
  llvm::StringSet<>::iterator di,de;
  for (di = DeclaredFunctions.begin(), de = DeclaredFunctions.end();
       di != de; di++) {
      llvm::StringRef r = di->getKey();
      if (copy_func_callgraph(r, Lib, Program, vvm, global_AS)) {
        Function *f = Program->getFunction(r);
        if ((f == NULL) ||
            (f->isDeclaration() &&
             // A target might want to expose the C99 printf in case not supporting
             // the OpenCL 1.2 printf.
             !f->getName().equals("printf") &&
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

  shared_copy(Program, Lib, log, vvm);

  return 0;
}

int copyKernelFromBitcode(const char* name, llvm::Module *parallel_bc,
                          const llvm::Module *program, unsigned global_AS,
                          const char **DevAuxFuncs) {
  ValueToValueMapTy vvm;

  // Copy all the globals from lib to program.
  // It probably is faster to just copy them all, than to inspect
  // both program and lib to find which actually are used.
  DB_PRINT("cloning the global variables:\n");
  llvm::Module::const_global_iterator gi,ge;
  for (gi=program->global_begin(), ge=program->global_end(); gi != ge; gi++) {
    DB_PRINT(" %s\n", gi->getName().data());
    GlobalVariable *GV = new GlobalVariable(
      *parallel_bc, gi->getValueType(), gi->isConstant(),
      gi->getLinkage(), (Constant*)0, gi->getName(), (GlobalVariable*)0,
      gi->getThreadLocalMode(), gi->getType()->getAddressSpace());
    GV->copyAttributesFrom(&*gi);
    vvm[&*gi]=GV;
  }

  const StringRef kernel_name(name);
  copy_func_callgraph(kernel_name, program, parallel_bc, vvm, global_AS);

  if (DevAuxFuncs) {
    const char **Func = DevAuxFuncs;
    while (*Func != nullptr) {
      copy_func_callgraph(*Func++, program, parallel_bc, vvm, global_AS);
    }
  }

  std::string log;
  shared_copy(parallel_bc, program, log, vvm);

  /* LLVM 13 complains about this being an invalid MDnode. */
  llvm::NamedMDNode *DebugCU = parallel_bc->getNamedMetadata("llvm.dbg.cu");
  if (DebugCU && DebugCU->getNumOperands()==0)
    parallel_bc->eraseNamedMetadata(DebugCU);

  return 0;
}

/* vim: set expandtab ts=4 : */

