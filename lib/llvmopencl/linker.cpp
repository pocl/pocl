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

#include <llvm/ADT/StringSet.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/PassInfo.h>
#include <llvm/PassRegistry.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include "pocl_cl.h"

#include "LLVMUtils.h"
#include "linker.h"


using namespace llvm;

// #include <cstdio>
// #define DB_PRINT(...) printf("linker:" __VA_ARGS__)
#define DB_PRINT(...)

namespace pocl {

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
      if (!Callee->hasName()) {
        if (F->getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
          // This is an SPIR-V entry point wrapper function: SPIR-V
          // translator generates these because OpenCL allows calling
          // kernels from kernels like they were device side functions
          // whereas SPIR-V entry points cannot call other entry points.

          Callee->setName(std::string("_spirv_wrapped_") + F->getName());

          // The SPIR-V translator loses the kernel's original DISubprogram
          // info, leaving it to the wrapper, thus even after inlining the
          // function to the kernel we do not get any debug info (LLVM checks
          // for DISubprogram for each function it generates the debug info
          // for). Just reuse the DISubprogram in the kernel here in that case.
          if (Callee->getSubprogram() != nullptr &&
              F->getSubprogram() == nullptr &&
              Callee->getSubprogram()->getName() == F->getName()) {
            F->setSubprogram(pocl::mimicDISubprogram(
                Callee->getSubprogram(),
                std::string("_spirv_wrapped_") + F->getName().str(), nullptr));
            CI->setDebugLoc(llvm::DILocation::get(
                Callee->getContext(), Callee->getSubprogram()->getLine(), 0,
                F->getSubprogram(), nullptr, true));
          }
        } else {
          Callee->setName("__noname_function");
        }
      }
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
         ValueToValueMapTy &   VVMap) {

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
                    ValueToValueMapTy &   vvm) {
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
        copy_func_callgraph(ci->getKey(), from, to, vvm);
      } else {
        DB_PRINT("%s is declaration, not recursing into it!\n",
                 SrcFunc->getName().str().c_str());
      }
      CopyFunc(ci->getKey(), from, to, vvm);
    }
    CopyFunc(func_name, from, to, vvm);
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

  /* LLVM 1x complains about this being an invalid MDnode. */
  llvm::NamedMDNode *DebugCU = program->getNamedMetadata("llvm.dbg.cu");
  if (DebugCU && DebugCU->getNumOperands() == 0)
    program->eraseNamedMetadata(DebugCU);
}



}

using namespace pocl;

int link(llvm::Module *Program, const llvm::Module *Lib,
         std::string &log, const char **DevAuxFuncs) {

  assert(Program);
  assert(Lib);
  ValueToValueMapTy vvm;
  llvm::StringSet<> DeclaredFunctions;

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
      if (copy_func_callgraph(r, Lib, Program, vvm)) {
        Function *f = Program->getFunction(r);

        if (f->getName().equals("__to_local") ||
            f->getName().equals("__to_global") ||
            f->getName().equals("__to_private")) {

          // Special handling for the AS cast built-ins: They do not use
          // type mangling, which complicates the CPU implementation which
          // sometimes gets all-0 AS input and sometimes not (SPIR-V).
          // Here, in case the target doesn't define these built-ins in the
          // bitcode library, we replace the calls directly with address
          // space casts. This works for uniform address space targets (CPUs),
          // while the disjoint AS targets can override the implementation
          // as needed.
          for (auto *U : f->users()) {
            if (llvm::CallInst *Call = dyn_cast<llvm::CallInst>(U)) {
              llvm::AddrSpaceCastInst *AsCast = new llvm::AddrSpaceCastInst(
                Call->getArgOperand(0),
                Call->getFunctionType()->getReturnType(),
                f->getName() + ".as_cast", Call);
              Call->replaceAllUsesWith(AsCast);
              Call->eraseFromParent();
            }
          }
          continue;
        }

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
                          const llvm::Module *program,
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
  copy_func_callgraph(kernel_name, program, parallel_bc, vvm);

  if (DevAuxFuncs) {
    const char **Func = DevAuxFuncs;
    while (*Func != nullptr) {
      copy_func_callgraph(*Func++, program, parallel_bc, vvm);
    }
  }

  std::string log;
  shared_copy(parallel_bc, program, log, vvm);

  if (pocl_get_bool_option("POCL_LLVM_ALWAYS_INLINE", 0)) {
    llvm::Module::iterator MI, ME;
    for (MI = parallel_bc->begin(), ME = parallel_bc->end(); MI != ME; ++MI) {
      Function *F = &*MI;
      if (F->isDeclaration())
          continue;
      // inline all except the kernel
      if (F->getName() != name) {
          F->addFnAttr(Attribute::AlwaysInline);
      }
    }

    llvm::legacy::PassManager Passes;
    Passes.add(createAlwaysInlinerLegacyPass());
    Passes.run(*parallel_bc);
  }
  return 0;
}

bool moveProgramScopeVarsOutOfProgramBc(llvm::LLVMContext *Context,
                                        llvm::Module *ProgramBC,
                                        llvm::Module *OutputBC,
                                        unsigned DeviceLocalAS) {

  ValueToValueMapTy VVM;
  llvm::Module::global_iterator GI, GE;

  // Copy all the globals from input to output module.
  DB_PRINT("cloning the global variables:\n");
  for (GI = ProgramBC->global_begin(), GE = ProgramBC->global_end(); GI != GE;
       GI++) {
    DB_PRINT(" %s\n", GI->getName().data());
    GlobalVariable *GV = new GlobalVariable(
        *OutputBC, GI->getValueType(), GI->isConstant(), GI->getLinkage(),
        (Constant *)0, GI->getName(), (GlobalVariable *)0,
        GI->getThreadLocalMode(), GI->getType()->getAddressSpace());
    GV->copyAttributesFrom(&*GI);
    // for program scope vars, change linkage to external
    if (isProgramScopeVariable(*GV, DeviceLocalAS)) {
      GV->setDSOLocal(false);
      GV->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
    }
    VVM[&*GI] = GV;
  }

  // add initializers to global vars, copy aliases & MD
  std::string log;
  shared_copy(OutputBC, ProgramBC, log, VVM);

  // change program-scope variables in Input module to references
  std::list<GlobalVariable *> GVarsToErase;
  std::list<GlobalVariable *> ProgramGVars;
  for (GI = ProgramBC->global_begin(), GE = ProgramBC->global_end(); GI != GE;
       GI++) {
    ProgramGVars.push_back(&*GI);
  }

  // we can't use iteration over ProgramBC->global_begin/end here, because we're
  // adding new GVars into the Module during iteration
  for (auto GI : ProgramGVars) {
    DB_PRINT(" %s\n", GI->getName().data());
    if (isProgramScopeVariable(*GI, DeviceLocalAS)) {
      std::string OrigName = GI->getName().str();
      GI->setName(Twine(OrigName, "__old"));
      GlobalVariable *NewGV = new GlobalVariable(
          *ProgramBC, GI->getValueType(), GI->isConstant(),
          GlobalValue::LinkageTypes::ExternalLinkage,
          (Constant *)nullptr, // initializer
          OrigName,
          (GlobalVariable *)nullptr, // insert-before
          GI->getThreadLocalMode(), GI->getType()->getAddressSpace());

      NewGV->copyAttributesFrom(GI);
      NewGV->setDSOLocal(false);
      GI->replaceAllUsesWith(NewGV);
      GVarsToErase.push_back(GI);
    }
  }
  for (auto GV : GVarsToErase) {
    GV->eraseFromParent();
  }

  // copy functions to the output.bc, then replace them with references
  // in the original program.bc.
  // This can be useful in combination with enabled JIT, if the user program's
  // kernels call a lot of subroutines, and those subroutines are shared
  // (called) by multiple kernels.
  // With this enabled, subroutines are separated & compiled to ZE module only
  // once (together with program-scope variables), and then linked (with ZE
  // linker). If disabled, subroutines are copied & compiled for each kernel
  // separately.
  if (pocl_get_bool_option("POCL_LLVM_MOVE_NONKERNELS", 0)) {
    std::list<llvm::Function *> FunctionsToErase;
    llvm::Module::iterator MI, ME;
    for (MI = ProgramBC->begin(), ME = ProgramBC->end(); MI != ME; ++MI) {
      Function *F = &*MI;
      if (F->isDeclaration())
          continue;

      if (!F->hasName()) {
          F->setName("anonymous_func__");
      }

      std::string FName = F->getName().str();
      if (F->getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
          // POCL_MSG_WARN("NOT copying kernel function %s\n", FName.c_str());
      } else {
          if (F->getCallingConv() == llvm::CallingConv::SPIR_FUNC)
          POCL_MSG_WARN("Copying non-kernel function %s\n", FName.c_str());
          copy_func_callgraph(F->getName(), ProgramBC, OutputBC, VVM);
          Function *DestF = OutputBC->getFunction(FName);
          assert(DestF);
          DestF->setDSOLocal(false);
          DestF->setVisibility(GlobalValue::VisibilityTypes::DefaultVisibility);
          DestF->setLinkage(llvm::GlobalValue::ExternalLinkage);
          Twine TempName(FName, "__decl");
          Function *NewFDecl =
              Function::Create(F->getFunctionType(), Function::ExternalLinkage,
                               F->getAddressSpace(), TempName, ProgramBC);
          assert(NewFDecl);
          F->setName(Twine(FName, "___old"));
          F->replaceAllUsesWith(NewFDecl);
          NewFDecl->setName(FName);
          FunctionsToErase.push_back(F);
      }
    }

    for (auto FF : FunctionsToErase) {
      FF->eraseFromParent();
    }
  }

  return true;
}

/* vim: set expandtab ts=4 : */

