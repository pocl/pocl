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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS

IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugLoc.h"
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
#include <llvm/Transforms/Utils/AMDGPUEmitPrintf.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include "pocl_cl.h"
#include "pocl_llvm_api.h"

#include "EmitPrintf.hh"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "linker.h"

using namespace llvm;

// #include <cstdio>
// #define DB_PRINT(...) printf("linker:" __VA_ARGS__)
#define DB_PRINT(...)

namespace pocl {

// A workaround for issue #889. In some cases, functions seem
// to get multiple DISubprogram nodes attached. This causes
// the llvm::verifyModule to complain, and
// LLVM to segfault in some configurations (ARM, x86 in distro mode, ...)
// Erase the MD nodes if we detect the condition.
// TODO this needs further investigation & a proper fix
static bool removeDuplicateDbgInfo(Module *Mod) {

  bool Erased = false;

  for (Function &F : Mod->functions()) {

    bool EraseMdDbg = false;
    // Get the function metadata attachments.
    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    F.getAllMetadata(MDs);
    unsigned NumDebugAttachments = 0;

    for (const auto &I : MDs) {
      if (I.first == LLVMContext::MD_dbg) {
        if (isa<DISubprogram>(I.second)) {
          DISubprogram *DI = cast<DISubprogram>(I.second);
          if (!DI->isDistinct()) {
            EraseMdDbg = true;
          }
        }
      }
    }

    if (EraseMdDbg) {
      F.eraseMetadata(LLVMContext::MD_dbg);
      Erased = true;
    }
  }
  return Erased;
}

// fix mismatches between calling conv. This should not happen,
// but sometimes can, esp with SPIR(-V) input
static void fixCallingConv(llvm::Module *Mod, std::string &Log) {
  for (llvm::Module::iterator MI = Mod->begin(); MI != Mod->end(); ++MI) {
    llvm::Function *F = &*MI;
    if (F->isDeclaration())
      continue;

    for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
      for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE;
           ++BI) {
        Instruction *Instr = dyn_cast<Instruction>(BI);
        if (!llvm::isa<CallInst>(Instr))
          continue;
        CallInst *CallInstr = dyn_cast<CallInst>(Instr);
        Function *Callee = CallInstr->getCalledFunction();

        if ((Callee == nullptr) || Callee->getName().starts_with("llvm.") ||
            Callee->isDeclaration())
          continue;

        if (CallInstr->getCallingConv() != Callee->getCallingConv()) {
          std::string CalleeName, CallerName;
          if (F->hasName())
            CallerName = F->getName().str();
          else
            CallerName = "unnamed";
          if (Callee->hasName())
            CalleeName = Callee->getName().str();
          else
            CalleeName = "unnamed";

          Log.append("Warning: CallingConv mismatch: \n Caller is: ");
          Log.append(CallerName);
          Log.append(" and Callee is: ");
          Log.append(CalleeName);
          Log.append("; fixing \n");
          CallInstr->setCallingConv(Callee->getCallingConv());
        }
      }
    }
  }
}

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
      if (!Callee->hasName()) {
        Callee->setName("__anonymous_function");
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
      llvm::StringRef Name = ci->getKey();
      llvm::Function *SrcFunc = from->getFunction(Name);
      if (!SrcFunc->isDeclaration()) {
        copy_func_callgraph(Name, from, to, vvm);
      } else {
        DB_PRINT("%s is declaration, not recursing into it!\n",
                 SrcFunc->getName().str().c_str());
      }
      CopyFunc(Name, from, to, vvm);
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

/* Replace printf calls with generated bitcode that stores the format-string
 * and the arguments to a printf buffer. The replacement bitcode generated
 * by emitPrintfCall is:
 *
 * 1) get the format string length and argument sizes
 * 2) call __printf_alloc to allocate storage from device's printf buffer
 *    to allocate the size from 0)
 * 3) store the arguments using Store or Memcpy instructions
 * 4) optionally call __printf_flush_buffer (if the device supports
 *    immediate flushing) to immediately print the buffer content
 *
 * At some point the host-side code in printf_buffer.c decodes
 * the printf buffer content and writes it to STDOUT.
 * Note that this gets rid of variadic arguments on the kernel side,
 * however, Clang still does printf() argument promotions when compiling
 * OpenCL C source code, and these promotions also depends on target device.
 * Currently this is solved by passing a bunch of flags here (from device
 * properties) to the emitPrintfCall, which stores them in the "control word"
 * in the printf buffer. The decoding code on the host side reads the flags
 * to know the promotions used for arguments. This could also be implemented
 * by storing each argument's size in the printf buffer, but that's TBD
*/
static void handleDeviceSidePrintf(
    llvm::Module *Program, const llvm::Module *Lib, std::string &Log,
    ValueToValueMapTy &vvm, cl_device_id ClDev) {

  // if a device supports immediate flush, a declaration must exist in the
  // the device's builtin library (the definition is on the host side in
  // libpocl)
  bool DeviceSupportsImmediateFlush =
      Lib->getFunction("__printf_flush_buffer") != nullptr;

  PrintfCallFlags PFlags;
  PFlags.PrintfBufferAS = ClDev->global_as_id;
  PFlags.IsBuffered = true;
  PFlags.DontAlign = true;
  PFlags.FlushBuffer = DeviceSupportsImmediateFlush;

  if (ClDev->type == CL_DEVICE_TYPE_CPU) {
    // for CPU, store constant format strings as simply a pointer
    // the original AMDGPU emitPrintfCall uses MD5 but that is
    // unnecessary complication for CPU devices
    PFlags.StorePtrInsteadOfMD5 = true;
    PFlags.AlwaysStoreFmtPtr = false;
#if defined(__arm__) || defined(__aarch64__)
    /* ARM seems to promote char2 to int */
    PFlags.ArgPromotionChar2 = true;
#endif
  } else {
    // for non-CPU devices, always store the format string in the buffer,
    // even if it's a constant format string, since we cannot use pointers
    PFlags.StorePtrInsteadOfMD5 = false;
    PFlags.AlwaysStoreFmtPtr = true;
  }

  // C promotion of char/short -> int32
  PFlags.ArgPromotionCharShort = true;
  // C promotion of float -> double only if device supports double
  if (ClDev->double_fp_config == 0) {
    PFlags.ArgPromotionFloat = false;
  }
  // big endian support is not implemented at all currently
  if (!ClDev->endian_little) {
    PFlags.IsBigEndian = true;
  }

  Function *CalledPrintf = Program->getFunction("printf");
  if (CalledPrintf) {

    Function *PrintfAlloc = Lib->getFunction("__printf_alloc");
    assert(PrintfAlloc != nullptr);
    copy_func_callgraph("__printf_alloc", Lib, Program, vvm);
    copy_func_callgraph("__pocl_printf_alloc", Lib, Program, vvm);
    if (DeviceSupportsImmediateFlush)
      copy_func_callgraph("__printf_flush_buffer", Lib, Program, vvm);

    std::set<CallInst *> Calls;
    for (auto U : CalledPrintf->users()) {
      CallInst *CI = dyn_cast<CallInst>(U);
      if (CI == nullptr)
        continue;
      if (CI->getCalledFunction() == nullptr)
        continue;
      if (CI->getCalledFunction() != CalledPrintf)
        continue;
      Calls.insert(CI);
    }

    for (CallInst *C : Calls) {
      llvm::IRBuilder<> Builder(C);
      llvm::SmallVector<llvm::Value *> Args(C->arg_begin(), C->arg_end());
      Value *Replacement = pocl::emitPrintfCall(Builder, Args, PFlags);
      C->replaceAllUsesWith(Replacement);
    }

    for (CallInst *C : Calls) {
      C->eraseFromParent();
    }

    if (CalledPrintf->getNumUses() == 0)
      CalledPrintf->eraseFromParent();
  }
}

static void replaceIntrinsics(llvm::Module *Program, const llvm::Module *Lib,
                              ValueToValueMapTy &vvm, cl_device_id ClDev) {
  llvm_intrin_replace_fn IntrinRepl = ClDev->llvm_intrin_replace;
  std::map<Function *, Function *> EraseMap;
  llvm::Module::iterator FI, FE;
  StringRef LlvmIntrins("llvm.");
  for (FI = Program->begin(), FE = Program->end(); FI != FE; FI++) {
    if (FI->isDeclaration()) {
      if (IntrinRepl && FI->hasName() &&
          FI->getName().starts_with(LlvmIntrins)) {
        const char *ReplacementName =
            IntrinRepl(FI->getName().data(), FI->getName().size());
        if (ReplacementName) {
          copy_func_callgraph(ReplacementName, Lib, Program, vvm);
          Function *Repl = Program->getFunction(ReplacementName);
          assert(Repl);
          EraseMap[&*FI] = Repl;
          continue;
        }
      }
    }
  }

  for (auto [Intrin, Repl] : EraseMap) {
    Intrin->replaceAllUsesWith(Repl);
    Intrin->eraseFromParent();
  }
}
}

using namespace pocl;

int link(llvm::Module *Program, const llvm::Module *Lib, std::string &Log,
         cl_device_id ClDev) {

  assert(Program);
  assert(Lib);
  ValueToValueMapTy vvm;
  llvm::StringSet<> DeclaredFunctions;

  // Include auxiliary functions required by the device at hand.
  if (ClDev->device_aux_functions) {
    const char **Func = ClDev->device_aux_functions;
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
      fi->setName("__anonymous_function");
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

        if (f->getName() == "__to_local" ||
            f->getName() == "__to_global" ||
            f->getName() == "__to_private") {

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
            PointerType *ArgPT =
                dyn_cast<PointerType>(Call->getArgOperand(0)->getType());
            PointerType *RetPT =
                dyn_cast<PointerType>(Call->getFunctionType()->getReturnType());
            if (ArgPT == nullptr || RetPT == nullptr) {
              Log.append("Invalid use of operator __to_{local,global,private}");
              found_all_undefined = false;
              break;
            }
            if (ArgPT->getAddressSpace() == RetPT->getAddressSpace()) {
              Value *V = Call->getArgOperand(0);
              Call->replaceAllUsesWith(V);
              Call->eraseFromParent();
            } else {
              llvm::AddrSpaceCastInst *AsCast = new llvm::AddrSpaceCastInst(
                  Call->getArgOperand(0),
                  Call->getFunctionType()->getReturnType(),
                  f->getName() + ".as_cast", Call);
              Call->replaceAllUsesWith(AsCast);
              Call->eraseFromParent();
            }
            }
          }
          continue;
        }

        if ((f == NULL) || (f->isDeclaration() &&
                            // A target might want to expose the C99 printf in
                            // case not supporting the OpenCL 1.2 printf.
                            f->getName() != "printf" &&
                            f->getName() != pocl_sampler_handler &&
                            !f->getName().starts_with(llvm_intrins))) {
          Log.append("Cannot find symbol ");
          Log.append(r.str());
          Log.append(" in kernel library\n");
          found_all_undefined = false;
        }
      }
  }

  if (!found_all_undefined)
    return 1;

  shared_copy(Program, Lib, Log, vvm);

  removeDuplicateDbgInfo(Program);

  fixCallingConv(Program, Log);

  if (ClDev->device_side_printf)
    handleDeviceSidePrintf(Program, Lib, Log, vvm, ClDev);

  replaceIntrinsics(Program, Lib, vvm, ClDev);
  return 0;
}

int copyKernelFromBitcode(const char* Name, llvm::Module *ParallelBC,
                          const llvm::Module *Program,
                          const char **DevAuxFuncs) {
  ValueToValueMapTy vvm;

  // Copy all the globals from lib to program.
  // It probably is faster to just copy them all, than to inspect
  // both program and lib to find which actually are used.
  DB_PRINT("cloning the global variables:\n");
  llvm::Module::const_global_iterator gi,ge;
  for (gi=Program->global_begin(), ge=Program->global_end(); gi != ge; gi++) {
    DB_PRINT(" %s\n", gi->getName().data());
    GlobalVariable *GV = new GlobalVariable(
      *ParallelBC, gi->getValueType(), gi->isConstant(),
      gi->getLinkage(), (Constant*)0, gi->getName(), (GlobalVariable*)0,
      gi->getThreadLocalMode(), gi->getType()->getAddressSpace());
    GV->copyAttributesFrom(&*gi);
    vvm[&*gi]=GV;
  }

  const StringRef KernelName(Name);
  copy_func_callgraph(KernelName, Program, ParallelBC, vvm);

  if (DevAuxFuncs) {
    const char **Func = DevAuxFuncs;
    while (*Func != nullptr) {
      copy_func_callgraph(*Func++, Program, ParallelBC, vvm);
    }
  }

  std::string Log;
  shared_copy(ParallelBC, Program, Log, vvm);

  if (pocl_get_bool_option("POCL_LLVM_ALWAYS_INLINE", 0)) {
    llvm::Module::iterator MI, ME;
    for (MI = ParallelBC->begin(), ME = ParallelBC->end(); MI != ME; ++MI) {
      Function *F = &*MI;
      if (F->isDeclaration())
          continue;
      // inline all except the kernel
      if (F->getName() != Name) {
          F->addFnAttr(Attribute::AlwaysInline);
      }
    }

    llvm::legacy::PassManager Passes;
    Passes.add(createAlwaysInlinerLegacyPass());
    Passes.run(*ParallelBC);
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

