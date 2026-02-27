// Lightweight bitcode linker to replace llvm::Linker.
//
// Copyright (c) 2014 Kalle Raiskila
//               2016-2022 Pekka Jääskeläinen
//               2023-2024 Pekka Jääskeläinen / Intel Finland Oy
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
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
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
#include "pocl_spir.h"

#include "Barrier.h"
#include "EmitPrintf.hh"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "linker.h"

using namespace llvm;

// #include <cstdio>
// #define DB_PRINT(...) fprintf(stderr, "linker:" __VA_ARGS__)
#define DB_PRINT(...)

namespace pocl {

// The purpose is to lower SPIR-V "Target Extension" types when we're
// linking incoming bitcode from SPIR-V translator or Clang SPIRV target.
// These types in particular:
//    target("spirv.Image", ...)
//    target("spirv.Sampler")
// since we cannot do anything useful with them (like bitcasting),
// we copy the functions using them with a Remapper that maps
// arguments/values of these types to 'ptr addrspace(1)' type.
// This is necessary after recent change in SPIRV translator, see
// commit 2feb3e3eafc48553e4768d3b0118afb469365b2f in
// https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/3287

class PoclTargetExtTypeRemapper : public ValueMapTypeRemapper {
public:
  PoclTargetExtTypeRemapper() {}
  virtual ~PoclTargetExtTypeRemapper() {}

  virtual Type *remapType(Type *SrcTy) {
    if (SrcTy->isTargetExtTy()) {
      PointerType *PT =
          PointerType::get(SrcTy->getContext(), SPIR_ADDRESS_SPACE_GLOBAL);
      if (SrcTy->getTargetExtName() == "spirv.Image" ||
          SrcTy->getTargetExtName() == "spirv.Sampler")
        return PT;
    }
    return SrcTy;
  }
};

using ValueToSizeTMapTy = ValueMap<const Value *, size_t>;



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

static bool modIsNvptx(llvm::Module *Mod) {
#if LLVM_MAJOR > 20
  return (Mod->getTargetTriple().getArch() == llvm::Triple::ArchType::nvptx ||
          Mod->getTargetTriple().getArch() == llvm::Triple::ArchType::nvptx64);
#else
  return Mod->getTargetTriple().compare(0, 5, "nvptx") == 0;
#endif
}

// fix mismatches between calling conv. This should not happen,
// but sometimes can, esp with SPIR(-V) input
static void fixCallingConv(llvm::Module *Mod, std::string &Log) {
#if LLVM_MAJOR > 18
  if (modIsNvptx(Mod)) {
    for (llvm::Module::iterator MI = Mod->begin(); MI != Mod->end(); ++MI) {
      llvm::Function *F = &*MI;
      if (F->isDeclaration())
        continue;

      if (!F->hasName())
        continue;
      if (F->getName().starts_with("printf"))
        continue;
      if (F->getName().starts_with("llvm."))
        continue;

      if (isKernelToProcess(*F))
        F->setCallingConv(llvm::CallingConv::PTX_Kernel);
      else
        F->setCallingConv(llvm::CallingConv::PTX_Device);
    }
  }
#endif

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

        if ((Callee == nullptr) || Callee->isDeclaration())
          continue;
        // all functions should have name at this point
        assert(Callee->hasName());
        if (Callee->getName().starts_with("llvm."))
          continue;

        // Loosen the CC to the default one. It should be always the
        // preferred one to SPIR_FUNC at this stage.
        // note: this is necessary for working SPIR-V on Apple ARM64 target
        // note: must not change the SPIR_KERNEL CC because of MinLegalVecSize
        if (Callee->getCallingConv() == llvm::CallingConv::SPIR_FUNC ||
            CallInstr->getCallingConv() == llvm::CallingConv::SPIR_FUNC) {
          Callee->setCallingConv(llvm::CallingConv::C);
          CallInstr->setCallingConv(llvm::CallingConv::C);
        }

        // special handling for DebugInfo of SPIR-V from llvm-spirv translator
        if ((Callee->getName().starts_with("__anonymous_function")) &&
            (F->getCallingConv() == llvm::CallingConv::SPIR_KERNEL)) {
          // This is an SPIR-V entry point wrapper function: SPIR-V
          // translator generates these because OpenCL allows calling
          // kernels from kernels like they were device side functions
          // whereas SPIR-V entry points cannot call other entry points.
          std::string CalleeName("_spirv_wrapped_");
          CalleeName += F->getName().str();
          Callee->setName(CalleeName);

          // The SPIR-V translator loses the kernel's original DISubprogram
          // info, leaving it to the wrapper, thus even after inlining the
          // function to the kernel we do not get any debug info (LLVM checks
          // for DISubprogram for each function it generates the debug info
          // for). Just reuse the DISubprogram in the kernel here in that case.
          if (Callee->getSubprogram() != nullptr &&
            F->getSubprogram() == nullptr &&
            Callee->getSubprogram()->getName() == F->getName()) {
            F->setSubprogram(pocl::mimicDISubprogram(
              Callee->getSubprogram(), CalleeName, nullptr));
            CallInstr->setDebugLoc(llvm::DILocation::get(
              Callee->getContext(), Callee->getSubprogram()->getLine(), 0,
              F->getSubprogram(), nullptr, true));
          }
        }

        if (CallInstr->getCallingConv() != Callee->getCallingConv()) {
          std::string CalleeName, CallerName;
          assert(F->hasName());
          CallerName = F->getName().str();
          assert(Callee->hasName());
          CalleeName = Callee->getName().str();

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

using SmallFunctionSet = llvm::SmallSet<llvm::Function *, 8>;

// Find all functions in the calltree of F, including declarations.
// Returns the list of Functions called in CalledFuncList,
// in depth-first order (to make cloning simpler);
// returns pointer to recursive function on error, nullptr on success
static llvm::Function *
find_called_functions(llvm::Function *F,
                      llvm::SmallVector<llvm::Function *> &CalledFuncList,
                      SmallFunctionSet &CallStack) {
  if (F->isDeclaration()) {
    DB_PRINT("it's a declaration, return\n");
    return nullptr;
  }

  CallStack.insert(F);

  assert(F->hasName());
  std::string FName = F->getName().str();

  for (auto &I : instructions(F)) {

    CallInst *CI = dyn_cast<CallInst>(&I);
    if (CI == nullptr)
      continue;

    llvm::Function *Callee = CI->getCalledFunction();
    // this happens with e.g. inline asm calls
    if (Callee == nullptr) {
      DB_PRINT("search: %s callee NULL\n", FName.c_str());
      continue;
    }

    assert(Callee->hasName());
    std::string CName = Callee->getName().str();

    if (CallStack.contains(Callee)) {
      DB_PRINT("Recursion detected: %s\n", CName.c_str());
      return Callee;
    }
    DB_PRINT("Function %s calls %s\n", FName.c_str(), CName.c_str());

    auto It = std::find(CalledFuncList.begin(), CalledFuncList.end(), Callee);
    if (It != CalledFuncList.end()) {
      DB_PRINT("already contained in CalledList: %s\n", CName.c_str());
      continue;
    } else {
      DB_PRINT("function %s not seen before, recursing into it\n",
               CName.c_str());
      if (auto *R = find_called_functions(Callee, CalledFuncList, CallStack))
        return R;
      DB_PRINT("inserting %s into CalledList\n", CName.c_str());
      CalledFuncList.push_back(Callee);
    }
  }

  CallStack.erase(F);

  return nullptr;
}

// Copies one function from one module to another
// does not inspect it for callgraphs
static void CopyFunc(const llvm::StringRef Name, const llvm::Module *From,
                     llvm::Module *To, ValueToValueMapTy &VVMap) {

  llvm::Function *SrcFunc = From->getFunction(Name);
  // TODO: is this the linker error "not found", and not an assert?
  assert(SrcFunc && "Did not find function to copy in kernel library");
  llvm::Function *DstFunc = To->getFunction(Name);

  if (DstFunc == NULL) {
    DB_PRINT("   %s not found in destination module, creating\n", Name.data());
    DstFunc = Function::Create(cast<FunctionType>(SrcFunc->getValueType()),
                               SrcFunc->getLinkage(), SrcFunc->getName(), To);
    DstFunc->copyAttributesFrom(SrcFunc);
  } else if (DstFunc->size() > 0) {
    // We have already encountered and copied this function.
    return;
  }
  VVMap[SrcFunc] = DstFunc;

  Function::arg_iterator DstArgI = DstFunc->arg_begin();
  for (Function::const_arg_iterator SrcArgI = SrcFunc->arg_begin(),
                                    SrcArgE = SrcFunc->arg_end();
       SrcArgI != SrcArgE; ++SrcArgI) {
    DstArgI->setName(SrcArgI->getName());
    VVMap[&*SrcArgI] = &*DstArgI;
    ++DstArgI;
  }
  if (!SrcFunc->isDeclaration()) {
    SmallVector<ReturnInst *, 8> RI; // Ignore returns cloned.
    DB_PRINT("  cloning %s\n", Name.data());

    llvm::ClonedCodeInfo CodeInfo;
    CloneFunctionIntoAbs(DstFunc, SrcFunc, VVMap, RI,
                         false,     // same module
                         "",        // suffix
                         &CodeInfo, // codeInfo
                         nullptr,   // type remapper
                         nullptr);  // materializer
  } else {
    DB_PRINT("  found %s, but its a declaration, do nothing\n", Name.data());
  }
}

/* Copy function F and all the functions in its call graph
 * that are defined in 'from', into 'to', adding the mappings to
 * 'vvm'.
 */
static int CopyFuncCallgraph(const llvm::StringRef FuncName,
                             const llvm::Module *From, llvm::Module *To,
                             ValueToValueMapTy &VVMap) {
  llvm::Function *RootFunc = From->getFunction(FuncName);
  if (RootFunc == NULL) {
    return -1;
  }

  SmallFunctionSet CallStack;
  llvm::SmallVector<llvm::Function *> Callees;
  // error if recursion occurs
  if (find_called_functions(RootFunc, Callees, CallStack))
    return -2;

  DB_PRINT("copying function %s with callgraph\n", RootFunc->getName().data());

  // First copy the callees of func, then the function itself.
  for (auto F : Callees) {
    CopyFunc(F->getName(), From, To, VVMap);
  }
  CopyFunc(FuncName, From, To, VVMap);
  assert(To->getFunction(FuncName) != nullptr);

  return 0;
}

/* Estimates the size of stack frame used by a function and all functions
 * it calls. Since OpenCL forbids recursion, we can error out if it happens.
 * The estimate should be the worst-case, since the code at this point
 * is not optimized at all, and the optimization should move some
 * variables to registers.
 */
static size_t estimateFunctionStackSize(llvm::Function *Func,
                                        const llvm::Module *Mod,
                                        std::vector<Function *> &CallChain,
                                        ValueToSizeTMapTy &StackSizesMap) {
  if (Func == nullptr)
    return 0;
  DB_PRINT("estimating function stack size of %s\n", Func->getName().data());
  size_t TotalSelfSize = 0;
  size_t TotalSubSize = 0;
  CallChain.push_back(Func);

  for (auto FIter = Func->begin(); FIter != Func->end(); FIter++) {
    for (auto BIter = FIter->begin(); BIter != FIter->end(); BIter++) {

      AllocaInst *AI = dyn_cast<AllocaInst>(BIter);
      if (AI) {
        auto AllocatedType = AI->getAllocatedType();
        const llvm::DataLayout &DL = Mod->getDataLayout();
        if (auto AllocaSize = AI->getAllocationSize(DL))
          TotalSelfSize += AllocaSize->getKnownMinValue();
        continue;
      }

      CallInst *CI = dyn_cast<CallInst>(BIter);
      Function *Callee;
      if (CI && (Callee = CI->getCalledFunction())) {
        if (std::find(CallChain.begin(), CallChain.end(), Callee) !=
            CallChain.end()) {
          DB_PRINT("error: encountered recursion!\n");
          CallChain.pop_back();
          return 0;
        }

        auto It = StackSizesMap.find(Callee);
        if (It != StackSizesMap.end()) {
          TotalSubSize = std::max(It->second, TotalSubSize);
        } else {
          size_t SubSize =
              estimateFunctionStackSize(Callee, Mod, CallChain, StackSizesMap);
          StackSizesMap.insert(std::make_pair(Callee, SubSize));
          TotalSubSize = std::max(SubSize, TotalSubSize);
        }
      }
    }
  }

  size_t TotalSize = TotalSelfSize + TotalSubSize;
  CallChain.pop_back();
  StackSizesMap.insert(std::make_pair(Func, TotalSize));
  return TotalSize;
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

// Special handling for the AS cast operators: They do not use
// type mangling, which complicates the implementation which
// can be called with pointer arguments with different AS based
// on compilation input (OpenCL C source vs SPIR-V).
//
// Here, we replace the calls directly with address
// space casts. This works for uniform address space targets (CPUs),
// while the disjoint AS targets can override the implementation
// as needed.
static bool convertAddrSpaceOperator(llvm::Function *Func, std::string &Log) {
  if (Func == nullptr) {
    return true;
  }

  if (!Func->isDeclaration()) {
    // If the builtin library provides an implementation, don't touch it
    return true;
  }

  for (auto *U : Func->users()) {
    if (llvm::CallInst *Call = dyn_cast<llvm::CallInst>(U)) {
      PointerType *ArgPT =
          dyn_cast<PointerType>(Call->getArgOperand(0)->getType());
      PointerType *RetPT =
          dyn_cast<PointerType>(Call->getFunctionType()->getReturnType());
      if (ArgPT == nullptr || RetPT == nullptr) {
        Log.append("Invalid use of operator __to_{local,global,private}");
        return false;
      }
      if (ArgPT->getAddressSpace() == RetPT->getAddressSpace()) {
        Value *V = Call->getArgOperand(0);
        Call->replaceAllUsesWith(V);
        Call->eraseFromParent();
      } else {
        llvm::AddrSpaceCastInst *AsCast = new llvm::AddrSpaceCastInst(
            Call->getArgOperand(0), Call->getFunctionType()->getReturnType(),
            Func->getName() + ".as_cast",
#if LLVM_MAJOR < 20
            Call);
#else
            Call->getIterator());
#endif
        Call->replaceAllUsesWith(AsCast);
        Call->eraseFromParent();
      }
    }
  }

  Func->eraseFromParent();
  return true;
}

// Find all functions in the calltree of F, including declarations,
// using postorder traversal, and puts them into CalledFuncList
// @return: bool true on success, false on recursion detected
static bool
get_postorder_callstack(llvm::Function &F,
                        llvm::SmallVector<llvm::Function *> &CalledFuncList,
                        SmallFunctionSet &CallStack) {
  // if the function is already in the CalledFuncList because
  // it was called by another function, return immediately
  auto It = std::find(CalledFuncList.begin(), CalledFuncList.end(), &F);
  if (It != CalledFuncList.end()) {
    return true;
  }

  CallStack.insert(&F);
  assert(F.hasName());
  std::string FName = F.getName().str();

  for (auto &I : instructions(F)) {

    CallInst *CI = dyn_cast<CallInst>(&I);
    if (CI == nullptr)
      continue;

    llvm::Function *Callee = CI->getCalledFunction();
    // this happens with e.g. inline asm calls
    if (Callee == nullptr) {
      DB_PRINT("search: %s callee NULL\n", FName.c_str());
      continue;
    }

    assert(Callee->hasName());
    std::string CName = Callee->getName().str();

    if (CallStack.contains(Callee)) {
      DB_PRINT("Recursion detected: %s\n", CName.c_str());
      return false;
    }
    DB_PRINT("Function %s calls %s\n", FName.c_str(), CName.c_str());

    auto It = std::find(CalledFuncList.begin(), CalledFuncList.end(), Callee);
    if (It != CalledFuncList.end()) {
      DB_PRINT("already contained in CalledList: %s\n", CName.c_str());
      continue;
    } else {
      DB_PRINT("function %s not seen before, recursing into it\n",
               CName.c_str());
      if (!get_postorder_callstack(*Callee, CalledFuncList, CallStack))
        return false;
      DB_PRINT("inserting %s into CalledList\n", CName.c_str());
      CalledFuncList.push_back(Callee);
    }
  }

  CalledFuncList.push_back(&F);
  CallStack.erase(&F);
  return true;
}

// Find all functions in module M, return them for postorder traversal
static bool
sort_functions_postorder(llvm::Module &M,
                         llvm::SmallVector<llvm::Function *> &CalledFuncList) {

  for (auto &F : M.functions()) {
    if (F.isDeclaration()) {
      CalledFuncList.push_back(&F);
    }
  }

  SmallFunctionSet CallStack;
  for (auto &F : M.functions()) {
    if (F.isDeclaration())
      continue;
    if (!get_postorder_callstack(F, CalledFuncList, CallStack))
      return false;
  }
  return true;
}

// Copies a function with TargetExt types remapped to 'ptr' type
// @returs true if copy was performed, false if skipped because no remapping
static bool CopyFuncNoTargetExt(llvm::Function *Src, llvm::Module &M,
                                ValueToValueMapTy &VVMap) {
  PoclTargetExtTypeRemapper TR;

  FunctionType *SrcFT = cast<FunctionType>(Src->getValueType());
  SmallVector<Type *> DstArgT;
  for (auto SrcA : SrcFT->params()) {
    DstArgT.push_back(TR.remapType(SrcA));
  }
  Type *SrcRet = SrcFT->getReturnType();
  Type *DstRet = TR.remapType(SrcRet);
  assert(Src->hasName());
  FunctionType *DstFT = FunctionType::get(DstRet, DstArgT, SrcFT->isVarArg());
  if (DstFT == SrcFT) {
    DB_PRINT("CopyFuncNoTargetExt: function %s signature identical, "
             "skipping remap copy\n",
             Src->getName().c_str());
    return false;
  } else {
    DB_PRINT("CopyFuncNoTargetExt: function %s signature DIFFERENT, "
             "performing remap copy\n",
             Src->getName().c_str());
  }

  StringRef NewName = Src->getName();
  Src->setName(NewName + ".beforeTgtExt");
  llvm::Function *Dst = Function::Create(DstFT, Src->getLinkage(), NewName, M);
  Dst->copyAttributesFrom(Src);

  VVMap[Src] = Dst;

  Function::arg_iterator DstArgI = Dst->arg_begin();
  for (Function::const_arg_iterator SrcArgI = Src->arg_begin(),
                                    SrcArgE = Src->arg_end();
       SrcArgI != SrcArgE; ++SrcArgI) {
    DstArgI->setName(SrcArgI->getName());
    VVMap[&*SrcArgI] = &*DstArgI;
    ++DstArgI;
  }
  if (!Src->isDeclaration()) {
    SmallVector<ReturnInst *, 8> RI; // Ignore returns cloned.
    DB_PRINT("  cloning %s\n", Name.data());

    llvm::ClonedCodeInfo CodeInfo;
    CloneFunctionIntoAbs(Dst, Src, VVMap, RI,
                         true,        // same module
                         ".noTgtExt", // suffix
                         &CodeInfo,   // codeInfo
                         &TR,         // type remapper
                         nullptr);    // materializer
  } else {
    DB_PRINT("  found %s, but its a declaration, do nothing\n", Name.data());
  }
  return true;
}

static bool convertSpirvTargetExtTypes(llvm::Module &Mod, std::string &Log) {
  PoclTargetExtTypeRemapper TypeMapper;
  ValueToValueMapTy VVM;
  for (auto &F : Mod.functions()) {
    if (!F.hasName())
      F.setName("__anonymous_function");
  }

  SmallVector<Function *> CalledFuncList;
  if (!sort_functions_postorder(Mod, CalledFuncList)) {
    Log.append("convertSpirvTargetExtTypes: Failed to sort fuctions b/c of "
               "recursion\n");
    return false;
  }

  ValueToValueMapTy VVMap;
  SmallVector<Function *> RemoveFuncList;
  for (auto *F : CalledFuncList) {
    if (CopyFuncNoTargetExt(F, Mod, VVMap))
      // removal needs to be in reverse order
      RemoveFuncList.insert(RemoveFuncList.begin(), F);
  }

  for (auto *F : RemoveFuncList) {
    if (F->use_empty())
      F->eraseFromParent();
  }
  // std::cerr << "*******************************************************";
  // std::cerr << "  AFTER convertSpirvTargetExtTypes: *******************";
  // Mod.dump();
  // std::cerr << "*******************************************************";
  // std::cerr << "*******************************************************";
  return true;
}

/* Replace printf calls with generated bitcode that stores the format-string
 * and the arguments to a printf buffer. The replacement bitcode generated
 * by emitPrintfCall is:
 *
 * 1) get the format string length and argument sizes
 * 2) call pocl_printf_alloc_stub to allocate storage from device's printf buffer
 *    to allocate the size from 0)
 * 3) store the arguments using Store or Memcpy instructions
 * 4) optionally call pocl_flush_printf_buffer (if the device supports
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
      Lib->getFunction("pocl_flush_printf_buffer") != nullptr;

  PrintfCallFlags PFlags;
  PFlags.PrintfBufferAS = ClDev->global_as_id;
  PFlags.IsBuffered = true;
  PFlags.DontAlign = true;
  PFlags.FlushBuffer = DeviceSupportsImmediateFlush;
  PFlags.Pointers32Bit = ClDev->address_bits == 32;

  if (ClDev->type == CL_DEVICE_TYPE_CPU) {
    // for CPU, store constant format strings as simply a pointer
    // the original AMDGPU emitPrintfCall uses MD5 but that is
    // unnecessary complication for CPU devices
    PFlags.StorePtrInsteadOfMD5 = true;
    PFlags.AlwaysStoreFmtPtr = false;
  } else {
    // for non-CPU devices, always store the format string in the buffer,
    // even if it's a constant format string, since we cannot use pointers
    PFlags.StorePtrInsteadOfMD5 = false;
    PFlags.AlwaysStoreFmtPtr = true;
  }

  // C promotion of float -> double only if device supports double
  if (ClDev->double_fp_config == 0) {
    PFlags.ArgPromotionFP64 = false;
  }
  // big endian support is not implemented at all currently
  if (!ClDev->endian_little) {
    PFlags.IsBigEndian = true;
  }

  Function *CalledPrintf = Program->getFunction("printf");
  if (CalledPrintf) {

    Function *PrintfAlloc = Lib->getFunction("pocl_printf_alloc_stub");
    assert(PrintfAlloc != nullptr);
    CopyFuncCallgraph("pocl_printf_alloc_stub", Lib, Program, vvm);
    CopyFuncCallgraph("pocl_printf_alloc", Lib, Program, vvm);
    if (DeviceSupportsImmediateFlush)
      CopyFuncCallgraph("pocl_flush_printf_buffer", Lib, Program, vvm);

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
  if (IntrinRepl == nullptr)
    return;
  std::map<Function *, Function *> EraseMap;
  llvm::Module::iterator FI, FE;
  StringRef LlvmIntrins("llvm.");
  for (FI = Program->begin(), FE = Program->end(); FI != FE; FI++) {
    if (FI->isDeclaration()) {
      if (FI->hasName() && FI->getName().starts_with(LlvmIntrins)) {
        const char *ReplacementName =
            IntrinRepl(FI->getName().data(), FI->getName().size());
        if (ReplacementName) {
          CopyFuncCallgraph(ReplacementName, Lib, Program, vvm);
          Function *Repl = Program->getFunction(ReplacementName);
          assert(Repl);
          EraseMap[&*FI] = Repl;
          continue;
        }
      }
    }
  }

  for (auto It : EraseMap) {
    llvm::Function *Intrin = It.first;
    llvm::Function *Repl = It.second;
    Intrin->replaceAllUsesWith(Repl);
    Intrin->eraseFromParent();
  }
}
}

static void convertPoclTrap(llvm::Module *Program, cl_device_id ClDev) {
  Function *PoclTrapFn = Program->getFunction("__pocl_trap");
  if (!PoclTrapFn || PoclTrapFn->use_empty())
    return;

  // Ensure proper attributes regardless of what the frontend provided.
  // These are needed by optimization passes that run before ConvertPoclExit.
  PoclTrapFn->addFnAttr(Attribute::NoReturn);
  PoclTrapFn->addFnAttr(Attribute::Cold);
  PoclTrapFn->addFnAttr(Attribute::NoUnwind);

  // CPU: leave for the ConvertPoclExit pass (flag-store + return)
  if (!ClDev->spmd)
    return;

  Function *TrapIntrinsic =
      Intrinsic::getOrInsertDeclaration(Program, Intrinsic::trap);

  SmallVector<CallInst *, 4> TrapCalls;
  for (User *U : PoclTrapFn->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      TrapCalls.push_back(CI);

  for (auto *CI : TrapCalls) {
    IRBuilder<> Builder(CI);
    Builder.CreateCall(TrapIntrinsic);
    CI->eraseFromParent();
  }

  if (PoclTrapFn->use_empty())
    PoclTrapFn->eraseFromParent();
}

static void convertPoclExit(llvm::Module *Program, cl_device_id ClDev) {
  Function *ExitFn = Program->getFunction("__pocl_exit");
  if (!ExitFn || ExitFn->use_empty())
    return;

  // Ensure proper attributes regardless of what the frontend provided.
  // These are needed by optimization passes that run before ConvertPoclExit.
  ExitFn->addFnAttr(Attribute::NoReturn);
  ExitFn->addFnAttr(Attribute::Cold);
  ExitFn->addFnAttr(Attribute::NoUnwind);

  // CPU: leave for the ConvertPoclExit pass (runs before UTR)
  if (!ClDev->spmd)
    return;

  // Collect calls before modifying
  SmallVector<CallInst *, 4> Calls;
  for (User *U : ExitFn->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      Calls.push_back(CI);

  if (pocl::modIsNvptx(Program)) {
    // CUDA/PTX: inline asm "exit;"
    InlineAsm *ExitAsm = InlineAsm::get(
        FunctionType::get(Type::getVoidTy(Program->getContext()), false),
        "exit;", "", /*hasSideEffects=*/true);
    for (auto *CI : Calls) {
      IRBuilder<> Builder(CI);
      Builder.CreateCall(ExitAsm);
      CI->eraseFromParent();
    }
  } else {
    // Other GPU: fallback to llvm.trap
    Function *TrapIntrinsic =
        Intrinsic::getOrInsertDeclaration(Program, Intrinsic::trap);
    for (auto *CI : Calls) {
      IRBuilder<> Builder(CI);
      Builder.CreateCall(TrapIntrinsic);
      CI->eraseFromParent();
    }
  }

  if (ExitFn->use_empty())
    ExitFn->eraseFromParent();
}

using namespace pocl;

int link(llvm::Module *Program, const llvm::Module *Lib, std::string &Log,
         cl_device_id ClDev, bool StripAllDebugInfo) {

  assert(Program);
  assert(Lib);
  ValueToValueMapTy vvm;
  llvm::StringSet<> DeclaredFunctions;

  pocl::removeClangGeneratedKernelStubs(Program);

  if (!pocl::convertSpirvTargetExtTypes(*Program, Log))
    return -1;

  // Include auxiliary functions required by the device at hand.
  if (ClDev->device_aux_functions) {
    const char **Func = ClDev->device_aux_functions;
    while (*Func != nullptr) {
      DeclaredFunctions.insert(*Func++);
    }
  }

  llvm::Module::iterator FI, FE;
  // assign names to all functions
  for (FI = Program->begin(), FE = Program->end(); FI != FE; FI++) {
    // anonymous functions have no name, which breaks the algorithm later
    // when it searches for undefined functions in the kernel library.
    // assign a name here, this should be made unique by setName()
    if (!FI->hasName()) {
      FI->setName("__anonymous_function");
    }
  }

  // Inspect the program, find undefined functions
  for (FI = Program->begin(), FE = Program->end(); FI != FE; FI++) {
    std::string FName = FI->getName().str();
    if (FI->isDeclaration()) {
      DB_PRINT("Pre-link: %s is not defined\n", FName.c_str());
      DeclaredFunctions.insert(FI->getName());
      continue;
    }
    DB_PRINT("Function '%s' is defined, checking which funcs it calls\n",
             FName.c_str());

    SmallFunctionSet CallStack;
    llvm::SmallVector<llvm::Function *> Callees;
    if (auto *R = find_called_functions(&*FI, Callees, CallStack)) {
      Log.append("Recursion detected in function: '");
      std::string PrettyName = tryDemangleWithoutAddressSpaces(FName);
      Log.append(PrettyName.c_str());
      Log.append("'\n");
      Log.append("-> Infringing function: '");
      PrettyName = tryDemangleWithoutAddressSpaces(R->getName().str());
      Log.append(PrettyName.c_str());
      Log.append("'\n");
      return -1;
    }
    for (auto F : Callees) {
      DB_PRINT("Adding function '%s' to list of called funcs\n", FName.c_str());
      DeclaredFunctions.insert(F->getName());
    }
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
  for (auto &DeclIter : DeclaredFunctions) {
    llvm::StringRef FName = DeclIter.getKey();
    int Res = CopyFuncCallgraph(FName, Lib, Program, vvm);
    if (Res == -2) {
      Log.append("Recursion detected while copying function: '");
      Log.append(FName.str());
      Log.append("'\n");
      return -1;
    }
  }

  convertAddrSpaceOperator(Program->getFunction("__to_local"), Log);
  convertAddrSpaceOperator(Program->getFunction("__to_private"), Log);
  convertAddrSpaceOperator(Program->getFunction("__to_global"), Log);
  convertPoclTrap(Program, ClDev);
  convertPoclExit(Program, ClDev);

  // check all function declarations in the program
  // *after* we have linked functions from the library
  DeclaredFunctions.clear();
  for (auto &F : *Program) {
    if (F.isDeclaration()) {
      DB_PRINT("Post-link: %s is not defined\n", F.getName().data());
      DeclaredFunctions.insert(F.getName());
      continue;
    }
  }

  bool FoundAllUndefined = true;
  // this one is a handled with a special pocl LLVM pass
  StringRef pocl_sampler_handler("__translate_sampler_initializer");

  if (!modIsNvptx(Program)) {
    for (auto &DeclIter : DeclaredFunctions) {
      llvm::StringRef FName = DeclIter.getKey();
      Function *F = Program->getFunction(FName);

      if ((F == NULL) ||
          (F->isDeclaration() &&
           // A target might want to expose the C99 printf in
           // case not supporting the OpenCL 1.2 printf.
           F->getName() != "printf" && F->getName() != pocl_sampler_handler &&
           !F->getName().starts_with("llvm.") &&
           F->getName() != BARRIER_FUNCTION_NAME &&
           F->getName() != "__pocl_local_mem_alloca" &&
           F->getName() != "__pocl_work_group_alloca" &&
           F->getName() != "__pocl_trap" &&
           F->getName() != "__pocl_exit")) {
        Log.append("Cannot find symbol ");
        Log.append(FName.str());
        Log.append(" in kernel library\n");
        FoundAllUndefined = false;
      }
    }
  }

  if (!FoundAllUndefined)
    return -1;

  shared_copy(Program, Lib, Log, vvm);

  if (StripAllDebugInfo)
    llvm::StripDebugInfo(*Program);
  else
    removeDuplicateDbgInfo(Program);

  fixCallingConv(Program, Log);

  if (ClDev->device_side_printf)
    handleDeviceSidePrintf(Program, Lib, Log, vvm, ClDev);

  replaceIntrinsics(Program, Lib, vvm, ClDev);

  // If we prefer to use compiler expansion of ID functions instead of the
  // software-defined ones (with switch...cases for dim), replace the functions
  // with declarations so they act as markers for the kernel compiler.
  // This can be done only if the dim is static.
  if (!ClDev->spmd) {
    for (auto &F : *Program) {
      if (!isWorkitemFunctionWithOnlyCompilerExpandableCalls(F))
        continue;
      F.deleteBody();
    }
  }

  std::vector<Function *> CallChain;
  ValueToSizeTMapTy FuncStackSizeMap;
  for (auto &F : *Program) {
    if (F.isDeclaration())
      continue;
    if (!F.hasName()) {
      F.setName("__anonymous_function");
    }
    if (isKernelToProcess(F)) {
      size_t EstStackSize =
          estimateFunctionStackSize(&F, Program, CallChain, FuncStackSizeMap);
      DB_PRINT("Kernel %s Estimated stack size: %zu \n", F.getName().data(),
               EstStackSize);
      if (EstStackSize > 0) {
        std::string MetadataKey = F.getName().str();
        MetadataKey.append(".meta.est.stack.size");
        setModuleIntMetadata(Program, MetadataKey.c_str(), EstStackSize);
      }
    }
  }

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

  if (DevAuxFuncs) {
    const char **Func = DevAuxFuncs;
    while (*Func != nullptr) {
      CopyFuncCallgraph(*Func++, Program, ParallelBC, vvm);
    }
  }

  const StringRef KernelName(Name);
  if (CopyFuncCallgraph(KernelName, Program, ParallelBC, vvm)) {
    POCL_MSG_ERR("Failed to copy kernel %s from bitcode\n", Name);
    return -1;
  }

#ifdef CPU_USE_LLD_LINK_WIN32
  // create a global constant "int32 _fltused"
  // this is necessary if we're not linking against C runtime on Windows
  auto TT = ParallelBC->getTargetTriple();
  if ((TT.rfind("x86", 0) == 0) && (TT.find("windows") != std::string::npos)) {
    IntegerType *Int32Ty = IntegerType::getInt32Ty(ParallelBC->getContext());
    ConstantInt *Initializer = ConstantInt::get(Int32Ty, 0);
    GlobalVariable *FltUsed = new GlobalVariable(
        /*Module=*/*ParallelBC,
        /*Type=*/Int32Ty,
        /*isConstant=*/true,
        /*Linkage=*/GlobalValue::CommonLinkage, // GlobalValue::ExternalLinkage
        /*Initializer=*/Initializer,
        /*Name=*/"_fltused");
  }
#endif

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
            POCL_MSG_PRINT_LLVM("Copying non-kernel function %s\n",
                                FName.c_str());
          CopyFuncCallgraph(F->getName(), ProgramBC, OutputBC, VVM);
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

/* vim: set expandtab ts=2 : */

