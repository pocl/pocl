// FlattenBarrierSubs, a pass to force inlining of non-kernel functions
// with barrier calls
//
// Copyright (c) 2018 Michal Babej / TUT
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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include <llvm/IR/Instructions.h>

#include "Barrier.h"
#include "FlattenBarrierSubs.hh"
#include "LLVMUtils.h"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include "pocl_llvm_api.h"

#include <iostream>
#include <string>

//#define DEBUG_FLATTEN_SUBS

#define PASS_NAME "flatten-barrier-subs"
#define PASS_CLASS pocl::FlattenBarrierSubs
#define PASS_DESC "Flatten subroutines with barriers and/or local arguments"

namespace pocl {

using namespace llvm;

static bool recursivelyInlineBarrierUsers(Function *F, bool ChangeInlineFlag) {

  bool BarrierIsCalled = false;

#ifdef DEBUG_FLATTEN_SUBS
  std::cerr << "### FlattenBarrierSubs: SCANNING " << F->getName().str()
            << std::endl;
#endif

  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {
      Instruction *Instr = dyn_cast<Instruction>(BI);
      if (!llvm::isa<CallInst>(Instr))
        continue;
      CallInst *CallInstr = dyn_cast<CallInst>(Instr);
      Function *Callee = CallInstr->getCalledFunction();

      if ((Callee == nullptr) || Callee->getName().starts_with("llvm."))
        continue;

      if (llvm::isa<pocl::Barrier>(CallInstr))
        BarrierIsCalled = true;
      // we cannot break the loop here, since we want to inline all functions
      // that could call a barrier, not just the first one
      else if (recursivelyInlineBarrierUsers(Callee, true))
        BarrierIsCalled = true;
    }
  }

  if (ChangeInlineFlag & BarrierIsCalled) {
    F->removeFnAttr(Attribute::NoInline);
    F->removeFnAttr(Attribute::OptimizeNone);
    F->addFnAttr(Attribute::AlwaysInline);
    F->setLinkage(llvm::GlobalValue::InternalLinkage);
#ifdef DEBUG_FLATTEN_SUBS
    std::cerr << "### FlattenBarrierSubs: AlwaysInline ENABLED on "
              << F->getName().str() << std::endl;
#endif
  }
  return BarrierIsCalled;
}

static bool flattenBarrierSubs(Module &M) {
  bool Changed = false;

  std::string KernelName;
  getModuleStringMetadata(M, "KernelName", KernelName);

  for (llvm::Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    llvm::Function *F = &*I;
    if (F->isDeclaration())
      continue;

    if (KernelName == F->getName().str() || isKernelToProcess(*F)) {

#ifdef DEBUG_FLATTEN_SUBS
      std::cerr << "### FlattenBarrierSubs Pass running on " << KernelName
                << std::endl;
#endif
      // we don't want to set alwaysInline on a Kernel, only its subroutines.
      Changed = recursivelyInlineBarrierUsers(F, false);
    }
  }
  return Changed;
}


llvm::PreservedAnalyses
FlattenBarrierSubs::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  return flattenBarrierSubs(M) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_MPASS(PASS_NAME, PASS_CLASS, PASS_DESC);


} // namespace pocl
