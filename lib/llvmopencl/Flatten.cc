// LLVM module pass that setups inline attributes so ALL called functions
// will be inlined into the kernel.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos
//               2012-2015 Pekka Jääskeläinen
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
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"

#include "Flatten.hh"
#include "LLVMUtils.h"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include <iostream>
#include <set>
#include <string>

#include "pocl_llvm_api.h"

//#define DEBUG_FLATTEN

#define PASS_NAME "flatten-inline-all"
#define PASS_CLASS pocl::FlattenAll
#define PASS_DESC "Kernel function flattening pass - flatten everything"

namespace pocl {

using namespace llvm;

static bool flattenAll(Module &M) {
  bool changed = false;

  std::string DevAuxFunctionsList;
  bool FoundMetadata =
      getModuleStringMetadata(M, "device_aux_functions", DevAuxFunctionsList);

  std::set<std::string> AuxFuncs;
  if (FoundMetadata && DevAuxFunctionsList.size() > 0) {
    size_t pos = 0;
    std::string token;
    while ((pos = DevAuxFunctionsList.find(";")) != std::string::npos) {
      token = DevAuxFunctionsList.substr(0, pos);
      AuxFuncs.insert(token);
      DevAuxFunctionsList.erase(0, pos + 1);
    }
  }

  for (llvm::Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    llvm::Function *F = &*i;
    if (F->isDeclaration() || F->getName().starts_with("__pocl_print") ||
        F->getName() == "__printf_alloc" ||
        F->getName() == "__printf_flush_buffer" ||
        AuxFuncs.find(F->getName().str()) != AuxFuncs.end())
      continue;

    AttributeSet Attrs;
    changed = true;
    decltype(Attribute::AlwaysInline) replaceThisAttr, replacementAttr;
    decltype(llvm::GlobalValue::ExternalLinkage) linkage;

    bool OnlyDynamicWIFuncCallsFound = false;
    bool IsStaticWIFuncCall = false;
    // Check if this is a WIF which is called only with constant
    // arguments. We want to leave the calls intact to avoid cluttering
    // the code. They will be expanded by
    // WorkitemHandler::handleWorkitemFunctions().
    // If there's even one caller that sets the dimidx dynamically,
    // we have to retain the function.
    Instruction::use_iterator UI = F->use_begin(), UE = F->use_end();
    for (; UI != UE; ++UI) {
      llvm::CallInst *Call = dyn_cast<llvm::CallInst>(UI->getUser());
      if (Call == nullptr)
        continue;
      IsStaticWIFuncCall = isCompilerExpandableWIFunctionCall(*Call);
      if (!IsStaticWIFuncCall)
        break;
    }

    OnlyDynamicWIFuncCallsFound = IsStaticWIFuncCall && UI == UE;

    if (pocl::isKernelToProcess(*F) || OnlyDynamicWIFuncCallsFound) {
      replaceThisAttr = Attribute::AlwaysInline;
      replacementAttr = Attribute::NoInline;
      linkage = llvm::GlobalValue::ExternalLinkage;
#ifdef DEBUG_FLATTEN
      std::cerr << "### NoInline for " << f->getName().str() << std::endl;
#endif
    } else {
      replaceThisAttr = Attribute::NoInline;
      replacementAttr = Attribute::AlwaysInline;
      linkage = llvm::GlobalValue::InternalLinkage;
#ifdef DEBUG_FLATTEN
      std::cerr << "### AlwaysInline for " << f->getName().str() << std::endl;
#endif
    }
    F->setAttributes(F->getAttributes()
                         .removeFnAttribute(M.getContext(), replaceThisAttr)
                         .addFnAttribute(F->getContext(), replacementAttr));
    if (F->hasFnAttribute(Attribute::AlwaysInline) &&
        F->hasFnAttribute(Attribute::OptimizeNone)) {
      F->removeFnAttr(Attribute::OptimizeNone);
    }

    F->setLinkage(linkage);
  }
  return changed;
}


llvm::PreservedAnalyses FlattenAll::run(llvm::Module &M,
                                        llvm::ModuleAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  return flattenAll(M) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_MPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
