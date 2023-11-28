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
    llvm::Function *f = &*i;
    if (f->isDeclaration() || f->getName().startswith("__pocl_print") ||
        AuxFuncs.find(f->getName().str()) != AuxFuncs.end())
      continue;

    AttributeSet Attrs;
    changed = true;
    decltype(Attribute::AlwaysInline) replaceThisAttr, replacementAttr;
    decltype(llvm::GlobalValue::ExternalLinkage) linkage;
    if (pocl::isKernelToProcess(*f)) {
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
#if LLVM_MAJOR < 14
    f->removeAttributes(AttributeList::FunctionIndex,
                        Attrs.addAttribute(M.getContext(), replaceThisAttr));
    f->addFnAttr(replacementAttr);
#else
    f->setAttributes(f->getAttributes()
                         .removeFnAttribute(M.getContext(), replaceThisAttr)
                         .addFnAttribute(f->getContext(), replacementAttr));
#endif

    f->setLinkage(linkage);
  }
  return changed;
}

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
char FlattenAll::ID = 0;

bool FlattenAll::runOnModule(Module &M) { return flattenAll(M); }

void FlattenAll::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addPreserved<WorkitemHandlerChooser>();
}

REGISTER_OLD_MPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#else

llvm::PreservedAnalyses FlattenAll::run(llvm::Module &M,
                                        llvm::ModuleAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  return flattenAll(M) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_MPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#endif

} // namespace pocl
