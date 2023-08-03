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

#include <iostream>
#include <string>
#include <set>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "config.h"
#include "pocl.h"
#include "pocl_llvm_api.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"

#include "Workgroup.h"

POP_COMPILER_DIAGS

using namespace llvm;

namespace {
  class Flatten : public ModulePass {

  public:
    static char ID;
    Flatten() : ModulePass(ID) {}

    virtual bool runOnModule(Module &M);
  };

}

char Flatten::ID = 0;
static RegisterPass<Flatten>
    X("flatten-inline-all",
      "Kernel function flattening pass - flatten everything");

//#define DEBUG_FLATTEN

bool
Flatten::runOnModule(Module &M)
{
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
#ifdef LLVM_OLDER_THAN_14_0
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

