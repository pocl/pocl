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
#include "pocl_cl.h"

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

extern cl::opt<std::string> KernelName;
extern cl_device_id currentPoclDevice;

char Flatten::ID = 0;
static RegisterPass<Flatten>
    X("flatten-inline-all",
      "Kernel function flattening pass - flatten everything");

//#define DEBUG_FLATTEN

bool
Flatten::runOnModule(Module &M)
{
  bool changed = false;

  std::set<std::string> AuxFuncs;
  if (const char **DevAuxFuncs = currentPoclDevice->device_aux_functions) {
    const char **Func = DevAuxFuncs;
    while (*Func != nullptr) {
      AuxFuncs.insert(*Func++);
    }
  }

  for (llvm::Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    llvm::Function *f = &*i;
    if (f->isDeclaration() || f->getName().startswith("__pocl_print") ||
        AuxFuncs.find(f->getName().str()) != AuxFuncs.end())
      continue;
    if (KernelName == f->getName() && pocl::Workgroup::isKernelToProcess(*f)) {
      AttributeSet Attrs;
      f->removeAttributes(AttributeList::FunctionIndex,
                          Attrs.addAttribute(M.getContext(),
                                             Attribute::AlwaysInline));

      f->addFnAttr(Attribute::NoInline);

      f->setLinkage(llvm::GlobalValue::ExternalLinkage);
      changed = true;
#ifdef DEBUG_FLATTEN
      std::cerr << "### NoInline for " << f->getName().str() << std::endl;
#endif
    } else {
      AttributeSet Attrs;
      f->removeAttributes(AttributeList::FunctionIndex,
                          Attrs.addAttribute(M.getContext(),
                                             Attribute::NoInline));
      f->addFnAttr(Attribute::AlwaysInline);

      f->setLinkage(llvm::GlobalValue::InternalLinkage);
      changed = true;
#ifdef DEBUG_FLATTEN
      std::cerr << "### AlwaysInline for " << f->getName().str() << std::endl;
#endif
    }
  }
  return changed;
}

