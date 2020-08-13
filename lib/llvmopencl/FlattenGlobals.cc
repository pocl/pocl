// LLVM module pass to inline only required functions (those accessing
// per-workgroup variables) into the kernel.
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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "config.h"
#include "pocl.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

#include "Workgroup.h"

POP_COMPILER_DIAGS

using namespace llvm;

namespace {
class FlattenGlobals : public ModulePass {

public:
  static char ID;
  FlattenGlobals() : ModulePass(ID) {}

  virtual bool runOnModule(Module &M);
};
}

extern cl::opt<std::string> KernelName;

char FlattenGlobals::ID = 0;
static RegisterPass<FlattenGlobals>
    X("flatten-globals",
      "Kernel function flattening pass - flatten global vars' users only");

//#define DEBUG_FLATTEN

static const char *workgroup_variables[] = {"_local_id_x",
                                            "_local_id_y",
                                            "_local_id_z",
                                            "_local_size_x",
                                            "_local_size_y",
                                            "_local_size_z",
                                            "_work_dim",
                                            "_num_groups_x",
                                            "_num_groups_y",
                                            "_num_groups_z",
                                            "_group_id_x",
                                            "_group_id_y",
                                            "_group_id_z",
                                            "_global_offset_x",
                                            "_global_offset_y",
                                            "_global_offset_z",
                                            NULL};

bool FlattenGlobals::runOnModule(Module &M) {
  SmallPtrSet<Function *, 8> functions_to_inline;
  SmallVector<Value *, 8> pending;

  const char **s = workgroup_variables;
  while (*s != NULL) {
    GlobalVariable *gv = M.getGlobalVariable(*s);
    if (gv != NULL)
      pending.push_back(gv);

    ++s;
  }

  while (!pending.empty()) {
    Value *v = pending.back();
    pending.pop_back();

    for (Value::use_iterator i = v->use_begin(), e = v->use_end(); i != e;
         ++i) {
      llvm::User *user = i->getUser();
      if (Instruction *ci = dyn_cast<Instruction>(user)) {
        // Prevent infinite looping on recursive functions
        // (though OpenCL does not allow this?)
        Function *f = ci->getParent()->getParent();
        ;
        assert(
            (f != NULL) &&
            "Per-workgroup global variable used on function with no parent!");
        if (functions_to_inline.count(f))
          continue;

        // if it's an OpenCL kernel with OptNone attribute, assume we're debugging,
        // and don't inline the kernel into the workgroup launcher.
        // this makes it possible to debug kernel code with GDB.
        if (pocl::Workgroup::isKernelToProcess(*f) && f->hasFnAttribute(Attribute::OptimizeNone))
          continue;

        functions_to_inline.insert(f);
        pending.push_back(f);
      }
    }
  }

  for (SmallPtrSet<Function *, 8>::iterator i = functions_to_inline.begin(),
                                            e = functions_to_inline.end();
       i != e; ++i) {
    (*i)->removeFnAttr(Attribute::NoInline);
    (*i)->removeFnAttr(Attribute::OptimizeNone);
    (*i)->addFnAttr(Attribute::AlwaysInline);
  }

  StringRef barrier("_Z7barrierj");
  for (llvm::Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    llvm::Function *f = &*i;
    if (f->isDeclaration())
      continue;
    if (f->getName().equals(barrier)) {
      f->removeFnAttr(Attribute::NoInline);
      f->removeFnAttr(Attribute::OptimizeNone);
      f->addFnAttr(Attribute::AlwaysInline);
    }
  }

  return true;
}
