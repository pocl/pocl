// Header for LLVMUtils, useful common LLVM-related functionality.
//
// Copyright (c) 2013-2019 Pekka Jääskeläinen
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

#ifndef _POCL_LLVM_UTILS_H
#define _POCL_LLVM_UTILS_H

#include <map>
#include <string>

#include "pocl.h"

#include <llvm/IR/Module.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/DerivedTypes.h>

#include "TargetAddressSpaces.h"

namespace llvm {
    class Module;
    class Function;
    class GlobalVariable;
}

namespace pocl {

typedef std::map<llvm::Function*, llvm::Function*> FunctionMapping;

void
regenerate_kernel_metadata(llvm::Module &M, FunctionMapping &kernels);

// Remove a function from a module, along with all callsites.
void eraseFunctionAndCallers(llvm::Function *Function);

inline bool
isAutomaticLocal(const std::string &FuncName, llvm::GlobalVariable &Var) {
#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
  return Var.getName().startswith(FuncName + ".") &&
    llvm::isa<llvm::PointerType>(Var.getType()) &&
    Var.getType()->getPointerAddressSpace() == POCL_FAKE_AS_LOCAL;
#else
  // Without the fake address space IDs, there is no reliable way to figure out
  // if the address space is local from the bitcode. We could check its AS
  // against the device's local address space id, but for now lets rely on the
  // naming convention only. Only relying on the naming convention has the problem
  // that LLVM can move private const arrays to the global space which make
  // them look like local arrays (see Github Issue 445). This should be properly
  // fixed in Clang side with e.g. a naming convention for the local arrays to
  // detect them robstly without having logical address space info in the IR.
  return Var.getName().startswith(FuncName + ".") &&
    llvm::isa<llvm::PointerType>(Var.getType()) && !Var.isConstant();
#endif
}

inline bool
is_image_type(const llvm::Type& t) 
{
  if (t.isPointerTy() && t.getPointerElementType()->isStructTy()) {
    llvm::StringRef name = t.getPointerElementType()->getStructName();
    if (name.startswith("opencl.image2d_") || name.startswith("opencl.image3d_") ||
        name.startswith("opencl.image1d_") || name.startswith("struct._pocl_image"))
      return true;
  }
  return false;
}

inline bool
is_sampler_type(const llvm::Type& t)
{
  if (t.isPointerTy() && t.getPointerElementType()->isStructTy())
    {
      llvm::StringRef name = t.getPointerElementType()->getStructName();
      if (name.startswith("opencl.sampler_t")) return true;
    }
  return false;
}

// Checks if the given argument of Func is a local buffer.
bool
isLocalMemFunctionArg(llvm::Function *Func, unsigned ArgIndex);

// Sets the address space metadata of the given function argument.
// Note: The address space ids must be SPIR ids. If it encounters
// argument indices without address space ids in the list, sets
// them to globals.
void
setFuncArgAddressSpaceMD(llvm::Function *Func, unsigned ArgIndex, unsigned AS);

llvm::Metadata *
createConstantIntMD(llvm::LLVMContext &C, int32_t Val);

}

#endif
