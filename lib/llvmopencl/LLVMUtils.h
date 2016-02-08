// Header for LLVMUtils, useful common LLVM-related functionality.
// 
// Copyright (c) 2013 Pekka Jääskeläinen / TUT
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

namespace llvm {
    class Module;
    class Function;
    class GlobalVariable;
}

namespace pocl {

typedef std::map<llvm::Function*, llvm::Function*> FunctionMapping;

void
regenerate_kernel_metadata(llvm::Module &M, FunctionMapping &kernels);

inline bool
is_automatic_local(const std::string& funcName, llvm::GlobalVariable &var) 
{
  return var.getName().startswith(funcName + ".") &&
    llvm::isa<llvm::PointerType>(var.getType()) &&
    var.getType()->getPointerAddressSpace() == POCL_ADDRESS_SPACE_LOCAL;
}

inline bool
is_image_type(const llvm::Type& t) 
{
  if (t.isPointerTy() && t.getPointerElementType()->isStructTy()) {
    llvm::StringRef name = t.getPointerElementType()->getStructName();
    if (name.startswith("opencl.image2d_t") || name.startswith("opencl.image3d_t") ||
        name.startswith("opencl.image1d_t") || name.startswith("struct._pocl_image"))
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
      if (name.startswith("opencl.sampler_t_")) return true;     
    }
  return false;
}

}

#endif
