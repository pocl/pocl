// LLVM function pass to replicate the kernel body for all work items 
// in a work group.
// 
// Copyright (c) 2011-2012 Carlos Sánchez de La Lama / URJC and
//                         Pekka Jääskeläinen / TUT
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

#include "llvm/Support/CommandLine.h"
#include "llvm/Metadata.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "WorkitemHandler.h"
#include "Kernel.h"

namespace pocl {

using namespace llvm;

llvm::cl::list<int>
LocalSize("local-size",
          llvm::cl::desc("Local size (x y z)"),
          llvm::cl::multi_val(3));

void
WorkitemHandler::CheckLocalSize(Kernel *K)
{
  llvm::Module *M = K->getParent();
  
  LocalSizeX = LocalSize[0];
  LocalSizeY = LocalSize[1];
  LocalSizeZ = LocalSize[2];
  
  llvm::NamedMDNode *size_info = M->getNamedMetadata("opencl.kernel_wg_size_info");
  if (size_info) {
    for (unsigned i = 0, e = size_info->getNumOperands(); i != e; ++i) {
      llvm::MDNode *KernelSizeInfo = size_info->getOperand(i);
      if (KernelSizeInfo->getOperand(0) == K) {
        LocalSizeX = (llvm::cast<ConstantInt>(KernelSizeInfo->getOperand(1)))->getLimitedValue();
        LocalSizeY = (llvm::cast<ConstantInt>(KernelSizeInfo->getOperand(2)))->getLimitedValue();
        LocalSizeZ = (llvm::cast<ConstantInt>(KernelSizeInfo->getOperand(3)))->getLimitedValue();
      }
    }
  }
}

} // namespace pocl
