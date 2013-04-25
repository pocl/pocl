// Implementation of LLVMUtils, useful common LLVM-related functionality.
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

#include "LLVMUtils.h"

#include "config.h"

#ifdef LLVM_3_2
#include <llvm/Module.h>
#include <llvm/Metadata.h>
#else
#include <llvm/IR/Module.h>
#include <llvm/IR/Metadata.h>
#endif

using namespace llvm;

/**
 * Regenerates the metadata that points to the original kernel
 * (of which finger print was modified) to point to the new
 * kernel.
 *
 * Only checks if the first operand of the metadata is the kernel
 * function.
 */
void
regenerate_kernel_metadata(llvm::Module &M, FunctionMapping &kernels)
{
  // reproduce the opencl.kernel_wg_size_info metadata
  NamedMDNode *wg_sizes = M.getNamedMetadata("opencl.kernel_wg_size_info");
  if (wg_sizes != NULL && wg_sizes->getNumOperands() > 0) 
    {
      for (std::size_t mni = 0; mni < wg_sizes->getNumOperands(); ++mni)
        {
          MDNode *wgsizeMD = dyn_cast<MDNode>(wg_sizes->getOperand(mni));
          for (FunctionMapping::const_iterator i = kernels.begin(),
                 e = kernels.end(); i != e; ++i) 
            {
              Function *old_kernel = (*i).first;
              Function *new_kernel = (*i).second;
              if (old_kernel == new_kernel || wgsizeMD->getNumOperands() == 0 ||
                  dyn_cast<Function>(wgsizeMD->getOperand(0)) != old_kernel) 
                continue;
              // found a wg size metadata that points to the old kernel, copy its
              // operands except the first one to a new MDNode
              SmallVector<Value*, 8> operands;
              operands.push_back(new_kernel);
              for (unsigned opr = 1; opr < wgsizeMD->getNumOperands(); ++opr)
                {
                  operands.push_back(wgsizeMD->getOperand(opr));
                }
              MDNode *new_wg_md = MDNode::get(M.getContext(), operands);
              wg_sizes->addOperand(new_wg_md);
            } 
        }
    }

  // reproduce the opencl.kernels metadata
  NamedMDNode *nmd = M.getNamedMetadata("opencl.kernels");
  if (nmd)
    M.eraseNamedMetadata(nmd);

  nmd = M.getOrInsertNamedMetadata("opencl.kernels");
  for (FunctionMapping::const_iterator i = kernels.begin(),
         e = kernels.end();
       i != e; ++i) {
    MDNode *md = MDNode::get(M.getContext(), ArrayRef<Value *>((*i).second));
    nmd->addOperand(md);
  }
}

