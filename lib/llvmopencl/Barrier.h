// Class for barrier instructions, modelled as a CallInstr.
// 
// Copyright (c) 2011 Universidad Rey Juan Carlos
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

#include "config.h"
#if (defined LLVM_3_1 or defined LLVM_3_2)
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#else
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#endif

#include "llvm/Support/Casting.h"

#define BARRIER_FUNCTION_NAME "pocl.barrier"

namespace pocl {
  
  class Barrier : public llvm::CallInst {

  public:
    static void GetBarriers(llvm::SmallVectorImpl<Barrier *> &B,
                            llvm::Module &M) {
      llvm::Function *F = M.getFunction(BARRIER_FUNCTION_NAME);
      if (F != NULL) {
        for (llvm::Function::use_iterator i = F->use_begin(), e = F->use_end();
             i != e; ++i)
          B.push_back(llvm::cast<Barrier>(*i));
      }
    }
    /**
     * Creates a new barrier before the given instruction.
     *
     * If there was already a barrier there, returns the old one.
     */
    static Barrier *Create(llvm::Instruction *InsertBefore) {
      llvm::Module *M = InsertBefore->getParent()->getParent()->getParent();

      if (InsertBefore != &InsertBefore->getParent()->front() && 
          llvm::isa<Barrier>(InsertBefore->getPrevNode()))
        return llvm::cast<Barrier>(InsertBefore->getPrevNode());

      llvm::Function *F = llvm::cast<llvm::Function>
        (M->getOrInsertFunction(BARRIER_FUNCTION_NAME,
                                llvm::Type::getVoidTy(M->getContext()),
                                NULL));
      return llvm::cast<pocl::Barrier>
        (llvm::CallInst::Create(F, "", InsertBefore));
    }
    static bool classof(const Barrier *) { return true; };
    static bool classof(const llvm::CallInst *C) {
      return C->getCalledFunction() != NULL &&
        C->getCalledFunction()->getName() == BARRIER_FUNCTION_NAME;
    }
    static bool classof(const Instruction *I) {
      return (llvm::isa<llvm::CallInst>(I) &&
              classof(llvm::cast<llvm::CallInst>(I)));
    }
    static bool classof(const User *U) {
      return (llvm::isa<Instruction>(U) &&
              classof(llvm::cast<llvm::Instruction>(U)));
    }


    static bool hasOnlyBarrier(const llvm::BasicBlock *bb) 
    {
      return endsWithBarrier(bb) && bb->size() == 2;
    }

    static bool hasBarrier(const llvm::BasicBlock *bb) 
    {
      for (llvm::BasicBlock::const_iterator i = bb->begin(), e = bb->end();
           i != e; ++i) 
        {
          if (llvm::isa<Barrier>(i)) return true;
        }
      return false;
    }

    // returns true in case the given basic block ends with a barrier,
    // that is, contains only a branch instruction after a barrier call
    static bool endsWithBarrier(const llvm::BasicBlock *bb) 
    {
      const llvm::TerminatorInst *t = bb->getTerminator();
      if (t == NULL) return false;
      return bb->size() > 1 && t->getPrevNode() != NULL && 
          llvm::isa<Barrier>(t->getPrevNode());
    }
  };

}

