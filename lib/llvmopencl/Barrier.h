// Class for barrier instructions, modelled as a CallInstr.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos
//               2011-2019 Pekka Jääskeläinen
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

#ifndef POCL_BARRIERS_H
#define POCL_BARRIERS_H

#include "config.h"

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ValueSymbolTable.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/Support/Casting.h>

#define BARRIER_FUNCTION_NAME "pocl.barrier"

namespace pocl {

  class Barrier : public llvm::CallInst {
  public:
    static void GetBarriers(llvm::SmallVectorImpl<Barrier *> &B,
                            llvm::Module &M) {
      llvm::Function *F = M.getFunction(BARRIER_FUNCTION_NAME);
      if (F != NULL) {
        for (llvm::Function::use_iterator I = F->use_begin(), E = F->use_end();
             I != E; ++I)
          B.push_back(llvm::cast<Barrier>(*I));
      }
    }

    static bool IsLoopWithBarrier(llvm::Loop &L) {
      for (llvm::Loop::block_iterator i = L.block_begin(), e = L.block_end();
           i != e; ++i) {
        for (llvm::BasicBlock::iterator j = (*i)->begin(), e = (*i)->end();
             j != e; ++j) {
          if (llvm::isa<Barrier>(j)) {
            return true;
          }
        }
      }
      return false;
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
      llvm::FunctionCallee FC =
        M->getOrInsertFunction(BARRIER_FUNCTION_NAME,
                                llvm::Type::getVoidTy(M->getContext()));
      llvm::Function *F = llvm::cast<llvm::Function>(FC.getCallee());
      F->addFnAttr(llvm::Attribute::Convergent);
      return llvm::cast<pocl::Barrier>
        (llvm::CallInst::Create(F, "", InsertBefore));
    }
    static bool classof(const Barrier *) { return true; }
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
    static bool classof(const Value *V) {
      return (llvm::isa<User>(V) &&
              classof(llvm::cast<llvm::User>(V)));
    }

    static bool hasOnlyBarrier(const llvm::BasicBlock *BB) {
      return endsWithBarrier(BB) && BB->size() == 2;
    }

    static bool hasBarrier(const llvm::BasicBlock *BB)
    {
      for (llvm::BasicBlock::const_iterator I = BB->begin(), E = BB->end();
           I != E; ++I)
        {
          if (llvm::isa<Barrier>(I)) return true;
        }
      return false;
    }

    // Returns true in case the given basic block starts with a barrier,
    // that is, contains a branch instruction after possible PHI nodes.
    static bool startsWithBarrier(const llvm::BasicBlock *BB) {
      const llvm::Instruction *Inst = BB->getFirstNonPHI();
      if (Inst == NULL)
        return false;
      return llvm::isa<Barrier>(Inst);
    }

    // Returns true in case the given basic block ends with a barrier,
    // that is, contains only a branch instruction after a barrier call.
    static bool endsWithBarrier(const llvm::BasicBlock *BB) {
      const llvm::Instruction *Inst = BB->getTerminator();
      if (Inst == NULL)
        return false;
      return BB->size() > 1 && Inst->getPrevNode() != NULL &&
          llvm::isa<Barrier>(Inst->getPrevNode());
    }
  };

}

#endif
