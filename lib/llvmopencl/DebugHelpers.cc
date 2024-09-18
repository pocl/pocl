// Helpers for debugging the kernel compiler.
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

#include <fstream>
#include <iostream>
#include <sstream>
#include <set>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "pocl.h"

#include <llvm/Analysis/RegionInfo.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "Barrier.h"
#include "DebugHelpers.h"
#include "LLVMUtils.h"
#include "Workgroup.h"
#include "pocl_file_util.h"

POP_COMPILER_DIAGS

using namespace llvm;

namespace pocl {

static std::string getDotBasicBlockID(llvm::BasicBlock* bb) {
  std::ostringstream namess;
  namess << "BB" << std::hex << bb;
  return namess.str();
}

static void printBranches(
  llvm::BasicBlock* b, std::ostream& s, bool /*highlighted*/) {

  auto term = b->getTerminator();
  for (unsigned i = 0; i < term->getNumSuccessors(); ++i)
    {
      BasicBlock *succ = term->getSuccessor(i);
      s << getDotBasicBlockID(b) << " -> " << getDotBasicBlockID(succ) << ";"
        << std::endl;
    }
  s << std::endl;
}

static void printBasicBlock(
  llvm::BasicBlock* b, std::ostream& s, bool highlighted) {
  //      if (!Barrier::hasBarrier(b)) continue;
  s << getDotBasicBlockID(b);
  s << "[shape=rect,style=";
  if (Barrier::hasBarrier(b))
    s << "dotted";
  else
    s << "solid";

  if (highlighted) {
    s << ",color=red,style=filled";
  }
  s << ",label=\"" << b->getName().str() << ":\\n";

  // The work-item loop control structures.
  if (b->getName().starts_with("pregion_for_cond")) {
    s << "wi-loop branch\\n";
  } else if (b->getName().starts_with("pregion_for_inc")) {
    s << "local_id_* increment\\n";
  } else if (b->getName().starts_with("pregion_for_init")) {
    s << "wi-loop init\\n";
  } else if (b->getName().starts_with("pregion_for_end")) {
    s << "wi-loop exit\\n";
  } else {
    // analyze the contents of the BB
    int previousNonBarriers = 0;
    for (llvm::BasicBlock::iterator instr = b->begin();
         instr != b->end(); ++instr) {

        if (isa<Barrier>(instr)) {
          s << "BARRIER\\n";
          previousNonBarriers = 0;
        } else if (isa<BranchInst>(instr)) {
          s << "branch\\n";
          previousNonBarriers = 0;
        } else if (isa<PHINode>(instr)) {
          s << "PHI\\n";
          previousNonBarriers = 0;
        } else if (isa<ReturnInst>(instr)) {
          s << "RETURN\\n";
          previousNonBarriers = 0;
        } else if (isa<UnreachableInst>(instr)) {
          s << "UNREACHABLE\\n";
          previousNonBarriers = 0;
        } else {
          if (previousNonBarriers == 0)
            s << "...program instructions...\\n";
          previousNonBarriers++;
        }
      }
  }
  s << "\"";
  s << "]";
  s << ";" << std::endl << std::endl;
}

/**
 * pocl-specific dumping of the LLVM Function as a control flow graph in the
 * Graphviz dot format.
 *
 * @param F the function to dump
 * @param fname the target file name
 * @param regions highlight these parallel regions in the graph
 * @param highlights highlight these basic blocks in the graph
 */
void dumpCFG(llvm::Function &F, std::string fname,
             const std::vector<llvm::Region *> *Regions,
             const ParallelRegion::ParallelRegionVector *ParRegions,
             const std::set<llvm::BasicBlock *> *highlights) {

  unsigned LastRegID = 0;

  if (fname == "")
    fname = std::string("pocl_cfg.") + F.getName().str() + ".dot";

  std::string origName = fname;
  int counter = 0;
  while (pocl_exists (fname.c_str())) {
    std::ostringstream ss;
    ss << origName << "." << counter;
    fname = ss.str();
    ++counter;
  }

  std::ofstream s;
  s.open(fname.c_str(), std::ios::trunc);
  s << "digraph " << F.getName().str() << " {" << std::endl;

  std::set<BasicBlock*> regionBBs;

  if (Regions != nullptr && Regions->size()) {
    for (const Region *R : *Regions) {
      unsigned RegID = ++LastRegID;
      s << "\tsubgraph cluster" << RegID << " {" << std::endl;
      for (Region::const_block_iterator RI = R->block_begin(),
                                        RE = R->block_end();
           RI != RE; ++RI) {
        BasicBlock *BB = *RI;
        printBasicBlock(
            BB, s,
            (highlights != NULL && highlights->find(BB) != highlights->end()));
        regionBBs.insert(BB);
      }
      s << "label=\"Parallel region #" << RegID << "\";" << std::endl;
      s << "}" << std::endl;
    }
  }

  if (ParRegions != nullptr) {
    for (ParallelRegion::ParallelRegionVector::const_iterator
             RI = ParRegions->begin(),
             RE = ParRegions->end();
         RI != RE; ++RI) {
      ParallelRegion *PR = *RI;
      s << "\tsubgraph cluster" << PR->getID() << " {" << std::endl;
      for (ParallelRegion::iterator It = PR->begin(), E = PR->end(); It != E;
           ++It) {
        BasicBlock *BB = *It;
        printBasicBlock(
            BB, s,
            (highlights != NULL && highlights->find(BB) != highlights->end()));
        regionBBs.insert(BB);
      }
      s << "label=\"Parallel region #" << PR->getID() << "\";" << std::endl;
      s << "}" << std::endl;
    }
  }
  for (Function::iterator FI = F.begin(), e = F.end(); FI != e; ++FI) {
    BasicBlock *BB = &*FI;
    if (regionBBs.find(BB) != regionBBs.end())
      continue;
    printBasicBlock(
        BB, s, highlights != NULL && highlights->find(BB) != highlights->end());
  }

  for (Function::iterator FI = F.begin(), e = F.end(); FI != e; ++FI) {
    BasicBlock *BB = &*FI;
    printBranches(
        BB, s, highlights != NULL && highlights->find(BB) != highlights->end());
  }

  s << "}" << std::endl;
  s.close();
#if 0
  std::cout << "### dumped CFG to " << fname << std::endl;
#endif
}

bool chopBBs(llvm::Function &F, llvm::Pass &) {
  bool fchanged = false;
  const int MAX_INSTRUCTIONS_PER_BB = 70;
  do {
    fchanged = false;
    for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
      BasicBlock *b = &*i;
      
      if (b->size() > MAX_INSTRUCTIONS_PER_BB + 1)
        {
          int count = 0;
          BasicBlock::iterator splitPoint = b->begin();
          while (count < MAX_INSTRUCTIONS_PER_BB || isa<PHINode>(splitPoint))
            {
              ++splitPoint;
              ++count;
            }
          SplitBlock(b, &*splitPoint);
          fchanged = true;
          break;
        }
    }  

  } while (fchanged);
  return fchanged;
}

void PoCLCFGPrinter::dumpModule(llvm::Module &M) {
  for (llvm::Function &F : M) {
    std::string Name;
    if (F.hasName())
      Name += F.getName();
    else
      Name += "anonymous_func";
    Name = Prefix + Name + ".dot";
    // TODO somehow supply regions/highlights
    dumpCFG(F, Name, nullptr, nullptr);
  }
}


llvm::PreservedAnalyses PoCLCFGPrinter::run(llvm::Module &M,
                                            llvm::ModuleAnalysisManager &AM) {
  dumpModule(M);
  return PreservedAnalyses::all();
}

void PoCLCFGPrinter::registerWithPB(llvm::PassBuilder &PB) {
  PB.registerPipelineParsingCallback(
      [](::llvm::StringRef Name, ::llvm::ModulePassManager &MPM,
         llvm::ArrayRef<::llvm::PassBuilder::PipelineElement>) {
        if (Name == "print<pocl-cfg>") {
          MPM.addPass(PoCLCFGPrinter(llvm::errs()));
          return true;
        }
        // the string X in "print<pocl-cfg;X>" will be passed to constructor;
        // this can be used to run multiple times and dump to different files
        if (Name.consume_front("print<pocl-cfg;") && Name.consume_back(">")) {
          MPM.addPass(PoCLCFGPrinter(llvm::errs(), Name));
          return true;
        }

        return false;
      });
}

} // namespace pocl
