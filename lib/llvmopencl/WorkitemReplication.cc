// LLVM function pass to replicate kernel body for several workitems.
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

#include "WorkitemReplication.h"
#include "Workgroup.h"
#include "Barrier.h"
#include "Kernel.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/IRBuilder.h"

using namespace llvm;
using namespace pocl;

static bool block_has_barrier(const BasicBlock *bb);
// static void purge_subgraph(std::vector<BasicBlock *> &new_subgraph,
// 			   const std::vector<BasicBlock *> &original_subgraph,
// 			   const BasicBlock *exit);

namespace {
  static
  RegisterPass<WorkitemReplication> X("workitem", "Workitem replication pass");
}

cl::list<int>
LocalSize("local-size",
	  cl::desc("Local size (x y z)"),
	  cl::multi_val(3));

char WorkitemReplication::ID = 0;

void
WorkitemReplication::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.addRequired<DominatorTree>();
  AU.addRequired<LoopInfo>();
}

bool
WorkitemReplication::runOnFunction(Function &F)
{
  if (!Workgroup::isKernelToProcess(F))
    return false;

  DT = &getAnalysis<DominatorTree>();
  LI = &getAnalysis<LoopInfo>();

  return ProcessFunction(F);
}

bool
WorkitemReplication::ProcessFunction(Function &F)
{
  Module *M = F.getParent();

  CheckLocalSize(&F);

  LocalX = M->getGlobalVariable("_local_id_x");
  LocalY = M->getGlobalVariable("_local_id_y");
  LocalZ = M->getGlobalVariable("_local_id_z");

  // Allocate space for workitem reference maps. Workitem 0 does
  // not need it.
  unsigned workitem_count = LocalSizeZ * LocalSizeY * LocalSizeX;
  ReferenceMap = new ValueToValueMapTy[workitem_count - 1];

  BasicBlockVector original_bbs;
  for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
    if (!block_has_barrier(i))
        original_bbs.push_back(i);
  }

  // BasicBlockVector subgraph;

  // BasicBlock *exit = FindBarriersDFS(&(F.getEntryBlock()),
  //       			     &(F.getEntryBlock()),
  //       			     subgraph);

  // if (exit != NULL) {
  //   // There is a path from entry to exit that does not cross
  //   // any barrier, we need to replicate it now.
  //   replicateWorkitemSubgraph(subgraph, &(F.getEntryBlock()), exit);
  // }

  Kernel *K = cast<Kernel> (&F);

  SmallVector<ParallelRegion *, 8> parallel_regions[workitem_count];
  K->getParallelRegions(parallel_regions[0], *DT);

  for (int z = 0; z < LocalSizeZ; ++z) {
    for (int y = 0; y < LocalSizeY; ++y) {
      for (int x = 0; x < LocalSizeX ; ++x) {

        int index = (LocalSizeY * LocalSizeX * z +
                     LocalSizeX * y +
                     x);

        if (index == 0)
          continue;

        for (SmallVector<ParallelRegion *, 8>::iterator
               i = parallel_regions[0].begin(), e = parallel_regions[0].end();
             i != e; ++i) {
          ParallelRegion *original = (*i);
          ParallelRegion *nueva = original->replicate(ReferenceMap[index - 1],
                                                      (".wi_" + Twine(x) +
                                                       "_" + Twine(y) +
                                                       "_" + Twine(z)));
          parallel_regions[index].push_back(nueva);
          // parallel_regions[index].
          //   push_back((*i)->replicate(ReferenceMap[index - 1],
          //                             (".wi_" + Twine(x) +
          //                              "_" + Twine(y) +
          //                              "_" + Twine(z))));
        }
      }
    }
  }

  for (int z = 0; z < LocalSizeZ; ++z) {
    for (int y = 0; y < LocalSizeY; ++y) {
      for (int x = 0; x < LocalSizeX ; ++x) {

        int index = (LocalSizeY * LocalSizeX * z +
                     LocalSizeX * y +
                     x);

        for (unsigned i = 0, e = parallel_regions[index].size(); i != e; ++i) {
          ParallelRegion *region = parallel_regions[index][i];
          if (index != 0) {
            region->remap(ReferenceMap[index - 1]);
            region->chainAfter(parallel_regions[index - 1][i]);
            region->purge();
          }
          region->insertPrologue(x, y, z);
        }
      }
    }
  }

  // Add the suffixes to original (wi_0_0_0) basic blocks.
  for (BasicBlockVector::iterator i = original_bbs.begin(),
         e = original_bbs.end();
       i != e; ++i)
    (*i)->setName((*i)->getName() + ".wi_0_0_0");

  delete []ReferenceMap;

  // Initialize local size (done at the end to avoid unnecessary
  // replication).
  IRBuilder<> builder(F.getEntryBlock().getFirstNonPHI());

  GlobalVariable *gv;

  gv = M->getGlobalVariable("_local_size_x");
  if (gv != NULL)
    builder.CreateStore(ConstantInt::get(IntegerType::get(M->getContext(), 32),
					 LocalSizeX),
			gv);
  gv = M->getGlobalVariable("_local_size_y");
  if (gv != NULL)
    builder.CreateStore(ConstantInt::get(IntegerType::get(M->getContext(), 32),
					 LocalSizeY),
			gv);
  gv = M->getGlobalVariable("_local_size_z");
  if (gv != NULL)
    builder.CreateStore(ConstantInt::get(IntegerType::get(M->getContext(), 32),
					 LocalSizeZ),
			gv);

  return true;
}

void
WorkitemReplication::CheckLocalSize(Function *F)
{
  Module *M = F->getParent();

  LocalSizeX = LocalSize[0];
  LocalSizeY = LocalSize[1];
  LocalSizeZ = LocalSize[2];

  NamedMDNode *size_info = M->getNamedMetadata("opencl.kernel_wg_size_info");
  if (size_info) {
    for (unsigned i = 0, e = size_info->getNumOperands(); i != e; ++i) {
      MDNode *KernelSizeInfo = size_info->getOperand(i);
      if (KernelSizeInfo->getOperand(0) == F) {
        LocalSizeX = (cast<ConstantInt>(KernelSizeInfo->getOperand(1)))->getLimitedValue();
        LocalSizeY = (cast<ConstantInt>(KernelSizeInfo->getOperand(2)))->getLimitedValue();
        LocalSizeZ = (cast<ConstantInt>(KernelSizeInfo->getOperand(3)))->getLimitedValue();
      }
    }
  }
}

// // Perform a deep first seach on the subgraph starting on bb. On the
// // outermost recursive call, entry = bb. Barriers are handled by
// // recursive calls, so on return only subgraph needs still to be
// // replicated. Return value is last basicblock (exit) of original graph.
// BasicBlock*
// WorkitemReplication::FindBarriersDFS(BasicBlock *bb,
// 				     BasicBlock *entry,
// 				     BasicBlockVector &subgraph)
// {
//   // // Do nothing if basicblock already visited, to avoid
//   // // infinite recursion when processing loops.
//   // if (subgraph.count(bb))
//   //   return NULL;

//   TerminatorInst *t = bb->getTerminator();
  
//   if (block_has_barrier(bb) &&
//       (ProcessedBarriers.count(bb) == 0))
//     {      
//       BasicBlockVector pre_subgraph;
// #ifndef NDEBUG
//       bool found = 
// #endif
//           FindSubgraph(pre_subgraph, entry, bb);
//       assert(found && "Subgraph to a barrier does not reach the barrier!");
//       //pre_subgraph.erase(bb); // Remove barrier basicblock from subgraph.
//       for (std::vector<BasicBlock *>::const_iterator i = pre_subgraph.begin(),
// 	     e = pre_subgraph.end();
// 	   i != e; ++i) {
// 	if (block_has_barrier(*i) &&
// 	    ProcessedBarriers.count(*i) == 0) {
// 	  // Subgraph to this barriers has still unprocessed barriers. Do
// 	  // not process this barrier yet (it will be done when coming
// 	  // from the path through the previous barrier).
// 	  return NULL;
// 	}
//       }

//       // Mark this barrier as processed.
//       ProcessedBarriers.insert(bb);
      
//       // Replicate subgraph from entry to the barrier for
//       // all workitems.
//       replicateWorkitemSubgraph(pre_subgraph, entry, bb->getSinglePredecessor());

//       // Continue processing after the barrier.
//       BasicBlockVector post_subgraph;
//       assert (t->getNumSuccessors() == 1);
//       Loop *l = LI->getLoopFor(bb);
//       bb = t->getSuccessor(0);
//       // If this is a latch barrier there is nothing to
//       // process after the barrier.
//       if ((l != NULL) && (l->getHeader() == bb))
//         return NULL;
      
//       BasicBlock *exit = FindBarriersDFS(bb, bb, post_subgraph);
//       if (exit != NULL)
//         replicateWorkitemSubgraph(post_subgraph, bb, exit);

//       return NULL;
//     }

//   subgraph.push_back(bb);

//   if (t->getNumSuccessors() == 0)
//     return t->getParent();

//   // Find barriers in the successors (depth first).
//   BasicBlock *r = NULL;
//   BasicBlock *s;
//   Loop *l = LI->getLoopFor(bb);
//   for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i) {
//     BasicBlock *successor = t->getSuccessor(i);
//     if ((l != NULL) && (l->getHeader() == successor))
//       continue;
//     s = FindBarriersDFS(t->getSuccessor(i), entry, subgraph);
//     if (s != NULL) {
//       assert((r == NULL || r == s) &&
// 	     "More than one function tail within same barrier path!\n");
//       r = s;
//     }
//     if (t != bb->getTerminator()) {
//       // Terminator changed, this BB was made part of a parallel region
//       // and replicated, do not look for more successors.
//       break;
//     }
//   }

//   return r;
// }

// // Find subgraph between entry and exit basicblocks.
// bool
// WorkitemReplication::FindSubgraph(BasicBlockVector &subgraph,
//                                   BasicBlock *entry,
//                                   BasicBlock *exit)
// {
//   if (entry == exit) {
//     // DO NOT ADD THE BARRIER ITSELF.
//     //   subgraph.push_back(entry);
//     return true;
//   }

//   bool found = false;
//   const TerminatorInst *t = entry->getTerminator();
//   Loop *l = LI->getLoopFor(entry);
//   for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i) {
//     BasicBlock *b = t->getSuccessor(i);
//     if ((l != NULL) && (l->getHeader() == b)) {
//       // This is a loop backedge. Do not find subgraphs across
//       // those.
//       continue;
//     }
//     found |= FindSubgraph(subgraph, t->getSuccessor(i), exit);
//   }
    
//   if (found)
//     subgraph.push_back(entry);

//   return found;
// }

// void
// WorkitemReplication::SetBasicBlockNames(BasicBlockVector &subgraph)
// {
//   for (BasicBlockVector::iterator i = subgraph.begin(), e = subgraph.end();
//        i != e; ++i) {
//     BasicBlock *bb = *i;
//     StringRef s = bb->getName();
//     for (int z = 0; z < LocalSizeZ; ++z) {
//       for (int y = 0; y < LocalSizeY; ++y) {
//         for (int x = 0; x < LocalSizeX ; ++x) {
          
//           if ((z == 0) && (y == 0) && (x == 0))
//             continue;

//           int index = (LocalSizeY * LocalSizeX * z +
//                        LocalSizeX * y +
//                        x) - 1;
          
//           bb = cast<BasicBlock> (ReferenceMap[index][bb]);
//           bb->setName(s + ".wi_" + Twine(x) + "_" + Twine(y) + "_" + Twine(z));
//         }
//       }
//     }
//   }
// }

// void
// WorkitemReplication::replicateWorkitemSubgraph(BasicBlockVector subgraph,
// 					       BasicBlock *entry,
// 					       BasicBlock *exit)
// {
//   // This might happen if one barrier follow another. Do nothing.
//   if (subgraph.size() == 0)
//     return;

//   BasicBlockVector original_subgraph = subgraph;

//   BasicBlockVector s;

//   assert (entry != NULL && exit != NULL);

//   IRBuilder<> builder(entry->getContext());

//   for (int z = 0; z < LocalSizeZ; ++z) {
//     for (int y = 0; y < LocalSizeY; ++y) {
//       for (int x = 0; x < LocalSizeX; ++x) {
	
// 	if (x == (LocalSizeX - 1) &&
// 	    y == (LocalSizeY - 1) &&
// 	    z == (LocalSizeZ - 1)) {
// 	  builder.SetInsertPoint(entry, entry->getFirstNonPHI());
// 	  if (LocalX != NULL) {
// 	    builder.CreateStore(ConstantInt::get(IntegerType::
// 						 get(entry->getContext(),
// 						     32), x), LocalX);
// 	  }
//           // Might need to write other dimensions also in the last case,
//           // as we can have a size of 1 in any x and or y dimension.
//           if (x == 0) {
//             if (LocalY != NULL) {
//               builder.CreateStore(ConstantInt::get(IntegerType::
//                                                    get(entry->getContext(),
//                                                        32), y), LocalY);
//             }
//             if (y == 0) {
//               if (LocalZ != NULL) {
//                 builder.CreateStore(ConstantInt::get(IntegerType::
//                                                      get(entry->getContext(),
//                                                          32), z), LocalZ);
//               }
//             }
//           }

//           SetBasicBlockNames(original_subgraph);

//           // No need to update LoopInfo here, replicated code
//           // is never replicated again (FALSE, fails without it).
//           DT->runOnFunction(*(entry->getParent()));
//           LI->releaseMemory();
//           LI->getBase().Calculate(DT->getBase());
// 	  return;
// 	}

// 	int i = (LocalSizeY * LocalSizeX * z +
//                  LocalSizeX * y +
//                  x);

// 	replicateBasicblocks(s, ReferenceMap[i], subgraph);
// 	purge_subgraph(s, subgraph,
// 		       cast<BasicBlock> (ReferenceMap[i][exit]));
// 	updateReferences(s, ReferenceMap[i]);
	
// 	ReferenceMap[i].erase(exit->getTerminator());
// 	BranchInst::Create(cast<BasicBlock>(ReferenceMap[i][entry]),
// 			   exit->getTerminator());
//         // This is not true, barriers must have one succesor (barrier BBs are
//         // not replicated so we do not want functionality ther) and one predecessor
//         // so that paralle regions have a definite exit edge), but the parallel region
//         // itself might have more than one successor (for example, one edge going
//         // to the barrier and another one elsewhere).
//         // assert((exit->getTerminator()->getNumSuccessors() <= 1) &&
//         //        "Multiple succesors of parallel section (uncanonicalized barriers?)!");
// 	exit->getTerminator()->eraseFromParent();

// 	builder.SetInsertPoint(entry, entry->getFirstNonPHI());
// 	if (LocalX != NULL) {
// 	  builder.CreateStore(ConstantInt::get(IntegerType::
// 					       get(entry->getContext(),
// 						   32), x), LocalX);
	  
// 	}
// 	if (x == 0) {
// 	  if (LocalY != NULL) {
// 	    builder.CreateStore(ConstantInt::get(IntegerType::
// 						 get(entry->getContext(),
// 						     32), y), LocalY);
	    
// 	  }
// 	  if (y == 0) {
// 	    if (LocalZ != NULL) {
// 	      builder.CreateStore(ConstantInt::get(IntegerType::
// 						   get(entry->getContext(),
// 						       32), z), LocalZ);
// 	    }
// 	  }
// 	}

// 	subgraph = s;
// 	entry = cast<BasicBlock>(ReferenceMap[i][entry]);
// 	exit = cast<BasicBlock>(ReferenceMap[i][exit]);

// 	s.clear();
//       }
//     }
//   }

//   // We should never exit through here.
//   assert (0);
// }

// static void
// purge_subgraph(std::vector<BasicBlock *> &new_subgraph,
// 	       const std::vector<BasicBlock *> &original_subgraph,
// 	       const BasicBlock *exit)
// {
//   std::set<BasicBlock *> original(original_subgraph.begin(),
//                                   original_subgraph.end());

//   for (std::vector<BasicBlock *>::iterator i = new_subgraph.begin(),
// 	 e = new_subgraph.end();
//        i != e; ++i) {
//     if (*i == exit)
//       continue;
//     // Remove CFG edges going to basic blocks which
//     // were not contained in original graph (impossible
//     // branches).
//     TerminatorInst *t = (*i)->getTerminator();
//     for (unsigned u = 0; u < t->getNumSuccessors(); ++u) {
//       if (original.count(t->getSuccessor(u)) == 0) {
// 	BasicBlock *unreachable = BasicBlock::Create((*i)->getContext(),
// 						     "unreachable",
// 						     (*i)->getParent());
// 	new UnreachableInst((*i)->getContext(), unreachable);
// 	t->setSuccessor(u, unreachable);
//       }
//     }
//   }
// }

static bool
block_has_barrier(const BasicBlock *bb)
{
  for (BasicBlock::const_iterator i = bb->begin(), e = bb->end();
       i != e; ++i) {
    if (isa<Barrier>(i))
      return true;
  }

  return false;
}

// void
// WorkitemReplication::replicateBasicblocks(BasicBlockVector &new_graph,
// 					  ValueValueMap &reference_map,
// 					  const BasicBlockVector &graph)
// {
//   for (std::vector<BasicBlock *>::const_iterator i = graph.begin(),
// 	 e = graph.end();
//        i != e; ++i) {
//     BasicBlock *b = *i;

//     assert(!block_has_barrier(b) && "Barrier blocks should not be replicated!");

//     BasicBlock *new_b = BasicBlock::Create(b->getContext(),
// 					   b->getName(),
// 					   b->getParent());
    
//     reference_map[b] = new_b;
//     new_graph.push_back(new_b);

//     for (BasicBlock::iterator i2 = b->begin(), e2 = b->end();
// 	 i2 != e2; ++i2) {
//       if (isReplicable(i2)) {
// 	Instruction *i = i2->clone();
// 	reference_map[i2] = i;//.insert(std::make_pair(i2, i));
// 	new_b->getInstList().push_back(i);
//       }
//     }
//   }
// }

// void
// WorkitemReplication::updateReferences(const BasicBlockVector &graph,
// 				      const ValueValueMap &reference_map)
// {
//   for (std::vector<BasicBlock *>::const_iterator i = graph.begin(),
// 	 e = graph.end();
//        i != e; ++i) {
//     BasicBlock *b = *i;
//     for (BasicBlock::iterator i2 = b->begin(), e2 = b->end();
// 	 i2 != e2; ++i2) {
//       Instruction *i = i2;
//       for (std::map<Value *, Value *>::const_iterator i3 =
// 	     reference_map.begin(), e3 = reference_map.end();
// 	   i3 != e3; ++i3) {
// 	i->replaceUsesOfWith(i3->first, i3->second);
// 	// PHINode incoming BBs are no longer uses, we need a special case
// 	// for them.
// 	if (PHINode *phi = dyn_cast<PHINode>(i)) {
// 	  for (unsigned i4 = 0, e4 = phi->getNumIncomingValues(); i4 != e4; ++i4) {
// 	    if (phi->getIncomingBlock(i4) == i3->first) {
// 	      phi->setIncomingBlock(i4, cast<BasicBlock>(i3->second));
// 	    }
// 	  }
// 	}
//       }
//     }
//   }
// }

// bool
// WorkitemReplication::isReplicable(const Instruction *i)
// {
//   if (const StoreInst *s = dyn_cast<StoreInst>(i)) {
//     const Value *v = s->getPointerOperand();
//     const GlobalVariable *gv = dyn_cast<GlobalVariable>(v);
//     if (gv == LocalX || gv == LocalY || gv == LocalZ)
//       return false;
//   }

//   return true;
// }
