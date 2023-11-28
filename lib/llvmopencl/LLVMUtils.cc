// Implementation of LLVMUtils, useful common LLVM-related functionality.
//
// Copyright (c) 2013-2019 Pekka Jääskeläinen
//               2023 Pekka Jääskeläinen / Intel Finland Oy
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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/ADT/SmallSet.h>

// include all passes & analysis
#include "AllocasToEntry.h"
#include "AutomaticLocals.h"
#include "BarrierTailReplication.h"
#include "BreakConstantGEPs.h"
#include "CanonicalizeBarriers.h"
#include "DebugHelpers.h"
#include "Flatten.hh"
#include "FlattenBarrierSubs.hh"
#include "FlattenGlobals.hh"
#include "HandleSamplerInitialization.h"
#include "ImplicitConditionalBarriers.h"
#include "ImplicitLoopBarriers.h"
#include "InlineKernels.hh"
#include "IsolateRegions.h"
#include "LoopBarriers.h"
#include "MinLegalVecSize.hh"
#include "OptimizeWorkItemFuncCalls.h"
#include "PHIsToAllocas.h"
#include "ParallelRegion.h"
#include "RemoveBarrierCalls.h"
#include "RemoveOptnoneFromWIFunc.h"
#include "SubCFGFormation.h"
#include "VariableUniformityAnalysis.h"
#include "WorkItemAliasAnalysis.h"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
#include "WorkitemLoops.h"
#include "WorkitemReplication.h"

#include "LLVMUtils.h"
POP_COMPILER_DIAGS

#include "Barrier.h"

#include "pocl_llvm_api.h"
#include "pocl_spir.h"

#include <iostream>
#include <set>

using namespace llvm;

//#define DEBUG_LLVM_UTILS

static void findInstructionUsesImpl(Use &U, std::vector<Use *> &Uses,
                                    std::set<Use *> &Visited) {
  if (Visited.count(&U))
    return;
  Visited.insert(&U);

  assert(isa<Constant>(*U));
  if (isa<Instruction>(U.getUser())) {
    Uses.push_back(&U);
    return;
  }
  if (isa<Constant>(U.getUser())) {
    for (auto &U : U.getUser()->uses())
      findInstructionUsesImpl(U, Uses, Visited);
    return;
  }

  // Catch other user kinds - we may need to process them (somewhere but not
  // here).
  llvm_unreachable("Unexpected user kind.");
}

// Return list of non-constant leaf use edges whose users are instructions.
static std::vector<Use *> findInstructionUses(GlobalVariable *GVar) {
  std::vector<Use *> Uses;
  std::set<Use *> Visited;
  for (auto &U : GVar->uses())
    findInstructionUsesImpl(U, Uses, Visited);
  return Uses;
}


namespace pocl {

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
              Function *func_from_md;
              func_from_md = dyn_cast<Function>(
                dyn_cast<ValueAsMetadata>(wgsizeMD->getOperand(0))->getValue());
              if (old_kernel == new_kernel || wgsizeMD->getNumOperands() == 0 ||
                  func_from_md != old_kernel) 
                continue;
              // found a wg size metadata that points to the old kernel, copy its
              // operands except the first one to a new MDNode
              SmallVector<Metadata*, 8> operands;
              operands.push_back(llvm::ValueAsMetadata::get(new_kernel));
              for (unsigned opr = 1; opr < wgsizeMD->getNumOperands(); ++opr) {
                  operands.push_back(wgsizeMD->getOperand(opr));
              }
              MDNode *new_wg_md = MDNode::get(M.getContext(), operands);
              wg_sizes->addOperand(new_wg_md);
            }
        }
    }

  // reproduce the opencl.kernels metadata, if it exists
  // unconditionally adding opencl.kernels confuses the
  // metadata parser in pocl_llvm_metadata.cc, which uses
  // "opencl.kernels" to distinguish old SPIR format from new
  NamedMDNode *nmd = M.getNamedMetadata("opencl.kernels");
  if (nmd) {
    M.eraseNamedMetadata(nmd);

    nmd = M.getOrInsertNamedMetadata("opencl.kernels");
    for (FunctionMapping::const_iterator i = kernels.begin(),
         e = kernels.end();
       i != e; ++i) {
      MDNode *md = MDNode::get(M.getContext(), ArrayRef<Metadata *>(
        llvm::ValueAsMetadata::get((*i).second)));
      nmd->addOperand(md);
    }
  }

}

// Recursively descend a Value's users and convert any constant expressions into
// regular instructions.
void breakConstantExpressions(llvm::Value *Val, llvm::Function *Func) {
  std::vector<llvm::Value *> Users(Val->user_begin(), Val->user_end());
  for (auto *U : Users) {
    if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(U)) {
      // First, make sure no users of this constant expression are themselves
      // constant expressions.
      breakConstantExpressions(U, Func);

      // Convert this constant expression to an instruction.
      llvm::Instruction *I = CE->getAsInstruction();
      I->insertBefore(&*Func->begin()->begin());
      CE->replaceAllUsesWith(I);
      CE->destroyConstant();
    }
  }
}

static void
recursivelyFindCalledFunctions(llvm::SmallSet<llvm::Function *, 12> &FSet,
                               llvm::Function *F) {
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {
      Instruction *Instr = dyn_cast<Instruction>(BI);
      if (!llvm::isa<CallInst>(Instr))
        continue;
      CallInst *CallInstr = dyn_cast<CallInst>(Instr);
      Function *Callee = CallInstr->getCalledFunction();
      if (!Callee)
        continue;
      if (Callee->isDeclaration())
        continue;
#if LLVM_MAJOR < 11
      if (FSet.count(Callee) > 0)
#else
      if (FSet.contains(Callee))
#endif
        continue;
      FSet.insert(Callee);
      recursivelyFindCalledFunctions(FSet, Callee);
    }
  }
}

bool isGVarUsedByFunction(llvm::GlobalVariable *GVar, llvm::Function *F) {
  std::vector<Use *> Uses = findInstructionUses(GVar);
  // we must recursively search for each function called by F, because
  // this (isGVarUsedByFunction) is called by isAutomaticLocal(),
  // which in turn is called on "unprocessed" LLVM bitcode (or SPIRV),
  // where we haven't run any LLVM passes yet; in particular the pass
  // that inlines all functions using "special" variables and kernels
  llvm::SmallSet<llvm::Function *, 12> CalledFunctionSet;
  CalledFunctionSet.insert(F);
  recursivelyFindCalledFunctions(CalledFunctionSet, F);
  std::vector<Function *> Funcs;
  for (auto &U : Uses) {
    if (Instruction *I = dyn_cast<Instruction>(U->getUser()))
    {
#if LLVM_MAJOR < 11
      if (CalledFunctionSet.count(I->getFunction()) > 0)
#else
      if (CalledFunctionSet.contains(I->getFunction()))
#endif
        return true;
    }
  }
  return false;
}


bool
isAutomaticLocal(llvm::Function *F, llvm::GlobalVariable &Var) {
  // Without the fake address space IDs, there is no reliable way to figure out
  // if the address space is local from the bitcode. We could check its AS
  // against the device's local address space id, but for now lets rely on the
  // naming convention only. Only relying on the naming convention has the problem
  // that LLVM can move private const arrays to the global space which make
  // them look like local arrays (see Github Issue 445). This should be properly
  // fixed in Clang side with e.g. a naming convention for the local arrays to
  // detect them robstly without having logical address space info in the IR.
  std::string FuncName = F->getName().str();
  if (!llvm::isa<llvm::PointerType>(Var.getType()) || Var.isConstant())
    return false;
  if (Var.getName().startswith(FuncName + ".")) {
    assert(isGVarUsedByFunction(&Var, F) == true);
    return true;
  }

  // handle SPIR local AS (3)
  if (Var.getParent() && Var.getParent()->getNamedMetadata("spirv.Source") &&
      (Var.getType()->getAddressSpace() == SPIR_ADDRESS_SPACE_LOCAL)) {

    if (!Var.hasName())
      Var.setName(llvm::Twine(FuncName, ".__anon_gvar"));
    // check it's used by this particular function
    return isGVarUsedByFunction(&Var, F);
  }

  return false;
}

void eraseFunctionAndCallers(llvm::Function *Function) {
  if (!Function)
    return;

  std::vector<llvm::Value *> Callers(Function->user_begin(),
                                     Function->user_end());
  for (auto &U : Callers) {
    llvm::CallInst *Call = llvm::dyn_cast<llvm::CallInst>(U);
    if (!Call)
      continue;
    Call->eraseFromParent();
  }
  Function->eraseFromParent();
}

int getConstantIntMDValue(Metadata *MD) {
  ConstantInt *CI = mdconst::extract<ConstantInt>(MD);
  return CI->getLimitedValue();
}

llvm::Metadata *createConstantIntMD(llvm::LLVMContext &C, int32_t Val) {
  IntegerType *I32Type = IntegerType::get(C, 32);
  return ConstantAsMetadata::get(ConstantInt::get(I32Type, Val));
}

llvm::DISubprogram *mimicDISubprogram(llvm::DISubprogram *Old,
                                      const llvm::StringRef &NewFuncName,
                                      llvm::DIScope *Scope) {

  return DISubprogram::getDistinct(
      Old->getContext(), Old->getScope(), NewFuncName, "", Old->getFile(),
      Old->getLine(), Old->getType(), Old->getScopeLine(),
      Old->getContainingType(), Old->getVirtualIndex(),
      Old->getThisAdjustment(), Old->getFlags(), Old->getSPFlags(),
      Old->getUnit(), Old->getTemplateParams(), Old->getDeclaration());
}

bool isLocalMemFunctionArg(llvm::Function *F, unsigned ArgIndex) {

  MDNode *MD = F->getMetadata("kernel_arg_addr_space");

  if (MD == nullptr || MD->getNumOperands() <= ArgIndex)
    return false;
  else
    return getConstantIntMDValue(MD->getOperand(ArgIndex)) ==
           SPIR_ADDRESS_SPACE_LOCAL;
}

bool isProgramScopeVariable(GlobalVariable &GVar, unsigned DeviceLocalAS) {

  bool retval = false;

  // no need to handle constants
  if (GVar.isConstant()) {
    retval = false;
    goto END;
  }

  // program-scope variables from direct Clang compilation have external
  // linkage with Target AS numbers
  if (GVar.getLinkage() == GlobalValue::LinkageTypes::ExternalLinkage) {
    retval = true;
    goto END;
  }

#ifdef DEBUG_LLVM_UTILS
  std::cerr << "isProgramScopeVariable: checking variable: " <<
            GVar.getName().str() << "\n";
#endif

  // global variables from SPIR-V have internal linkage with SPIR AS numbers
  if (GVar.getLinkage() == GlobalValue::LinkageTypes::InternalLinkage) {
#ifdef DEBUG_LLVM_UTILS
    std::cerr << "isProgramScopeVariable: checking internal linkage\n";
#endif
    PointerType *GVarT = GVar.getType();
    assert(GVarT != nullptr);
    unsigned AddrSpace = GVarT->getAddressSpace();

    if (AddrSpace == SPIR_ADDRESS_SPACE_GLOBAL) {
#ifdef DEBUG_LLVM_UTILS
      std::cerr << "isProgramScopeVariable: AS = SPIR Global AS\n";
#endif
      if (!GVar.hasName()) {
        GVar.setName("__anonymous_gvar");
      }
      retval = true;
    }

    // variables in local AS cannot have initializer (OpenCL standard).
    // for CPU target, Local AS = Global AS = 0, and
    // function-scope variables ("static global X = {...};")
    // must be recognized as program-scope variables
    if (GVar.hasInitializer()) {
      Constant *C = GVar.getInitializer();
      bool isUndef = isa<UndefValue>(C);
      if (AddrSpace == DeviceLocalAS && !isUndef) {
#ifdef DEBUG_LLVM_UTILS
        std::cerr << "isProgramScopeVariable: AS = device's Local AS && "
                     "isUndef == false\n";
#endif
        if (!GVar.hasName()) {
          GVar.setName("__anonymous_gvar");
        }
        retval = true;
      }
    }
  }

END:
#ifdef DEBUG_LLVM_UTILS
  std::cerr << "isProgramScopeVariable: \n"
            << "Variable: " << GVar.getName().str()
            << " is ProgramScope variable: " << retval << "\n";

#endif
  return retval;
}

void setFuncArgAddressSpaceMD(llvm::Function *F, unsigned ArgIndex,
                              unsigned AS) {

  unsigned MDKind = F->getContext().getMDKindID("kernel_arg_addr_space");
  MDNode *OldMD = F->getMetadata(MDKind);

  assert(OldMD == nullptr || OldMD->getNumOperands() >= ArgIndex);

  LLVMContext &C = F->getContext();

  llvm::SmallVector<llvm::Metadata *, 8> AddressQuals;
  for (unsigned i = 0; i < ArgIndex; ++i) {
    AddressQuals.push_back(createConstantIntMD(
        C, OldMD != nullptr ? getConstantIntMDValue(OldMD->getOperand(i))
                            : SPIR_ADDRESS_SPACE_GLOBAL));
  }
  AddressQuals.push_back(createConstantIntMD(C, AS));
  F->setMetadata(MDKind, MDNode::get(F->getContext(), AddressQuals));
}

// Returns true in case the given function is a kernel that
// should be processed by the kernel compiler.
bool isKernelToProcess(const llvm::Function &F) {

  const Module *m = F.getParent();

  if (F.getMetadata("kernel_arg_access_qual") &&
      F.getMetadata("pocl_generated") == nullptr)
    return true;

  if (F.isDeclaration())
    return false;
  if (!F.hasName())
    return false;
  if (F.getName().startswith("@llvm"))
    return false;

  NamedMDNode *kernels = m->getNamedMetadata("opencl.kernels");
  if (kernels == NULL) {

    std::string KernelName;
    bool HasMeta = getModuleStringMetadata(*m, "KernelName", KernelName);

    if (HasMeta && KernelName.size() && F.getName().str() == KernelName)
      return true;

    return false;
  }

  for (unsigned i = 0, e = kernels->getNumOperands(); i != e; ++i) {
    if (kernels->getOperand(i)->getOperand(0) == NULL)
      continue; // globaldce might have removed uncalled kernels
    Function *k = cast<Function>(
        dyn_cast<ValueAsMetadata>(kernels->getOperand(i)->getOperand(0))
            ->getValue());
    if (&F == k)
      return true;
  }

  return false;
}

// Returns true in case the given function is a kernel with work-group
// barriers inside it.
bool hasWorkgroupBarriers(const llvm::Function &F) {
  for (llvm::Function::const_iterator i = F.begin(), e = F.end(); i != e; ++i) {
    const llvm::BasicBlock *bb = &*i;
    if (pocl::Barrier::hasBarrier(bb)) {

      // Ignore the implicit entry and exit barriers.
      if (pocl::Barrier::hasOnlyBarrier(bb) && bb == &F.getEntryBlock())
        continue;

      if (pocl::Barrier::hasOnlyBarrier(bb) &&
          bb->getTerminator()->getNumSuccessors() == 0)
        continue;

      return true;
    }
  }
  return false;
}

constexpr unsigned NumWorkgroupVariables = 18;
const char *WorkgroupVariablesArray[NumWorkgroupVariables+1] = {"_local_id_x",
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
                                    "_pocl_sub_group_size",
                                    PoclGVarBufferName,
                                    NULL};

const std::vector<std::string>
    WorkgroupVariablesVector(WorkgroupVariablesArray,
                             WorkgroupVariablesArray+NumWorkgroupVariables);


#if LLVM_MAJOR >= MIN_LLVM_NEW_PASSMANAGER
// register all PoCL analyses & passes with an LLVM PassBuilder instance,
// so that it can parse them from string representation
void registerPassBuilderPasses(llvm::PassBuilder &PB) {
  AllocasToEntry::registerWithPB(PB);
  AutomaticLocals::registerWithPB(PB);
  BarrierTailReplication::registerWithPB(PB);
  BreakConstantGEPs::registerWithPB(PB);
  CanonicalizeBarriers::registerWithPB(PB);
  FlattenAll::registerWithPB(PB);
  FlattenBarrierSubs::registerWithPB(PB);
  FlattenGlobals::registerWithPB(PB);
  HandleSamplerInitialization::registerWithPB(PB);
  ImplicitConditionalBarriers::registerWithPB(PB);
  ImplicitLoopBarriers::registerWithPB(PB);
  InlineKernels::registerWithPB(PB);
  IsolateRegions::registerWithPB(PB);
  LoopBarriers::registerWithPB(PB);
  FixMinVecSize::registerWithPB(PB);
  OptimizeWorkItemFuncCalls::registerWithPB(PB);
  PHIsToAllocas::registerWithPB(PB);
  RemoveBarrierCalls::registerWithPB(PB);
  RemoveOptnoneFromWIFunc::registerWithPB(PB);
  SubCFGFormation::registerWithPB(PB);
  Workgroup::registerWithPB(PB);
  WorkitemLoops::registerWithPB(PB);
  WorkitemReplication::registerWithPB(PB);
  PoCLCFGPrinter::registerWithPB(PB);
}

void registerFunctionAnalyses(llvm::PassBuilder &PB) {
  VariableUniformityAnalysis::registerWithPB(PB);
  WorkitemHandlerChooser::registerWithPB(PB);
  WorkItemAliasAnalysis::registerWithPB(PB);
}
#endif

} // namespace pocl
