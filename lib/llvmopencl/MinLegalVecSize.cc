// LLVM module pass. Sets AlwaysInline on functions that have arguments whose size
// exceeds the largest available register size of the Target
//
// Copyright (c) 2023 Michal Babej / Intel Finland Oy
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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

#include "LLVMUtils.h"
#include "MinLegalVecSize.hh"

#include "pocl_llvm_api.h"
POP_COMPILER_DIAGS

#include <iostream>
#include <set>
#include <string>

//#define DEBUG_VEC_SIZE

#define PASS_NAME "fix-min-legal-vec-size"
#define PASS_CLASS pocl::FixMinVecSize
#define PASS_DESC "Module pass stuff"

namespace pocl {

using namespace llvm;

#if LLVM_MAJOR < 14
static bool fixMinVecSize(Module &M) { return false; }
#else

static void setFunctionMinLegalVecAttr(llvm::Function *F, uint64_t Value) {
  Attribute replacementAttr = Attribute::get(
      F->getContext(), "min-legal-vector-width", std::to_string(Value));

  F->setAttributes(
      F->getAttributes()
          .removeFnAttribute(F->getContext(), "min-legal-vector-width")
          .addFnAttribute(F->getContext(), replacementAttr));
}

/* some functions have an incorrect value in attribute min-legal-vec-size,
 * find out the true value here, fix & return it */
static uint64_t getMinVecSizeFromPrototype(llvm::Function *F,
                                           const std::string &Spaces) {

  uint64_t ReportedSize = 0;

  // get the value from F Attribute
  llvm::Attribute Attr = F->getFnAttribute("min-legal-vector-width");
  if (Attr.isValid()) {
    Attr.getValueAsString().getAsInteger(0, ReportedSize);
  }

  uint64_t Max = ReportedSize;
  const DataLayout &DL = F->getParent()->getDataLayout();

  if (F->getReturnType()->isSized()) {
    TypeSize RetTySize = DL.getTypeAllocSizeInBits(F->getReturnType());
    assert(RetTySize.isScalable() == false);
    Max = std::max(Max, RetTySize.getFixedValue());
  }

  for (const llvm::Argument &A : F->args()) {
    Type *T = nullptr;
    if (A.hasByValAttr() && A.getType()->isPointerTy()) {
      T = A.getParamByValType();
    } else {
      T = A.getType();
    }
    if (!T->isSized())
      continue;
    TypeSize ArgTySize = DL.getTypeAllocSizeInBits(T);
    Max = std::max(Max, ArgTySize.getFixedValue());
  }

  if (Max > ReportedSize) {
#ifdef DEBUG_VEC_SIZE
    std::cerr << Spaces << "Function " << F->getName().str()
              << " has wrong min-legal-vec-size, "
                 "reported: "
              << ReportedSize << ", actual: " << Max << "\n";
#endif
    setFunctionMinLegalVecAttr(F, Max);
  }

  return Max;
}

/* recursively look & fix min-legal-vec-size  */
static uint64_t getAndFixLargestVecSize(llvm::Function *F, unsigned Justify) {
  SmallVector<llvm::Function *, 8> Calls;
  std::string Spaces(Justify, ' ');

  uint64_t ReportedSize = getMinVecSizeFromPrototype(F, Spaces);

  uint64_t MinVecSize = ReportedSize;

  // find called functions
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {
      Instruction *Instr = dyn_cast<Instruction>(BI);

      if (!llvm::isa<CallInst>(Instr))
        continue;

      CallInst *CallInstr = dyn_cast<CallInst>(Instr);

      llvm::Function *Callee = CallInstr->getCalledFunction();
      if (Callee == nullptr)
        continue;

      if (Callee->hasName() && Callee->getName().startswith("llvm."))
        continue;

      Calls.push_back(Callee);
    }
  }

#ifdef DEBUG_VEC_SIZE
  std::cerr << Spaces << "Num Callees: " << Calls.size() << "\n";
#endif

  for (llvm::Function *Callee : Calls) {
    if (Callee->hasName()) {
      std::string Name = Callee->getName().str();
#ifdef DEBUG_VEC_SIZE
      std::cerr << Spaces << "Found callee: " << Name << "\n";
#endif
    }
    uint64_t Width = getAndFixLargestVecSize(Callee, Justify + 2);
    if (Width > MinVecSize) {
      MinVecSize = Width;
#ifdef DEBUG_VEC_SIZE
      std::cerr << Spaces << "New min vec size: " << MinVecSize << "\n";
#endif
    }
  }

  if (MinVecSize > ReportedSize) {
    setFunctionMinLegalVecAttr(F, MinVecSize);
#ifdef DEBUG_VEC_SIZE
    std::cerr << Spaces << "Fixed to: " << MinVecSize << "\n";
#endif
  }

  return MinVecSize;
}

// find all kernels with SPIR_KERNEL CC and recursively fix their
// "min-legal-vector-width" attributes to the correct value. If any
// called function has has "min-legal-vector-width" larger than
// the device's native vector width size, also set the AlwaysInline
// attribute on that function
// the kernels htemselves are
static bool fixMinVecSize(Module &M) {

  unsigned long DeviceNativeVectorWidth = 0;
  if (!getModuleIntMetadata(M, "device_native_vec_width",
                            DeviceNativeVectorWidth))
    return false;

  // fix the "min-legal-vector-width" attr recursively
  for (llvm::Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    llvm::Function *F = &*i;
    if (F->isDeclaration())
      continue;
    if (F->hasName() && F->getName().startswith("llvm."))
      continue;

    // AttributeSet Attrs;
    if (pocl::isKernelToProcess(*F) &&
        (F->getCallingConv() == llvm::CallingConv::SPIR_KERNEL)) {
#ifdef DEBUG_VEC_SIZE
      std::cerr << "processing kernel: " << F->getName().str() << "\n";
#endif
      llvm::Attribute Attr = F->getFnAttribute("min-legal-vector-width");
      if (Attr.isValid()) {
        continue;
      }

      getAndFixLargestVecSize(F, 0);

      Attr = F->getFnAttribute("min-legal-vector-width");
      uint64_t NewWidth = 0;
      Attr.getValueAsString().getAsInteger(0, NewWidth);
#ifdef DEBUG_VEC_SIZE
      std::cerr << "Set kernel attr min-legal-vector-width to : " << NewWidth
                << "\n";
#endif
    }
  }

  // add the alwaysInline attribute where required
  for (llvm::Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    llvm::Function *F = &*i;
    if (F->isDeclaration())
      continue;
    if (F->hasName() && F->getName().startswith("llvm."))
      continue;
    if (pocl::isKernelToProcess(*F))
      continue;

    llvm::Attribute Attr = F->getFnAttribute("min-legal-vector-width");
    if (!Attr.isValid())
      continue;

    uint64_t FuncMinVecWidth = 0;
    Attr.getValueAsString().getAsInteger(0, FuncMinVecWidth);
#ifdef DEBUG_VEC_SIZE
    std::cerr << "Min Vec Width: " << FuncMinVecWidth << "\n";
#endif
    // force-inline functions larger than max vector size
    if (FuncMinVecWidth <= DeviceNativeVectorWidth)
      continue;

#ifdef DEBUG_VEC_SIZE
    std::cerr << "processing function: " << F->getName().str() << "\n";
    std::cerr << "MinVecSize " << FuncMinVecWidth
              << " larger than NativeVec: " << DeviceNativeVectorWidth
              << ", force-inlining\n";
#endif

    decltype(Attribute::AlwaysInline) replaceThisAttr, replacementAttr;
    replaceThisAttr = Attribute::NoInline;
    replacementAttr = Attribute::AlwaysInline;
    F->setAttributes(F->getAttributes()
                         .removeFnAttribute(F->getContext(), replaceThisAttr)
                         .addFnAttribute(F->getContext(), replacementAttr));
  }

  return false;
}
#endif

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
char FixMinVecSize::ID = 0;

bool FixMinVecSize::runOnModule(Module &M) { return fixMinVecSize(M); }

void FixMinVecSize::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

REGISTER_OLD_MPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#else

llvm::PreservedAnalyses FixMinVecSize::run(llvm::Module &M,
                                           llvm::ModuleAnalysisManager &AM) {
  fixMinVecSize(M);
  return PreservedAnalyses::all();
}

REGISTER_NEW_MPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#endif

} // namespace pocl
