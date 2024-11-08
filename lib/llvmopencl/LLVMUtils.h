// Header for LLVMUtils, useful common LLVM-related functionality.
//
// Copyright (c) 2013-2019 Pekka Jääskeläinen
//               2023 Pekka Jääskeläinen / Intel Finland Oy
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef POCL_LLVM_UTILS_H
#define POCL_LLVM_UTILS_H

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Transforms/Utils/Cloning.h> // for CloneFunctionIntoAbs
POP_COMPILER_DIAGS

#include <map>
#include <string>

#include "pocl.h"
#include "pocl_spir.h"


namespace llvm {
    class Module;
    class Function;
    class GlobalVariable;
}

namespace pocl {

typedef std::map<llvm::Function*, llvm::Function*> FunctionMapping;

constexpr unsigned NumWorkgroupVariables = 21;
extern const char *WorkgroupVariablesArray[];
extern const std::vector<std::string> WorkgroupVariablesVector;
// work-item function names
extern const std::vector<std::string> WIFuncNameVec;
// functions that should not be inlined because they're required by other
// passes: WI funcs are required by FlattenGlobals pass Printf funcs
// (pocl_printf_alloc etc) must exist until Workgroup pass
constexpr unsigned NumDIFuncNames = 13;
extern const char *DIFuncNameArray[];
extern const std::vector<std::string> DIFuncNameVec;

void regenerate_kernel_metadata(llvm::Module &M, FunctionMapping &kernels);

void breakConstantExpressions(llvm::Value *Val, llvm::Function *Func);

// Remove a function from a module, along with all callsites.
POCL_EXPORT
void eraseFunctionAndCallers(llvm::Function *Function);

bool isAutomaticLocal(llvm::Function *F, llvm::GlobalVariable &Var);

/// Returns true if \param Call is a work-item function call that can be
/// directly analyzed and expanded by the kernel compiler.
bool isCompilerExpandableWIFunctionCall(const llvm::CallInst &Call);

POCL_EXPORT
bool isGVarUsedByFunction(llvm::GlobalVariable *GVar, llvm::Function *F);

// Checks if the given argument of Func is a local buffer.
bool isLocalMemFunctionArg(llvm::Function *Func, unsigned ArgIndex);

// determines if GVar is OpenCL program-scope variable
// if it has empty name, sets it to __anonymous_global_as.XYZ
bool isProgramScopeVariable(llvm::GlobalVariable &GVar, unsigned DeviceLocalAS);

// Sets the address space metadata of the given function argument.
// Note: The address space ids must be SPIR ids. If it encounters
// argument indices without address space ids in the list, sets
// them to globals.
void setFuncArgAddressSpaceMD(llvm::Function *Func, unsigned ArgIndex,
                              unsigned AS);

bool hasWorkgroupBarriers(const llvm::Function &F);

POCL_EXPORT
bool areAllGvarsDefined(llvm::Module *Program, std::string &log,
                        std::set<llvm::GlobalVariable *> &GVarSet,
                        unsigned DeviceLocalAS);

POCL_EXPORT
size_t calculateGVarOffsetsSizes(
    const llvm::DataLayout &DL,
    std::map<llvm::GlobalVariable *, uint64_t> &GVarOffsets,
    std::set<llvm::GlobalVariable *> &GVarSet);

POCL_EXPORT
bool isKernelToProcess(const llvm::Function &F);

llvm::Metadata *createConstantIntMD(llvm::LLVMContext &C, int32_t Val);

// Fixes switch statements that have a default case that is a simple
// unreachable instruction. LLVM does this as "optimization", however it breaks
// the (post) dominator-tree analysis, because the unreachable instruction
// creates an additional function exit path. PoCL's ImplicitConditionalBarriers
// pass then erroneously adds barriers because it thinks some basic blocks are
// inside conditional blocks, when they're not.
// TODO this is fragile, LLVM can introduce more optimizations that create
// unreachable blocks. However I couldn't find any working way to make PDT
// ignore blocks with an unreachable inst.
void removeUnreachableSwitchCases(llvm::Function &F);

void markFunctionAlwaysInline(llvm::Function *F);

/**
 * \brief Clones a DISubprogram with changed function name and scope.
 *
 * \param [in] Old The DISubprogram to clone.
 * \param [in] NewFuncName A new function name.
 * \param [in] Scope New scope.
 * \returns a new DISubprogram.
 *
 */
llvm::DISubprogram *mimicDISubprogram(llvm::DISubprogram *Old,
                                      const llvm::StringRef &NewFuncName,
                                      llvm::DIScope *Scope = nullptr);

void registerPassBuilderPasses(llvm::PassBuilder &PB);

void registerFunctionAnalyses(llvm::PassBuilder &PB);

llvm::Type *SizeT(llvm::Module *M);

} // namespace pocl

template <typename VectorT>
void CloneFunctionIntoAbs(llvm::Function *NewFunc,
                          const llvm::Function *OldFunc,
                          llvm::ValueToValueMapTy &VMap, VectorT &Returns,
                          bool sameModule = true,
                          const char *NameSuffix = "",
                          llvm::ClonedCodeInfo *CodeInfo = nullptr,
                          llvm::ValueMapTypeRemapper *TypeMapper = nullptr,
                          llvm::ValueMaterializer *Materializer = nullptr) {

                    // ClonedModule DifferentModule LocalChangesOnly
                    // GlobalChanges
  CloneFunctionInto(NewFunc, OldFunc, VMap,
                    (sameModule ? llvm::CloneFunctionChangeType::GlobalChanges
                                : llvm::CloneFunctionChangeType::DifferentModule),
                    Returns, NameSuffix, CodeInfo, TypeMapper, Materializer);
}

#if LLVM_MAJOR < 15
// Globals
#define getValueType getType()->getElementType
#endif /* LLVM_OPAQUE_POINTERS */

// macros for registering LLVM passes & analyses with old & new PM

#define REGISTER_NEW_FPASS(PNAME, PCLASS, PDESC)                               \
  void PCLASS::registerWithPB(llvm::PassBuilder &PB) {                         \
    PB.registerPipelineParsingCallback(                                        \
        [](::llvm::StringRef Name, ::llvm::FunctionPassManager &FPM,           \
           llvm::ArrayRef<::llvm::PassBuilder::PipelineElement>) {             \
          if (Name == PNAME) {                                                 \
            FPM.addPass(PCLASS());                                             \
            return true;                                                       \
          }                                                                    \
          return false;                                                        \
        });                                                                    \
  }

#define REGISTER_NEW_MPASS(PNAME, PCLASS, PDESC)                               \
  void PCLASS::registerWithPB(llvm::PassBuilder &PB) {                         \
    PB.registerPipelineParsingCallback(                                        \
        [](::llvm::StringRef Name, ::llvm::ModulePassManager &MPM,             \
           llvm::ArrayRef<::llvm::PassBuilder::PipelineElement>) {             \
          if (Name == PNAME) {                                                 \
            MPM.addPass(PCLASS());                                             \
            return true;                                                       \
          } else                                                               \
            return false;                                                      \
        });                                                                    \
  }

#define REGISTER_NEW_LPASS(PNAME, PCLASS, PDESC)                               \
  void PCLASS::registerWithPB(llvm::PassBuilder &PB) {                         \
    PB.registerPipelineParsingCallback(                                        \
        [](::llvm::StringRef Name, ::llvm::LoopPassManager &LPM,               \
           ::llvm::ArrayRef<::llvm::PassBuilder::PipelineElement>) {           \
          if (Name == PNAME) {                                                 \
            LPM.addPass(PCLASS());                                             \
            return true;                                                       \
          } else                                                               \
            return false;                                                      \
        });                                                                    \
  }

#define REGISTER_NEW_FANALYSIS(PNAME, PCLASS, PDESC)                           \
  void PCLASS::registerWithPB(llvm::PassBuilder &PB) {                         \
    PB.registerPipelineParsingCallback(                                        \
        [](::llvm::StringRef Name, ::llvm::FunctionPassManager &FPM,           \
           llvm::ArrayRef<::llvm::PassBuilder::PipelineElement>) {             \
          if (Name == "require<" PNAME ">") {                                  \
            FPM.addPass(RequireAnalysisPass<PCLASS, llvm::Function>());        \
            return true;                                                       \
          } else                                                               \
            return false;                                                      \
        });                                                                    \
    PB.registerAnalysisRegistrationCallback(                                   \
        [](::llvm::FunctionAnalysisManager &FAM) {                             \
          FAM.registerPass([&] { return PCLASS(); });                          \
        });                                                                    \
  }

#define REGISTER_OLD_FPASS(PNAME, PCLASS, PDESC)                               \
  static llvm::RegisterPass<PCLASS> X(PNAME, PDESC)

#define REGISTER_OLD_MPASS(PNAME, PCLASS, PDESC)                               \
  static llvm::RegisterPass<PCLASS> X(PNAME, PDESC)

#define REGISTER_OLD_LPASS(PNAME, PCLASS, PDESC)                               \
  static llvm::RegisterPass<PCLASS> X(PNAME, PDESC)

#define REGISTER_OLD_FANALYSIS(PNAME, PCLASS, PDESC)                          \
  static llvm::RegisterPass<PCLASS> X (PNAME, PDESC)

#if LLVM_MAJOR < 16
// Avoid the deprecation warning with later LLVMs.
#define starts_with startswith
#endif

#endif
