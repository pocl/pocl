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

#ifndef _POCL_LLVM_UTILS_H
#define _POCL_LLVM_UTILS_H

#include <map>
#include <string>

#include "pocl.h"
#include "pocl_spir.h"
// #include "_libclang_versions_checks.h"

#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h> // for CloneFunctionIntoAbs

namespace llvm {
    class Module;
    class Function;
    class GlobalVariable;
}

namespace pocl {

typedef std::map<llvm::Function*, llvm::Function*> FunctionMapping;

void regenerate_kernel_metadata(llvm::Module &M, FunctionMapping &kernels);

void breakConstantExpressions(llvm::Value *Val, llvm::Function *Func);

// Remove a function from a module, along with all callsites.
POCL_EXPORT
void eraseFunctionAndCallers(llvm::Function *Function);

bool isAutomaticLocal(llvm::Function *F, llvm::GlobalVariable &Var);

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

llvm::Metadata *createConstantIntMD(llvm::LLVMContext &C, int32_t Val);

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
}

template <typename VectorT>
void CloneFunctionIntoAbs(llvm::Function *NewFunc,
                          const llvm::Function *OldFunc,
                          llvm::ValueToValueMapTy &VMap, VectorT &Returns,
                          bool sameModule = true,
                          const char *NameSuffix = "",
                          llvm::ClonedCodeInfo *CodeInfo = nullptr,
                          llvm::ValueMapTypeRemapper *TypeMapper = nullptr,
                          llvm::ValueMaterializer *Materializer = nullptr) {


#ifdef LLVM_OLDER_THAN_13_0
  CloneFunctionInto(NewFunc, OldFunc, VMap, true,
                    Returns, NameSuffix, CodeInfo, TypeMapper, Materializer);
#else
                    // ClonedModule DifferentModule LocalChangesOnly
                    // GlobalChanges
  CloneFunctionInto(NewFunc, OldFunc, VMap,
                    (sameModule ? llvm::CloneFunctionChangeType::GlobalChanges
                                : llvm::CloneFunctionChangeType::DifferentModule),
                    Returns, NameSuffix, CodeInfo, TypeMapper, Materializer);
#endif
}

#ifdef LLVM_OLDER_THAN_15_0
// Globals
#define getValueType getType()->getElementType
#endif /* LLVM_OPAQUE_POINTERS */

#ifdef LLVM_OLDER_THAN_14_0
#define LLVMBuildGEP2(A, B, C, D, E, F) LLVMBuildGEP(A, C, D, E, F)
#endif

#ifdef LLVM_OLDER_THAN_11_0
#define CBS_NO_PHIS_IN_SPLIT
#endif

#endif
