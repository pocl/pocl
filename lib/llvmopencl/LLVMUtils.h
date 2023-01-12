// Header for LLVMUtils, useful common LLVM-related functionality.
//
// Copyright (c) 2013-2019 Pekka Jääskeläinen
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
//#include "_libclang_versions_checks.h"

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

inline bool
isAutomaticLocal(const std::string &FuncName, llvm::GlobalVariable &Var) {
  // Without the fake address space IDs, there is no reliable way to figure out
  // if the address space is local from the bitcode. We could check its AS
  // against the device's local address space id, but for now lets rely on the
  // naming convention only. Only relying on the naming convention has the problem
  // that LLVM can move private const arrays to the global space which make
  // them look like local arrays (see Github Issue 445). This should be properly
  // fixed in Clang side with e.g. a naming convention for the local arrays to
  // detect them robstly without having logical address space info in the IR.
  if (!llvm::isa<llvm::PointerType>(Var.getType()) || Var.isConstant())
    return false;
  if (Var.getName().startswith(FuncName + "."))
    return true;

  // handle SPIR local AS (3)
  if (Var.getParent() && Var.getParent()->getNamedMetadata("spirv.Source") &&
      (Var.getType()->getAddressSpace() == SPIR_ADDRESS_SPACE_LOCAL))
    return true;

  return false;
}

// Checks if the given argument of Func is a local buffer.
bool isLocalMemFunctionArg(llvm::Function *Func, unsigned ArgIndex);

// Sets the address space metadata of the given function argument.
// Note: The address space ids must be SPIR ids. If it encounters
// argument indices without address space ids in the list, sets
// them to globals.
void setFuncArgAddressSpaceMD(llvm::Function *Func, unsigned ArgIndex,
                              unsigned AS);

llvm::Metadata *createConstantIntMD(llvm::LLVMContext &C, int32_t Val);
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

#endif
