/* Linker.cpp - Lightweight bitcode linker to link MLIR bytecode.

   Copyright (c) 2025 Topi Leppänen / Tampere University

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include <llvm/ADT/StringSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include "pocl/Transforms/Passes.hh"
#include "llvm/ADT/STLExtras.h"

#include "pocl_debug.h"
#include "pocl_llvm_api.h"

namespace {
#define GEN_PASS_DEF_LINKER
#include "pocl/Transforms/Passes.h.inc"
} // namespace

namespace {
struct Linker : public impl::LinkerBase<Linker> {

  Linker(mlir::ModuleOp BuiltinLib) : BuiltinModule(BuiltinLib) {}

  /* Copy function F and all the functions in its call graph
   * that are defined in 'from', into 'to'
   */
  static int copyMlirFuncCallgraph(const llvm::StringRef FuncName,
                                   mlir::ModuleOp From, mlir::ModuleOp To,
                                   std::vector<std::string> &NestedFuncs) {

    // First check if the function already exists in 'to' in case it has been
    // included twice.
    bool Exists = llvm::find(NestedFuncs, FuncName.str()) != NestedFuncs.end();
    if (Exists) {
      return 0;
    }

    llvm::StringSet<> Callees;
    mlir::IRRewriter Builder(To.front().getContext());
    mlir::Region *FromRegion = NULL;
    auto LibMods = From.getRegion().getOps<mlir::ModuleOp>();
    for (auto LibMod : LibMods) {
      auto &TmpRegion = LibMod.getRegion();
      auto FuncOpsOfKernelLib =
          TmpRegion.getBlocks().front().getOps<mlir::func::FuncOp>();
      for (auto Func : FuncOpsOfKernelLib) {
        if (Func.getName() == FuncName && !Func.isDeclaration()) {
          FromRegion = Func->getParentRegion();
          mlir::func::FuncOp FuncToReplace = NULL;
          for (auto Functoreplace :
               To.getRegion().getOps<mlir::func::FuncOp>()) {
            if (Functoreplace.getName() == FuncName) {
              FuncToReplace = Functoreplace;
              break;
            }
          }
          if (FuncToReplace == NULL) {
            return -1;
          }
          Builder.setInsertionPoint(FuncToReplace);
          Builder.cloneRegionBefore(Func.getRegion(), FuncToReplace.getRegion(),
                                    FuncToReplace.getRegion().begin());
          break;
        }
      }
    }
    NestedFuncs.push_back(FuncName.str());
    if (FromRegion == NULL) {
      return -1;
    }
    // Copy global variables
    auto ExistingGops = To.getRegion().getOps<mlir::memref::GlobalOp>();
    Builder.setInsertionPoint(&To.getRegion().front().front());
    FromRegion->walk([&](mlir::memref::GlobalOp Globalop) {
      bool ExistsAlready = false;
      for (auto ExistingGop : ExistingGops) {
        if (ExistingGop.getSymName() == Globalop.getSymName()) {
          ExistsAlready = true;
        }
      }
      if (!ExistsAlready) {
        Builder.clone(*Globalop);
      }
    });

    // Copy all the undefined functions, then call recursively until all them
    // are defined
    FromRegion->walk([&](mlir::func::FuncOp Func) {
      std::string FuncName = Func.getName().str();
      if (Func.isDeclaration()) {
        if (llvm::find(NestedFuncs, FuncName) == NestedFuncs.end()) {
          Builder.setInsertionPoint(&To.getRegion().front().back());
          Builder.clone(
              *Func); // Copy the empty function body to be the copying target
          copyMlirFuncCallgraph(Func.getName(), From, To, NestedFuncs);
        }
      }
    });

    // Delete any leftover declarations, these are only copies incase some
    // built-in function was included twice
    To.walk([&](mlir::func::FuncOp Func) {
      if (Func.isDeclaration() && FuncName == Func.getName()) {
        Builder.eraseOp(Func);
      }
    });

    return 0;
  }

  int mlirLink(mlir::ModuleOp &ProgramMod, mlir::ModuleOp &Lib) {

    llvm::StringSet<> DeclaredFunctions;
    ProgramMod.walk([&](mlir::func::FuncOp Func) {
      if (Func.isDeclaration()) {
        DeclaredFunctions.insert(Func.getName());
      }
    });

    std::vector<std::string> NestedFuncs;
    bool FoundAllUndefined = true;
    llvm::StringSet<>::iterator Di, De;
    for (Di = DeclaredFunctions.begin(), De = DeclaredFunctions.end(); Di != De;
         Di++) {
      llvm::StringRef R = Di->getKey();
      if (copyMlirFuncCallgraph(R, Lib, ProgramMod, NestedFuncs)) {
        FoundAllUndefined = false;
      }
    }
    if (!FoundAllUndefined)
      return 1;

    // sort all the kernel functions to end of the region
    auto &Block = ProgramMod.getRegion().front();

    auto &OldOps = Block.getOperations();
    auto Builder = mlir::IRRewriter(ProgramMod.getContext());
    std::vector<mlir::func::FuncOp> OpsToErase;
    std::vector<std::string> ClonedOps;
    for (auto &OldOp : OldOps) {
      if (auto Func = mlir::dyn_cast<mlir::func::FuncOp>(OldOp)) {
        bool IsKernel =
            Func->hasAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName());
        if (IsKernel &&
            (llvm::find(ClonedOps, Func.getName().str()) == ClonedOps.end())) {
          // move the kernel FuncOps to the very end
          Builder.setInsertionPointToEnd(&Block);
          Builder.clone(OldOp);
          OpsToErase.push_back(Func);
          ClonedOps.push_back(Func.getName().str());
        }
      }
    }
    for (auto &Op : OpsToErase) {
      Builder.eraseOp(Op);
    }

    return 0;
  }

  void runOnOperation() override {
    if (!BuiltinModule) {
      llvm::errs() << "Error: This pass cannot be run from CLI, as it requires "
                      "an external BuiltinLib as an ModuleOp argument\n";
      signalPassFailure();
      return;
    }
    mlir::ModuleOp ProgramMod = getOperation();
    mlirLink(ProgramMod, BuiltinModule);
  }

private:
  mlir::ModuleOp BuiltinModule = mlir::ModuleOp();
};
} // namespace

std::unique_ptr<mlir::Pass>
mlir::pocl::createLinkerPass(mlir::ModuleOp BuiltinLib) {
  return std::make_unique<Linker>(BuiltinLib);
}
