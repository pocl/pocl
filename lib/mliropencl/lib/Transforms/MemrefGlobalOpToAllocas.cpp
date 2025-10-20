/* MemrefGlobalOpToAllocas.cpp

   Copyright (c) 2025 Topi Leppänen / Tampere University

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include "pocl/Transforms/Passes.hh"

namespace {
#define GEN_PASS_DEF_MEMREFGLOBALOPTOALLOCAS
#include "pocl/Transforms/Passes.h.inc"
} // namespace

using namespace mlir;

namespace {
struct MemrefGlobalOpToAllocas
    : public impl::MemrefGlobalOpToAllocasBase<MemrefGlobalOpToAllocas> {

  void runOnOperation() override {

    auto Mod = getOperation();
    auto Opbuilder = mlir::OpBuilder(Mod);

    std::vector<mlir::Operation *> OpsToErase = {};
    Mod.walk([&](mlir::memref::GetGlobalOp GetGlobalOp) {
      auto Func = GetGlobalOp->getParentOfType<mlir::func::FuncOp>();
      Opbuilder.setInsertionPointToStart(&Func.getBody().front());
      auto Alloca = Opbuilder.create<memref::AllocaOp>(GetGlobalOp.getLoc(),
                                                       GetGlobalOp.getType());

      // Replace all uses of getGlobalOp with the new alloca
      GetGlobalOp.replaceAllUsesWith(Alloca.getResult());
      OpsToErase.push_back(GetGlobalOp);
    });
    Mod.walk([&](mlir::memref::GlobalOp GlobalOp) {
      OpsToErase.push_back(GlobalOp);
    });

    for (auto *Op : OpsToErase) {
      Op->erase();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::pocl::createMemrefGlobalOpToAllocasPass() {
  return std::make_unique<MemrefGlobalOpToAllocas>();
}
