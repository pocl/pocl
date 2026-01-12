/* AffineParallelToAffineFor.cpp

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

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "pocl/Transforms/Passes.hh"

namespace {
#define GEN_PASS_DEF_CONVERTAFFINEPARALLELTOAFFINEFOR
#include "pocl/Transforms/Passes.h.inc"
} // namespace

struct AffineParallelToAffineForPattern
    : public mlir::OpRewritePattern<mlir::affine::AffineParallelOp> {
  using mlir::OpRewritePattern<
      mlir::affine::AffineParallelOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::affine::AffineParallelOp ParallelOp,
                  mlir::PatternRewriter &Rewriter) const override {
    // Get the location of the original operation
    mlir::Location Loc = ParallelOp.getLoc();
    Rewriter.setInsertionPoint(ParallelOp);
    // Create a new AffineForOp for each dimension of the AffineParallelOp
    mlir::SmallVector<mlir::Value, 8> Ivs;
    for (unsigned I = 0, E = ParallelOp.getNumDims(); I < E; ++I) {
      auto LbMap = ParallelOp.getLowerBoundMap(I);
      auto UbMap = ParallelOp.getUpperBoundMap(I);
      auto LbOperands = ParallelOp.getLowerBoundsOperands();
      auto UbOperands = ParallelOp.getUpperBoundsOperands();
      auto Step = ParallelOp.getSteps()[I];

      auto ForOp = mlir::affine::AffineForOp::create(
          Rewriter, Loc, LbOperands, LbMap, UbOperands, UbMap, Step);

      Ivs.push_back(ForOp.getInductionVar());
      // Move the insertion point inside the newly created loop
      Rewriter.setInsertionPointToStart(ForOp.getBody());
    }

    // Clone the body of the original AffineParallelOp into the innermost
    // AffineForOp
    mlir::Block &EntryBlock = ParallelOp.getRegion().front();
    Rewriter.mergeBlocks(&EntryBlock, Rewriter.getInsertionBlock(), Ivs);

    // Remove the automatically created affine.yield in the innermost loop
    auto &InnermostLoopBody = Rewriter.getInsertionBlock()->getOperations();
    if (!InnermostLoopBody.empty() &&
        mlir::isa<mlir::affine::AffineYieldOp>(InnermostLoopBody.front())) {
      Rewriter.eraseOp(&InnermostLoopBody.front());
    }
    // Erase the original AffineParallelOp
    Rewriter.eraseOp(ParallelOp);

    return mlir::success();
  }
};

namespace {
// Register the pattern in a pass
struct ConvertAffineParallelToAffineFor
    : public impl::ConvertAffineParallelToAffineForBase<
          ConvertAffineParallelToAffineFor> {
  void runOnOperation() override {
    mlir::MLIRContext *Context = &getContext();
    mlir::RewritePatternSet Patterns(Context);
    Patterns.add<AffineParallelToAffineForPattern>(Context);
    (void)applyPatternsGreedily(getOperation(), std::move(Patterns));
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
mlir::pocl::createConvertAffineParallelToAffineForPass() {
  return std::make_unique<ConvertAffineParallelToAffineFor>();
}
