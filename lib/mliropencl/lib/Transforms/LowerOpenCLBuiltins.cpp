//===- LowerOpenCLBuiltins.cpp - Lower OpenCL builtins to MLIR dialects ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "pocl/Dialect/Dialect.hh"
#include "pocl/Dialect/PoclOps.hh"

#include "pocl/Transforms/Passes.hh"

#define DEBUG_TYPE "lower-opencl-builtins"
#define REPORT_DEBUG_TYPE DEBUG_TYPE "-report"

namespace mlir {
namespace pocl {
#define GEN_PASS_DEF_LOWEROPENCLBUILTINS
#include "pocl/Transforms/Passes.h.inc"
} // namespace pocl
} // namespace mlir

using namespace mlir;

namespace {

template <typename GPUOpType>
static LogicalResult lowerOpenCL3DimFunc(func::CallOp Op,
                                         PatternRewriter &Rewriter) {
  auto Loc = Op.getLoc();
  auto I64Type = Rewriter.getI64Type();
  auto IndexType = Rewriter.getIndexType();
  std::array<gpu::Dimension, 3> Dims = {gpu::Dimension::x, gpu::Dimension::y,
                                        gpu::Dimension::z};
  Value IndexVal =
      arith::IndexCastOp::create(Rewriter, Loc, IndexType, Op.getOperand(0));
  auto SwitchOp = scf::IndexSwitchOp::create(
      Rewriter, Loc, I64Type, IndexVal,
      /*caseValues=*/ArrayRef<int64_t>{0, 1, 2}, Dims.size());

  // === Default case ===
  {
    Region &CaseRegion = SwitchOp.getDefaultRegion();
    Block *CaseBlock = new Block();
    CaseRegion.push_back(CaseBlock);
    OpBuilder CaseBuilder(CaseBlock, CaseBlock->begin());
    Value C0I64 = arith::ConstantOp::create(CaseBuilder, Loc, I64Type,
                                            Rewriter.getI64IntegerAttr(0));
    scf::YieldOp::create(CaseBuilder, Loc, C0I64);
  }
  for (size_t I = 0; I < Dims.size(); ++I) {
    auto Dim = Dims[I];
    Region &CaseRegion = SwitchOp.getCaseRegions()[I];
    Block *CaseBlock = new Block();
    CaseRegion.push_back(CaseBlock);
    OpBuilder CaseBuilder(CaseBlock, CaseBlock->begin());
    Value TidX = GPUOpType::create(CaseBuilder, Loc, Dim);
    Value TidXCast =
        arith::IndexCastOp::create(CaseBuilder, Loc, I64Type, TidX);
    scf::YieldOp::create(CaseBuilder, Loc, TidXCast);
  }
  Rewriter.replaceOp(Op, SwitchOp);
  return LogicalResult::success();
}

static LogicalResult lowerOpenCLBarrierFunc(func::CallOp Op,
                                            PatternRewriter &Rewriter) {
  Rewriter.replaceOpWithNewOp<gpu::BarrierOp>(Op);
  return LogicalResult::success();
}

static LogicalResult lowerOpenCLWorkDimFunc(func::CallOp Op,
                                            PatternRewriter &Rewriter) {
  Rewriter.replaceOpWithNewOp<pocl::WorkDimOp>(Op);
  return LogicalResult::success();
}

static const llvm::StringMap<
    std::function<LogicalResult(func::CallOp, PatternRewriter &)>>
    FuncMap = {
        {"_Z12get_group_idj",
         [](auto Op, auto &Rewriter) {
           return lowerOpenCL3DimFunc<gpu::BlockIdOp>(Op, Rewriter);
         }},
        {"_Z12get_local_idj",
         [](auto Op, auto &Rewriter) {
           return lowerOpenCL3DimFunc<gpu::ThreadIdOp>(Op, Rewriter);
         }},
        {"_Z14get_local_sizej",
         [](auto Op, auto &Rewriter) {
           return lowerOpenCL3DimFunc<gpu::BlockDimOp>(Op, Rewriter);
         }},
        {"_Z14get_num_groupsj",
         [](auto Op, auto &Rewriter) {
           return lowerOpenCL3DimFunc<gpu::GridDimOp>(Op, Rewriter);
         }},
        {"_Z7barrierj",
         [](auto Op, auto &Rewriter) {
           return lowerOpenCLBarrierFunc(Op, Rewriter);
         }},
        {"_Z12get_work_dimv",
         [](auto Op, auto &Rewriter) {
           return lowerOpenCLWorkDimFunc(Op, Rewriter);
         }},
};

struct ConvertCallOpPattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp CallOp,
                                PatternRewriter &Rewriter) const override {
    auto FuncName = CallOp.getCallee();
    auto It = FuncMap.find(FuncName);
    if (It != FuncMap.end())
      return It->second(CallOp, Rewriter);
    return failure();
  }
};

struct ConvertFuncOpPattern : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp FuncOp,
                                PatternRewriter &Rewriter) const override {
    auto FuncName = FuncOp.getName();
    auto It = FuncMap.find(FuncName);
    if (It != FuncMap.end()) {
      Rewriter.eraseOp(FuncOp);
      return success();
    }
    return failure();
  }
};

class LowerOpenCLBuiltinsPass
    : public pocl::impl::LowerOpenCLBuiltinsBase<LowerOpenCLBuiltinsPass> {

  void runOnOperation() override {
    MLIRContext *Ctx = &getContext();
    RewritePatternSet Patterns(Ctx);
    Patterns.add<ConvertCallOpPattern, ConvertFuncOpPattern>(Ctx);
    if (mlir::failed(
            applyPatternsGreedily(getOperation(), std::move(Patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace pocl {
std::unique_ptr<Pass> createLowerOpenCLBuiltinsPass() {
  return std::make_unique<LowerOpenCLBuiltinsPass>();
}
} // namespace pocl
} // namespace mlir
