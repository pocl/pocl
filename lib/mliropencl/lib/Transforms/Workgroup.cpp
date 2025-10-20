/* Workgroup.cc

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

#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/Argument.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Polygeist/Dialect/Dialect.h>
#include <mlir/Polygeist/Dialect/Ops.h>
#include <mlir/Transforms/InliningUtils.h>

#include "pocl/Dialect/PoclOps.hh"
#include "pocl/Transforms/Passes.hh"

#include "pocl_util.h"

#define POCL_CONTEXT_NUM_ELEMENTS_AS_ARGS (15)
#define GROUP_ID_NUM_DIMENSIONS_AS_ARGS (3)

namespace {
#define GEN_PASS_DEF_WORKGROUP
#include "pocl/Transforms/Passes.h.inc"
} // namespace

namespace {
struct Workgroup : public impl::WorkgroupBase<Workgroup> {

  mlir::affine::AffineParallelOp
  createAffineParallelLoop(mlir::func::FuncOp FuncOp,
                           mlir::OpBuilder &Builder) {
    //  Create a builder with the same context as the block
    auto *Context = Builder.getContext();
    auto Loc = Builder.getUnknownLoc();
    Builder.setInsertionPointToEnd(&FuncOp.getRegion().front());

    std::vector<mlir::Operation *> OpsNotToCopy = {};

    // Create the bounds for the loop
    mlir::AffineMap LbMapX = mlir::AffineMap::getConstantMap(0, Context);
    mlir::AffineMap LbMapY = mlir::AffineMap::getConstantMap(0, Context);
    mlir::AffineMap LbMapZ = mlir::AffineMap::getConstantMap(0, Context);
    mlir::AffineMap UbMapX, UbMapY, UbMapZ;
    mlir::ValueRange UpperBoundValues = mlir::ValueRange();
    if (WGDynamicLocalSize) {
      mlir::AffineExpr D0 = Builder.getAffineDimExpr(0);
      mlir::AffineExpr D1 = Builder.getAffineDimExpr(1);
      mlir::AffineExpr D2 = Builder.getAffineDimExpr(2);
      UbMapX = mlir::AffineMap::get(3, 0, {D0}, Context);
      UbMapY = mlir::AffineMap::get(3, 0, {D1}, Context);
      UbMapZ = mlir::AffineMap::get(3, 0, {D2}, Context);

      auto TotalNumOfArguments = FuncOp.getFunctionBody().getNumArguments();
      int LocalSizeOffsetFromEnd = GROUP_ID_NUM_DIMENSIONS_AS_ARGS +
                                   POCL_CONTEXT_NUM_ELEMENTS_AS_ARGS - 6;
      mlir::Value LocalSizeX = FuncOp.getFunctionBody().getArgument(
          TotalNumOfArguments - LocalSizeOffsetFromEnd + 0);
      auto LocalSizeXCasted = Builder.create<mlir::arith::IndexCastOp>(
          Builder.getUnknownLoc(), Builder.getIndexType(), LocalSizeX);
      mlir::Value LocalSizeY = FuncOp.getFunctionBody().getArgument(
          TotalNumOfArguments - LocalSizeOffsetFromEnd + 1);
      auto LocalSizeYCasted = Builder.create<mlir::arith::IndexCastOp>(
          Builder.getUnknownLoc(), Builder.getIndexType(), LocalSizeY);
      mlir::Value LocalSizeZ = FuncOp.getFunctionBody().getArgument(
          TotalNumOfArguments - LocalSizeOffsetFromEnd + 2);
      auto LocalSizeZCasted = Builder.create<mlir::arith::IndexCastOp>(
          Builder.getUnknownLoc(), Builder.getIndexType(), LocalSizeZ);
      UpperBoundValues = mlir::ValueRange(
          {LocalSizeXCasted, LocalSizeYCasted, LocalSizeZCasted});
      OpsNotToCopy.push_back(LocalSizeXCasted);
      OpsNotToCopy.push_back(LocalSizeYCasted);
      OpsNotToCopy.push_back(LocalSizeZCasted);
    } else {
      UbMapX = mlir::AffineMap::getConstantMap(WGLocalSize[0], Context);
      UbMapY = mlir::AffineMap::getConstantMap(WGLocalSize[1], Context);
      UbMapZ = mlir::AffineMap::getConstantMap(WGLocalSize[2], Context);
    }

    int64_t Step = 1;
    mlir::affine::AffineParallelOp AffineParallelOp =
        Builder.create<mlir::affine::AffineParallelOp>(
            Loc, mlir::TypeRange(),
            llvm::ArrayRef<mlir::arith::AtomicRMWKind>(),
            llvm::ArrayRef<mlir::AffineMap>{LbMapX, LbMapY, LbMapZ},
            mlir::ValueRange(),
            llvm::ArrayRef<mlir::AffineMap>{UbMapX, UbMapY, UbMapZ},
            UpperBoundValues, llvm::ArrayRef<int64_t>{Step, Step, Step});
    OpsNotToCopy.push_back(AffineParallelOp);

    std::vector<mlir::Operation *> ClonedOps;
    std::vector<mlir::Operation *> OrigOpReplaced;
    std::vector<mlir::Block *> BlocksToErase;

    mlir::IRMapping Mapping;

    auto *YieldOp = AffineParallelOp.getRegion().front().getTerminator();
    // Move the operations from the original block to the loop body
    // Clone the operations and keep track of the clones
    int BlockIdx = 0;
    for (auto &Block : FuncOp.getRegion()) {
      mlir::Block *NewBlock;
      if (BlockIdx == 0) {
        NewBlock = &AffineParallelOp.getRegion().front();
      } else {
        mlir::SmallVector<mlir::Location, 4> Arglocs(Block.getNumArguments(),
                                                     Builder.getUnknownLoc());
        NewBlock = Builder.createBlock(&AffineParallelOp.getRegion(),
                                       AffineParallelOp.getRegion().end(),
                                       Block.getArgumentTypes(), Arglocs);
      }
      for (mlir::Operation &OpTmp :
           llvm::make_early_inc_range(Block.getOperations())) {
        auto *Op = &OpTmp;
        if (std::find(OpsNotToCopy.begin(), OpsNotToCopy.end(), Op) !=
            OpsNotToCopy.end())
          continue;
        if (auto AllocaOp = mlir::dyn_cast<mlir::memref::AllocaOp>(Op)) {
          auto Memref = AllocaOp.getMemref();
          auto Memreftype = Memref.getType();
          auto As = Memreftype.getMemorySpaceAsInt();
          // if (as == mlir::gpu::GPUDialect::getWorkgroupAddressSpace()) {
          if (As == 5)
            continue;
        }
        Builder.setInsertionPointToEnd(NewBlock);
        auto *ClonedOp = Builder.clone(*Op, Mapping);
        ClonedOps.push_back(ClonedOp);
        OrigOpReplaced.push_back(Op);
      }
      BlocksToErase.push_back(&Block);
      BlockIdx++;
    }

    Builder.setInsertionPointToEnd(&AffineParallelOp.getRegion().back());
    auto *ClonedYield = Builder.clone(*YieldOp, Mapping);
    ClonedOps.push_back(ClonedYield);
    OrigOpReplaced.push_back(YieldOp);

    // Replace all uses of the original operations with the clones
    for (size_t I = 0; I < OrigOpReplaced.size(); ++I) {
      OrigOpReplaced[I]->replaceAllUsesWith(ClonedOps[I]);
    }

    // Now it should be safe to erase the original operations
    for (auto It = OrigOpReplaced.rbegin(); It != OrigOpReplaced.rend(); ++It) {
      auto *Op = *It;
      Op->erase();
    }

    // Move return op from the middle of parallel loop to the end of function
    // TODO: fix this to work properly in more complex return situations
    Builder.setInsertionPointToEnd(
        &AffineParallelOp->getParentRegion()->back());
    AffineParallelOp->walk([&](mlir::func::ReturnOp ReturnOp) {
      Builder.clone(*ReturnOp);
      ReturnOp->erase();
      return;
    });

    std::vector<mlir::Operation *> OpsToErase;
    int I = 0;
    for (auto Dim : {mlir::gpu::Dimension::x, mlir::gpu::Dimension::y,
                     mlir::gpu::Dimension::z}) {
      auto LocalIdValue = AffineParallelOp.getRegion().front().getArgument(I);
      FuncOp->walk([&](mlir::gpu::ThreadIdOp Gop) {
        if (Gop.getDimension() == Dim) {
          Gop.replaceAllUsesWith(LocalIdValue);
          OpsToErase.push_back(Gop);
        }
      });
      I++;
    }
    for (auto *Op : OpsToErase) {
      Op->erase();
    }
    mlir::Value IdxX = AffineParallelOp.getRegion().front().getArgument(0);
    mlir::Value IdxY = AffineParallelOp.getRegion().front().getArgument(1);
    mlir::Value IdxZ = AffineParallelOp.getRegion().front().getArgument(2);
    mlir::SmallVector<mlir::Value, 3> BarrierArgs{IdxX, IdxY, IdxZ};
    std::vector<mlir::gpu::BarrierOp> OldBarriers;
    FuncOp->walk([&](mlir::gpu::BarrierOp Op) {
      Builder.setInsertionPoint(Op);
      Builder.create<mlir::polygeist::BarrierOp>(Loc, BarrierArgs);
      OldBarriers.push_back(Op);
    });
    for (auto Op : OldBarriers) {
      Op.erase();
    }

    int Counter = 0;
    std::vector<mlir::Operation *> OldAffineOpsToDelete;
    FuncOp->walk([&](mlir::affine::AffineStoreOp Storeop) {
      Builder.setInsertionPoint(Storeop);
      mlir::AffineMap AffineMap = Storeop.getAffineMap();
      llvm::SmallVector<mlir::Value, 4> Indices;
      for (unsigned I = 0, E = AffineMap.getNumResults(); I < E; ++I) {
        auto ApplyOp = Builder.create<mlir::affine::AffineApplyOp>(
            Storeop.getLoc(), AffineMap.getSliceMap(I, 1),
            Storeop.getMapOperands());
        Indices.push_back(ApplyOp);
      }
      mlir::memref::StoreOp NewStoreOp = Builder.create<mlir::memref::StoreOp>(
          Storeop.getLoc(), Storeop.getValueToStore(), Storeop.getMemref(),
          Indices);
      Storeop->replaceAllUsesWith(NewStoreOp);
      OldAffineOpsToDelete.push_back(Storeop);
      Counter++;
    });
    FuncOp->walk([&](mlir::affine::AffineLoadOp Loadop) {
      Builder.setInsertionPoint(Loadop);
      mlir::AffineMap AffineMap = Loadop.getAffineMap();
      llvm::SmallVector<mlir::Value, 4> Indices;
      for (unsigned I = 0, E = AffineMap.getNumResults(); I < E; ++I) {
        auto ApplyOp = Builder.create<mlir::affine::AffineApplyOp>(
            Loadop.getLoc(), AffineMap.getSliceMap(I, 1),
            Loadop.getMapOperands());
        Indices.push_back(ApplyOp);
      }
      mlir::memref::LoadOp NewLoadOp = Builder.create<mlir::memref::LoadOp>(
          Loadop.getLoc(), Loadop.getMemRef(), Indices);
      Loadop.replaceAllUsesWith((mlir::Operation *)NewLoadOp);
      OldAffineOpsToDelete.push_back(Loadop);

      Counter++;
    });
    for (auto *Op : OldAffineOpsToDelete) {
      Op->erase();
    }

    return AffineParallelOp;
  }

  void
  createWrapperWithoutArgBuffer(mlir::func::FuncOp &F, mlir::OpBuilder &Builder,
                                std::vector<mlir::Value> &NewKernelArguments) {

    mlir::MLIRContext *Context = F->getContext();
    auto OldName = F.getName();
    std::string NewName = "pocl_mlir_";
    NewName += OldName;
    F.setName(NewName);

    mlir::FunctionType ExistingFuncType = F.getFunctionType();
    llvm::SmallVector<mlir::Type> OldArgs(ExistingFuncType.getInputs());
    mlir::Block *EntryBlock = &F.front();

    llvm::SmallVector<mlir::Type> UpdatedArgs;

    uint64_t ArgCount = OldArgs.size();
    mlir::IntegerType IntType = mlir::IntegerType::get(Context, 64);
    F->setAttr("CL_arg_count", mlir::IntegerAttr::get(IntType, ArgCount));
    for (size_t I = 0; I < ArgCount; ++I) {
      auto Argtype = OldArgs[I];
      UpdatedArgs.push_back(Argtype);
      auto Arg = EntryBlock->getArgument(I);
      NewKernelArguments.push_back(Arg);
    }
    mlir::Type SizeTArgType = Builder.getIntegerType(SizeTWidth);
    mlir::Type IntArgType = Builder.getIntegerType(32);
    // The pocl_context as hidden args
    for (int I = 0; I < 11; I++) {
      UpdatedArgs.push_back(SizeTArgType);
      EntryBlock->addArgument(SizeTArgType, Builder.getUnknownLoc());
    }
    // printf_buffer_capacity
    UpdatedArgs.push_back(IntArgType);
    EntryBlock->addArgument(IntArgType, Builder.getUnknownLoc());
    // global_var_buffer
    UpdatedArgs.push_back(SizeTArgType);
    EntryBlock->addArgument(SizeTArgType, Builder.getUnknownLoc());
    // work_dim
    UpdatedArgs.push_back(IntArgType);
    EntryBlock->addArgument(IntArgType, Builder.getUnknownLoc());

    UpdatedArgs.push_back(IntArgType);
    EntryBlock->addArgument(IntArgType, Builder.getUnknownLoc());

    // The global workgroup ids are passed to kernel as hidden args
    for (int I = 0; I < GROUP_ID_NUM_DIMENSIONS_AS_ARGS; I++) {
      UpdatedArgs.push_back(SizeTArgType);
      EntryBlock->addArgument(SizeTArgType, Builder.getUnknownLoc());
    }

    mlir::FunctionType UpdatedFuncType =
        mlir::FunctionType::get(Context, UpdatedArgs, {});
    F.setFunctionType(UpdatedFuncType);
  }

  void privatizeMLIRContextGPU(mlir::func::FuncOp &F,
                               mlir::OpBuilder &Builder) {
    std::vector<mlir::Operation *> OpsToErase;
    int I = 0;
    for (auto Dim : {mlir::gpu::Dimension::x, mlir::gpu::Dimension::y,
                     mlir::gpu::Dimension::z}) {
      F->walk([&](mlir::gpu::BlockDimOp LocalSizeOp) {
        if (LocalSizeOp.getDimension() == Dim) {
          mlir::Operation *LocalSizeValue;
          int LocalSizeOffsetFromEnd = GROUP_ID_NUM_DIMENSIONS_AS_ARGS +
                                       POCL_CONTEXT_NUM_ELEMENTS_AS_ARGS - 6;
          if (WGDynamicLocalSize) {
            int TotalNumOfArguments = F.getFunctionBody().getNumArguments();
            auto LocalSize = F.getFunctionBody().getArgument(
                TotalNumOfArguments - LocalSizeOffsetFromEnd + I);

            LocalSizeValue = Builder.create<mlir::arith::IndexCastOp>(
                LocalSizeOp.getLoc(), Builder.getIndexType(), LocalSize);
          } else {
            LocalSizeValue = Builder.create<mlir::arith::ConstantIndexOp>(
                LocalSizeOp.getLoc(), WGLocalSize[I]);
          }
          LocalSizeOp->replaceAllUsesWith(LocalSizeValue);
          OpsToErase.push_back(LocalSizeOp);
        }
      });
      I++;
    }
    I = 0;
    for (auto Dim : {mlir::gpu::Dimension::x, mlir::gpu::Dimension::y,
                     mlir::gpu::Dimension::z}) {
      F->walk([&](mlir::gpu::GridDimOp GridDimOp) {
        if (GridDimOp.getDimension() == Dim) {
          int GridDimOffsetFromEnd = GROUP_ID_NUM_DIMENSIONS_AS_ARGS +
                                     POCL_CONTEXT_NUM_ELEMENTS_AS_ARGS;
          int TotalNumOfArguments = F.getFunctionBody().getNumArguments();
          auto GridDim = F.getFunctionBody().getArgument(
              TotalNumOfArguments - GridDimOffsetFromEnd + I);

          auto GridDimValue = Builder.create<mlir::arith::IndexCastOp>(
              GridDimOp.getLoc(), Builder.getIndexType(), GridDim);
          GridDimOp->replaceAllUsesWith(GridDimValue);
          OpsToErase.push_back(GridDimOp);
        }
      });
      I++;
    }
    I = 0;
    for (auto Dim : {mlir::gpu::Dimension::x, mlir::gpu::Dimension::y,
                     mlir::gpu::Dimension::z}) {
      F->walk([&](mlir::gpu::BlockIdOp GroupIdOp) {
        if (GroupIdOp.getDimension() == Dim) {
          // replace all the loadops that used the value of this getGlobalOp
          int TotalNumOfArguments = F.getFunctionBody().getNumArguments();
          auto GroupId = F.getFunctionBody().getArgument(
              TotalNumOfArguments - GROUP_ID_NUM_DIMENSIONS_AS_ARGS + I);

          auto CastedGroupId = Builder.create<mlir::arith::IndexCastOp>(
              GroupIdOp.getLoc(), Builder.getIndexType(), GroupId);
          GroupIdOp->replaceAllUsesWith(CastedGroupId);
          OpsToErase.push_back(GroupIdOp);
        }
      });
      I++;
    }
    F->walk([&](mlir::pocl::WorkDimOp WorkDimOp) {
      int WorkDimOffsetFromEnd = GROUP_ID_NUM_DIMENSIONS_AS_ARGS +
                                 POCL_CONTEXT_NUM_ELEMENTS_AS_ARGS - 13;
      int TotalNumOfArguments = F.getFunctionBody().getNumArguments();
      auto GroupId = F.getFunctionBody().getArgument(TotalNumOfArguments -
                                                     WorkDimOffsetFromEnd);
      WorkDimOp->getResult(0).replaceAllUsesWith(GroupId);
      OpsToErase.push_back(WorkDimOp);
    });

    for (auto *Op : OpsToErase) {
      Op->erase();
    }
  }

  void runOnOperation() override {
    SizeTWidth = 64;
    auto Mod = getOperation();

    if (auto LocalSizeAttr =
            Mod->getAttrOfType<mlir::DenseI64ArrayAttr>("gpu.workgroup_size")) {
      auto ArrayRef = LocalSizeAttr.asArrayRef();
      if (ArrayRef.size() != 3) {
        POCL_MSG_ERR(
            "Incorrect number of dimensions in gpu.workgroup_size attribute\n");
        signalPassFailure();
      }
      WGLocalSize[0] = ArrayRef[0];
      WGLocalSize[1] = ArrayRef[1];
      WGLocalSize[2] = ArrayRef[2];
      WGDynamicLocalSize = false;
    } else {
      WGDynamicLocalSize = true;
    }
    mlir::IRRewriter Builder(Mod.getContext());

    Mod.walk([&](mlir::func::FuncOp FuncOp) {
      bool IsKernel =
          FuncOp->hasAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName());
      if (!IsKernel) {
        return mlir::WalkResult::advance();
      }

      Builder.setInsertionPointToStart(&FuncOp.getFunctionBody().front());

      std::vector<mlir::Value> KernelArguments;
      createWrapperWithoutArgBuffer(FuncOp, Builder, KernelArguments);

      privatizeMLIRContextGPU(FuncOp, Builder);

      createAffineParallelLoop(FuncOp, Builder);

      return mlir::WalkResult::interrupt();
    });
  }

  // Set to the hidden context argument.
  llvm::Argument *ContextArg;

  // Number of hidden args added to the work-group function.
  unsigned HiddenArgs = 0;

  // The width of the size_t data type in the current target.
  int SizeTWidth = 64;
  llvm::Type *SizeT = nullptr;
  llvm::Type *PoclContextT = nullptr;

  // Copies of compilation parameters
  std::string KernelName;
  unsigned long AddressBits;
  bool WGAssumeZeroGlobalOffset = true;
  bool WGDynamicLocalSize = false;
  bool DeviceUsingArgBufferLauncher;
  bool DeviceUsingGridLauncher;
  bool DeviceIsSPMD;
  unsigned long WGLocalSize[3] = {8, 1, 1};
  unsigned long WGMaxGridDimWidth;

  unsigned long DeviceGlobalASid;
  unsigned long DeviceLocalASid;
  unsigned long DeviceConstantASid;
  unsigned long DeviceContextASid;
  unsigned long DeviceArgsASid;
  bool DeviceSidePrintf;
  bool DeviceAllocaLocals;
  unsigned long DeviceMaxWItemDim;
  unsigned long DeviceMaxWItemSizes[3];
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::pocl::createWorkgroupPass() {
  return std::make_unique<Workgroup>();
}
