
//===- PoclDialect.cpp - Pocl dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "pocl/Dialect/Dialect.h"
#include "pocl/Dialect/PoclOps.h"

#define GET_OP_CLASSES
#include "pocl/Dialect/PoclOps.cpp.inc"

namespace mlir {
namespace pocl {

void PoclDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pocl/Dialect/PoclOps.cpp.inc"
      >();
}
} // namespace pocl
} // namespace mlir

#include "pocl/Dialect/PoclOpsDialect.cpp.inc"
