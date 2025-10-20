
#ifndef POCL_DIALECT_POCLOPS_H
#define POCL_DIALECT_POCLOPS_H

#include <mlir/Interfaces/InferTypeOpInterface.h>

#pragma GCC visibility push(default)

#define GET_OP_CLASSES
#include "pocl/Dialect/PoclOps.h.inc"

#pragma GCC visibility pop

#endif
