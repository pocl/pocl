include(TableGen)
include(AddMLIR)
include(AddLLVM)

include_directories(${MLIR_INCLUDE_DIRS})
add_mlir_dialect(PoclOps pocl)
