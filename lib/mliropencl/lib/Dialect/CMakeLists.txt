
add_mlir_dialect_library(PoclDialect
			PoclDialect.cpp
			DEPENDS
			MLIRPoclOpsIncGen
			)

target_include_directories(PoclDialect PRIVATE ${MLIR_INCLUDE_DIRS})
target_include_directories(PoclDialect PRIVATE "${CMAKE_SOURCE_DIR}/lib/mliropencl/include")
target_include_directories(PoclDialect PRIVATE "${CMAKE_BINARY_DIR}/lib/mliropencl/include")

