// RUN: pocl-mlir-opt --pocl-affine-parallel-to-for --allow-unregistered-dialect --split-input-file %s | FileCheck %s
module {
  func.func @pocl_mlir_command_buffer(%arg0: memref<?xi32>) {
    affine.parallel (%arg1, %arg2, %arg3) = (0, 0, 0) to (256, 1, 1) {
      affine.parallel (%arg4, %arg5, %arg6) = (0, 0, 0) to (1, 3, 1) {
        // Pass all loop iterators to the test op
        "test.do_something"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (memref<?xi32>, index, index, index, index, index, index) -> ()
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @pocl_mlir_command_buffer
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9_]+]]: memref<?xi32>)

// Outer loops
// CHECK: affine.for %[[IV0:[a-zA-Z0-9_]+]] = 0 to 256
// CHECK:   affine.for %[[IV1:[a-zA-Z0-9_]+]] = 0 to 1
// CHECK:     affine.for %[[IV2:[a-zA-Z0-9_]+]] = 0 to 1

// Inner loops
// CHECK:           affine.for %[[IV3:[a-zA-Z0-9_]+]] = 0 to 1
// CHECK:             affine.for %[[IV4:[a-zA-Z0-9_]+]] = 0 to 3
// CHECK:               affine.for %[[IV5:[a-zA-Z0-9_]+]] = 0 to 1
// CHECK:                 "test.do_something"(%[[ARG0]], %[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]], %[[IV4]], %[[IV5]]) : (memref<?xi32>, index, index, index, index, index, index) -> ()
// CHECK: return
