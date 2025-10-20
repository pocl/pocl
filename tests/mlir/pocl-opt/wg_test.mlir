// RUN: pocl-opt --pocl-workgroup --allow-unregistered-dialect --split-input-file %s | FileCheck %s

module attributes {gpu.workgroup_size = array<i64: 1, 1, 1>} {
  func.func @vecadd_kernel(%arg0: memref<?xi32>, %arg1: i32) attributes {gpu.kernel} {
    %c1_i32 = arith.constant 1 : i32
    %block_id_x = gpu.block_id  x
    %0 = arith.index_cast %block_id_x : index to i64
    %block_dim_x = gpu.block_dim  x
    %1 = arith.index_cast %block_dim_x : index to i64
    %2 = arith.muli %0, %1 : i64
    %thread_id_x = gpu.thread_id  x
    %3 = arith.index_cast %thread_id_x : index to i64
    %4 = arith.addi %2, %3 : i64
    %5 = arith.trunci %4 : i64 to i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg0[%6] : memref<?xi32>
    %8 = arith.addi %7, %c1_i32 : i32
    memref.store %8, %arg0[%6] : memref<?xi32>
    return
  }
}

//      CHECK:	func.func @pocl_mlir_vecadd_kernel(%[[arg0:.+]]: memref<?xi32>, %[[arg1:.+]]: i32, %[[arg2:.+]]: i64, %[[arg3:.+]]: i64, %[[arg4:.+]]: i64) attributes {CL_arg_count = 2 : i64, gpu.kernel} {
// CHECK-NEXT:		affine.parallel (%[[arg5:.+]], %[[arg6:.+]], %[[arg7:.+]]) = (0, 0, 0) to (1, 1, 1) {
// CHECK-NEXT:			%[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:			%[[x0:.+]] = arith.index_cast %[[arg2]] : i64 to index
// CHECK-NEXT:			%[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-NEXT:			%[[x1:.+]] = arith.index_cast %[[x0]] : index to i64
// CHECK-NEXT:			%[[x2:.+]] = arith.index_cast %[[c1]] : index to i64
// CHECK-NEXT:			%[[x3:.+]] = arith.muli %[[x1]], %[[x2]] : i64
// CHECK-NEXT:			%[[x4:.+]] = arith.index_cast %[[arg5]] : index to i64
// CHECK-NEXT:			%[[x5:.+]] = arith.addi %[[x3]], %[[x4]] : i64
// CHECK-NEXT:			%[[x6:.+]] = arith.trunci %[[x5]] : i64 to i32
// CHECK-NEXT:			%[[x7:.+]] = arith.index_cast %[[x6]] : i32 to index
// CHECK-NEXT:			%[[x8:.+]] = memref.load %[[arg0]][%[[x7]]] : memref<?xi32>
// CHECK-NEXT:			%[[x9:.+]] = arith.addi %[[x8]], %[[c1_i32]] : i32
// CHECK-NEXT:			memref.store %[[x9]], %[[arg0]][%[[x7]]] : memref<?xi32>
// CHECK-NEXT:		}
// CHECK-NEXT:		return
// CHECK-NEXT:	}
