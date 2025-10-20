// RUN: pocl-mlir-opt --pocl-workgroup --allow-unregistered-dialect --split-input-file %s | FileCheck %s

module attributes {gpu.workgroup_size = array<i64: 8, 1, 1>} {
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
    %5 = arith.index_cast %4 : i64 to index
    %6 = memref.load %arg0[%5] : memref<?xi32>
    %7 = arith.addi %6, %c1_i32 : i32
    memref.store %7, %arg0[%5] : memref<?xi32>
    return
  }
}

// CHECK:    func.func @pocl_mlir_vecadd_kernel(%[[arg0:.+]]: memref<?xi32>, %[[arg1:.+]]: i32, %[[arg2:.+]]: i64, %[[arg3:.+]]: i64, %[[arg4:.+]]: i64)
// CHECK-SAME: attributes {CL_arg_count = 2 : i64, gpu.kernel}

// CHECK:      affine.parallel (%[[arg5:.+]], %[[arg6:.+]], %[[arg7:.+]]) = (0, 0, 0) to (8, 1, 1) {
// CHECK-DAG:    %[[c8:.+]] = arith.constant 8 : index
// CHECK-DAG:    %[[x0:.+]] = arith.index_cast %[[arg2]] : i64 to index
// CHECK-DAG:    %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-DAG:    %[[x1:.+]] = arith.index_cast %[[x0]] : index to i64
// CHECK-DAG:    %[[x2:.+]] = arith.index_cast %[[c8]] : index to i64
// CHECK:        %[[x3:.+]] = arith.muli %[[x1]], %[[x2]] : i64
// CHECK:        %[[x4:.+]] = arith.index_cast %[[arg5]] : index to i64
// CHECK:        %[[x5:.+]] = arith.addi %[[x3]], %[[x4]] : i64
// CHECK:        %[[x6:.+]] = arith.index_cast %[[x5]] : i64 to index
// CHECK:        %[[x7:.+]] = memref.load %[[arg0]][%[[x6]]] : memref<?xi32>
// CHECK:        %[[x8:.+]] = arith.addi %[[x7]], %[[c1_i32]] : i32
// CHECK:        memref.store %[[x8]], %[[arg0]][%[[x6]]] : memref<?xi32>
// CHECK:      }
// CHECK:      return
// CHECK:    }
