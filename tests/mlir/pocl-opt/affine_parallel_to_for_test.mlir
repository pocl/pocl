// RUN: pocl-opt --pocl-affine-parallel-to-for --allow-unregistered-dialect --split-input-file %s | FileCheck %s
module {
  func.func @pocl_mlir_command_buffer(%arg0: memref<?xi32>) attributes {gpu.kernel} {
    %c256_i32 = arith.constant 256 : i32
    affine.parallel (%arg1, %arg2, %arg3) = (0, 0, 0) to (256, 1, 1) {
      %0 = arith.index_cast %arg1 : index to i64
      %1 = arith.index_cast %arg2 : index to i64
      %2 = arith.index_cast %arg3 : index to i64
      memref.alloca_scope  {
        scf.execute_region {
          %c1_i32 = arith.constant 1 : i32
          affine.parallel (%arg4, %arg5, %arg6) = (0, 0, 0) to (1, 1, 1) {
            %3 = arith.index_cast %arg4 : index to i64
            %4 = arith.addi %0, %3 : i64
            %5 = arith.trunci %4 : i64 to i32
            %6 = arith.index_cast %5 : i32 to index
            %7 = memref.load %arg0[%6] : memref<?xi32>
            %8 = arith.addi %7, %c1_i32 : i32
            memref.store %8, %arg0[%6] : memref<?xi32>
          }
          scf.yield
        }
      }
    }
    return
  }
}

// 			CHECK:  func.func @pocl_mlir_command_buffer(%[[arg0:.+]]: memref<?xi32>) attributes {gpu.kernel} {
// CHECK-NEXT:		%[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-NEXT:    affine.for %[[arg1:.+]] = 0 to 256 {
// CHECK-NEXT:      affine.for %[[arg2:.+]] = 0 to 1 {
// CHECK-NEXT:        affine.for %[[arg3:.+]] = 0 to 1 {
// CHECK-NEXT:          %[[x0:.+]] = arith.index_cast %[[arg1]] : index to i64
// CHECK-NEXT:          memref.alloca_scope  {
// CHECK-NEXT:            scf.execute_region {
// CHECK-NEXT:              affine.for %[[arg4:.+]] = 0 to 1 {
// CHECK-NEXT:                affine.for %[[arg5:.+]] = 0 to 1 {
// CHECK-NEXT:                  affine.for %[[arg6:.+]] = 0 to 1 {
// CHECK-NEXT:                    %[[x1:.+]] = arith.index_cast %[[arg4]] : index to i64
// CHECK-NEXT:                    %[[x2:.+]] = arith.addi %[[x0]], %[[x1]] : i64
// CHECK-NEXT:                    %[[x3:.+]] = arith.trunci %[[x2]] : i64 to i32
// CHECK-NEXT:                    %[[x4:.+]] = arith.index_cast %[[x3]] : i32 to index
// CHECK-NEXT:                    %[[x5:.+]] = memref.load %[[arg0]][%[[x4]]] : memref<?xi32>
// CHECK-NEXT:                    %[[x6:.+]] = arith.addi %[[x5]], %[[c1_i32]] : i32
// CHECK-NEXT:                    memref.store %[[x6]], %[[arg0]][%[[x4]]] : memref<?xi32>
// CHECK-NEXT:                  }
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:              scf.yield
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
