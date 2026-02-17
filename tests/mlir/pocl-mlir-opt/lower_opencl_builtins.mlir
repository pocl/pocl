// RUN: pocl-mlir-opt --pocl-lower-opencl-builtins --split-input-file %s | %FileCheck %s
module {
  func.func private @_Z14get_num_groupsj(i32) -> i64
  func.func private @_Z14get_local_sizej(i32) -> i64
  func.func @_Z15get_global_sizej(%arg0: i32) -> i64 {
    %0 = call @_Z14get_num_groupsj(%arg0) : (i32) -> i64
    %1 = call @_Z14get_local_sizej(%arg0) : (i32) -> i64
    %2 = arith.muli %0, %1 : i64
    return %2 : i64
  }
}

// CHECK-LABEL: func.func @_Z15get_global_sizej
// CHECK-SAME:  (%[[ARG:[a-zA-Z0-9_]+]]: i32) -> i64
// CHECK-DAG:   %[[C0:[a-zA-Z0-9_]+]] = arith.constant 0 : i64
// CHECK-DAG:   %[[IDX:[a-zA-Z0-9_]+]] = arith.index_cast %[[ARG]] : i32 to index
// CHECK:       %[[SWITCH1:[a-zA-Z0-9_]+]] = scf.index_switch %[[IDX]] -> i64
// CHECK:       case 0 {
// CHECK:         %[[GRID_X:[a-zA-Z0-9_]+]] = gpu.grid_dim x
// CHECK:         %[[CAST_X:[a-zA-Z0-9_]+]] = arith.index_cast %[[GRID_X]] : index to i64
// CHECK:         scf.yield %[[CAST_X]] : i64
// CHECK:       }
// CHECK:       case 1 {
// CHECK:         = gpu.grid_dim y
// CHECK:       }
// CHECK:       case 2 {
// CHECK:         = gpu.grid_dim z
// CHECK:       }
// CHECK:       default {
// CHECK:         scf.yield %[[C0]] : i64
// CHECK:       }
// CHECK:       %[[IDX2:[a-zA-Z0-9_]+]] = arith.index_cast %[[ARG]] : i32 to index
// CHECK:       %[[SWITCH2:[a-zA-Z0-9_]+]] = scf.index_switch %[[IDX2]] -> i64
// CHECK:       case 0 {
// CHECK:         = gpu.block_dim x
// CHECK:       }
// CHECK:       case 1 {
// CHECK:         = gpu.block_dim y
// CHECK:       }
// CHECK:       case 2 {
// CHECK:         = gpu.block_dim z
// CHECK:       }
// CHECK:       default {
// CHECK:         scf.yield %[[C0]] : i64
// CHECK:       }
// CHECK:       %[[MUL:[a-zA-Z0-9_]+]] = arith.muli %[[SWITCH1]], %[[SWITCH2]] : i64
// CHECK:       return %[[MUL]] : i64

