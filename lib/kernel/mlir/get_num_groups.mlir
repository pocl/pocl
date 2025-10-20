module {
  func.func private @_Z14get_num_groupsj(%arg0: i32) -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = scf.index_switch %0 -> i64
    case 0 {
      %thread_id_x = gpu.grid_dim  x
      %2 = arith.index_cast %thread_id_x : index to i64
      scf.yield %2 : i64
    }
    case 1 {
      %thread_id_y = gpu.grid_dim  y
      %2 = arith.index_cast %thread_id_y : index to i64
      scf.yield %2 : i64
    }
    case 2 {
      %thread_id_z = gpu.grid_dim  z
      %2 = arith.index_cast %thread_id_z : index to i64
      scf.yield %2 : i64
    }
    default {
      scf.yield %c0_i32 : i64
    }
    return %1 : i64
  }
}
