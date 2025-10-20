module {
  func.func private @_Z14get_local_sizej(%arg0: i32) -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = scf.index_switch %0 -> i64
    case 0 {
      %block_dim_x = gpu.block_dim  x
      %2 = arith.index_cast %block_dim_x : index to i64
      scf.yield %2 : i64
    }
    case 1 {
      %block_dim_y = gpu.block_dim  y
      %2 = arith.index_cast %block_dim_y : index to i64
      scf.yield %2 : i64
    }
    case 2 {
      %block_dim_z = gpu.block_dim  z
      %2 = arith.index_cast %block_dim_z : index to i64
      scf.yield %2 : i64
    }
    default {
      scf.yield %c0_i32 : i64
    }
    return %1 : i64
  }
}
