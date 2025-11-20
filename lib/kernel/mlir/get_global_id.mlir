module {
  func.func private @_Z13get_global_idj(%dim: i32) -> i64 {
    %group_id = func.call @_Z12get_group_idj(%dim) : (i32) -> i64
    %local_size = func.call @_Z14get_local_sizej(%dim) : (i32) -> i64
    %local_id = func.call @_Z12get_local_idj(%dim) : (i32) -> i64

    %group_id_idx = arith.index_cast %group_id : i64 to index
    %local_size_idx = arith.index_cast %local_size : i64 to index
    %local_id_idx = arith.index_cast %local_id : i64 to index

    %global_id = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)> ()[%group_id_idx, %local_size_idx, %local_id_idx]

    %global_id_i64 = arith.index_cast %global_id : index to i64
    func.return %global_id_i64 : i64
  }

  func.func private @_Z12get_group_idj(%dim: i32) -> i64
  func.func private @_Z14get_local_sizej(%dim: i32) -> i64
  func.func private @_Z12get_local_idj(%dim: i32) -> i64
}
