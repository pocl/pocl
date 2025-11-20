module {
  func.func private @_Z12get_work_dimv() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = pocl.work_dim
    return %0 : i32
  }
}
