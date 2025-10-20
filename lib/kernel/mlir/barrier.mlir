
module {
  func.func private @_Z7barrierj(%flags: i32) -> () attributes {llvm.linkage = #llvm.linkage<external>} {
    gpu.barrier
    return
  }
}
