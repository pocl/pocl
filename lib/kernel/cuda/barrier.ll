declare void @llvm.nvvm.barrier0()

define void @_Z7barrierj(i32 %flags) noduplicate {
entry:
  call void @llvm.nvvm.barrier0()
  ret void
}
