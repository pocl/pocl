declare void @pocl.barrier()

define void @barrier(i32 %flags) {
entry:
  call void @pocl.barrier()
  ret void
}