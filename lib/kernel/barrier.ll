declare void @_pocl_barrier()

define void @barrier(i32 %flags) {
entry:
  call void @_pocl_barrier()
  ret void
}
