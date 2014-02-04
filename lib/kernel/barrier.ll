; This is an "illegal" C function name on purpose. It's a magic
; handle based on which we know it's the special WG barrier function.
declare void @pocl.barrier()

define void @_Z7barrierj(i32 %flags) {
entry:
  call void @pocl.barrier()
  ret void
}
