; This is an "illegal" C function name on purpose. It's a magic
; handle based on which we know it's the special WG barrier function.
declare void @pocl.barrier() convergent

define void @"?barrier@@$$J0YAXI@Z"(i32 %flags) convergent {
entry:
  call void @pocl.barrier()
  ret void
}
