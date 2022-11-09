; This is an "illegal" C function name on purpose. It's a magic
; handle based on which we know it's the special WG barrier function.
declare void @pocl.barrier() noduplicate

; this is merely to enforce opaque-pointers
; required because there are no other pointers in the code,
; and llvm-link just sets opaque-pointers setting to whatever default
declare void @__pocl_unused(ptr %arg1) noduplicate

define void @_Z7barrierj(i32 %flags) noduplicate {
entry:
  call void @pocl.barrier()
  ret void
}
