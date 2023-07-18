declare void @llvm.nvvm.barrier0()

; this is merely to enforce non-opaque-pointers
; required because there are no other pointers in the code,
; and llvm-link just sets opaque-pointers setting to whatever default
declare void @__pocl_unused(i8* %arg1) noduplicate

define void @_Z7barrierj(i32 %flags) noduplicate {
entry:
  call void @llvm.nvvm.barrier0()
  ret void
}
