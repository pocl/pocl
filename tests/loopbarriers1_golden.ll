; ModuleID = 'loopbarriers1_input.ll'

declare void @pocl.barrier()

define void @loopbarriers1() {
a:
  br label %barrier

barrier:                                          ; preds = %barrier, %a
  call void @pocl.barrier()
  br i1 true, label %barrier, label %b

b:                                                ; preds = %barrier
  ret void
}
