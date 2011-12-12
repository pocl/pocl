; ModuleID = 'loopbarriers1_input.ll'

declare void @pocl.barrier()

define void @forbarrier1() {
a.loopbarrier:
  call void @pocl.barrier()
  br label %barrier

barrier:                                          ; preds = %barrier, %a.loopbarrier
  call void @pocl.barrier()
  br i1 true, label %barrier, label %b

b:                                                ; preds = %barrier
  ret void
}
