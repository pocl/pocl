; ModuleID = 'barriers1_input.ll'

declare void @pocl.barrier()

declare void @foo()

define void @barriers1() {
entry.barrier:
  call void @pocl.barrier()
  br label %barrier.prebarrier

barrier.prebarrier:                               ; preds = %entry.barrier
  call void @foo()
  br label %barrier

barrier:                                          ; preds = %barrier.prebarrier
  call void @pocl.barrier()
  br label %barrier.postbarrier

barrier.postbarrier:                              ; preds = %barrier
  call void @foo()
  br label %exit.barrier

exit.barrier:                                     ; preds = %barrier.postbarrier
  call void @pocl.barrier()
  ret void
}
