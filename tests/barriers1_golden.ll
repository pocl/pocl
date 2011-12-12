; ModuleID = 'barriers1_input.ll'

declare void @pocl.barrier()

declare void @foo()

define void @barrier1() {
barrier.prebarrier:
  call void @foo()
  br label %barrier

barrier:                                          ; preds = %barrier.prebarrier
  call void @pocl.barrier()
  br label %barrier.postbarrier

barrier.postbarrier:                              ; preds = %barrier
  call void @foo()
  ret void
}
