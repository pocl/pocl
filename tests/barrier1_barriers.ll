; ModuleID = 'barrier1.ll'

declare void @pocl.barrier()

define void @barrier1() {
barrier.prebarrier:
  br label %barrier

barrier:                                          ; preds = %barrier.prebarrier
  call void @pocl.barrier()
  br label %barrier.postbarrier

barrier.postbarrier:                              ; preds = %barrier
  ret void
}
