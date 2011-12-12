; ModuleID = 'forbarrier1_loops_btr.ll'

declare void @pocl.barrier()

define void @forbarrier1() {
a.loopbarrier.prebarrier:
  br label %a.loopbarrier

a.loopbarrier:                                    ; preds = %a.loopbarrier.prebarrier
  call void @pocl.barrier()
  br label %barrier.prebarrier

barrier.prebarrier:                               ; preds = %barrier.postbarrier, %a.loopbarrier
  br label %barrier

barrier:                                          ; preds = %barrier.prebarrier
  call void @pocl.barrier()
  br label %barrier.postbarrier

barrier.postbarrier:                              ; preds = %barrier
  br i1 true, label %barrier.prebarrier, label %b

b:                                                ; preds = %barrier.postbarrier
  ret void
}
