; ModuleID = 'barrier1_barriers.ll'

declare void @pocl.barrier()

define void @barrier1() {
barrier.prebarrier.wi_0_0_0:
  br label %barrier.prebarrier.wi_1_0_0

barrier:                                          ; preds = %barrier.prebarrier.wi_1_1_1
  call void @pocl.barrier()
  br label %barrier.postbarrier.wi_0_0_0

barrier.postbarrier.wi_0_0_0:                     ; preds = %barrier
  br label %barrier.postbarrier.wi_1_0_0

barrier.prebarrier.wi_1_0_0:                      ; preds = %barrier.prebarrier.wi_0_0_0
  br label %barrier.prebarrier.wi_0_1_0

barrier.prebarrier.wi_0_1_0:                      ; preds = %barrier.prebarrier.wi_1_0_0
  br label %barrier.prebarrier.wi_1_1_0

barrier.prebarrier.wi_1_1_0:                      ; preds = %barrier.prebarrier.wi_0_1_0
  br label %barrier.prebarrier.wi_0_0_1

barrier.prebarrier.wi_0_0_1:                      ; preds = %barrier.prebarrier.wi_1_1_0
  br label %barrier.prebarrier.wi_1_0_1

barrier.prebarrier.wi_1_0_1:                      ; preds = %barrier.prebarrier.wi_0_0_1
  br label %barrier.prebarrier.wi_0_1_1

barrier.prebarrier.wi_0_1_1:                      ; preds = %barrier.prebarrier.wi_1_0_1
  br label %barrier.prebarrier.wi_1_1_1

barrier.prebarrier.wi_1_1_1:                      ; preds = %barrier.prebarrier.wi_0_1_1
  br label %barrier

barrier.postbarrier.wi_1_0_0:                     ; preds = %barrier.postbarrier.wi_0_0_0
  br label %barrier.postbarrier.wi_0_1_0

barrier.postbarrier.wi_0_1_0:                     ; preds = %barrier.postbarrier.wi_1_0_0
  br label %barrier.postbarrier.wi_1_1_0

barrier.postbarrier.wi_1_1_0:                     ; preds = %barrier.postbarrier.wi_0_1_0
  br label %barrier.postbarrier.wi_0_0_1

barrier.postbarrier.wi_0_0_1:                     ; preds = %barrier.postbarrier.wi_1_1_0
  br label %barrier.postbarrier.wi_1_0_1

barrier.postbarrier.wi_1_0_1:                     ; preds = %barrier.postbarrier.wi_0_0_1
  br label %barrier.postbarrier.wi_0_1_1

barrier.postbarrier.wi_0_1_1:                     ; preds = %barrier.postbarrier.wi_1_0_1
  br label %barrier.postbarrier.wi_1_1_1

barrier.postbarrier.wi_1_1_1:                     ; preds = %barrier.postbarrier.wi_0_1_1
  ret void
}
