; ModuleID = 'forbarrier1_btr_loops_barriers.ll'

declare void @barrier(i32)

define void @forbarrier1() {
a.loopbarrier.prebarrier.wi_0_0_0:
  br label %a.loopbarrier.prebarrier.wi_1_0_0

a.loopbarrier:                                    ; preds = %a.loopbarrier.prebarrier.wi_1_1_1
  call void @barrier(i32 0)
  br label %barrier.prebarrier.wi_0_0_0

barrier.prebarrier.wi_0_0_0:                      ; preds = %barrier.postbarrier.wi_0_0_0, %a.loopbarrier
  br label %barrier.prebarrier.wi_1_0_0

barrier:                                          ; preds = %barrier.prebarrier.wi_1_1_1
  call void @barrier(i32 0)
  br label %barrier.postbarrier.wi_0_0_0

barrier.postbarrier.wi_0_0_0:                     ; preds = %barrier
  br i1 true, label %barrier.prebarrier.wi_0_0_0, label %b.wi_0_0_0

b.wi_0_0_0:                                       ; preds = %barrier.postbarrier.wi_0_0_0
  br label %barrier.postbarrier.wi_1_0_0

a.loopbarrier.prebarrier.wi_1_0_0:                ; preds = %a.loopbarrier.prebarrier.wi_0_0_0
  br label %a.loopbarrier.prebarrier.wi_0_1_0

a.loopbarrier.prebarrier.wi_0_1_0:                ; preds = %a.loopbarrier.prebarrier.wi_1_0_0
  br label %a.loopbarrier.prebarrier.wi_1_1_0

a.loopbarrier.prebarrier.wi_1_1_0:                ; preds = %a.loopbarrier.prebarrier.wi_0_1_0
  br label %a.loopbarrier.prebarrier.wi_0_0_1

a.loopbarrier.prebarrier.wi_0_0_1:                ; preds = %a.loopbarrier.prebarrier.wi_1_1_0
  br label %a.loopbarrier.prebarrier.wi_1_0_1

a.loopbarrier.prebarrier.wi_1_0_1:                ; preds = %a.loopbarrier.prebarrier.wi_0_0_1
  br label %a.loopbarrier.prebarrier.wi_0_1_1

a.loopbarrier.prebarrier.wi_0_1_1:                ; preds = %a.loopbarrier.prebarrier.wi_1_0_1
  br label %a.loopbarrier.prebarrier.wi_1_1_1

a.loopbarrier.prebarrier.wi_1_1_1:                ; preds = %a.loopbarrier.prebarrier.wi_0_1_1
  br label %a.loopbarrier

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

barrier.postbarrier.wi_1_0_0:                     ; preds = %b.wi_0_0_0
  br i1 true, label %unreachable, label %b.wi_1_0_0

b.wi_1_0_0:                                       ; preds = %barrier.postbarrier.wi_1_0_0
  br label %barrier.postbarrier.wi_0_1_0

unreachable:                                      ; preds = %barrier.postbarrier.wi_1_0_0
  unreachable

barrier.postbarrier.wi_0_1_0:                     ; preds = %b.wi_1_0_0
  br i1 true, label %unreachable19, label %b.wi_0_1_0

b.wi_0_1_0:                                       ; preds = %barrier.postbarrier.wi_0_1_0
  br label %barrier.postbarrier.wi_1_1_0

unreachable19:                                    ; preds = %barrier.postbarrier.wi_0_1_0
  unreachable

barrier.postbarrier.wi_1_1_0:                     ; preds = %b.wi_0_1_0
  br i1 true, label %unreachable22, label %b.wi_1_1_0

b.wi_1_1_0:                                       ; preds = %barrier.postbarrier.wi_1_1_0
  br label %barrier.postbarrier.wi_0_0_1

unreachable22:                                    ; preds = %barrier.postbarrier.wi_1_1_0
  unreachable

barrier.postbarrier.wi_0_0_1:                     ; preds = %b.wi_1_1_0
  br i1 true, label %unreachable25, label %b.wi_0_0_1

b.wi_0_0_1:                                       ; preds = %barrier.postbarrier.wi_0_0_1
  br label %barrier.postbarrier.wi_1_0_1

unreachable25:                                    ; preds = %barrier.postbarrier.wi_0_0_1
  unreachable

barrier.postbarrier.wi_1_0_1:                     ; preds = %b.wi_0_0_1
  br i1 true, label %unreachable28, label %b.wi_1_0_1

b.wi_1_0_1:                                       ; preds = %barrier.postbarrier.wi_1_0_1
  br label %barrier.postbarrier.wi_0_1_1

unreachable28:                                    ; preds = %barrier.postbarrier.wi_1_0_1
  unreachable

barrier.postbarrier.wi_0_1_1:                     ; preds = %b.wi_1_0_1
  br i1 true, label %unreachable31, label %b.wi_0_1_1

b.wi_0_1_1:                                       ; preds = %barrier.postbarrier.wi_0_1_1
  br label %barrier.postbarrier.wi_1_1_1

unreachable31:                                    ; preds = %barrier.postbarrier.wi_0_1_1
  unreachable

barrier.postbarrier.wi_1_1_1:                     ; preds = %b.wi_0_1_1
  br i1 true, label %unreachable34, label %b.wi_1_1_1

b.wi_1_1_1:                                       ; preds = %barrier.postbarrier.wi_1_1_1
  ret void

unreachable34:                                    ; preds = %barrier.postbarrier.wi_1_1_1
  unreachable
}
