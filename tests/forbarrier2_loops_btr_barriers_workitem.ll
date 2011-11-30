; ModuleID = 'forbarrier2_loops_btr_barriers.ll'

declare void @barrier(i32)

define void @forbarrier2() {
a.loopbarrier.prebarrier.wi_0_0_0:
  br label %a.loopbarrier.prebarrier.wi_1_0_0

a.loopbarrier:                                    ; preds = %a.loopbarrier.prebarrier.wi_1_1_1
  call void @barrier(i32 0)
  br label %b.loopbarrier.prebarrier.wi_0_0_0

b.loopbarrier.prebarrier.wi_0_0_0:                ; preds = %c.latchbarrier.postbarrier.wi_0_0_0, %a.loopbarrier
  br label %b.loopbarrier.prebarrier.wi_1_0_0

b.loopbarrier:                                    ; preds = %b.loopbarrier.prebarrier.wi_1_1_1
  call void @barrier(i32 0)
  br label %barrier.prebarrier.wi_0_0_0

barrier.prebarrier.wi_0_0_0:                      ; preds = %barrier.postbarrier.wi_0_0_0, %b.loopbarrier
  br label %barrier.prebarrier.wi_1_0_0

barrier:                                          ; preds = %barrier.prebarrier.wi_1_1_1
  call void @barrier(i32 0)
  br label %barrier.postbarrier.wi_0_0_0

barrier.postbarrier.wi_0_0_0:                     ; preds = %barrier
  br i1 true, label %barrier.prebarrier.wi_0_0_0, label %c.latchbarrier.prebarrier.wi_0_0_0

c.latchbarrier.prebarrier.wi_0_0_0:               ; preds = %barrier.postbarrier.wi_0_0_0
  br label %barrier.postbarrier.wi_1_0_0

c.latchbarrier:                                   ; preds = %c.latchbarrier.prebarrier.wi_1_1_1
  call void @barrier(i32 0)
  br label %c.latchbarrier.postbarrier.wi_0_0_0

c.latchbarrier.postbarrier.wi_0_0_0:              ; preds = %c.latchbarrier
  br i1 true, label %b.loopbarrier.prebarrier.wi_0_0_0, label %d.wi_0_0_0

d.wi_0_0_0:                                       ; preds = %c.latchbarrier.postbarrier.wi_0_0_0
  br label %c.latchbarrier.postbarrier.wi_1_0_0

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

b.loopbarrier.prebarrier.wi_1_0_0:                ; preds = %b.loopbarrier.prebarrier.wi_0_0_0
  br label %b.loopbarrier.prebarrier.wi_0_1_0

b.loopbarrier.prebarrier.wi_0_1_0:                ; preds = %b.loopbarrier.prebarrier.wi_1_0_0
  br label %b.loopbarrier.prebarrier.wi_1_1_0

b.loopbarrier.prebarrier.wi_1_1_0:                ; preds = %b.loopbarrier.prebarrier.wi_0_1_0
  br label %b.loopbarrier.prebarrier.wi_0_0_1

b.loopbarrier.prebarrier.wi_0_0_1:                ; preds = %b.loopbarrier.prebarrier.wi_1_1_0
  br label %b.loopbarrier.prebarrier.wi_1_0_1

b.loopbarrier.prebarrier.wi_1_0_1:                ; preds = %b.loopbarrier.prebarrier.wi_0_0_1
  br label %b.loopbarrier.prebarrier.wi_0_1_1

b.loopbarrier.prebarrier.wi_0_1_1:                ; preds = %b.loopbarrier.prebarrier.wi_1_0_1
  br label %b.loopbarrier.prebarrier.wi_1_1_1

b.loopbarrier.prebarrier.wi_1_1_1:                ; preds = %b.loopbarrier.prebarrier.wi_0_1_1
  br label %b.loopbarrier

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

c.latchbarrier.prebarrier.wi_1_0_0:               ; preds = %barrier.postbarrier.wi_1_0_0
  br label %barrier.postbarrier.wi_0_1_0

barrier.postbarrier.wi_1_0_0:                     ; preds = %c.latchbarrier.prebarrier.wi_0_0_0
  br i1 true, label %unreachable, label %c.latchbarrier.prebarrier.wi_1_0_0

unreachable:                                      ; preds = %barrier.postbarrier.wi_1_0_0
  unreachable

c.latchbarrier.prebarrier.wi_0_1_0:               ; preds = %barrier.postbarrier.wi_0_1_0
  br label %barrier.postbarrier.wi_1_1_0

barrier.postbarrier.wi_0_1_0:                     ; preds = %c.latchbarrier.prebarrier.wi_1_0_0
  br i1 true, label %unreachable26, label %c.latchbarrier.prebarrier.wi_0_1_0

unreachable26:                                    ; preds = %barrier.postbarrier.wi_0_1_0
  unreachable

c.latchbarrier.prebarrier.wi_1_1_0:               ; preds = %barrier.postbarrier.wi_1_1_0
  br label %barrier.postbarrier.wi_0_0_1

barrier.postbarrier.wi_1_1_0:                     ; preds = %c.latchbarrier.prebarrier.wi_0_1_0
  br i1 true, label %unreachable29, label %c.latchbarrier.prebarrier.wi_1_1_0

unreachable29:                                    ; preds = %barrier.postbarrier.wi_1_1_0
  unreachable

c.latchbarrier.prebarrier.wi_0_0_1:               ; preds = %barrier.postbarrier.wi_0_0_1
  br label %barrier.postbarrier.wi_1_0_1

barrier.postbarrier.wi_0_0_1:                     ; preds = %c.latchbarrier.prebarrier.wi_1_1_0
  br i1 true, label %unreachable32, label %c.latchbarrier.prebarrier.wi_0_0_1

unreachable32:                                    ; preds = %barrier.postbarrier.wi_0_0_1
  unreachable

c.latchbarrier.prebarrier.wi_1_0_1:               ; preds = %barrier.postbarrier.wi_1_0_1
  br label %barrier.postbarrier.wi_0_1_1

barrier.postbarrier.wi_1_0_1:                     ; preds = %c.latchbarrier.prebarrier.wi_0_0_1
  br i1 true, label %unreachable35, label %c.latchbarrier.prebarrier.wi_1_0_1

unreachable35:                                    ; preds = %barrier.postbarrier.wi_1_0_1
  unreachable

c.latchbarrier.prebarrier.wi_0_1_1:               ; preds = %barrier.postbarrier.wi_0_1_1
  br label %barrier.postbarrier.wi_1_1_1

barrier.postbarrier.wi_0_1_1:                     ; preds = %c.latchbarrier.prebarrier.wi_1_0_1
  br i1 true, label %unreachable38, label %c.latchbarrier.prebarrier.wi_0_1_1

unreachable38:                                    ; preds = %barrier.postbarrier.wi_0_1_1
  unreachable

c.latchbarrier.prebarrier.wi_1_1_1:               ; preds = %barrier.postbarrier.wi_1_1_1
  br label %c.latchbarrier

barrier.postbarrier.wi_1_1_1:                     ; preds = %c.latchbarrier.prebarrier.wi_0_1_1
  br i1 true, label %unreachable41, label %c.latchbarrier.prebarrier.wi_1_1_1

unreachable41:                                    ; preds = %barrier.postbarrier.wi_1_1_1
  unreachable

c.latchbarrier.postbarrier.wi_1_0_0:              ; preds = %d.wi_0_0_0
  br i1 true, label %unreachable44, label %d.wi_1_0_0

d.wi_1_0_0:                                       ; preds = %c.latchbarrier.postbarrier.wi_1_0_0
  br label %c.latchbarrier.postbarrier.wi_0_1_0

unreachable44:                                    ; preds = %c.latchbarrier.postbarrier.wi_1_0_0
  unreachable

c.latchbarrier.postbarrier.wi_0_1_0:              ; preds = %d.wi_1_0_0
  br i1 true, label %unreachable47, label %d.wi_0_1_0

d.wi_0_1_0:                                       ; preds = %c.latchbarrier.postbarrier.wi_0_1_0
  br label %c.latchbarrier.postbarrier.wi_1_1_0

unreachable47:                                    ; preds = %c.latchbarrier.postbarrier.wi_0_1_0
  unreachable

c.latchbarrier.postbarrier.wi_1_1_0:              ; preds = %d.wi_0_1_0
  br i1 true, label %unreachable50, label %d.wi_1_1_0

d.wi_1_1_0:                                       ; preds = %c.latchbarrier.postbarrier.wi_1_1_0
  br label %c.latchbarrier.postbarrier.wi_0_0_1

unreachable50:                                    ; preds = %c.latchbarrier.postbarrier.wi_1_1_0
  unreachable

c.latchbarrier.postbarrier.wi_0_0_1:              ; preds = %d.wi_1_1_0
  br i1 true, label %unreachable53, label %d.wi_0_0_1

d.wi_0_0_1:                                       ; preds = %c.latchbarrier.postbarrier.wi_0_0_1
  br label %c.latchbarrier.postbarrier.wi_1_0_1

unreachable53:                                    ; preds = %c.latchbarrier.postbarrier.wi_0_0_1
  unreachable

c.latchbarrier.postbarrier.wi_1_0_1:              ; preds = %d.wi_0_0_1
  br i1 true, label %unreachable56, label %d.wi_1_0_1

d.wi_1_0_1:                                       ; preds = %c.latchbarrier.postbarrier.wi_1_0_1
  br label %c.latchbarrier.postbarrier.wi_0_1_1

unreachable56:                                    ; preds = %c.latchbarrier.postbarrier.wi_1_0_1
  unreachable

c.latchbarrier.postbarrier.wi_0_1_1:              ; preds = %d.wi_1_0_1
  br i1 true, label %unreachable59, label %d.wi_0_1_1

d.wi_0_1_1:                                       ; preds = %c.latchbarrier.postbarrier.wi_0_1_1
  br label %c.latchbarrier.postbarrier.wi_1_1_1

unreachable59:                                    ; preds = %c.latchbarrier.postbarrier.wi_0_1_1
  unreachable

c.latchbarrier.postbarrier.wi_1_1_1:              ; preds = %d.wi_0_1_1
  br i1 true, label %unreachable62, label %d.wi_1_1_1

d.wi_1_1_1:                                       ; preds = %c.latchbarrier.postbarrier.wi_1_1_1
  ret void

unreachable62:                                    ; preds = %c.latchbarrier.postbarrier.wi_1_1_1
  unreachable
}
