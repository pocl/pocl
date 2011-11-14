; ModuleID = '../../../src/pocl.loopbarriers/tests/forifbarrier1_barriers.ll'

declare void @barrier(i32)

define void @forifbarrier1() {
a.loopbarrier.prebarrier:
  br label %a.loopbarrier

a.loopbarrier:                                    ; preds = %a.loopbarrier.prebarrier
  call void @barrier(i32 0)
  br label %b

b:                                                ; preds = %d.latchbarrier.postbarrier.btr, %d.latchbarrier.postbarrier, %a.loopbarrier
  br i1 true, label %c, label %barrier.prebarrier

c:                                                ; preds = %b
  br label %d.latchbarrier.prebarrier

barrier.prebarrier:                               ; preds = %b
  br label %barrier

barrier:                                          ; preds = %barrier.prebarrier
  call void @barrier(i32 0)
  br label %barrier.postbarrier

barrier.postbarrier:                              ; preds = %barrier
  br label %d.latchbarrier.prebarrier.btr

d.latchbarrier.prebarrier:                        ; preds = %c
  br label %d.latchbarrier

d.latchbarrier:                                   ; preds = %d.latchbarrier.prebarrier
  call void @barrier(i32 0)
  br label %d.latchbarrier.postbarrier

d.latchbarrier.postbarrier:                       ; preds = %d.latchbarrier
  br i1 true, label %b, label %e

e:                                                ; preds = %d.latchbarrier.postbarrier
  ret void

d.latchbarrier.prebarrier.btr:                    ; preds = %barrier.postbarrier
  br label %d.latchbarrier.btr

d.latchbarrier.btr:                               ; preds = %d.latchbarrier.prebarrier.btr
  call void @barrier(i32 0)
  br label %d.latchbarrier.postbarrier.btr

d.latchbarrier.postbarrier.btr:                   ; preds = %d.latchbarrier.btr
  br i1 true, label %b, label %e.btr

e.btr:                                            ; preds = %d.latchbarrier.postbarrier.btr
  ret void
}
