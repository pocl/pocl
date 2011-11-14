; ModuleID = '../../../src/pocl.loopbarriers/tests/forifbarrier1_btr.ll'

declare void @barrier(i32)

define void @forifbarrier1() {
a.loopbarrier.prebarrier:
  br label %a.loopbarrier

a.loopbarrier:                                    ; preds = %a.loopbarrier.prebarrier
  call void @barrier(i32 0)
  br label %b

b:                                                ; preds = %d.btr.latchbarrier.postbarrier, %d, %a.loopbarrier
  br i1 true, label %c, label %barrier.prebarrier

c:                                                ; preds = %b
  br label %d

barrier.prebarrier:                               ; preds = %b
  br label %barrier

barrier:                                          ; preds = %barrier.prebarrier
  call void @barrier(i32 0)
  br label %barrier.postbarrier

barrier.postbarrier:                              ; preds = %barrier
  br label %d.btr.latchbarrier.prebarrier

d:                                                ; preds = %c
  br i1 true, label %b, label %e

e:                                                ; preds = %d
  ret void

d.btr.latchbarrier.prebarrier:                    ; preds = %barrier.postbarrier
  br label %d.btr.latchbarrier

d.btr.latchbarrier:                               ; preds = %d.btr.latchbarrier.prebarrier
  call void @barrier(i32 0)
  br label %d.btr.latchbarrier.postbarrier

d.btr.latchbarrier.postbarrier:                   ; preds = %d.btr.latchbarrier
  br i1 true, label %b, label %e.btr

e.btr:                                            ; preds = %d.btr.latchbarrier.postbarrier
  ret void
}
