; ModuleID = '../../../src/pocl.loopbarriers/tests/ifbarrier1_btr.ll'

declare void @barrier(i32)

define void @ifbarrier1() {
a:
  br i1 true, label %b, label %barrier.prebarrier

b:                                                ; preds = %a
  br label %c

barrier.prebarrier:                               ; preds = %a
  br label %barrier

barrier:                                          ; preds = %barrier.prebarrier
  call void @barrier(i32 0)
  br label %barrier.postbarrier

barrier.postbarrier:                              ; preds = %barrier
  br label %c.btr

c:                                                ; preds = %b
  ret void

c.btr:                                            ; preds = %barrier.postbarrier
  ret void
}
