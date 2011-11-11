; ModuleID = '../../../src/pocl.loopbarriers/tests/forbarrier1_barriers.ll'

declare void @barrier(i32)

define void @forbarrier1() {
a.prebarrier:
  br label %a

a:                                                ; preds = %a.prebarrier
  call void @barrier(i32 0)
  br label %barrier.prebarrier.btr

barrier.prebarrier:                               ; preds = %barrier.postbarrier
  br label %barrier

barrier:                                          ; preds = %barrier.prebarrier
  call void @barrier(i32 0)
  br label %barrier.postbarrier

barrier.postbarrier:                              ; preds = %barrier
  br i1 true, label %barrier.prebarrier, label %b

b:                                                ; preds = %barrier.postbarrier
  ret void

barrier.prebarrier.btr:                           ; preds = %barrier.postbarrier.btr.btr, %a, %barrier.postbarrier.btr
  br label %barrier.btr

barrier.btr:                                      ; preds = %barrier.prebarrier.btr
  call void @barrier(i32 0)
  br label %barrier.postbarrier.btr.btr

barrier.postbarrier.btr:                          ; No predecessors!
  br i1 true, label %barrier.prebarrier.btr, label %b.btr

b.btr:                                            ; preds = %barrier.postbarrier.btr
  ret void

barrier.postbarrier.btr.btr:                      ; preds = %barrier.btr
  br i1 true, label %barrier.prebarrier.btr, label %b.btr.btr

b.btr.btr:                                        ; preds = %barrier.postbarrier.btr.btr
  ret void
}
