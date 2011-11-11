; ModuleID = '../../../src/pocl.loopbarriers/tests/forbarrier1.ll'

declare void @barrier(i32)

define void @forbarrier1() {
a.prebarrier:
  br label %a

a:                                                ; preds = %a.prebarrier
  call void @barrier(i32 0)
  br label %barrier.prebarrier

barrier.prebarrier:                               ; preds = %barrier.postbarrier, %a
  br label %barrier

barrier:                                          ; preds = %barrier.prebarrier
  call void @barrier(i32 0)
  br label %barrier.postbarrier

barrier.postbarrier:                              ; preds = %barrier
  br i1 true, label %barrier.prebarrier, label %b

b:                                                ; preds = %barrier.postbarrier
  ret void
}
