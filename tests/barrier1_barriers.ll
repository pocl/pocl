; ModuleID = 'barrier1.ll'

declare void @barrier(i32)

define void @barrier1() {
barrier.prebarrier:
  br label %barrier

barrier:                                          ; preds = %barrier.prebarrier
  call void @barrier(i32 0)
  br label %barrier.postbarrier

barrier.postbarrier:                              ; preds = %barrier
  ret void
}
