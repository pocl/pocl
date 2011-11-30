; ModuleID = 'barriers1_input.ll'

declare void @barrier(i32)

declare void @foo()

define void @barrier1() {
barrier.prebarrier:
  call void @foo()
  br label %barrier

barrier:                                          ; preds = %barrier.prebarrier
  call void @barrier(i32 0)
  br label %barrier.postbarrier

barrier.postbarrier:                              ; preds = %barrier
  call void @foo()
  ret void
}
