; ModuleID = 'loopbarriers1_input.ll'

declare void @barrier(i32)

define void @forbarrier1() {
a.loopbarrier:
  call void @barrier(i32 0)
  br label %barrier

barrier:                                          ; preds = %barrier, %a.loopbarrier
  call void @barrier(i32 0)
  br i1 true, label %barrier, label %b

b:                                                ; preds = %barrier
  ret void
}
