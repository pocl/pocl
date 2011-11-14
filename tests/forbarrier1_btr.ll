; ModuleID = '../../../src/pocl.loopbarriers/tests/forbarrier1.ll'

declare void @barrier(i32)

define void @forbarrier1() {
a:
  br label %barrier

barrier:                                          ; preds = %barrier, %a
  call void @barrier(i32 0)
  br i1 true, label %barrier, label %b

b:                                                ; preds = %barrier
  ret void
}
