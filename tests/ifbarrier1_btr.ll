; ModuleID = '../../../src/pocl.loopbarriers/tests/ifbarrier1.ll'

declare void @barrier(i32)

define void @ifbarrier1() {
a:
  br i1 true, label %b, label %barrier

b:                                                ; preds = %a
  br label %c

barrier:                                          ; preds = %a
  call void @barrier(i32 0)
  br label %c.btr

c:                                                ; preds = %b
  ret void

c.btr:                                            ; preds = %barrier
  ret void
}
