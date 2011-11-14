; ModuleID = '../../../src/pocl.loopbarriers/tests/forifbarrier1.ll'

declare void @barrier(i32)

define void @forifbarrier1() {
a:
  br label %b

b:                                                ; preds = %d.btr, %d, %a
  br i1 true, label %c, label %barrier

c:                                                ; preds = %b
  br label %d

barrier:                                          ; preds = %b
  call void @barrier(i32 0)
  br label %d.btr

d:                                                ; preds = %c
  br i1 true, label %b, label %e

e:                                                ; preds = %d
  ret void

d.btr:                                            ; preds = %barrier
  br i1 true, label %b, label %e.btr

e.btr:                                            ; preds = %d.btr
  ret void
}
