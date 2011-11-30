; ModuleID = 'barriertails1_input.ll'

declare void @barrier(i32)

define void @ifbarrier2() {
a:
  br i1 true, label %b, label %c

b:                                                ; preds = %a
  br label %f

c:                                                ; preds = %a
  br i1 true, label %d, label %barrier

d:                                                ; preds = %c
  br label %e

barrier:                                          ; preds = %c
  call void @barrier(i32 0)
  br label %e.btr

e:                                                ; preds = %d
  br label %f

f:                                                ; preds = %e, %b
  ret void

e.btr:                                            ; preds = %barrier
  br label %f.btr

f.btr:                                            ; preds = %e.btr
  ret void
}
