; ModuleID = 'forbarrier2.ll'

declare void @barrier(i32)

define void @forbarrier2() {
a:
  br label %b

b:                                                ; preds = %c, %a
  br label %barrier

barrier:                                          ; preds = %barrier, %b
  call void @barrier(i32 0)
  br i1 true, label %barrier, label %c

c:                                                ; preds = %barrier
  br i1 true, label %b, label %d

d:                                                ; preds = %c
  ret void
}
