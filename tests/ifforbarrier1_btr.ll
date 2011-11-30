; ModuleID = 'ifforbarrier1.ll'

declare void @barrier(i32)

define void @ifforbarrier1() {
a:
  br i1 true, label %b, label %barrier

b:                                                ; preds = %a
  br label %d

barrier:                                          ; preds = %c, %a
  call void @barrier(i32 0)
  br label %c

c:                                                ; preds = %barrier
  br i1 true, label %barrier, label %d.btr

d:                                                ; preds = %b
  ret void

d.btr:                                            ; preds = %c
  ret void
}
