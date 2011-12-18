; ModuleID = 'ifbarrier3.ll'

declare void @pocl.barrier()

define void @ifbarrier3() {
a:
  br i1 true, label %b, label %c

b:                                                ; preds = %a
  br i1 true, label %f, label %e

c:                                                ; preds = %a
  br i1 true, label %d, label %barrier

d:                                                ; preds = %c
  br label %e

barrier:                                          ; preds = %c
  call void @pocl.barrier()
  br label %e.btr

e:                                                ; preds = %d, %b
  br label %f

f:                                                ; preds = %e, %b
  ret void

e.btr:                                            ; preds = %barrier
  br label %f.btr

f.btr:                                            ; preds = %e.btr
  ret void
}
