; ModuleID = 'barriertails2_input.ll'

declare void @pocl.barrier()

define void @ifbarrier4() {
a:
  br i1 true, label %b, label %barrier

barrier:                                          ; preds = %a
  call void @pocl.barrier()
  br label %c.btr

b:                                                ; preds = %a
  br i1 true, label %d, label %c

c:                                                ; preds = %b
  ret void

d:                                                ; preds = %b
  ret void

c.btr:                                            ; preds = %barrier
  ret void
}
