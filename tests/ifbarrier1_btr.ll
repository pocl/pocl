; ModuleID = 'ifbarrier1.ll'

declare void @pocl.barrier()

define void @ifbarrier1() {
a:
  br i1 true, label %b, label %barrier

b:                                                ; preds = %a
  br label %c

barrier:                                          ; preds = %a
  call void @pocl.barrier()
  br label %c.btr

c:                                                ; preds = %b
  ret void

c.btr:                                            ; preds = %barrier
  ret void
}
