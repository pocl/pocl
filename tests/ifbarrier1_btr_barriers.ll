; ModuleID = 'ifbarrier1_btr.ll'

declare void @pocl.barrier()

define void @ifbarrier1() {
a:
  br i1 true, label %b, label %barrier.prebarrier

b:                                                ; preds = %a
  br label %c

barrier.prebarrier:                               ; preds = %a
  br label %barrier

barrier:                                          ; preds = %barrier.prebarrier
  call void @pocl.barrier()
  br label %c.btr

c:                                                ; preds = %b
  ret void

c.btr:                                            ; preds = %barrier
  ret void
}
