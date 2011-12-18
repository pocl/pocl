; ModuleID = 'forbarrier2.ll'

declare void @pocl.barrier()

define void @forbarrier2() {
a.loopbarrier:
  call void @pocl.barrier()
  br label %b.loopbarrier

b.loopbarrier:                                    ; preds = %c.latchbarrier, %a.loopbarrier
  call void @pocl.barrier()
  br label %barrier

barrier:                                          ; preds = %barrier, %b.loopbarrier
  call void @pocl.barrier()
  br i1 true, label %barrier, label %c.latchbarrier

c.latchbarrier:                                   ; preds = %barrier
  call void @pocl.barrier()
  br i1 true, label %b.loopbarrier, label %d

d:                                                ; preds = %c.latchbarrier
  ret void
}
