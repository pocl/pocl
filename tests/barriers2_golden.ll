; ModuleID = 'barriers2_input.ll'

declare void @pocl.barrier()

define void @loopbarriers2() {
entry.barrier:
  call void @pocl.barrier()
  br label %a

a:                                                ; preds = %entry.barrier
  br label %b.loopbarrier

b.loopbarrier:                                    ; preds = %c.latchbarrier, %a
  call void @pocl.barrier()
  br label %barrier

barrier:                                          ; preds = %barrier, %b.loopbarrier
  call void @pocl.barrier()
  br i1 true, label %barrier, label %c.latchbarrier

c.latchbarrier:                                   ; preds = %barrier
  call void @pocl.barrier()
  br i1 true, label %b.loopbarrier, label %d

d:                                                ; preds = %c.latchbarrier
  br label %exit.barrier

exit.barrier:                                     ; preds = %d
  call void @pocl.barrier()
  ret void
}
