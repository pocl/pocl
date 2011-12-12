; ModuleID = 'forifbarrier1_loops.ll'

declare void @pocl.barrier()

define void @forifbarrier1() {
a.loopbarrier:
  call void @pocl.barrier()
  br label %b

b:                                                ; preds = %d.latchbarrier.btr, %d.latchbarrier, %a.loopbarrier
  br i1 true, label %c, label %barrier

c:                                                ; preds = %b
  br label %d.latchbarrier

barrier:                                          ; preds = %b
  call void @pocl.barrier()
  br label %d.latchbarrier.btr

d.latchbarrier:                                   ; preds = %c
  call void @pocl.barrier()
  br i1 true, label %b, label %e

e:                                                ; preds = %d.latchbarrier
  ret void

d.latchbarrier.btr:                               ; preds = %barrier
  call void @pocl.barrier()
  br i1 true, label %b, label %e.btr

e.btr:                                            ; preds = %d.latchbarrier.btr
  ret void
}
