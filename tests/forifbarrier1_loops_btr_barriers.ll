; ModuleID = 'forifbarrier1_loops_btr.ll'

declare void @pocl.barrier()

define void @forifbarrier1() {
a.loopbarrier.prebarrier:
  br label %a.loopbarrier

a.loopbarrier:                                    ; preds = %a.loopbarrier.prebarrier
  call void @pocl.barrier()
  br label %b

b:                                                ; preds = %d.latchbarrier.btr.postbarrier, %d.latchbarrier.postbarrier, %a.loopbarrier
  br i1 true, label %c, label %barrier.prebarrier

c:                                                ; preds = %b
  br label %d.latchbarrier

barrier.prebarrier:                               ; preds = %b
  br label %barrier

barrier:                                          ; preds = %barrier.prebarrier
  call void @pocl.barrier()
  br label %d.latchbarrier.btr

d.latchbarrier:                                   ; preds = %c
  call void @pocl.barrier()
  br label %d.latchbarrier.postbarrier

d.latchbarrier.postbarrier:                       ; preds = %d.latchbarrier
  br i1 true, label %b, label %e

e:                                                ; preds = %d.latchbarrier.postbarrier
  ret void

d.latchbarrier.btr:                               ; preds = %barrier
  call void @pocl.barrier()
  br label %d.latchbarrier.btr.postbarrier

d.latchbarrier.btr.postbarrier:                   ; preds = %d.latchbarrier.btr
  br i1 true, label %b, label %e.btr

e.btr:                                            ; preds = %d.latchbarrier.btr.postbarrier
  ret void
}
