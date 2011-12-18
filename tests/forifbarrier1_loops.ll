; ModuleID = 'forifbarrier1.ll'

declare void @pocl.barrier()

define void @forifbarrier1() {
a.loopbarrier:
  call void @pocl.barrier()
  br label %b

b:                                                ; preds = %d.latchbarrier, %a.loopbarrier
  br i1 true, label %c, label %barrier

c:                                                ; preds = %b
  br label %d.latchbarrier

barrier:                                          ; preds = %b
  call void @pocl.barrier()
  br label %d.latchbarrier

d.latchbarrier:                                   ; preds = %barrier, %c
  call void @pocl.barrier()
  br i1 true, label %b, label %e

e:                                                ; preds = %d.latchbarrier
  ret void
}
