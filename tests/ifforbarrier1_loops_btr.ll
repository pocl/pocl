; ModuleID = 'ifforbarrier1_loops.ll'

declare void @barrier(i32)

define void @ifforbarrier1() {
a:
  br i1 true, label %b, label %barrier.preheader.loopbarrier

barrier.preheader.loopbarrier:                    ; preds = %a
  call void @barrier(i32 0)
  br label %barrier

b:                                                ; preds = %a
  br label %d

barrier:                                          ; preds = %c.latchbarrier, %barrier.preheader.loopbarrier
  call void @barrier(i32 0)
  br label %c.latchbarrier

c.latchbarrier:                                   ; preds = %barrier
  call void @barrier(i32 0)
  br i1 true, label %barrier, label %d.loopexit

d.loopexit:                                       ; preds = %c.latchbarrier
  br label %d.btr

d:                                                ; preds = %b
  ret void

d.btr:                                            ; preds = %d.loopexit
  ret void
}
