; ModuleID = 'forbarrier2.ll'

declare void @barrier(i32)

define void @forbarrier2() {
a.loopbarrier:
  call void @barrier(i32 0)
  br label %b.loopbarrier

b.loopbarrier:                                    ; preds = %c.latchbarrier, %a.loopbarrier
  call void @barrier(i32 0)
  br label %barrier

barrier:                                          ; preds = %barrier, %b.loopbarrier
  call void @barrier(i32 0)
  br i1 true, label %barrier, label %c.latchbarrier

c.latchbarrier:                                   ; preds = %barrier
  call void @barrier(i32 0)
  br i1 true, label %b.loopbarrier, label %d

d:                                                ; preds = %c.latchbarrier
  ret void
}
