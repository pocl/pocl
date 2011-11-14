; ModuleID = '../../../src/pocl.loopbarriers/tests/ifforbarrier1.ll'

declare void @barrier(i32)

define void @ifforbarrier1() {
a:
  br i1 true, label %b, label %barrier.preheader

barrier.preheader:                                ; preds = %a
  br label %barrier

b:                                                ; preds = %a
  br label %d

barrier:                                          ; preds = %barrier.preheader, %c
  call void @barrier(i32 0)
  br label %c

c:                                                ; preds = %barrier
  br i1 true, label %barrier, label %d.loopexit

d.loopexit:                                       ; preds = %c
  br label %d

d:                                                ; preds = %d.loopexit, %b
  ret void
}
