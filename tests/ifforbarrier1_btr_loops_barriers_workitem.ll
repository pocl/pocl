; ModuleID = 'ifforbarrier1_btr_loops_barriers.ll'

declare void @barrier(i32)

define void @ifforbarrier1() {
a.wi_0_0_0:
  br i1 true, label %b.wi_0_0_0, label %barrier.preheader.loopbarrier.prebarrier.wi_0_0_0

barrier.preheader.loopbarrier.prebarrier.wi_0_0_0: ; preds = %a.wi_0_0_0
  br label %a.wi_1_0_0

barrier.preheader.loopbarrier:                    ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_1_1_1
  call void @barrier(i32 0)
  br label %barrier.prebarrier.wi_0_0_0

b.wi_0_0_0:                                       ; preds = %a.wi_0_0_0
  br label %d.wi_0_0_0

barrier.prebarrier.wi_0_0_0:                      ; preds = %c.latchbarrier.postbarrier.wi_0_0_0, %barrier.preheader.loopbarrier
  br label %barrier.prebarrier.wi_1_0_0

barrier:                                          ; preds = %barrier.prebarrier.wi_1_1_1
  call void @barrier(i32 0)
  br label %c.latchbarrier

c.latchbarrier:                                   ; preds = %barrier
  call void @barrier(i32 0)
  br label %c.latchbarrier.postbarrier.wi_0_0_0

c.latchbarrier.postbarrier.wi_0_0_0:              ; preds = %c.latchbarrier
  br i1 true, label %barrier.prebarrier.wi_0_0_0, label %d.btr.wi_0_0_0

d.wi_0_0_0:                                       ; preds = %b.wi_0_0_0
  br label %a.wi_1_0_084

d.btr.wi_0_0_0:                                   ; preds = %c.latchbarrier.postbarrier.wi_0_0_0
  br label %c.latchbarrier.postbarrier.wi_1_0_0

barrier.preheader.loopbarrier.prebarrier.wi_1_0_0: ; preds = %a.wi_1_0_0
  br label %a.wi_0_1_0

a.wi_1_0_0:                                       ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_0_0_0
  br i1 true, label %unreachable, label %barrier.preheader.loopbarrier.prebarrier.wi_1_0_0

unreachable:                                      ; preds = %a.wi_1_0_0
  unreachable

barrier.preheader.loopbarrier.prebarrier.wi_0_1_0: ; preds = %a.wi_0_1_0
  br label %a.wi_1_1_0

a.wi_0_1_0:                                       ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_1_0_0
  br i1 true, label %unreachable5, label %barrier.preheader.loopbarrier.prebarrier.wi_0_1_0

unreachable5:                                     ; preds = %a.wi_0_1_0
  unreachable

barrier.preheader.loopbarrier.prebarrier.wi_1_1_0: ; preds = %a.wi_1_1_0
  br label %a.wi_0_0_1

a.wi_1_1_0:                                       ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_0_1_0
  br i1 true, label %unreachable8, label %barrier.preheader.loopbarrier.prebarrier.wi_1_1_0

unreachable8:                                     ; preds = %a.wi_1_1_0
  unreachable

barrier.preheader.loopbarrier.prebarrier.wi_0_0_1: ; preds = %a.wi_0_0_1
  br label %a.wi_1_0_1

a.wi_0_0_1:                                       ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_1_1_0
  br i1 true, label %unreachable11, label %barrier.preheader.loopbarrier.prebarrier.wi_0_0_1

unreachable11:                                    ; preds = %a.wi_0_0_1
  unreachable

barrier.preheader.loopbarrier.prebarrier.wi_1_0_1: ; preds = %a.wi_1_0_1
  br label %a.wi_0_1_1

a.wi_1_0_1:                                       ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_0_0_1
  br i1 true, label %unreachable14, label %barrier.preheader.loopbarrier.prebarrier.wi_1_0_1

unreachable14:                                    ; preds = %a.wi_1_0_1
  unreachable

barrier.preheader.loopbarrier.prebarrier.wi_0_1_1: ; preds = %a.wi_0_1_1
  br label %a.wi_1_1_1

a.wi_0_1_1:                                       ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_1_0_1
  br i1 true, label %unreachable17, label %barrier.preheader.loopbarrier.prebarrier.wi_0_1_1

unreachable17:                                    ; preds = %a.wi_0_1_1
  unreachable

barrier.preheader.loopbarrier.prebarrier.wi_1_1_1: ; preds = %a.wi_1_1_1
  br label %barrier.preheader.loopbarrier

a.wi_1_1_1:                                       ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_0_1_1
  br i1 true, label %unreachable20, label %barrier.preheader.loopbarrier.prebarrier.wi_1_1_1

unreachable20:                                    ; preds = %a.wi_1_1_1
  unreachable

barrier.prebarrier.wi_1_0_0:                      ; preds = %barrier.prebarrier.wi_0_0_0
  br label %barrier.prebarrier.wi_0_1_0

barrier.prebarrier.wi_0_1_0:                      ; preds = %barrier.prebarrier.wi_1_0_0
  br label %barrier.prebarrier.wi_1_1_0

barrier.prebarrier.wi_1_1_0:                      ; preds = %barrier.prebarrier.wi_0_1_0
  br label %barrier.prebarrier.wi_0_0_1

barrier.prebarrier.wi_0_0_1:                      ; preds = %barrier.prebarrier.wi_1_1_0
  br label %barrier.prebarrier.wi_1_0_1

barrier.prebarrier.wi_1_0_1:                      ; preds = %barrier.prebarrier.wi_0_0_1
  br label %barrier.prebarrier.wi_0_1_1

barrier.prebarrier.wi_0_1_1:                      ; preds = %barrier.prebarrier.wi_1_0_1
  br label %barrier.prebarrier.wi_1_1_1

barrier.prebarrier.wi_1_1_1:                      ; preds = %barrier.prebarrier.wi_0_1_1
  br label %barrier

c.latchbarrier.postbarrier.wi_1_0_0:              ; preds = %d.btr.wi_0_0_0
  br i1 true, label %unreachable30, label %d.btr.wi_1_0_0

d.btr.wi_1_0_0:                                   ; preds = %c.latchbarrier.postbarrier.wi_1_0_0
  br label %c.latchbarrier.postbarrier.wi_0_1_0

unreachable30:                                    ; preds = %c.latchbarrier.postbarrier.wi_1_0_0
  unreachable

c.latchbarrier.postbarrier.wi_0_1_0:              ; preds = %d.btr.wi_1_0_0
  br i1 true, label %unreachable33, label %d.btr.wi_0_1_0

d.btr.wi_0_1_0:                                   ; preds = %c.latchbarrier.postbarrier.wi_0_1_0
  br label %c.latchbarrier.postbarrier.wi_1_1_0

unreachable33:                                    ; preds = %c.latchbarrier.postbarrier.wi_0_1_0
  unreachable

c.latchbarrier.postbarrier.wi_1_1_0:              ; preds = %d.btr.wi_0_1_0
  br i1 true, label %unreachable36, label %d.btr.wi_1_1_0

d.btr.wi_1_1_0:                                   ; preds = %c.latchbarrier.postbarrier.wi_1_1_0
  br label %c.latchbarrier.postbarrier.wi_0_0_1

unreachable36:                                    ; preds = %c.latchbarrier.postbarrier.wi_1_1_0
  unreachable

c.latchbarrier.postbarrier.wi_0_0_1:              ; preds = %d.btr.wi_1_1_0
  br i1 true, label %unreachable39, label %d.btr.wi_0_0_1

d.btr.wi_0_0_1:                                   ; preds = %c.latchbarrier.postbarrier.wi_0_0_1
  br label %c.latchbarrier.postbarrier.wi_1_0_1

unreachable39:                                    ; preds = %c.latchbarrier.postbarrier.wi_0_0_1
  unreachable

c.latchbarrier.postbarrier.wi_1_0_1:              ; preds = %d.btr.wi_0_0_1
  br i1 true, label %unreachable42, label %d.btr.wi_1_0_1

d.btr.wi_1_0_1:                                   ; preds = %c.latchbarrier.postbarrier.wi_1_0_1
  br label %c.latchbarrier.postbarrier.wi_0_1_1

unreachable42:                                    ; preds = %c.latchbarrier.postbarrier.wi_1_0_1
  unreachable

c.latchbarrier.postbarrier.wi_0_1_1:              ; preds = %d.btr.wi_1_0_1
  br i1 true, label %unreachable45, label %d.btr.wi_0_1_1

d.btr.wi_0_1_1:                                   ; preds = %c.latchbarrier.postbarrier.wi_0_1_1
  br label %c.latchbarrier.postbarrier.wi_1_1_1

unreachable45:                                    ; preds = %c.latchbarrier.postbarrier.wi_0_1_1
  unreachable

c.latchbarrier.postbarrier.wi_1_1_1:              ; preds = %d.btr.wi_0_1_1
  br i1 true, label %unreachable48, label %d.btr.wi_1_1_1

d.btr.wi_1_1_1:                                   ; preds = %c.latchbarrier.postbarrier.wi_1_1_1
  ret void

unreachable48:                                    ; preds = %c.latchbarrier.postbarrier.wi_1_1_1
  unreachable

a.wi_1_0_084:                                     ; preds = %d.wi_0_0_0
  br i1 true, label %b.wi_1_0_0, label %barrier.preheader.loopbarrier.prebarrier.wi_1_0_091

b.wi_1_0_0:                                       ; preds = %a.wi_1_0_084
  br label %d.wi_1_0_0

d.wi_1_0_0:                                       ; preds = %b.wi_1_0_0
  br label %a.wi_0_1_085

barrier.preheader.loopbarrier.prebarrier.wi_1_0_091: ; preds = %a.wi_1_0_084
  br label %unreachable53

unreachable53:                                    ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_1_0_091
  unreachable

a.wi_0_1_085:                                     ; preds = %d.wi_1_0_0
  br i1 true, label %b.wi_0_1_0, label %barrier.preheader.loopbarrier.prebarrier.wi_0_1_092

b.wi_0_1_0:                                       ; preds = %a.wi_0_1_085
  br label %d.wi_0_1_0

d.wi_0_1_0:                                       ; preds = %b.wi_0_1_0
  br label %a.wi_1_1_086

barrier.preheader.loopbarrier.prebarrier.wi_0_1_092: ; preds = %a.wi_0_1_085
  br label %unreachable58

unreachable58:                                    ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_0_1_092
  unreachable

a.wi_1_1_086:                                     ; preds = %d.wi_0_1_0
  br i1 true, label %b.wi_1_1_0, label %barrier.preheader.loopbarrier.prebarrier.wi_1_1_093

b.wi_1_1_0:                                       ; preds = %a.wi_1_1_086
  br label %d.wi_1_1_0

d.wi_1_1_0:                                       ; preds = %b.wi_1_1_0
  br label %a.wi_0_0_187

barrier.preheader.loopbarrier.prebarrier.wi_1_1_093: ; preds = %a.wi_1_1_086
  br label %unreachable63

unreachable63:                                    ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_1_1_093
  unreachable

a.wi_0_0_187:                                     ; preds = %d.wi_1_1_0
  br i1 true, label %b.wi_0_0_1, label %barrier.preheader.loopbarrier.prebarrier.wi_0_0_194

b.wi_0_0_1:                                       ; preds = %a.wi_0_0_187
  br label %d.wi_0_0_1

d.wi_0_0_1:                                       ; preds = %b.wi_0_0_1
  br label %a.wi_1_0_188

barrier.preheader.loopbarrier.prebarrier.wi_0_0_194: ; preds = %a.wi_0_0_187
  br label %unreachable68

unreachable68:                                    ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_0_0_194
  unreachable

a.wi_1_0_188:                                     ; preds = %d.wi_0_0_1
  br i1 true, label %b.wi_1_0_1, label %barrier.preheader.loopbarrier.prebarrier.wi_1_0_195

b.wi_1_0_1:                                       ; preds = %a.wi_1_0_188
  br label %d.wi_1_0_1

d.wi_1_0_1:                                       ; preds = %b.wi_1_0_1
  br label %a.wi_0_1_189

barrier.preheader.loopbarrier.prebarrier.wi_1_0_195: ; preds = %a.wi_1_0_188
  br label %unreachable73

unreachable73:                                    ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_1_0_195
  unreachable

a.wi_0_1_189:                                     ; preds = %d.wi_1_0_1
  br i1 true, label %b.wi_0_1_1, label %barrier.preheader.loopbarrier.prebarrier.wi_0_1_196

b.wi_0_1_1:                                       ; preds = %a.wi_0_1_189
  br label %d.wi_0_1_1

d.wi_0_1_1:                                       ; preds = %b.wi_0_1_1
  br label %a.wi_1_1_190

barrier.preheader.loopbarrier.prebarrier.wi_0_1_196: ; preds = %a.wi_0_1_189
  br label %unreachable78

unreachable78:                                    ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_0_1_196
  unreachable

a.wi_1_1_190:                                     ; preds = %d.wi_0_1_1
  br i1 true, label %b.wi_1_1_1, label %barrier.preheader.loopbarrier.prebarrier.wi_1_1_197

b.wi_1_1_1:                                       ; preds = %a.wi_1_1_190
  br label %d.wi_1_1_1

d.wi_1_1_1:                                       ; preds = %b.wi_1_1_1
  ret void

barrier.preheader.loopbarrier.prebarrier.wi_1_1_197: ; preds = %a.wi_1_1_190
  br label %unreachable83

unreachable83:                                    ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_1_1_197
  unreachable
}
