; ModuleID = 'ifforbarrier1_loops_btr_barriers.ll'

declare void @pocl.barrier()

define void @ifforbarrier1() {
a.wi_0_0_0:
  br i1 true, label %b.wi_0_0_0, label %barrier.preheader.loopbarrier.prebarrier.wi_0_0_0

barrier.preheader.loopbarrier.prebarrier.wi_0_0_0: ; preds = %a.wi_0_0_0
  br label %a.wi_1_0_0

barrier.preheader.loopbarrier:                    ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_1_1_1
  call void @pocl.barrier()
  br label %barrier.prebarrier.wi_0_0_0

b.wi_0_0_0:                                       ; preds = %a.wi_0_0_0
  br label %d.wi_0_0_0

barrier.prebarrier.wi_0_0_0:                      ; preds = %c.latchbarrier.postbarrier.wi_0_0_0, %barrier.preheader.loopbarrier
  br label %barrier.prebarrier.wi_1_0_0

barrier:                                          ; preds = %barrier.prebarrier.wi_1_1_1
  call void @pocl.barrier()
  br label %c.latchbarrier

c.latchbarrier:                                   ; preds = %barrier
  call void @pocl.barrier()
  br label %c.latchbarrier.postbarrier.wi_0_0_0

c.latchbarrier.postbarrier.wi_0_0_0:              ; preds = %c.latchbarrier
  br i1 true, label %barrier.prebarrier.wi_0_0_0, label %d.loopexit.wi_0_0_0

d.loopexit.wi_0_0_0:                              ; preds = %c.latchbarrier.postbarrier.wi_0_0_0
  br label %d.btr.wi_0_0_0

d.wi_0_0_0:                                       ; preds = %b.wi_0_0_0
  br label %a.wi_1_0_091

d.btr.wi_0_0_0:                                   ; preds = %d.loopexit.wi_0_0_0
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
  br i1 true, label %unreachable31, label %d.loopexit.wi_1_0_0

d.loopexit.wi_1_0_0:                              ; preds = %c.latchbarrier.postbarrier.wi_1_0_0
  br label %d.btr.wi_1_0_0

d.btr.wi_1_0_0:                                   ; preds = %d.loopexit.wi_1_0_0
  br label %c.latchbarrier.postbarrier.wi_0_1_0

unreachable31:                                    ; preds = %c.latchbarrier.postbarrier.wi_1_0_0
  unreachable

c.latchbarrier.postbarrier.wi_0_1_0:              ; preds = %d.btr.wi_1_0_0
  br i1 true, label %unreachable35, label %d.loopexit.wi_0_1_0

d.loopexit.wi_0_1_0:                              ; preds = %c.latchbarrier.postbarrier.wi_0_1_0
  br label %d.btr.wi_0_1_0

d.btr.wi_0_1_0:                                   ; preds = %d.loopexit.wi_0_1_0
  br label %c.latchbarrier.postbarrier.wi_1_1_0

unreachable35:                                    ; preds = %c.latchbarrier.postbarrier.wi_0_1_0
  unreachable

c.latchbarrier.postbarrier.wi_1_1_0:              ; preds = %d.btr.wi_0_1_0
  br i1 true, label %unreachable39, label %d.loopexit.wi_1_1_0

d.loopexit.wi_1_1_0:                              ; preds = %c.latchbarrier.postbarrier.wi_1_1_0
  br label %d.btr.wi_1_1_0

d.btr.wi_1_1_0:                                   ; preds = %d.loopexit.wi_1_1_0
  br label %c.latchbarrier.postbarrier.wi_0_0_1

unreachable39:                                    ; preds = %c.latchbarrier.postbarrier.wi_1_1_0
  unreachable

c.latchbarrier.postbarrier.wi_0_0_1:              ; preds = %d.btr.wi_1_1_0
  br i1 true, label %unreachable43, label %d.loopexit.wi_0_0_1

d.loopexit.wi_0_0_1:                              ; preds = %c.latchbarrier.postbarrier.wi_0_0_1
  br label %d.btr.wi_0_0_1

d.btr.wi_0_0_1:                                   ; preds = %d.loopexit.wi_0_0_1
  br label %c.latchbarrier.postbarrier.wi_1_0_1

unreachable43:                                    ; preds = %c.latchbarrier.postbarrier.wi_0_0_1
  unreachable

c.latchbarrier.postbarrier.wi_1_0_1:              ; preds = %d.btr.wi_0_0_1
  br i1 true, label %unreachable47, label %d.loopexit.wi_1_0_1

d.loopexit.wi_1_0_1:                              ; preds = %c.latchbarrier.postbarrier.wi_1_0_1
  br label %d.btr.wi_1_0_1

d.btr.wi_1_0_1:                                   ; preds = %d.loopexit.wi_1_0_1
  br label %c.latchbarrier.postbarrier.wi_0_1_1

unreachable47:                                    ; preds = %c.latchbarrier.postbarrier.wi_1_0_1
  unreachable

c.latchbarrier.postbarrier.wi_0_1_1:              ; preds = %d.btr.wi_1_0_1
  br i1 true, label %unreachable51, label %d.loopexit.wi_0_1_1

d.loopexit.wi_0_1_1:                              ; preds = %c.latchbarrier.postbarrier.wi_0_1_1
  br label %d.btr.wi_0_1_1

d.btr.wi_0_1_1:                                   ; preds = %d.loopexit.wi_0_1_1
  br label %c.latchbarrier.postbarrier.wi_1_1_1

unreachable51:                                    ; preds = %c.latchbarrier.postbarrier.wi_0_1_1
  unreachable

c.latchbarrier.postbarrier.wi_1_1_1:              ; preds = %d.btr.wi_0_1_1
  br i1 true, label %unreachable55, label %d.loopexit.wi_1_1_1

d.loopexit.wi_1_1_1:                              ; preds = %c.latchbarrier.postbarrier.wi_1_1_1
  br label %d.btr.wi_1_1_1

d.btr.wi_1_1_1:                                   ; preds = %d.loopexit.wi_1_1_1
  ret void

unreachable55:                                    ; preds = %c.latchbarrier.postbarrier.wi_1_1_1
  unreachable

a.wi_1_0_091:                                     ; preds = %d.wi_0_0_0
  br i1 true, label %b.wi_1_0_0, label %barrier.preheader.loopbarrier.prebarrier.wi_1_0_098

b.wi_1_0_0:                                       ; preds = %a.wi_1_0_091
  br label %d.wi_1_0_0

d.wi_1_0_0:                                       ; preds = %b.wi_1_0_0
  br label %a.wi_0_1_092

barrier.preheader.loopbarrier.prebarrier.wi_1_0_098: ; preds = %a.wi_1_0_091
  br label %unreachable60

unreachable60:                                    ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_1_0_098
  unreachable

a.wi_0_1_092:                                     ; preds = %d.wi_1_0_0
  br i1 true, label %b.wi_0_1_0, label %barrier.preheader.loopbarrier.prebarrier.wi_0_1_099

b.wi_0_1_0:                                       ; preds = %a.wi_0_1_092
  br label %d.wi_0_1_0

d.wi_0_1_0:                                       ; preds = %b.wi_0_1_0
  br label %a.wi_1_1_093

barrier.preheader.loopbarrier.prebarrier.wi_0_1_099: ; preds = %a.wi_0_1_092
  br label %unreachable65

unreachable65:                                    ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_0_1_099
  unreachable

a.wi_1_1_093:                                     ; preds = %d.wi_0_1_0
  br i1 true, label %b.wi_1_1_0, label %barrier.preheader.loopbarrier.prebarrier.wi_1_1_0100

b.wi_1_1_0:                                       ; preds = %a.wi_1_1_093
  br label %d.wi_1_1_0

d.wi_1_1_0:                                       ; preds = %b.wi_1_1_0
  br label %a.wi_0_0_194

barrier.preheader.loopbarrier.prebarrier.wi_1_1_0100: ; preds = %a.wi_1_1_093
  br label %unreachable70

unreachable70:                                    ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_1_1_0100
  unreachable

a.wi_0_0_194:                                     ; preds = %d.wi_1_1_0
  br i1 true, label %b.wi_0_0_1, label %barrier.preheader.loopbarrier.prebarrier.wi_0_0_1101

b.wi_0_0_1:                                       ; preds = %a.wi_0_0_194
  br label %d.wi_0_0_1

d.wi_0_0_1:                                       ; preds = %b.wi_0_0_1
  br label %a.wi_1_0_195

barrier.preheader.loopbarrier.prebarrier.wi_0_0_1101: ; preds = %a.wi_0_0_194
  br label %unreachable75

unreachable75:                                    ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_0_0_1101
  unreachable

a.wi_1_0_195:                                     ; preds = %d.wi_0_0_1
  br i1 true, label %b.wi_1_0_1, label %barrier.preheader.loopbarrier.prebarrier.wi_1_0_1102

b.wi_1_0_1:                                       ; preds = %a.wi_1_0_195
  br label %d.wi_1_0_1

d.wi_1_0_1:                                       ; preds = %b.wi_1_0_1
  br label %a.wi_0_1_196

barrier.preheader.loopbarrier.prebarrier.wi_1_0_1102: ; preds = %a.wi_1_0_195
  br label %unreachable80

unreachable80:                                    ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_1_0_1102
  unreachable

a.wi_0_1_196:                                     ; preds = %d.wi_1_0_1
  br i1 true, label %b.wi_0_1_1, label %barrier.preheader.loopbarrier.prebarrier.wi_0_1_1103

b.wi_0_1_1:                                       ; preds = %a.wi_0_1_196
  br label %d.wi_0_1_1

d.wi_0_1_1:                                       ; preds = %b.wi_0_1_1
  br label %a.wi_1_1_197

barrier.preheader.loopbarrier.prebarrier.wi_0_1_1103: ; preds = %a.wi_0_1_196
  br label %unreachable85

unreachable85:                                    ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_0_1_1103
  unreachable

a.wi_1_1_197:                                     ; preds = %d.wi_0_1_1
  br i1 true, label %b.wi_1_1_1, label %barrier.preheader.loopbarrier.prebarrier.wi_1_1_1104

b.wi_1_1_1:                                       ; preds = %a.wi_1_1_197
  br label %d.wi_1_1_1

d.wi_1_1_1:                                       ; preds = %b.wi_1_1_1
  ret void

barrier.preheader.loopbarrier.prebarrier.wi_1_1_1104: ; preds = %a.wi_1_1_197
  br label %unreachable90

unreachable90:                                    ; preds = %barrier.preheader.loopbarrier.prebarrier.wi_1_1_1104
  unreachable
}
