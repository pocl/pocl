; ModuleID = 'forifbarrier1_loops_btr_barriers.ll'

declare void @barrier(i32)

define void @forifbarrier1() {
a.loopbarrier.prebarrier.wi_0_0_0:
  br label %a.loopbarrier.prebarrier.wi_1_0_0

a.loopbarrier:                                    ; preds = %a.loopbarrier.prebarrier.wi_1_1_1
  call void @barrier(i32 0)
  br label %b.wi_0_0_0

b.wi_0_0_0:                                       ; preds = %d.latchbarrier.btr.postbarrier.wi_0_0_0, %d.latchbarrier.postbarrier.wi_0_0_0, %a.loopbarrier
  br i1 true, label %c.wi_0_0_0, label %barrier.prebarrier.wi_0_0_0

c.wi_0_0_0:                                       ; preds = %b.wi_0_0_0
  br label %b.wi_1_0_0

barrier.prebarrier.wi_0_0_0:                      ; preds = %b.wi_0_0_0
  br label %b.wi_1_0_070

barrier:                                          ; preds = %barrier.prebarrier.wi_1_1_1
  call void @barrier(i32 0)
  br label %d.latchbarrier.btr

d.latchbarrier:                                   ; preds = %c.wi_1_1_1
  call void @barrier(i32 0)
  br label %d.latchbarrier.postbarrier.wi_0_0_0

d.latchbarrier.postbarrier.wi_0_0_0:              ; preds = %d.latchbarrier
  br i1 true, label %b.wi_0_0_0, label %e.wi_0_0_0

e.wi_0_0_0:                                       ; preds = %d.latchbarrier.postbarrier.wi_0_0_0
  br label %d.latchbarrier.postbarrier.wi_1_0_0

d.latchbarrier.btr:                               ; preds = %barrier
  call void @barrier(i32 0)
  br label %d.latchbarrier.btr.postbarrier.wi_0_0_0

d.latchbarrier.btr.postbarrier.wi_0_0_0:          ; preds = %d.latchbarrier.btr
  br i1 true, label %b.wi_0_0_0, label %e.btr.wi_0_0_0

e.btr.wi_0_0_0:                                   ; preds = %d.latchbarrier.btr.postbarrier.wi_0_0_0
  br label %d.latchbarrier.btr.postbarrier.wi_1_0_0

a.loopbarrier.prebarrier.wi_1_0_0:                ; preds = %a.loopbarrier.prebarrier.wi_0_0_0
  br label %a.loopbarrier.prebarrier.wi_0_1_0

a.loopbarrier.prebarrier.wi_0_1_0:                ; preds = %a.loopbarrier.prebarrier.wi_1_0_0
  br label %a.loopbarrier.prebarrier.wi_1_1_0

a.loopbarrier.prebarrier.wi_1_1_0:                ; preds = %a.loopbarrier.prebarrier.wi_0_1_0
  br label %a.loopbarrier.prebarrier.wi_0_0_1

a.loopbarrier.prebarrier.wi_0_0_1:                ; preds = %a.loopbarrier.prebarrier.wi_1_1_0
  br label %a.loopbarrier.prebarrier.wi_1_0_1

a.loopbarrier.prebarrier.wi_1_0_1:                ; preds = %a.loopbarrier.prebarrier.wi_0_0_1
  br label %a.loopbarrier.prebarrier.wi_0_1_1

a.loopbarrier.prebarrier.wi_0_1_1:                ; preds = %a.loopbarrier.prebarrier.wi_1_0_1
  br label %a.loopbarrier.prebarrier.wi_1_1_1

a.loopbarrier.prebarrier.wi_1_1_1:                ; preds = %a.loopbarrier.prebarrier.wi_0_1_1
  br label %a.loopbarrier

c.wi_1_0_0:                                       ; preds = %b.wi_1_0_0
  br label %b.wi_0_1_0

b.wi_1_0_0:                                       ; preds = %c.wi_0_0_0
  br i1 true, label %c.wi_1_0_0, label %unreachable

unreachable:                                      ; preds = %b.wi_1_0_0
  unreachable

c.wi_0_1_0:                                       ; preds = %b.wi_0_1_0
  br label %b.wi_1_1_0

b.wi_0_1_0:                                       ; preds = %c.wi_1_0_0
  br i1 true, label %c.wi_0_1_0, label %unreachable12

unreachable12:                                    ; preds = %b.wi_0_1_0
  unreachable

c.wi_1_1_0:                                       ; preds = %b.wi_1_1_0
  br label %b.wi_0_0_1

b.wi_1_1_0:                                       ; preds = %c.wi_0_1_0
  br i1 true, label %c.wi_1_1_0, label %unreachable15

unreachable15:                                    ; preds = %b.wi_1_1_0
  unreachable

c.wi_0_0_1:                                       ; preds = %b.wi_0_0_1
  br label %b.wi_1_0_1

b.wi_0_0_1:                                       ; preds = %c.wi_1_1_0
  br i1 true, label %c.wi_0_0_1, label %unreachable18

unreachable18:                                    ; preds = %b.wi_0_0_1
  unreachable

c.wi_1_0_1:                                       ; preds = %b.wi_1_0_1
  br label %b.wi_0_1_1

b.wi_1_0_1:                                       ; preds = %c.wi_0_0_1
  br i1 true, label %c.wi_1_0_1, label %unreachable21

unreachable21:                                    ; preds = %b.wi_1_0_1
  unreachable

c.wi_0_1_1:                                       ; preds = %b.wi_0_1_1
  br label %b.wi_1_1_1

b.wi_0_1_1:                                       ; preds = %c.wi_1_0_1
  br i1 true, label %c.wi_0_1_1, label %unreachable24

unreachable24:                                    ; preds = %b.wi_0_1_1
  unreachable

c.wi_1_1_1:                                       ; preds = %b.wi_1_1_1
  br label %d.latchbarrier

b.wi_1_1_1:                                       ; preds = %c.wi_0_1_1
  br i1 true, label %c.wi_1_1_1, label %unreachable27

unreachable27:                                    ; preds = %b.wi_1_1_1
  unreachable

d.latchbarrier.postbarrier.wi_1_0_0:              ; preds = %e.wi_0_0_0
  br i1 true, label %unreachable30, label %e.wi_1_0_0

e.wi_1_0_0:                                       ; preds = %d.latchbarrier.postbarrier.wi_1_0_0
  br label %d.latchbarrier.postbarrier.wi_0_1_0

unreachable30:                                    ; preds = %d.latchbarrier.postbarrier.wi_1_0_0
  unreachable

d.latchbarrier.postbarrier.wi_0_1_0:              ; preds = %e.wi_1_0_0
  br i1 true, label %unreachable33, label %e.wi_0_1_0

e.wi_0_1_0:                                       ; preds = %d.latchbarrier.postbarrier.wi_0_1_0
  br label %d.latchbarrier.postbarrier.wi_1_1_0

unreachable33:                                    ; preds = %d.latchbarrier.postbarrier.wi_0_1_0
  unreachable

d.latchbarrier.postbarrier.wi_1_1_0:              ; preds = %e.wi_0_1_0
  br i1 true, label %unreachable36, label %e.wi_1_1_0

e.wi_1_1_0:                                       ; preds = %d.latchbarrier.postbarrier.wi_1_1_0
  br label %d.latchbarrier.postbarrier.wi_0_0_1

unreachable36:                                    ; preds = %d.latchbarrier.postbarrier.wi_1_1_0
  unreachable

d.latchbarrier.postbarrier.wi_0_0_1:              ; preds = %e.wi_1_1_0
  br i1 true, label %unreachable39, label %e.wi_0_0_1

e.wi_0_0_1:                                       ; preds = %d.latchbarrier.postbarrier.wi_0_0_1
  br label %d.latchbarrier.postbarrier.wi_1_0_1

unreachable39:                                    ; preds = %d.latchbarrier.postbarrier.wi_0_0_1
  unreachable

d.latchbarrier.postbarrier.wi_1_0_1:              ; preds = %e.wi_0_0_1
  br i1 true, label %unreachable42, label %e.wi_1_0_1

e.wi_1_0_1:                                       ; preds = %d.latchbarrier.postbarrier.wi_1_0_1
  br label %d.latchbarrier.postbarrier.wi_0_1_1

unreachable42:                                    ; preds = %d.latchbarrier.postbarrier.wi_1_0_1
  unreachable

d.latchbarrier.postbarrier.wi_0_1_1:              ; preds = %e.wi_1_0_1
  br i1 true, label %unreachable45, label %e.wi_0_1_1

e.wi_0_1_1:                                       ; preds = %d.latchbarrier.postbarrier.wi_0_1_1
  br label %d.latchbarrier.postbarrier.wi_1_1_1

unreachable45:                                    ; preds = %d.latchbarrier.postbarrier.wi_0_1_1
  unreachable

d.latchbarrier.postbarrier.wi_1_1_1:              ; preds = %e.wi_0_1_1
  br i1 true, label %unreachable48, label %e.wi_1_1_1

e.wi_1_1_1:                                       ; preds = %d.latchbarrier.postbarrier.wi_1_1_1
  ret void

unreachable48:                                    ; preds = %d.latchbarrier.postbarrier.wi_1_1_1
  unreachable

barrier.prebarrier.wi_1_0_0:                      ; preds = %b.wi_1_0_070
  br label %b.wi_0_1_071

b.wi_1_0_070:                                     ; preds = %barrier.prebarrier.wi_0_0_0
  br i1 true, label %unreachable51, label %barrier.prebarrier.wi_1_0_0

unreachable51:                                    ; preds = %b.wi_1_0_070
  unreachable

barrier.prebarrier.wi_0_1_0:                      ; preds = %b.wi_0_1_071
  br label %b.wi_1_1_072

b.wi_0_1_071:                                     ; preds = %barrier.prebarrier.wi_1_0_0
  br i1 true, label %unreachable54, label %barrier.prebarrier.wi_0_1_0

unreachable54:                                    ; preds = %b.wi_0_1_071
  unreachable

barrier.prebarrier.wi_1_1_0:                      ; preds = %b.wi_1_1_072
  br label %b.wi_0_0_173

b.wi_1_1_072:                                     ; preds = %barrier.prebarrier.wi_0_1_0
  br i1 true, label %unreachable57, label %barrier.prebarrier.wi_1_1_0

unreachable57:                                    ; preds = %b.wi_1_1_072
  unreachable

barrier.prebarrier.wi_0_0_1:                      ; preds = %b.wi_0_0_173
  br label %b.wi_1_0_174

b.wi_0_0_173:                                     ; preds = %barrier.prebarrier.wi_1_1_0
  br i1 true, label %unreachable60, label %barrier.prebarrier.wi_0_0_1

unreachable60:                                    ; preds = %b.wi_0_0_173
  unreachable

barrier.prebarrier.wi_1_0_1:                      ; preds = %b.wi_1_0_174
  br label %b.wi_0_1_175

b.wi_1_0_174:                                     ; preds = %barrier.prebarrier.wi_0_0_1
  br i1 true, label %unreachable63, label %barrier.prebarrier.wi_1_0_1

unreachable63:                                    ; preds = %b.wi_1_0_174
  unreachable

barrier.prebarrier.wi_0_1_1:                      ; preds = %b.wi_0_1_175
  br label %b.wi_1_1_176

b.wi_0_1_175:                                     ; preds = %barrier.prebarrier.wi_1_0_1
  br i1 true, label %unreachable66, label %barrier.prebarrier.wi_0_1_1

unreachable66:                                    ; preds = %b.wi_0_1_175
  unreachable

barrier.prebarrier.wi_1_1_1:                      ; preds = %b.wi_1_1_176
  br label %barrier

b.wi_1_1_176:                                     ; preds = %barrier.prebarrier.wi_0_1_1
  br i1 true, label %unreachable69, label %barrier.prebarrier.wi_1_1_1

unreachable69:                                    ; preds = %b.wi_1_1_176
  unreachable

d.latchbarrier.btr.postbarrier.wi_1_0_0:          ; preds = %e.btr.wi_0_0_0
  br i1 true, label %unreachable79, label %e.btr.wi_1_0_0

e.btr.wi_1_0_0:                                   ; preds = %d.latchbarrier.btr.postbarrier.wi_1_0_0
  br label %d.latchbarrier.btr.postbarrier.wi_0_1_0

unreachable79:                                    ; preds = %d.latchbarrier.btr.postbarrier.wi_1_0_0
  unreachable

d.latchbarrier.btr.postbarrier.wi_0_1_0:          ; preds = %e.btr.wi_1_0_0
  br i1 true, label %unreachable82, label %e.btr.wi_0_1_0

e.btr.wi_0_1_0:                                   ; preds = %d.latchbarrier.btr.postbarrier.wi_0_1_0
  br label %d.latchbarrier.btr.postbarrier.wi_1_1_0

unreachable82:                                    ; preds = %d.latchbarrier.btr.postbarrier.wi_0_1_0
  unreachable

d.latchbarrier.btr.postbarrier.wi_1_1_0:          ; preds = %e.btr.wi_0_1_0
  br i1 true, label %unreachable85, label %e.btr.wi_1_1_0

e.btr.wi_1_1_0:                                   ; preds = %d.latchbarrier.btr.postbarrier.wi_1_1_0
  br label %d.latchbarrier.btr.postbarrier.wi_0_0_1

unreachable85:                                    ; preds = %d.latchbarrier.btr.postbarrier.wi_1_1_0
  unreachable

d.latchbarrier.btr.postbarrier.wi_0_0_1:          ; preds = %e.btr.wi_1_1_0
  br i1 true, label %unreachable88, label %e.btr.wi_0_0_1

e.btr.wi_0_0_1:                                   ; preds = %d.latchbarrier.btr.postbarrier.wi_0_0_1
  br label %d.latchbarrier.btr.postbarrier.wi_1_0_1

unreachable88:                                    ; preds = %d.latchbarrier.btr.postbarrier.wi_0_0_1
  unreachable

d.latchbarrier.btr.postbarrier.wi_1_0_1:          ; preds = %e.btr.wi_0_0_1
  br i1 true, label %unreachable91, label %e.btr.wi_1_0_1

e.btr.wi_1_0_1:                                   ; preds = %d.latchbarrier.btr.postbarrier.wi_1_0_1
  br label %d.latchbarrier.btr.postbarrier.wi_0_1_1

unreachable91:                                    ; preds = %d.latchbarrier.btr.postbarrier.wi_1_0_1
  unreachable

d.latchbarrier.btr.postbarrier.wi_0_1_1:          ; preds = %e.btr.wi_1_0_1
  br i1 true, label %unreachable94, label %e.btr.wi_0_1_1

e.btr.wi_0_1_1:                                   ; preds = %d.latchbarrier.btr.postbarrier.wi_0_1_1
  br label %d.latchbarrier.btr.postbarrier.wi_1_1_1

unreachable94:                                    ; preds = %d.latchbarrier.btr.postbarrier.wi_0_1_1
  unreachable

d.latchbarrier.btr.postbarrier.wi_1_1_1:          ; preds = %e.btr.wi_0_1_1
  br i1 true, label %unreachable97, label %e.btr.wi_1_1_1

e.btr.wi_1_1_1:                                   ; preds = %d.latchbarrier.btr.postbarrier.wi_1_1_1
  ret void

unreachable97:                                    ; preds = %d.latchbarrier.btr.postbarrier.wi_1_1_1
  unreachable
}
