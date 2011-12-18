; ModuleID = 'ifbarrier1_btr_barriers.ll'

declare void @pocl.barrier()

define void @ifbarrier1() {
a.wi_0_0_0:
  br i1 true, label %b.wi_0_0_0, label %barrier.prebarrier.wi_0_0_0

b.wi_0_0_0:                                       ; preds = %a.wi_0_0_0
  br label %c.wi_0_0_0

barrier.prebarrier.wi_0_0_0:                      ; preds = %a.wi_0_0_0
  br label %a.wi_1_0_0

barrier:                                          ; preds = %barrier.prebarrier.wi_1_1_1
  call void @pocl.barrier()
  br label %c.btr.wi_0_0_0

c.wi_0_0_0:                                       ; preds = %b.wi_0_0_0
  br label %a.wi_1_0_063

c.btr.wi_0_0_0:                                   ; preds = %barrier
  br label %c.btr.wi_1_0_0

barrier.prebarrier.wi_1_0_0:                      ; preds = %a.wi_1_0_0
  br label %a.wi_0_1_0

a.wi_1_0_0:                                       ; preds = %barrier.prebarrier.wi_0_0_0
  br i1 true, label %unreachable, label %barrier.prebarrier.wi_1_0_0

unreachable:                                      ; preds = %a.wi_1_0_0
  unreachable

barrier.prebarrier.wi_0_1_0:                      ; preds = %a.wi_0_1_0
  br label %a.wi_1_1_0

a.wi_0_1_0:                                       ; preds = %barrier.prebarrier.wi_1_0_0
  br i1 true, label %unreachable5, label %barrier.prebarrier.wi_0_1_0

unreachable5:                                     ; preds = %a.wi_0_1_0
  unreachable

barrier.prebarrier.wi_1_1_0:                      ; preds = %a.wi_1_1_0
  br label %a.wi_0_0_1

a.wi_1_1_0:                                       ; preds = %barrier.prebarrier.wi_0_1_0
  br i1 true, label %unreachable8, label %barrier.prebarrier.wi_1_1_0

unreachable8:                                     ; preds = %a.wi_1_1_0
  unreachable

barrier.prebarrier.wi_0_0_1:                      ; preds = %a.wi_0_0_1
  br label %a.wi_1_0_1

a.wi_0_0_1:                                       ; preds = %barrier.prebarrier.wi_1_1_0
  br i1 true, label %unreachable11, label %barrier.prebarrier.wi_0_0_1

unreachable11:                                    ; preds = %a.wi_0_0_1
  unreachable

barrier.prebarrier.wi_1_0_1:                      ; preds = %a.wi_1_0_1
  br label %a.wi_0_1_1

a.wi_1_0_1:                                       ; preds = %barrier.prebarrier.wi_0_0_1
  br i1 true, label %unreachable14, label %barrier.prebarrier.wi_1_0_1

unreachable14:                                    ; preds = %a.wi_1_0_1
  unreachable

barrier.prebarrier.wi_0_1_1:                      ; preds = %a.wi_0_1_1
  br label %a.wi_1_1_1

a.wi_0_1_1:                                       ; preds = %barrier.prebarrier.wi_1_0_1
  br i1 true, label %unreachable17, label %barrier.prebarrier.wi_0_1_1

unreachable17:                                    ; preds = %a.wi_0_1_1
  unreachable

barrier.prebarrier.wi_1_1_1:                      ; preds = %a.wi_1_1_1
  br label %barrier

a.wi_1_1_1:                                       ; preds = %barrier.prebarrier.wi_0_1_1
  br i1 true, label %unreachable20, label %barrier.prebarrier.wi_1_1_1

unreachable20:                                    ; preds = %a.wi_1_1_1
  unreachable

c.btr.wi_1_0_0:                                   ; preds = %c.btr.wi_0_0_0
  br label %c.btr.wi_0_1_0

c.btr.wi_0_1_0:                                   ; preds = %c.btr.wi_1_0_0
  br label %c.btr.wi_1_1_0

c.btr.wi_1_1_0:                                   ; preds = %c.btr.wi_0_1_0
  br label %c.btr.wi_0_0_1

c.btr.wi_0_0_1:                                   ; preds = %c.btr.wi_1_1_0
  br label %c.btr.wi_1_0_1

c.btr.wi_1_0_1:                                   ; preds = %c.btr.wi_0_0_1
  br label %c.btr.wi_0_1_1

c.btr.wi_0_1_1:                                   ; preds = %c.btr.wi_1_0_1
  br label %c.btr.wi_1_1_1

c.btr.wi_1_1_1:                                   ; preds = %c.btr.wi_0_1_1
  ret void

a.wi_1_0_063:                                     ; preds = %c.wi_0_0_0
  br i1 true, label %b.wi_1_0_0, label %barrier.prebarrier.wi_1_0_070

b.wi_1_0_0:                                       ; preds = %a.wi_1_0_063
  br label %c.wi_1_0_0

c.wi_1_0_0:                                       ; preds = %b.wi_1_0_0
  br label %a.wi_0_1_064

barrier.prebarrier.wi_1_0_070:                    ; preds = %a.wi_1_0_063
  br label %unreachable32

unreachable32:                                    ; preds = %barrier.prebarrier.wi_1_0_070
  unreachable

a.wi_0_1_064:                                     ; preds = %c.wi_1_0_0
  br i1 true, label %b.wi_0_1_0, label %barrier.prebarrier.wi_0_1_071

b.wi_0_1_0:                                       ; preds = %a.wi_0_1_064
  br label %c.wi_0_1_0

c.wi_0_1_0:                                       ; preds = %b.wi_0_1_0
  br label %a.wi_1_1_065

barrier.prebarrier.wi_0_1_071:                    ; preds = %a.wi_0_1_064
  br label %unreachable37

unreachable37:                                    ; preds = %barrier.prebarrier.wi_0_1_071
  unreachable

a.wi_1_1_065:                                     ; preds = %c.wi_0_1_0
  br i1 true, label %b.wi_1_1_0, label %barrier.prebarrier.wi_1_1_072

b.wi_1_1_0:                                       ; preds = %a.wi_1_1_065
  br label %c.wi_1_1_0

c.wi_1_1_0:                                       ; preds = %b.wi_1_1_0
  br label %a.wi_0_0_166

barrier.prebarrier.wi_1_1_072:                    ; preds = %a.wi_1_1_065
  br label %unreachable42

unreachable42:                                    ; preds = %barrier.prebarrier.wi_1_1_072
  unreachable

a.wi_0_0_166:                                     ; preds = %c.wi_1_1_0
  br i1 true, label %b.wi_0_0_1, label %barrier.prebarrier.wi_0_0_173

b.wi_0_0_1:                                       ; preds = %a.wi_0_0_166
  br label %c.wi_0_0_1

c.wi_0_0_1:                                       ; preds = %b.wi_0_0_1
  br label %a.wi_1_0_167

barrier.prebarrier.wi_0_0_173:                    ; preds = %a.wi_0_0_166
  br label %unreachable47

unreachable47:                                    ; preds = %barrier.prebarrier.wi_0_0_173
  unreachable

a.wi_1_0_167:                                     ; preds = %c.wi_0_0_1
  br i1 true, label %b.wi_1_0_1, label %barrier.prebarrier.wi_1_0_174

b.wi_1_0_1:                                       ; preds = %a.wi_1_0_167
  br label %c.wi_1_0_1

c.wi_1_0_1:                                       ; preds = %b.wi_1_0_1
  br label %a.wi_0_1_168

barrier.prebarrier.wi_1_0_174:                    ; preds = %a.wi_1_0_167
  br label %unreachable52

unreachable52:                                    ; preds = %barrier.prebarrier.wi_1_0_174
  unreachable

a.wi_0_1_168:                                     ; preds = %c.wi_1_0_1
  br i1 true, label %b.wi_0_1_1, label %barrier.prebarrier.wi_0_1_175

b.wi_0_1_1:                                       ; preds = %a.wi_0_1_168
  br label %c.wi_0_1_1

c.wi_0_1_1:                                       ; preds = %b.wi_0_1_1
  br label %a.wi_1_1_169

barrier.prebarrier.wi_0_1_175:                    ; preds = %a.wi_0_1_168
  br label %unreachable57

unreachable57:                                    ; preds = %barrier.prebarrier.wi_0_1_175
  unreachable

a.wi_1_1_169:                                     ; preds = %c.wi_0_1_1
  br i1 true, label %b.wi_1_1_1, label %barrier.prebarrier.wi_1_1_176

b.wi_1_1_1:                                       ; preds = %a.wi_1_1_169
  br label %c.wi_1_1_1

c.wi_1_1_1:                                       ; preds = %b.wi_1_1_1
  ret void

barrier.prebarrier.wi_1_1_176:                    ; preds = %a.wi_1_1_169
  br label %unreachable62

unreachable62:                                    ; preds = %barrier.prebarrier.wi_1_1_176
  unreachable
}
