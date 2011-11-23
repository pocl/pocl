; ModuleID = 'ifbarrier1_btr_barriers.ll'

declare void @barrier(i32)

define void @ifbarrier1() {
a.wi_0_0_0:
  br i1 true, label %b.wi_0_0_0, label %barrier.prebarrier.wi_0_0_0

b.wi_0_0_0:                                       ; preds = %a.wi_0_0_0
  br label %c.wi_0_0_0

barrier.prebarrier.wi_0_0_0:                      ; preds = %a.wi_0_0_0
  br label %a.wi_1_0_0

barrier:                                          ; preds = %barrier.prebarrier.wi_1_1_1
  call void @barrier(i32 0)
  br label %c.btr.wi_0_0_0

c.wi_0_0_0:                                       ; preds = %b.wi_0_0_0
  br label %a.wi_1_0_070

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

a.wi_0_0_1:                                       ; preds = %barrier.prebarrier.wi_1_1_0
  br i1 true, label %unreachable11, label %barrier.prebarrier.wi_0_0_1

barrier.prebarrier.wi_0_0_1:                      ; preds = %a.wi_0_0_1
  br label %a.wi_1_0_1

unreachable11:                                    ; preds = %a.wi_0_0_1
  unreachable

a.wi_1_0_1:                                       ; preds = %barrier.prebarrier.wi_0_0_1
  br i1 true, label %unreachable14, label %barrier.prebarrier.wi_1_0_1

barrier.prebarrier.wi_1_0_1:                      ; preds = %a.wi_1_0_1
  br label %a.wi_0_1_1

unreachable14:                                    ; preds = %a.wi_1_0_1
  unreachable

a.wi_0_1_1:                                       ; preds = %barrier.prebarrier.wi_1_0_1
  br i1 true, label %unreachable17, label %barrier.prebarrier.wi_0_1_1

barrier.prebarrier.wi_0_1_1:                      ; preds = %a.wi_0_1_1
  br label %a.wi_1_1_1

unreachable17:                                    ; preds = %a.wi_0_1_1
  unreachable

a.wi_1_1_1:                                       ; preds = %barrier.prebarrier.wi_0_1_1
  br i1 true, label %unreachable20, label %barrier.prebarrier.wi_1_1_1

barrier.prebarrier.wi_1_1_1:                      ; preds = %a.wi_1_1_1
  br label %barrier

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

b.wi_1_0_0:                                       ; preds = %a.wi_1_0_070
  br label %c.wi_1_0_0

barrier.prebarrier.wi_1_0_063:                    ; preds = %a.wi_1_0_070
  br label %unreachable32

a.wi_1_0_070:                                     ; preds = %c.wi_0_0_0
  br i1 true, label %b.wi_1_0_0, label %barrier.prebarrier.wi_1_0_063

c.wi_1_0_0:                                       ; preds = %b.wi_1_0_0
  br label %a.wi_0_1_071

unreachable32:                                    ; preds = %barrier.prebarrier.wi_1_0_063
  unreachable

b.wi_0_1_0:                                       ; preds = %a.wi_0_1_071
  br label %c.wi_0_1_0

barrier.prebarrier.wi_0_1_064:                    ; preds = %a.wi_0_1_071
  br label %unreachable37

a.wi_0_1_071:                                     ; preds = %c.wi_1_0_0
  br i1 true, label %b.wi_0_1_0, label %barrier.prebarrier.wi_0_1_064

c.wi_0_1_0:                                       ; preds = %b.wi_0_1_0
  br label %a.wi_1_1_072

unreachable37:                                    ; preds = %barrier.prebarrier.wi_0_1_064
  unreachable

b.wi_1_1_0:                                       ; preds = %a.wi_1_1_072
  br label %c.wi_1_1_0

barrier.prebarrier.wi_1_1_065:                    ; preds = %a.wi_1_1_072
  br label %unreachable42

a.wi_1_1_072:                                     ; preds = %c.wi_0_1_0
  br i1 true, label %b.wi_1_1_0, label %barrier.prebarrier.wi_1_1_065

c.wi_1_1_0:                                       ; preds = %b.wi_1_1_0
  br label %a.wi_0_0_173

unreachable42:                                    ; preds = %barrier.prebarrier.wi_1_1_065
  unreachable

b.wi_0_0_1:                                       ; preds = %a.wi_0_0_173
  br label %c.wi_0_0_1

barrier.prebarrier.wi_0_0_166:                    ; preds = %a.wi_0_0_173
  br label %unreachable47

a.wi_0_0_173:                                     ; preds = %c.wi_1_1_0
  br i1 true, label %b.wi_0_0_1, label %barrier.prebarrier.wi_0_0_166

c.wi_0_0_1:                                       ; preds = %b.wi_0_0_1
  br label %a.wi_1_0_174

unreachable47:                                    ; preds = %barrier.prebarrier.wi_0_0_166
  unreachable

barrier.prebarrier.wi_1_0_167:                    ; preds = %a.wi_1_0_174
  br label %unreachable52

b.wi_1_0_1:                                       ; preds = %a.wi_1_0_174
  br label %c.wi_1_0_1

a.wi_1_0_174:                                     ; preds = %c.wi_0_0_1
  br i1 true, label %b.wi_1_0_1, label %barrier.prebarrier.wi_1_0_167

c.wi_1_0_1:                                       ; preds = %b.wi_1_0_1
  br label %a.wi_0_1_175

unreachable52:                                    ; preds = %barrier.prebarrier.wi_1_0_167
  unreachable

barrier.prebarrier.wi_0_1_168:                    ; preds = %a.wi_0_1_175
  br label %unreachable57

b.wi_0_1_1:                                       ; preds = %a.wi_0_1_175
  br label %c.wi_0_1_1

a.wi_0_1_175:                                     ; preds = %c.wi_1_0_1
  br i1 true, label %b.wi_0_1_1, label %barrier.prebarrier.wi_0_1_168

c.wi_0_1_1:                                       ; preds = %b.wi_0_1_1
  br label %a.wi_1_1_176

unreachable57:                                    ; preds = %barrier.prebarrier.wi_0_1_168
  unreachable

barrier.prebarrier.wi_1_1_169:                    ; preds = %a.wi_1_1_176
  br label %unreachable62

b.wi_1_1_1:                                       ; preds = %a.wi_1_1_176
  br label %c.wi_1_1_1

a.wi_1_1_176:                                     ; preds = %c.wi_0_1_1
  br i1 true, label %b.wi_1_1_1, label %barrier.prebarrier.wi_1_1_169

c.wi_1_1_1:                                       ; preds = %b.wi_1_1_1
  ret void

unreachable62:                                    ; preds = %barrier.prebarrier.wi_1_1_169
  unreachable
}
