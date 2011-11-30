; ModuleID = 'nobarrier1.ll'

define void @nobarrier1() {
a.wi_0_0_0:
  br label %b.wi_0_0_0

b.wi_0_0_0:                                       ; preds = %a.wi_0_0_0
  br label %a.wi_1_0_0

a.wi_1_0_0:                                       ; preds = %b.wi_0_0_0
  br label %b.wi_1_0_0

b.wi_1_0_0:                                       ; preds = %a.wi_1_0_0
  br label %a.wi_0_1_0

a.wi_0_1_0:                                       ; preds = %b.wi_1_0_0
  br label %b.wi_0_1_0

b.wi_0_1_0:                                       ; preds = %a.wi_0_1_0
  br label %a.wi_1_1_0

a.wi_1_1_0:                                       ; preds = %b.wi_0_1_0
  br label %b.wi_1_1_0

b.wi_1_1_0:                                       ; preds = %a.wi_1_1_0
  br label %a.wi_0_0_1

a.wi_0_0_1:                                       ; preds = %b.wi_1_1_0
  br label %b.wi_0_0_1

b.wi_0_0_1:                                       ; preds = %a.wi_0_0_1
  br label %a.wi_1_0_1

a.wi_1_0_1:                                       ; preds = %b.wi_0_0_1
  br label %b.wi_1_0_1

b.wi_1_0_1:                                       ; preds = %a.wi_1_0_1
  br label %a.wi_0_1_1

a.wi_0_1_1:                                       ; preds = %b.wi_1_0_1
  br label %b.wi_0_1_1

b.wi_0_1_1:                                       ; preds = %a.wi_0_1_1
  br label %a.wi_1_1_1

a.wi_1_1_1:                                       ; preds = %b.wi_0_1_1
  br label %b.wi_1_1_1

b.wi_1_1_1:                                       ; preds = %a.wi_1_1_1
  ret void
}
