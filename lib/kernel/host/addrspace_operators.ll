target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define dso_local i32 @_Z9get_fencePU9CLgenericv(i8* %address) local_unnamed_addr #0 {
  ret i32 3
}

define dso_local i32 @_Z9get_fencePU9CLprivatev(i8* %address) local_unnamed_addr #0 {
  ret i32 3
}

define dso_local i32 @_Z9get_fencePU8CLglobalv(i8* %address) local_unnamed_addr #0 {
  ret i32 3
}

define dso_local i32 @_Z9get_fencePU7CLlocalv(i8* %address) local_unnamed_addr #0 {
  ret i32 3
}

define dso_local i32 @_Z9get_fencePU9CLgenericKv(i8* %address) local_unnamed_addr #0 {
  ret i32 3
}

define dso_local i32 @_Z9get_fencePU9CLprivateKv(i8* %address) local_unnamed_addr #0 {
  ret i32 3
}

define dso_local i32 @_Z9get_fencePU8CLglobalKv(i8* %address) local_unnamed_addr #0 {
  ret i32 3
}

define dso_local i32 @_Z9get_fencePU7CLlocalKv(i8* %address) local_unnamed_addr #0 {
  ret i32 3
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{i32 1, !"wchar_size", i32 4}
