target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define dso_local i8 addrspace(3)* @__to_local(i8 addrspace(4)* %address) local_unnamed_addr #0 {
  %1 = addrspacecast i8 addrspace(4)* %address to i8 addrspace(3)*
  ret i8 addrspace(3)* %1
}

define dso_local i8 addrspace(1)* @__to_global(i8 addrspace(4)* %address) local_unnamed_addr #0 {
  %1 = addrspacecast i8 addrspace(4)* %address to i8 addrspace(1)*
  ret i8 addrspace(1)* %1
}

define dso_local i8* @__to_private(i8 addrspace(4)* %address) local_unnamed_addr #0 {
  %1 = addrspacecast i8 addrspace(4)* %address to i8*
  ret i8* %1
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{i32 1, !"wchar_size", i32 4}
