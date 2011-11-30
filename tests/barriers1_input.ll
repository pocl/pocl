declare void @barrier(i32 %flags)

declare void @foo()

define void @barrier1() {
barrier:
  call void @foo()
  call void @barrier(i32 0)
  call void @foo()
  ret void
}
