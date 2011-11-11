declare void @barrier(i32 %flags)

define void @forbarrier1() {

a:
  br label %barrier

barrier:
  call void @barrier(i32 0)
  br i1 1, label %barrier, label %b

b:
  ret void
}