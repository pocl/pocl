declare void @barrier(i32 %flags)

define void @forbarrier2() {
a:
  br label %b

b:
  br label %barrier

barrier:
  call void @barrier(i32 0)
  br i1 1, label %barrier, label %c

c:
  br i1 1, label %b, label %d

d:
  ret void
}
