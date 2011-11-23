declare void @barrier(i32 %flags)

define void @ifbarrier4() {

a:
  br i1 1, label %b, label %barrier

barrier:
  call void @barrier(i32 0)
  br label %c

b:
  br i1 1, label %d, label %c

c:
  ret void

d:
  ret void
}