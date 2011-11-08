declare void @barrier(i32 %flags)

define void @ifbarrier1() {

a:
  br i1 1, label %b, label %barrier

b:
  br label %c

barrier:
  call void @barrier(i32 0)
  br label %c

c:
  ret void
}