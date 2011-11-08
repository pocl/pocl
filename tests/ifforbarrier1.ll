declare void @barrier(i32 %flags)

define void @ifforbarrier1() {

a:
  br i1 1, label %b, label %barrier

b:
  br label %c

barrier:
  call void @barrier(i32 0)
  br i1 1, label %barrier, label %c

c:
  ret void
}