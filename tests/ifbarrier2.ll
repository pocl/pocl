declare void @barrier(i32 %flags)

define void @ifbarrier2() {

a:
  br i1 1, label %b, label %c

b:
  br label %f

c:
  br i1 1, label %d, label %barrier

d:
  br label %e

barrier:
  call void @barrier(i32 0)
  br label %e

e:
  br label %f

f:
  ret void
}