declare void @pocl.barrier()

define void @forifbarrier1() {

a:
  br label %b

b:
  br i1 1, label %c, label %barrier

c:
  br label %d

barrier:
  call void @pocl.barrier()
  br label %d

d:
  br i1 1, label %b, label %e

e:
  ret void
}