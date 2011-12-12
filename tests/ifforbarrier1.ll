declare void @pocl.barrier()

define void @ifforbarrier1() {

a:
  br i1 1, label %b, label %barrier

b:
  br label %d

barrier:
  call void @pocl.barrier()
  br label %c

c:
  br i1 1, label %barrier, label %d

d:
  ret void
}