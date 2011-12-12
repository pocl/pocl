declare void @pocl.barrier()

define void @ifbarrier1() {

a:
  br i1 1, label %b, label %barrier

b:
  br label %c

barrier:
  call void @pocl.barrier()
  br label %c

c:
  ret void
}