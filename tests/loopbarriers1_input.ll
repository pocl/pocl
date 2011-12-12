declare void @pocl.barrier()

define void @forbarrier1() {
a:
  br label %barrier

barrier:
  call void @pocl.barrier()
  br i1 1, label %barrier, label %b

b:
  ret void
}
