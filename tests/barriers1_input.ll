declare void @pocl.barrier()

declare void @foo()

define void @barriers1() {
barrier:
  call void @foo()
  call void @pocl.barrier()
  call void @foo()
  ret void
}
