declare void @pocl.barrier()

define void @barrier1() {

barrier:
  call void @pocl.barrier()
  ret void
}