declare void @pocl.barrier()

; Use noduplicate to avoid unwanted (illegal in OpenCL C semantics)
; code motion / replication of barriers.
define void @barrier(i32 %flags) noduplicate {
entry:
  call void @pocl.barrier()
  ret void
}
