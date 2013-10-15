declare void @_pocl_barrier()

; Use noduplicate to avoid unwanted (illegal in OpenCL C semantics)
; code motion / replication of barriers.
define void @barrier(i32 %flags) noduplicate {
entry:
  call void @_pocl_barrier()
  ret void
}
