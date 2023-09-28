define i32 @_Z31__pocl_atomic_fetch_max__globalPU3AS1VU7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw max i32 addrspace(1)* %0, i32 %1 monotonic
  ret i32 %res
}

define i32 @_Z31__pocl_atomic_fetch_max__globalPU3AS1VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw umax i32 addrspace(1)* %0, i32 %1 monotonic
  ret i32 %res
}

define i64 @_Z31__pocl_atomic_fetch_max__globalPU3AS1VU7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw max i64 addrspace(1)* %0, i64 %1 monotonic
  ret i64 %res
}

define i64 @_Z31__pocl_atomic_fetch_max__globalPU3AS1VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw umax i64 addrspace(1)* %0, i64 %1 monotonic
  ret i64 %res
}

define i32 @_Z30__pocl_atomic_fetch_max__localPU3AS3VU7_Atomicii12memory_order12memory_scope(i32 addrspace(3)* noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw max i32 addrspace(3)* %0, i32 %1 monotonic
  ret i32 %res
}

define i32 @_Z30__pocl_atomic_fetch_max__localPU3AS3VU7_Atomicjj12memory_order12memory_scope(i32 addrspace(3)* noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw umax i32 addrspace(3)* %0, i32 %1 monotonic
  ret i32 %res
}

define i64 @_Z30__pocl_atomic_fetch_max__localPU3AS3VU7_Atomicll12memory_order12memory_scope(i64 addrspace(3)* noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw max i64 addrspace(3)* %0, i64 %1 monotonic
  ret i64 %res
}

define i64 @_Z30__pocl_atomic_fetch_max__localPU3AS3VU7_Atomicmm12memory_order12memory_scope(i64 addrspace(3)* noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw umax i64 addrspace(3)* %0, i64 %1 monotonic
  ret i64 %res
}


define i32 @_Z32__pocl_atomic_fetch_max__genericPVU7_Atomicii12memory_order12memory_scope(i32* noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw max i32* %0, i32 %1 monotonic
  ret i32 %res
}

define i32 @_Z32__pocl_atomic_fetch_max__genericPVU7_Atomicjj12memory_order12memory_scope(i32* noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw umax i32* %0, i32 %1 monotonic
  ret i32 %res
}

define i64 @_Z32__pocl_atomic_fetch_max__genericPVU7_Atomicll12memory_order12memory_scope(i64* noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw max i64* %0, i64 %1 monotonic
  ret i64 %res
}

define i64 @_Z32__pocl_atomic_fetch_max__genericPVU7_Atomicmm12memory_order12memory_scope(i64* noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw umax i64* %0, i64 %1 monotonic
  ret i64 %res
}

