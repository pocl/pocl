define i32 @_Z30__pocl_atomic_fetch_or__globalPU3AS1VU7_Atomicii12memory_order12memory_scope(ptr addrspace(1) noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw or ptr addrspace(1) %0, i32 %1 monotonic
  ret i32 %res
}

define i32 @_Z30__pocl_atomic_fetch_or__globalPU3AS1VU7_Atomicjj12memory_order12memory_scope(ptr addrspace(1) noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw or ptr addrspace(1) %0, i32 %1 monotonic
  ret i32 %res
}

define i64 @_Z30__pocl_atomic_fetch_or__globalPU3AS1VU7_Atomicll12memory_order12memory_scope(ptr addrspace(1) noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw or ptr addrspace(1) %0, i64 %1 monotonic
  ret i64 %res
}

define i64 @_Z30__pocl_atomic_fetch_or__globalPU3AS1VU7_Atomicmm12memory_order12memory_scope(ptr addrspace(1) noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw or ptr addrspace(1) %0, i64 %1 monotonic
  ret i64 %res
}

define i32 @_Z29__pocl_atomic_fetch_or__localPU3AS3VU7_Atomicii12memory_order12memory_scope(ptr addrspace(3) noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw or ptr addrspace(3) %0, i32 %1 monotonic
  ret i32 %res
}

define i32 @_Z29__pocl_atomic_fetch_or__localPU3AS3VU7_Atomicjj12memory_order12memory_scope(ptr addrspace(3) noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw or ptr addrspace(3) %0, i32 %1 monotonic
  ret i32 %res
}

define i64 @_Z29__pocl_atomic_fetch_or__localPU3AS3VU7_Atomicll12memory_order12memory_scope(ptr addrspace(3) noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw or ptr addrspace(3) %0, i64 %1 monotonic
  ret i64 %res
}

define i64 @_Z29__pocl_atomic_fetch_or__localPU3AS3VU7_Atomicmm12memory_order12memory_scope(ptr addrspace(3) noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw or ptr addrspace(3) %0, i64 %1 monotonic
  ret i64 %res
}


define i32 @_Z31__pocl_atomic_fetch_or__genericPVU7_Atomicii12memory_order12memory_scope(ptr  noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw or ptr  %0, i32 %1 monotonic
  ret i32 %res
}

define i32 @_Z31__pocl_atomic_fetch_or__genericPVU7_Atomicjj12memory_order12memory_scope(ptr  noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw or ptr  %0, i32 %1 monotonic
  ret i32 %res
}

define i64 @_Z31__pocl_atomic_fetch_or__genericPVU7_Atomicll12memory_order12memory_scope(ptr  noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw or ptr  %0, i64 %1 monotonic
  ret i64 %res
}

define i64 @_Z31__pocl_atomic_fetch_or__genericPVU7_Atomicmm12memory_order12memory_scope(ptr  noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw or ptr  %0, i64 %1 monotonic
  ret i64 %res
}

