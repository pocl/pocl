define i32 @_Z30__pocl_atomic_exchange__globalPU3AS1VU7_Atomicii12memory_order12memory_scope(ptr addrspace(1) noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw xchg ptr addrspace(1) %0, i32 %1 monotonic
  ret i32 %res
}

define i32 @_Z30__pocl_atomic_exchange__globalPU3AS1VU7_Atomicjj12memory_order12memory_scope(ptr addrspace(1) noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw xchg ptr addrspace(1) %0, i32 %1 monotonic
  ret i32 %res
}

define i64 @_Z30__pocl_atomic_exchange__globalPU3AS1VU7_Atomicll12memory_order12memory_scope(ptr addrspace(1) noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw xchg ptr addrspace(1) %0, i64 %1 monotonic
  ret i64 %res
}

define i64 @_Z30__pocl_atomic_exchange__globalPU3AS1VU7_Atomicmm12memory_order12memory_scope(ptr addrspace(1) noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw xchg ptr addrspace(1) %0, i64 %1 monotonic
  ret i64 %res
}

define i32 @_Z29__pocl_atomic_exchange__localPU3AS3VU7_Atomicii12memory_order12memory_scope(ptr addrspace(3) noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw xchg ptr addrspace(3) %0, i32 %1 monotonic
  ret i32 %res
}

define i32 @_Z29__pocl_atomic_exchange__localPU3AS3VU7_Atomicjj12memory_order12memory_scope(ptr addrspace(3) noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw xchg ptr addrspace(3) %0, i32 %1 monotonic
  ret i32 %res
}

define i64 @_Z29__pocl_atomic_exchange__localPU3AS3VU7_Atomicll12memory_order12memory_scope(ptr addrspace(3) noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw xchg ptr addrspace(3) %0, i64 %1 monotonic
  ret i64 %res
}

define i64 @_Z29__pocl_atomic_exchange__localPU3AS3VU7_Atomicmm12memory_order12memory_scope(ptr addrspace(3) noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw xchg ptr addrspace(3) %0, i64 %1 monotonic
  ret i64 %res
}


define i32 @_Z31__pocl_atomic_exchange__genericPVU7_Atomicii12memory_order12memory_scope(ptr  noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw xchg ptr  %0, i32 %1 monotonic
  ret i32 %res
}

define i32 @_Z31__pocl_atomic_exchange__genericPVU7_Atomicjj12memory_order12memory_scope(ptr  noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw xchg ptr  %0, i32 %1 monotonic
  ret i32 %res
}

define i64 @_Z31__pocl_atomic_exchange__genericPVU7_Atomicll12memory_order12memory_scope(ptr  noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw xchg ptr  %0, i64 %1 monotonic
  ret i64 %res
}

define i64 @_Z31__pocl_atomic_exchange__genericPVU7_Atomicmm12memory_order12memory_scope(ptr  noundef %0, i64 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %res = atomicrmw xchg ptr  %0, i64 %1 monotonic
  ret i64 %res
}


; #####################################################################

define double @_Z29__pocl_atomic_exchange__localPU3AS3VU7_Atomicdd12memory_order12memory_scope(ptr addrspace(3) noundef %0, double noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %iptr = bitcast ptr addrspace(3) %0 to ptr addrspace(3)
  %ival = bitcast double %1 to i64
  %ires = atomicrmw xchg ptr addrspace(3) %iptr, i64 %ival monotonic
  %fres = bitcast i64 %ires to double
  ret double %fres
}

define double @_Z30__pocl_atomic_exchange__globalPU3AS1VU7_Atomicdd12memory_order12memory_scope(ptr addrspace(1) noundef %0, double noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %iptr = bitcast ptr addrspace(1) %0 to ptr addrspace(1)
  %ival = bitcast double %1 to i64
  %ires = atomicrmw xchg ptr addrspace(1) %iptr, i64 %ival monotonic
  %fres = bitcast i64 %ires to double
  ret double %fres
}

define double @_Z31__pocl_atomic_exchange__genericPVU7_Atomicdd12memory_order12memory_scope(ptr  noundef %0, double noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %iptr = bitcast ptr  %0 to ptr 
  %ival = bitcast double %1 to i64
  %ires = atomicrmw xchg ptr  %iptr, i64 %ival monotonic
  %fres = bitcast i64 %ires to double
  ret double %fres
}




define float @_Z29__pocl_atomic_exchange__localPU3AS3VU7_Atomicff12memory_order12memory_scope(ptr addrspace(3) noundef %0, float noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %iptr = bitcast ptr addrspace(3) %0 to ptr addrspace(3)
  %ival = bitcast float %1 to i32
  %ires = atomicrmw xchg ptr addrspace(3) %iptr, i32 %ival monotonic
  %fres = bitcast i32 %ires to float
  ret float %fres
}

define float @_Z30__pocl_atomic_exchange__globalPU3AS1VU7_Atomicff12memory_order12memory_scope(ptr addrspace(1) noundef %0, float noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %iptr = bitcast ptr addrspace(1) %0 to ptr addrspace(1)
  %ival = bitcast float %1 to i32
  %ires = atomicrmw xchg ptr addrspace(1) %iptr, i32 %ival monotonic
  %fres = bitcast i32 %ires to float
  ret float %fres
}

define float @_Z31__pocl_atomic_exchange__genericPVU7_Atomicff12memory_order12memory_scope(ptr  noundef %0, float noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
entry:
  %iptr = bitcast ptr  %0 to ptr 
  %ival = bitcast float %1 to i32
  %ires = atomicrmw xchg ptr  %iptr, i32 %ival monotonic
  %fres = bitcast i32 %ires to float
  ret float %fres
}
