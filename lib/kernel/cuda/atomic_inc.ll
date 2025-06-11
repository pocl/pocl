define i32 @_Z14_cl_atomic_incPU3AS1Vi(ptr addrspace(1) %ptr) {
entry:
  %0 = atomicrmw add ptr addrspace(1) %ptr, i32 1 monotonic
  ret i32 %0
}

define i32 @_Z14_cl_atomic_incPU3AS1Vj(ptr addrspace(1) %ptr) {
entry:
  %0 = atomicrmw add ptr addrspace(1) %ptr, i32 1 monotonic
  ret i32 %0
}

define i32 @_Z14_cl_atomic_incPU3AS3Vi(ptr addrspace(3) %ptr) {
entry:
  %0 = atomicrmw add ptr addrspace(3) %ptr, i32 1 monotonic
  ret i32 %0
}

define i32 @_Z14_cl_atomic_incPU3AS3Vj(ptr addrspace(3) %ptr) {
entry:
  %0 = atomicrmw add ptr addrspace(3) %ptr, i32 1 monotonic
  ret i32 %0
}

define i64 @_Z14_cl_atomic_incPU3AS1Vl(ptr addrspace(1) %ptr) {
entry:
  %0 = atomicrmw add ptr addrspace(1) %ptr, i64 1 monotonic
  ret i64 %0
}

define i64 @_Z14_cl_atomic_incPU3AS1Vm(ptr addrspace(1) %ptr) {
entry:
  %0 = atomicrmw add ptr addrspace(1) %ptr, i64 1 monotonic
  ret i64 %0
}

define i64 @_Z14_cl_atomic_incPU3AS3Vl(ptr addrspace(3) %ptr) {
entry:
  %0 = atomicrmw add ptr addrspace(3) %ptr, i64 1 monotonic
  ret i64 %0
}

define i64 @_Z14_cl_atomic_incPU3AS3Vm(ptr addrspace(3) %ptr) {
entry:
  %0 = atomicrmw add ptr addrspace(3) %ptr, i64 1 monotonic
  ret i64 %0
}
