define i32 @_Z14_cl_atomic_addPU3AS1Vii(ptr addrspace(1) %ptr, i32 %val) {
entry:
  %0 = atomicrmw add ptr addrspace(1) %ptr, i32 %val monotonic
  ret i32 %0
}

define i32 @_Z14_cl_atomic_addPU3AS1Vjj(ptr addrspace(1) %ptr, i32 %val) {
entry:
  %0 = atomicrmw add ptr addrspace(1) %ptr, i32 %val monotonic
  ret i32 %0
}

define i32 @_Z14_cl_atomic_addPU3AS3Vii(ptr addrspace(3) %ptr, i32 %val) {
entry:
  %0 = atomicrmw add ptr addrspace(3) %ptr, i32 %val monotonic
  ret i32 %0
}

define i32 @_Z14_cl_atomic_addPU3AS3Vjj(ptr addrspace(3) %ptr, i32 %val) {
entry:
  %0 = atomicrmw add ptr addrspace(3) %ptr, i32 %val monotonic
  ret i32 %0
}

define i64 @_Z14_cl_atomic_addPU3AS1Vll(ptr addrspace(1) %ptr, i64 %val) {
entry:
  %0 = atomicrmw add ptr addrspace(1) %ptr, i64 %val monotonic
  ret i64 %0
}

define i64 @_Z14_cl_atomic_addPU3AS1Vmm(ptr addrspace(1) %ptr, i64 %val) {
entry:
  %0 = atomicrmw add ptr addrspace(1) %ptr, i64 %val monotonic
  ret i64 %0
}

define i64 @_Z14_cl_atomic_addPU3AS3Vll(ptr addrspace(3) %ptr, i64 %val) {
entry:
  %0 = atomicrmw add ptr addrspace(3) %ptr, i64 %val monotonic
  ret i64 %0
}

define i64 @_Z14_cl_atomic_addPU3AS3Vmm(ptr addrspace(3) %ptr, i64 %val) {
entry:
  %0 = atomicrmw add ptr addrspace(3) %ptr, i64 %val monotonic
  ret i64 %0
}
