define i32 @_Z13_cl_atomic_orPU3AS1Vii(ptr addrspace(1) %ptr, i32 %val) {
entry:
  %0 = atomicrmw or ptr addrspace(1) %ptr, i32 %val monotonic
  ret i32 %0
}

define i32 @_Z13_cl_atomic_orPU3AS1Vjj(ptr addrspace(1) %ptr, i32 %val) {
entry:
  %0 = atomicrmw or ptr addrspace(1) %ptr, i32 %val monotonic
  ret i32 %0
}

define i32 @_Z13_cl_atomic_orPU3AS3Vii(ptr addrspace(3) %ptr, i32 %val) {
entry:
  %0 = atomicrmw or ptr addrspace(3) %ptr, i32 %val monotonic
  ret i32 %0
}

define i32 @_Z13_cl_atomic_orPU3AS3Vjj(ptr addrspace(3) %ptr, i32 %val) {
entry:
  %0 = atomicrmw or ptr addrspace(3) %ptr, i32 %val monotonic
  ret i32 %0
}

define i64 @_Z13_cl_atomic_orPU3AS1Vll(ptr addrspace(1) %ptr, i64 %val) {
entry:
  %0 = atomicrmw or ptr addrspace(1) %ptr, i64 %val monotonic
  ret i64 %0
}

define i64 @_Z13_cl_atomic_orPU3AS1Vmm(ptr addrspace(1) %ptr, i64 %val) {
entry:
  %0 = atomicrmw or ptr addrspace(1) %ptr, i64 %val monotonic
  ret i64 %0
}

define i64 @_Z13_cl_atomic_orPU3AS3Vll(ptr addrspace(3) %ptr, i64 %val) {
entry:
  %0 = atomicrmw or ptr addrspace(3) %ptr, i64 %val monotonic
  ret i64 %0
}

define i64 @_Z13_cl_atomic_orPU3AS3Vmm(ptr addrspace(3) %ptr, i64 %val) {
entry:
  %0 = atomicrmw or ptr addrspace(3) %ptr, i64 %val monotonic
  ret i64 %0
}
