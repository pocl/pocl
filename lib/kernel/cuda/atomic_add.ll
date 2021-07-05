define i32 @_Z14_cl_atomic_addPU3AS1Vii(i32 addrspace(1)* %ptr, i32 %val) {
entry:
  %0 = atomicrmw add i32 addrspace(1)* %ptr, i32 %val monotonic
  ret i32 %0
}

define i32 @_Z14_cl_atomic_addPU3AS1Vjj(i32 addrspace(1)* %ptr, i32 %val) {
entry:
  %0 = atomicrmw add i32 addrspace(1)* %ptr, i32 %val monotonic
  ret i32 %0
}

define i32 @_Z14_cl_atomic_addPU3AS3Vii(i32 addrspace(3)* %ptr, i32 %val) {
entry:
  %0 = atomicrmw add i32 addrspace(3)* %ptr, i32 %val monotonic
  ret i32 %0
}

define i32 @_Z14_cl_atomic_addPU3AS3Vjj(i32 addrspace(3)* %ptr, i32 %val) {
entry:
  %0 = atomicrmw add i32 addrspace(3)* %ptr, i32 %val monotonic
  ret i32 %0
}

define i64 @_Z14_cl_atomic_addPU3AS1Vll(i64 addrspace(1)* %ptr, i64 %val) {
entry:
  %0 = atomicrmw add i64 addrspace(1)* %ptr, i64 %val monotonic
  ret i64 %0
}

define i64 @_Z14_cl_atomic_addPU3AS1Vmm(i64 addrspace(1)* %ptr, i64 %val) {
entry:
  %0 = atomicrmw add i64 addrspace(1)* %ptr, i64 %val monotonic
  ret i64 %0
}

define i64 @_Z14_cl_atomic_addPU3AS3Vll(i64 addrspace(3)* %ptr, i64 %val) {
entry:
  %0 = atomicrmw add i64 addrspace(3)* %ptr, i64 %val monotonic
  ret i64 %0
}

define i64 @_Z14_cl_atomic_addPU3AS3Vmm(i64 addrspace(3)* %ptr, i64 %val) {
entry:
  %0 = atomicrmw add i64 addrspace(3)* %ptr, i64 %val monotonic
  ret i64 %0
}
