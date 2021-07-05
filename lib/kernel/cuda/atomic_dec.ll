define i32 @_Z14_cl_atomic_decPU3AS1Vi(i32 addrspace(1)* %ptr) {
entry:
  %0 = atomicrmw sub i32 addrspace(1)* %ptr, i32 1 monotonic
  ret i32 %0
}

define i32 @_Z14_cl_atomic_decPU3AS1Vj(i32 addrspace(1)* %ptr) {
entry:
  %0 = atomicrmw sub i32 addrspace(1)* %ptr, i32 1 monotonic
  ret i32 %0
}

define i32 @_Z14_cl_atomic_decPU3AS3Vi(i32 addrspace(3)* %ptr) {
entry:
  %0 = atomicrmw sub i32 addrspace(3)* %ptr, i32 1 monotonic
  ret i32 %0
}

define i32 @_Z14_cl_atomic_decPU3AS3Vj(i32 addrspace(3)* %ptr) {
entry:
  %0 = atomicrmw sub i32 addrspace(3)* %ptr, i32 1 monotonic
  ret i32 %0
}

define i64 @_Z14_cl_atomic_decPU3AS1Vl(i64 addrspace(1)* %ptr) {
entry:
  %0 = atomicrmw sub i64 addrspace(1)* %ptr, i64 1 monotonic
  ret i64 %0
}

define i64 @_Z14_cl_atomic_decPU3AS1Vm(i64 addrspace(1)* %ptr) {
entry:
  %0 = atomicrmw sub i64 addrspace(1)* %ptr, i64 1 monotonic
  ret i64 %0
}

define i64 @_Z14_cl_atomic_decPU3AS3Vl(i64 addrspace(3)* %ptr) {
entry:
  %0 = atomicrmw sub i64 addrspace(3)* %ptr, i64 1 monotonic
  ret i64 %0
}

define i64 @_Z14_cl_atomic_decPU3AS3Vm(i64 addrspace(3)* %ptr) {
entry:
  %0 = atomicrmw sub i64 addrspace(3)* %ptr, i64 1 monotonic
  ret i64 %0
}
