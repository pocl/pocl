define i32 @_Z15_cl_atomic_xchgPU3AS1Vii(ptr addrspace(1) %ptr, i32 %val) {
entry:
  %0 = atomicrmw xchg ptr addrspace(1) %ptr, i32 %val monotonic
  ret i32 %0
}

define i32 @_Z15_cl_atomic_xchgPU3AS1Vjj(ptr addrspace(1) %ptr, i32 %val) {
entry:
  %0 = atomicrmw xchg ptr addrspace(1) %ptr, i32 %val monotonic
  ret i32 %0
}

define i32 @_Z15_cl_atomic_xchgPU3AS3Vii(ptr addrspace(3) %ptr, i32 %val) {
entry:
  %0 = atomicrmw xchg ptr addrspace(3) %ptr, i32 %val monotonic
  ret i32 %0
}

define i32 @_Z15_cl_atomic_xchgPU3AS3Vjj(ptr addrspace(3) %ptr, i32 %val) {
entry:
  %0 = atomicrmw xchg ptr addrspace(3) %ptr, i32 %val monotonic
  ret i32 %0
}

define i64 @_Z15_cl_atomic_xchgPU3AS1Vll(ptr addrspace(1) %ptr, i64 %val) {
entry:
  %0 = atomicrmw xchg ptr addrspace(1) %ptr, i64 %val monotonic
  ret i64 %0
}

define i64 @_Z15_cl_atomic_xchgPU3AS1Vmm(ptr addrspace(1) %ptr, i64 %val) {
entry:
  %0 = atomicrmw xchg ptr addrspace(1) %ptr, i64 %val monotonic
  ret i64 %0
}

define i64 @_Z15_cl_atomic_xchgPU3AS3Vll(ptr addrspace(3) %ptr, i64 %val) {
entry:
  %0 = atomicrmw xchg ptr addrspace(3) %ptr, i64 %val monotonic
  ret i64 %0
}

define i64 @_Z15_cl_atomic_xchgPU3AS3Vmm(ptr addrspace(3) %ptr, i64 %val) {
entry:
  %0 = atomicrmw xchg ptr addrspace(3) %ptr, i64 %val monotonic
  ret i64 %0
}

define float @_Z15_cl_atomic_xchgPU3AS1Vff(ptr addrspace(1) %ptr, float %val) {
entry:
  %iptr = bitcast ptr addrspace(1) %ptr to ptr addrspace(1)
  %ival = bitcast float %val to i32
  %ires = atomicrmw xchg ptr addrspace(1) %iptr, i32 %ival monotonic
  %fres = bitcast i32 %ires to float
  ret float %fres
}

define float @_Z15_cl_atomic_xchgPU3AS3Vff(ptr addrspace(3) %ptr, float %val) {
entry:
  %iptr = bitcast ptr addrspace(3) %ptr to ptr addrspace(3)
  %ival = bitcast float %val to i32
  %ires = atomicrmw xchg ptr addrspace(3) %iptr, i32 %ival monotonic
  %fres = bitcast i32 %ires to float
  ret float %fres
}
