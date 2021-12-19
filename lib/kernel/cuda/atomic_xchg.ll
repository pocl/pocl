define i32 @_Z15_cl_atomic_xchgPU3AS1Vii(i32 addrspace(1)* %ptr, i32 %val) {
entry:
  %0 = atomicrmw xchg i32 addrspace(1)* %ptr, i32 %val monotonic
  ret i32 %0
}

define i32 @_Z15_cl_atomic_xchgPU3AS1Vjj(i32 addrspace(1)* %ptr, i32 %val) {
entry:
  %0 = atomicrmw xchg i32 addrspace(1)* %ptr, i32 %val monotonic
  ret i32 %0
}

define i32 @_Z15_cl_atomic_xchgPU3AS3Vii(i32 addrspace(3)* %ptr, i32 %val) {
entry:
  %0 = atomicrmw xchg i32 addrspace(3)* %ptr, i32 %val monotonic
  ret i32 %0
}

define i32 @_Z15_cl_atomic_xchgPU3AS3Vjj(i32 addrspace(3)* %ptr, i32 %val) {
entry:
  %0 = atomicrmw xchg i32 addrspace(3)* %ptr, i32 %val monotonic
  ret i32 %0
}

define i64 @_Z15_cl_atomic_xchgPU3AS1Vll(i64 addrspace(1)* %ptr, i64 %val) {
entry:
  %0 = atomicrmw xchg i64 addrspace(1)* %ptr, i64 %val monotonic
  ret i64 %0
}

define i64 @_Z15_cl_atomic_xchgPU3AS1Vmm(i64 addrspace(1)* %ptr, i64 %val) {
entry:
  %0 = atomicrmw xchg i64 addrspace(1)* %ptr, i64 %val monotonic
  ret i64 %0
}

define i64 @_Z15_cl_atomic_xchgPU3AS3Vll(i64 addrspace(3)* %ptr, i64 %val) {
entry:
  %0 = atomicrmw xchg i64 addrspace(3)* %ptr, i64 %val monotonic
  ret i64 %0
}

define i64 @_Z15_cl_atomic_xchgPU3AS3Vmm(i64 addrspace(3)* %ptr, i64 %val) {
entry:
  %0 = atomicrmw xchg i64 addrspace(3)* %ptr, i64 %val monotonic
  ret i64 %0
}

define float @_Z15_cl_atomic_xchgPU3AS1Vff(float addrspace(1)* %ptr, float %val) {
entry:
  %iptr = bitcast float addrspace(1)* %ptr to i32 addrspace(1)*
  %ival = bitcast float %val to i32
  %ires = atomicrmw xchg i32 addrspace(1)* %iptr, i32 %ival monotonic
  %fres = bitcast i32 %ires to float
  ret float %fres
}

define float @_Z15_cl_atomic_xchgPU3AS3Vff(float addrspace(3)* %ptr, float %val) {
entry:
  %iptr = bitcast float addrspace(3)* %ptr to i32 addrspace(3)*
  %ival = bitcast float %val to i32
  %ires = atomicrmw xchg i32 addrspace(3)* %iptr, i32 %ival monotonic
  %fres = bitcast i32 %ires to float
  ret float %fres
}
