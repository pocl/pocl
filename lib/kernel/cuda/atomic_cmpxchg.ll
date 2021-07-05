define i32 @_Z18_cl_atomic_cmpxchgPU3AS1Viii(i32 addrspace(1)* %ptr, i32 %cmp, i32 %val) {
entry:
  %0 = cmpxchg i32 addrspace(1)* %ptr, i32 %cmp, i32 %val monotonic monotonic
  %1 = extractvalue {i32, i1} %0, 0
  ret i32 %1
}

define i32 @_Z18_cl_atomic_cmpxchgPU3AS1Vjjj(i32 addrspace(1)* %ptr, i32 %cmp, i32 %val) {
entry:
  %0 = cmpxchg i32 addrspace(1)* %ptr, i32 %cmp, i32 %val monotonic monotonic
  %1 = extractvalue {i32, i1} %0, 0
  ret i32 %1
}

define i32 @_Z18_cl_atomic_cmpxchgPU3AS3Viii(i32 addrspace(3)* %ptr, i32 %cmp, i32 %val) {
entry:
  %0 = cmpxchg i32 addrspace(3)* %ptr, i32 %cmp, i32 %val monotonic monotonic
  %1 = extractvalue {i32, i1} %0, 0
  ret i32 %1
}

define i32 @_Z18_cl_atomic_cmpxchgPU3AS3Vjjj(i32 addrspace(3)* %ptr, i32 %cmp, i32 %val) {
entry:
  %0 = cmpxchg i32 addrspace(3)* %ptr, i32 %cmp, i32 %val monotonic monotonic
  %1 = extractvalue {i32, i1} %0, 0
  ret i32 %1
}

define i64 @_Z18_cl_atomic_cmpxchgPU3AS1Vlll(i64 addrspace(1)* %ptr, i64 %cmp, i64 %val) {
entry:
  %0 = cmpxchg i64 addrspace(1)* %ptr, i64 %cmp, i64 %val monotonic monotonic
  %1 = extractvalue {i64, i1} %0, 0
  ret i64 %1
}

define i64 @_Z18_cl_atomic_cmpxchgPU3AS1Vmmm(i64 addrspace(1)* %ptr, i64 %cmp, i64 %val) {
entry:
  %0 = cmpxchg i64 addrspace(1)* %ptr, i64 %cmp, i64 %val monotonic monotonic
  %1 = extractvalue {i64, i1} %0, 0
  ret i64 %1
}

define i64 @_Z18_cl_atomic_cmpxchgPU3AS3Vlll(i64 addrspace(3)* %ptr, i64 %cmp, i64 %val) {
entry:
  %0 = cmpxchg i64 addrspace(3)* %ptr, i64 %cmp, i64 %val monotonic monotonic
  %1 = extractvalue {i64, i1} %0, 0
  ret i64 %1
}

define i64 @_Z18_cl_atomic_cmpxchgPU3AS3Vmmm(i64 addrspace(3)* %ptr, i64 %cmp, i64 %val) {
entry:
  %0 = cmpxchg i64 addrspace(3)* %ptr, i64 %cmp, i64 %val monotonic monotonic
  %1 = extractvalue {i64, i1} %0, 0
  ret i64 %1
}
