define i32 @_Z18_cl_atomic_cmpxchgPU3AS1Viii(ptr addrspace(1) %ptr, i32 %cmp, i32 %val) {
entry:
  %0 = cmpxchg ptr addrspace(1) %ptr, i32 %cmp, i32 %val monotonic monotonic
  %1 = extractvalue {i32, i1} %0, 0
  ret i32 %1
}

define i32 @_Z18_cl_atomic_cmpxchgPU3AS1Vjjj(ptr addrspace(1) %ptr, i32 %cmp, i32 %val) {
entry:
  %0 = cmpxchg ptr addrspace(1) %ptr, i32 %cmp, i32 %val monotonic monotonic
  %1 = extractvalue {i32, i1} %0, 0
  ret i32 %1
}

define i32 @_Z18_cl_atomic_cmpxchgPU3AS3Viii(ptr addrspace(3) %ptr, i32 %cmp, i32 %val) {
entry:
  %0 = cmpxchg ptr addrspace(3) %ptr, i32 %cmp, i32 %val monotonic monotonic
  %1 = extractvalue {i32, i1} %0, 0
  ret i32 %1
}

define i32 @_Z18_cl_atomic_cmpxchgPU3AS3Vjjj(ptr addrspace(3) %ptr, i32 %cmp, i32 %val) {
entry:
  %0 = cmpxchg ptr addrspace(3) %ptr, i32 %cmp, i32 %val monotonic monotonic
  %1 = extractvalue {i32, i1} %0, 0
  ret i32 %1
}

define i64 @_Z18_cl_atomic_cmpxchgPU3AS1Vlll(ptr addrspace(1) %ptr, i64 %cmp, i64 %val) {
entry:
  %0 = cmpxchg ptr addrspace(1) %ptr, i64 %cmp, i64 %val monotonic monotonic
  %1 = extractvalue {i64, i1} %0, 0
  ret i64 %1
}

define i64 @_Z18_cl_atomic_cmpxchgPU3AS1Vmmm(ptr addrspace(1) %ptr, i64 %cmp, i64 %val) {
entry:
  %0 = cmpxchg ptr addrspace(1) %ptr, i64 %cmp, i64 %val monotonic monotonic
  %1 = extractvalue {i64, i1} %0, 0
  ret i64 %1
}

define i64 @_Z18_cl_atomic_cmpxchgPU3AS3Vlll(ptr addrspace(3) %ptr, i64 %cmp, i64 %val) {
entry:
  %0 = cmpxchg ptr addrspace(3) %ptr, i64 %cmp, i64 %val monotonic monotonic
  %1 = extractvalue {i64, i1} %0, 0
  ret i64 %1
}

define i64 @_Z18_cl_atomic_cmpxchgPU3AS3Vmmm(ptr addrspace(3) %ptr, i64 %cmp, i64 %val) {
entry:
  %0 = cmpxchg ptr addrspace(3) %ptr, i64 %cmp, i64 %val monotonic monotonic
  %1 = extractvalue {i64, i1} %0, 0
  ret i64 %1
}
