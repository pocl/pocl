
;     volatile __global A *object,
;    __local C *expected,
;    C desired,
;    memory_order success,
;    memory_order failure,
;    memory_scope scope)

; ######################### LOCAL


define zeroext i1 @_Z44__pocl_atomic_compare_exchange_strong__localPU3AS3VU7_AtomiciPii12memory_orderS4_12memory_scope(i32 addrspace(3)* noundef %0, i32* noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %cmpval = load atomic i32, i32* %1 monotonic , align 4
  %res = cmpxchg i32 addrspace(3)* %0, i32 %cmpval, i32 %2 monotonic monotonic
  %val_loaded = extractvalue {i32, i1} %res, 0
  %success = extractvalue { i32, i1 } %res, 1
  br i1 %success, label %true, label %false

false:
  store atomic i32 %val_loaded, i32* %1 monotonic, align 4
  br label %true

true:
  ret i1 %success
}

define zeroext i1 @_Z44__pocl_atomic_compare_exchange_strong__localPU3AS3VU7_AtomicjPjj12memory_orderS4_12memory_scope(i32 addrspace(3)* noundef %0, i32* noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %res = tail call i1 @_Z44__pocl_atomic_compare_exchange_strong__localPU3AS3VU7_AtomiciPii12memory_orderS4_12memory_scope(i32 addrspace(3)* %0, i32* %1, i32 %2, i32 %3, i32 %4, i32 %5)
  ret i1 %res
}

define zeroext i1 @_Z44__pocl_atomic_compare_exchange_strong__localPU3AS3VU7_AtomicfPff12memory_orderS4_12memory_scope(i32 addrspace(3)* noundef %0, i32* noundef %1, float noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %newval = bitcast float %2 to i32
  %res = tail call i1 @_Z44__pocl_atomic_compare_exchange_strong__localPU3AS3VU7_AtomiciPii12memory_orderS4_12memory_scope(i32 addrspace(3)* %0, i32* %1, i32 %newval, i32 %3, i32 %4, i32 %5)
  ret i1 %res
}





define zeroext i1 @_Z44__pocl_atomic_compare_exchange_strong__localPU3AS3VU7_AtomiclPll12memory_orderS4_12memory_scope(i64 addrspace(3)* noundef %0, i64* noundef %1, i64 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %cmpval = load atomic i64, i64* %1 monotonic , align 8
  %res = cmpxchg i64 addrspace(3)* %0, i64 %cmpval, i64 %2 monotonic monotonic
  %val_loaded = extractvalue {i64, i1} %res, 0
  %success = extractvalue { i64, i1 } %res, 1
  br i1 %success, label %true, label %false

false:
  store atomic i64 %val_loaded, i64* %1 monotonic, align 8
  br label %true

true:
  ret i1 %success
}

define zeroext i1 @_Z44__pocl_atomic_compare_exchange_strong__localPU3AS3VU7_AtomicmPmm12memory_orderS4_12memory_scope(i64 addrspace(3)* noundef %0, i64* noundef %1, i64 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %res = tail call i1 @_Z44__pocl_atomic_compare_exchange_strong__localPU3AS3VU7_AtomiclPll12memory_orderS4_12memory_scope(i64 addrspace(3)* %0, i64* %1, i64 %2, i32 %3, i32 %4, i32 %5)
  ret i1 %res
}

define zeroext i1 @_Z44__pocl_atomic_compare_exchange_strong__localPU3AS3VU7_AtomicdPdd12memory_orderS4_12memory_scope(i64 addrspace(3)* noundef %0, i64* noundef %1, double noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %newval = bitcast double %2 to i64
  %res = tail call i1 @_Z44__pocl_atomic_compare_exchange_strong__localPU3AS3VU7_AtomiclPll12memory_orderS4_12memory_scope(i64 addrspace(3)* %0, i64* %1, i64 %newval, i32 %3, i32 %4, i32 %5)
  ret i1 %res
}


; ######################### GLOBAL

define zeroext i1 @_Z45__pocl_atomic_compare_exchange_strong__globalPU3AS1VU7_AtomiciPii12memory_orderS4_12memory_scope(i32 addrspace(1)* noundef %0, i32* noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %cmpval = load atomic i32, i32* %1 monotonic , align 4
  %res = cmpxchg i32 addrspace(1)* %0, i32 %cmpval, i32 %2 monotonic monotonic
  %val_loaded = extractvalue {i32, i1} %res, 0
  %success = extractvalue { i32, i1 } %res, 1
  br i1 %success, label %true, label %false

false:
  store atomic i32 %val_loaded, i32* %1 monotonic, align 4
  br label %true

true:
  ret i1 %success
}

define zeroext i1 @_Z45__pocl_atomic_compare_exchange_strong__globalPU3AS1VU7_AtomicjPjj12memory_orderS4_12memory_scope(i32 addrspace(1)* noundef %0, i32* noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %res = tail call i1 @_Z45__pocl_atomic_compare_exchange_strong__globalPU3AS1VU7_AtomiciPii12memory_orderS4_12memory_scope(i32 addrspace(1)* %0, i32* %1, i32 %2, i32 %3, i32 %4, i32 %5)
  ret i1 %res
}

define zeroext i1 @_Z45__pocl_atomic_compare_exchange_strong__globalPU3AS1VU7_AtomicfPff12memory_orderS4_12memory_scope(i32 addrspace(1)* noundef %0, i32* noundef %1, float noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %newval = bitcast float %2 to i32
  %res = tail call i1 @_Z45__pocl_atomic_compare_exchange_strong__globalPU3AS1VU7_AtomiciPii12memory_orderS4_12memory_scope(i32 addrspace(1)* %0, i32* %1, i32 %newval, i32 %3, i32 %4, i32 %5)
  ret i1 %res
}





define zeroext i1 @_Z45__pocl_atomic_compare_exchange_strong__globalPU3AS1VU7_AtomiclPll12memory_orderS4_12memory_scope(i64 addrspace(1)* noundef %0, i64* noundef %1, i64 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %cmpval = load atomic i64, i64* %1 monotonic , align 8
  %res = cmpxchg i64 addrspace(1)* %0, i64 %cmpval, i64 %2 monotonic monotonic
  %val_loaded = extractvalue {i64, i1} %res, 0
  %success = extractvalue { i64, i1 } %res, 1
  br i1 %success, label %true, label %false

false:
  store atomic i64 %val_loaded, i64* %1 monotonic, align 8
  br label %true

true:
  ret i1 %success
}

define zeroext i1 @_Z45__pocl_atomic_compare_exchange_strong__globalPU3AS1VU7_AtomicmPmm12memory_orderS4_12memory_scope(i64 addrspace(1)* noundef %0, i64* noundef %1, i64 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %res = tail call i1 @_Z45__pocl_atomic_compare_exchange_strong__globalPU3AS1VU7_AtomiclPll12memory_orderS4_12memory_scope(i64 addrspace(1)* %0, i64* %1, i64 %2, i32 %3, i32 %4, i32 %5)
  ret i1 %res
}

define zeroext i1 @_Z45__pocl_atomic_compare_exchange_strong__globalPU3AS1VU7_AtomicdPdd12memory_orderS4_12memory_scope(i64 addrspace(1)* noundef %0, i64* noundef %1, double noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %newval = bitcast double %2 to i64
  %res = tail call i1 @_Z45__pocl_atomic_compare_exchange_strong__globalPU3AS1VU7_AtomiclPll12memory_orderS4_12memory_scope(i64 addrspace(1)* %0, i64* %1, i64 %newval, i32 %3, i32 %4, i32 %5)
  ret i1 %res
}


; ######################### GENERIC


define zeroext i1 @_Z46__pocl_atomic_compare_exchange_strong__genericPVU7_AtomiciPii12memory_orderS4_12memory_scope(i32* noundef %0, i32* noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %cmpval = load atomic i32, i32* %1 monotonic , align 4
  %res = cmpxchg i32* %0, i32 %cmpval, i32 %2 monotonic monotonic
  %val_loaded = extractvalue {i32, i1} %res, 0
  %success = extractvalue { i32, i1 } %res, 1
  br i1 %success, label %true, label %false

false:
  store atomic i32 %val_loaded, i32* %1 monotonic, align 4
  br label %true

true:
  ret i1 %success
}

define zeroext i1 @_Z46__pocl_atomic_compare_exchange_strong__genericPVU7_AtomicjPjj12memory_orderS4_12memory_scope(i32* noundef %0, i32* noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %res = tail call i1 @_Z46__pocl_atomic_compare_exchange_strong__genericPVU7_AtomiciPii12memory_orderS4_12memory_scope(i32* %0, i32* %1, i32 %2, i32 %3, i32 %4, i32 %5)
  ret i1 %res
}

define zeroext i1 @_Z46__pocl_atomic_compare_exchange_strong__genericPVU7_AtomicfPff12memory_orderS4_12memory_scope(i32* noundef %0, i32* noundef %1, float noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %newval = bitcast float %2 to i32
  %res = tail call i1 @_Z46__pocl_atomic_compare_exchange_strong__genericPVU7_AtomiciPii12memory_orderS4_12memory_scope(i32* %0, i32* %1, i32 %newval, i32 %3, i32 %4, i32 %5)
  ret i1 %res
}







define zeroext i1 @_Z46__pocl_atomic_compare_exchange_strong__genericPVU7_AtomiclPll12memory_orderS4_12memory_scope(i64* noundef %0, i64* noundef %1, i64 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %cmpval = load atomic i64, i64* %1 monotonic , align 8
  %res = cmpxchg i64* %0, i64 %cmpval, i64 %2 monotonic monotonic
  %val_loaded = extractvalue {i64, i1} %res, 0
  %success = extractvalue { i64, i1 } %res, 1
  br i1 %success, label %true, label %false

false:
  store atomic i64 %val_loaded, i64* %1 monotonic, align 8
  br label %true

true:
  ret i1 %success
}

define zeroext i1 @_Z46__pocl_atomic_compare_exchange_strong__genericPVU7_AtomicmPmm12memory_orderS4_12memory_scope(i64* noundef %0, i64* noundef %1, i64 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %res = tail call i1 @_Z46__pocl_atomic_compare_exchange_strong__genericPVU7_AtomiclPll12memory_orderS4_12memory_scope(i64* %0, i64* %1, i64 %2, i32 %3, i32 %4, i32 %5)
  ret i1 %res
}

define zeroext i1 @_Z46__pocl_atomic_compare_exchange_strong__genericPVU7_AtomicdPdd12memory_orderS4_12memory_scope(i64* noundef %0, i64* noundef %1, double noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
entry:
  %newval = bitcast double %2 to i64
  %res = tail call i1 @_Z46__pocl_atomic_compare_exchange_strong__genericPVU7_AtomiclPll12memory_orderS4_12memory_scope(i64* %0, i64* %1, i64 %newval, i32 %3, i32 %4, i32 %5)
  ret i1 %res
}

