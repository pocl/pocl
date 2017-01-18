; ModuleID = 'svm_atomics_hsail.cl.bc'
target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:64:64-p8:32:32-p9:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "hsail64"

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z37pocl_atomic_flag_test_and_set__globalPVU3AS1U7_Atomici12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xchg i32 addrspace(1)* %object, i32 1 monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xchg i32 addrspace(1)* %object, i32 1 acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xchg i32 addrspace(1)* %object, i32 1 release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xchg i32 addrspace(1)* %object, i32 1 acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xchg i32 addrspace(1)* %object, i32 1 seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  %9 = icmp ne i32 %.0, 0
  ret i1 %9
}

; Function Attrs: nounwind uwtable
define void @_Z30pocl_atomic_flag_clear__globalPVU3AS1U7_Atomici12memory_order12memory_scope(i32 addrspace(1)* nocapture %object, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %1 [
    i32 0, label %.thread
    i32 1, label %.thread
    i32 2, label %.thread1
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  store atomic volatile i32 0, i32 addrspace(1)* %object monotonic, align 4
  br label %2

.thread1:                                         ; preds = %0
  store atomic volatile i32 0, i32 addrspace(1)* %object release, align 4
  br label %2

; <label>:1                                       ; preds = %0
  store atomic volatile i32 0, i32 addrspace(1)* %object seq_cst, align 4
  br label %2

; <label>:2                                       ; preds = %1, %.thread1, %.thread
  ret void
}

; Function Attrs: nounwind uwtable
define void @_Z25pocl_atomic_store__globalPVU3AS1U7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* nocapture %object, i32 %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %1 [
    i32 0, label %.thread
    i32 1, label %.thread
    i32 2, label %.thread1
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  store atomic volatile i32 %desired, i32 addrspace(1)* %object monotonic, align 4
  br label %2

.thread1:                                         ; preds = %0
  store atomic volatile i32 %desired, i32 addrspace(1)* %object release, align 4
  br label %2

; <label>:1                                       ; preds = %0
  store atomic volatile i32 %desired, i32 addrspace(1)* %object seq_cst, align 4
  br label %2

; <label>:2                                       ; preds = %1, %.thread1, %.thread
  ret void
}

; Function Attrs: nounwind uwtable
define i32 @_Z24pocl_atomic_load__globalPVU3AS1U7_Atomici12memory_order12memory_scope(i32 addrspace(1)* nocapture readonly %object, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %3 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  %1 = load atomic volatile i32, i32 addrspace(1)* %object monotonic, align 4
  br label %5

.thread1:                                         ; preds = %0
  %2 = load atomic volatile i32, i32 addrspace(1)* %object acquire, align 4
  br label %5

; <label>:3                                       ; preds = %0
  %4 = load atomic volatile i32, i32 addrspace(1)* %object seq_cst, align 4
  br label %5

; <label>:5                                       ; preds = %3, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %4, %3 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z28pocl_atomic_exchange__globalPVU3AS1U7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xchg i32 addrspace(1)* %object, i32 %desired monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xchg i32 addrspace(1)* %object, i32 %desired acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xchg i32 addrspace(1)* %object, i32 %desired release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xchg i32 addrspace(1)* %object, i32 %desired acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xchg i32 addrspace(1)* %object, i32 %desired seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z43pocl_atomic_compare_exchange_strong__globalPVU3AS1U7_AtomiciPii12memory_orderS3_12memory_scope(i32 addrspace(1)* %object, i32* nocapture %expected, i32 %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = icmp eq i32 %failure, 0
  br i1 %11, label %19, label %12

; <label>:12                                      ; preds = %9
  %13 = icmp eq i32 %failure, 1
  br i1 %13, label %19, label %14

; <label>:14                                      ; preds = %12
  %15 = icmp eq i32 %failure, 2
  br i1 %15, label %19, label %16

; <label>:16                                      ; preds = %14
  %17 = icmp eq i32 %failure, 3
  %18 = select i1 %17, i32 4, i32 5
  br label %19

; <label>:19                                      ; preds = %12, %14, %16, %9
  %20 = phi i32 [ 0, %9 ], [ 2, %12 ], [ %18, %16 ], [ 3, %14 ]
  switch i32 %10, label %28 [
    i32 1, label %21
    i32 2, label %21
    i32 3, label %50
    i32 4, label %23
    i32 5, label %25
  ]

; <label>:21                                      ; preds = %19, %19
  %.off = add nsw i32 %20, -1
  %switch = icmp ult i32 %.off, 2
  %22 = load i32, i32* %expected, align 4
  br i1 %switch, label %39, label %36

; <label>:23                                      ; preds = %19
  %.off1 = add nsw i32 %20, -1
  %switch2 = icmp ult i32 %.off1, 2
  %24 = load i32, i32* %expected, align 4
  br i1 %switch2, label %61, label %58

; <label>:25                                      ; preds = %19
  switch i32 %20, label %72 [
    i32 1, label %76
    i32 2, label %76
    i32 5, label %80
  ]

; <label>:26                                      ; preds = %86, %90, %94, %66, %70, %44, %48, %56, %34
  %.0 = phi i8 [ %35, %34 ], [ %87, %86 ], [ %95, %94 ], [ %91, %90 ], [ %71, %70 ], [ %67, %66 ], [ %57, %56 ], [ %49, %48 ], [ %45, %44 ]
  %27 = icmp ne i8 %.0, 0
  ret i1 %27

; <label>:28                                      ; preds = %19
  %29 = load i32, i32* %expected, align 4
  %30 = cmpxchg volatile i32 addrspace(1)* %object, i32 %29, i32 %desired monotonic monotonic
  %31 = extractvalue { i32, i1 } %30, 1
  br i1 %31, label %34, label %32

; <label>:32                                      ; preds = %28
  %33 = extractvalue { i32, i1 } %30, 0
  store i32 %33, i32* %expected, align 4
  br label %34

; <label>:34                                      ; preds = %32, %28
  %35 = zext i1 %31 to i8
  br label %26

; <label>:36                                      ; preds = %21
  %37 = cmpxchg volatile i32 addrspace(1)* %object, i32 %22, i32 %desired acquire monotonic
  %38 = extractvalue { i32, i1 } %37, 1
  br i1 %38, label %44, label %42

; <label>:39                                      ; preds = %21
  %40 = cmpxchg volatile i32 addrspace(1)* %object, i32 %22, i32 %desired acquire acquire
  %41 = extractvalue { i32, i1 } %40, 1
  br i1 %41, label %48, label %46

; <label>:42                                      ; preds = %36
  %43 = extractvalue { i32, i1 } %37, 0
  store i32 %43, i32* %expected, align 4
  br label %44

; <label>:44                                      ; preds = %42, %36
  %45 = zext i1 %38 to i8
  br label %26

; <label>:46                                      ; preds = %39
  %47 = extractvalue { i32, i1 } %40, 0
  store i32 %47, i32* %expected, align 4
  br label %48

; <label>:48                                      ; preds = %46, %39
  %49 = zext i1 %41 to i8
  br label %26

; <label>:50                                      ; preds = %19
  %51 = load i32, i32* %expected, align 4
  %52 = cmpxchg volatile i32 addrspace(1)* %object, i32 %51, i32 %desired release monotonic
  %53 = extractvalue { i32, i1 } %52, 1
  br i1 %53, label %56, label %54

; <label>:54                                      ; preds = %50
  %55 = extractvalue { i32, i1 } %52, 0
  store i32 %55, i32* %expected, align 4
  br label %56

; <label>:56                                      ; preds = %54, %50
  %57 = zext i1 %53 to i8
  br label %26

; <label>:58                                      ; preds = %23
  %59 = cmpxchg volatile i32 addrspace(1)* %object, i32 %24, i32 %desired acq_rel monotonic
  %60 = extractvalue { i32, i1 } %59, 1
  br i1 %60, label %66, label %64

; <label>:61                                      ; preds = %23
  %62 = cmpxchg volatile i32 addrspace(1)* %object, i32 %24, i32 %desired acq_rel acquire
  %63 = extractvalue { i32, i1 } %62, 1
  br i1 %63, label %70, label %68

; <label>:64                                      ; preds = %58
  %65 = extractvalue { i32, i1 } %59, 0
  store i32 %65, i32* %expected, align 4
  br label %66

; <label>:66                                      ; preds = %64, %58
  %67 = zext i1 %60 to i8
  br label %26

; <label>:68                                      ; preds = %61
  %69 = extractvalue { i32, i1 } %62, 0
  store i32 %69, i32* %expected, align 4
  br label %70

; <label>:70                                      ; preds = %68, %61
  %71 = zext i1 %63 to i8
  br label %26

; <label>:72                                      ; preds = %25
  %73 = load i32, i32* %expected, align 4
  %74 = cmpxchg volatile i32 addrspace(1)* %object, i32 %73, i32 %desired seq_cst monotonic
  %75 = extractvalue { i32, i1 } %74, 1
  br i1 %75, label %86, label %84

; <label>:76                                      ; preds = %25, %25
  %77 = load i32, i32* %expected, align 4
  %78 = cmpxchg volatile i32 addrspace(1)* %object, i32 %77, i32 %desired seq_cst acquire
  %79 = extractvalue { i32, i1 } %78, 1
  br i1 %79, label %90, label %88

; <label>:80                                      ; preds = %25
  %81 = load i32, i32* %expected, align 4
  %82 = cmpxchg volatile i32 addrspace(1)* %object, i32 %81, i32 %desired seq_cst seq_cst
  %83 = extractvalue { i32, i1 } %82, 1
  br i1 %83, label %94, label %92

; <label>:84                                      ; preds = %72
  %85 = extractvalue { i32, i1 } %74, 0
  store i32 %85, i32* %expected, align 4
  br label %86

; <label>:86                                      ; preds = %84, %72
  %87 = zext i1 %75 to i8
  br label %26

; <label>:88                                      ; preds = %76
  %89 = extractvalue { i32, i1 } %78, 0
  store i32 %89, i32* %expected, align 4
  br label %90

; <label>:90                                      ; preds = %88, %76
  %91 = zext i1 %79 to i8
  br label %26

; <label>:92                                      ; preds = %80
  %93 = extractvalue { i32, i1 } %82, 0
  store i32 %93, i32* %expected, align 4
  br label %94

; <label>:94                                      ; preds = %92, %80
  %95 = zext i1 %83 to i8
  br label %26
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z41pocl_atomic_compare_exchange_weak__globalPVU3AS1U7_AtomiciPii12memory_orderS3_12memory_scope(i32 addrspace(1)* %object, i32* nocapture %expected, i32 %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = icmp eq i32 %failure, 0
  br i1 %11, label %19, label %12

; <label>:12                                      ; preds = %9
  %13 = icmp eq i32 %failure, 1
  br i1 %13, label %19, label %14

; <label>:14                                      ; preds = %12
  %15 = icmp eq i32 %failure, 2
  br i1 %15, label %19, label %16

; <label>:16                                      ; preds = %14
  %17 = icmp eq i32 %failure, 3
  %18 = select i1 %17, i32 4, i32 5
  br label %19

; <label>:19                                      ; preds = %12, %14, %16, %9
  %20 = phi i32 [ 0, %9 ], [ 2, %12 ], [ %18, %16 ], [ 3, %14 ]
  switch i32 %10, label %28 [
    i32 1, label %21
    i32 2, label %21
    i32 3, label %50
    i32 4, label %23
    i32 5, label %25
  ]

; <label>:21                                      ; preds = %19, %19
  %.off = add nsw i32 %20, -1
  %switch = icmp ult i32 %.off, 2
  %22 = load i32, i32* %expected, align 4
  br i1 %switch, label %39, label %36

; <label>:23                                      ; preds = %19
  %.off1 = add nsw i32 %20, -1
  %switch2 = icmp ult i32 %.off1, 2
  %24 = load i32, i32* %expected, align 4
  br i1 %switch2, label %61, label %58

; <label>:25                                      ; preds = %19
  switch i32 %20, label %72 [
    i32 1, label %76
    i32 2, label %76
    i32 5, label %80
  ]

; <label>:26                                      ; preds = %86, %90, %94, %66, %70, %44, %48, %56, %34
  %.0 = phi i8 [ %35, %34 ], [ %87, %86 ], [ %95, %94 ], [ %91, %90 ], [ %71, %70 ], [ %67, %66 ], [ %57, %56 ], [ %49, %48 ], [ %45, %44 ]
  %27 = icmp ne i8 %.0, 0
  ret i1 %27

; <label>:28                                      ; preds = %19
  %29 = load i32, i32* %expected, align 4
  %30 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %29, i32 %desired monotonic monotonic
  %31 = extractvalue { i32, i1 } %30, 1
  br i1 %31, label %34, label %32

; <label>:32                                      ; preds = %28
  %33 = extractvalue { i32, i1 } %30, 0
  store i32 %33, i32* %expected, align 4
  br label %34

; <label>:34                                      ; preds = %32, %28
  %35 = zext i1 %31 to i8
  br label %26

; <label>:36                                      ; preds = %21
  %37 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %22, i32 %desired acquire monotonic
  %38 = extractvalue { i32, i1 } %37, 1
  br i1 %38, label %44, label %42

; <label>:39                                      ; preds = %21
  %40 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %22, i32 %desired acquire acquire
  %41 = extractvalue { i32, i1 } %40, 1
  br i1 %41, label %48, label %46

; <label>:42                                      ; preds = %36
  %43 = extractvalue { i32, i1 } %37, 0
  store i32 %43, i32* %expected, align 4
  br label %44

; <label>:44                                      ; preds = %42, %36
  %45 = zext i1 %38 to i8
  br label %26

; <label>:46                                      ; preds = %39
  %47 = extractvalue { i32, i1 } %40, 0
  store i32 %47, i32* %expected, align 4
  br label %48

; <label>:48                                      ; preds = %46, %39
  %49 = zext i1 %41 to i8
  br label %26

; <label>:50                                      ; preds = %19
  %51 = load i32, i32* %expected, align 4
  %52 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %51, i32 %desired release monotonic
  %53 = extractvalue { i32, i1 } %52, 1
  br i1 %53, label %56, label %54

; <label>:54                                      ; preds = %50
  %55 = extractvalue { i32, i1 } %52, 0
  store i32 %55, i32* %expected, align 4
  br label %56

; <label>:56                                      ; preds = %54, %50
  %57 = zext i1 %53 to i8
  br label %26

; <label>:58                                      ; preds = %23
  %59 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %24, i32 %desired acq_rel monotonic
  %60 = extractvalue { i32, i1 } %59, 1
  br i1 %60, label %66, label %64

; <label>:61                                      ; preds = %23
  %62 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %24, i32 %desired acq_rel acquire
  %63 = extractvalue { i32, i1 } %62, 1
  br i1 %63, label %70, label %68

; <label>:64                                      ; preds = %58
  %65 = extractvalue { i32, i1 } %59, 0
  store i32 %65, i32* %expected, align 4
  br label %66

; <label>:66                                      ; preds = %64, %58
  %67 = zext i1 %60 to i8
  br label %26

; <label>:68                                      ; preds = %61
  %69 = extractvalue { i32, i1 } %62, 0
  store i32 %69, i32* %expected, align 4
  br label %70

; <label>:70                                      ; preds = %68, %61
  %71 = zext i1 %63 to i8
  br label %26

; <label>:72                                      ; preds = %25
  %73 = load i32, i32* %expected, align 4
  %74 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %73, i32 %desired seq_cst monotonic
  %75 = extractvalue { i32, i1 } %74, 1
  br i1 %75, label %86, label %84

; <label>:76                                      ; preds = %25, %25
  %77 = load i32, i32* %expected, align 4
  %78 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %77, i32 %desired seq_cst acquire
  %79 = extractvalue { i32, i1 } %78, 1
  br i1 %79, label %90, label %88

; <label>:80                                      ; preds = %25
  %81 = load i32, i32* %expected, align 4
  %82 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %81, i32 %desired seq_cst seq_cst
  %83 = extractvalue { i32, i1 } %82, 1
  br i1 %83, label %94, label %92

; <label>:84                                      ; preds = %72
  %85 = extractvalue { i32, i1 } %74, 0
  store i32 %85, i32* %expected, align 4
  br label %86

; <label>:86                                      ; preds = %84, %72
  %87 = zext i1 %75 to i8
  br label %26

; <label>:88                                      ; preds = %76
  %89 = extractvalue { i32, i1 } %78, 0
  store i32 %89, i32* %expected, align 4
  br label %90

; <label>:90                                      ; preds = %88, %76
  %91 = zext i1 %79 to i8
  br label %26

; <label>:92                                      ; preds = %80
  %93 = extractvalue { i32, i1 } %82, 0
  store i32 %93, i32* %expected, align 4
  br label %94

; <label>:94                                      ; preds = %92, %80
  %95 = zext i1 %83 to i8
  br label %26
}

; Function Attrs: nounwind uwtable
define i32 @_Z29pocl_atomic_fetch_add__globalPVU3AS1U7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile add i32 addrspace(1)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile add i32 addrspace(1)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile add i32 addrspace(1)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile add i32 addrspace(1)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile add i32 addrspace(1)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z29pocl_atomic_fetch_sub__globalPVU3AS1U7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile sub i32 addrspace(1)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile sub i32 addrspace(1)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile sub i32 addrspace(1)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile sub i32 addrspace(1)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile sub i32 addrspace(1)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z28pocl_atomic_fetch_or__globalPVU3AS1U7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile or i32 addrspace(1)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile or i32 addrspace(1)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile or i32 addrspace(1)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile or i32 addrspace(1)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile or i32 addrspace(1)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z29pocl_atomic_fetch_xor__globalPVU3AS1U7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xor i32 addrspace(1)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xor i32 addrspace(1)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xor i32 addrspace(1)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xor i32 addrspace(1)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xor i32 addrspace(1)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z29pocl_atomic_fetch_and__globalPVU3AS1U7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile and i32 addrspace(1)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile and i32 addrspace(1)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile and i32 addrspace(1)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile and i32 addrspace(1)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile and i32 addrspace(1)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z29pocl_atomic_fetch_min__globalPVU3AS1U7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile min i32 addrspace(1)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile min i32 addrspace(1)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile min i32 addrspace(1)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile min i32 addrspace(1)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile min i32 addrspace(1)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z29pocl_atomic_fetch_max__globalPVU3AS1U7_Atomicii12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile max i32 addrspace(1)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile max i32 addrspace(1)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile max i32 addrspace(1)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile max i32 addrspace(1)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile max i32 addrspace(1)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define void @_Z25pocl_atomic_store__globalPVU3AS1U7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* nocapture %object, i32 %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %1 [
    i32 0, label %.thread
    i32 1, label %.thread
    i32 2, label %.thread1
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  store atomic volatile i32 %desired, i32 addrspace(1)* %object monotonic, align 4
  br label %2

.thread1:                                         ; preds = %0
  store atomic volatile i32 %desired, i32 addrspace(1)* %object release, align 4
  br label %2

; <label>:1                                       ; preds = %0
  store atomic volatile i32 %desired, i32 addrspace(1)* %object seq_cst, align 4
  br label %2

; <label>:2                                       ; preds = %1, %.thread1, %.thread
  ret void
}

; Function Attrs: nounwind uwtable
define i32 @_Z24pocl_atomic_load__globalPVU3AS1U7_Atomicj12memory_order12memory_scope(i32 addrspace(1)* nocapture readonly %object, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %3 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  %1 = load atomic volatile i32, i32 addrspace(1)* %object monotonic, align 4
  br label %5

.thread1:                                         ; preds = %0
  %2 = load atomic volatile i32, i32 addrspace(1)* %object acquire, align 4
  br label %5

; <label>:3                                       ; preds = %0
  %4 = load atomic volatile i32, i32 addrspace(1)* %object seq_cst, align 4
  br label %5

; <label>:5                                       ; preds = %3, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %4, %3 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z28pocl_atomic_exchange__globalPVU3AS1U7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xchg i32 addrspace(1)* %object, i32 %desired monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xchg i32 addrspace(1)* %object, i32 %desired acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xchg i32 addrspace(1)* %object, i32 %desired release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xchg i32 addrspace(1)* %object, i32 %desired acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xchg i32 addrspace(1)* %object, i32 %desired seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z43pocl_atomic_compare_exchange_strong__globalPVU3AS1U7_AtomicjPjj12memory_orderS3_12memory_scope(i32 addrspace(1)* %object, i32* nocapture %expected, i32 %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = icmp eq i32 %failure, 0
  br i1 %11, label %19, label %12

; <label>:12                                      ; preds = %9
  %13 = icmp eq i32 %failure, 1
  br i1 %13, label %19, label %14

; <label>:14                                      ; preds = %12
  %15 = icmp eq i32 %failure, 2
  br i1 %15, label %19, label %16

; <label>:16                                      ; preds = %14
  %17 = icmp eq i32 %failure, 3
  %18 = select i1 %17, i32 4, i32 5
  br label %19

; <label>:19                                      ; preds = %12, %14, %16, %9
  %20 = phi i32 [ 0, %9 ], [ 2, %12 ], [ %18, %16 ], [ 3, %14 ]
  switch i32 %10, label %28 [
    i32 1, label %21
    i32 2, label %21
    i32 3, label %50
    i32 4, label %23
    i32 5, label %25
  ]

; <label>:21                                      ; preds = %19, %19
  %.off = add nsw i32 %20, -1
  %switch = icmp ult i32 %.off, 2
  %22 = load i32, i32* %expected, align 4
  br i1 %switch, label %39, label %36

; <label>:23                                      ; preds = %19
  %.off1 = add nsw i32 %20, -1
  %switch2 = icmp ult i32 %.off1, 2
  %24 = load i32, i32* %expected, align 4
  br i1 %switch2, label %61, label %58

; <label>:25                                      ; preds = %19
  switch i32 %20, label %72 [
    i32 1, label %76
    i32 2, label %76
    i32 5, label %80
  ]

; <label>:26                                      ; preds = %86, %90, %94, %66, %70, %44, %48, %56, %34
  %.0 = phi i8 [ %35, %34 ], [ %87, %86 ], [ %95, %94 ], [ %91, %90 ], [ %71, %70 ], [ %67, %66 ], [ %57, %56 ], [ %49, %48 ], [ %45, %44 ]
  %27 = icmp ne i8 %.0, 0
  ret i1 %27

; <label>:28                                      ; preds = %19
  %29 = load i32, i32* %expected, align 4
  %30 = cmpxchg volatile i32 addrspace(1)* %object, i32 %29, i32 %desired monotonic monotonic
  %31 = extractvalue { i32, i1 } %30, 1
  br i1 %31, label %34, label %32

; <label>:32                                      ; preds = %28
  %33 = extractvalue { i32, i1 } %30, 0
  store i32 %33, i32* %expected, align 4
  br label %34

; <label>:34                                      ; preds = %32, %28
  %35 = zext i1 %31 to i8
  br label %26

; <label>:36                                      ; preds = %21
  %37 = cmpxchg volatile i32 addrspace(1)* %object, i32 %22, i32 %desired acquire monotonic
  %38 = extractvalue { i32, i1 } %37, 1
  br i1 %38, label %44, label %42

; <label>:39                                      ; preds = %21
  %40 = cmpxchg volatile i32 addrspace(1)* %object, i32 %22, i32 %desired acquire acquire
  %41 = extractvalue { i32, i1 } %40, 1
  br i1 %41, label %48, label %46

; <label>:42                                      ; preds = %36
  %43 = extractvalue { i32, i1 } %37, 0
  store i32 %43, i32* %expected, align 4
  br label %44

; <label>:44                                      ; preds = %42, %36
  %45 = zext i1 %38 to i8
  br label %26

; <label>:46                                      ; preds = %39
  %47 = extractvalue { i32, i1 } %40, 0
  store i32 %47, i32* %expected, align 4
  br label %48

; <label>:48                                      ; preds = %46, %39
  %49 = zext i1 %41 to i8
  br label %26

; <label>:50                                      ; preds = %19
  %51 = load i32, i32* %expected, align 4
  %52 = cmpxchg volatile i32 addrspace(1)* %object, i32 %51, i32 %desired release monotonic
  %53 = extractvalue { i32, i1 } %52, 1
  br i1 %53, label %56, label %54

; <label>:54                                      ; preds = %50
  %55 = extractvalue { i32, i1 } %52, 0
  store i32 %55, i32* %expected, align 4
  br label %56

; <label>:56                                      ; preds = %54, %50
  %57 = zext i1 %53 to i8
  br label %26

; <label>:58                                      ; preds = %23
  %59 = cmpxchg volatile i32 addrspace(1)* %object, i32 %24, i32 %desired acq_rel monotonic
  %60 = extractvalue { i32, i1 } %59, 1
  br i1 %60, label %66, label %64

; <label>:61                                      ; preds = %23
  %62 = cmpxchg volatile i32 addrspace(1)* %object, i32 %24, i32 %desired acq_rel acquire
  %63 = extractvalue { i32, i1 } %62, 1
  br i1 %63, label %70, label %68

; <label>:64                                      ; preds = %58
  %65 = extractvalue { i32, i1 } %59, 0
  store i32 %65, i32* %expected, align 4
  br label %66

; <label>:66                                      ; preds = %64, %58
  %67 = zext i1 %60 to i8
  br label %26

; <label>:68                                      ; preds = %61
  %69 = extractvalue { i32, i1 } %62, 0
  store i32 %69, i32* %expected, align 4
  br label %70

; <label>:70                                      ; preds = %68, %61
  %71 = zext i1 %63 to i8
  br label %26

; <label>:72                                      ; preds = %25
  %73 = load i32, i32* %expected, align 4
  %74 = cmpxchg volatile i32 addrspace(1)* %object, i32 %73, i32 %desired seq_cst monotonic
  %75 = extractvalue { i32, i1 } %74, 1
  br i1 %75, label %86, label %84

; <label>:76                                      ; preds = %25, %25
  %77 = load i32, i32* %expected, align 4
  %78 = cmpxchg volatile i32 addrspace(1)* %object, i32 %77, i32 %desired seq_cst acquire
  %79 = extractvalue { i32, i1 } %78, 1
  br i1 %79, label %90, label %88

; <label>:80                                      ; preds = %25
  %81 = load i32, i32* %expected, align 4
  %82 = cmpxchg volatile i32 addrspace(1)* %object, i32 %81, i32 %desired seq_cst seq_cst
  %83 = extractvalue { i32, i1 } %82, 1
  br i1 %83, label %94, label %92

; <label>:84                                      ; preds = %72
  %85 = extractvalue { i32, i1 } %74, 0
  store i32 %85, i32* %expected, align 4
  br label %86

; <label>:86                                      ; preds = %84, %72
  %87 = zext i1 %75 to i8
  br label %26

; <label>:88                                      ; preds = %76
  %89 = extractvalue { i32, i1 } %78, 0
  store i32 %89, i32* %expected, align 4
  br label %90

; <label>:90                                      ; preds = %88, %76
  %91 = zext i1 %79 to i8
  br label %26

; <label>:92                                      ; preds = %80
  %93 = extractvalue { i32, i1 } %82, 0
  store i32 %93, i32* %expected, align 4
  br label %94

; <label>:94                                      ; preds = %92, %80
  %95 = zext i1 %83 to i8
  br label %26
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z41pocl_atomic_compare_exchange_weak__globalPVU3AS1U7_AtomicjPjj12memory_orderS3_12memory_scope(i32 addrspace(1)* %object, i32* nocapture %expected, i32 %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = icmp eq i32 %failure, 0
  br i1 %11, label %19, label %12

; <label>:12                                      ; preds = %9
  %13 = icmp eq i32 %failure, 1
  br i1 %13, label %19, label %14

; <label>:14                                      ; preds = %12
  %15 = icmp eq i32 %failure, 2
  br i1 %15, label %19, label %16

; <label>:16                                      ; preds = %14
  %17 = icmp eq i32 %failure, 3
  %18 = select i1 %17, i32 4, i32 5
  br label %19

; <label>:19                                      ; preds = %12, %14, %16, %9
  %20 = phi i32 [ 0, %9 ], [ 2, %12 ], [ %18, %16 ], [ 3, %14 ]
  switch i32 %10, label %28 [
    i32 1, label %21
    i32 2, label %21
    i32 3, label %50
    i32 4, label %23
    i32 5, label %25
  ]

; <label>:21                                      ; preds = %19, %19
  %.off = add nsw i32 %20, -1
  %switch = icmp ult i32 %.off, 2
  %22 = load i32, i32* %expected, align 4
  br i1 %switch, label %39, label %36

; <label>:23                                      ; preds = %19
  %.off1 = add nsw i32 %20, -1
  %switch2 = icmp ult i32 %.off1, 2
  %24 = load i32, i32* %expected, align 4
  br i1 %switch2, label %61, label %58

; <label>:25                                      ; preds = %19
  switch i32 %20, label %72 [
    i32 1, label %76
    i32 2, label %76
    i32 5, label %80
  ]

; <label>:26                                      ; preds = %86, %90, %94, %66, %70, %44, %48, %56, %34
  %.0 = phi i8 [ %35, %34 ], [ %87, %86 ], [ %95, %94 ], [ %91, %90 ], [ %71, %70 ], [ %67, %66 ], [ %57, %56 ], [ %49, %48 ], [ %45, %44 ]
  %27 = icmp ne i8 %.0, 0
  ret i1 %27

; <label>:28                                      ; preds = %19
  %29 = load i32, i32* %expected, align 4
  %30 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %29, i32 %desired monotonic monotonic
  %31 = extractvalue { i32, i1 } %30, 1
  br i1 %31, label %34, label %32

; <label>:32                                      ; preds = %28
  %33 = extractvalue { i32, i1 } %30, 0
  store i32 %33, i32* %expected, align 4
  br label %34

; <label>:34                                      ; preds = %32, %28
  %35 = zext i1 %31 to i8
  br label %26

; <label>:36                                      ; preds = %21
  %37 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %22, i32 %desired acquire monotonic
  %38 = extractvalue { i32, i1 } %37, 1
  br i1 %38, label %44, label %42

; <label>:39                                      ; preds = %21
  %40 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %22, i32 %desired acquire acquire
  %41 = extractvalue { i32, i1 } %40, 1
  br i1 %41, label %48, label %46

; <label>:42                                      ; preds = %36
  %43 = extractvalue { i32, i1 } %37, 0
  store i32 %43, i32* %expected, align 4
  br label %44

; <label>:44                                      ; preds = %42, %36
  %45 = zext i1 %38 to i8
  br label %26

; <label>:46                                      ; preds = %39
  %47 = extractvalue { i32, i1 } %40, 0
  store i32 %47, i32* %expected, align 4
  br label %48

; <label>:48                                      ; preds = %46, %39
  %49 = zext i1 %41 to i8
  br label %26

; <label>:50                                      ; preds = %19
  %51 = load i32, i32* %expected, align 4
  %52 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %51, i32 %desired release monotonic
  %53 = extractvalue { i32, i1 } %52, 1
  br i1 %53, label %56, label %54

; <label>:54                                      ; preds = %50
  %55 = extractvalue { i32, i1 } %52, 0
  store i32 %55, i32* %expected, align 4
  br label %56

; <label>:56                                      ; preds = %54, %50
  %57 = zext i1 %53 to i8
  br label %26

; <label>:58                                      ; preds = %23
  %59 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %24, i32 %desired acq_rel monotonic
  %60 = extractvalue { i32, i1 } %59, 1
  br i1 %60, label %66, label %64

; <label>:61                                      ; preds = %23
  %62 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %24, i32 %desired acq_rel acquire
  %63 = extractvalue { i32, i1 } %62, 1
  br i1 %63, label %70, label %68

; <label>:64                                      ; preds = %58
  %65 = extractvalue { i32, i1 } %59, 0
  store i32 %65, i32* %expected, align 4
  br label %66

; <label>:66                                      ; preds = %64, %58
  %67 = zext i1 %60 to i8
  br label %26

; <label>:68                                      ; preds = %61
  %69 = extractvalue { i32, i1 } %62, 0
  store i32 %69, i32* %expected, align 4
  br label %70

; <label>:70                                      ; preds = %68, %61
  %71 = zext i1 %63 to i8
  br label %26

; <label>:72                                      ; preds = %25
  %73 = load i32, i32* %expected, align 4
  %74 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %73, i32 %desired seq_cst monotonic
  %75 = extractvalue { i32, i1 } %74, 1
  br i1 %75, label %86, label %84

; <label>:76                                      ; preds = %25, %25
  %77 = load i32, i32* %expected, align 4
  %78 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %77, i32 %desired seq_cst acquire
  %79 = extractvalue { i32, i1 } %78, 1
  br i1 %79, label %90, label %88

; <label>:80                                      ; preds = %25
  %81 = load i32, i32* %expected, align 4
  %82 = cmpxchg weak volatile i32 addrspace(1)* %object, i32 %81, i32 %desired seq_cst seq_cst
  %83 = extractvalue { i32, i1 } %82, 1
  br i1 %83, label %94, label %92

; <label>:84                                      ; preds = %72
  %85 = extractvalue { i32, i1 } %74, 0
  store i32 %85, i32* %expected, align 4
  br label %86

; <label>:86                                      ; preds = %84, %72
  %87 = zext i1 %75 to i8
  br label %26

; <label>:88                                      ; preds = %76
  %89 = extractvalue { i32, i1 } %78, 0
  store i32 %89, i32* %expected, align 4
  br label %90

; <label>:90                                      ; preds = %88, %76
  %91 = zext i1 %79 to i8
  br label %26

; <label>:92                                      ; preds = %80
  %93 = extractvalue { i32, i1 } %82, 0
  store i32 %93, i32* %expected, align 4
  br label %94

; <label>:94                                      ; preds = %92, %80
  %95 = zext i1 %83 to i8
  br label %26
}

; Function Attrs: nounwind uwtable
define i32 @_Z29pocl_atomic_fetch_add__globalPVU3AS1U7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile add i32 addrspace(1)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile add i32 addrspace(1)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile add i32 addrspace(1)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile add i32 addrspace(1)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile add i32 addrspace(1)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z29pocl_atomic_fetch_sub__globalPVU3AS1U7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile sub i32 addrspace(1)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile sub i32 addrspace(1)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile sub i32 addrspace(1)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile sub i32 addrspace(1)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile sub i32 addrspace(1)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z28pocl_atomic_fetch_or__globalPVU3AS1U7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile or i32 addrspace(1)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile or i32 addrspace(1)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile or i32 addrspace(1)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile or i32 addrspace(1)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile or i32 addrspace(1)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z29pocl_atomic_fetch_xor__globalPVU3AS1U7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xor i32 addrspace(1)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xor i32 addrspace(1)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xor i32 addrspace(1)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xor i32 addrspace(1)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xor i32 addrspace(1)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z29pocl_atomic_fetch_and__globalPVU3AS1U7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile and i32 addrspace(1)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile and i32 addrspace(1)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile and i32 addrspace(1)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile and i32 addrspace(1)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile and i32 addrspace(1)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z29pocl_atomic_fetch_min__globalPVU3AS1U7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile umin i32 addrspace(1)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile umin i32 addrspace(1)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile umin i32 addrspace(1)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile umin i32 addrspace(1)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile umin i32 addrspace(1)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z29pocl_atomic_fetch_max__globalPVU3AS1U7_Atomicjj12memory_order12memory_scope(i32 addrspace(1)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile umax i32 addrspace(1)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile umax i32 addrspace(1)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile umax i32 addrspace(1)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile umax i32 addrspace(1)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile umax i32 addrspace(1)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define void @_Z25pocl_atomic_store__globalPVU3AS1U7_Atomicff12memory_order12memory_scope(float addrspace(1)* nocapture %object, float %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %5 [
    i32 0, label %.thread
    i32 1, label %.thread
    i32 2, label %.thread1
  ]

.thread1:                                         ; preds = %0
  %1 = bitcast float %desired to i32
  %2 = bitcast float addrspace(1)* %object to i32 addrspace(1)*
  store atomic volatile i32 %1, i32 addrspace(1)* %2 release, align 4
  br label %13

.thread:                                          ; preds = %0, %0
  %3 = bitcast float %desired to i32
  %4 = bitcast float addrspace(1)* %object to i32 addrspace(1)*
  br label %9

; <label>:5                                       ; preds = %0
  %6 = icmp eq i32 %order, 3
  %7 = bitcast float %desired to i32
  %8 = bitcast float addrspace(1)* %object to i32 addrspace(1)*
  br i1 %6, label %9, label %12

; <label>:9                                       ; preds = %5, %.thread
  %10 = phi i32 addrspace(1)* [ %4, %.thread ], [ %8, %5 ]
  %11 = phi i32 [ %3, %.thread ], [ %7, %5 ]
  store atomic volatile i32 %11, i32 addrspace(1)* %10 monotonic, align 4
  br label %13

; <label>:12                                      ; preds = %5
  store atomic volatile i32 %7, i32 addrspace(1)* %8 seq_cst, align 4
  br label %13

; <label>:13                                      ; preds = %12, %.thread1, %9
  ret void
}

; Function Attrs: nounwind uwtable
define float @_Z24pocl_atomic_load__globalPVU3AS1U7_Atomicf12memory_order12memory_scope(float addrspace(1)* nocapture readonly %object, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %4 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread
  ]

.thread1:                                         ; preds = %0
  %1 = bitcast float addrspace(1)* %object to i32 addrspace(1)*
  %2 = load atomic volatile i32, i32 addrspace(1)* %1 acquire, align 4
  br label %12

.thread:                                          ; preds = %0, %0
  %3 = bitcast float addrspace(1)* %object to i32 addrspace(1)*
  br label %7

; <label>:4                                       ; preds = %0
  %5 = icmp eq i32 %order, 3
  %6 = bitcast float addrspace(1)* %object to i32 addrspace(1)*
  br i1 %5, label %7, label %10

; <label>:7                                       ; preds = %4, %.thread
  %8 = phi i32 addrspace(1)* [ %3, %.thread ], [ %6, %4 ]
  %9 = load atomic volatile i32, i32 addrspace(1)* %8 monotonic, align 4
  br label %12

; <label>:10                                      ; preds = %4
  %11 = load atomic volatile i32, i32 addrspace(1)* %6 seq_cst, align 4
  br label %12

; <label>:12                                      ; preds = %10, %.thread1, %7
  %.sroa.0.0 = phi i32 [ %9, %7 ], [ %11, %10 ], [ %2, %.thread1 ]
  %13 = bitcast i32 %.sroa.0.0 to float
  ret float %13
}

; Function Attrs: nounwind uwtable
define float @_Z28pocl_atomic_exchange__globalPVU3AS1U7_Atomicff12memory_order12memory_scope(float addrspace(1)* %object, float %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %10 [
    i32 0, label %.thread
    i32 1, label %.thread2
    i32 2, label %.thread3
  ]

.thread:                                          ; preds = %0
  %1 = bitcast float %desired to i32
  %2 = bitcast float addrspace(1)* %object to i32 addrspace(1)*
  %3 = atomicrmw volatile xchg i32 addrspace(1)* %2, i32 %1 monotonic
  br label %18

.thread2:                                         ; preds = %0
  %4 = bitcast float %desired to i32
  %5 = bitcast float addrspace(1)* %object to i32 addrspace(1)*
  %6 = atomicrmw volatile xchg i32 addrspace(1)* %5, i32 %4 acquire
  br label %18

.thread3:                                         ; preds = %0
  %7 = bitcast float %desired to i32
  %8 = bitcast float addrspace(1)* %object to i32 addrspace(1)*
  %9 = atomicrmw volatile xchg i32 addrspace(1)* %8, i32 %7 release
  br label %18

; <label>:10                                      ; preds = %0
  %11 = icmp eq i32 %order, 3
  %12 = bitcast float %desired to i32
  %13 = bitcast float addrspace(1)* %object to i32 addrspace(1)*
  br i1 %11, label %14, label %16

; <label>:14                                      ; preds = %10
  %15 = atomicrmw volatile xchg i32 addrspace(1)* %13, i32 %12 acq_rel
  br label %18

; <label>:16                                      ; preds = %10
  %17 = atomicrmw volatile xchg i32 addrspace(1)* %13, i32 %12 seq_cst
  br label %18

; <label>:18                                      ; preds = %16, %14, %.thread3, %.thread2, %.thread
  %.sroa.0.0 = phi i32 [ %3, %.thread ], [ %17, %16 ], [ %15, %14 ], [ %9, %.thread3 ], [ %6, %.thread2 ]
  %19 = bitcast i32 %.sroa.0.0 to float
  ret float %19
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z43pocl_atomic_compare_exchange_strong__globalPVU3AS1U7_AtomicfPff12memory_orderS3_12memory_scope(float addrspace(1)* %object, float* nocapture %expected, float %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = bitcast float %desired to i32
  %12 = icmp eq i32 %failure, 0
  br i1 %12, label %20, label %13

; <label>:13                                      ; preds = %9
  %14 = icmp eq i32 %failure, 1
  br i1 %14, label %20, label %15

; <label>:15                                      ; preds = %13
  %16 = icmp eq i32 %failure, 2
  br i1 %16, label %20, label %17

; <label>:17                                      ; preds = %15
  %18 = icmp eq i32 %failure, 3
  %19 = select i1 %18, i32 4, i32 5
  br label %20

; <label>:20                                      ; preds = %13, %15, %17, %9
  %21 = phi i32 [ 0, %9 ], [ 2, %13 ], [ %19, %17 ], [ 3, %15 ]
  %22 = bitcast float addrspace(1)* %object to i32 addrspace(1)*
  %23 = bitcast float* %expected to i32*
  switch i32 %10, label %31 [
    i32 1, label %24
    i32 2, label %24
    i32 3, label %53
    i32 4, label %26
    i32 5, label %28
  ]

; <label>:24                                      ; preds = %20, %20
  %.off = add nsw i32 %21, -1
  %switch = icmp ult i32 %.off, 2
  %25 = load i32, i32* %23, align 4
  br i1 %switch, label %42, label %39

; <label>:26                                      ; preds = %20
  %.off1 = add nsw i32 %21, -1
  %switch2 = icmp ult i32 %.off1, 2
  %27 = load i32, i32* %23, align 4
  br i1 %switch2, label %64, label %61

; <label>:28                                      ; preds = %20
  switch i32 %21, label %75 [
    i32 1, label %79
    i32 2, label %79
    i32 5, label %83
  ]

; <label>:29                                      ; preds = %89, %93, %97, %69, %73, %47, %51, %59, %37
  %.0 = phi i8 [ %38, %37 ], [ %90, %89 ], [ %98, %97 ], [ %94, %93 ], [ %74, %73 ], [ %70, %69 ], [ %60, %59 ], [ %52, %51 ], [ %48, %47 ]
  %30 = icmp ne i8 %.0, 0
  ret i1 %30

; <label>:31                                      ; preds = %20
  %32 = load i32, i32* %23, align 4
  %33 = cmpxchg volatile i32 addrspace(1)* %22, i32 %32, i32 %11 monotonic monotonic
  %34 = extractvalue { i32, i1 } %33, 1
  br i1 %34, label %37, label %35

; <label>:35                                      ; preds = %31
  %36 = extractvalue { i32, i1 } %33, 0
  store i32 %36, i32* %23, align 4
  br label %37

; <label>:37                                      ; preds = %35, %31
  %38 = zext i1 %34 to i8
  br label %29

; <label>:39                                      ; preds = %24
  %40 = cmpxchg volatile i32 addrspace(1)* %22, i32 %25, i32 %11 acquire monotonic
  %41 = extractvalue { i32, i1 } %40, 1
  br i1 %41, label %47, label %45

; <label>:42                                      ; preds = %24
  %43 = cmpxchg volatile i32 addrspace(1)* %22, i32 %25, i32 %11 acquire acquire
  %44 = extractvalue { i32, i1 } %43, 1
  br i1 %44, label %51, label %49

; <label>:45                                      ; preds = %39
  %46 = extractvalue { i32, i1 } %40, 0
  store i32 %46, i32* %23, align 4
  br label %47

; <label>:47                                      ; preds = %45, %39
  %48 = zext i1 %41 to i8
  br label %29

; <label>:49                                      ; preds = %42
  %50 = extractvalue { i32, i1 } %43, 0
  store i32 %50, i32* %23, align 4
  br label %51

; <label>:51                                      ; preds = %49, %42
  %52 = zext i1 %44 to i8
  br label %29

; <label>:53                                      ; preds = %20
  %54 = load i32, i32* %23, align 4
  %55 = cmpxchg volatile i32 addrspace(1)* %22, i32 %54, i32 %11 release monotonic
  %56 = extractvalue { i32, i1 } %55, 1
  br i1 %56, label %59, label %57

; <label>:57                                      ; preds = %53
  %58 = extractvalue { i32, i1 } %55, 0
  store i32 %58, i32* %23, align 4
  br label %59

; <label>:59                                      ; preds = %57, %53
  %60 = zext i1 %56 to i8
  br label %29

; <label>:61                                      ; preds = %26
  %62 = cmpxchg volatile i32 addrspace(1)* %22, i32 %27, i32 %11 acq_rel monotonic
  %63 = extractvalue { i32, i1 } %62, 1
  br i1 %63, label %69, label %67

; <label>:64                                      ; preds = %26
  %65 = cmpxchg volatile i32 addrspace(1)* %22, i32 %27, i32 %11 acq_rel acquire
  %66 = extractvalue { i32, i1 } %65, 1
  br i1 %66, label %73, label %71

; <label>:67                                      ; preds = %61
  %68 = extractvalue { i32, i1 } %62, 0
  store i32 %68, i32* %23, align 4
  br label %69

; <label>:69                                      ; preds = %67, %61
  %70 = zext i1 %63 to i8
  br label %29

; <label>:71                                      ; preds = %64
  %72 = extractvalue { i32, i1 } %65, 0
  store i32 %72, i32* %23, align 4
  br label %73

; <label>:73                                      ; preds = %71, %64
  %74 = zext i1 %66 to i8
  br label %29

; <label>:75                                      ; preds = %28
  %76 = load i32, i32* %23, align 4
  %77 = cmpxchg volatile i32 addrspace(1)* %22, i32 %76, i32 %11 seq_cst monotonic
  %78 = extractvalue { i32, i1 } %77, 1
  br i1 %78, label %89, label %87

; <label>:79                                      ; preds = %28, %28
  %80 = load i32, i32* %23, align 4
  %81 = cmpxchg volatile i32 addrspace(1)* %22, i32 %80, i32 %11 seq_cst acquire
  %82 = extractvalue { i32, i1 } %81, 1
  br i1 %82, label %93, label %91

; <label>:83                                      ; preds = %28
  %84 = load i32, i32* %23, align 4
  %85 = cmpxchg volatile i32 addrspace(1)* %22, i32 %84, i32 %11 seq_cst seq_cst
  %86 = extractvalue { i32, i1 } %85, 1
  br i1 %86, label %97, label %95

; <label>:87                                      ; preds = %75
  %88 = extractvalue { i32, i1 } %77, 0
  store i32 %88, i32* %23, align 4
  br label %89

; <label>:89                                      ; preds = %87, %75
  %90 = zext i1 %78 to i8
  br label %29

; <label>:91                                      ; preds = %79
  %92 = extractvalue { i32, i1 } %81, 0
  store i32 %92, i32* %23, align 4
  br label %93

; <label>:93                                      ; preds = %91, %79
  %94 = zext i1 %82 to i8
  br label %29

; <label>:95                                      ; preds = %83
  %96 = extractvalue { i32, i1 } %85, 0
  store i32 %96, i32* %23, align 4
  br label %97

; <label>:97                                      ; preds = %95, %83
  %98 = zext i1 %86 to i8
  br label %29
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z41pocl_atomic_compare_exchange_weak__globalPVU3AS1U7_AtomicfPff12memory_orderS3_12memory_scope(float addrspace(1)* %object, float* nocapture %expected, float %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = bitcast float %desired to i32
  %12 = icmp eq i32 %failure, 0
  br i1 %12, label %20, label %13

; <label>:13                                      ; preds = %9
  %14 = icmp eq i32 %failure, 1
  br i1 %14, label %20, label %15

; <label>:15                                      ; preds = %13
  %16 = icmp eq i32 %failure, 2
  br i1 %16, label %20, label %17

; <label>:17                                      ; preds = %15
  %18 = icmp eq i32 %failure, 3
  %19 = select i1 %18, i32 4, i32 5
  br label %20

; <label>:20                                      ; preds = %13, %15, %17, %9
  %21 = phi i32 [ 0, %9 ], [ 2, %13 ], [ %19, %17 ], [ 3, %15 ]
  %22 = bitcast float addrspace(1)* %object to i32 addrspace(1)*
  %23 = bitcast float* %expected to i32*
  switch i32 %10, label %31 [
    i32 1, label %24
    i32 2, label %24
    i32 3, label %53
    i32 4, label %26
    i32 5, label %28
  ]

; <label>:24                                      ; preds = %20, %20
  %.off = add nsw i32 %21, -1
  %switch = icmp ult i32 %.off, 2
  %25 = load i32, i32* %23, align 4
  br i1 %switch, label %42, label %39

; <label>:26                                      ; preds = %20
  %.off1 = add nsw i32 %21, -1
  %switch2 = icmp ult i32 %.off1, 2
  %27 = load i32, i32* %23, align 4
  br i1 %switch2, label %64, label %61

; <label>:28                                      ; preds = %20
  switch i32 %21, label %75 [
    i32 1, label %79
    i32 2, label %79
    i32 5, label %83
  ]

; <label>:29                                      ; preds = %89, %93, %97, %69, %73, %47, %51, %59, %37
  %.0 = phi i8 [ %38, %37 ], [ %90, %89 ], [ %98, %97 ], [ %94, %93 ], [ %74, %73 ], [ %70, %69 ], [ %60, %59 ], [ %52, %51 ], [ %48, %47 ]
  %30 = icmp ne i8 %.0, 0
  ret i1 %30

; <label>:31                                      ; preds = %20
  %32 = load i32, i32* %23, align 4
  %33 = cmpxchg weak volatile i32 addrspace(1)* %22, i32 %32, i32 %11 monotonic monotonic
  %34 = extractvalue { i32, i1 } %33, 1
  br i1 %34, label %37, label %35

; <label>:35                                      ; preds = %31
  %36 = extractvalue { i32, i1 } %33, 0
  store i32 %36, i32* %23, align 4
  br label %37

; <label>:37                                      ; preds = %35, %31
  %38 = zext i1 %34 to i8
  br label %29

; <label>:39                                      ; preds = %24
  %40 = cmpxchg weak volatile i32 addrspace(1)* %22, i32 %25, i32 %11 acquire monotonic
  %41 = extractvalue { i32, i1 } %40, 1
  br i1 %41, label %47, label %45

; <label>:42                                      ; preds = %24
  %43 = cmpxchg weak volatile i32 addrspace(1)* %22, i32 %25, i32 %11 acquire acquire
  %44 = extractvalue { i32, i1 } %43, 1
  br i1 %44, label %51, label %49

; <label>:45                                      ; preds = %39
  %46 = extractvalue { i32, i1 } %40, 0
  store i32 %46, i32* %23, align 4
  br label %47

; <label>:47                                      ; preds = %45, %39
  %48 = zext i1 %41 to i8
  br label %29

; <label>:49                                      ; preds = %42
  %50 = extractvalue { i32, i1 } %43, 0
  store i32 %50, i32* %23, align 4
  br label %51

; <label>:51                                      ; preds = %49, %42
  %52 = zext i1 %44 to i8
  br label %29

; <label>:53                                      ; preds = %20
  %54 = load i32, i32* %23, align 4
  %55 = cmpxchg weak volatile i32 addrspace(1)* %22, i32 %54, i32 %11 release monotonic
  %56 = extractvalue { i32, i1 } %55, 1
  br i1 %56, label %59, label %57

; <label>:57                                      ; preds = %53
  %58 = extractvalue { i32, i1 } %55, 0
  store i32 %58, i32* %23, align 4
  br label %59

; <label>:59                                      ; preds = %57, %53
  %60 = zext i1 %56 to i8
  br label %29

; <label>:61                                      ; preds = %26
  %62 = cmpxchg weak volatile i32 addrspace(1)* %22, i32 %27, i32 %11 acq_rel monotonic
  %63 = extractvalue { i32, i1 } %62, 1
  br i1 %63, label %69, label %67

; <label>:64                                      ; preds = %26
  %65 = cmpxchg weak volatile i32 addrspace(1)* %22, i32 %27, i32 %11 acq_rel acquire
  %66 = extractvalue { i32, i1 } %65, 1
  br i1 %66, label %73, label %71

; <label>:67                                      ; preds = %61
  %68 = extractvalue { i32, i1 } %62, 0
  store i32 %68, i32* %23, align 4
  br label %69

; <label>:69                                      ; preds = %67, %61
  %70 = zext i1 %63 to i8
  br label %29

; <label>:71                                      ; preds = %64
  %72 = extractvalue { i32, i1 } %65, 0
  store i32 %72, i32* %23, align 4
  br label %73

; <label>:73                                      ; preds = %71, %64
  %74 = zext i1 %66 to i8
  br label %29

; <label>:75                                      ; preds = %28
  %76 = load i32, i32* %23, align 4
  %77 = cmpxchg weak volatile i32 addrspace(1)* %22, i32 %76, i32 %11 seq_cst monotonic
  %78 = extractvalue { i32, i1 } %77, 1
  br i1 %78, label %89, label %87

; <label>:79                                      ; preds = %28, %28
  %80 = load i32, i32* %23, align 4
  %81 = cmpxchg weak volatile i32 addrspace(1)* %22, i32 %80, i32 %11 seq_cst acquire
  %82 = extractvalue { i32, i1 } %81, 1
  br i1 %82, label %93, label %91

; <label>:83                                      ; preds = %28
  %84 = load i32, i32* %23, align 4
  %85 = cmpxchg weak volatile i32 addrspace(1)* %22, i32 %84, i32 %11 seq_cst seq_cst
  %86 = extractvalue { i32, i1 } %85, 1
  br i1 %86, label %97, label %95

; <label>:87                                      ; preds = %75
  %88 = extractvalue { i32, i1 } %77, 0
  store i32 %88, i32* %23, align 4
  br label %89

; <label>:89                                      ; preds = %87, %75
  %90 = zext i1 %78 to i8
  br label %29

; <label>:91                                      ; preds = %79
  %92 = extractvalue { i32, i1 } %81, 0
  store i32 %92, i32* %23, align 4
  br label %93

; <label>:93                                      ; preds = %91, %79
  %94 = zext i1 %82 to i8
  br label %29

; <label>:95                                      ; preds = %83
  %96 = extractvalue { i32, i1 } %85, 0
  store i32 %96, i32* %23, align 4
  br label %97

; <label>:97                                      ; preds = %95, %83
  %98 = zext i1 %86 to i8
  br label %29
}

; Function Attrs: nounwind uwtable
define void @_Z25pocl_atomic_store__globalPVU3AS1U7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* nocapture %object, i64 %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %1 [
    i32 0, label %.thread
    i32 1, label %.thread
    i32 2, label %.thread1
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  store atomic volatile i64 %desired, i64 addrspace(1)* %object monotonic, align 8
  br label %2

.thread1:                                         ; preds = %0
  store atomic volatile i64 %desired, i64 addrspace(1)* %object release, align 8
  br label %2

; <label>:1                                       ; preds = %0
  store atomic volatile i64 %desired, i64 addrspace(1)* %object seq_cst, align 8
  br label %2

; <label>:2                                       ; preds = %1, %.thread1, %.thread
  ret void
}

; Function Attrs: nounwind uwtable
define i64 @_Z24pocl_atomic_load__globalPVU3AS1U7_Atomicl12memory_order12memory_scope(i64 addrspace(1)* nocapture readonly %object, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %3 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  %1 = load atomic volatile i64, i64 addrspace(1)* %object monotonic, align 8
  br label %5

.thread1:                                         ; preds = %0
  %2 = load atomic volatile i64, i64 addrspace(1)* %object acquire, align 8
  br label %5

; <label>:3                                       ; preds = %0
  %4 = load atomic volatile i64, i64 addrspace(1)* %object seq_cst, align 8
  br label %5

; <label>:5                                       ; preds = %3, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %4, %3 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z28pocl_atomic_exchange__globalPVU3AS1U7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* %object, i64 %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xchg i64 addrspace(1)* %object, i64 %desired monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xchg i64 addrspace(1)* %object, i64 %desired acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xchg i64 addrspace(1)* %object, i64 %desired release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xchg i64 addrspace(1)* %object, i64 %desired acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xchg i64 addrspace(1)* %object, i64 %desired seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z43pocl_atomic_compare_exchange_strong__globalPVU3AS1U7_AtomiclPll12memory_orderS3_12memory_scope(i64 addrspace(1)* %object, i64* nocapture %expected, i64 %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = icmp eq i32 %failure, 0
  br i1 %11, label %19, label %12

; <label>:12                                      ; preds = %9
  %13 = icmp eq i32 %failure, 1
  br i1 %13, label %19, label %14

; <label>:14                                      ; preds = %12
  %15 = icmp eq i32 %failure, 2
  br i1 %15, label %19, label %16

; <label>:16                                      ; preds = %14
  %17 = icmp eq i32 %failure, 3
  %18 = select i1 %17, i32 4, i32 5
  br label %19

; <label>:19                                      ; preds = %12, %14, %16, %9
  %20 = phi i32 [ 0, %9 ], [ 2, %12 ], [ %18, %16 ], [ 3, %14 ]
  switch i32 %10, label %28 [
    i32 1, label %21
    i32 2, label %21
    i32 3, label %50
    i32 4, label %23
    i32 5, label %25
  ]

; <label>:21                                      ; preds = %19, %19
  %.off = add nsw i32 %20, -1
  %switch = icmp ult i32 %.off, 2
  %22 = load i64, i64* %expected, align 8
  br i1 %switch, label %39, label %36

; <label>:23                                      ; preds = %19
  %.off1 = add nsw i32 %20, -1
  %switch2 = icmp ult i32 %.off1, 2
  %24 = load i64, i64* %expected, align 8
  br i1 %switch2, label %61, label %58

; <label>:25                                      ; preds = %19
  switch i32 %20, label %72 [
    i32 1, label %76
    i32 2, label %76
    i32 5, label %80
  ]

; <label>:26                                      ; preds = %86, %90, %94, %66, %70, %44, %48, %56, %34
  %.0 = phi i8 [ %35, %34 ], [ %87, %86 ], [ %95, %94 ], [ %91, %90 ], [ %71, %70 ], [ %67, %66 ], [ %57, %56 ], [ %49, %48 ], [ %45, %44 ]
  %27 = icmp ne i8 %.0, 0
  ret i1 %27

; <label>:28                                      ; preds = %19
  %29 = load i64, i64* %expected, align 8
  %30 = cmpxchg volatile i64 addrspace(1)* %object, i64 %29, i64 %desired monotonic monotonic
  %31 = extractvalue { i64, i1 } %30, 1
  br i1 %31, label %34, label %32

; <label>:32                                      ; preds = %28
  %33 = extractvalue { i64, i1 } %30, 0
  store i64 %33, i64* %expected, align 8
  br label %34

; <label>:34                                      ; preds = %32, %28
  %35 = zext i1 %31 to i8
  br label %26

; <label>:36                                      ; preds = %21
  %37 = cmpxchg volatile i64 addrspace(1)* %object, i64 %22, i64 %desired acquire monotonic
  %38 = extractvalue { i64, i1 } %37, 1
  br i1 %38, label %44, label %42

; <label>:39                                      ; preds = %21
  %40 = cmpxchg volatile i64 addrspace(1)* %object, i64 %22, i64 %desired acquire acquire
  %41 = extractvalue { i64, i1 } %40, 1
  br i1 %41, label %48, label %46

; <label>:42                                      ; preds = %36
  %43 = extractvalue { i64, i1 } %37, 0
  store i64 %43, i64* %expected, align 8
  br label %44

; <label>:44                                      ; preds = %42, %36
  %45 = zext i1 %38 to i8
  br label %26

; <label>:46                                      ; preds = %39
  %47 = extractvalue { i64, i1 } %40, 0
  store i64 %47, i64* %expected, align 8
  br label %48

; <label>:48                                      ; preds = %46, %39
  %49 = zext i1 %41 to i8
  br label %26

; <label>:50                                      ; preds = %19
  %51 = load i64, i64* %expected, align 8
  %52 = cmpxchg volatile i64 addrspace(1)* %object, i64 %51, i64 %desired release monotonic
  %53 = extractvalue { i64, i1 } %52, 1
  br i1 %53, label %56, label %54

; <label>:54                                      ; preds = %50
  %55 = extractvalue { i64, i1 } %52, 0
  store i64 %55, i64* %expected, align 8
  br label %56

; <label>:56                                      ; preds = %54, %50
  %57 = zext i1 %53 to i8
  br label %26

; <label>:58                                      ; preds = %23
  %59 = cmpxchg volatile i64 addrspace(1)* %object, i64 %24, i64 %desired acq_rel monotonic
  %60 = extractvalue { i64, i1 } %59, 1
  br i1 %60, label %66, label %64

; <label>:61                                      ; preds = %23
  %62 = cmpxchg volatile i64 addrspace(1)* %object, i64 %24, i64 %desired acq_rel acquire
  %63 = extractvalue { i64, i1 } %62, 1
  br i1 %63, label %70, label %68

; <label>:64                                      ; preds = %58
  %65 = extractvalue { i64, i1 } %59, 0
  store i64 %65, i64* %expected, align 8
  br label %66

; <label>:66                                      ; preds = %64, %58
  %67 = zext i1 %60 to i8
  br label %26

; <label>:68                                      ; preds = %61
  %69 = extractvalue { i64, i1 } %62, 0
  store i64 %69, i64* %expected, align 8
  br label %70

; <label>:70                                      ; preds = %68, %61
  %71 = zext i1 %63 to i8
  br label %26

; <label>:72                                      ; preds = %25
  %73 = load i64, i64* %expected, align 8
  %74 = cmpxchg volatile i64 addrspace(1)* %object, i64 %73, i64 %desired seq_cst monotonic
  %75 = extractvalue { i64, i1 } %74, 1
  br i1 %75, label %86, label %84

; <label>:76                                      ; preds = %25, %25
  %77 = load i64, i64* %expected, align 8
  %78 = cmpxchg volatile i64 addrspace(1)* %object, i64 %77, i64 %desired seq_cst acquire
  %79 = extractvalue { i64, i1 } %78, 1
  br i1 %79, label %90, label %88

; <label>:80                                      ; preds = %25
  %81 = load i64, i64* %expected, align 8
  %82 = cmpxchg volatile i64 addrspace(1)* %object, i64 %81, i64 %desired seq_cst seq_cst
  %83 = extractvalue { i64, i1 } %82, 1
  br i1 %83, label %94, label %92

; <label>:84                                      ; preds = %72
  %85 = extractvalue { i64, i1 } %74, 0
  store i64 %85, i64* %expected, align 8
  br label %86

; <label>:86                                      ; preds = %84, %72
  %87 = zext i1 %75 to i8
  br label %26

; <label>:88                                      ; preds = %76
  %89 = extractvalue { i64, i1 } %78, 0
  store i64 %89, i64* %expected, align 8
  br label %90

; <label>:90                                      ; preds = %88, %76
  %91 = zext i1 %79 to i8
  br label %26

; <label>:92                                      ; preds = %80
  %93 = extractvalue { i64, i1 } %82, 0
  store i64 %93, i64* %expected, align 8
  br label %94

; <label>:94                                      ; preds = %92, %80
  %95 = zext i1 %83 to i8
  br label %26
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z41pocl_atomic_compare_exchange_weak__globalPVU3AS1U7_AtomiclPll12memory_orderS3_12memory_scope(i64 addrspace(1)* %object, i64* nocapture %expected, i64 %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = icmp eq i32 %failure, 0
  br i1 %11, label %19, label %12

; <label>:12                                      ; preds = %9
  %13 = icmp eq i32 %failure, 1
  br i1 %13, label %19, label %14

; <label>:14                                      ; preds = %12
  %15 = icmp eq i32 %failure, 2
  br i1 %15, label %19, label %16

; <label>:16                                      ; preds = %14
  %17 = icmp eq i32 %failure, 3
  %18 = select i1 %17, i32 4, i32 5
  br label %19

; <label>:19                                      ; preds = %12, %14, %16, %9
  %20 = phi i32 [ 0, %9 ], [ 2, %12 ], [ %18, %16 ], [ 3, %14 ]
  switch i32 %10, label %28 [
    i32 1, label %21
    i32 2, label %21
    i32 3, label %50
    i32 4, label %23
    i32 5, label %25
  ]

; <label>:21                                      ; preds = %19, %19
  %.off = add nsw i32 %20, -1
  %switch = icmp ult i32 %.off, 2
  %22 = load i64, i64* %expected, align 8
  br i1 %switch, label %39, label %36

; <label>:23                                      ; preds = %19
  %.off1 = add nsw i32 %20, -1
  %switch2 = icmp ult i32 %.off1, 2
  %24 = load i64, i64* %expected, align 8
  br i1 %switch2, label %61, label %58

; <label>:25                                      ; preds = %19
  switch i32 %20, label %72 [
    i32 1, label %76
    i32 2, label %76
    i32 5, label %80
  ]

; <label>:26                                      ; preds = %86, %90, %94, %66, %70, %44, %48, %56, %34
  %.0 = phi i8 [ %35, %34 ], [ %87, %86 ], [ %95, %94 ], [ %91, %90 ], [ %71, %70 ], [ %67, %66 ], [ %57, %56 ], [ %49, %48 ], [ %45, %44 ]
  %27 = icmp ne i8 %.0, 0
  ret i1 %27

; <label>:28                                      ; preds = %19
  %29 = load i64, i64* %expected, align 8
  %30 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %29, i64 %desired monotonic monotonic
  %31 = extractvalue { i64, i1 } %30, 1
  br i1 %31, label %34, label %32

; <label>:32                                      ; preds = %28
  %33 = extractvalue { i64, i1 } %30, 0
  store i64 %33, i64* %expected, align 8
  br label %34

; <label>:34                                      ; preds = %32, %28
  %35 = zext i1 %31 to i8
  br label %26

; <label>:36                                      ; preds = %21
  %37 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %22, i64 %desired acquire monotonic
  %38 = extractvalue { i64, i1 } %37, 1
  br i1 %38, label %44, label %42

; <label>:39                                      ; preds = %21
  %40 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %22, i64 %desired acquire acquire
  %41 = extractvalue { i64, i1 } %40, 1
  br i1 %41, label %48, label %46

; <label>:42                                      ; preds = %36
  %43 = extractvalue { i64, i1 } %37, 0
  store i64 %43, i64* %expected, align 8
  br label %44

; <label>:44                                      ; preds = %42, %36
  %45 = zext i1 %38 to i8
  br label %26

; <label>:46                                      ; preds = %39
  %47 = extractvalue { i64, i1 } %40, 0
  store i64 %47, i64* %expected, align 8
  br label %48

; <label>:48                                      ; preds = %46, %39
  %49 = zext i1 %41 to i8
  br label %26

; <label>:50                                      ; preds = %19
  %51 = load i64, i64* %expected, align 8
  %52 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %51, i64 %desired release monotonic
  %53 = extractvalue { i64, i1 } %52, 1
  br i1 %53, label %56, label %54

; <label>:54                                      ; preds = %50
  %55 = extractvalue { i64, i1 } %52, 0
  store i64 %55, i64* %expected, align 8
  br label %56

; <label>:56                                      ; preds = %54, %50
  %57 = zext i1 %53 to i8
  br label %26

; <label>:58                                      ; preds = %23
  %59 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %24, i64 %desired acq_rel monotonic
  %60 = extractvalue { i64, i1 } %59, 1
  br i1 %60, label %66, label %64

; <label>:61                                      ; preds = %23
  %62 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %24, i64 %desired acq_rel acquire
  %63 = extractvalue { i64, i1 } %62, 1
  br i1 %63, label %70, label %68

; <label>:64                                      ; preds = %58
  %65 = extractvalue { i64, i1 } %59, 0
  store i64 %65, i64* %expected, align 8
  br label %66

; <label>:66                                      ; preds = %64, %58
  %67 = zext i1 %60 to i8
  br label %26

; <label>:68                                      ; preds = %61
  %69 = extractvalue { i64, i1 } %62, 0
  store i64 %69, i64* %expected, align 8
  br label %70

; <label>:70                                      ; preds = %68, %61
  %71 = zext i1 %63 to i8
  br label %26

; <label>:72                                      ; preds = %25
  %73 = load i64, i64* %expected, align 8
  %74 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %73, i64 %desired seq_cst monotonic
  %75 = extractvalue { i64, i1 } %74, 1
  br i1 %75, label %86, label %84

; <label>:76                                      ; preds = %25, %25
  %77 = load i64, i64* %expected, align 8
  %78 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %77, i64 %desired seq_cst acquire
  %79 = extractvalue { i64, i1 } %78, 1
  br i1 %79, label %90, label %88

; <label>:80                                      ; preds = %25
  %81 = load i64, i64* %expected, align 8
  %82 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %81, i64 %desired seq_cst seq_cst
  %83 = extractvalue { i64, i1 } %82, 1
  br i1 %83, label %94, label %92

; <label>:84                                      ; preds = %72
  %85 = extractvalue { i64, i1 } %74, 0
  store i64 %85, i64* %expected, align 8
  br label %86

; <label>:86                                      ; preds = %84, %72
  %87 = zext i1 %75 to i8
  br label %26

; <label>:88                                      ; preds = %76
  %89 = extractvalue { i64, i1 } %78, 0
  store i64 %89, i64* %expected, align 8
  br label %90

; <label>:90                                      ; preds = %88, %76
  %91 = zext i1 %79 to i8
  br label %26

; <label>:92                                      ; preds = %80
  %93 = extractvalue { i64, i1 } %82, 0
  store i64 %93, i64* %expected, align 8
  br label %94

; <label>:94                                      ; preds = %92, %80
  %95 = zext i1 %83 to i8
  br label %26
}

; Function Attrs: nounwind uwtable
define i64 @_Z29pocl_atomic_fetch_add__globalPVU3AS1U7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile add i64 addrspace(1)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile add i64 addrspace(1)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile add i64 addrspace(1)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile add i64 addrspace(1)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile add i64 addrspace(1)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z29pocl_atomic_fetch_sub__globalPVU3AS1U7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile sub i64 addrspace(1)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile sub i64 addrspace(1)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile sub i64 addrspace(1)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile sub i64 addrspace(1)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile sub i64 addrspace(1)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z28pocl_atomic_fetch_or__globalPVU3AS1U7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile or i64 addrspace(1)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile or i64 addrspace(1)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile or i64 addrspace(1)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile or i64 addrspace(1)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile or i64 addrspace(1)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z29pocl_atomic_fetch_xor__globalPVU3AS1U7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xor i64 addrspace(1)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xor i64 addrspace(1)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xor i64 addrspace(1)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xor i64 addrspace(1)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xor i64 addrspace(1)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z29pocl_atomic_fetch_and__globalPVU3AS1U7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile and i64 addrspace(1)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile and i64 addrspace(1)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile and i64 addrspace(1)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile and i64 addrspace(1)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile and i64 addrspace(1)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: noreturn nounwind uwtable
define i64 @_Z29pocl_atomic_fetch_min__globalPVU3AS1U7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* nocapture readnone %object, i64 %operand, i32 %order, i32 %scope) #1 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile min i64 addrspace(1)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile min i64 addrspace(1)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile min i64 addrspace(1)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile min i64 addrspace(1)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile min i64 addrspace(1)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: noreturn nounwind uwtable
define i64 @_Z29pocl_atomic_fetch_max__globalPVU3AS1U7_Atomicll12memory_order12memory_scope(i64 addrspace(1)* nocapture readnone %object, i64 %operand, i32 %order, i32 %scope) #1 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile max i64 addrspace(1)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile max i64 addrspace(1)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile max i64 addrspace(1)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile max i64 addrspace(1)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile max i64 addrspace(1)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define void @_Z25pocl_atomic_store__globalPVU3AS1U7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* nocapture %object, i64 %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %1 [
    i32 0, label %.thread
    i32 1, label %.thread
    i32 2, label %.thread1
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  store atomic volatile i64 %desired, i64 addrspace(1)* %object monotonic, align 8
  br label %2

.thread1:                                         ; preds = %0
  store atomic volatile i64 %desired, i64 addrspace(1)* %object release, align 8
  br label %2

; <label>:1                                       ; preds = %0
  store atomic volatile i64 %desired, i64 addrspace(1)* %object seq_cst, align 8
  br label %2

; <label>:2                                       ; preds = %1, %.thread1, %.thread
  ret void
}

; Function Attrs: nounwind uwtable
define i64 @_Z24pocl_atomic_load__globalPVU3AS1U7_Atomicm12memory_order12memory_scope(i64 addrspace(1)* nocapture readonly %object, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %3 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  %1 = load atomic volatile i64, i64 addrspace(1)* %object monotonic, align 8
  br label %5

.thread1:                                         ; preds = %0
  %2 = load atomic volatile i64, i64 addrspace(1)* %object acquire, align 8
  br label %5

; <label>:3                                       ; preds = %0
  %4 = load atomic volatile i64, i64 addrspace(1)* %object seq_cst, align 8
  br label %5

; <label>:5                                       ; preds = %3, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %4, %3 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z28pocl_atomic_exchange__globalPVU3AS1U7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* %object, i64 %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xchg i64 addrspace(1)* %object, i64 %desired monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xchg i64 addrspace(1)* %object, i64 %desired acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xchg i64 addrspace(1)* %object, i64 %desired release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xchg i64 addrspace(1)* %object, i64 %desired acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xchg i64 addrspace(1)* %object, i64 %desired seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z43pocl_atomic_compare_exchange_strong__globalPVU3AS1U7_AtomicmPmm12memory_orderS3_12memory_scope(i64 addrspace(1)* %object, i64* nocapture %expected, i64 %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = icmp eq i32 %failure, 0
  br i1 %11, label %19, label %12

; <label>:12                                      ; preds = %9
  %13 = icmp eq i32 %failure, 1
  br i1 %13, label %19, label %14

; <label>:14                                      ; preds = %12
  %15 = icmp eq i32 %failure, 2
  br i1 %15, label %19, label %16

; <label>:16                                      ; preds = %14
  %17 = icmp eq i32 %failure, 3
  %18 = select i1 %17, i32 4, i32 5
  br label %19

; <label>:19                                      ; preds = %12, %14, %16, %9
  %20 = phi i32 [ 0, %9 ], [ 2, %12 ], [ %18, %16 ], [ 3, %14 ]
  switch i32 %10, label %28 [
    i32 1, label %21
    i32 2, label %21
    i32 3, label %50
    i32 4, label %23
    i32 5, label %25
  ]

; <label>:21                                      ; preds = %19, %19
  %.off = add nsw i32 %20, -1
  %switch = icmp ult i32 %.off, 2
  %22 = load i64, i64* %expected, align 8
  br i1 %switch, label %39, label %36

; <label>:23                                      ; preds = %19
  %.off1 = add nsw i32 %20, -1
  %switch2 = icmp ult i32 %.off1, 2
  %24 = load i64, i64* %expected, align 8
  br i1 %switch2, label %61, label %58

; <label>:25                                      ; preds = %19
  switch i32 %20, label %72 [
    i32 1, label %76
    i32 2, label %76
    i32 5, label %80
  ]

; <label>:26                                      ; preds = %86, %90, %94, %66, %70, %44, %48, %56, %34
  %.0 = phi i8 [ %35, %34 ], [ %87, %86 ], [ %95, %94 ], [ %91, %90 ], [ %71, %70 ], [ %67, %66 ], [ %57, %56 ], [ %49, %48 ], [ %45, %44 ]
  %27 = icmp ne i8 %.0, 0
  ret i1 %27

; <label>:28                                      ; preds = %19
  %29 = load i64, i64* %expected, align 8
  %30 = cmpxchg volatile i64 addrspace(1)* %object, i64 %29, i64 %desired monotonic monotonic
  %31 = extractvalue { i64, i1 } %30, 1
  br i1 %31, label %34, label %32

; <label>:32                                      ; preds = %28
  %33 = extractvalue { i64, i1 } %30, 0
  store i64 %33, i64* %expected, align 8
  br label %34

; <label>:34                                      ; preds = %32, %28
  %35 = zext i1 %31 to i8
  br label %26

; <label>:36                                      ; preds = %21
  %37 = cmpxchg volatile i64 addrspace(1)* %object, i64 %22, i64 %desired acquire monotonic
  %38 = extractvalue { i64, i1 } %37, 1
  br i1 %38, label %44, label %42

; <label>:39                                      ; preds = %21
  %40 = cmpxchg volatile i64 addrspace(1)* %object, i64 %22, i64 %desired acquire acquire
  %41 = extractvalue { i64, i1 } %40, 1
  br i1 %41, label %48, label %46

; <label>:42                                      ; preds = %36
  %43 = extractvalue { i64, i1 } %37, 0
  store i64 %43, i64* %expected, align 8
  br label %44

; <label>:44                                      ; preds = %42, %36
  %45 = zext i1 %38 to i8
  br label %26

; <label>:46                                      ; preds = %39
  %47 = extractvalue { i64, i1 } %40, 0
  store i64 %47, i64* %expected, align 8
  br label %48

; <label>:48                                      ; preds = %46, %39
  %49 = zext i1 %41 to i8
  br label %26

; <label>:50                                      ; preds = %19
  %51 = load i64, i64* %expected, align 8
  %52 = cmpxchg volatile i64 addrspace(1)* %object, i64 %51, i64 %desired release monotonic
  %53 = extractvalue { i64, i1 } %52, 1
  br i1 %53, label %56, label %54

; <label>:54                                      ; preds = %50
  %55 = extractvalue { i64, i1 } %52, 0
  store i64 %55, i64* %expected, align 8
  br label %56

; <label>:56                                      ; preds = %54, %50
  %57 = zext i1 %53 to i8
  br label %26

; <label>:58                                      ; preds = %23
  %59 = cmpxchg volatile i64 addrspace(1)* %object, i64 %24, i64 %desired acq_rel monotonic
  %60 = extractvalue { i64, i1 } %59, 1
  br i1 %60, label %66, label %64

; <label>:61                                      ; preds = %23
  %62 = cmpxchg volatile i64 addrspace(1)* %object, i64 %24, i64 %desired acq_rel acquire
  %63 = extractvalue { i64, i1 } %62, 1
  br i1 %63, label %70, label %68

; <label>:64                                      ; preds = %58
  %65 = extractvalue { i64, i1 } %59, 0
  store i64 %65, i64* %expected, align 8
  br label %66

; <label>:66                                      ; preds = %64, %58
  %67 = zext i1 %60 to i8
  br label %26

; <label>:68                                      ; preds = %61
  %69 = extractvalue { i64, i1 } %62, 0
  store i64 %69, i64* %expected, align 8
  br label %70

; <label>:70                                      ; preds = %68, %61
  %71 = zext i1 %63 to i8
  br label %26

; <label>:72                                      ; preds = %25
  %73 = load i64, i64* %expected, align 8
  %74 = cmpxchg volatile i64 addrspace(1)* %object, i64 %73, i64 %desired seq_cst monotonic
  %75 = extractvalue { i64, i1 } %74, 1
  br i1 %75, label %86, label %84

; <label>:76                                      ; preds = %25, %25
  %77 = load i64, i64* %expected, align 8
  %78 = cmpxchg volatile i64 addrspace(1)* %object, i64 %77, i64 %desired seq_cst acquire
  %79 = extractvalue { i64, i1 } %78, 1
  br i1 %79, label %90, label %88

; <label>:80                                      ; preds = %25
  %81 = load i64, i64* %expected, align 8
  %82 = cmpxchg volatile i64 addrspace(1)* %object, i64 %81, i64 %desired seq_cst seq_cst
  %83 = extractvalue { i64, i1 } %82, 1
  br i1 %83, label %94, label %92

; <label>:84                                      ; preds = %72
  %85 = extractvalue { i64, i1 } %74, 0
  store i64 %85, i64* %expected, align 8
  br label %86

; <label>:86                                      ; preds = %84, %72
  %87 = zext i1 %75 to i8
  br label %26

; <label>:88                                      ; preds = %76
  %89 = extractvalue { i64, i1 } %78, 0
  store i64 %89, i64* %expected, align 8
  br label %90

; <label>:90                                      ; preds = %88, %76
  %91 = zext i1 %79 to i8
  br label %26

; <label>:92                                      ; preds = %80
  %93 = extractvalue { i64, i1 } %82, 0
  store i64 %93, i64* %expected, align 8
  br label %94

; <label>:94                                      ; preds = %92, %80
  %95 = zext i1 %83 to i8
  br label %26
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z41pocl_atomic_compare_exchange_weak__globalPVU3AS1U7_AtomicmPmm12memory_orderS3_12memory_scope(i64 addrspace(1)* %object, i64* nocapture %expected, i64 %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = icmp eq i32 %failure, 0
  br i1 %11, label %19, label %12

; <label>:12                                      ; preds = %9
  %13 = icmp eq i32 %failure, 1
  br i1 %13, label %19, label %14

; <label>:14                                      ; preds = %12
  %15 = icmp eq i32 %failure, 2
  br i1 %15, label %19, label %16

; <label>:16                                      ; preds = %14
  %17 = icmp eq i32 %failure, 3
  %18 = select i1 %17, i32 4, i32 5
  br label %19

; <label>:19                                      ; preds = %12, %14, %16, %9
  %20 = phi i32 [ 0, %9 ], [ 2, %12 ], [ %18, %16 ], [ 3, %14 ]
  switch i32 %10, label %28 [
    i32 1, label %21
    i32 2, label %21
    i32 3, label %50
    i32 4, label %23
    i32 5, label %25
  ]

; <label>:21                                      ; preds = %19, %19
  %.off = add nsw i32 %20, -1
  %switch = icmp ult i32 %.off, 2
  %22 = load i64, i64* %expected, align 8
  br i1 %switch, label %39, label %36

; <label>:23                                      ; preds = %19
  %.off1 = add nsw i32 %20, -1
  %switch2 = icmp ult i32 %.off1, 2
  %24 = load i64, i64* %expected, align 8
  br i1 %switch2, label %61, label %58

; <label>:25                                      ; preds = %19
  switch i32 %20, label %72 [
    i32 1, label %76
    i32 2, label %76
    i32 5, label %80
  ]

; <label>:26                                      ; preds = %86, %90, %94, %66, %70, %44, %48, %56, %34
  %.0 = phi i8 [ %35, %34 ], [ %87, %86 ], [ %95, %94 ], [ %91, %90 ], [ %71, %70 ], [ %67, %66 ], [ %57, %56 ], [ %49, %48 ], [ %45, %44 ]
  %27 = icmp ne i8 %.0, 0
  ret i1 %27

; <label>:28                                      ; preds = %19
  %29 = load i64, i64* %expected, align 8
  %30 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %29, i64 %desired monotonic monotonic
  %31 = extractvalue { i64, i1 } %30, 1
  br i1 %31, label %34, label %32

; <label>:32                                      ; preds = %28
  %33 = extractvalue { i64, i1 } %30, 0
  store i64 %33, i64* %expected, align 8
  br label %34

; <label>:34                                      ; preds = %32, %28
  %35 = zext i1 %31 to i8
  br label %26

; <label>:36                                      ; preds = %21
  %37 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %22, i64 %desired acquire monotonic
  %38 = extractvalue { i64, i1 } %37, 1
  br i1 %38, label %44, label %42

; <label>:39                                      ; preds = %21
  %40 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %22, i64 %desired acquire acquire
  %41 = extractvalue { i64, i1 } %40, 1
  br i1 %41, label %48, label %46

; <label>:42                                      ; preds = %36
  %43 = extractvalue { i64, i1 } %37, 0
  store i64 %43, i64* %expected, align 8
  br label %44

; <label>:44                                      ; preds = %42, %36
  %45 = zext i1 %38 to i8
  br label %26

; <label>:46                                      ; preds = %39
  %47 = extractvalue { i64, i1 } %40, 0
  store i64 %47, i64* %expected, align 8
  br label %48

; <label>:48                                      ; preds = %46, %39
  %49 = zext i1 %41 to i8
  br label %26

; <label>:50                                      ; preds = %19
  %51 = load i64, i64* %expected, align 8
  %52 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %51, i64 %desired release monotonic
  %53 = extractvalue { i64, i1 } %52, 1
  br i1 %53, label %56, label %54

; <label>:54                                      ; preds = %50
  %55 = extractvalue { i64, i1 } %52, 0
  store i64 %55, i64* %expected, align 8
  br label %56

; <label>:56                                      ; preds = %54, %50
  %57 = zext i1 %53 to i8
  br label %26

; <label>:58                                      ; preds = %23
  %59 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %24, i64 %desired acq_rel monotonic
  %60 = extractvalue { i64, i1 } %59, 1
  br i1 %60, label %66, label %64

; <label>:61                                      ; preds = %23
  %62 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %24, i64 %desired acq_rel acquire
  %63 = extractvalue { i64, i1 } %62, 1
  br i1 %63, label %70, label %68

; <label>:64                                      ; preds = %58
  %65 = extractvalue { i64, i1 } %59, 0
  store i64 %65, i64* %expected, align 8
  br label %66

; <label>:66                                      ; preds = %64, %58
  %67 = zext i1 %60 to i8
  br label %26

; <label>:68                                      ; preds = %61
  %69 = extractvalue { i64, i1 } %62, 0
  store i64 %69, i64* %expected, align 8
  br label %70

; <label>:70                                      ; preds = %68, %61
  %71 = zext i1 %63 to i8
  br label %26

; <label>:72                                      ; preds = %25
  %73 = load i64, i64* %expected, align 8
  %74 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %73, i64 %desired seq_cst monotonic
  %75 = extractvalue { i64, i1 } %74, 1
  br i1 %75, label %86, label %84

; <label>:76                                      ; preds = %25, %25
  %77 = load i64, i64* %expected, align 8
  %78 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %77, i64 %desired seq_cst acquire
  %79 = extractvalue { i64, i1 } %78, 1
  br i1 %79, label %90, label %88

; <label>:80                                      ; preds = %25
  %81 = load i64, i64* %expected, align 8
  %82 = cmpxchg weak volatile i64 addrspace(1)* %object, i64 %81, i64 %desired seq_cst seq_cst
  %83 = extractvalue { i64, i1 } %82, 1
  br i1 %83, label %94, label %92

; <label>:84                                      ; preds = %72
  %85 = extractvalue { i64, i1 } %74, 0
  store i64 %85, i64* %expected, align 8
  br label %86

; <label>:86                                      ; preds = %84, %72
  %87 = zext i1 %75 to i8
  br label %26

; <label>:88                                      ; preds = %76
  %89 = extractvalue { i64, i1 } %78, 0
  store i64 %89, i64* %expected, align 8
  br label %90

; <label>:90                                      ; preds = %88, %76
  %91 = zext i1 %79 to i8
  br label %26

; <label>:92                                      ; preds = %80
  %93 = extractvalue { i64, i1 } %82, 0
  store i64 %93, i64* %expected, align 8
  br label %94

; <label>:94                                      ; preds = %92, %80
  %95 = zext i1 %83 to i8
  br label %26
}

; Function Attrs: nounwind uwtable
define i64 @_Z29pocl_atomic_fetch_add__globalPVU3AS1U7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile add i64 addrspace(1)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile add i64 addrspace(1)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile add i64 addrspace(1)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile add i64 addrspace(1)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile add i64 addrspace(1)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z29pocl_atomic_fetch_sub__globalPVU3AS1U7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile sub i64 addrspace(1)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile sub i64 addrspace(1)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile sub i64 addrspace(1)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile sub i64 addrspace(1)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile sub i64 addrspace(1)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z28pocl_atomic_fetch_or__globalPVU3AS1U7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile or i64 addrspace(1)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile or i64 addrspace(1)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile or i64 addrspace(1)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile or i64 addrspace(1)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile or i64 addrspace(1)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z29pocl_atomic_fetch_xor__globalPVU3AS1U7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xor i64 addrspace(1)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xor i64 addrspace(1)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xor i64 addrspace(1)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xor i64 addrspace(1)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xor i64 addrspace(1)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z29pocl_atomic_fetch_and__globalPVU3AS1U7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile and i64 addrspace(1)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile and i64 addrspace(1)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile and i64 addrspace(1)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile and i64 addrspace(1)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile and i64 addrspace(1)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: noreturn nounwind uwtable
define i64 @_Z29pocl_atomic_fetch_min__globalPVU3AS1U7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* nocapture readnone %object, i64 %operand, i32 %order, i32 %scope) #1 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile umin i64 addrspace(1)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile umin i64 addrspace(1)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile umin i64 addrspace(1)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile umin i64 addrspace(1)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile umin i64 addrspace(1)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: noreturn nounwind uwtable
define i64 @_Z29pocl_atomic_fetch_max__globalPVU3AS1U7_Atomicmm12memory_order12memory_scope(i64 addrspace(1)* nocapture readnone %object, i64 %operand, i32 %order, i32 %scope) #1 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile umax i64 addrspace(1)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile umax i64 addrspace(1)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile umax i64 addrspace(1)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile umax i64 addrspace(1)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile umax i64 addrspace(1)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define void @_Z25pocl_atomic_store__globalPVU3AS1U7_Atomicdd12memory_order12memory_scope(double addrspace(1)* nocapture %object, double %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %5 [
    i32 0, label %.thread
    i32 1, label %.thread
    i32 2, label %.thread1
  ]

.thread1:                                         ; preds = %0
  %1 = bitcast double %desired to i64
  %2 = bitcast double addrspace(1)* %object to i64 addrspace(1)*
  store atomic volatile i64 %1, i64 addrspace(1)* %2 release, align 8
  br label %13

.thread:                                          ; preds = %0, %0
  %3 = bitcast double %desired to i64
  %4 = bitcast double addrspace(1)* %object to i64 addrspace(1)*
  br label %9

; <label>:5                                       ; preds = %0
  %6 = icmp eq i32 %order, 3
  %7 = bitcast double %desired to i64
  %8 = bitcast double addrspace(1)* %object to i64 addrspace(1)*
  br i1 %6, label %9, label %12

; <label>:9                                       ; preds = %5, %.thread
  %10 = phi i64 addrspace(1)* [ %4, %.thread ], [ %8, %5 ]
  %11 = phi i64 [ %3, %.thread ], [ %7, %5 ]
  store atomic volatile i64 %11, i64 addrspace(1)* %10 monotonic, align 8
  br label %13

; <label>:12                                      ; preds = %5
  store atomic volatile i64 %7, i64 addrspace(1)* %8 seq_cst, align 8
  br label %13

; <label>:13                                      ; preds = %12, %.thread1, %9
  ret void
}

; Function Attrs: nounwind uwtable
define double @_Z24pocl_atomic_load__globalPVU3AS1U7_Atomicd12memory_order12memory_scope(double addrspace(1)* nocapture readonly %object, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %4 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread
  ]

.thread1:                                         ; preds = %0
  %1 = bitcast double addrspace(1)* %object to i64 addrspace(1)*
  %2 = load atomic volatile i64, i64 addrspace(1)* %1 acquire, align 8
  br label %12

.thread:                                          ; preds = %0, %0
  %3 = bitcast double addrspace(1)* %object to i64 addrspace(1)*
  br label %7

; <label>:4                                       ; preds = %0
  %5 = icmp eq i32 %order, 3
  %6 = bitcast double addrspace(1)* %object to i64 addrspace(1)*
  br i1 %5, label %7, label %10

; <label>:7                                       ; preds = %4, %.thread
  %8 = phi i64 addrspace(1)* [ %3, %.thread ], [ %6, %4 ]
  %9 = load atomic volatile i64, i64 addrspace(1)* %8 monotonic, align 8
  br label %12

; <label>:10                                      ; preds = %4
  %11 = load atomic volatile i64, i64 addrspace(1)* %6 seq_cst, align 8
  br label %12

; <label>:12                                      ; preds = %10, %.thread1, %7
  %.sroa.0.0 = phi i64 [ %9, %7 ], [ %11, %10 ], [ %2, %.thread1 ]
  %13 = bitcast i64 %.sroa.0.0 to double
  ret double %13
}

; Function Attrs: nounwind uwtable
define double @_Z28pocl_atomic_exchange__globalPVU3AS1U7_Atomicdd12memory_order12memory_scope(double addrspace(1)* %object, double %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %10 [
    i32 0, label %.thread
    i32 1, label %.thread2
    i32 2, label %.thread3
  ]

.thread:                                          ; preds = %0
  %1 = bitcast double %desired to i64
  %2 = bitcast double addrspace(1)* %object to i64 addrspace(1)*
  %3 = atomicrmw volatile xchg i64 addrspace(1)* %2, i64 %1 monotonic
  br label %18

.thread2:                                         ; preds = %0
  %4 = bitcast double %desired to i64
  %5 = bitcast double addrspace(1)* %object to i64 addrspace(1)*
  %6 = atomicrmw volatile xchg i64 addrspace(1)* %5, i64 %4 acquire
  br label %18

.thread3:                                         ; preds = %0
  %7 = bitcast double %desired to i64
  %8 = bitcast double addrspace(1)* %object to i64 addrspace(1)*
  %9 = atomicrmw volatile xchg i64 addrspace(1)* %8, i64 %7 release
  br label %18

; <label>:10                                      ; preds = %0
  %11 = icmp eq i32 %order, 3
  %12 = bitcast double %desired to i64
  %13 = bitcast double addrspace(1)* %object to i64 addrspace(1)*
  br i1 %11, label %14, label %16

; <label>:14                                      ; preds = %10
  %15 = atomicrmw volatile xchg i64 addrspace(1)* %13, i64 %12 acq_rel
  br label %18

; <label>:16                                      ; preds = %10
  %17 = atomicrmw volatile xchg i64 addrspace(1)* %13, i64 %12 seq_cst
  br label %18

; <label>:18                                      ; preds = %16, %14, %.thread3, %.thread2, %.thread
  %.sroa.0.0 = phi i64 [ %3, %.thread ], [ %17, %16 ], [ %15, %14 ], [ %9, %.thread3 ], [ %6, %.thread2 ]
  %19 = bitcast i64 %.sroa.0.0 to double
  ret double %19
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z43pocl_atomic_compare_exchange_strong__globalPVU3AS1U7_AtomicdPdd12memory_orderS3_12memory_scope(double addrspace(1)* %object, double* nocapture %expected, double %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = bitcast double %desired to i64
  %12 = icmp eq i32 %failure, 0
  br i1 %12, label %20, label %13

; <label>:13                                      ; preds = %9
  %14 = icmp eq i32 %failure, 1
  br i1 %14, label %20, label %15

; <label>:15                                      ; preds = %13
  %16 = icmp eq i32 %failure, 2
  br i1 %16, label %20, label %17

; <label>:17                                      ; preds = %15
  %18 = icmp eq i32 %failure, 3
  %19 = select i1 %18, i32 4, i32 5
  br label %20

; <label>:20                                      ; preds = %13, %15, %17, %9
  %21 = phi i32 [ 0, %9 ], [ 2, %13 ], [ %19, %17 ], [ 3, %15 ]
  %22 = bitcast double addrspace(1)* %object to i64 addrspace(1)*
  %23 = bitcast double* %expected to i64*
  switch i32 %10, label %31 [
    i32 1, label %24
    i32 2, label %24
    i32 3, label %53
    i32 4, label %26
    i32 5, label %28
  ]

; <label>:24                                      ; preds = %20, %20
  %.off = add nsw i32 %21, -1
  %switch = icmp ult i32 %.off, 2
  %25 = load i64, i64* %23, align 8
  br i1 %switch, label %42, label %39

; <label>:26                                      ; preds = %20
  %.off1 = add nsw i32 %21, -1
  %switch2 = icmp ult i32 %.off1, 2
  %27 = load i64, i64* %23, align 8
  br i1 %switch2, label %64, label %61

; <label>:28                                      ; preds = %20
  switch i32 %21, label %75 [
    i32 1, label %79
    i32 2, label %79
    i32 5, label %83
  ]

; <label>:29                                      ; preds = %89, %93, %97, %69, %73, %47, %51, %59, %37
  %.0 = phi i8 [ %38, %37 ], [ %90, %89 ], [ %98, %97 ], [ %94, %93 ], [ %74, %73 ], [ %70, %69 ], [ %60, %59 ], [ %52, %51 ], [ %48, %47 ]
  %30 = icmp ne i8 %.0, 0
  ret i1 %30

; <label>:31                                      ; preds = %20
  %32 = load i64, i64* %23, align 8
  %33 = cmpxchg volatile i64 addrspace(1)* %22, i64 %32, i64 %11 monotonic monotonic
  %34 = extractvalue { i64, i1 } %33, 1
  br i1 %34, label %37, label %35

; <label>:35                                      ; preds = %31
  %36 = extractvalue { i64, i1 } %33, 0
  store i64 %36, i64* %23, align 8
  br label %37

; <label>:37                                      ; preds = %35, %31
  %38 = zext i1 %34 to i8
  br label %29

; <label>:39                                      ; preds = %24
  %40 = cmpxchg volatile i64 addrspace(1)* %22, i64 %25, i64 %11 acquire monotonic
  %41 = extractvalue { i64, i1 } %40, 1
  br i1 %41, label %47, label %45

; <label>:42                                      ; preds = %24
  %43 = cmpxchg volatile i64 addrspace(1)* %22, i64 %25, i64 %11 acquire acquire
  %44 = extractvalue { i64, i1 } %43, 1
  br i1 %44, label %51, label %49

; <label>:45                                      ; preds = %39
  %46 = extractvalue { i64, i1 } %40, 0
  store i64 %46, i64* %23, align 8
  br label %47

; <label>:47                                      ; preds = %45, %39
  %48 = zext i1 %41 to i8
  br label %29

; <label>:49                                      ; preds = %42
  %50 = extractvalue { i64, i1 } %43, 0
  store i64 %50, i64* %23, align 8
  br label %51

; <label>:51                                      ; preds = %49, %42
  %52 = zext i1 %44 to i8
  br label %29

; <label>:53                                      ; preds = %20
  %54 = load i64, i64* %23, align 8
  %55 = cmpxchg volatile i64 addrspace(1)* %22, i64 %54, i64 %11 release monotonic
  %56 = extractvalue { i64, i1 } %55, 1
  br i1 %56, label %59, label %57

; <label>:57                                      ; preds = %53
  %58 = extractvalue { i64, i1 } %55, 0
  store i64 %58, i64* %23, align 8
  br label %59

; <label>:59                                      ; preds = %57, %53
  %60 = zext i1 %56 to i8
  br label %29

; <label>:61                                      ; preds = %26
  %62 = cmpxchg volatile i64 addrspace(1)* %22, i64 %27, i64 %11 acq_rel monotonic
  %63 = extractvalue { i64, i1 } %62, 1
  br i1 %63, label %69, label %67

; <label>:64                                      ; preds = %26
  %65 = cmpxchg volatile i64 addrspace(1)* %22, i64 %27, i64 %11 acq_rel acquire
  %66 = extractvalue { i64, i1 } %65, 1
  br i1 %66, label %73, label %71

; <label>:67                                      ; preds = %61
  %68 = extractvalue { i64, i1 } %62, 0
  store i64 %68, i64* %23, align 8
  br label %69

; <label>:69                                      ; preds = %67, %61
  %70 = zext i1 %63 to i8
  br label %29

; <label>:71                                      ; preds = %64
  %72 = extractvalue { i64, i1 } %65, 0
  store i64 %72, i64* %23, align 8
  br label %73

; <label>:73                                      ; preds = %71, %64
  %74 = zext i1 %66 to i8
  br label %29

; <label>:75                                      ; preds = %28
  %76 = load i64, i64* %23, align 8
  %77 = cmpxchg volatile i64 addrspace(1)* %22, i64 %76, i64 %11 seq_cst monotonic
  %78 = extractvalue { i64, i1 } %77, 1
  br i1 %78, label %89, label %87

; <label>:79                                      ; preds = %28, %28
  %80 = load i64, i64* %23, align 8
  %81 = cmpxchg volatile i64 addrspace(1)* %22, i64 %80, i64 %11 seq_cst acquire
  %82 = extractvalue { i64, i1 } %81, 1
  br i1 %82, label %93, label %91

; <label>:83                                      ; preds = %28
  %84 = load i64, i64* %23, align 8
  %85 = cmpxchg volatile i64 addrspace(1)* %22, i64 %84, i64 %11 seq_cst seq_cst
  %86 = extractvalue { i64, i1 } %85, 1
  br i1 %86, label %97, label %95

; <label>:87                                      ; preds = %75
  %88 = extractvalue { i64, i1 } %77, 0
  store i64 %88, i64* %23, align 8
  br label %89

; <label>:89                                      ; preds = %87, %75
  %90 = zext i1 %78 to i8
  br label %29

; <label>:91                                      ; preds = %79
  %92 = extractvalue { i64, i1 } %81, 0
  store i64 %92, i64* %23, align 8
  br label %93

; <label>:93                                      ; preds = %91, %79
  %94 = zext i1 %82 to i8
  br label %29

; <label>:95                                      ; preds = %83
  %96 = extractvalue { i64, i1 } %85, 0
  store i64 %96, i64* %23, align 8
  br label %97

; <label>:97                                      ; preds = %95, %83
  %98 = zext i1 %86 to i8
  br label %29
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z41pocl_atomic_compare_exchange_weak__globalPVU3AS1U7_AtomicdPdd12memory_orderS3_12memory_scope(double addrspace(1)* %object, double* nocapture %expected, double %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = bitcast double %desired to i64
  %12 = icmp eq i32 %failure, 0
  br i1 %12, label %20, label %13

; <label>:13                                      ; preds = %9
  %14 = icmp eq i32 %failure, 1
  br i1 %14, label %20, label %15

; <label>:15                                      ; preds = %13
  %16 = icmp eq i32 %failure, 2
  br i1 %16, label %20, label %17

; <label>:17                                      ; preds = %15
  %18 = icmp eq i32 %failure, 3
  %19 = select i1 %18, i32 4, i32 5
  br label %20

; <label>:20                                      ; preds = %13, %15, %17, %9
  %21 = phi i32 [ 0, %9 ], [ 2, %13 ], [ %19, %17 ], [ 3, %15 ]
  %22 = bitcast double addrspace(1)* %object to i64 addrspace(1)*
  %23 = bitcast double* %expected to i64*
  switch i32 %10, label %31 [
    i32 1, label %24
    i32 2, label %24
    i32 3, label %53
    i32 4, label %26
    i32 5, label %28
  ]

; <label>:24                                      ; preds = %20, %20
  %.off = add nsw i32 %21, -1
  %switch = icmp ult i32 %.off, 2
  %25 = load i64, i64* %23, align 8
  br i1 %switch, label %42, label %39

; <label>:26                                      ; preds = %20
  %.off1 = add nsw i32 %21, -1
  %switch2 = icmp ult i32 %.off1, 2
  %27 = load i64, i64* %23, align 8
  br i1 %switch2, label %64, label %61

; <label>:28                                      ; preds = %20
  switch i32 %21, label %75 [
    i32 1, label %79
    i32 2, label %79
    i32 5, label %83
  ]

; <label>:29                                      ; preds = %89, %93, %97, %69, %73, %47, %51, %59, %37
  %.0 = phi i8 [ %38, %37 ], [ %90, %89 ], [ %98, %97 ], [ %94, %93 ], [ %74, %73 ], [ %70, %69 ], [ %60, %59 ], [ %52, %51 ], [ %48, %47 ]
  %30 = icmp ne i8 %.0, 0
  ret i1 %30

; <label>:31                                      ; preds = %20
  %32 = load i64, i64* %23, align 8
  %33 = cmpxchg weak volatile i64 addrspace(1)* %22, i64 %32, i64 %11 monotonic monotonic
  %34 = extractvalue { i64, i1 } %33, 1
  br i1 %34, label %37, label %35

; <label>:35                                      ; preds = %31
  %36 = extractvalue { i64, i1 } %33, 0
  store i64 %36, i64* %23, align 8
  br label %37

; <label>:37                                      ; preds = %35, %31
  %38 = zext i1 %34 to i8
  br label %29

; <label>:39                                      ; preds = %24
  %40 = cmpxchg weak volatile i64 addrspace(1)* %22, i64 %25, i64 %11 acquire monotonic
  %41 = extractvalue { i64, i1 } %40, 1
  br i1 %41, label %47, label %45

; <label>:42                                      ; preds = %24
  %43 = cmpxchg weak volatile i64 addrspace(1)* %22, i64 %25, i64 %11 acquire acquire
  %44 = extractvalue { i64, i1 } %43, 1
  br i1 %44, label %51, label %49

; <label>:45                                      ; preds = %39
  %46 = extractvalue { i64, i1 } %40, 0
  store i64 %46, i64* %23, align 8
  br label %47

; <label>:47                                      ; preds = %45, %39
  %48 = zext i1 %41 to i8
  br label %29

; <label>:49                                      ; preds = %42
  %50 = extractvalue { i64, i1 } %43, 0
  store i64 %50, i64* %23, align 8
  br label %51

; <label>:51                                      ; preds = %49, %42
  %52 = zext i1 %44 to i8
  br label %29

; <label>:53                                      ; preds = %20
  %54 = load i64, i64* %23, align 8
  %55 = cmpxchg weak volatile i64 addrspace(1)* %22, i64 %54, i64 %11 release monotonic
  %56 = extractvalue { i64, i1 } %55, 1
  br i1 %56, label %59, label %57

; <label>:57                                      ; preds = %53
  %58 = extractvalue { i64, i1 } %55, 0
  store i64 %58, i64* %23, align 8
  br label %59

; <label>:59                                      ; preds = %57, %53
  %60 = zext i1 %56 to i8
  br label %29

; <label>:61                                      ; preds = %26
  %62 = cmpxchg weak volatile i64 addrspace(1)* %22, i64 %27, i64 %11 acq_rel monotonic
  %63 = extractvalue { i64, i1 } %62, 1
  br i1 %63, label %69, label %67

; <label>:64                                      ; preds = %26
  %65 = cmpxchg weak volatile i64 addrspace(1)* %22, i64 %27, i64 %11 acq_rel acquire
  %66 = extractvalue { i64, i1 } %65, 1
  br i1 %66, label %73, label %71

; <label>:67                                      ; preds = %61
  %68 = extractvalue { i64, i1 } %62, 0
  store i64 %68, i64* %23, align 8
  br label %69

; <label>:69                                      ; preds = %67, %61
  %70 = zext i1 %63 to i8
  br label %29

; <label>:71                                      ; preds = %64
  %72 = extractvalue { i64, i1 } %65, 0
  store i64 %72, i64* %23, align 8
  br label %73

; <label>:73                                      ; preds = %71, %64
  %74 = zext i1 %66 to i8
  br label %29

; <label>:75                                      ; preds = %28
  %76 = load i64, i64* %23, align 8
  %77 = cmpxchg weak volatile i64 addrspace(1)* %22, i64 %76, i64 %11 seq_cst monotonic
  %78 = extractvalue { i64, i1 } %77, 1
  br i1 %78, label %89, label %87

; <label>:79                                      ; preds = %28, %28
  %80 = load i64, i64* %23, align 8
  %81 = cmpxchg weak volatile i64 addrspace(1)* %22, i64 %80, i64 %11 seq_cst acquire
  %82 = extractvalue { i64, i1 } %81, 1
  br i1 %82, label %93, label %91

; <label>:83                                      ; preds = %28
  %84 = load i64, i64* %23, align 8
  %85 = cmpxchg weak volatile i64 addrspace(1)* %22, i64 %84, i64 %11 seq_cst seq_cst
  %86 = extractvalue { i64, i1 } %85, 1
  br i1 %86, label %97, label %95

; <label>:87                                      ; preds = %75
  %88 = extractvalue { i64, i1 } %77, 0
  store i64 %88, i64* %23, align 8
  br label %89

; <label>:89                                      ; preds = %87, %75
  %90 = zext i1 %78 to i8
  br label %29

; <label>:91                                      ; preds = %79
  %92 = extractvalue { i64, i1 } %81, 0
  store i64 %92, i64* %23, align 8
  br label %93

; <label>:93                                      ; preds = %91, %79
  %94 = zext i1 %82 to i8
  br label %29

; <label>:95                                      ; preds = %83
  %96 = extractvalue { i64, i1 } %85, 0
  store i64 %96, i64* %23, align 8
  br label %97

; <label>:97                                      ; preds = %95, %83
  %98 = zext i1 %86 to i8
  br label %29
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z36pocl_atomic_flag_test_and_set__localPVU3AS2U7_Atomici12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xchg i32 addrspace(2)* %object, i32 1 monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xchg i32 addrspace(2)* %object, i32 1 acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xchg i32 addrspace(2)* %object, i32 1 release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xchg i32 addrspace(2)* %object, i32 1 acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xchg i32 addrspace(2)* %object, i32 1 seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  %9 = icmp ne i32 %.0, 0
  ret i1 %9
}

; Function Attrs: nounwind uwtable
define void @_Z29pocl_atomic_flag_clear__localPVU3AS2U7_Atomici12memory_order12memory_scope(i32 addrspace(2)* nocapture %object, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %1 [
    i32 0, label %.thread
    i32 1, label %.thread
    i32 2, label %.thread1
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  store atomic volatile i32 0, i32 addrspace(2)* %object monotonic, align 4
  br label %2

.thread1:                                         ; preds = %0
  store atomic volatile i32 0, i32 addrspace(2)* %object release, align 4
  br label %2

; <label>:1                                       ; preds = %0
  store atomic volatile i32 0, i32 addrspace(2)* %object seq_cst, align 4
  br label %2

; <label>:2                                       ; preds = %1, %.thread1, %.thread
  ret void
}

; Function Attrs: nounwind uwtable
define void @_Z24pocl_atomic_store__localPVU3AS2U7_Atomicii12memory_order12memory_scope(i32 addrspace(2)* nocapture %object, i32 %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %1 [
    i32 0, label %.thread
    i32 1, label %.thread
    i32 2, label %.thread1
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  store atomic volatile i32 %desired, i32 addrspace(2)* %object monotonic, align 4
  br label %2

.thread1:                                         ; preds = %0
  store atomic volatile i32 %desired, i32 addrspace(2)* %object release, align 4
  br label %2

; <label>:1                                       ; preds = %0
  store atomic volatile i32 %desired, i32 addrspace(2)* %object seq_cst, align 4
  br label %2

; <label>:2                                       ; preds = %1, %.thread1, %.thread
  ret void
}

; Function Attrs: nounwind uwtable
define i32 @_Z23pocl_atomic_load__localPVU3AS2U7_Atomici12memory_order12memory_scope(i32 addrspace(2)* nocapture readonly %object, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %3 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  %1 = load atomic volatile i32, i32 addrspace(2)* %object monotonic, align 4
  br label %5

.thread1:                                         ; preds = %0
  %2 = load atomic volatile i32, i32 addrspace(2)* %object acquire, align 4
  br label %5

; <label>:3                                       ; preds = %0
  %4 = load atomic volatile i32, i32 addrspace(2)* %object seq_cst, align 4
  br label %5

; <label>:5                                       ; preds = %3, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %4, %3 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z27pocl_atomic_exchange__localPVU3AS2U7_Atomicii12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xchg i32 addrspace(2)* %object, i32 %desired monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xchg i32 addrspace(2)* %object, i32 %desired acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xchg i32 addrspace(2)* %object, i32 %desired release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xchg i32 addrspace(2)* %object, i32 %desired acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xchg i32 addrspace(2)* %object, i32 %desired seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z42pocl_atomic_compare_exchange_strong__localPVU3AS2U7_AtomiciPii12memory_orderS3_12memory_scope(i32 addrspace(2)* %object, i32* nocapture %expected, i32 %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = icmp eq i32 %failure, 0
  br i1 %11, label %19, label %12

; <label>:12                                      ; preds = %9
  %13 = icmp eq i32 %failure, 1
  br i1 %13, label %19, label %14

; <label>:14                                      ; preds = %12
  %15 = icmp eq i32 %failure, 2
  br i1 %15, label %19, label %16

; <label>:16                                      ; preds = %14
  %17 = icmp eq i32 %failure, 3
  %18 = select i1 %17, i32 4, i32 5
  br label %19

; <label>:19                                      ; preds = %12, %14, %16, %9
  %20 = phi i32 [ 0, %9 ], [ 2, %12 ], [ %18, %16 ], [ 3, %14 ]
  switch i32 %10, label %28 [
    i32 1, label %21
    i32 2, label %21
    i32 3, label %50
    i32 4, label %23
    i32 5, label %25
  ]

; <label>:21                                      ; preds = %19, %19
  %.off = add nsw i32 %20, -1
  %switch = icmp ult i32 %.off, 2
  %22 = load i32, i32* %expected, align 4
  br i1 %switch, label %39, label %36

; <label>:23                                      ; preds = %19
  %.off1 = add nsw i32 %20, -1
  %switch2 = icmp ult i32 %.off1, 2
  %24 = load i32, i32* %expected, align 4
  br i1 %switch2, label %61, label %58

; <label>:25                                      ; preds = %19
  switch i32 %20, label %72 [
    i32 1, label %76
    i32 2, label %76
    i32 5, label %80
  ]

; <label>:26                                      ; preds = %86, %90, %94, %66, %70, %44, %48, %56, %34
  %.0 = phi i8 [ %35, %34 ], [ %87, %86 ], [ %95, %94 ], [ %91, %90 ], [ %71, %70 ], [ %67, %66 ], [ %57, %56 ], [ %49, %48 ], [ %45, %44 ]
  %27 = icmp ne i8 %.0, 0
  ret i1 %27

; <label>:28                                      ; preds = %19
  %29 = load i32, i32* %expected, align 4
  %30 = cmpxchg volatile i32 addrspace(2)* %object, i32 %29, i32 %desired monotonic monotonic
  %31 = extractvalue { i32, i1 } %30, 1
  br i1 %31, label %34, label %32

; <label>:32                                      ; preds = %28
  %33 = extractvalue { i32, i1 } %30, 0
  store i32 %33, i32* %expected, align 4
  br label %34

; <label>:34                                      ; preds = %32, %28
  %35 = zext i1 %31 to i8
  br label %26

; <label>:36                                      ; preds = %21
  %37 = cmpxchg volatile i32 addrspace(2)* %object, i32 %22, i32 %desired acquire monotonic
  %38 = extractvalue { i32, i1 } %37, 1
  br i1 %38, label %44, label %42

; <label>:39                                      ; preds = %21
  %40 = cmpxchg volatile i32 addrspace(2)* %object, i32 %22, i32 %desired acquire acquire
  %41 = extractvalue { i32, i1 } %40, 1
  br i1 %41, label %48, label %46

; <label>:42                                      ; preds = %36
  %43 = extractvalue { i32, i1 } %37, 0
  store i32 %43, i32* %expected, align 4
  br label %44

; <label>:44                                      ; preds = %42, %36
  %45 = zext i1 %38 to i8
  br label %26

; <label>:46                                      ; preds = %39
  %47 = extractvalue { i32, i1 } %40, 0
  store i32 %47, i32* %expected, align 4
  br label %48

; <label>:48                                      ; preds = %46, %39
  %49 = zext i1 %41 to i8
  br label %26

; <label>:50                                      ; preds = %19
  %51 = load i32, i32* %expected, align 4
  %52 = cmpxchg volatile i32 addrspace(2)* %object, i32 %51, i32 %desired release monotonic
  %53 = extractvalue { i32, i1 } %52, 1
  br i1 %53, label %56, label %54

; <label>:54                                      ; preds = %50
  %55 = extractvalue { i32, i1 } %52, 0
  store i32 %55, i32* %expected, align 4
  br label %56

; <label>:56                                      ; preds = %54, %50
  %57 = zext i1 %53 to i8
  br label %26

; <label>:58                                      ; preds = %23
  %59 = cmpxchg volatile i32 addrspace(2)* %object, i32 %24, i32 %desired acq_rel monotonic
  %60 = extractvalue { i32, i1 } %59, 1
  br i1 %60, label %66, label %64

; <label>:61                                      ; preds = %23
  %62 = cmpxchg volatile i32 addrspace(2)* %object, i32 %24, i32 %desired acq_rel acquire
  %63 = extractvalue { i32, i1 } %62, 1
  br i1 %63, label %70, label %68

; <label>:64                                      ; preds = %58
  %65 = extractvalue { i32, i1 } %59, 0
  store i32 %65, i32* %expected, align 4
  br label %66

; <label>:66                                      ; preds = %64, %58
  %67 = zext i1 %60 to i8
  br label %26

; <label>:68                                      ; preds = %61
  %69 = extractvalue { i32, i1 } %62, 0
  store i32 %69, i32* %expected, align 4
  br label %70

; <label>:70                                      ; preds = %68, %61
  %71 = zext i1 %63 to i8
  br label %26

; <label>:72                                      ; preds = %25
  %73 = load i32, i32* %expected, align 4
  %74 = cmpxchg volatile i32 addrspace(2)* %object, i32 %73, i32 %desired seq_cst monotonic
  %75 = extractvalue { i32, i1 } %74, 1
  br i1 %75, label %86, label %84

; <label>:76                                      ; preds = %25, %25
  %77 = load i32, i32* %expected, align 4
  %78 = cmpxchg volatile i32 addrspace(2)* %object, i32 %77, i32 %desired seq_cst acquire
  %79 = extractvalue { i32, i1 } %78, 1
  br i1 %79, label %90, label %88

; <label>:80                                      ; preds = %25
  %81 = load i32, i32* %expected, align 4
  %82 = cmpxchg volatile i32 addrspace(2)* %object, i32 %81, i32 %desired seq_cst seq_cst
  %83 = extractvalue { i32, i1 } %82, 1
  br i1 %83, label %94, label %92

; <label>:84                                      ; preds = %72
  %85 = extractvalue { i32, i1 } %74, 0
  store i32 %85, i32* %expected, align 4
  br label %86

; <label>:86                                      ; preds = %84, %72
  %87 = zext i1 %75 to i8
  br label %26

; <label>:88                                      ; preds = %76
  %89 = extractvalue { i32, i1 } %78, 0
  store i32 %89, i32* %expected, align 4
  br label %90

; <label>:90                                      ; preds = %88, %76
  %91 = zext i1 %79 to i8
  br label %26

; <label>:92                                      ; preds = %80
  %93 = extractvalue { i32, i1 } %82, 0
  store i32 %93, i32* %expected, align 4
  br label %94

; <label>:94                                      ; preds = %92, %80
  %95 = zext i1 %83 to i8
  br label %26
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z40pocl_atomic_compare_exchange_weak__localPVU3AS2U7_AtomiciPii12memory_orderS3_12memory_scope(i32 addrspace(2)* %object, i32* nocapture %expected, i32 %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = icmp eq i32 %failure, 0
  br i1 %11, label %19, label %12

; <label>:12                                      ; preds = %9
  %13 = icmp eq i32 %failure, 1
  br i1 %13, label %19, label %14

; <label>:14                                      ; preds = %12
  %15 = icmp eq i32 %failure, 2
  br i1 %15, label %19, label %16

; <label>:16                                      ; preds = %14
  %17 = icmp eq i32 %failure, 3
  %18 = select i1 %17, i32 4, i32 5
  br label %19

; <label>:19                                      ; preds = %12, %14, %16, %9
  %20 = phi i32 [ 0, %9 ], [ 2, %12 ], [ %18, %16 ], [ 3, %14 ]
  switch i32 %10, label %28 [
    i32 1, label %21
    i32 2, label %21
    i32 3, label %50
    i32 4, label %23
    i32 5, label %25
  ]

; <label>:21                                      ; preds = %19, %19
  %.off = add nsw i32 %20, -1
  %switch = icmp ult i32 %.off, 2
  %22 = load i32, i32* %expected, align 4
  br i1 %switch, label %39, label %36

; <label>:23                                      ; preds = %19
  %.off1 = add nsw i32 %20, -1
  %switch2 = icmp ult i32 %.off1, 2
  %24 = load i32, i32* %expected, align 4
  br i1 %switch2, label %61, label %58

; <label>:25                                      ; preds = %19
  switch i32 %20, label %72 [
    i32 1, label %76
    i32 2, label %76
    i32 5, label %80
  ]

; <label>:26                                      ; preds = %86, %90, %94, %66, %70, %44, %48, %56, %34
  %.0 = phi i8 [ %35, %34 ], [ %87, %86 ], [ %95, %94 ], [ %91, %90 ], [ %71, %70 ], [ %67, %66 ], [ %57, %56 ], [ %49, %48 ], [ %45, %44 ]
  %27 = icmp ne i8 %.0, 0
  ret i1 %27

; <label>:28                                      ; preds = %19
  %29 = load i32, i32* %expected, align 4
  %30 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %29, i32 %desired monotonic monotonic
  %31 = extractvalue { i32, i1 } %30, 1
  br i1 %31, label %34, label %32

; <label>:32                                      ; preds = %28
  %33 = extractvalue { i32, i1 } %30, 0
  store i32 %33, i32* %expected, align 4
  br label %34

; <label>:34                                      ; preds = %32, %28
  %35 = zext i1 %31 to i8
  br label %26

; <label>:36                                      ; preds = %21
  %37 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %22, i32 %desired acquire monotonic
  %38 = extractvalue { i32, i1 } %37, 1
  br i1 %38, label %44, label %42

; <label>:39                                      ; preds = %21
  %40 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %22, i32 %desired acquire acquire
  %41 = extractvalue { i32, i1 } %40, 1
  br i1 %41, label %48, label %46

; <label>:42                                      ; preds = %36
  %43 = extractvalue { i32, i1 } %37, 0
  store i32 %43, i32* %expected, align 4
  br label %44

; <label>:44                                      ; preds = %42, %36
  %45 = zext i1 %38 to i8
  br label %26

; <label>:46                                      ; preds = %39
  %47 = extractvalue { i32, i1 } %40, 0
  store i32 %47, i32* %expected, align 4
  br label %48

; <label>:48                                      ; preds = %46, %39
  %49 = zext i1 %41 to i8
  br label %26

; <label>:50                                      ; preds = %19
  %51 = load i32, i32* %expected, align 4
  %52 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %51, i32 %desired release monotonic
  %53 = extractvalue { i32, i1 } %52, 1
  br i1 %53, label %56, label %54

; <label>:54                                      ; preds = %50
  %55 = extractvalue { i32, i1 } %52, 0
  store i32 %55, i32* %expected, align 4
  br label %56

; <label>:56                                      ; preds = %54, %50
  %57 = zext i1 %53 to i8
  br label %26

; <label>:58                                      ; preds = %23
  %59 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %24, i32 %desired acq_rel monotonic
  %60 = extractvalue { i32, i1 } %59, 1
  br i1 %60, label %66, label %64

; <label>:61                                      ; preds = %23
  %62 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %24, i32 %desired acq_rel acquire
  %63 = extractvalue { i32, i1 } %62, 1
  br i1 %63, label %70, label %68

; <label>:64                                      ; preds = %58
  %65 = extractvalue { i32, i1 } %59, 0
  store i32 %65, i32* %expected, align 4
  br label %66

; <label>:66                                      ; preds = %64, %58
  %67 = zext i1 %60 to i8
  br label %26

; <label>:68                                      ; preds = %61
  %69 = extractvalue { i32, i1 } %62, 0
  store i32 %69, i32* %expected, align 4
  br label %70

; <label>:70                                      ; preds = %68, %61
  %71 = zext i1 %63 to i8
  br label %26

; <label>:72                                      ; preds = %25
  %73 = load i32, i32* %expected, align 4
  %74 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %73, i32 %desired seq_cst monotonic
  %75 = extractvalue { i32, i1 } %74, 1
  br i1 %75, label %86, label %84

; <label>:76                                      ; preds = %25, %25
  %77 = load i32, i32* %expected, align 4
  %78 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %77, i32 %desired seq_cst acquire
  %79 = extractvalue { i32, i1 } %78, 1
  br i1 %79, label %90, label %88

; <label>:80                                      ; preds = %25
  %81 = load i32, i32* %expected, align 4
  %82 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %81, i32 %desired seq_cst seq_cst
  %83 = extractvalue { i32, i1 } %82, 1
  br i1 %83, label %94, label %92

; <label>:84                                      ; preds = %72
  %85 = extractvalue { i32, i1 } %74, 0
  store i32 %85, i32* %expected, align 4
  br label %86

; <label>:86                                      ; preds = %84, %72
  %87 = zext i1 %75 to i8
  br label %26

; <label>:88                                      ; preds = %76
  %89 = extractvalue { i32, i1 } %78, 0
  store i32 %89, i32* %expected, align 4
  br label %90

; <label>:90                                      ; preds = %88, %76
  %91 = zext i1 %79 to i8
  br label %26

; <label>:92                                      ; preds = %80
  %93 = extractvalue { i32, i1 } %82, 0
  store i32 %93, i32* %expected, align 4
  br label %94

; <label>:94                                      ; preds = %92, %80
  %95 = zext i1 %83 to i8
  br label %26
}

; Function Attrs: nounwind uwtable
define i32 @_Z28pocl_atomic_fetch_add__localPVU3AS2U7_Atomicii12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile add i32 addrspace(2)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile add i32 addrspace(2)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile add i32 addrspace(2)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile add i32 addrspace(2)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile add i32 addrspace(2)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z28pocl_atomic_fetch_sub__localPVU3AS2U7_Atomicii12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile sub i32 addrspace(2)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile sub i32 addrspace(2)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile sub i32 addrspace(2)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile sub i32 addrspace(2)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile sub i32 addrspace(2)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z27pocl_atomic_fetch_or__localPVU3AS2U7_Atomicii12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile or i32 addrspace(2)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile or i32 addrspace(2)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile or i32 addrspace(2)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile or i32 addrspace(2)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile or i32 addrspace(2)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z28pocl_atomic_fetch_xor__localPVU3AS2U7_Atomicii12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xor i32 addrspace(2)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xor i32 addrspace(2)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xor i32 addrspace(2)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xor i32 addrspace(2)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xor i32 addrspace(2)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z28pocl_atomic_fetch_and__localPVU3AS2U7_Atomicii12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile and i32 addrspace(2)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile and i32 addrspace(2)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile and i32 addrspace(2)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile and i32 addrspace(2)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile and i32 addrspace(2)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z28pocl_atomic_fetch_min__localPVU3AS2U7_Atomicii12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile min i32 addrspace(2)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile min i32 addrspace(2)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile min i32 addrspace(2)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile min i32 addrspace(2)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile min i32 addrspace(2)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0

}

; Function Attrs: nounwind uwtable
define i32 @_Z28pocl_atomic_fetch_max__localPVU3AS2U7_Atomicii12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile max i32 addrspace(2)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile max i32 addrspace(2)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile max i32 addrspace(2)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile max i32 addrspace(2)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile max i32 addrspace(2)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define void @_Z24pocl_atomic_store__localPVU3AS2U7_Atomicjj12memory_order12memory_scope(i32 addrspace(2)* nocapture %object, i32 %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %1 [
    i32 0, label %.thread
    i32 1, label %.thread
    i32 2, label %.thread1
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  store atomic volatile i32 %desired, i32 addrspace(2)* %object monotonic, align 4
  br label %2

.thread1:                                         ; preds = %0
  store atomic volatile i32 %desired, i32 addrspace(2)* %object release, align 4
  br label %2

; <label>:1                                       ; preds = %0
  store atomic volatile i32 %desired, i32 addrspace(2)* %object seq_cst, align 4
  br label %2

; <label>:2                                       ; preds = %1, %.thread1, %.thread
  ret void
}

; Function Attrs: nounwind uwtable
define i32 @_Z23pocl_atomic_load__localPVU3AS2U7_Atomicj12memory_order12memory_scope(i32 addrspace(2)* nocapture readonly %object, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %3 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  %1 = load atomic volatile i32, i32 addrspace(2)* %object monotonic, align 4
  br label %5

.thread1:                                         ; preds = %0
  %2 = load atomic volatile i32, i32 addrspace(2)* %object acquire, align 4
  br label %5

; <label>:3                                       ; preds = %0
  %4 = load atomic volatile i32, i32 addrspace(2)* %object seq_cst, align 4
  br label %5

; <label>:5                                       ; preds = %3, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %4, %3 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z27pocl_atomic_exchange__localPVU3AS2U7_Atomicjj12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xchg i32 addrspace(2)* %object, i32 %desired monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xchg i32 addrspace(2)* %object, i32 %desired acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xchg i32 addrspace(2)* %object, i32 %desired release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xchg i32 addrspace(2)* %object, i32 %desired acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xchg i32 addrspace(2)* %object, i32 %desired seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z42pocl_atomic_compare_exchange_strong__localPVU3AS2U7_AtomicjPjj12memory_orderS3_12memory_scope(i32 addrspace(2)* %object, i32* nocapture %expected, i32 %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = icmp eq i32 %failure, 0
  br i1 %11, label %19, label %12

; <label>:12                                      ; preds = %9
  %13 = icmp eq i32 %failure, 1
  br i1 %13, label %19, label %14

; <label>:14                                      ; preds = %12
  %15 = icmp eq i32 %failure, 2
  br i1 %15, label %19, label %16

; <label>:16                                      ; preds = %14
  %17 = icmp eq i32 %failure, 3
  %18 = select i1 %17, i32 4, i32 5
  br label %19

; <label>:19                                      ; preds = %12, %14, %16, %9
  %20 = phi i32 [ 0, %9 ], [ 2, %12 ], [ %18, %16 ], [ 3, %14 ]
  switch i32 %10, label %28 [
    i32 1, label %21
    i32 2, label %21
    i32 3, label %50
    i32 4, label %23
    i32 5, label %25
  ]

; <label>:21                                      ; preds = %19, %19
  %.off = add nsw i32 %20, -1
  %switch = icmp ult i32 %.off, 2
  %22 = load i32, i32* %expected, align 4
  br i1 %switch, label %39, label %36

; <label>:23                                      ; preds = %19
  %.off1 = add nsw i32 %20, -1
  %switch2 = icmp ult i32 %.off1, 2
  %24 = load i32, i32* %expected, align 4
  br i1 %switch2, label %61, label %58

; <label>:25                                      ; preds = %19
  switch i32 %20, label %72 [
    i32 1, label %76
    i32 2, label %76
    i32 5, label %80
  ]

; <label>:26                                      ; preds = %86, %90, %94, %66, %70, %44, %48, %56, %34
  %.0 = phi i8 [ %35, %34 ], [ %87, %86 ], [ %95, %94 ], [ %91, %90 ], [ %71, %70 ], [ %67, %66 ], [ %57, %56 ], [ %49, %48 ], [ %45, %44 ]
  %27 = icmp ne i8 %.0, 0
  ret i1 %27

; <label>:28                                      ; preds = %19
  %29 = load i32, i32* %expected, align 4
  %30 = cmpxchg volatile i32 addrspace(2)* %object, i32 %29, i32 %desired monotonic monotonic
  %31 = extractvalue { i32, i1 } %30, 1
  br i1 %31, label %34, label %32

; <label>:32                                      ; preds = %28
  %33 = extractvalue { i32, i1 } %30, 0
  store i32 %33, i32* %expected, align 4
  br label %34

; <label>:34                                      ; preds = %32, %28
  %35 = zext i1 %31 to i8
  br label %26

; <label>:36                                      ; preds = %21
  %37 = cmpxchg volatile i32 addrspace(2)* %object, i32 %22, i32 %desired acquire monotonic
  %38 = extractvalue { i32, i1 } %37, 1
  br i1 %38, label %44, label %42

; <label>:39                                      ; preds = %21
  %40 = cmpxchg volatile i32 addrspace(2)* %object, i32 %22, i32 %desired acquire acquire
  %41 = extractvalue { i32, i1 } %40, 1
  br i1 %41, label %48, label %46

; <label>:42                                      ; preds = %36
  %43 = extractvalue { i32, i1 } %37, 0
  store i32 %43, i32* %expected, align 4
  br label %44

; <label>:44                                      ; preds = %42, %36
  %45 = zext i1 %38 to i8
  br label %26

; <label>:46                                      ; preds = %39
  %47 = extractvalue { i32, i1 } %40, 0
  store i32 %47, i32* %expected, align 4
  br label %48

; <label>:48                                      ; preds = %46, %39
  %49 = zext i1 %41 to i8
  br label %26

; <label>:50                                      ; preds = %19
  %51 = load i32, i32* %expected, align 4
  %52 = cmpxchg volatile i32 addrspace(2)* %object, i32 %51, i32 %desired release monotonic
  %53 = extractvalue { i32, i1 } %52, 1
  br i1 %53, label %56, label %54

; <label>:54                                      ; preds = %50
  %55 = extractvalue { i32, i1 } %52, 0
  store i32 %55, i32* %expected, align 4
  br label %56

; <label>:56                                      ; preds = %54, %50
  %57 = zext i1 %53 to i8
  br label %26

; <label>:58                                      ; preds = %23
  %59 = cmpxchg volatile i32 addrspace(2)* %object, i32 %24, i32 %desired acq_rel monotonic
  %60 = extractvalue { i32, i1 } %59, 1
  br i1 %60, label %66, label %64

; <label>:61                                      ; preds = %23
  %62 = cmpxchg volatile i32 addrspace(2)* %object, i32 %24, i32 %desired acq_rel acquire
  %63 = extractvalue { i32, i1 } %62, 1
  br i1 %63, label %70, label %68

; <label>:64                                      ; preds = %58
  %65 = extractvalue { i32, i1 } %59, 0
  store i32 %65, i32* %expected, align 4
  br label %66

; <label>:66                                      ; preds = %64, %58
  %67 = zext i1 %60 to i8
  br label %26

; <label>:68                                      ; preds = %61
  %69 = extractvalue { i32, i1 } %62, 0
  store i32 %69, i32* %expected, align 4
  br label %70

; <label>:70                                      ; preds = %68, %61
  %71 = zext i1 %63 to i8
  br label %26

; <label>:72                                      ; preds = %25
  %73 = load i32, i32* %expected, align 4
  %74 = cmpxchg volatile i32 addrspace(2)* %object, i32 %73, i32 %desired seq_cst monotonic
  %75 = extractvalue { i32, i1 } %74, 1
  br i1 %75, label %86, label %84

; <label>:76                                      ; preds = %25, %25
  %77 = load i32, i32* %expected, align 4
  %78 = cmpxchg volatile i32 addrspace(2)* %object, i32 %77, i32 %desired seq_cst acquire
  %79 = extractvalue { i32, i1 } %78, 1
  br i1 %79, label %90, label %88

; <label>:80                                      ; preds = %25
  %81 = load i32, i32* %expected, align 4
  %82 = cmpxchg volatile i32 addrspace(2)* %object, i32 %81, i32 %desired seq_cst seq_cst
  %83 = extractvalue { i32, i1 } %82, 1
  br i1 %83, label %94, label %92

; <label>:84                                      ; preds = %72
  %85 = extractvalue { i32, i1 } %74, 0
  store i32 %85, i32* %expected, align 4
  br label %86

; <label>:86                                      ; preds = %84, %72
  %87 = zext i1 %75 to i8
  br label %26

; <label>:88                                      ; preds = %76
  %89 = extractvalue { i32, i1 } %78, 0
  store i32 %89, i32* %expected, align 4
  br label %90

; <label>:90                                      ; preds = %88, %76
  %91 = zext i1 %79 to i8
  br label %26

; <label>:92                                      ; preds = %80
  %93 = extractvalue { i32, i1 } %82, 0
  store i32 %93, i32* %expected, align 4
  br label %94

; <label>:94                                      ; preds = %92, %80
  %95 = zext i1 %83 to i8
  br label %26
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z40pocl_atomic_compare_exchange_weak__localPVU3AS2U7_AtomicjPjj12memory_orderS3_12memory_scope(i32 addrspace(2)* %object, i32* nocapture %expected, i32 %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = icmp eq i32 %failure, 0
  br i1 %11, label %19, label %12

; <label>:12                                      ; preds = %9
  %13 = icmp eq i32 %failure, 1
  br i1 %13, label %19, label %14

; <label>:14                                      ; preds = %12
  %15 = icmp eq i32 %failure, 2
  br i1 %15, label %19, label %16

; <label>:16                                      ; preds = %14
  %17 = icmp eq i32 %failure, 3
  %18 = select i1 %17, i32 4, i32 5
  br label %19

; <label>:19                                      ; preds = %12, %14, %16, %9
  %20 = phi i32 [ 0, %9 ], [ 2, %12 ], [ %18, %16 ], [ 3, %14 ]
  switch i32 %10, label %28 [
    i32 1, label %21
    i32 2, label %21
    i32 3, label %50
    i32 4, label %23
    i32 5, label %25
  ]

; <label>:21                                      ; preds = %19, %19
  %.off = add nsw i32 %20, -1
  %switch = icmp ult i32 %.off, 2
  %22 = load i32, i32* %expected, align 4
  br i1 %switch, label %39, label %36

; <label>:23                                      ; preds = %19
  %.off1 = add nsw i32 %20, -1
  %switch2 = icmp ult i32 %.off1, 2
  %24 = load i32, i32* %expected, align 4
  br i1 %switch2, label %61, label %58

; <label>:25                                      ; preds = %19
  switch i32 %20, label %72 [
    i32 1, label %76
    i32 2, label %76
    i32 5, label %80
  ]

; <label>:26                                      ; preds = %86, %90, %94, %66, %70, %44, %48, %56, %34
  %.0 = phi i8 [ %35, %34 ], [ %87, %86 ], [ %95, %94 ], [ %91, %90 ], [ %71, %70 ], [ %67, %66 ], [ %57, %56 ], [ %49, %48 ], [ %45, %44 ]
  %27 = icmp ne i8 %.0, 0
  ret i1 %27

; <label>:28                                      ; preds = %19
  %29 = load i32, i32* %expected, align 4
  %30 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %29, i32 %desired monotonic monotonic
  %31 = extractvalue { i32, i1 } %30, 1
  br i1 %31, label %34, label %32

; <label>:32                                      ; preds = %28
  %33 = extractvalue { i32, i1 } %30, 0
  store i32 %33, i32* %expected, align 4
  br label %34

; <label>:34                                      ; preds = %32, %28
  %35 = zext i1 %31 to i8
  br label %26

; <label>:36                                      ; preds = %21
  %37 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %22, i32 %desired acquire monotonic
  %38 = extractvalue { i32, i1 } %37, 1
  br i1 %38, label %44, label %42

; <label>:39                                      ; preds = %21
  %40 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %22, i32 %desired acquire acquire
  %41 = extractvalue { i32, i1 } %40, 1
  br i1 %41, label %48, label %46

; <label>:42                                      ; preds = %36
  %43 = extractvalue { i32, i1 } %37, 0
  store i32 %43, i32* %expected, align 4
  br label %44

; <label>:44                                      ; preds = %42, %36
  %45 = zext i1 %38 to i8
  br label %26

; <label>:46                                      ; preds = %39
  %47 = extractvalue { i32, i1 } %40, 0
  store i32 %47, i32* %expected, align 4
  br label %48

; <label>:48                                      ; preds = %46, %39
  %49 = zext i1 %41 to i8
  br label %26

; <label>:50                                      ; preds = %19
  %51 = load i32, i32* %expected, align 4
  %52 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %51, i32 %desired release monotonic
  %53 = extractvalue { i32, i1 } %52, 1
  br i1 %53, label %56, label %54

; <label>:54                                      ; preds = %50
  %55 = extractvalue { i32, i1 } %52, 0
  store i32 %55, i32* %expected, align 4
  br label %56

; <label>:56                                      ; preds = %54, %50
  %57 = zext i1 %53 to i8
  br label %26

; <label>:58                                      ; preds = %23
  %59 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %24, i32 %desired acq_rel monotonic
  %60 = extractvalue { i32, i1 } %59, 1
  br i1 %60, label %66, label %64

; <label>:61                                      ; preds = %23
  %62 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %24, i32 %desired acq_rel acquire
  %63 = extractvalue { i32, i1 } %62, 1
  br i1 %63, label %70, label %68

; <label>:64                                      ; preds = %58
  %65 = extractvalue { i32, i1 } %59, 0
  store i32 %65, i32* %expected, align 4
  br label %66

; <label>:66                                      ; preds = %64, %58
  %67 = zext i1 %60 to i8
  br label %26

; <label>:68                                      ; preds = %61
  %69 = extractvalue { i32, i1 } %62, 0
  store i32 %69, i32* %expected, align 4
  br label %70

; <label>:70                                      ; preds = %68, %61
  %71 = zext i1 %63 to i8
  br label %26

; <label>:72                                      ; preds = %25
  %73 = load i32, i32* %expected, align 4
  %74 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %73, i32 %desired seq_cst monotonic
  %75 = extractvalue { i32, i1 } %74, 1
  br i1 %75, label %86, label %84

; <label>:76                                      ; preds = %25, %25
  %77 = load i32, i32* %expected, align 4
  %78 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %77, i32 %desired seq_cst acquire
  %79 = extractvalue { i32, i1 } %78, 1
  br i1 %79, label %90, label %88

; <label>:80                                      ; preds = %25
  %81 = load i32, i32* %expected, align 4
  %82 = cmpxchg weak volatile i32 addrspace(2)* %object, i32 %81, i32 %desired seq_cst seq_cst
  %83 = extractvalue { i32, i1 } %82, 1
  br i1 %83, label %94, label %92

; <label>:84                                      ; preds = %72
  %85 = extractvalue { i32, i1 } %74, 0
  store i32 %85, i32* %expected, align 4
  br label %86

; <label>:86                                      ; preds = %84, %72
  %87 = zext i1 %75 to i8
  br label %26

; <label>:88                                      ; preds = %76
  %89 = extractvalue { i32, i1 } %78, 0
  store i32 %89, i32* %expected, align 4
  br label %90

; <label>:90                                      ; preds = %88, %76
  %91 = zext i1 %79 to i8
  br label %26

; <label>:92                                      ; preds = %80
  %93 = extractvalue { i32, i1 } %82, 0
  store i32 %93, i32* %expected, align 4
  br label %94

; <label>:94                                      ; preds = %92, %80
  %95 = zext i1 %83 to i8
  br label %26
}

; Function Attrs: nounwind uwtable
define i32 @_Z28pocl_atomic_fetch_add__localPVU3AS2U7_Atomicjj12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile add i32 addrspace(2)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile add i32 addrspace(2)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile add i32 addrspace(2)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile add i32 addrspace(2)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile add i32 addrspace(2)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z28pocl_atomic_fetch_sub__localPVU3AS2U7_Atomicjj12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile sub i32 addrspace(2)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile sub i32 addrspace(2)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile sub i32 addrspace(2)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile sub i32 addrspace(2)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile sub i32 addrspace(2)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z27pocl_atomic_fetch_or__localPVU3AS2U7_Atomicjj12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile or i32 addrspace(2)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile or i32 addrspace(2)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile or i32 addrspace(2)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile or i32 addrspace(2)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile or i32 addrspace(2)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z28pocl_atomic_fetch_xor__localPVU3AS2U7_Atomicjj12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xor i32 addrspace(2)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xor i32 addrspace(2)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xor i32 addrspace(2)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xor i32 addrspace(2)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xor i32 addrspace(2)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z28pocl_atomic_fetch_and__localPVU3AS2U7_Atomicjj12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile and i32 addrspace(2)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile and i32 addrspace(2)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile and i32 addrspace(2)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile and i32 addrspace(2)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile and i32 addrspace(2)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z28pocl_atomic_fetch_min__localPVU3AS2U7_Atomicjj12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile umin i32 addrspace(2)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile umin i32 addrspace(2)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile umin i32 addrspace(2)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile umin i32 addrspace(2)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile umin i32 addrspace(2)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define i32 @_Z28pocl_atomic_fetch_max__localPVU3AS2U7_Atomicjj12memory_order12memory_scope(i32 addrspace(2)* %object, i32 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile umax i32 addrspace(2)* %object, i32 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile umax i32 addrspace(2)* %object, i32 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile umax i32 addrspace(2)* %object, i32 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile umax i32 addrspace(2)* %object, i32 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile umax i32 addrspace(2)* %object, i32 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i32 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i32 %.0
}

; Function Attrs: nounwind uwtable
define void @_Z24pocl_atomic_store__localPVU3AS2U7_Atomicff12memory_order12memory_scope(float addrspace(2)* nocapture %object, float %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %5 [
    i32 0, label %.thread
    i32 1, label %.thread
    i32 2, label %.thread1
  ]

.thread1:                                         ; preds = %0
  %1 = bitcast float %desired to i32
  %2 = bitcast float addrspace(2)* %object to i32 addrspace(2)*
  store atomic volatile i32 %1, i32 addrspace(2)* %2 release, align 4
  br label %13

.thread:                                          ; preds = %0, %0
  %3 = bitcast float %desired to i32
  %4 = bitcast float addrspace(2)* %object to i32 addrspace(2)*
  br label %9

; <label>:5                                       ; preds = %0
  %6 = icmp eq i32 %order, 3
  %7 = bitcast float %desired to i32
  %8 = bitcast float addrspace(2)* %object to i32 addrspace(2)*
  br i1 %6, label %9, label %12

; <label>:9                                       ; preds = %5, %.thread
  %10 = phi i32 addrspace(2)* [ %4, %.thread ], [ %8, %5 ]
  %11 = phi i32 [ %3, %.thread ], [ %7, %5 ]
  store atomic volatile i32 %11, i32 addrspace(2)* %10 monotonic, align 4
  br label %13

; <label>:12                                      ; preds = %5
  store atomic volatile i32 %7, i32 addrspace(2)* %8 seq_cst, align 4
  br label %13

; <label>:13                                      ; preds = %12, %.thread1, %9
  ret void
}

; Function Attrs: nounwind uwtable
define float @_Z23pocl_atomic_load__localPVU3AS2U7_Atomicf12memory_order12memory_scope(float addrspace(2)* nocapture readonly %object, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %4 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread
  ]

.thread1:                                         ; preds = %0
  %1 = bitcast float addrspace(2)* %object to i32 addrspace(2)*
  %2 = load atomic volatile i32, i32 addrspace(2)* %1 acquire, align 4
  br label %12

.thread:                                          ; preds = %0, %0
  %3 = bitcast float addrspace(2)* %object to i32 addrspace(2)*
  br label %7

; <label>:4                                       ; preds = %0
  %5 = icmp eq i32 %order, 3
  %6 = bitcast float addrspace(2)* %object to i32 addrspace(2)*
  br i1 %5, label %7, label %10

; <label>:7                                       ; preds = %4, %.thread
  %8 = phi i32 addrspace(2)* [ %3, %.thread ], [ %6, %4 ]
  %9 = load atomic volatile i32, i32 addrspace(2)* %8 monotonic, align 4
  br label %12

; <label>:10                                      ; preds = %4
  %11 = load atomic volatile i32, i32 addrspace(2)* %6 seq_cst, align 4
  br label %12

; <label>:12                                      ; preds = %10, %.thread1, %7
  %.sroa.0.0 = phi i32 [ %9, %7 ], [ %11, %10 ], [ %2, %.thread1 ]
  %13 = bitcast i32 %.sroa.0.0 to float
  ret float %13
}

; Function Attrs: nounwind uwtable
define float @_Z27pocl_atomic_exchange__localPVU3AS2U7_Atomicff12memory_order12memory_scope(float addrspace(2)* %object, float %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %10 [
    i32 0, label %.thread
    i32 1, label %.thread2
    i32 2, label %.thread3
  ]

.thread:                                          ; preds = %0
  %1 = bitcast float %desired to i32
  %2 = bitcast float addrspace(2)* %object to i32 addrspace(2)*
  %3 = atomicrmw volatile xchg i32 addrspace(2)* %2, i32 %1 monotonic
  br label %18

.thread2:                                         ; preds = %0
  %4 = bitcast float %desired to i32
  %5 = bitcast float addrspace(2)* %object to i32 addrspace(2)*
  %6 = atomicrmw volatile xchg i32 addrspace(2)* %5, i32 %4 acquire
  br label %18

.thread3:                                         ; preds = %0
  %7 = bitcast float %desired to i32
  %8 = bitcast float addrspace(2)* %object to i32 addrspace(2)*
  %9 = atomicrmw volatile xchg i32 addrspace(2)* %8, i32 %7 release
  br label %18

; <label>:10                                      ; preds = %0
  %11 = icmp eq i32 %order, 3
  %12 = bitcast float %desired to i32
  %13 = bitcast float addrspace(2)* %object to i32 addrspace(2)*
  br i1 %11, label %14, label %16

; <label>:14                                      ; preds = %10
  %15 = atomicrmw volatile xchg i32 addrspace(2)* %13, i32 %12 acq_rel
  br label %18

; <label>:16                                      ; preds = %10
  %17 = atomicrmw volatile xchg i32 addrspace(2)* %13, i32 %12 seq_cst
  br label %18

; <label>:18                                      ; preds = %16, %14, %.thread3, %.thread2, %.thread
  %.sroa.0.0 = phi i32 [ %3, %.thread ], [ %17, %16 ], [ %15, %14 ], [ %9, %.thread3 ], [ %6, %.thread2 ]
  %19 = bitcast i32 %.sroa.0.0 to float
  ret float %19
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z42pocl_atomic_compare_exchange_strong__localPVU3AS2U7_AtomicfPff12memory_orderS3_12memory_scope(float addrspace(2)* %object, float* nocapture %expected, float %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = bitcast float %desired to i32
  %12 = icmp eq i32 %failure, 0
  br i1 %12, label %20, label %13

; <label>:13                                      ; preds = %9
  %14 = icmp eq i32 %failure, 1
  br i1 %14, label %20, label %15

; <label>:15                                      ; preds = %13
  %16 = icmp eq i32 %failure, 2
  br i1 %16, label %20, label %17

; <label>:17                                      ; preds = %15
  %18 = icmp eq i32 %failure, 3
  %19 = select i1 %18, i32 4, i32 5
  br label %20

; <label>:20                                      ; preds = %13, %15, %17, %9
  %21 = phi i32 [ 0, %9 ], [ 2, %13 ], [ %19, %17 ], [ 3, %15 ]
  %22 = bitcast float addrspace(2)* %object to i32 addrspace(2)*
  %23 = bitcast float* %expected to i32*
  switch i32 %10, label %31 [
    i32 1, label %24
    i32 2, label %24
    i32 3, label %53
    i32 4, label %26
    i32 5, label %28
  ]

; <label>:24                                      ; preds = %20, %20
  %.off = add nsw i32 %21, -1
  %switch = icmp ult i32 %.off, 2
  %25 = load i32, i32* %23, align 4
  br i1 %switch, label %42, label %39

; <label>:26                                      ; preds = %20
  %.off1 = add nsw i32 %21, -1
  %switch2 = icmp ult i32 %.off1, 2
  %27 = load i32, i32* %23, align 4
  br i1 %switch2, label %64, label %61

; <label>:28                                      ; preds = %20
  switch i32 %21, label %75 [
    i32 1, label %79
    i32 2, label %79
    i32 5, label %83
  ]

; <label>:29                                      ; preds = %89, %93, %97, %69, %73, %47, %51, %59, %37
  %.0 = phi i8 [ %38, %37 ], [ %90, %89 ], [ %98, %97 ], [ %94, %93 ], [ %74, %73 ], [ %70, %69 ], [ %60, %59 ], [ %52, %51 ], [ %48, %47 ]
  %30 = icmp ne i8 %.0, 0
  ret i1 %30

; <label>:31                                      ; preds = %20
  %32 = load i32, i32* %23, align 4
  %33 = cmpxchg volatile i32 addrspace(2)* %22, i32 %32, i32 %11 monotonic monotonic
  %34 = extractvalue { i32, i1 } %33, 1
  br i1 %34, label %37, label %35

; <label>:35                                      ; preds = %31
  %36 = extractvalue { i32, i1 } %33, 0
  store i32 %36, i32* %23, align 4
  br label %37

; <label>:37                                      ; preds = %35, %31
  %38 = zext i1 %34 to i8
  br label %29

; <label>:39                                      ; preds = %24
  %40 = cmpxchg volatile i32 addrspace(2)* %22, i32 %25, i32 %11 acquire monotonic
  %41 = extractvalue { i32, i1 } %40, 1
  br i1 %41, label %47, label %45

; <label>:42                                      ; preds = %24
  %43 = cmpxchg volatile i32 addrspace(2)* %22, i32 %25, i32 %11 acquire acquire
  %44 = extractvalue { i32, i1 } %43, 1
  br i1 %44, label %51, label %49

; <label>:45                                      ; preds = %39
  %46 = extractvalue { i32, i1 } %40, 0
  store i32 %46, i32* %23, align 4
  br label %47

; <label>:47                                      ; preds = %45, %39
  %48 = zext i1 %41 to i8
  br label %29

; <label>:49                                      ; preds = %42
  %50 = extractvalue { i32, i1 } %43, 0
  store i32 %50, i32* %23, align 4
  br label %51

; <label>:51                                      ; preds = %49, %42
  %52 = zext i1 %44 to i8
  br label %29

; <label>:53                                      ; preds = %20
  %54 = load i32, i32* %23, align 4
  %55 = cmpxchg volatile i32 addrspace(2)* %22, i32 %54, i32 %11 release monotonic
  %56 = extractvalue { i32, i1 } %55, 1
  br i1 %56, label %59, label %57

; <label>:57                                      ; preds = %53
  %58 = extractvalue { i32, i1 } %55, 0
  store i32 %58, i32* %23, align 4
  br label %59

; <label>:59                                      ; preds = %57, %53
  %60 = zext i1 %56 to i8
  br label %29

; <label>:61                                      ; preds = %26
  %62 = cmpxchg volatile i32 addrspace(2)* %22, i32 %27, i32 %11 acq_rel monotonic
  %63 = extractvalue { i32, i1 } %62, 1
  br i1 %63, label %69, label %67

; <label>:64                                      ; preds = %26
  %65 = cmpxchg volatile i32 addrspace(2)* %22, i32 %27, i32 %11 acq_rel acquire
  %66 = extractvalue { i32, i1 } %65, 1
  br i1 %66, label %73, label %71

; <label>:67                                      ; preds = %61
  %68 = extractvalue { i32, i1 } %62, 0
  store i32 %68, i32* %23, align 4
  br label %69

; <label>:69                                      ; preds = %67, %61
  %70 = zext i1 %63 to i8
  br label %29

; <label>:71                                      ; preds = %64
  %72 = extractvalue { i32, i1 } %65, 0
  store i32 %72, i32* %23, align 4
  br label %73

; <label>:73                                      ; preds = %71, %64
  %74 = zext i1 %66 to i8
  br label %29

; <label>:75                                      ; preds = %28
  %76 = load i32, i32* %23, align 4
  %77 = cmpxchg volatile i32 addrspace(2)* %22, i32 %76, i32 %11 seq_cst monotonic
  %78 = extractvalue { i32, i1 } %77, 1
  br i1 %78, label %89, label %87

; <label>:79                                      ; preds = %28, %28
  %80 = load i32, i32* %23, align 4
  %81 = cmpxchg volatile i32 addrspace(2)* %22, i32 %80, i32 %11 seq_cst acquire
  %82 = extractvalue { i32, i1 } %81, 1
  br i1 %82, label %93, label %91

; <label>:83                                      ; preds = %28
  %84 = load i32, i32* %23, align 4
  %85 = cmpxchg volatile i32 addrspace(2)* %22, i32 %84, i32 %11 seq_cst seq_cst
  %86 = extractvalue { i32, i1 } %85, 1
  br i1 %86, label %97, label %95

; <label>:87                                      ; preds = %75
  %88 = extractvalue { i32, i1 } %77, 0
  store i32 %88, i32* %23, align 4
  br label %89

; <label>:89                                      ; preds = %87, %75
  %90 = zext i1 %78 to i8
  br label %29

; <label>:91                                      ; preds = %79
  %92 = extractvalue { i32, i1 } %81, 0
  store i32 %92, i32* %23, align 4
  br label %93

; <label>:93                                      ; preds = %91, %79
  %94 = zext i1 %82 to i8
  br label %29

; <label>:95                                      ; preds = %83
  %96 = extractvalue { i32, i1 } %85, 0
  store i32 %96, i32* %23, align 4
  br label %97

; <label>:97                                      ; preds = %95, %83
  %98 = zext i1 %86 to i8
  br label %29
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z40pocl_atomic_compare_exchange_weak__localPVU3AS2U7_AtomicfPff12memory_orderS3_12memory_scope(float addrspace(2)* %object, float* nocapture %expected, float %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = bitcast float %desired to i32
  %12 = icmp eq i32 %failure, 0
  br i1 %12, label %20, label %13

; <label>:13                                      ; preds = %9
  %14 = icmp eq i32 %failure, 1
  br i1 %14, label %20, label %15

; <label>:15                                      ; preds = %13
  %16 = icmp eq i32 %failure, 2
  br i1 %16, label %20, label %17

; <label>:17                                      ; preds = %15
  %18 = icmp eq i32 %failure, 3
  %19 = select i1 %18, i32 4, i32 5
  br label %20

; <label>:20                                      ; preds = %13, %15, %17, %9
  %21 = phi i32 [ 0, %9 ], [ 2, %13 ], [ %19, %17 ], [ 3, %15 ]
  %22 = bitcast float addrspace(2)* %object to i32 addrspace(2)*
  %23 = bitcast float* %expected to i32*
  switch i32 %10, label %31 [
    i32 1, label %24
    i32 2, label %24
    i32 3, label %53
    i32 4, label %26
    i32 5, label %28
  ]

; <label>:24                                      ; preds = %20, %20
  %.off = add nsw i32 %21, -1
  %switch = icmp ult i32 %.off, 2
  %25 = load i32, i32* %23, align 4
  br i1 %switch, label %42, label %39

; <label>:26                                      ; preds = %20
  %.off1 = add nsw i32 %21, -1
  %switch2 = icmp ult i32 %.off1, 2
  %27 = load i32, i32* %23, align 4
  br i1 %switch2, label %64, label %61

; <label>:28                                      ; preds = %20
  switch i32 %21, label %75 [
    i32 1, label %79
    i32 2, label %79
    i32 5, label %83
  ]

; <label>:29                                      ; preds = %89, %93, %97, %69, %73, %47, %51, %59, %37
  %.0 = phi i8 [ %38, %37 ], [ %90, %89 ], [ %98, %97 ], [ %94, %93 ], [ %74, %73 ], [ %70, %69 ], [ %60, %59 ], [ %52, %51 ], [ %48, %47 ]
  %30 = icmp ne i8 %.0, 0
  ret i1 %30

; <label>:31                                      ; preds = %20
  %32 = load i32, i32* %23, align 4
  %33 = cmpxchg weak volatile i32 addrspace(2)* %22, i32 %32, i32 %11 monotonic monotonic
  %34 = extractvalue { i32, i1 } %33, 1
  br i1 %34, label %37, label %35

; <label>:35                                      ; preds = %31
  %36 = extractvalue { i32, i1 } %33, 0
  store i32 %36, i32* %23, align 4
  br label %37

; <label>:37                                      ; preds = %35, %31
  %38 = zext i1 %34 to i8
  br label %29

; <label>:39                                      ; preds = %24
  %40 = cmpxchg weak volatile i32 addrspace(2)* %22, i32 %25, i32 %11 acquire monotonic
  %41 = extractvalue { i32, i1 } %40, 1
  br i1 %41, label %47, label %45

; <label>:42                                      ; preds = %24
  %43 = cmpxchg weak volatile i32 addrspace(2)* %22, i32 %25, i32 %11 acquire acquire
  %44 = extractvalue { i32, i1 } %43, 1
  br i1 %44, label %51, label %49

; <label>:45                                      ; preds = %39
  %46 = extractvalue { i32, i1 } %40, 0
  store i32 %46, i32* %23, align 4
  br label %47

; <label>:47                                      ; preds = %45, %39
  %48 = zext i1 %41 to i8
  br label %29

; <label>:49                                      ; preds = %42
  %50 = extractvalue { i32, i1 } %43, 0
  store i32 %50, i32* %23, align 4
  br label %51

; <label>:51                                      ; preds = %49, %42
  %52 = zext i1 %44 to i8
  br label %29

; <label>:53                                      ; preds = %20
  %54 = load i32, i32* %23, align 4
  %55 = cmpxchg weak volatile i32 addrspace(2)* %22, i32 %54, i32 %11 release monotonic
  %56 = extractvalue { i32, i1 } %55, 1
  br i1 %56, label %59, label %57

; <label>:57                                      ; preds = %53
  %58 = extractvalue { i32, i1 } %55, 0
  store i32 %58, i32* %23, align 4
  br label %59

; <label>:59                                      ; preds = %57, %53
  %60 = zext i1 %56 to i8
  br label %29

; <label>:61                                      ; preds = %26
  %62 = cmpxchg weak volatile i32 addrspace(2)* %22, i32 %27, i32 %11 acq_rel monotonic
  %63 = extractvalue { i32, i1 } %62, 1
  br i1 %63, label %69, label %67

; <label>:64                                      ; preds = %26
  %65 = cmpxchg weak volatile i32 addrspace(2)* %22, i32 %27, i32 %11 acq_rel acquire
  %66 = extractvalue { i32, i1 } %65, 1
  br i1 %66, label %73, label %71

; <label>:67                                      ; preds = %61
  %68 = extractvalue { i32, i1 } %62, 0
  store i32 %68, i32* %23, align 4
  br label %69

; <label>:69                                      ; preds = %67, %61
  %70 = zext i1 %63 to i8
  br label %29

; <label>:71                                      ; preds = %64
  %72 = extractvalue { i32, i1 } %65, 0
  store i32 %72, i32* %23, align 4
  br label %73

; <label>:73                                      ; preds = %71, %64
  %74 = zext i1 %66 to i8
  br label %29

; <label>:75                                      ; preds = %28
  %76 = load i32, i32* %23, align 4
  %77 = cmpxchg weak volatile i32 addrspace(2)* %22, i32 %76, i32 %11 seq_cst monotonic
  %78 = extractvalue { i32, i1 } %77, 1
  br i1 %78, label %89, label %87

; <label>:79                                      ; preds = %28, %28
  %80 = load i32, i32* %23, align 4
  %81 = cmpxchg weak volatile i32 addrspace(2)* %22, i32 %80, i32 %11 seq_cst acquire
  %82 = extractvalue { i32, i1 } %81, 1
  br i1 %82, label %93, label %91

; <label>:83                                      ; preds = %28
  %84 = load i32, i32* %23, align 4
  %85 = cmpxchg weak volatile i32 addrspace(2)* %22, i32 %84, i32 %11 seq_cst seq_cst
  %86 = extractvalue { i32, i1 } %85, 1
  br i1 %86, label %97, label %95

; <label>:87                                      ; preds = %75
  %88 = extractvalue { i32, i1 } %77, 0
  store i32 %88, i32* %23, align 4
  br label %89

; <label>:89                                      ; preds = %87, %75
  %90 = zext i1 %78 to i8
  br label %29

; <label>:91                                      ; preds = %79
  %92 = extractvalue { i32, i1 } %81, 0
  store i32 %92, i32* %23, align 4
  br label %93

; <label>:93                                      ; preds = %91, %79
  %94 = zext i1 %82 to i8
  br label %29

; <label>:95                                      ; preds = %83
  %96 = extractvalue { i32, i1 } %85, 0
  store i32 %96, i32* %23, align 4
  br label %97

; <label>:97                                      ; preds = %95, %83
  %98 = zext i1 %86 to i8
  br label %29
}

; Function Attrs: nounwind uwtable
define void @_Z24pocl_atomic_store__localPVU3AS2U7_Atomicll12memory_order12memory_scope(i64 addrspace(2)* nocapture %object, i64 %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %1 [
    i32 0, label %.thread
    i32 1, label %.thread
    i32 2, label %.thread1
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  store atomic volatile i64 %desired, i64 addrspace(2)* %object monotonic, align 8
  br label %2

.thread1:                                         ; preds = %0
  store atomic volatile i64 %desired, i64 addrspace(2)* %object release, align 8
  br label %2

; <label>:1                                       ; preds = %0
  store atomic volatile i64 %desired, i64 addrspace(2)* %object seq_cst, align 8
  br label %2

; <label>:2                                       ; preds = %1, %.thread1, %.thread
  ret void
}

; Function Attrs: nounwind uwtable
define i64 @_Z23pocl_atomic_load__localPVU3AS2U7_Atomicl12memory_order12memory_scope(i64 addrspace(2)* nocapture readonly %object, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %3 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  %1 = load atomic volatile i64, i64 addrspace(2)* %object monotonic, align 8
  br label %5

.thread1:                                         ; preds = %0
  %2 = load atomic volatile i64, i64 addrspace(2)* %object acquire, align 8
  br label %5

; <label>:3                                       ; preds = %0
  %4 = load atomic volatile i64, i64 addrspace(2)* %object seq_cst, align 8
  br label %5

; <label>:5                                       ; preds = %3, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %4, %3 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z27pocl_atomic_exchange__localPVU3AS2U7_Atomicll12memory_order12memory_scope(i64 addrspace(2)* %object, i64 %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xchg i64 addrspace(2)* %object, i64 %desired monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xchg i64 addrspace(2)* %object, i64 %desired acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xchg i64 addrspace(2)* %object, i64 %desired release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xchg i64 addrspace(2)* %object, i64 %desired acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xchg i64 addrspace(2)* %object, i64 %desired seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z42pocl_atomic_compare_exchange_strong__localPVU3AS2U7_AtomiclPll12memory_orderS3_12memory_scope(i64 addrspace(2)* %object, i64* nocapture %expected, i64 %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = icmp eq i32 %failure, 0
  br i1 %11, label %19, label %12

; <label>:12                                      ; preds = %9
  %13 = icmp eq i32 %failure, 1
  br i1 %13, label %19, label %14

; <label>:14                                      ; preds = %12
  %15 = icmp eq i32 %failure, 2
  br i1 %15, label %19, label %16

; <label>:16                                      ; preds = %14
  %17 = icmp eq i32 %failure, 3
  %18 = select i1 %17, i32 4, i32 5
  br label %19

; <label>:19                                      ; preds = %12, %14, %16, %9
  %20 = phi i32 [ 0, %9 ], [ 2, %12 ], [ %18, %16 ], [ 3, %14 ]
  switch i32 %10, label %28 [
    i32 1, label %21
    i32 2, label %21
    i32 3, label %50
    i32 4, label %23
    i32 5, label %25
  ]

; <label>:21                                      ; preds = %19, %19
  %.off = add nsw i32 %20, -1
  %switch = icmp ult i32 %.off, 2
  %22 = load i64, i64* %expected, align 8
  br i1 %switch, label %39, label %36

; <label>:23                                      ; preds = %19
  %.off1 = add nsw i32 %20, -1
  %switch2 = icmp ult i32 %.off1, 2
  %24 = load i64, i64* %expected, align 8
  br i1 %switch2, label %61, label %58

; <label>:25                                      ; preds = %19
  switch i32 %20, label %72 [
    i32 1, label %76
    i32 2, label %76
    i32 5, label %80
  ]

; <label>:26                                      ; preds = %86, %90, %94, %66, %70, %44, %48, %56, %34
  %.0 = phi i8 [ %35, %34 ], [ %87, %86 ], [ %95, %94 ], [ %91, %90 ], [ %71, %70 ], [ %67, %66 ], [ %57, %56 ], [ %49, %48 ], [ %45, %44 ]
  %27 = icmp ne i8 %.0, 0
  ret i1 %27

; <label>:28                                      ; preds = %19
  %29 = load i64, i64* %expected, align 8
  %30 = cmpxchg volatile i64 addrspace(2)* %object, i64 %29, i64 %desired monotonic monotonic
  %31 = extractvalue { i64, i1 } %30, 1
  br i1 %31, label %34, label %32

; <label>:32                                      ; preds = %28
  %33 = extractvalue { i64, i1 } %30, 0
  store i64 %33, i64* %expected, align 8
  br label %34

; <label>:34                                      ; preds = %32, %28
  %35 = zext i1 %31 to i8
  br label %26

; <label>:36                                      ; preds = %21
  %37 = cmpxchg volatile i64 addrspace(2)* %object, i64 %22, i64 %desired acquire monotonic
  %38 = extractvalue { i64, i1 } %37, 1
  br i1 %38, label %44, label %42

; <label>:39                                      ; preds = %21
  %40 = cmpxchg volatile i64 addrspace(2)* %object, i64 %22, i64 %desired acquire acquire
  %41 = extractvalue { i64, i1 } %40, 1
  br i1 %41, label %48, label %46

; <label>:42                                      ; preds = %36
  %43 = extractvalue { i64, i1 } %37, 0
  store i64 %43, i64* %expected, align 8
  br label %44

; <label>:44                                      ; preds = %42, %36
  %45 = zext i1 %38 to i8
  br label %26

; <label>:46                                      ; preds = %39
  %47 = extractvalue { i64, i1 } %40, 0
  store i64 %47, i64* %expected, align 8
  br label %48

; <label>:48                                      ; preds = %46, %39
  %49 = zext i1 %41 to i8
  br label %26

; <label>:50                                      ; preds = %19
  %51 = load i64, i64* %expected, align 8
  %52 = cmpxchg volatile i64 addrspace(2)* %object, i64 %51, i64 %desired release monotonic
  %53 = extractvalue { i64, i1 } %52, 1
  br i1 %53, label %56, label %54

; <label>:54                                      ; preds = %50
  %55 = extractvalue { i64, i1 } %52, 0
  store i64 %55, i64* %expected, align 8
  br label %56

; <label>:56                                      ; preds = %54, %50
  %57 = zext i1 %53 to i8
  br label %26

; <label>:58                                      ; preds = %23
  %59 = cmpxchg volatile i64 addrspace(2)* %object, i64 %24, i64 %desired acq_rel monotonic
  %60 = extractvalue { i64, i1 } %59, 1
  br i1 %60, label %66, label %64

; <label>:61                                      ; preds = %23
  %62 = cmpxchg volatile i64 addrspace(2)* %object, i64 %24, i64 %desired acq_rel acquire
  %63 = extractvalue { i64, i1 } %62, 1
  br i1 %63, label %70, label %68

; <label>:64                                      ; preds = %58
  %65 = extractvalue { i64, i1 } %59, 0
  store i64 %65, i64* %expected, align 8
  br label %66

; <label>:66                                      ; preds = %64, %58
  %67 = zext i1 %60 to i8
  br label %26

; <label>:68                                      ; preds = %61
  %69 = extractvalue { i64, i1 } %62, 0
  store i64 %69, i64* %expected, align 8
  br label %70

; <label>:70                                      ; preds = %68, %61
  %71 = zext i1 %63 to i8
  br label %26

; <label>:72                                      ; preds = %25
  %73 = load i64, i64* %expected, align 8
  %74 = cmpxchg volatile i64 addrspace(2)* %object, i64 %73, i64 %desired seq_cst monotonic
  %75 = extractvalue { i64, i1 } %74, 1
  br i1 %75, label %86, label %84

; <label>:76                                      ; preds = %25, %25
  %77 = load i64, i64* %expected, align 8
  %78 = cmpxchg volatile i64 addrspace(2)* %object, i64 %77, i64 %desired seq_cst acquire
  %79 = extractvalue { i64, i1 } %78, 1
  br i1 %79, label %90, label %88

; <label>:80                                      ; preds = %25
  %81 = load i64, i64* %expected, align 8
  %82 = cmpxchg volatile i64 addrspace(2)* %object, i64 %81, i64 %desired seq_cst seq_cst
  %83 = extractvalue { i64, i1 } %82, 1
  br i1 %83, label %94, label %92

; <label>:84                                      ; preds = %72
  %85 = extractvalue { i64, i1 } %74, 0
  store i64 %85, i64* %expected, align 8
  br label %86

; <label>:86                                      ; preds = %84, %72
  %87 = zext i1 %75 to i8
  br label %26

; <label>:88                                      ; preds = %76
  %89 = extractvalue { i64, i1 } %78, 0
  store i64 %89, i64* %expected, align 8
  br label %90

; <label>:90                                      ; preds = %88, %76
  %91 = zext i1 %79 to i8
  br label %26

; <label>:92                                      ; preds = %80
  %93 = extractvalue { i64, i1 } %82, 0
  store i64 %93, i64* %expected, align 8
  br label %94

; <label>:94                                      ; preds = %92, %80
  %95 = zext i1 %83 to i8
  br label %26
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z40pocl_atomic_compare_exchange_weak__localPVU3AS2U7_AtomiclPll12memory_orderS3_12memory_scope(i64 addrspace(2)* %object, i64* nocapture %expected, i64 %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = icmp eq i32 %failure, 0
  br i1 %11, label %19, label %12

; <label>:12                                      ; preds = %9
  %13 = icmp eq i32 %failure, 1
  br i1 %13, label %19, label %14

; <label>:14                                      ; preds = %12
  %15 = icmp eq i32 %failure, 2
  br i1 %15, label %19, label %16

; <label>:16                                      ; preds = %14
  %17 = icmp eq i32 %failure, 3
  %18 = select i1 %17, i32 4, i32 5
  br label %19

; <label>:19                                      ; preds = %12, %14, %16, %9
  %20 = phi i32 [ 0, %9 ], [ 2, %12 ], [ %18, %16 ], [ 3, %14 ]
  switch i32 %10, label %28 [
    i32 1, label %21
    i32 2, label %21
    i32 3, label %50
    i32 4, label %23
    i32 5, label %25
  ]

; <label>:21                                      ; preds = %19, %19
  %.off = add nsw i32 %20, -1
  %switch = icmp ult i32 %.off, 2
  %22 = load i64, i64* %expected, align 8
  br i1 %switch, label %39, label %36

; <label>:23                                      ; preds = %19
  %.off1 = add nsw i32 %20, -1
  %switch2 = icmp ult i32 %.off1, 2
  %24 = load i64, i64* %expected, align 8
  br i1 %switch2, label %61, label %58

; <label>:25                                      ; preds = %19
  switch i32 %20, label %72 [
    i32 1, label %76
    i32 2, label %76
    i32 5, label %80
  ]

; <label>:26                                      ; preds = %86, %90, %94, %66, %70, %44, %48, %56, %34
  %.0 = phi i8 [ %35, %34 ], [ %87, %86 ], [ %95, %94 ], [ %91, %90 ], [ %71, %70 ], [ %67, %66 ], [ %57, %56 ], [ %49, %48 ], [ %45, %44 ]
  %27 = icmp ne i8 %.0, 0
  ret i1 %27

; <label>:28                                      ; preds = %19
  %29 = load i64, i64* %expected, align 8
  %30 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %29, i64 %desired monotonic monotonic
  %31 = extractvalue { i64, i1 } %30, 1
  br i1 %31, label %34, label %32

; <label>:32                                      ; preds = %28
  %33 = extractvalue { i64, i1 } %30, 0
  store i64 %33, i64* %expected, align 8
  br label %34

; <label>:34                                      ; preds = %32, %28
  %35 = zext i1 %31 to i8
  br label %26

; <label>:36                                      ; preds = %21
  %37 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %22, i64 %desired acquire monotonic
  %38 = extractvalue { i64, i1 } %37, 1
  br i1 %38, label %44, label %42

; <label>:39                                      ; preds = %21
  %40 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %22, i64 %desired acquire acquire
  %41 = extractvalue { i64, i1 } %40, 1
  br i1 %41, label %48, label %46

; <label>:42                                      ; preds = %36
  %43 = extractvalue { i64, i1 } %37, 0
  store i64 %43, i64* %expected, align 8
  br label %44

; <label>:44                                      ; preds = %42, %36
  %45 = zext i1 %38 to i8
  br label %26

; <label>:46                                      ; preds = %39
  %47 = extractvalue { i64, i1 } %40, 0
  store i64 %47, i64* %expected, align 8
  br label %48

; <label>:48                                      ; preds = %46, %39
  %49 = zext i1 %41 to i8
  br label %26

; <label>:50                                      ; preds = %19
  %51 = load i64, i64* %expected, align 8
  %52 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %51, i64 %desired release monotonic
  %53 = extractvalue { i64, i1 } %52, 1
  br i1 %53, label %56, label %54

; <label>:54                                      ; preds = %50
  %55 = extractvalue { i64, i1 } %52, 0
  store i64 %55, i64* %expected, align 8
  br label %56

; <label>:56                                      ; preds = %54, %50
  %57 = zext i1 %53 to i8
  br label %26

; <label>:58                                      ; preds = %23
  %59 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %24, i64 %desired acq_rel monotonic
  %60 = extractvalue { i64, i1 } %59, 1
  br i1 %60, label %66, label %64

; <label>:61                                      ; preds = %23
  %62 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %24, i64 %desired acq_rel acquire
  %63 = extractvalue { i64, i1 } %62, 1
  br i1 %63, label %70, label %68

; <label>:64                                      ; preds = %58
  %65 = extractvalue { i64, i1 } %59, 0
  store i64 %65, i64* %expected, align 8
  br label %66

; <label>:66                                      ; preds = %64, %58
  %67 = zext i1 %60 to i8
  br label %26

; <label>:68                                      ; preds = %61
  %69 = extractvalue { i64, i1 } %62, 0
  store i64 %69, i64* %expected, align 8
  br label %70

; <label>:70                                      ; preds = %68, %61
  %71 = zext i1 %63 to i8
  br label %26

; <label>:72                                      ; preds = %25
  %73 = load i64, i64* %expected, align 8
  %74 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %73, i64 %desired seq_cst monotonic
  %75 = extractvalue { i64, i1 } %74, 1
  br i1 %75, label %86, label %84

; <label>:76                                      ; preds = %25, %25
  %77 = load i64, i64* %expected, align 8
  %78 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %77, i64 %desired seq_cst acquire
  %79 = extractvalue { i64, i1 } %78, 1
  br i1 %79, label %90, label %88

; <label>:80                                      ; preds = %25
  %81 = load i64, i64* %expected, align 8
  %82 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %81, i64 %desired seq_cst seq_cst
  %83 = extractvalue { i64, i1 } %82, 1
  br i1 %83, label %94, label %92

; <label>:84                                      ; preds = %72
  %85 = extractvalue { i64, i1 } %74, 0
  store i64 %85, i64* %expected, align 8
  br label %86

; <label>:86                                      ; preds = %84, %72
  %87 = zext i1 %75 to i8
  br label %26

; <label>:88                                      ; preds = %76
  %89 = extractvalue { i64, i1 } %78, 0
  store i64 %89, i64* %expected, align 8
  br label %90

; <label>:90                                      ; preds = %88, %76
  %91 = zext i1 %79 to i8
  br label %26

; <label>:92                                      ; preds = %80
  %93 = extractvalue { i64, i1 } %82, 0
  store i64 %93, i64* %expected, align 8
  br label %94

; <label>:94                                      ; preds = %92, %80
  %95 = zext i1 %83 to i8
  br label %26
}

; Function Attrs: nounwind uwtable
define i64 @_Z28pocl_atomic_fetch_add__localPVU3AS2U7_Atomicll12memory_order12memory_scope(i64 addrspace(2)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile add i64 addrspace(2)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile add i64 addrspace(2)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile add i64 addrspace(2)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile add i64 addrspace(2)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile add i64 addrspace(2)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z28pocl_atomic_fetch_sub__localPVU3AS2U7_Atomicll12memory_order12memory_scope(i64 addrspace(2)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile sub i64 addrspace(2)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile sub i64 addrspace(2)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile sub i64 addrspace(2)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile sub i64 addrspace(2)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile sub i64 addrspace(2)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z27pocl_atomic_fetch_or__localPVU3AS2U7_Atomicll12memory_order12memory_scope(i64 addrspace(2)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile or i64 addrspace(2)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile or i64 addrspace(2)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile or i64 addrspace(2)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile or i64 addrspace(2)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile or i64 addrspace(2)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z28pocl_atomic_fetch_xor__localPVU3AS2U7_Atomicll12memory_order12memory_scope(i64 addrspace(2)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xor i64 addrspace(2)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xor i64 addrspace(2)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xor i64 addrspace(2)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xor i64 addrspace(2)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xor i64 addrspace(2)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z28pocl_atomic_fetch_and__localPVU3AS2U7_Atomicll12memory_order12memory_scope(i64 addrspace(2)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile and i64 addrspace(2)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile and i64 addrspace(2)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile and i64 addrspace(2)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile and i64 addrspace(2)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile and i64 addrspace(2)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: noreturn nounwind uwtable
define i64 @_Z28pocl_atomic_fetch_min__localPVU3AS2U7_Atomicll12memory_order12memory_scope(i64 addrspace(2)* nocapture readnone %object, i64 %operand, i32 %order, i32 %scope) #1 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile min i64 addrspace(2)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile min i64 addrspace(2)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile min i64 addrspace(2)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile min i64 addrspace(2)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile min i64 addrspace(2)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: noreturn nounwind uwtable
define i64 @_Z28pocl_atomic_fetch_max__localPVU3AS2U7_Atomicll12memory_order12memory_scope(i64 addrspace(2)* nocapture readnone %object, i64 %operand, i32 %order, i32 %scope) #1 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile max i64 addrspace(2)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile max i64 addrspace(2)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile max i64 addrspace(2)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile max i64 addrspace(2)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile max i64 addrspace(2)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0

}

; Function Attrs: nounwind uwtable
define void @_Z24pocl_atomic_store__localPVU3AS2U7_Atomicmm12memory_order12memory_scope(i64 addrspace(2)* nocapture %object, i64 %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %1 [
    i32 0, label %.thread
    i32 1, label %.thread
    i32 2, label %.thread1
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  store atomic volatile i64 %desired, i64 addrspace(2)* %object monotonic, align 8
  br label %2

.thread1:                                         ; preds = %0
  store atomic volatile i64 %desired, i64 addrspace(2)* %object release, align 8
  br label %2

; <label>:1                                       ; preds = %0
  store atomic volatile i64 %desired, i64 addrspace(2)* %object seq_cst, align 8
  br label %2

; <label>:2                                       ; preds = %1, %.thread1, %.thread
  ret void
}

; Function Attrs: nounwind uwtable
define i64 @_Z23pocl_atomic_load__localPVU3AS2U7_Atomicm12memory_order12memory_scope(i64 addrspace(2)* nocapture readonly %object, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %3 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread
    i32 3, label %.thread
  ]

.thread:                                          ; preds = %0, %0, %0
  %1 = load atomic volatile i64, i64 addrspace(2)* %object monotonic, align 8
  br label %5

.thread1:                                         ; preds = %0
  %2 = load atomic volatile i64, i64 addrspace(2)* %object acquire, align 8
  br label %5

; <label>:3                                       ; preds = %0
  %4 = load atomic volatile i64, i64 addrspace(2)* %object seq_cst, align 8
  br label %5

; <label>:5                                       ; preds = %3, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %4, %3 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z27pocl_atomic_exchange__localPVU3AS2U7_Atomicmm12memory_order12memory_scope(i64 addrspace(2)* %object, i64 %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xchg i64 addrspace(2)* %object, i64 %desired monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xchg i64 addrspace(2)* %object, i64 %desired acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xchg i64 addrspace(2)* %object, i64 %desired release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xchg i64 addrspace(2)* %object, i64 %desired acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xchg i64 addrspace(2)* %object, i64 %desired seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z42pocl_atomic_compare_exchange_strong__localPVU3AS2U7_AtomicmPmm12memory_orderS3_12memory_scope(i64 addrspace(2)* %object, i64* nocapture %expected, i64 %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = icmp eq i32 %failure, 0
  br i1 %11, label %19, label %12

; <label>:12                                      ; preds = %9
  %13 = icmp eq i32 %failure, 1
  br i1 %13, label %19, label %14

; <label>:14                                      ; preds = %12
  %15 = icmp eq i32 %failure, 2
  br i1 %15, label %19, label %16

; <label>:16                                      ; preds = %14
  %17 = icmp eq i32 %failure, 3
  %18 = select i1 %17, i32 4, i32 5
  br label %19

; <label>:19                                      ; preds = %12, %14, %16, %9
  %20 = phi i32 [ 0, %9 ], [ 2, %12 ], [ %18, %16 ], [ 3, %14 ]
  switch i32 %10, label %28 [
    i32 1, label %21
    i32 2, label %21
    i32 3, label %50
    i32 4, label %23
    i32 5, label %25
  ]

; <label>:21                                      ; preds = %19, %19
  %.off = add nsw i32 %20, -1
  %switch = icmp ult i32 %.off, 2
  %22 = load i64, i64* %expected, align 8
  br i1 %switch, label %39, label %36

; <label>:23                                      ; preds = %19
  %.off1 = add nsw i32 %20, -1
  %switch2 = icmp ult i32 %.off1, 2
  %24 = load i64, i64* %expected, align 8
  br i1 %switch2, label %61, label %58

; <label>:25                                      ; preds = %19
  switch i32 %20, label %72 [
    i32 1, label %76
    i32 2, label %76
    i32 5, label %80
  ]

; <label>:26                                      ; preds = %86, %90, %94, %66, %70, %44, %48, %56, %34
  %.0 = phi i8 [ %35, %34 ], [ %87, %86 ], [ %95, %94 ], [ %91, %90 ], [ %71, %70 ], [ %67, %66 ], [ %57, %56 ], [ %49, %48 ], [ %45, %44 ]
  %27 = icmp ne i8 %.0, 0
  ret i1 %27

; <label>:28                                      ; preds = %19
  %29 = load i64, i64* %expected, align 8
  %30 = cmpxchg volatile i64 addrspace(2)* %object, i64 %29, i64 %desired monotonic monotonic
  %31 = extractvalue { i64, i1 } %30, 1
  br i1 %31, label %34, label %32

; <label>:32                                      ; preds = %28
  %33 = extractvalue { i64, i1 } %30, 0
  store i64 %33, i64* %expected, align 8
  br label %34

; <label>:34                                      ; preds = %32, %28
  %35 = zext i1 %31 to i8
  br label %26

; <label>:36                                      ; preds = %21
  %37 = cmpxchg volatile i64 addrspace(2)* %object, i64 %22, i64 %desired acquire monotonic
  %38 = extractvalue { i64, i1 } %37, 1
  br i1 %38, label %44, label %42

; <label>:39                                      ; preds = %21
  %40 = cmpxchg volatile i64 addrspace(2)* %object, i64 %22, i64 %desired acquire acquire
  %41 = extractvalue { i64, i1 } %40, 1
  br i1 %41, label %48, label %46

; <label>:42                                      ; preds = %36
  %43 = extractvalue { i64, i1 } %37, 0
  store i64 %43, i64* %expected, align 8
  br label %44

; <label>:44                                      ; preds = %42, %36
  %45 = zext i1 %38 to i8
  br label %26

; <label>:46                                      ; preds = %39
  %47 = extractvalue { i64, i1 } %40, 0
  store i64 %47, i64* %expected, align 8
  br label %48

; <label>:48                                      ; preds = %46, %39
  %49 = zext i1 %41 to i8
  br label %26

; <label>:50                                      ; preds = %19
  %51 = load i64, i64* %expected, align 8
  %52 = cmpxchg volatile i64 addrspace(2)* %object, i64 %51, i64 %desired release monotonic
  %53 = extractvalue { i64, i1 } %52, 1
  br i1 %53, label %56, label %54

; <label>:54                                      ; preds = %50
  %55 = extractvalue { i64, i1 } %52, 0
  store i64 %55, i64* %expected, align 8
  br label %56

; <label>:56                                      ; preds = %54, %50
  %57 = zext i1 %53 to i8
  br label %26

; <label>:58                                      ; preds = %23
  %59 = cmpxchg volatile i64 addrspace(2)* %object, i64 %24, i64 %desired acq_rel monotonic
  %60 = extractvalue { i64, i1 } %59, 1
  br i1 %60, label %66, label %64

; <label>:61                                      ; preds = %23
  %62 = cmpxchg volatile i64 addrspace(2)* %object, i64 %24, i64 %desired acq_rel acquire
  %63 = extractvalue { i64, i1 } %62, 1
  br i1 %63, label %70, label %68

; <label>:64                                      ; preds = %58
  %65 = extractvalue { i64, i1 } %59, 0
  store i64 %65, i64* %expected, align 8
  br label %66

; <label>:66                                      ; preds = %64, %58
  %67 = zext i1 %60 to i8
  br label %26

; <label>:68                                      ; preds = %61
  %69 = extractvalue { i64, i1 } %62, 0
  store i64 %69, i64* %expected, align 8
  br label %70

; <label>:70                                      ; preds = %68, %61
  %71 = zext i1 %63 to i8
  br label %26

; <label>:72                                      ; preds = %25
  %73 = load i64, i64* %expected, align 8
  %74 = cmpxchg volatile i64 addrspace(2)* %object, i64 %73, i64 %desired seq_cst monotonic
  %75 = extractvalue { i64, i1 } %74, 1
  br i1 %75, label %86, label %84

; <label>:76                                      ; preds = %25, %25
  %77 = load i64, i64* %expected, align 8
  %78 = cmpxchg volatile i64 addrspace(2)* %object, i64 %77, i64 %desired seq_cst acquire
  %79 = extractvalue { i64, i1 } %78, 1
  br i1 %79, label %90, label %88

; <label>:80                                      ; preds = %25
  %81 = load i64, i64* %expected, align 8
  %82 = cmpxchg volatile i64 addrspace(2)* %object, i64 %81, i64 %desired seq_cst seq_cst
  %83 = extractvalue { i64, i1 } %82, 1
  br i1 %83, label %94, label %92

; <label>:84                                      ; preds = %72
  %85 = extractvalue { i64, i1 } %74, 0
  store i64 %85, i64* %expected, align 8
  br label %86

; <label>:86                                      ; preds = %84, %72
  %87 = zext i1 %75 to i8
  br label %26

; <label>:88                                      ; preds = %76
  %89 = extractvalue { i64, i1 } %78, 0
  store i64 %89, i64* %expected, align 8
  br label %90

; <label>:90                                      ; preds = %88, %76
  %91 = zext i1 %79 to i8
  br label %26

; <label>:92                                      ; preds = %80
  %93 = extractvalue { i64, i1 } %82, 0
  store i64 %93, i64* %expected, align 8
  br label %94

; <label>:94                                      ; preds = %92, %80
  %95 = zext i1 %83 to i8
  br label %26
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z40pocl_atomic_compare_exchange_weak__localPVU3AS2U7_AtomicmPmm12memory_orderS3_12memory_scope(i64 addrspace(2)* %object, i64* nocapture %expected, i64 %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = icmp eq i32 %failure, 0
  br i1 %11, label %19, label %12

; <label>:12                                      ; preds = %9
  %13 = icmp eq i32 %failure, 1
  br i1 %13, label %19, label %14

; <label>:14                                      ; preds = %12
  %15 = icmp eq i32 %failure, 2
  br i1 %15, label %19, label %16

; <label>:16                                      ; preds = %14
  %17 = icmp eq i32 %failure, 3
  %18 = select i1 %17, i32 4, i32 5
  br label %19

; <label>:19                                      ; preds = %12, %14, %16, %9
  %20 = phi i32 [ 0, %9 ], [ 2, %12 ], [ %18, %16 ], [ 3, %14 ]
  switch i32 %10, label %28 [
    i32 1, label %21
    i32 2, label %21
    i32 3, label %50
    i32 4, label %23
    i32 5, label %25
  ]

; <label>:21                                      ; preds = %19, %19
  %.off = add nsw i32 %20, -1
  %switch = icmp ult i32 %.off, 2
  %22 = load i64, i64* %expected, align 8
  br i1 %switch, label %39, label %36

; <label>:23                                      ; preds = %19
  %.off1 = add nsw i32 %20, -1
  %switch2 = icmp ult i32 %.off1, 2
  %24 = load i64, i64* %expected, align 8
  br i1 %switch2, label %61, label %58

; <label>:25                                      ; preds = %19
  switch i32 %20, label %72 [
    i32 1, label %76
    i32 2, label %76
    i32 5, label %80
  ]

; <label>:26                                      ; preds = %86, %90, %94, %66, %70, %44, %48, %56, %34
  %.0 = phi i8 [ %35, %34 ], [ %87, %86 ], [ %95, %94 ], [ %91, %90 ], [ %71, %70 ], [ %67, %66 ], [ %57, %56 ], [ %49, %48 ], [ %45, %44 ]
  %27 = icmp ne i8 %.0, 0
  ret i1 %27

; <label>:28                                      ; preds = %19
  %29 = load i64, i64* %expected, align 8
  %30 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %29, i64 %desired monotonic monotonic
  %31 = extractvalue { i64, i1 } %30, 1
  br i1 %31, label %34, label %32

; <label>:32                                      ; preds = %28
  %33 = extractvalue { i64, i1 } %30, 0
  store i64 %33, i64* %expected, align 8
  br label %34

; <label>:34                                      ; preds = %32, %28
  %35 = zext i1 %31 to i8
  br label %26

; <label>:36                                      ; preds = %21
  %37 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %22, i64 %desired acquire monotonic
  %38 = extractvalue { i64, i1 } %37, 1
  br i1 %38, label %44, label %42

; <label>:39                                      ; preds = %21
  %40 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %22, i64 %desired acquire acquire
  %41 = extractvalue { i64, i1 } %40, 1
  br i1 %41, label %48, label %46

; <label>:42                                      ; preds = %36
  %43 = extractvalue { i64, i1 } %37, 0
  store i64 %43, i64* %expected, align 8
  br label %44

; <label>:44                                      ; preds = %42, %36
  %45 = zext i1 %38 to i8
  br label %26

; <label>:46                                      ; preds = %39
  %47 = extractvalue { i64, i1 } %40, 0
  store i64 %47, i64* %expected, align 8
  br label %48

; <label>:48                                      ; preds = %46, %39
  %49 = zext i1 %41 to i8
  br label %26

; <label>:50                                      ; preds = %19
  %51 = load i64, i64* %expected, align 8
  %52 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %51, i64 %desired release monotonic
  %53 = extractvalue { i64, i1 } %52, 1
  br i1 %53, label %56, label %54

; <label>:54                                      ; preds = %50
  %55 = extractvalue { i64, i1 } %52, 0
  store i64 %55, i64* %expected, align 8
  br label %56

; <label>:56                                      ; preds = %54, %50
  %57 = zext i1 %53 to i8
  br label %26

; <label>:58                                      ; preds = %23
  %59 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %24, i64 %desired acq_rel monotonic
  %60 = extractvalue { i64, i1 } %59, 1
  br i1 %60, label %66, label %64

; <label>:61                                      ; preds = %23
  %62 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %24, i64 %desired acq_rel acquire
  %63 = extractvalue { i64, i1 } %62, 1
  br i1 %63, label %70, label %68

; <label>:64                                      ; preds = %58
  %65 = extractvalue { i64, i1 } %59, 0
  store i64 %65, i64* %expected, align 8
  br label %66

; <label>:66                                      ; preds = %64, %58
  %67 = zext i1 %60 to i8
  br label %26

; <label>:68                                      ; preds = %61
  %69 = extractvalue { i64, i1 } %62, 0
  store i64 %69, i64* %expected, align 8
  br label %70

; <label>:70                                      ; preds = %68, %61
  %71 = zext i1 %63 to i8
  br label %26

; <label>:72                                      ; preds = %25
  %73 = load i64, i64* %expected, align 8
  %74 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %73, i64 %desired seq_cst monotonic
  %75 = extractvalue { i64, i1 } %74, 1
  br i1 %75, label %86, label %84

; <label>:76                                      ; preds = %25, %25
  %77 = load i64, i64* %expected, align 8
  %78 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %77, i64 %desired seq_cst acquire
  %79 = extractvalue { i64, i1 } %78, 1
  br i1 %79, label %90, label %88

; <label>:80                                      ; preds = %25
  %81 = load i64, i64* %expected, align 8
  %82 = cmpxchg weak volatile i64 addrspace(2)* %object, i64 %81, i64 %desired seq_cst seq_cst
  %83 = extractvalue { i64, i1 } %82, 1
  br i1 %83, label %94, label %92

; <label>:84                                      ; preds = %72
  %85 = extractvalue { i64, i1 } %74, 0
  store i64 %85, i64* %expected, align 8
  br label %86

; <label>:86                                      ; preds = %84, %72
  %87 = zext i1 %75 to i8
  br label %26

; <label>:88                                      ; preds = %76
  %89 = extractvalue { i64, i1 } %78, 0
  store i64 %89, i64* %expected, align 8
  br label %90

; <label>:90                                      ; preds = %88, %76
  %91 = zext i1 %79 to i8
  br label %26

; <label>:92                                      ; preds = %80
  %93 = extractvalue { i64, i1 } %82, 0
  store i64 %93, i64* %expected, align 8
  br label %94

; <label>:94                                      ; preds = %92, %80
  %95 = zext i1 %83 to i8
  br label %26
}

; Function Attrs: nounwind uwtable
define i64 @_Z28pocl_atomic_fetch_add__localPVU3AS2U7_Atomicmm12memory_order12memory_scope(i64 addrspace(2)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile add i64 addrspace(2)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile add i64 addrspace(2)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile add i64 addrspace(2)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile add i64 addrspace(2)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile add i64 addrspace(2)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z28pocl_atomic_fetch_sub__localPVU3AS2U7_Atomicmm12memory_order12memory_scope(i64 addrspace(2)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile sub i64 addrspace(2)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile sub i64 addrspace(2)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile sub i64 addrspace(2)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile sub i64 addrspace(2)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile sub i64 addrspace(2)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z27pocl_atomic_fetch_or__localPVU3AS2U7_Atomicmm12memory_order12memory_scope(i64 addrspace(2)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile or i64 addrspace(2)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile or i64 addrspace(2)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile or i64 addrspace(2)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile or i64 addrspace(2)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile or i64 addrspace(2)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z28pocl_atomic_fetch_xor__localPVU3AS2U7_Atomicmm12memory_order12memory_scope(i64 addrspace(2)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile xor i64 addrspace(2)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile xor i64 addrspace(2)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile xor i64 addrspace(2)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile xor i64 addrspace(2)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile xor i64 addrspace(2)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: nounwind uwtable
define i64 @_Z28pocl_atomic_fetch_and__localPVU3AS2U7_Atomicmm12memory_order12memory_scope(i64 addrspace(2)* %object, i64 %operand, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile and i64 addrspace(2)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile and i64 addrspace(2)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile and i64 addrspace(2)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile and i64 addrspace(2)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile and i64 addrspace(2)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: noreturn nounwind uwtable
define i64 @_Z28pocl_atomic_fetch_min__localPVU3AS2U7_Atomicmm12memory_order12memory_scope(i64 addrspace(2)* nocapture readnone %object, i64 %operand, i32 %order, i32 %scope) #1 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile umin i64 addrspace(2)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile umin i64 addrspace(2)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile umin i64 addrspace(2)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile umin i64 addrspace(2)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile umin i64 addrspace(2)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0
}

; Function Attrs: noreturn nounwind uwtable
define i64 @_Z28pocl_atomic_fetch_max__localPVU3AS2U7_Atomicmm12memory_order12memory_scope(i64 addrspace(2)* nocapture readnone %object, i64 %operand, i32 %order, i32 %scope) #1 {
  switch i32 %order, label %6 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread2
    i32 3, label %4
  ]

.thread:                                          ; preds = %0
  %1 = atomicrmw volatile umax i64 addrspace(2)* %object, i64 %operand monotonic
  br label %8

.thread1:                                         ; preds = %0
  %2 = atomicrmw volatile umax i64 addrspace(2)* %object, i64 %operand acquire
  br label %8

.thread2:                                         ; preds = %0
  %3 = atomicrmw volatile umax i64 addrspace(2)* %object, i64 %operand release
  br label %8

; <label>:4                                       ; preds = %0
  %5 = atomicrmw volatile umax i64 addrspace(2)* %object, i64 %operand acq_rel
  br label %8

; <label>:6                                       ; preds = %0
  %7 = atomicrmw volatile umax i64 addrspace(2)* %object, i64 %operand seq_cst
  br label %8

; <label>:8                                       ; preds = %6, %4, %.thread2, %.thread1, %.thread
  %.0 = phi i64 [ %1, %.thread ], [ %7, %6 ], [ %5, %4 ], [ %3, %.thread2 ], [ %2, %.thread1 ]
  ret i64 %.0

}

; Function Attrs: nounwind uwtable
define void @_Z24pocl_atomic_store__localPVU3AS2U7_Atomicdd12memory_order12memory_scope(double addrspace(2)* nocapture %object, double %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %5 [
    i32 0, label %.thread
    i32 1, label %.thread
    i32 2, label %.thread1
  ]

.thread1:                                         ; preds = %0
  %1 = bitcast double %desired to i64
  %2 = bitcast double addrspace(2)* %object to i64 addrspace(2)*
  store atomic volatile i64 %1, i64 addrspace(2)* %2 release, align 8
  br label %13

.thread:                                          ; preds = %0, %0
  %3 = bitcast double %desired to i64
  %4 = bitcast double addrspace(2)* %object to i64 addrspace(2)*
  br label %9

; <label>:5                                       ; preds = %0
  %6 = icmp eq i32 %order, 3
  %7 = bitcast double %desired to i64
  %8 = bitcast double addrspace(2)* %object to i64 addrspace(2)*
  br i1 %6, label %9, label %12

; <label>:9                                       ; preds = %5, %.thread
  %10 = phi i64 addrspace(2)* [ %4, %.thread ], [ %8, %5 ]
  %11 = phi i64 [ %3, %.thread ], [ %7, %5 ]
  store atomic volatile i64 %11, i64 addrspace(2)* %10 monotonic, align 8
  br label %13

; <label>:12                                      ; preds = %5
  store atomic volatile i64 %7, i64 addrspace(2)* %8 seq_cst, align 8
  br label %13

; <label>:13                                      ; preds = %12, %.thread1, %9
  ret void
}

; Function Attrs: nounwind uwtable
define double @_Z23pocl_atomic_load__localPVU3AS2U7_Atomicd12memory_order12memory_scope(double addrspace(2)* nocapture readonly %object, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %4 [
    i32 0, label %.thread
    i32 1, label %.thread1
    i32 2, label %.thread
  ]

.thread1:                                         ; preds = %0
  %1 = bitcast double addrspace(2)* %object to i64 addrspace(2)*
  %2 = load atomic volatile i64, i64 addrspace(2)* %1 acquire, align 8
  br label %12

.thread:                                          ; preds = %0, %0
  %3 = bitcast double addrspace(2)* %object to i64 addrspace(2)*
  br label %7

; <label>:4                                       ; preds = %0
  %5 = icmp eq i32 %order, 3
  %6 = bitcast double addrspace(2)* %object to i64 addrspace(2)*
  br i1 %5, label %7, label %10

; <label>:7                                       ; preds = %4, %.thread
  %8 = phi i64 addrspace(2)* [ %3, %.thread ], [ %6, %4 ]
  %9 = load atomic volatile i64, i64 addrspace(2)* %8 monotonic, align 8
  br label %12

; <label>:10                                      ; preds = %4
  %11 = load atomic volatile i64, i64 addrspace(2)* %6 seq_cst, align 8
  br label %12

; <label>:12                                      ; preds = %10, %.thread1, %7
  %.sroa.0.0 = phi i64 [ %9, %7 ], [ %11, %10 ], [ %2, %.thread1 ]
  %13 = bitcast i64 %.sroa.0.0 to double
  ret double %13
}

; Function Attrs: nounwind uwtable
define double @_Z27pocl_atomic_exchange__localPVU3AS2U7_Atomicdd12memory_order12memory_scope(double addrspace(2)* %object, double %desired, i32 %order, i32 %scope) #0 {
  switch i32 %order, label %10 [
    i32 0, label %.thread
    i32 1, label %.thread2
    i32 2, label %.thread3
  ]

.thread:                                          ; preds = %0
  %1 = bitcast double %desired to i64
  %2 = bitcast double addrspace(2)* %object to i64 addrspace(2)*
  %3 = atomicrmw volatile xchg i64 addrspace(2)* %2, i64 %1 monotonic
  br label %18

.thread2:                                         ; preds = %0
  %4 = bitcast double %desired to i64
  %5 = bitcast double addrspace(2)* %object to i64 addrspace(2)*
  %6 = atomicrmw volatile xchg i64 addrspace(2)* %5, i64 %4 acquire
  br label %18

.thread3:                                         ; preds = %0
  %7 = bitcast double %desired to i64
  %8 = bitcast double addrspace(2)* %object to i64 addrspace(2)*
  %9 = atomicrmw volatile xchg i64 addrspace(2)* %8, i64 %7 release
  br label %18

; <label>:10                                      ; preds = %0
  %11 = icmp eq i32 %order, 3
  %12 = bitcast double %desired to i64
  %13 = bitcast double addrspace(2)* %object to i64 addrspace(2)*
  br i1 %11, label %14, label %16

; <label>:14                                      ; preds = %10
  %15 = atomicrmw volatile xchg i64 addrspace(2)* %13, i64 %12 acq_rel
  br label %18

; <label>:16                                      ; preds = %10
  %17 = atomicrmw volatile xchg i64 addrspace(2)* %13, i64 %12 seq_cst
  br label %18

; <label>:18                                      ; preds = %16, %14, %.thread3, %.thread2, %.thread
  %.sroa.0.0 = phi i64 [ %3, %.thread ], [ %17, %16 ], [ %15, %14 ], [ %9, %.thread3 ], [ %6, %.thread2 ]
  %19 = bitcast i64 %.sroa.0.0 to double
  ret double %19
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z42pocl_atomic_compare_exchange_strong__localPVU3AS2U7_AtomicdPdd12memory_orderS3_12memory_scope(double addrspace(2)* %object, double* nocapture %expected, double %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = bitcast double %desired to i64
  %12 = icmp eq i32 %failure, 0
  br i1 %12, label %20, label %13

; <label>:13                                      ; preds = %9
  %14 = icmp eq i32 %failure, 1
  br i1 %14, label %20, label %15

; <label>:15                                      ; preds = %13
  %16 = icmp eq i32 %failure, 2
  br i1 %16, label %20, label %17

; <label>:17                                      ; preds = %15
  %18 = icmp eq i32 %failure, 3
  %19 = select i1 %18, i32 4, i32 5
  br label %20

; <label>:20                                      ; preds = %13, %15, %17, %9
  %21 = phi i32 [ 0, %9 ], [ 2, %13 ], [ %19, %17 ], [ 3, %15 ]
  %22 = bitcast double addrspace(2)* %object to i64 addrspace(2)*
  %23 = bitcast double* %expected to i64*
  switch i32 %10, label %31 [
    i32 1, label %24
    i32 2, label %24
    i32 3, label %53
    i32 4, label %26
    i32 5, label %28
  ]

; <label>:24                                      ; preds = %20, %20
  %.off = add nsw i32 %21, -1
  %switch = icmp ult i32 %.off, 2
  %25 = load i64, i64* %23, align 8
  br i1 %switch, label %42, label %39

; <label>:26                                      ; preds = %20
  %.off1 = add nsw i32 %21, -1
  %switch2 = icmp ult i32 %.off1, 2
  %27 = load i64, i64* %23, align 8
  br i1 %switch2, label %64, label %61

; <label>:28                                      ; preds = %20
  switch i32 %21, label %75 [
    i32 1, label %79
    i32 2, label %79
    i32 5, label %83
  ]

; <label>:29                                      ; preds = %89, %93, %97, %69, %73, %47, %51, %59, %37
  %.0 = phi i8 [ %38, %37 ], [ %90, %89 ], [ %98, %97 ], [ %94, %93 ], [ %74, %73 ], [ %70, %69 ], [ %60, %59 ], [ %52, %51 ], [ %48, %47 ]
  %30 = icmp ne i8 %.0, 0
  ret i1 %30

; <label>:31                                      ; preds = %20
  %32 = load i64, i64* %23, align 8
  %33 = cmpxchg volatile i64 addrspace(2)* %22, i64 %32, i64 %11 monotonic monotonic
  %34 = extractvalue { i64, i1 } %33, 1
  br i1 %34, label %37, label %35

; <label>:35                                      ; preds = %31
  %36 = extractvalue { i64, i1 } %33, 0
  store i64 %36, i64* %23, align 8
  br label %37

; <label>:37                                      ; preds = %35, %31
  %38 = zext i1 %34 to i8
  br label %29

; <label>:39                                      ; preds = %24
  %40 = cmpxchg volatile i64 addrspace(2)* %22, i64 %25, i64 %11 acquire monotonic
  %41 = extractvalue { i64, i1 } %40, 1
  br i1 %41, label %47, label %45

; <label>:42                                      ; preds = %24
  %43 = cmpxchg volatile i64 addrspace(2)* %22, i64 %25, i64 %11 acquire acquire
  %44 = extractvalue { i64, i1 } %43, 1
  br i1 %44, label %51, label %49

; <label>:45                                      ; preds = %39
  %46 = extractvalue { i64, i1 } %40, 0
  store i64 %46, i64* %23, align 8
  br label %47

; <label>:47                                      ; preds = %45, %39
  %48 = zext i1 %41 to i8
  br label %29

; <label>:49                                      ; preds = %42
  %50 = extractvalue { i64, i1 } %43, 0
  store i64 %50, i64* %23, align 8
  br label %51

; <label>:51                                      ; preds = %49, %42
  %52 = zext i1 %44 to i8
  br label %29

; <label>:53                                      ; preds = %20
  %54 = load i64, i64* %23, align 8
  %55 = cmpxchg volatile i64 addrspace(2)* %22, i64 %54, i64 %11 release monotonic
  %56 = extractvalue { i64, i1 } %55, 1
  br i1 %56, label %59, label %57

; <label>:57                                      ; preds = %53
  %58 = extractvalue { i64, i1 } %55, 0
  store i64 %58, i64* %23, align 8
  br label %59

; <label>:59                                      ; preds = %57, %53
  %60 = zext i1 %56 to i8
  br label %29

; <label>:61                                      ; preds = %26
  %62 = cmpxchg volatile i64 addrspace(2)* %22, i64 %27, i64 %11 acq_rel monotonic
  %63 = extractvalue { i64, i1 } %62, 1
  br i1 %63, label %69, label %67

; <label>:64                                      ; preds = %26
  %65 = cmpxchg volatile i64 addrspace(2)* %22, i64 %27, i64 %11 acq_rel acquire
  %66 = extractvalue { i64, i1 } %65, 1
  br i1 %66, label %73, label %71

; <label>:67                                      ; preds = %61
  %68 = extractvalue { i64, i1 } %62, 0
  store i64 %68, i64* %23, align 8
  br label %69

; <label>:69                                      ; preds = %67, %61
  %70 = zext i1 %63 to i8
  br label %29

; <label>:71                                      ; preds = %64
  %72 = extractvalue { i64, i1 } %65, 0
  store i64 %72, i64* %23, align 8
  br label %73

; <label>:73                                      ; preds = %71, %64
  %74 = zext i1 %66 to i8
  br label %29

; <label>:75                                      ; preds = %28
  %76 = load i64, i64* %23, align 8
  %77 = cmpxchg volatile i64 addrspace(2)* %22, i64 %76, i64 %11 seq_cst monotonic
  %78 = extractvalue { i64, i1 } %77, 1
  br i1 %78, label %89, label %87

; <label>:79                                      ; preds = %28, %28
  %80 = load i64, i64* %23, align 8
  %81 = cmpxchg volatile i64 addrspace(2)* %22, i64 %80, i64 %11 seq_cst acquire
  %82 = extractvalue { i64, i1 } %81, 1
  br i1 %82, label %93, label %91

; <label>:83                                      ; preds = %28
  %84 = load i64, i64* %23, align 8
  %85 = cmpxchg volatile i64 addrspace(2)* %22, i64 %84, i64 %11 seq_cst seq_cst
  %86 = extractvalue { i64, i1 } %85, 1
  br i1 %86, label %97, label %95

; <label>:87                                      ; preds = %75
  %88 = extractvalue { i64, i1 } %77, 0
  store i64 %88, i64* %23, align 8
  br label %89

; <label>:89                                      ; preds = %87, %75
  %90 = zext i1 %78 to i8
  br label %29

; <label>:91                                      ; preds = %79
  %92 = extractvalue { i64, i1 } %81, 0
  store i64 %92, i64* %23, align 8
  br label %93

; <label>:93                                      ; preds = %91, %79
  %94 = zext i1 %82 to i8
  br label %29

; <label>:95                                      ; preds = %83
  %96 = extractvalue { i64, i1 } %85, 0
  store i64 %96, i64* %23, align 8
  br label %97

; <label>:97                                      ; preds = %95, %83
  %98 = zext i1 %86 to i8
  br label %29
}

; Function Attrs: nounwind uwtable
define zeroext i1 @_Z40pocl_atomic_compare_exchange_weak__localPVU3AS2U7_AtomicdPdd12memory_orderS3_12memory_scope(double addrspace(2)* %object, double* nocapture %expected, double %desired, i32 %success, i32 %failure, i32 %scope) #0 {
  %1 = icmp eq i32 %success, 0
  br i1 %1, label %9, label %2

; <label>:2                                       ; preds = %0
  %3 = icmp eq i32 %success, 1
  br i1 %3, label %9, label %4

; <label>:4                                       ; preds = %2
  %5 = icmp eq i32 %success, 2
  br i1 %5, label %9, label %6

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 %success, 3
  %8 = select i1 %7, i32 4, i32 5
  br label %9

; <label>:9                                       ; preds = %2, %4, %6, %0
  %10 = phi i32 [ 0, %0 ], [ 2, %2 ], [ %8, %6 ], [ 3, %4 ]
  %11 = bitcast double %desired to i64
  %12 = icmp eq i32 %failure, 0
  br i1 %12, label %20, label %13

; <label>:13                                      ; preds = %9
  %14 = icmp eq i32 %failure, 1
  br i1 %14, label %20, label %15

; <label>:15                                      ; preds = %13
  %16 = icmp eq i32 %failure, 2
  br i1 %16, label %20, label %17

; <label>:17                                      ; preds = %15
  %18 = icmp eq i32 %failure, 3
  %19 = select i1 %18, i32 4, i32 5
  br label %20

; <label>:20                                      ; preds = %13, %15, %17, %9
  %21 = phi i32 [ 0, %9 ], [ 2, %13 ], [ %19, %17 ], [ 3, %15 ]
  %22 = bitcast double addrspace(2)* %object to i64 addrspace(2)*
  %23 = bitcast double* %expected to i64*
  switch i32 %10, label %31 [
    i32 1, label %24
    i32 2, label %24
    i32 3, label %53
    i32 4, label %26
    i32 5, label %28
  ]

; <label>:24                                      ; preds = %20, %20
  %.off = add nsw i32 %21, -1
  %switch = icmp ult i32 %.off, 2
  %25 = load i64, i64* %23, align 8
  br i1 %switch, label %42, label %39

; <label>:26                                      ; preds = %20
  %.off1 = add nsw i32 %21, -1
  %switch2 = icmp ult i32 %.off1, 2
  %27 = load i64, i64* %23, align 8
  br i1 %switch2, label %64, label %61

; <label>:28                                      ; preds = %20
  switch i32 %21, label %75 [
    i32 1, label %79
    i32 2, label %79
    i32 5, label %83
  ]

; <label>:29                                      ; preds = %89, %93, %97, %69, %73, %47, %51, %59, %37
  %.0 = phi i8 [ %38, %37 ], [ %90, %89 ], [ %98, %97 ], [ %94, %93 ], [ %74, %73 ], [ %70, %69 ], [ %60, %59 ], [ %52, %51 ], [ %48, %47 ]
  %30 = icmp ne i8 %.0, 0
  ret i1 %30

; <label>:31                                      ; preds = %20
  %32 = load i64, i64* %23, align 8
  %33 = cmpxchg weak volatile i64 addrspace(2)* %22, i64 %32, i64 %11 monotonic monotonic
  %34 = extractvalue { i64, i1 } %33, 1
  br i1 %34, label %37, label %35

; <label>:35                                      ; preds = %31
  %36 = extractvalue { i64, i1 } %33, 0
  store i64 %36, i64* %23, align 8
  br label %37

; <label>:37                                      ; preds = %35, %31
  %38 = zext i1 %34 to i8
  br label %29

; <label>:39                                      ; preds = %24
  %40 = cmpxchg weak volatile i64 addrspace(2)* %22, i64 %25, i64 %11 acquire monotonic
  %41 = extractvalue { i64, i1 } %40, 1
  br i1 %41, label %47, label %45

; <label>:42                                      ; preds = %24
  %43 = cmpxchg weak volatile i64 addrspace(2)* %22, i64 %25, i64 %11 acquire acquire
  %44 = extractvalue { i64, i1 } %43, 1
  br i1 %44, label %51, label %49

; <label>:45                                      ; preds = %39
  %46 = extractvalue { i64, i1 } %40, 0
  store i64 %46, i64* %23, align 8
  br label %47

; <label>:47                                      ; preds = %45, %39
  %48 = zext i1 %41 to i8
  br label %29

; <label>:49                                      ; preds = %42
  %50 = extractvalue { i64, i1 } %43, 0
  store i64 %50, i64* %23, align 8
  br label %51

; <label>:51                                      ; preds = %49, %42
  %52 = zext i1 %44 to i8
  br label %29

; <label>:53                                      ; preds = %20
  %54 = load i64, i64* %23, align 8
  %55 = cmpxchg weak volatile i64 addrspace(2)* %22, i64 %54, i64 %11 release monotonic
  %56 = extractvalue { i64, i1 } %55, 1
  br i1 %56, label %59, label %57

; <label>:57                                      ; preds = %53
  %58 = extractvalue { i64, i1 } %55, 0
  store i64 %58, i64* %23, align 8
  br label %59

; <label>:59                                      ; preds = %57, %53
  %60 = zext i1 %56 to i8
  br label %29

; <label>:61                                      ; preds = %26
  %62 = cmpxchg weak volatile i64 addrspace(2)* %22, i64 %27, i64 %11 acq_rel monotonic
  %63 = extractvalue { i64, i1 } %62, 1
  br i1 %63, label %69, label %67

; <label>:64                                      ; preds = %26
  %65 = cmpxchg weak volatile i64 addrspace(2)* %22, i64 %27, i64 %11 acq_rel acquire
  %66 = extractvalue { i64, i1 } %65, 1
  br i1 %66, label %73, label %71

; <label>:67                                      ; preds = %61
  %68 = extractvalue { i64, i1 } %62, 0
  store i64 %68, i64* %23, align 8
  br label %69

; <label>:69                                      ; preds = %67, %61
  %70 = zext i1 %63 to i8
  br label %29

; <label>:71                                      ; preds = %64
  %72 = extractvalue { i64, i1 } %65, 0
  store i64 %72, i64* %23, align 8
  br label %73

; <label>:73                                      ; preds = %71, %64
  %74 = zext i1 %66 to i8
  br label %29

; <label>:75                                      ; preds = %28
  %76 = load i64, i64* %23, align 8
  %77 = cmpxchg weak volatile i64 addrspace(2)* %22, i64 %76, i64 %11 seq_cst monotonic
  %78 = extractvalue { i64, i1 } %77, 1
  br i1 %78, label %89, label %87

; <label>:79                                      ; preds = %28, %28
  %80 = load i64, i64* %23, align 8
  %81 = cmpxchg weak volatile i64 addrspace(2)* %22, i64 %80, i64 %11 seq_cst acquire
  %82 = extractvalue { i64, i1 } %81, 1
  br i1 %82, label %93, label %91

; <label>:83                                      ; preds = %28
  %84 = load i64, i64* %23, align 8
  %85 = cmpxchg weak volatile i64 addrspace(2)* %22, i64 %84, i64 %11 seq_cst seq_cst
  %86 = extractvalue { i64, i1 } %85, 1
  br i1 %86, label %97, label %95

; <label>:87                                      ; preds = %75
  %88 = extractvalue { i64, i1 } %77, 0
  store i64 %88, i64* %23, align 8
  br label %89

; <label>:89                                      ; preds = %87, %75
  %90 = zext i1 %78 to i8
  br label %29

; <label>:91                                      ; preds = %79
  %92 = extractvalue { i64, i1 } %81, 0
  store i64 %92, i64* %23, align 8
  br label %93

; <label>:93                                      ; preds = %91, %79
  %94 = zext i1 %82 to i8
  br label %29

; <label>:95                                      ; preds = %83
  %96 = extractvalue { i64, i1 } %85, 0
  store i64 %96, i64* %23, align 8
  br label %97

; <label>:97                                      ; preds = %95, %83
  %98 = zext i1 %86 to i8
  br label %29
}

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

