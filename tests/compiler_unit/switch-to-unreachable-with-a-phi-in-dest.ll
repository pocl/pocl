; ModuleID = 'switch-to-unreachable-with-a-phi-in-dest'
;
; Derived from a Julia GPU kernel (getindex on Symmetric matrix) that
; originally had noreturn helper functions called before unreachable
; terminators. Those calls are removed here so that the unreachable
; blocks are pure dead code that UTR can delete, allowing the switch
; to be eliminated. The test verifies switch/PHI fix-up when cases
; are pruned, not error-handling path preservation.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-G1"
target triple = "spir64-unknown-unknown"

%structtype.0 = type { %structtype.1, i32 }
%structtype.1 = type { ptr addrspace(1), i64, [2 x i64], i64 }
%structtype.2 = type { %structtype.3 }
%structtype.3 = type { [1 x [1 x [1 x i64]]], [2 x [1 x [1 x [1 x i64]]]] }
%structtype.4 = type { %structtype }
%structtype = type { ptr addrspace(1), i64, [1 x i64], i64 }
%structtype.5 = type { %structtype.0 }
%structtype.6 = type { [1 x i64] }

; Function Attrs: nounwind
define internal spir_func void @julia_getindex_generated_19034(ptr readonly align 8 captures(none) %0, ptr readonly align 8 captures(none) %1, i64 signext %2, ptr readonly align 8 captures(none) %3) #0 {
top:
  %4 = alloca [2 x i64], align 8
  %5 = bitcast ptr %3 to ptr
  %.sroa.023.sroa.0.0.copyload26 = load ptr addrspace(1), ptr %5, align 8
  %6 = add i64 %2, -1
  %.unpack.elt = getelementptr inbounds [1 x [2 x i64]], ptr addrspace(1) %.sroa.023.sroa.0.0.copyload26, i64 %6, i64 0, i64 0
  %.unpack.unpack = load i64, ptr addrspace(1) %.unpack.elt, align 8
  %.unpack.elt27 = getelementptr inbounds [1 x [2 x i64]], ptr addrspace(1) %.sroa.023.sroa.0.0.copyload26, i64 %6, i64 0, i64 1
  %.unpack.unpack28 = load i64, ptr addrspace(1) %.unpack.elt27, align 8
  %7 = getelementptr inbounds [2 x i64], ptr %4, i64 0, i64 0
  store i64 %.unpack.unpack, ptr %7, align 8
  %8 = getelementptr inbounds [2 x i64], ptr %4, i64 0, i64 1
  store i64 %.unpack.unpack28, ptr %8, align 8
  %9 = getelementptr inbounds %structtype.0, ptr %1, i64 0, i32 0, i32 2, i64 0
  %10 = getelementptr inbounds %structtype.0, ptr %1, i64 0, i32 0, i32 2, i64 1
  %11 = add i64 %.unpack.unpack, -1
  %12 = load i64, ptr %9, align 8
  %13 = icmp uge i64 %11, %12
  %14 = add i64 %.unpack.unpack28, -1
  %15 = load i64, ptr %10, align 8
  %16 = icmp uge i64 %14, %15
  %.not31 = or i1 %13, %16
  br i1 %.not31, label %L79, label %L82

L79:                                              ; preds = %top
  unreachable

L82:                                              ; preds = %top
  %.not = icmp eq i64 %.unpack.unpack, %.unpack.unpack28
  br i1 %.not, label %L84, label %L124

L84:                                              ; preds = %L82
  %17 = mul i64 %12, %14
  %18 = add i64 %11, %17
  %19 = bitcast ptr %1 to ptr
  %20 = load ptr addrspace(1), ptr %19, align 8
  %21 = getelementptr inbounds float, ptr addrspace(1) %20, i64 %18
  %22 = getelementptr inbounds %structtype.0, ptr %1, i64 0, i32 1
  %23 = load i32, ptr %22, align 8
  switch i32 %23, label %L121 [
    i32 1426063360, label %L196
    i32 1275068416, label %L196
  ]

L121:                                             ; preds = %L84
  unreachable

L124:                                             ; preds = %L82
  %24 = getelementptr inbounds %structtype.0, ptr %1, i64 0, i32 1
  %25 = load i32, ptr %24, align 8
  %26 = icmp eq i32 %25, 1426063360
  %27 = icmp slt i64 %.unpack.unpack, %.unpack.unpack28
  %28 = icmp ne i1 %27, %26
  %29 = bitcast ptr %1 to ptr
  %30 = load ptr addrspace(1), ptr %29, align 8
  br i1 %28, label %L158, label %L131

L131:                                             ; preds = %L124
  %31 = mul i64 %12, %14
  %32 = add i64 %11, %31
  %33 = getelementptr inbounds float, ptr addrspace(1) %30, i64 %32
  br label %L196

L158:                                             ; preds = %L124
  %34 = mul i64 %12, %11
  %35 = add i64 %14, %34
  %36 = getelementptr inbounds float, ptr addrspace(1) %30, i64 %35
  br label %L196

L196:                                             ; preds = %L158, %L131, %L84, %L84
  %value_phi.in = phi ptr addrspace(1) [ %33, %L131 ], [ %36, %L158 ], [ %21, %L84 ], [ %21, %L84 ]
  %value_phi = load float, ptr addrspace(1) %value_phi.in, align 4
  %37 = bitcast ptr %0 to ptr
  %38 = load ptr addrspace(1), ptr %37, align 8
  %39 = getelementptr inbounds float, ptr addrspace(1) %38, i64 %6
  store float %value_phi, ptr addrspace(1) %39, align 4
  ret void
}

; Function Attrs: nounwind
define spir_kernel void @_Z19gpu_getindex_kernel16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi1E5TupleI5OneToI5Int64EEE7NDRangeILi1ES0_S0_S8_S8_EE13CLDeviceArrayI7Float32Li1ELi1EE9SymmetricISD_SC_ISD_Li2ELi1EEES3_IS5_ESC_I14CartesianIndexILi2EELi1ELi1EE(ptr byval(%structtype.2) %0, ptr byval(%structtype.4) %1, ptr byval(%structtype.5) %2, ptr byval(%structtype.6) %3, ptr byval(%structtype.4) %4) #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_type_qual !7 !kernel_arg_base_type !7 !spirv.ParameterDecorations !8 {
conversion:
  %Is = alloca [1 x %structtype], align 8
  %.sroa.0.0..sroa_cast = bitcast ptr %4 to ptr
  %Is56 = bitcast ptr %Is to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %Is56, ptr align 8 %.sroa.0.0..sroa_cast, i64 32, i1 false)
  %5 = call spir_func i64 @_Z12get_group_idj(i32 0) #2
  %6 = insertelement <3 x i64> poison, i64 %5, i32 0
  %7 = call spir_func i64 @_Z12get_group_idj(i32 1) #2
  %8 = insertelement <3 x i64> %6, i64 %7, i32 1
  %9 = call spir_func i64 @_Z12get_group_idj(i32 2) #2
  %10 = insertelement <3 x i64> %8, i64 %9, i32 2
  %11 = extractelement <3 x i64> %10, i32 0
  %12 = select i1 true, i64 %11, i64 0
  %13 = call spir_func i64 @_Z12get_local_idj(i32 0) #2
  %14 = insertelement <3 x i64> poison, i64 %13, i32 0
  %15 = call spir_func i64 @_Z12get_local_idj(i32 1) #2
  %16 = insertelement <3 x i64> %14, i64 %15, i32 1
  %17 = call spir_func i64 @_Z12get_local_idj(i32 2) #2
  %18 = insertelement <3 x i64> %16, i64 %17, i32 2
  %19 = extractelement <3 x i64> %18, i32 0
  %20 = select i1 true, i64 %19, i64 0
  %21 = add i64 %20, 1
  %22 = getelementptr inbounds %structtype.2, ptr %0, i64 0, i32 0, i32 1, i64 0, i64 0, i64 0, i64 0
  %23 = load i64, ptr %22, align 8
  %24 = icmp sgt i64 %23, 0
  %25 = getelementptr inbounds %structtype.2, ptr %0, i64 0, i32 0, i32 1, i64 1, i64 0, i64 0, i64 0
  %26 = load i64, ptr %25, align 8
  %27 = icmp sgt i64 %26, 0
  %28 = mul i64 %26, %12
  %29 = add i64 %21, %28
  %30 = icmp slt i64 %29, 1
  %31 = getelementptr inbounds %structtype.2, ptr %0, i64 0, i32 0, i32 0, i64 0, i64 0, i64 0
  %32 = load i64, ptr %31, align 8
  %33 = icmp sgt i64 %29, %32
  %.not4 = or i1 %30, %33
  br i1 %.not4, label %L133, label %L125

L125:                                             ; preds = %conversion
  %34 = getelementptr inbounds %structtype.6, ptr %3, i64 0, i32 0
  %35 = getelementptr inbounds %structtype.5, ptr %2, i64 0, i32 0
  %36 = getelementptr inbounds %structtype.4, ptr %1, i64 0, i32 0
  %37 = call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %38 = insertelement <3 x i64> poison, i64 %37, i32 0
  %39 = call spir_func i64 @_Z13get_global_idj(i32 1) #2
  %40 = insertelement <3 x i64> %38, i64 %39, i32 1
  %41 = call spir_func i64 @_Z13get_global_idj(i32 2) #2
  %42 = insertelement <3 x i64> %40, i64 %41, i32 2
  %43 = extractelement <3 x i64> %42, i32 0
  %44 = select i1 true, i64 %43, i64 0
  %45 = add i64 %44, 1
  %46 = getelementptr inbounds [1 x %structtype], ptr %Is, i64 0, i64 0
  call spir_func void @julia_getindex_generated_19034(ptr readonly align 8 captures(none) %36, ptr readonly align 8 captures(none) %35, i64 signext %45, ptr readonly align 8 captures(none) %46) #0
  br label %L133

L133:                                             ; preds = %L125, %conversion
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z12get_group_idj(i32) #2

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z12get_local_idj(i32) #2

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z13get_global_idj(i32) #2

attributes #0 = { nounwind }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind willreturn memory(none) }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!2}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!3}
!spirv.Generator = !{!4}

!0 = !{i32 2, i32 2}
!1 = !{i32 3, i32 200000}
!2 = !{i32 2, i32 0}
!3 = !{}
!4 = !{i16 6, i16 14}
!5 = !{i32 0, i32 0, i32 0, i32 0, i32 0}
!6 = !{!"none", !"none", !"none", !"none", !"none"}
!7 = !{!"", !"", !"", !"", !""}
!8 = !{!9, !9, !9, !9, !9}
!9 = !{!10}
!10 = !{i32 38, i32 2}
