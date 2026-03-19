; LLVM IR source for the trap-vectorization compiler unit test.
; Not compiled by the build system; kept for reference / regeneration.
;
; To regenerate the SPIR-V binary:
;   llvm-as trap-vectorization.ll -o /tmp/trap-vectorization.bc
;   llvm-spirv /tmp/trap-vectorization.bc -o trap-vectorization.spirv
;
; These kernels test that __pocl_trap() flag stores do not prevent LLVM's
; loop vectorizer from vectorizing the WI loop in WorkitemLoops.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-G1"
target triple = "spir64-unknown-unknown"

declare spir_func i64 @_Z13get_global_idj(i32) nounwind readnone
declare void @__pocl_trap() cold noreturn nounwind

;; ============================================================================
;; baseline: c[gid] = a[gid] + b[gid]
;; Control case: pure compute, must always vectorize.
;; ============================================================================
define spir_kernel void @baseline(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c) !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_type_qual !8 !kernel_arg_base_type !7 {
entry:
  %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
  %pa = getelementptr float, ptr addrspace(1) %a, i64 %gid
  %va = load float, ptr addrspace(1) %pa, align 4
  %pb = getelementptr float, ptr addrspace(1) %b, i64 %gid
  %vb = load float, ptr addrspace(1) %pb, align 4
  %sum = fadd float %va, %vb
  %pc = getelementptr float, ptr addrspace(1) %c, i64 %gid
  store float %sum, ptr addrspace(1) %pc, align 4
  ret void
}

;; ============================================================================
;; trap_nonuniform: if (gid >= n) __pocl_trap(); out[gid] = 1.0
;; Single non-uniform trap site. The flag store to __pocl_context_unreachable
;; must not prevent vectorization. Requires:
;;   - Shared flag-store block (ConvertPoclExit)
;;   - Preserved !llvm.access.group metadata (Workgroup::privatizeGlobalStores)
;; ============================================================================
define spir_kernel void @trap_nonuniform(ptr addrspace(1) %out, i64 %n) !kernel_arg_addr_space !9 !kernel_arg_access_qual !10 !kernel_arg_type !11 !kernel_arg_type_qual !12 !kernel_arg_base_type !11 {
entry:
  %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
  %cmp = icmp uge i64 %gid, %n
  br i1 %cmp, label %do_trap, label %compute

do_trap:
  call void @__pocl_trap()
  unreachable

compute:
  %pout = getelementptr float, ptr addrspace(1) %out, i64 %gid
  store float 1.0, ptr addrspace(1) %pout, align 4
  ret void
}

;; ============================================================================
;; multi_trap: two trap sites + compute
;; if (gid >= n) __pocl_trap(); if (a[gid] < 0) __pocl_trap(); out[gid] = a[gid]*2
;; Multiple non-uniform trap sites. All must share a single flag-store block
;; to avoid "write to a loop invariant address could not be vectorized".
;; ============================================================================
define spir_kernel void @multi_trap(ptr addrspace(1) %a, ptr addrspace(1) %out, i64 %n) !kernel_arg_addr_space !13 !kernel_arg_access_qual !14 !kernel_arg_type !15 !kernel_arg_type_qual !16 !kernel_arg_base_type !15 {
entry:
  %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
  %bounds_cmp = icmp uge i64 %gid, %n
  br i1 %bounds_cmp, label %bounds_trap, label %check_val

bounds_trap:
  call void @__pocl_trap()
  unreachable

check_val:
  %pa = getelementptr float, ptr addrspace(1) %a, i64 %gid
  %val = load float, ptr addrspace(1) %pa, align 4
  %val_cmp = fcmp olt float %val, 0.0
  br i1 %val_cmp, label %val_trap, label %compute

val_trap:
  call void @__pocl_trap()
  unreachable

compute:
  %result = fmul float %val, 2.0
  %pout = getelementptr float, ptr addrspace(1) %out, i64 %gid
  store float %result, ptr addrspace(1) %pout, align 4
  ret void
}

!spirv.MemoryModel = !{!0}
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

; baseline: 3 pointer args
!5 = !{i32 1, i32 1, i32 1}
!6 = !{!"none", !"none", !"none"}
!7 = !{!"float*", !"float*", !"float*"}
!8 = !{!"", !"", !""}

; trap_nonuniform: 1 pointer + 1 scalar
!9 = !{i32 1, i32 0}
!10 = !{!"none", !"none"}
!11 = !{!"float*", !"long"}
!12 = !{!"", !""}

; multi_trap: 2 pointers + 1 scalar
!13 = !{i32 1, i32 1, i32 0}
!14 = !{!"none", !"none", !"none"}
!15 = !{!"float*", !"float*", !"long"}
!16 = !{!"", !"", !""}
