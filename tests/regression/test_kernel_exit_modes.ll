; LLVM IR source for the kernel exit modes regression test.
; Not compiled by the build system; kept for reference / regeneration.
;
; To regenerate the SPIR-V binary:
;   llvm-as test_kernel_exit_modes.ll -o /tmp/test_kernel_exit_modes.bc
;   llvm-spirv /tmp/test_kernel_exit_modes.bc -o test_kernel_exit_modes.spv

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-G1"
target triple = "spir64-unknown-unknown"

; Declarations
declare void @__pocl_trap() cold noreturn nounwind
declare void @__pocl_exit() cold noreturn nounwind

; Trivial wrapper with no side effects.  do_something() has no unreachable,
; so flatten-globals does NOT force-inline it — the call survives to UTR.
; However, function-attrs infers willreturn (the body is just ret void), so
; UTR classifies the unreachable block as pure dead code and deletes it.
; Result: kernel completes normally (CL_COMPLETE).
define spir_func void @do_something() noinline nounwind {
  ret void
}

; Wrapper around __pocl_trap
define spir_func void @wrapper_trap() noinline nounwind {
  call void @__pocl_trap()
  ret void
}

; Wrapper around __pocl_exit
define spir_func void @wrapper_exit() noinline nounwind {
  call void @__pocl_exit()
  ret void
}

; Kernel 1: unreachable after willreturn call (UTR deletes block) -> CL_COMPLETE
define spir_kernel void @test_unreachable(i64 %n) !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_type_qual !8 !kernel_arg_base_type !7 {
entry:
  %cmp = icmp slt i64 %n, 1
  br i1 %cmp, label %bad, label %good
bad:
  call spir_func void @do_something()
  unreachable
good:
  ret void
}

; Kernel 2: __pocl_trap through noinline wrapper -> CL_FAILED
define spir_kernel void @test_trap(i64 %n) !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_type_qual !8 !kernel_arg_base_type !7 {
entry:
  %cmp = icmp slt i64 %n, 1
  br i1 %cmp, label %bad, label %good
bad:
  call spir_func void @wrapper_trap()
  unreachable
good:
  ret void
}

; Kernel 3: __pocl_exit through noinline wrapper -> CL_SUCCESS
define spir_kernel void @test_exit(i64 %n) !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_type_qual !8 !kernel_arg_base_type !7 {
entry:
  %cmp = icmp slt i64 %n, 1
  br i1 %cmp, label %bad, label %good
bad:
  call spir_func void @wrapper_exit()
  unreachable
good:
  ret void
}

declare spir_func i32 @printf(ptr addrspace(2), ...) nounwind

@.str = internal unnamed_addr addrspace(2) constant [1 x i8] c"\00", align 1

; Side-effecting wrapper: calls printf (not willreturn), then returns.
; No unreachable -> not force-inlined by flatten-globals.
; function-attrs cannot infer willreturn (printf is an external without it).
; UTR sees the call as potentially non-returning -> preserves block -> CL_FAILED.
; Uses an empty format string so no output is produced.
define spir_func void @do_something_with_sideeffect() noinline nounwind {
  %ret = call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @.str)
  ret void
}

; Kernel 4: unreachable after side-effecting call (UTR preserves) -> CL_FAILED
define spir_kernel void @test_unreachable_sideeffect(i64 %n) !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_type_qual !8 !kernel_arg_base_type !7 {
entry:
  %cmp = icmp slt i64 %n, 1
  br i1 %cmp, label %bad, label %good
bad:
  call spir_func void @do_something_with_sideeffect()
  unreachable
good:
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
!5 = !{i32 0}
!6 = !{!"none"}
!7 = !{!"long"}
!8 = !{!""}
