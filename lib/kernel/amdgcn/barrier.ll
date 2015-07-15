; ModuleID = 'kernel-amdgcn--amdhsa.bc'
target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p24:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--amdhsa"

; Borrowed from libclc.

declare void @llvm.AMDGPU.barrier.local() nounwind noduplicate
declare void @llvm.AMDGPU.barrier.global() nounwind noduplicate

define void @_Z7barrierj(i32 %flags) nounwind noduplicate alwaysinline {
barrier_local_test:
  %CLK_LOCAL_MEM_FENCE = call i32 @__clc_clk_local_mem_fence()
  %0 = and i32 %flags, %CLK_LOCAL_MEM_FENCE
  %1 = icmp ne i32 %0, 0
  br i1 %1, label %barrier_local, label %barrier_global_test

barrier_local:
  call void @llvm.AMDGPU.barrier.local() noduplicate
  br label %barrier_global_test

barrier_global_test:
  %CLK_GLOBAL_MEM_FENCE = call i32 @__clc_clk_global_mem_fence()
  %2 = and i32 %flags, %CLK_GLOBAL_MEM_FENCE
  %3 = icmp ne i32 %2, 0
  br i1 %3, label %barrier_global, label %done

barrier_global:
  call void @llvm.AMDGPU.barrier.global() noduplicate
  br label %done

done:
  ret void
}

define i32 @__clc_clk_local_mem_fence() nounwind noduplicate alwaysinline {
entry:
  ret i32 1
}

define i32 @__clc_clk_global_mem_fence() nounwind noduplicate alwaysinline {
entry:
  ret i32 2
}

