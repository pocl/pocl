; ModuleID = 'kernel-amdgcn--amdhsa.bc'
target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p24:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--amdhsa"

; Borrowed from libclc.

declare i32 @llvm.AMDGPU.read.workdim() nounwind readnone

define i32 @_Z13get_work_dimv() nounwind readnone alwaysinline {
  %x = call i32 @llvm.AMDGPU.read.workdim() nounwind readnone , !range !0
  ret i32 %x
}

!0 = !{ i32 1, i32 4 }
