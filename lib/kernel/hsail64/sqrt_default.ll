target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:64:64-p8:32:32-p9:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "hsail64"

; This is taken from tests/CodeGen/HSAIL/llvm.hsail.fsqrt.ll

declare float @llvm.hsail.fsqrt.f32(i1, i32, float) #0
declare double @llvm.hsail.fsqrt.f64(i1, i32, double) #0

define float @_Z17_sqrt_default_f32f(float %x) #0 {
  %ret = call float @llvm.hsail.fsqrt.f32(i1 false, i32 2, float %x) #0
  ret float %ret
}

define double @_Z17_sqrt_default_f64d(double %x) #0 {
  %ret = call double @llvm.hsail.fsqrt.f64(i1 false, i32 2, double %x) #0
  ret double %ret
}

attributes #0 = { nounwind readnone }
