declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()

declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()

define i32 @get_nvvm_tid_x() {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  ret i32 %0
}

define i32 @get_nvvm_tid_y() {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  ret i32 %0
}

define i32 @get_nvvm_tid_z() {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  ret i32 %0
}

define i32 @get_nvvm_ntid_x() {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  ret i32 %0
}

define i32 @get_nvvm_ntid_y() {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  ret i32 %0
}

define i32 @get_nvvm_ntid_z() {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  ret i32 %0
}

define i32 @get_nvvm_ctaid_x() {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  ret i32 %0
}

define i32 @get_nvvm_ctaid_y() {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  ret i32 %0
}

define i32 @get_nvvm_ctaid_z() {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  ret i32 %0
}

define i32 @get_nvvm_nctaid_x() {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  ret i32 %0
}

define i32 @get_nvvm_nctaid_y() {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  ret i32 %0
}

define i32 @get_nvvm_nctaid_z() {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  ret i32 %0
}

declare float @__nv_frexpf(float %a, i32* %b)

define { float, i32 } @frexpf_f32_i32(float %a) {
entry:
  %0 = alloca i32, align 4
  %1 = call float @__nv_frexpf(float %a, i32* %0)
  %2 = load i32, i32* %0
  %3 = insertvalue { float, i32 } undef, float %1, 0
  %4 = insertvalue { float, i32 } %3, i32 %2, 1
  ret { float, i32 } %4
}

declare double @__nv_frexp(double %a, i32* %b)

define { double, i32 } @frexp_f64_i32(double %a) {
entry:
  %0 = alloca i32, align 4
  %1 = call double @__nv_frexp(double %a, i32* %0)
  %2 = load i32, i32* %0
  %3 = insertvalue { double, i32 } undef, double %1, 0
  %4 = insertvalue { double, i32 } %3, i32 %2, 1
  ret { double, i32 } %4
}
