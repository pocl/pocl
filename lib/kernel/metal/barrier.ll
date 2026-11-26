; OpenCL built-in library: barriers and memory fences for the Metal (AIR) target
;
; Copyright (c) 2026 PoCL developers
;
; Permission is hereby granted, free of charge, to any person obtaining a copy
; of this software and associated documentation files (the "Software"), to
; deal in the Software without restriction, including without limitation the
; rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
; sell copies of the Software, and to permit persons to whom the Software is
; furnished to do so, subject to the following conditions:
;
; The above copyright notice and this permission notice shall be included in
; all copies or substantial portions of the Software.
;
; THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
; IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
; FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
; AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
; LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
; FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
; IN THE SOFTWARE.

; Work-group barriers call @pocl.barrier (the one undefined function PoCL's
; linker accepts for this), which the Metal driver's AIR lowering pass turns
; into an air.wg.barrier ordering both device and threadgroup memory.
; Memory fences are LLVM fences, which the pass lowers to air.atomic.fence.
;
; The functions use the spir_func calling convention, because the kernel
; library and the user programs are compiled for the spir64 target.

declare void @pocl.barrier() convergent nounwind

define spir_func void @_Z7barrierj(i32 %flags) convergent nounwind alwaysinline {
entry:
  call void @pocl.barrier()
  ret void
}

define spir_func void @_Z22_cl_work_group_barrierj(i32 %flags) convergent nounwind alwaysinline {
entry:
  call void @pocl.barrier()
  ret void
}

define spir_func void @_Z22_cl_work_group_barrierj12memory_scope(i32 %flags, i32 %scope) convergent nounwind alwaysinline {
entry:
  call void @pocl.barrier()
  ret void
}

define spir_func void @_Z18_cl_read_mem_fencej(i32 %flags) nounwind alwaysinline {
entry:
  fence syncscope("workgroup") seq_cst
  ret void
}

define spir_func void @_Z19_cl_write_mem_fencej(i32 %flags) nounwind alwaysinline {
entry:
  fence syncscope("workgroup") seq_cst
  ret void
}

define spir_func void @_Z13_cl_mem_fencej(i32 %flags) nounwind alwaysinline {
entry:
  fence syncscope("workgroup") seq_cst
  ret void
}
