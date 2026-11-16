; OpenCL built-in library: device-side printf buffer allocation for the Metal
; (AIR) target
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

; PoCL's linker replaces printf() calls with code that stores the format
; string and the arguments to the memory returned by pocl_printf_alloc_stub.
; On Metal, the printf buffer is a device buffer that the driver passes to
; every kernel; the AIR lowering pass replaces the @_pocl_air_printf_buffer
; placeholder with that kernel argument. Its layout is:
;
;   uint position;  // bytes used so far, advanced atomically
;   uint capacity;  // bytes available for data, set by the host
;   char data[capacity];
;
; These functions use the default calling convention (not spir_func, like
; the rest of the library), because the linker emits the calls to the stub
; with the default one.

@_pocl_air_printf_buffer = external global ptr addrspace(1)

define ptr addrspace(1) @pocl_printf_alloc(ptr addrspace(1) %buf, i32 %bytes) noinline nounwind {
entry:
  %pos = atomicrmw add ptr addrspace(1) %buf, i32 %bytes monotonic, align 4
  %cap.ptr = getelementptr inbounds i8, ptr addrspace(1) %buf, i64 4
  %cap = load i32, ptr addrspace(1) %cap.ptr, align 4
  %end = add i32 %pos, %bytes
  %fits = icmp ule i32 %end, %cap
  %overflow = icmp ult i32 %end, %pos
  %nooverflow = xor i1 %overflow, true
  %ok = and i1 %fits, %nooverflow
  br i1 %ok, label %alloc, label %full

alloc:
  %data = getelementptr inbounds i8, ptr addrspace(1) %buf, i64 8
  %pos64 = zext i32 %pos to i64
  %ret = getelementptr inbounds i8, ptr addrspace(1) %data, i64 %pos64
  ret ptr addrspace(1) %ret

full:
  ret ptr addrspace(1) null
}

define ptr addrspace(1) @pocl_printf_alloc_stub(i32 %bytes) noinline nounwind {
entry:
  %buf = load ptr addrspace(1), ptr @_pocl_air_printf_buffer, align 8
  %ret = call ptr addrspace(1) @pocl_printf_alloc(ptr addrspace(1) %buf, i32 %bytes)
  ret ptr addrspace(1) %ret
}
