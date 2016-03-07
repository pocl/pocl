# Process this file with automake to produce Makefile.in
# 
# Copyright (c) 2011 Universidad Rey Juan Carlos
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.



KERNEL_TARGET = @OCL_KERNEL_TARGET@

# Use HOST flags:
CLANG_FLAGS = @HOST_CLANG_FLAGS@ -Xclang -ffake-address-space-map -emit-llvm -ffp-contract=off
LLC_FLAGS   = @HOST_LLC_FLAGS@
LD_FLAGS    = @HOST_LD_FLAGS@
DEVICE_CL_FLAGS = -D__OPENCL_VERSION__=200 @HOST_DEVICE_EXTENSION_DEFINES@

include ../rules.mk
include ../sources.mk
if USE_VECMATHLIB
include ../sources-vml.mk
endif

if HOST_CPU_IS_X86_64
  LKERNEL_SRCS_EXTRA2 = svm_atomics_x86_64.ll svm_atomics.cl
else
  LKERNEL_SRCS_EXTRA2 = svm_atomics_host.cl svm_atomics.cl
endif



EXTRA_DIST = CMakeLists.txt
