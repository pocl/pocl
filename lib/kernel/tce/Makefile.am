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



KERNEL_TARGET = tce-tut-llvm

# Use TARGET flags:
CLANG_FLAGS = @TARGET_CLANG_FLAGS@ -Xclang -ffake-address-space-map -Xclang -menable-no-nans -emit-llvm -ffp-contract=off -target tce-tut-llvm  -isystem `tce-config --prefix`/tce-llvm/include
LLC_FLAGS   = @TARGET_LLC_FLAGS@
LD_FLAGS    = @TARGET_LD_FLAGS@
DEVICE_CL_FLAGS = -D__OPENCL_VERSION__=200 @TCE_DEVICE_EXTENSION_DEFINES@

include ../rules.mk
include ../sources.mk

# Use the libc (newlib) printf() for now because the pocl's
# fails in TCE code gen:
# SplitVectorResult #0: 0x24c5ae0: v8i16,ch = vaarg 0x20628e8, 0x2423ed0, 0x24c59e0, 0x24c4fe0 [ORD=223] [ID=0]
# LLVM ERROR: Do not know how to split the result of this operator!
LKERNEL_SRCS_EXCLUDE = printf.c svm_atomics_host.cl atomics.cl
LKERNEL_SRCS_EXTRA = printf_constant.c

EXTRA_DIST = CMakeLists.txt

