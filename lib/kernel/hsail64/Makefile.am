# Process this file with automake to produce Makefile.in
#
# Copyright (c) 2015 Pekka Jääskeläinen
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

KERNEL_TARGET = hsail64

# Use TARGET flags:
CLANG_FLAGS = @TARGET_CLANG_FLAGS@ -Xclang -ffake-address-space-map -emit-llvm -target ${KERNEL_TARGET}
LLC_FLAGS   = @TARGET_LLC_FLAGS@
LD_FLAGS    = @TARGET_LD_FLAGS@

include ../rules.mk
include ../sources.mk

# TODO: a global memory ring buffer stdout based printf
LKERNEL_SRCS_EXCLUDE = printf.c barrier.ll
LKERNEL_SRCS_EXTRA = barrier.c

EXTRA_DIST = CMakeLists.txt
