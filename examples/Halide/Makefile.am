# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/examples/Halide
# 
# Copyright (c) 2014 Pekka Jääskeläinen / Tampere University of Technology
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

if TEST_SUITE_HALIDE

.PHONY: build

prepare-examples:
	make USE_LLVM_SHARED_LIB=1 -sC Halide bin/tutorial_lesson_12_using_the_gpu
	HL_TARGET=opencl make USE_LLVM_SHARED_LIB=1 -sC Halide/apps/local_laplacian
	HL_TARGET=opencl make USE_LLVM_SHARED_LIB=1 -sC Halide/apps/bilateral_grid
	HL_TARGET=opencl make USE_LLVM_SHARED_LIB=1 -sC Halide/apps/interpolate
	if test ! -e buhd4k.png; then wget http://www.cs.tut.fi/~pjaaskel/buhd4k.png -O buhd4k.png; fi

clean-examples:
	make -C Halide clean

endif
