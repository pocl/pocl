# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for tests/regression.
# 
# Copyright (c) 2012 Cyril Roelandt and Pekka Jääskeläinen
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

noinst_PROGRAMS= test_clFinish test_clGetDeviceInfo test_clGetEventInfo \
	test_read-copy-write-buffer test_event_cycle test_event_free \
	test_clCreateProgramWithBinary test_clGetSupportedImageFormats \
	test_clSetEventCallback test_clEnqueueNativeKernel test_clBuildProgram \
	test_link_error test_kernel_cache_includes \
	test_clCreateSubDevices test_enqueue_kernel_from_binary \
	test_clCreateKernelsInProgram test_clCreateKernel test_version \
	test_clGetKernelArgInfo

EXTRA_DIST= \
	test_kernel_src_in_pwd.h test_clSetEventCallback_expout.txt \
	test_clCreateKernelsInProgram.cl CMakeLists.txt \
	test_data/test_kernel_src_in_another_dir.h \
        test_kernel_cache_includes.cl test_kernel_cache_includes_expout.txt \
	clGetKernelArgInfo.spir32_meta clGetKernelArgInfo.spir32_nometa \
	clGetKernelArgInfo.spir64_meta clGetKernelArgInfo.spir64_nometa

AM_LDFLAGS = @OPENCL_LIBS@ ../../lib/poclu/libpoclu.la
AM_CPPFLAGS = -I$(top_srcdir)/fix-include -I$(top_srcdir)/include @OPENCL_CFLAGS@
