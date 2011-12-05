# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/lib/CL.
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

lib_LTLIBRARIES = libCL.la

libCL_la_SOURCES = clCreateContextFromType.c	\
                   clReleaseContext.c		\
                   clGetContextInfo.c		\
                   clCreateCommandQueue.c	\
                   clReleaseCommandQueue.c	\
                   clCreateBuffer.c		\
	           clEnqueueReadBuffer.c	\
                   clReleaseMemObject.c		\
                   clCreateProgramWithSource.c	\
                   clReleaseProgram.c		\
                   clBuildProgram.c		\
                   clCreateKernel.c		\
                   clReleaseKernel.c		\
                   clSetKernelArg.c		\
                   clEnqueueNDRangeKernel.c	\
                   clGetPlatformIDs.c		\
                   clGetPlatformInfo.c		\
                   clGetDeviceIDs.c		\
                   clGetDeviceInfo.c		\
                   clCreateContext.c		\
                   clGetProgramBuildInfo.c	\
                   clEnqueueWriteBuffer.c	\
                   clFinish.c			\
                   pocl_cl.h			\
                   devices/devices.h		\
                   devices/devices.c		\
                   devices/native/native.h	\
                   devices/native/native.c	\
                   devices/pthread/pthread.h	\
                   devices/pthread/pthread.c

libCL_la_CPPFLAGS = -I$(top_srcdir)/include -U__APPLE__
libCL_la_LDFLAGS = -lltdl
