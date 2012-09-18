# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/lib/CL.
# 
# Copyright (c) 2011-2012 Universidad Rey Juan Carlos and 
#                         Tampere University of Technology
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

SUBDIRS = devices

lib_LTLIBRARIES = libpocl.la

if ! BUILD_ICD

# If we do not build with the ICD enabled, the final .so and its soname
# should be libOpenCL.so for direct linkage. Some benchmarks such as
# Parboil require the soname to be libOpenCL.so (check with ldd) so
# symlinking is not enough.

lib_LTLIBRARIES += libOpenCL.la

endif

libpocl_la_SOURCES = clCreateContextFromType.c	\
                   clReleaseContext.c		\
                   clRetainContext.c		\
                   clGetContextInfo.c		\
                   clCreateCommandQueue.c	\
                   clReleaseCommandQueue.c	\
                   clRetainCommandQueue.c	\
                   clGetCommandQueueInfo.c	\
                   clCreateBuffer.c		\
                   clCreateSubBuffer.c		\
                   clEnqueueReadBuffer.c	\
                   clEnqueueReadBufferRect.c	\
                   clEnqueueMapBuffer.c	\
                   clEnqueueUnmapMemObject.c	\
                   clReleaseMemObject.c		\
                   clRetainMemObject.c		\
                   clGetMemObjectInfo.c		\
                   clSetMemObjectDestructorCallback.c	\
                   clCreateProgramWithSource.c	\
                   clCreateProgramWithBinary.c	\
                   clReleaseProgram.c		\
                   clRetainProgram.c		\
                   clBuildProgram.c		\
                   clCreateKernel.c		\
                   clReleaseKernel.c		\
                   clRetainKernel.c		\
                   clSetKernelArg.c		\
                   clEnqueueNDRangeKernel.c	\
                   clEnqueueNativeKernel.c	\
                   clCreateKernelsInProgram.c		\
                   clGetKernelInfo.c		\
                   clGetPlatformIDs.c		\
                   clGetPlatformInfo.c		\
                   clGetDeviceIDs.c		\
                   clGetDeviceInfo.c		\
                   clCreateContext.c		\
                   clGetProgramBuildInfo.c	\
                   clEnqueueWriteBuffer.c	\
                   clEnqueueWriteBufferRect.c	\
                   clEnqueueCopyBuffer.c	\
                   clEnqueueCopyBufferRect.c	\
                   clEnqueueCopyBufferToImage.c	\
                   clEnqueueCopyImageToBuffer.c	\
                   clFinish.c			\
                   clFlush.c			\
                   clEnqueueBarrier.c		\
                   clEnqueueMarker.c		\
                   clGetKernelWorkGroupInfo.c	\
                   clGetProgramInfo.c		\
                   clGetEventProfilingInfo.c	\
                   clWaitForEvents.c		\
                   clEnqueueWaitForEvents.c	\
                   clReleaseEvent.c		\
                   clRetainEvent.c		\
                   clSetEventCallback.c		\
                   clGetEventInfo.c		\
                   clCreateUserEvent.c		\
                   clSetUserEventStatus.c		\
                   clEnqueueTask.c		\
                   clCreateImage.c		\
                   clCreateImage2D.c		\
                   clCreateImage3D.c		\
                   clEnqueueReadImage.c	\
                   clEnqueueWriteImage.c	\
                   clCreateSampler.c	\
                   clReleaseSampler.c	\
                   clRetainSampler.c \
                   clGetSamplerInfo.c \
                   clEnqueueCopyImage.c		\
                   clEnqueueMapImage.c		\
                   clGetImageInfo.c		\
                   clCreateFromGLTexture2D.c	\
                   clCreateFromGLTexture3D.c	\
                   clUnloadCompiler.c		\
                   clGetSupportedImageFormats.c	\
                   clGetExtensionFunctionAddress.c \
                   clIcdGetPlatformIDsKHR.c \
                   clReleaseDevice.c \
                   clRetainDevice.c \
                   pocl_cl.h \
                   pocl_util.c pocl_util.h \
                   pocl_image_util.c pocl_image_util.h \
                   pocl_icd.h \
                   pocl_intfn.h


libpocl_la_CPPFLAGS = -I$(top_srcdir)/fix-include -I$(top_srcdir)/include -I$(top_srcdir)/lib/CL/devices $(OCL_ICD_CFLAGS)
libpocl_la_LDFLAGS = -lltdl @PTHREAD_CFLAGS@ --version-info ${LIB_VERSION}
libpocl_la_LDFLAGS += -Wl,-z,defs 

if TCE_AVAILABLE
# The ttasim device is written in C++ as it accesses the ttasim library of TCE
# directly.
libpocl_la_LDFLAGS += -lstdc++ `@TCE_CONFIG@ --libs`
endif

libpocl_la_LIBADD = devices/libpocl-devices.la 

if ! BUILD_ICD

libOpenCL_la_SOURCES = ${libpocl_la_SOURCES}
libOpenCL_la_CPPFLAGS = ${libpocl_la_CPPFLAGS}
libOpenCL_la_LDFLAGS = ${libpocl_la_LDFLAGS}
libOpenCL_la_LIBADD = ${libpocl_la_LIBADD}

endif
