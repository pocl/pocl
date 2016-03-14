# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/lib/CL.
# 
# Copyright (c) 2011-2014 Universidad Rey Juan Carlos and 
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
                   clCreateCommandQueueWithProperties.c	\
                   clReleaseCommandQueue.c	\
                   clRetainCommandQueue.c	\
                   clGetCommandQueueInfo.c	\
                   clCreateBuffer.c		\
                   clCreateSubBuffer.c		\
                   clEnqueueFillImage.c	\
                   clEnqueueFillBuffer.c	\
                   clEnqueueReadBuffer.c	\
                   clEnqueueReadBufferRect.c	\
                   clEnqueueMapBuffer.c	\
                   clEnqueueMapBuffer.h	\
                   clEnqueueMigrateMemObjects.c \
                   clEnqueueUnmapMemObject.c	\
                   clEnqueueMarkerWithWaitList.c \
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
                   clGetKernelArgInfo.c		\
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
                   clEnqueueBarrierWithWaitList.c \
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
                   clCreateSubDevices.c \
                   clReleaseDevice.c \
                   clRetainDevice.c \
                   clSVMAlloc.c \
                   clSVMFree.c \
                   clEnqueueSVMFree.c \
                   clEnqueueSVMMap.c \
                   clEnqueueSVMUnmap.c \
                   clEnqueueSVMMemcpy.c \
                   clEnqueueSVMMemFill.c \
                   clSetKernelArgSVMPointer.c \
                   clSetKernelExecInfo.c \
                   pocl_binary.c pocl_binary.h \
                   pocl_cl.h \
                   pocl_util.c pocl_util.h \
                   pocl_tracing.c pocl_tracing.h \
                   pocl_queue_util.c pocl_queue_util.h \
                   pocl_image_util.c pocl_image_util.h \
                   pocl_img_buf_cpy.h pocl_img_buf_cpy.c \
                   pocl_timing.c pocl_timing.h \
                   pocl_intfn.h pocl_icd.h \
                   pocl_llvm.h \
                   pocl_runtime_config.c pocl_runtime_config.h \
                   pocl_mem_management.c pocl_mem_management.h \
                   pocl_hash.c pocl_hash.h pocl_cache.c \
                   pocl_debug.c pocl_debug.h

libpocl_la_CPPFLAGS = -I$(top_srcdir)/fix-include -I$(top_srcdir)/fix-include/OpenCL -I$(top_srcdir)/include -I$(top_srcdir)/lib/CL/devices $(OCL_ICD_CFLAGS) @LTTNG_UST_CFLAGS@
libpocl_la_LDFLAGS = -lltdl @PTHREAD_CFLAGS@ -version-info ${LIB_VERSION}
#libpocl_la_LDFLAGS += -Wl,-z,defs 

if LTTNG_UST_AVAILABLE
libpocl_la_SOURCES += pocl_lttng.h pocl_lttng.c
endif


if TCE_AVAILABLE
# The ttasim device is written in C++ as it accesses the ttasim library of TCE
# directly.
libpocl_la_LDFLAGS += -lstdc++ `@TCE_CONFIG@ --libs` -ltce
endif

if BUILD_HSA
libpocl_la_LDFLAGS += ${HSA_LIBS}
endif

libpocl_la_LIBADD = devices/libpocl-devices.la
libpocl_la_LIBADD += ${LTDL_LIBS}

# The POCL_DEBUG output requires the clock_gettime and older glibs
# have it in a separate lib.
# Mac doesnt have lrt, or clock_gettime

if HAVE_CLOCK_GETTIME
libpocl_la_LIBADD += -lrt
endif

#Kludge: compile pocl_llvm_api.cc into a library of its own.
#The source file is necessarely a C++ file, and having a C++ file
#causes the entire pocl project to be linked with g++, which in turn
#exposes a gcc/autoconf m4 macro issue
# see: http://stackoverflow.com/questions/17055279/autotools-for-pthreads-not-setting-correct-linker-flags
noinst_LTLIBRARIES = libpoclllvm.la

libpoclllvm_la_SOURCES = pocl_llvm_api.cc
libpoclllvm_la_CPPFLAGS = ${libpocl_la_CPPFLAGS} -I$(top_srcdir)/lib/llvmopencl/
libpoclllvm_la_LDFLAGS = @LLVM_LDFLAGS@
libpocl_la_LDFLAGS += @LLVM_LDFLAGS@
AM_CXXFLAGS = @PTHREAD_CFLAGS@ @LLVM_CXX_FLAGS@

libpocl_la_LDFLAGS += -L${top_builddir}/lib/CL/ @ICD_LD_FLAGS@
libpocl_la_LIBADD += libpoclllvm.la
libpocl_la_LIBADD += ${top_builddir}/lib/llvmopencl/libllvmpasses.la
libpocl_la_LIBADD += -lclangFrontend -lclangDriver -lclangParse -lclangSema -lclangEdit -lclangLex -lclangSerialization
libpocl_la_LIBADD += -lclangAST -lclangBasic -lclangFrontendTool -lclangRewriteFrontend 
libpocl_la_LIBADD += -lclangStaticAnalyzerFrontend -lclangStaticAnalyzerCore
libpocl_la_LIBADD += -lclangAnalysis -lclangCodeGen -lclangAST
libpocl_la_LIBADD += -lstdc++ -lm @LLVM_LIBS@ @LTTNG_UST_LIBS@


libpocl_la_LIBADD += -lclangASTMatchers -lclangBasic

if ! BUILD_ICD
libOpenCL_la_SOURCES = ${libpocl_la_SOURCES}
libOpenCL_la_CPPFLAGS = ${libpocl_la_CPPFLAGS}
libOpenCL_la_LDFLAGS = ${libpocl_la_LDFLAGS}
libOpenCL_la_LIBADD = ${libpocl_la_LIBADD}
endif

CLEANFILES = libpocl.la libOpenCL.la kernellib_hash.h

EXTRA_DIST = CMakeLists.txt

BUILT_SOURCES = kernellib_hash.h

FORCE:

kernellib_hash.h: FORCE
	echo '#define POCL_KERNELLIB_SHA1 "'`sha1sum ../../include/_kernel.h ../../include/_kernel_c.h ../../include/pocl_types.h ../kernel/*.cl ../kernel/*.c ../kernel/vecmathlib-pocl/*.cl ../kernel/vecmathlib-pocl/*.cc | sha1sum -`'"' > kernellib_hash.new
	cmp kernellib_hash.new kernellib_hash.h || mv kernellib_hash.new kernellib_hash.h
