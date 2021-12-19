#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2018 pocl developers
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================

# this is required, because we're compiling the C++ files of this directory
# with LLVM-provided flags, which usually contain their own setting of
# C++ standard, which may be in conflict with pocl's default C++ standard.
unset(CMAKE_CXX_STANDARD)

include_directories(AFTER "devices" ".")

add_subdirectory("devices")

###############################################################################

set(POCL_LIB_SOURCES  "clCreateContextFromType.c"
                   "clReleaseContext.c"
                   "clRetainContext.c"
                   "clGetContextInfo.c"
                   "clCreateCommandQueue.c"
                   "clCreateCommandQueueWithProperties.c"
                   "clReleaseCommandQueue.c"
                   "clRetainCommandQueue.c"
                   "clGetCommandQueueInfo.c"
                   "clCreateBuffer.c"
                   "clCreateSubBuffer.c"
                   "clEnqueueFillImage.c"
                   "clEnqueueFillBuffer.c"
                   "clEnqueueReadBuffer.c"
                   "clEnqueueReadBufferRect.c"
                   "clEnqueueMapBuffer.c"
                   "clEnqueueUnmapMemObject.c"
                   "clEnqueueMarkerWithWaitList.c"
                   "clEnqueueBarrierWithWaitList.c"
                   "clReleaseMemObject.c"
                   "clRetainMemObject.c"
                   "clGetMemObjectInfo.c"
                   "clSetMemObjectDestructorCallback.c"
                   "clCreateProgramWithSource.c"
                   "clCreateProgramWithBinary.c"
                   "clCreateProgramWithIL.c"
                   "clCreateProgramWithBuiltInKernels.c"
                   "clReleaseProgram.c"
                   "clRetainProgram.c"
                   "clBuildProgram.c"
                   "pocl_build.c"
                   "clCompileProgram.c"
                   "clLinkProgram.c"
                   "clCreateKernel.c"
                   "clReleaseKernel.c"
                   "clRetainKernel.c"
                   "clSetKernelArg.c"
                   "clEnqueueNDRangeKernel.c"
                   "clEnqueueNativeKernel.c"
                   "clCreateKernelsInProgram.c"
                   "clGetKernelInfo.c"
                   "clGetKernelArgInfo.c"
                   "clGetPlatformIDs.c"
                   "clGetPlatformInfo.c"
                   "clGetDeviceAndHostTimer.c"
                   "clGetDeviceIDs.c"
                   "clGetDeviceInfo.c"
                   "clCreateContext.c"
                   "clGetProgramBuildInfo.c"
                   "clEnqueueWriteBuffer.c"
                   "clEnqueueWriteBufferRect.c"
                   "clEnqueueCopyBuffer.c"
                   "clEnqueueCopyBufferRect.c"
                   "clEnqueueCopyBufferToImage.c"
                   "clEnqueueCopyImageToBuffer.c"
                   "clEnqueueMigrateMemObjects.c"
                   "clFinish.c"
                   "clFlush.c"
                   "clEnqueueBarrier.c"
                   "clEnqueueMarker.c"
                   "clGetKernelWorkGroupInfo.c"
                   "clGetKernelSubGroupInfo.c"
                   "clGetProgramInfo.c"
                   "clGetEventProfilingInfo.c"
                   "clWaitForEvents.c"
                   "clEnqueueWaitForEvents.c"
                   "clReleaseEvent.c"
                   "clRetainEvent.c"
                   "clSetEventCallback.c"
                   "clGetEventInfo.c"
                   "clCreateUserEvent.c"
                   "clSetUserEventStatus.c"
                   "clEnqueueTask.c"
                   "clCreateImage.c"
                   "clCreateImage2D.c"
                   "clCreateImage3D.c"
                   "clEnqueueReadImage.c"
                   "clEnqueueWriteImage.c"
                   "clCreateSampler.c"
                   "clReleaseSampler.c"
                   "clRetainSampler.c"
                   "clGetSamplerInfo.c"
                   "clEnqueueCopyImage.c"
                   "clEnqueueMapImage.c"
                   "clGetHostTimer.c"
                   "clGetImageInfo.c"
                   "clCreateFromGLTexture.c"
                   "clCreateFromGLTexture2D.c"
                   "clCreateFromGLTexture3D.c"
                   "clEnqueueAcquireGLObjects.c"
                   "clEnqueueReleaseGLObjects.c"
                   "clGetGLContextInfoKHR.c"
                   "clUnloadCompiler.c"
                   "clGetSupportedImageFormats.c"
                   "clGetExtensionFunctionAddress.c"
                   "clGetExtensionFunctionAddressForPlatform.c"
                   "clIcdGetPlatformIDsKHR.c"
                   "clReleaseDevice.c"
                   "clRetainDevice.c"
                   "clCreateSubDevices.c"
                   "clUnloadPlatformCompiler.c"
                   "clSetContentSizeBufferPoCL.c"
                   "clCreatePipe.c"
                   "clGetPipeInfo.c"
                   "pocl_cl.h" "pocl_util.h" "pocl_util.c"
                   "pocl_image_util.c" "pocl_image_util.h"
                   "pocl_img_buf_cpy.c"
                   "pocl_icd.h" "pocl_llvm.h"
                   "pocl_tracing.h" "pocl_tracing.c"
                   "pocl_runtime_config.c" "pocl_runtime_config.h"
                   "pocl_mem_management.c"  "pocl_mem_management.h"
                   "pocl_hash.c" "pocl_file_util.c"
                   "pocl_debug.h" "pocl_debug.c" "pocl_timing.c"
                   "clSVMAlloc.c" "clSVMFree.c" "clEnqueueSVMFree.c"
                   "clEnqueueSVMMap.c" "clEnqueueSVMUnmap.c"
                   "clEnqueueSVMMemcpy.c" "clEnqueueSVMMemFill.c"
                   "clSetKernelArgSVMPointer.c" "clSetKernelExecInfo.c"
                   "clSetDefaultDeviceCommandQueue.c"
                   "pocl_binary.c" "pocl_opengl.c" "pocl_cq_profiling.c")

if(ANDROID)
  list(APPEND POCL_LIB_SOURCES "pocl_mkstemp.c")
endif()

set(LIBPOCL_OBJS "$<TARGET_OBJECTS:libpocl_unlinked_objs>"
                 "$<TARGET_OBJECTS:pocl_cache>")

if(ENABLE_LOADABLE_DRIVERS)
  list(APPEND LIBPOCL_OBJS "$<TARGET_OBJECTS:pocl-devices>")
  if(ENABLE_HOST_CPU_DEVICES OR ENABLE_HSA)
    list(APPEND LIBPOCL_OBJS "$<TARGET_OBJECTS:pocl-devices-topology>")
  endif()
else()
  # link drivers into libpocl
  list(APPEND LIBPOCL_OBJS ${POCL_DEVICES_OBJS})
endif()

add_library("pocl_cache" OBJECT "pocl_cache.c")
harden("pocl_cache")


if (ENABLE_LLVM)
  include_directories(${LLVM_INCLUDE_DIRS})
  set(LLVM_API_SOURCES "pocl_llvm_build.cc" "pocl_llvm_metadata.cc" "pocl_llvm_utils.cc" "pocl_llvm_wg.cc")
  set_source_files_properties(${LLVM_API_SOURCES} PROPERTIES COMPILE_FLAGS "${LLVM_CXXFLAGS} -I\"${CMAKE_CURRENT_SOURCE_DIR}/../llvmopencl\"")

  add_library("lib_cl_llvm" OBJECT ${LLVM_API_SOURCES})
  harden("lib_cl_llvm")

  list(APPEND LIBPOCL_OBJS "$<TARGET_OBJECTS:llvmpasses>")
  list(APPEND LIBPOCL_OBJS "$<TARGET_OBJECTS:lib_cl_llvm>")

  # pocl_cache.c depends on a SHA1 hash of all built kernel-<target>.bc
  add_dependencies("pocl_cache" "kernellib_hash")
endif()

if(ENABLE_ICD)
  add_compile_options(${OCL_ICD_CFLAGS})
endif()

if(HAVE_LTTNG_UST)
  add_compile_options(${LTTNG_UST_CFLAGS})
  list(APPEND POCL_LIB_SOURCES "pocl_lttng.c" "pocl_lttng.h")
endif()

if(MSVC)
  set_source_files_properties( ${POCL_LIB_SOURCES} PROPERTIES LANGUAGE CXX )
endif(MSVC)

# this is so that we don't compile twice when building both libpocl and libOpenCL
add_library("libpocl_unlinked_objs" OBJECT ${POCL_LIB_SOURCES})
harden("libpocl_unlinked_objs")

#################################################################

set(POCL_PRIVATE_LINK_LIST)

# this must come 1st
if(SANITIZER_OPTIONS)
  set(POCL_PRIVATE_LINK_LIST ${SANITIZER_LIBS})
endif()

if(POCL_DEVICES_LINK_LIST)
  list(APPEND POCL_PRIVATE_LINK_LIST ${POCL_DEVICES_LINK_LIST})
endif()

if(ENABLE_LLVM)
   list(APPEND POCL_PRIVATE_LINK_LIST ${CLANG_LIBFILES} ${POCL_LLVM_LIBS} ${LLVM_SYSLIBS})
endif()

if(HAVE_LTTNG_UST)
  list(APPEND POCL_PRIVATE_LINK_LIST ${LTTNG_UST_LDFLAGS})
endif()

# -lrt is required for glibc < 2.17
if(HAVE_CLOCK_GETTIME AND CMAKE_SYSTEM_NAME MATCHES "Linux")
  list(APPEND POCL_PRIVATE_LINK_LIST "rt")
endif()

list(APPEND POCL_PRIVATE_LINK_LIST ${LIBMATH} ${CMAKE_DL_LIBS} ${PTHREAD_LIBRARY})

# see lib/CMakeLists.txt
set(POCL_TRANSITIVE_LIBS ${POCL_PRIVATE_LINK_LIST} PARENT_SCOPE)

#################################################################

# most Android systems have their own libOpenCL, often directly linked
# to the graphics drivers, which in turn are linked into every GUI program
# ...and the libOpenCL is not an ICD, therefore building dynamically
# on Android is pointless.
if(ANDROID)
  add_library("${POCL_LIBRARY_NAME}" STATIC ${LIBPOCL_OBJS})
else()
  add_library("${POCL_LIBRARY_NAME}" ${LIBPOCL_OBJS})
endif()
harden("${POCL_LIBRARY_NAME}")

set_target_properties("${POCL_LIBRARY_NAME}" PROPERTIES SOVERSION "${LIB_API_VERSION}"
  VERSION "${LIB_BUILD_VERSION}" 
  LINK_FLAGS "${LLVM_LDFLAGS}  ${ICD_LD_FLAGS}")

target_link_libraries("${POCL_LIBRARY_NAME}" PRIVATE ${POCL_PRIVATE_LINK_LIST})
install(TARGETS "${POCL_LIBRARY_NAME}"
        ARCHIVE DESTINATION ${POCL_INSTALL_PUBLIC_LIBDIR}/static
        COMPONENT "dev"
        LIBRARY DESTINATION ${POCL_INSTALL_PUBLIC_LIBDIR}
        COMPONENT "lib"
)

if(ENABLE_IPO)
  if (ENABLE_LLVM)
    set_property(TARGET "lib_cl_llvm" PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()
  set_property(TARGET "${POCL_LIBRARY_NAME}" PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
