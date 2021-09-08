#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014 pocl developers
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

# do not link test_dlopen with -lOpenCL
if (UNIX AND HAVE_DLFCN_H)
  add_executable("test_dlopen" "test_dlopen.c")
  if(SANITIZER_OPTIONS)
    target_link_libraries("test_dlopen" ${SANITIZER_LIBS})
  endif()
  target_link_libraries("test_dlopen" ${DL_LIB})
endif ()

set(PROGRAMS_TO_BUILD test_clFinish test_clGetDeviceInfo test_clGetEventInfo
  test_clCreateProgramWithBinary test_clGetSupportedImageFormats
  test_clSetEventCallback test_clEnqueueNativeKernel test_clBuildProgram
  test_clCreateKernelsInProgram test_clCreateKernel test_clGetKernelArgInfo
  test_version test_kernel_cache_includes test_event_cycle test_link_error
  test_read-copy-write-buffer test_buffer-image-copy test_clCreateSubDevices test_event_free
  test_event_double_wait test_buffer_migration test_buffer_ping_pong
  test_enqueue_kernel_from_binary test_user_event test_fill-buffer
  test_clSetMemObjectDestructorCallback
  test_cl_pocl_content_size)

add_compile_options(${OPENCL_CFLAGS})

foreach(PROG ${PROGRAMS_TO_BUILD})
  if(MSVC)
    set_source_files_properties( "${PROG}.c" PROPERTIES LANGUAGE CXX )
  endif(MSVC)
  add_executable("${PROG}" "${PROG}.c")
  target_link_libraries("${PROG}" ${POCLU_LINK_OPTIONS})
endforeach()

#######################################################################


add_test_pocl(NAME "runtime/clGetDeviceInfo" COMMAND "test_clGetDeviceInfo")

add_test_pocl(NAME "runtime/clEnqueueNativeKernel" COMMAND "test_clEnqueueNativeKernel")

add_test_pocl(NAME "runtime/clGetEventInfo" COMMAND "test_clGetEventInfo")

add_test_pocl(NAME "runtime/clCreateProgramWithBinary" COMMAND "test_clCreateProgramWithBinary")

add_test_pocl(NAME "runtime/clBuildProgram"
              WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
              COMMAND "test_clBuildProgram")

add_test_pocl(NAME "runtime/test_kernel_cache_includes"
              WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
              COMMAND "test_kernel_cache_includes")

add_test_pocl(NAME "runtime/clFinish" COMMAND "test_clFinish")

add_test_pocl(NAME "runtime/test_event_cycle" COMMAND "test_event_cycle")

add_test_pocl(NAME "runtime/test_link_error" COMMAND "test_link_error")

add_test_pocl(NAME "runtime/test_read-copy-write-buffer" COMMAND "test_read-copy-write-buffer")

add_test_pocl(NAME "runtime/test_fill-buffer" COMMAND "test_fill-buffer")

add_test_pocl(NAME "runtime/test_buffer-image-copy" COMMAND "test_buffer-image-copy")

add_test_pocl(NAME "runtime/clCreateKernel" COMMAND "test_clCreateKernel")

add_test_pocl(NAME "runtime/clGetKernelArgInfo" COMMAND "test_clGetKernelArgInfo")

add_test_pocl(NAME "runtime/clSetEventCallback"
              COMMAND "test_clSetEventCallback"
              EXPECTED_OUTPUT "test_clSetEventCallback_expout.txt" )

add_test_pocl(NAME "runtime/clGetSupportedImageFormats" COMMAND "test_clGetSupportedImageFormats")

add_test_pocl(NAME "runtime/clCreateKernelsInProgram" COMMAND "test_clCreateKernelsInProgram")

add_test_pocl(NAME "runtime/clCreateSubDevices" COMMAND  "test_clCreateSubDevices")

add_test_pocl(NAME "runtime/test_event_free" COMMAND  "test_event_free")

add_test_pocl(NAME "runtime/test_event_double_wait" COMMAND  "test_event_double_wait")

add_test_pocl(NAME "runtime/test_enqueue_kernel_from_binary" COMMAND "test_enqueue_kernel_from_binary")

add_test_pocl(NAME "runtime/test_user_event" COMMAND  "test_user_event")

add_test_pocl(NAME "runtime/test_buffer_migration" COMMAND  "test_buffer_migration")

add_test_pocl(NAME "runtime/test_buffer_ping_pong" COMMAND  "test_buffer_ping_pong")

add_test_pocl(NAME "runtime/clSetMemObjectDestructorCallback" COMMAND  "test_clSetMemObjectDestructorCallback")

add_test_pocl(NAME "runtime/test_cl_pocl_content_size" COMMAND "test_cl_pocl_content_size")

set_tests_properties( "runtime/clGetDeviceInfo" "runtime/clEnqueueNativeKernel"
  "runtime/clGetEventInfo" "runtime/clCreateProgramWithBinary"
  "runtime/clBuildProgram" "runtime/clFinish" "runtime/clSetEventCallback"
  "runtime/clGetSupportedImageFormats" "runtime/clCreateKernelsInProgram"
  "runtime/clCreateKernel" "runtime/clGetKernelArgInfo"
  "runtime/test_kernel_cache_includes" "runtime/test_event_cycle"
  "runtime/test_read-copy-write-buffer" "runtime/test_buffer-image-copy"
  "runtime/test_fill-buffer"
  "runtime/test_event_free" "runtime/test_event_double_wait" "runtime/clCreateSubDevices"
  "runtime/test_enqueue_kernel_from_binary" "runtime/test_user_event"
  "runtime/clSetMemObjectDestructorCallback" "runtime/test_link_error"
  "runtime/test_cl_pocl_content_size"

  PROPERTIES
    COST 2.0
    PROCESSORS 1
    DEPENDS "pocl_version_check"
    LABELS "internal;runtime")

if(NOT ENABLE_ANYSAN)
  set_tests_properties("runtime/clCreateKernelsInProgram"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "Hello\nWorld")

  set_tests_properties("runtime/clFinish"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "ABABC;A\nB\nA\nB\nC\n")

  set_tests_properties("runtime/test_kernel_cache_includes"
  PROPERTIES PASS_REGULAR_EXPRESSION
  "function 1.*first include.*function 2.*second include")
endif()

# Label tests that work with CUDA backend
set_property(TEST
  "runtime/clGetDeviceInfo"
  "runtime/clGetEventInfo"
  "runtime/clCreateProgramWithBinary"
  "runtime/test_kernel_cache_includes"
  "runtime/clFinish"
  "runtime/test_read-copy-write-buffer"
  "runtime/test_fill-buffer"
  "runtime/test_buffer-image-copy"
  "runtime/clSetEventCallback"
  "runtime/clGetSupportedImageFormats"
  "runtime/clCreateKernelsInProgram"
  "runtime/test_event_cycle"
  "runtime/test_event_free"
  "runtime/test_event_double_wait"
  "runtime/test_user_event"
  "runtime/clSetMemObjectDestructorCallback"
  APPEND PROPERTY LABELS "cuda")

set_property(TEST
  "runtime/clGetDeviceInfo"
  "runtime/clGetEventInfo"
  "runtime/clCreateProgramWithBinary"
  "runtime/clBuildProgram"
  "runtime/test_kernel_cache_includes"
  "runtime/test_event_cycle"
  "runtime/test_link_error"
  "runtime/test_buffer-image-copy"
  "runtime/clCreateKernel"
  "runtime/clGetKernelArgInfo"
  "runtime/clGetSupportedImageFormats"
  "runtime/clSetEventCallback"
  "runtime/clCreateKernelsInProgram"
  "runtime/clFinish"
  "runtime/test_enqueue_kernel_from_binary"
  "runtime/test_user_event"
  "runtime/clSetMemObjectDestructorCallback"
  APPEND PROPERTY LABELS "hsa-native")

set_property(TEST
  "runtime/clGetDeviceInfo"
  "runtime/clGetEventInfo"
  "runtime/clCreateProgramWithBinary"
  "runtime/clFinish"
  "runtime/test_read-copy-write-buffer"
  "runtime/test_fill-buffer"
  "runtime/test_buffer-image-copy"
  "runtime/clSetEventCallback"
  "runtime/clGetSupportedImageFormats"
  "runtime/clCreateKernelsInProgram"
  "runtime/test_event_cycle"
  "runtime/test_event_free"
  "runtime/test_event_double_wait"
  "runtime/test_user_event"
  "runtime/clSetMemObjectDestructorCallback"
  APPEND PROPERTY LABELS "proxy")
