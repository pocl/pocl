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


set(PROGRAMS_TO_BUILD test_clFinish test_clGetDeviceInfo test_clGetEventInfo
  test_clCreateProgramWithBinary test_clGetSupportedImageFormats
  test_clSetEventCallback test_clEnqueueNativeKernel test_clBuildProgram
  test_clCreateKernelsInProgram test_clCreateKernel test_clGetKernelArgInfo
  test_version)

#EXTRA_DIST= \
# test_kernel_src_in_pwd.h \
# test_clCreateKernelsInProgram.cl \
# test_data/test_kernel_src_in_another_dir.h

#AM_LDFLAGS = @OPENCL_LIBS@ ../../lib/poclu/libpoclu.la
# POCLU_LINK_OPTIONS

#AM_CPPFLAGS = -I$(top_srcdir)/fix-include -I$(top_srcdir)/include @OPENCL_CFLAGS@
add_compile_options(${OPENCL_CFLAGS})

foreach(PROG ${PROGRAMS_TO_BUILD})
  if(MSVC)
    set_source_files_properties( "${PROG}.c" PROPERTIES LANGUAGE CXX )
  endif(MSVC)
  add_executable("${PROG}" "${PROG}.c")
  target_link_libraries("${PROG}" ${POCLU_LINK_OPTIONS})
endforeach()

#######################################################################


add_test("runtime/clGetDeviceInfo" "test_clGetDeviceInfo")

add_test("runtime/clEnqueueNativeKernel" "test_clEnqueueNativeKernel")

add_test("runtime/clGetEventInfo" "test_clGetEventInfo")

add_test("runtime/clCreateProgramWithBinary" "test_clCreateProgramWithBinary")

add_test(NAME "runtime/clBuildProgram"
         WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
         COMMAND "${CMAKE_CURRENT_BINARY_DIR}/test_clBuildProgram")

add_test("runtime/clFinish" "test_clFinish")

add_test("runtime/clCreateKernel" "test_clCreateKernel")

add_test("runtime/clGetKernelArgInfo" "test_clGetKernelArgInfo")

add_test_custom("${CMAKE_CURRENT_BINARY_DIR}/test_clSetEventCallback"
                "runtime/clSetEventCallback" "test_clSetEventCallback_expout.txt" )

add_test("runtime/clGetSupportedImageFormats" "test_clGetSupportedImageFormats")

add_test("runtime/clCreateKernelsInProgram" "test_clCreateKernelsInProgram")

set_tests_properties( "runtime/clGetDeviceInfo" "runtime/clEnqueueNativeKernel"
  "runtime/clGetEventInfo" "runtime/clCreateProgramWithBinary"
  "runtime/clBuildProgram" "runtime/clFinish" "runtime/clSetEventCallback"
  "runtime/clGetSupportedImageFormats" "runtime/clCreateKernelsInProgram"
  "runtime/clCreateKernel" "runtime/clGetKernelArgInfo"
  PROPERTIES
    COST 2.0
    PROCESSORS 1
    DEPENDS "pocl_version_check")

set_tests_properties("runtime/clGetSupportedImageFormats"
  PROPERTIES
    ENVIRONMENT "POCL_DEVICES=pthread\ pthread")

set_tests_properties("runtime/clCreateKernelsInProgram"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "Hello\nWorld")

if(LLVM_3_2)
  set_tests_properties("runtime/clGetKernelArgInfo"
    PROPERTIES WILL_FAIL 1)
endif()



set_tests_properties("runtime/clFinish"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "ABABC")

