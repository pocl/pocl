#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2022 pocl developers
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

set_opencl_header_includes()

add_test(NAME pocl_version_check COMMAND test_version)

if(ENABLE_HOST_CPU_DEVICES)
    # If basic OpenCL device is not included, we assume the device is chosen using
    # the POCL_DEVICES env. Otherwise, use the basic dev to check that we are using
    # pocl.
    set_tests_properties("pocl_version_check"
      PROPERTIES
      ENVIRONMENT "POCL_DEVICES=basic"
      PASS_REGULAR_EXPRESSION "basic"
      LABELS "internal;cuda")
endif()

#######################################################################
if(UNIX AND HAVE_DLFCN_H)

if(ENABLE_ICD)
add_test(NAME pocl_test_dlopen_libpocl COMMAND test_dlopen)
set_tests_properties("pocl_test_dlopen_libpocl" PROPERTIES LABELS "internal;dlopen;vulkan")
set_property(TEST "pocl_version_check" APPEND PROPERTY DEPENDS "pocl_test_dlopen_libpocl")
endif()

if(ENABLE_ICD AND ENABLE_LOADABLE_DRIVERS)

if(BUILD_BASIC)
  add_test(NAME pocl_test_dlopen_device_basic COMMAND test_dlopen basic)
  set_tests_properties("pocl_test_dlopen_device_basic" PROPERTIES LABELS "internal;dlopen")
  set_property(TEST "pocl_version_check" APPEND PROPERTY DEPENDS "pocl_test_dlopen_device_basic")
endif()

if(BUILD_PTHREAD)
  add_test(NAME pocl_test_dlopen_device_pthread COMMAND test_dlopen pthread)
  set_tests_properties("pocl_test_dlopen_device_pthread" PROPERTIES LABELS "internal;dlopen")
  set_property(TEST "pocl_version_check" APPEND PROPERTY DEPENDS "pocl_test_dlopen_device_pthread")
endif()

if(BUILD_ALMAIF)
  add_test(NAME pocl_test_dlopen_device_almaif COMMAND test_dlopen almaif)
  set_tests_properties("pocl_test_dlopen_device_almaif" PROPERTIES LABELS "internal;dlopen")
  set_property(TEST "pocl_version_check" APPEND PROPERTY DEPENDS "pocl_test_dlopen_device_almaif")
endif()

if(ENABLE_TCE)
  add_test(NAME pocl_test_dlopen_device_tce COMMAND test_dlopen tce)
  set_tests_properties("pocl_test_dlopen_device_tce" PROPERTIES LABELS "internal;dlopen")
  set_property(TEST "pocl_version_check" APPEND PROPERTY DEPENDS "pocl_test_dlopen_device_tce")
endif()

if(ENABLE_HSA)
  add_test(NAME pocl_test_dlopen_device_hsa COMMAND test_dlopen hsa)
  set_tests_properties("pocl_test_dlopen_device_hsa" PROPERTIES LABELS "internal;dlopen")
  set_property(TEST "pocl_version_check" APPEND PROPERTY DEPENDS "pocl_test_dlopen_device_hsa")
endif()

if(ENABLE_CUDA)
  add_test(NAME pocl_test_dlopen_device_cuda COMMAND test_dlopen cuda)
  set_tests_properties("pocl_test_dlopen_device_cuda" PROPERTIES LABELS "internal;dlopen")
  set_property(TEST "pocl_version_check" APPEND PROPERTY DEPENDS "pocl_test_dlopen_device_cuda")
endif()

if(ENABLE_PROXY)
  add_test(NAME pocl_test_dlopen_device_proxy COMMAND test_dlopen proxy)
  set_tests_properties("pocl_test_dlopen_device_proxy" PROPERTIES LABELS "internal;dlopen")
  set_property(TEST "pocl_version_check" APPEND PROPERTY DEPENDS "pocl_test_dlopen_device_proxy")
endif()

if(ENABLE_VULKAN)
  add_test(NAME pocl_test_dlopen_device_vulkan COMMAND test_dlopen vulkan)
  set_tests_properties("pocl_test_dlopen_device_vulkan" PROPERTIES LABELS "internal;dlopen;vulkan")
  set_property(TEST "pocl_version_check" APPEND PROPERTY DEPENDS "pocl_test_dlopen_device_vulkan")
endif()
endif()

endif()
#######################################################################

add_subdirectory("kernel")
add_subdirectory("regression")
add_subdirectory("runtime")
add_subdirectory("workgroup")
if(ENABLE_TCE)
  add_subdirectory("tce")
endif()
if(ENABLE_SPIRV)
  add_subdirectory("spirv")
endif()

