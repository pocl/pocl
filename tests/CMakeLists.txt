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

add_test("pocl_version_check" "runtime/test_version")

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

add_subdirectory("kernel")
add_subdirectory("regression")
add_subdirectory("runtime")
add_subdirectory("workgroup")
if(ENABLE_TCE)
  add_subdirectory("tce")
endif()

