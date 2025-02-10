#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2025 Michal Babej / Intel Finland oy
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

if(NOT ENABLE_HOST_CPU_VECTORIZE_BUILTINS)
  return()
endif()
# the vectorization of this code only works with LLVM 19; earlier LLVM versions can
# vectorize some but not all of the builtins in vecbuiltin.cl, therefore llvm-filecheck
# will report a failure
if(LLVM_VERSION VERSION_LESS 19.0)
  return()
endif()

add_compile_options(${OPENCL_CFLAGS})

add_executable("vecbuiltin" vecbuiltin.c vecbuiltin_exec.c vecbuiltin.cl)
target_link_libraries("vecbuiltin" ${POCLU_LINK_OPTIONS})
add_symlink_to_built_opencl_dynlib("vecbuiltin")

add_test_pocl(NAME "examples/vecbuiltin" COMMAND "vecbuiltin" LLVM_FILECHECK "vecbuiltin.fc" WORKITEM_HANDLER "loopvec")

set_tests_properties(
  "examples/vecbuiltin"
  PROPERTIES
    COST 3.0
    PASS_REGULAR_EXPRESSION "OK"
    PROCESSORS 1
    LABELS "cpu;vulkan;level0"
    DEPENDS "pocl_version_check")
