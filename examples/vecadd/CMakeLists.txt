#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2018 pocl developers
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

add_compile_options(${OPENCL_CFLAGS})

add_executable("vecadd" vecadd.c vecadd_exec.c vecadd.cl)
target_link_libraries("vecadd" ${POCLU_LINK_OPTIONS})
add_symlink_to_built_opencl_dynlib("vecadd")

add_test(NAME "examples/vecadd" COMMAND "vecadd")

add_test(NAME "examples/vecadd_large_grid" COMMAND "vecadd" "128000" "128" "10000" "100" "1" "1")

if(ENABLE_HOST_CPU_VECTORIZE_SVML)
  # The SVML provider must remain dormant for an unrelated JIT kernel.
  add_cpu_jit_test("vecadd_no_svml" vecadd)
  set_tests_properties("cpu_jit/vecadd_no_svml" PROPERTIES
    ENVIRONMENT "POCL_DEVICES=cpu;POCL_CPU_JIT=1;POCL_WORK_GROUP_METHOD=loopvec;POCL_KERNEL_CACHE=0;POCL_FAULT_INJECT_JIT_SVML=1")
endif()

set(PROPS)
if(NOT ENABLE_ANYSAN)
  set(PROPS
    PASS_REGULAR_EXPRESSION "OK")
endif()

set_tests_properties(
  "examples/vecadd"
  "examples/vecadd_large_grid"
  PROPERTIES
    COST 3.0
    ${PROPS}
    PROCESSORS 1
    LABELS "vulkan;level0;mlir"
    DEPENDS "pocl_version_check")
