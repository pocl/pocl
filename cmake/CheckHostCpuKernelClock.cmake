#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2026 pocl developers
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

set(HOST_CPU_ENABLE_CL_KHR_KERNEL_CLOCK 0)

# Probe native host CPU support for cl_khr_kernel_clock
# __builtin_readcyclecounter() is only auto-enabled for native x86 host builds.
# On other architectures the builtin may compile but still trap at runtime.

if(ENABLE_LLVM AND ENABLE_HOST_CPU_DEVICES AND X86
   AND NOT CMAKE_CROSSCOMPILING)
  set(HOST_KERNEL_CLOCK_TEST_SRC
      "${CMAKE_BINARY_DIR}/check_host_kernel_clock.c")
  set(HOST_KERNEL_CLOCK_TEST_BIN
      "${CMAKE_BINARY_DIR}/check_host_kernel_clock${CMAKE_EXECUTABLE_SUFFIX}")

  file(WRITE "${HOST_KERNEL_CLOCK_TEST_SRC}" [[
int main(void) {
  unsigned long long t0 = __builtin_readcyclecounter();
  for (volatile int i = 0; i < 1000; ++i) {
  }
  unsigned long long t1 = __builtin_readcyclecounter();
  return (t0 > 0 && t1 > t0) ? 0 : 1;
}
]])

  set(HOST_KERNEL_CLOCK_TEST_FLAGS "${DEFAULT_HOST_CLANG_FLAGS}")
  separate_arguments(HOST_KERNEL_CLOCK_TEST_FLAGS)
  list(APPEND HOST_KERNEL_CLOCK_TEST_FLAGS
       "-o" "${HOST_KERNEL_CLOCK_TEST_BIN}")

  execute_process(
    COMMAND "${HOST_CLANG}" ${HOST_KERNEL_CLOCK_TEST_FLAGS}
            "${HOST_KERNEL_CLOCK_TEST_SRC}"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    RESULT_VARIABLE HOST_KERNEL_CLOCK_TEST_COMPILE_RES)

  if(HOST_KERNEL_CLOCK_TEST_COMPILE_RES EQUAL 0)
    execute_process(
      COMMAND "${HOST_KERNEL_CLOCK_TEST_BIN}"
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      RESULT_VARIABLE HOST_KERNEL_CLOCK_TEST_RUN_RES)
    if(HOST_KERNEL_CLOCK_TEST_RUN_RES EQUAL 0)
      set(HOST_CPU_ENABLE_CL_KHR_KERNEL_CLOCK 1)
    endif()
  endif()
endif()

message(STATUS
        "HOST_CPU_ENABLE_CL_KHR_KERNEL_CLOCK: ${HOST_CPU_ENABLE_CL_KHR_KERNEL_CLOCK}")
