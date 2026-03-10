#=============================================================================
#   CMake build file for compiler unit tests
#
#   Copyright (c) 2025 Pekka Jääskeläinen / Intel Finland Oy
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
#
# The tests in this directory rely on LLVM FileCheck only and are not
# executed. The infra is designed such that it's easy and fun to add
# tests for specific LLVM IR transformations/fixes in cases where it's
# difficult or impossible to inspect the effect otherwise. It relies
# on poclcc to build the SPIR-V/OpenCL C kernels.

function(add_llvm_check)
  set(options SORT_OUTPUT)
  set(oneValueArgs SPIRV_NAME OPENCLC_NAME ENVIRONMENT)
  set(multiValueArgs COMMAND WORKITEM_HANDLER)
  cmake_parse_arguments(POCL_UTEST "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN})

  if (POCL_UTEST_WORKITEM_HANDLER)
    set(WORKITEM_HANDLER ${POCL_UTEST_WORKITEM_HANDLER})
  else()
    set(WORKITEM_HANDLER "loopvec")
  endif()
  if (POCL_UTEST_SPIRV_NAME)
    if (ENABLE_SPIRV)
      add_test_pocl(NAME "kernel_compiler_unit_tests/${POCL_UTEST_SPIRV_NAME}"
        COMMAND "poclcc" -s -o ${POCL_UTEST_SPIRV_NAME}.out
        "${CMAKE_CURRENT_SOURCE_DIR}/${POCL_UTEST_SPIRV_NAME}.spirv"
        WORKITEM_HANDLER "${WORKITEM_HANDLER}"
        LLVM_FILECHECK "${POCL_UTEST_SPIRV_NAME}.fc"
        ONLY_FILECHECK 1
        ENVIRONMENT "${POCL_UTEST_ENVIRONMENT}")
    endif()
  endif()
  if (POCL_UTEST_OPENCLC_NAME)
    add_test_pocl(NAME "kernel_compiler_unit_tests/${POCL_UTEST_OPENCLC_NAME}"
      COMMAND "poclcc" -o ${POCL_UTEST_OPENCLC_NAME}.out
      "${CMAKE_CURRENT_SOURCE_DIR}/${POCL_UTEST_OPENCLC_NAME}.cl"
      WORKITEM_HANDLER "${WORKITEM_HANDLER}"
      LLVM_FILECHECK "${POCL_UTEST_OPENCLC_NAME}.fc"
      ONLY_FILECHECK 1
      ENVIRONMENT "${POCL_UTEST_ENVIRONMENT}")
  endif()

endfunction()

add_llvm_check(SPIRV_NAME dirty-as-casts)
add_llvm_check(SPIRV_NAME issue-1871-unreachables-remaining)
add_llvm_check(SPIRV_NAME issue-1872-switch-to-a-bb-with-unreachable)
add_llvm_check(SPIRV_NAME switch-to-unreachable-with-a-phi-in-dest)
add_llvm_check(OPENCLC_NAME force-vector-width ENVIRONMENT "POCL_VECTORIZER_FORCE_VECTOR_WIDTH=4")
add_llvm_check(OPENCLC_NAME prefer-vector-width ENVIRONMENT "POCL_VECTORIZER_PREFER_VECTOR_WIDTH=512")
add_llvm_check(SPIRV_NAME trap-vectorization ENVIRONMENT "POCL_VECTORIZER_FORCE_VECTOR_WIDTH=4")
