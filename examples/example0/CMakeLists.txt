#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2019 pocl developers
#                 2024 Pekka Jääskeläinen / Intel Finland Oy
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

add_executable("example0" example0.c example0_exec.c example0.cl)
target_link_libraries("example0" ${POCLU_LINK_OPTIONS})
add_symlink_to_built_opencl_dynlib(example0)

add_test_pocl(NAME "examples/example0" COMMAND "example0" LLVM_FILECHECK "example0.fc" WORKITEM_HANDLER "loopvec")

if (ENABLE_SPIRV)
  add_test(NAME "examples/example0_spirv" COMMAND "example0" "v")
  list(APPEND SPIREX "examples/example0_spirv")
endif()

set(PROPS)
if(NOT ENABLE_ANYSAN)
  set(PROPS
    PASS_REGULAR_EXPRESSION "OK")
endif()

set_tests_properties( "examples/example0" ${SPIREX}
  PROPERTIES
    COST 3.0
    ${PROPS}
    PROCESSORS 1
    # devices which support SPIR
    LABELS "cuda;level0"
    DEPENDS "pocl_version_check")

# devices which don't support SPIR
set_property(TEST "examples/example0"
  APPEND PROPERTY LABELS "hsa;hsa-native;vulkan")

if (ENABLE_TCE)
  set_property(TEST "examples/example0"
    APPEND PROPERTY LABELS "tce;almaif")
endif()
