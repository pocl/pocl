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

add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

#cannot use add_compile_options, because we need this only for C files
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 ${OPENCL_CFLAGS_STR}")

######################################################################
add_executable("kernel" "kernel.c")
target_link_libraries("kernel" ${POCLU_LINK_OPTIONS})

add_test_pocl(NAME "kernel/test_as_type"
              COMMAND "kernel" "test_as_type")

add_test_pocl(NAME "kernel/test_convert_type_1"
              COMMAND "kernel" "test_convert_type_1")

add_test_pocl(NAME "kernel/test_convert_type_2"
              COMMAND "kernel" "test_convert_type_2")

add_test_pocl(NAME "kernel/test_convert_type_4"
              COMMAND "kernel" "test_convert_type_4")

add_test_pocl(NAME "kernel/test_convert_type_8"
              COMMAND "kernel" "test_convert_type_8")

add_test_pocl(NAME "kernel/test_convert_type_16"
              COMMAND "kernel" "test_convert_type_16")

add_test_pocl(NAME "kernel/test_bitselect"
              COMMAND "kernel" "test_bitselect")

add_test_pocl(NAME "kernel/test_hadd_loopvec"
              COMMAND "kernel" "test_hadd")

add_test_pocl(NAME "kernel/test_hadd_loops"
              COMMAND "kernel" "test_hadd")

set_tests_properties( "kernel/test_as_type" "kernel/test_bitselect"
  "kernel/test_convert_type_1" "kernel/test_convert_type_2" "kernel/test_convert_type_4"
  "kernel/test_convert_type_8" "kernel/test_convert_type_16"
  "kernel/test_hadd_loops" "kernel/test_hadd_loopvec"
  PROPERTIES
    COST 40.0
    FAIL_REGULAR_EXPRESSION "FAIL"
    PASS_REGULAR_EXPRESSION "\nOK\n"
    PROCESSORS 1
    DEPENDS "pocl_version_check"
    LABELS "kernel")

# convert_type_{4,8,16} on some machines takes >1 hour,
# so only add convert_type_1/2 to internal tests to keep them reasonably fast.

set_tests_properties( "kernel/test_as_type" "kernel/test_bitselect"
  "kernel/test_convert_type_1" "kernel/test_convert_type_2"
  PROPERTIES
    LABELS "internal;kernel")

set_tests_properties("kernel/test_hadd_loops"
  PROPERTIES ENVIRONMENT "POCL_WORK_GROUP_METHOD=loops")

set_tests_properties("kernel/test_hadd_loopvec"
  PROPERTIES ENVIRONMENT "POCL_WORK_GROUP_METHOD=loopvec")

add_test_pocl(NAME "kernel/test_min_max"
              COMMAND "kernel" "test_min_max")

add_test_pocl(NAME "kernel/test_length_distance"
              COMMAND "kernel" "test_length_distance")

add_test_pocl(NAME "kernel/test_fmin_fmax_fma"
              COMMAND "kernel" "test_fmin_fmax_fma")

add_test_pocl(NAME "kernel/test_local_struct_array"
              COMMAND "kernel" "test_local_struct_array")

add_test_pocl(NAME "kernel/test_convert_sat_regression"
              COMMAND "kernel" "test_convert_sat_regression")

add_test_pocl(NAME "kernel/test_rotate"
              COMMAND "kernel" "test_rotate")

add_test_pocl(NAME "kernel/test_fabs"
              COMMAND "kernel" "test_fabs")

add_test_pocl(NAME "kernel/test_copy_signbit"
              COMMAND "kernel" "test_copy_signbit")

add_test_pocl(NAME "kernel/test_ilogb"
              COMMAND "kernel" "test_ilogb")

add_test_pocl(NAME "kernel/test_ldexp"
              COMMAND "kernel" "test_ldexp")

add_test_pocl(NAME "kernel/test_isnan"
              COMMAND "kernel" "test_isnan")

add_test_pocl(NAME "kernel/test_short16"
              COMMAND "kernel" "test_short16")

add_test_pocl(NAME "kernel/test_frexp_modf"
              COMMAND "kernel" "test_frexp_modf")

set_tests_properties("kernel/test_min_max" "kernel/test_length_distance"
  "kernel/test_fmin_fmax_fma" "kernel/test_local_struct_array"
  "kernel/test_convert_sat_regression"   "kernel/test_fabs"
  "kernel/test_rotate" "kernel/test_copy_signbit" "kernel/test_ilogb"
  "kernel/test_ldexp" "kernel/test_isnan" "kernel/test_short16"
  "kernel/test_frexp_modf"
  PROPERTIES
    COST 4.0
    FAIL_REGULAR_EXPRESSION "FAIL"
    PASS_REGULAR_EXPRESSION "\nOK\n"
    PROCESSORS 1
    DEPENDS "pocl_version_check"
    LABELS "internal;kernel")

if(LLVM_ASSERTS_BUILD AND POCL_USE_FAKE_ADDR_SPACE_IDS)
  set_tests_properties("kernel/test_local_struct_array"
    PROPERTIES WILL_FAIL 1)
endif()

######################################################################
if(MSVC)
  set_source_files_properties( 
    sampler_address_clamp.c image_query_funcs.c
    test_shuffle.cc kernel.c PROPERTIES LANGUAGE CXX )
endif(MSVC)

add_executable("sampler_address_clamp" "sampler_address_clamp.c")
target_link_libraries("sampler_address_clamp" ${POCLU_LINK_OPTIONS})

add_executable("image_query_funcs" "image_query_funcs.c")
target_link_libraries("image_query_funcs" ${POCLU_LINK_OPTIONS})

add_test_pocl(NAME "kernel/test_sampler_address_clamp"
              COMMAND "sampler_address_clamp")

add_test_pocl(NAME "kernel/test_image_query_funcs"
              COMMAND "image_query_funcs")

set_tests_properties( "kernel/test_sampler_address_clamp"
  "kernel/test_image_query_funcs"
  PROPERTIES
    COST 4.0
    PASS_REGULAR_EXPRESSION "\nOK\n"
    PROCESSORS 1
    DEPENDS "pocl_version_check"
    LABELS "internal;kernel")

######################################################################

add_executable("test_shuffle" "test_shuffle.cc")
target_link_libraries("test_shuffle" ${POCLU_LINK_OPTIONS})

set(EXTRA_TESTS "")
add_test_pocl(NAME "kernel/test_shuffle_char"
              COMMAND "test_shuffle" "char")

add_test_pocl(NAME "kernel/test_shuffle_short"
              COMMAND "test_shuffle" "short")

add_test_pocl(NAME "kernel/test_shuffle_ushort"
              COMMAND "test_shuffle" "ushort")

if(NOT CL_DISABLE_HALF)
  add_test_pocl(NAME "kernel/test_shuffle_half"
              COMMAND "test_shuffle" "half")
  list(APPEND EXTRA_TESTS "kernel/test_shuffle_half")
endif()

add_test_pocl(NAME "kernel/test_shuffle_int"
              COMMAND "test_shuffle" "int")

add_test_pocl(NAME "kernel/test_shuffle_uint"
              COMMAND "test_shuffle" "uint")

add_test_pocl(NAME "kernel/test_shuffle_float"
              COMMAND "test_shuffle" "float")

add_test_pocl(NAME "kernel/test_shuffle_long"
              COMMAND "test_shuffle" "long")

add_test_pocl(NAME "kernel/test_shuffle_ulong"
              COMMAND "test_shuffle" "ulong")

######################################################################

add_test_pocl(NAME "kernel/test_printf"
              EXPECTED_OUTPUT "test_printf_expout.txt"
              COMMAND "kernel" "test_printf")

add_test_pocl(NAME "kernel/test_sizeof_uint"
              EXPECTED_OUTPUT "test_sizeof_expout.txt"
              COMMAND "kernel" "test_sizeof")

if(ENABLE_FP64)
  add_test_pocl(NAME "kernel/test_shuffle_double"
                COMMAND "test_shuffle" "double")
  list(APPEND EXTRA_TESTS "kernel/test_shuffle_double")
endif()

set_tests_properties("kernel/test_shuffle_char"
  "kernel/test_shuffle_short" "kernel/test_shuffle_ushort"
  "kernel/test_shuffle_int" "kernel/test_shuffle_uint"
  "kernel/test_shuffle_long" "kernel/test_shuffle_ulong"
  "kernel/test_shuffle_float" ${EXTRA_TESTS}
  "kernel/test_printf" "kernel/test_sizeof_uint"

  PROPERTIES
    COST 77
    PASS_REGULAR_EXPRESSION "OK\n"
    PROCESSORS 1
    DEPENDS "pocl_version_check"
    LABELS "internal;kernel")

######################################################################


######################################################################

#Comment out this one for now. Seems the test throws an exception, 
#and CTest doesn't think this is a WILL_FAIL, causing "Exception Other" 
#to be printed
#add_test_custom("${CMAKE_CURRENT_BINARY_DIR}/kernel" "kernel/test_block"
#                "test_block_expout.txt" "test_block")

#if(LLVM_3_4)
#  set_tests_properties("kernel/test_block"
#    PROPERTIES WILL_FAIL 1)
#endif()

#set_tests_properties("kernel/test_sizeof_uint" "kernel/test_block" "kernel/test_printf"
#  PROPERTIES
#    COST 2.0
#    PROCESSORS 1
#    DEPENDS "pocl_version_check")

# Label tests that work with CUDA backend
if(ENABLE_CUDA)
set_property(TEST
  "kernel/test_min_max"
  "kernel/test_length_distance"
  "kernel/test_fmin_fmax_fma"
  "kernel/test_convert_sat_regression"
  "kernel/test_rotate"
  "kernel/test_short16"
  "kernel/test_local_struct_array"
  "kernel/test_shuffle_char"
  "kernel/test_shuffle_short"
  "kernel/test_shuffle_ushort"
  "kernel/test_shuffle_int"
  "kernel/test_shuffle_uint"
  "kernel/test_shuffle_float"
  "kernel/test_shuffle_long"
  "kernel/test_shuffle_ulong"
  "kernel/test_shuffle_double"
  "kernel/test_convert_type_1"
  "kernel/test_convert_type_2"
  "kernel/test_convert_type_4"
  "kernel/test_as_type"
  "kernel/test_sizeof_uint"
  APPEND PROPERTY LABELS "cuda")
endif()
