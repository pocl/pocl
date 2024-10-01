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

add_test_pocl(NAME "kernel/test_hadd"
              WORKITEM_HANDLER "loops;loopvec;cbs"
              COMMAND "kernel" "test_hadd")



set(VARIANTS "loopvec;cbs")
foreach(VARIANT ${VARIANTS})
  set_tests_properties( "kernel/test_as_type_${VARIANT}" "kernel/test_bitselect_${VARIANT}"
    "kernel/test_convert_type_1_${VARIANT}" "kernel/test_convert_type_2_${VARIANT}" "kernel/test_convert_type_4_${VARIANT}"
    "kernel/test_convert_type_8_${VARIANT}" "kernel/test_convert_type_16_${VARIANT}"
    "kernel/test_hadd_${VARIANT}"
    PROPERTIES
      COST 40.0
      PROCESSORS 1
      DEPENDS "pocl_version_check"
      LABELS "kernel")
endforeach()

set_tests_properties( "kernel/test_hadd_loopvec"
  PROPERTIES
    COST 40.0
    PROCESSORS 1
    DEPENDS "pocl_version_check"
    LABELS "internal;kernel")


# convert_type_{4,8,16} on some machines takes >1 hour,
# so only add convert_type_1/2 to internal tests to keep them reasonably fast.

foreach(VARIANT ${VARIANTS})
  set_tests_properties( "kernel/test_as_type_${VARIANT}" "kernel/test_bitselect_${VARIANT}"
    "kernel/test_convert_type_1_${VARIANT}" "kernel/test_convert_type_2_${VARIANT}"
    "kernel/test_hadd_${VARIANT}"
    PROPERTIES
      LABELS "internal;kernel")
  # TODO: works on LLVM 15 but fails with LLVM 16, most likely reason:
  # https://github.com/pocl/pocl/issues/1525
  # https://github.com/grpc/grpc/issues/14587
  if(APPLE AND ARM64)
    set_tests_properties( "kernel/test_as_type_${VARIANT}" PROPERTIES DISABLED 1)
  endif()
endforeach()

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

add_test_pocl(NAME "kernel/test_halfs"
              COMMAND "kernel" "test_halfs")

foreach(VARIANT ${VARIANTS})
  set_tests_properties("kernel/test_min_max_${VARIANT}" "kernel/test_length_distance_${VARIANT}"
    "kernel/test_fmin_fmax_fma_${VARIANT}" "kernel/test_local_struct_array_${VARIANT}"
    "kernel/test_convert_sat_regression_${VARIANT}"   "kernel/test_fabs_${VARIANT}"
    "kernel/test_rotate_${VARIANT}" "kernel/test_copy_signbit_${VARIANT}" "kernel/test_ilogb_${VARIANT}"
    "kernel/test_ldexp_${VARIANT}" "kernel/test_isnan_${VARIANT}" "kernel/test_short16_${VARIANT}"
    "kernel/test_frexp_modf_${VARIANT}"
    PROPERTIES
      COST 4.0
      PROCESSORS 1
      DEPENDS "pocl_version_check"
      LABELS "internal;kernel")
endforeach()

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


foreach(VARIANT ${VARIANTS})
set_tests_properties( "kernel/test_sampler_address_clamp_${VARIANT}"
  PROPERTIES
    COST 4.0
    PASS_REGULAR_EXPRESSION "\nOK\n"
    SKIP_REGULAR_EXPRESSION "SKIP\n"
    PROCESSORS 1
    DEPENDS "pocl_version_check"
    LABELS "internal;kernel")

set_tests_properties( "kernel/test_image_query_funcs_${VARIANT}"
  PROPERTIES
    COST 4.0
    PASS_REGULAR_EXPRESSION "read imag1: 2,9,11,7\nread imag2: 2,9,11,7\n.*OK\n"
    SKIP_REGULAR_EXPRESSION "SKIP\n"
    PROCESSORS 1
    DEPENDS "pocl_version_check"
    LABELS "internal;kernel")
endforeach()
######################################################################

add_executable("test_shuffle" "test_shuffle.cc")
target_link_libraries("test_shuffle" ${POCLU_LINK_OPTIONS})

add_test_pocl(NAME "kernel/test_shuffle_char"
              COMMAND "test_shuffle" "char")

add_test_pocl(NAME "kernel/test_shuffle_short"
              COMMAND "test_shuffle" "short")

add_test_pocl(NAME "kernel/test_shuffle_ushort"
              COMMAND "test_shuffle" "ushort")

add_test_pocl(NAME "kernel/test_shuffle_int"
              COMMAND "test_shuffle" "int")

add_test_pocl(NAME "kernel/test_shuffle_uint"
              COMMAND "test_shuffle" "uint")

add_test_pocl(NAME "kernel/test_shuffle_half"
              COMMAND "test_shuffle" "half")

add_test_pocl(NAME "kernel/test_shuffle_float"
              COMMAND "test_shuffle" "float")

add_test_pocl(NAME "kernel/test_shuffle_double"
              COMMAND "test_shuffle" "double")

add_test_pocl(NAME "kernel/test_shuffle_long"
              COMMAND "test_shuffle" "long")

add_test_pocl(NAME "kernel/test_shuffle_ulong"
              COMMAND "test_shuffle" "ulong")

######################################################################

add_test_pocl(NAME "kernel/test_ucharn"
              EXPECTED_OUTPUT "test_ucharn_expout.txt"
              COMMAND "kernel" "test_ucharn")

add_test_pocl(NAME "kernel/test_printf"
              EXPECTED_OUTPUT "test_printf_expout.txt"
              COMMAND "kernel" "test_printf")

add_test_pocl(NAME "kernel/test_printf_vectors"
              EXPECTED_OUTPUT "test_printf_vectors_expout.txt"
              COMMAND "kernel" "test_printf_vectors")

add_test_pocl(NAME "kernel/test_printf_vectors_ulongn"
              EXPECTED_OUTPUT "test_printf_vectors_ulongn_expout.txt"
              COMMAND "kernel" "test_printf_vectors_ulongn")

add_test_pocl(NAME "kernel/test_sizeof_uint"
              EXPECTED_OUTPUT "test_sizeof_expout.txt"
              COMMAND "kernel" "test_sizeof")

foreach(VARIANT ${VARIANTS})

  set_tests_properties("kernel/test_shuffle_char_${VARIANT}"
    "kernel/test_shuffle_short_${VARIANT}" "kernel/test_shuffle_ushort_${VARIANT}"
    "kernel/test_shuffle_int_${VARIANT}" "kernel/test_shuffle_uint_${VARIANT}"
    "kernel/test_shuffle_long_${VARIANT}" "kernel/test_shuffle_ulong_${VARIANT}"
    "kernel/test_shuffle_half_${VARIANT}" "kernel/test_shuffle_float_${VARIANT}"
    "kernel/test_shuffle_double_${VARIANT}"
    "kernel/test_ucharn_${VARIANT}"
    "kernel/test_printf_${VARIANT}"
    "kernel/test_printf_vectors_${VARIANT}"
    "kernel/test_printf_vectors_ulongn_${VARIANT}"
    "kernel/test_sizeof_uint_${VARIANT}"

    PROPERTIES
      PROCESSORS 1
      COST 120
      DEPENDS "pocl_version_check"
      LABELS "internal;kernel")
endforeach()

######################################################################


######################################################################

#Comment out this one for now. Seems the test throws an exception, 
#and CTest doesn't think this is a WILL_FAIL, causing "Exception Other" 
#to be printed
#add_test_custom("${CMAKE_CURRENT_BINARY_DIR}/kernel" "kernel/test_block"
#                "test_block_expout.txt" "test_block")

#set_tests_properties("kernel/test_sizeof_uint" "kernel/test_block" "kernel/test_printf"
#  PROPERTIES
#    COST 2.0
#    PROCESSORS 1
#    DEPENDS "pocl_version_check")

# For the backend labels, we only need one variant per non-CPU backend enabled, so use loops.

# Label tests that work with CUDA backend
if(ENABLE_CUDA)
set_property(TEST
  "kernel/test_min_max_loopvec"
  "kernel/test_ldexp_loopvec"
  "kernel/test_length_distance_loopvec"
  "kernel/test_fmin_fmax_fma_loopvec"
  "kernel/test_frexp_modf_loopvec"
  "kernel/test_convert_sat_regression_loopvec"
  "kernel/test_short16_loopvec"
  "kernel/test_shuffle_char_loopvec"
  "kernel/test_shuffle_short_loopvec"
  "kernel/test_shuffle_ushort_loopvec"
  "kernel/test_shuffle_int_loopvec"
  "kernel/test_shuffle_uint_loopvec"
  "kernel/test_shuffle_float_loopvec"
  "kernel/test_shuffle_long_loopvec"
  "kernel/test_shuffle_ulong_loopvec"
  "kernel/test_shuffle_double_loopvec"
  "kernel/test_as_type_loopvec"
  "kernel/test_sizeof_uint_loopvec"
  APPEND PROPERTY LABELS "cuda")
endif()

set(EXTRA_TEST_VARIANT ${EXTRA_TESTS})
list(TRANSFORM EXTRA_TEST_VARIANT APPEND "_loopvec")

set_property(TEST
  "kernel/test_local_struct_array_loopvec"
  "kernel/test_printf_loopvec"
  "kernel/test_shuffle_char_loopvec"
  "kernel/test_shuffle_short_loopvec"
  "kernel/test_shuffle_ushort_loopvec"
  "kernel/test_shuffle_int_loopvec"
  "kernel/test_shuffle_uint_loopvec"
  "kernel/test_shuffle_float_loopvec"
  "kernel/test_shuffle_long_loopvec"
  "kernel/test_shuffle_ulong_loopvec"
  ${EXTRA_TEST_VARIANT}
  "kernel/test_as_type_loopvec"
  "kernel/test_convert_type_1_loopvec"
  "kernel/test_convert_type_2_loopvec"
  "kernel/test_convert_type_4_loopvec"
  "kernel/test_convert_type_8_loopvec"
  "kernel/test_convert_type_16_loopvec"
  "kernel/test_bitselect_loopvec"
  "kernel/test_min_max_loopvec"
  "kernel/test_length_distance_loopvec"
  "kernel/test_fmin_fmax_fma_loopvec"
  "kernel/test_convert_sat_regression_loopvec"
  "kernel/test_rotate_loopvec"
  "kernel/test_fabs_loopvec"
  "kernel/test_copy_signbit_loopvec"
  "kernel/test_ldexp_loopvec"
  "kernel/test_isnan_loopvec"
  "kernel/test_short16_loopvec"
  APPEND PROPERTY LABELS "hsa-native")

set_property(TEST
  "kernel/test_shuffle_char_loopvec"
  "kernel/test_shuffle_short_loopvec"
  "kernel/test_shuffle_ushort_loopvec"
  "kernel/test_shuffle_int_loopvec"
  "kernel/test_shuffle_uint_loopvec"
  "kernel/test_shuffle_float_loopvec"
  "kernel/test_shuffle_long_loopvec"
  "kernel/test_shuffle_ulong_loopvec"
  APPEND PROPERTY LABELS "proxy")

# disabled:
# kernel/test_convert_type_16 : crash
# all printf tests:
#  Test  #38: kernel/test_printf
#  Test  #39: kernel/test_printf_vectors
#  Test  #40: kernel/test_printf_vectors_ulongn
# kernel/test_fabs (incorrect values)
# kernel/test_ilogb (crash)
# works on iGPU but crashes in driver on Arc A750 (driver issue?):
#  "kernel/test_sampler_address_clamp_loopvec"
set_property(TEST
  "kernel/test_local_struct_array_loopvec"
  "kernel/test_shuffle_char_loopvec"
  "kernel/test_shuffle_short_loopvec"
  "kernel/test_shuffle_ushort_loopvec"
  "kernel/test_shuffle_int_loopvec"
  "kernel/test_shuffle_uint_loopvec"
  "kernel/test_shuffle_half_loopvec"
  "kernel/test_shuffle_float_loopvec"
  "kernel/test_shuffle_double_loopvec"
  "kernel/test_shuffle_long_loopvec"
  "kernel/test_shuffle_ulong_loopvec"
  "kernel/test_as_type_loopvec"
  "kernel/test_convert_type_1_loopvec"
  "kernel/test_convert_type_4_loopvec"
  "kernel/test_bitselect_loopvec"
  "kernel/test_hadd_loopvec"
  "kernel/test_min_max_loopvec"
  "kernel/test_length_distance_loopvec"
  "kernel/test_fmin_fmax_fma_loopvec"
  "kernel/test_frexp_modf_loopvec"
  "kernel/test_convert_sat_regression_loopvec"
  "kernel/test_rotate_loopvec"
  "kernel/test_copy_signbit_loopvec"
  "kernel/test_ldexp_loopvec"
  "kernel/test_isnan_loopvec"
  "kernel/test_short16_loopvec"
  "kernel/test_sizeof_uint_loopvec"
  "kernel/test_ucharn_loopvec"
  "kernel/test_image_query_funcs_loopvec"
  APPEND PROPERTY LABELS "level0")
