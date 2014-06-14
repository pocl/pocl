
#AM_CPPFLAGS = -I$(top_srcdir)/fix-include -I$(top_srcdir)/include -DSRCDIR='"$(abs_srcdir)"'
add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")


#kernel_CFLAGS = -std=c99 @OPENCL_CFLAGS@
#smapler_address_clamp_CFLAGS = -std=c99 @OPENCL_CFLAGS@
#image_query_funcs_CFLAGS = -std=c99 @OPENCL_CFLAGS@
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 ${OPENCL_CFLAGS}")

#kernel_LDADD = -lm @OPENCL_LIBS@ ../../lib/poclu/libpoclu.la
#sampler_address_clamp_LDADD = -lm @OPENCL_LIBS@ ../../lib/poclu/libpoclu.la
#image_query_funcs_LDADD = -lm @OPENCL_LIBS@ ../../lib/poclu/libpoclu.la
# -> POCLU_LINK_OPTIONS in toplevel cmake

######################################################################
add_executable("kernel" "kernel.c") # test_as_type.cl test_bitselect.cl test_convert_sat_regression.cl test_convert_type_*.cl test_fabs.cl test_fmin_fmax_fma.cl test_hadd.cl test_min_max.cl test_rotate.cl test_short16.cl test_sizeof.cl test_block.cl test_printf.cl
target_link_libraries("kernel" ${POCLU_LINK_OPTIONS})

add_test("kernel/test_as_type" "kernel" "test_as_type")

add_test("kernel/test_convert_type_1" "kernel" "test_convert_type_1")

add_test("kernel/test_convert_type_2" "kernel" "test_convert_type_2")

add_test("kernel/test_convert_type_4" "kernel" "test_convert_type_4")

add_test("kernel/test_convert_type_8" "kernel" "test_convert_type_8")

add_test("kernel/test_convert_type_16" "kernel" "test_convert_type_16")

add_test("kernel/test_bitselect" "kernel" "test_bitselect")

add_test("kernel/test_hadd" "kernel" "test_hadd")

set_tests_properties( "kernel/test_as_type" "kernel/test_bitselect"
  "kernel/test_convert_type_1" "kernel/test_convert_type_2" "kernel/test_convert_type_4"
  "kernel/test_convert_type_8" "kernel/test_convert_type_16" "kernel/test_hadd"
  PROPERTIES
    COST 40.0
    PASS_REGULAR_EXPRESSION "\nOK\n"
    PROCESSORS 1
    DEPENDS "pocl_version_check")

# TODO check if this works
set_tests_properties("kernel/test_bitselect"
  PROPERTIES
    ENVIRONMENT "POCL_VECTORIZE_WORK_GROUPS=0")

add_test("kernel/test_min_max" "kernel" "test_min_max")

add_test("kernel/test_fmin_fmax_fma" "kernel" "test_fmin_fmax_fma")

add_test("kernel/test_convert_sat_regression" "kernel" "test_convert_sat_regression")

add_test("kernel/test_rotate" "kernel" "test_rotate")

add_test("kernel/test_fabs" "kernel" "test_fabs")

add_test("kernel/test_short16" "kernel" "test_short16")

set_tests_properties("kernel/test_min_max" "kernel/test_fmin_fmax_fma"
  "kernel/test_convert_sat_regression"   "kernel/test_fabs"
  "kernel/test_rotate" "kernel/test_short16"
  PROPERTIES
    COST 4.0
    PASS_REGULAR_EXPRESSION "\nOK\n"
    PROCESSORS 1
    DEPENDS "pocl_version_check")

######################################################################

add_executable("sampler_address_clamp" "sampler_address_clamp.c") #test_sampler_address_clamp.cl
target_link_libraries("sampler_address_clamp" ${POCLU_LINK_OPTIONS})

add_executable("image_query_funcs" "image_query_funcs.c") #test_image_query_funcs.cl
target_link_libraries("image_query_funcs" ${POCLU_LINK_OPTIONS})

add_test("kernel/test_sampler_address_clamp" "sampler_address_clamp")

add_test("kernel/test_image_query_funcs" "image_query_funcs")

set_tests_properties( "kernel/test_sampler_address_clamp"
  "kernel/test_image_query_funcs"
  PROPERTIES
    COST 4.0
    PASS_REGULAR_EXPRESSION "\nOK\n"
    PROCESSORS 1
    DEPENDS "pocl_version_check")

######################################################################

add_executable("test_shuffle" "test_shuffle.cc")
target_link_libraries("test_shuffle" ${POCLU_LINK_OPTIONS})

add_test("kernel/test_shuffle_char" "test_shuffle" "char")

add_test("kernel/test_shuffle_short" "test_shuffle" "short")

add_test("kernel/test_shuffle_ushort" "test_shuffle" "ushort")

add_test("kernel/test_shuffle_half" "test_shuffle" "half")

add_test("kernel/test_shuffle_int" "test_shuffle" "int")

add_test("kernel/test_shuffle_uint" "test_shuffle" "uint")

add_test("kernel/test_shuffle_float" "test_shuffle" "float")

add_test("kernel/test_shuffle_long" "test_shuffle" "long")

add_test("kernel/test_shuffle_ulong" "test_shuffle" "ulong")

add_test("kernel/test_shuffle_double" "test_shuffle" "double")

set_tests_properties("kernel/test_shuffle_char" "kernel/test_shuffle_short"
  "kernel/test_shuffle_ushort"   "kernel/test_shuffle_half"
  "kernel/test_shuffle_int"   "kernel/test_shuffle_uint"
  "kernel/test_shuffle_float"   "kernel/test_shuffle_long"
  "kernel/test_shuffle_ulong"  "kernel/test_shuffle_double"
  PROPERTIES
    COST 77
    PASS_REGULAR_EXPRESSION "OK\n"
    PROCESSORS 1
    DEPENDS "pocl_version_check")

######################################################################



