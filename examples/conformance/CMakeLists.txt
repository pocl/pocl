#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2017 Michal Babej / Tampere University of Technology
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

set(TS_NAME "conformance")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")

if(NOT HAVE_GIT)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires git to checkout sources")
elseif(NOT TESTS_USE_ICD)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires ocl-icd")

elseif(NOT MAKE_COMMAND)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires make to build")

else()

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  GIT_REPOSITORY "https://github.com/KhronosGroup/OpenCL-CTS"
  GIT_TAG "cl12_trunk"
  # PATCH_COMMAND patch -p1 -i "${CMAKE_CURRENT_SOURCE_DIR}/squashed.patch"
  CONFIGURE_COMMAND /bin/true
  BUILD_COMMAND "${MAKE_PROGRAM}" -C "test_conformance" "COMPILERFLAGS=-Wno-narrowing -Wno-deprecated-declarations"
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND /bin/true
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

add_test(NAME "${TS_NAME}_allocations_buffer"
         COMMAND "${TS_BUILDDIR}/test_conformance/allocations/test_allocations" buffer buffer_non_blocking)
add_test(NAME "${TS_NAME}_allocations_image"
         COMMAND "${TS_BUILDDIR}/test_conformance/allocations/test_allocations" image2d_read  image2d_write  image2d_read_non_blocking  image2d_write_non_blocking)

add_test(NAME "${TS_NAME}_api"
         COMMAND "${TS_BUILDDIR}/test_conformance/api/test_api" )

add_test(NAME "${TS_NAME}_atomics"
         COMMAND "${TS_BUILDDIR}/test_conformance/atomics/test_atomics" )

add_test(NAME "${TS_NAME}_basic_math"
         COMMAND "${TS_BUILDDIR}/test_conformance/basic/test_basic" fpmath_float  fpmath_float2  fpmath_float4  intmath_int  intmath_int2  intmath_int4  intmath_long  intmath_long2  intmath_long4  int2float  float2int)
add_test(NAME "${TS_NAME}_basic_s2v"
         COMMAND "${TS_BUILDDIR}/test_conformance/basic/test_basic" explicit_s2v_bool  explicit_s2v_char  explicit_s2v_uchar  explicit_s2v_short  explicit_s2v_ushort  explicit_s2v_int  explicit_s2v_uint  explicit_s2v_long  explicit_s2v_ulong  explicit_s2v_float  explicit_s2v_double)
add_test(NAME "${TS_NAME}_basic_memory"
         COMMAND "${TS_BUILDDIR}/test_conformance/basic/test_basic" vload_global  vload_local  vload_constant  vload_private  vstore_global  vstore_local  vstore_private  bufferreadwriterect  arrayreadwrite  arraycopy  enqueue_map_buffer  async_copy_global_to_local  async_copy_local_to_global  async_strided_copy_global_to_local  async_strided_copy_local_to_global  kernel_memory_alignment_local  kernel_memory_alignment_global  kernel_memory_alignment_constant  kernel_memory_alignment_private)
add_test(NAME "${TS_NAME}_basic_image"
         COMMAND "${TS_BUILDDIR}/test_conformance/basic/test_basic" readimage  readimage_int16  readimage_fp32  writeimage  writeimage_int16  writeimage_fp32  image_r8  imagereadwrite  imagereadwrite3d  readimage3d  readimage3d_int16  readimage3d_fp32  imagearraycopy  imagearraycopy3d  imagecopy  imagecopy3d  imagerandomcopy  arrayimagecopy  arrayimagecopy3d  imagenpot  imagedim_pow2  imagedim_non_pow2  image_param  image_multipass_integer_coord  image_multipass_float_coord  enqueue_map_image)
add_test(NAME "${TS_NAME}_basic_other"
         COMMAND "${TS_BUILDDIR}/test_conformance/basic/test_basic" mri_one  mri_multiple  barrier  createkernelsinprogram  work_item_functions  astype  prefetch  kernel_call_kernel_function  host_numeric_constants  kernel_numeric_constants  kernel_limit_constants  kernel_preprocessor_macros  parameter_types  vector_creation  vec_type_hint  global_work_offsets  get_global_offset  hostptr  hiloeo  if  sizeof  loop  pointer_cast  local_arg_def  local_kernel_def  local_kernel_scope  constant  constant_source)

add_test(NAME "${TS_NAME}_buffers_read_async"
         COMMAND "${TS_BUILDDIR}/test_conformance/buffers/test_buffers" buffer_read_async_int  buffer_read_async_uint  buffer_read_async_long  buffer_read_async_ulong  buffer_read_async_short  buffer_read_async_ushort  buffer_read_async_char  buffer_read_async_uchar  buffer_read_async_float)
add_test(NAME "${TS_NAME}_buffers_write_async"
         COMMAND "${TS_BUILDDIR}/test_conformance/buffers/test_buffers"  buffer_write_async_int  buffer_write_async_uint  buffer_write_async_short  buffer_write_async_ushort  buffer_write_async_char  buffer_write_async_uchar  buffer_write_async_float  buffer_write_async_long  buffer_write_async_ulong)

add_test(NAME "${TS_NAME}_buffers_read_array_barrier"
         COMMAND "${TS_BUILDDIR}/test_conformance/buffers/test_buffers" buffer_read_array_barrier_int  buffer_read_array_barrier_uint  buffer_read_array_barrier_long  buffer_read_array_barrier_ulong  buffer_read_array_barrier_short  buffer_read_array_barrier_ushort  buffer_read_array_barrier_char  buffer_read_array_barrier_uchar  buffer_read_array_barrier_float)

add_test(NAME "${TS_NAME}_buffers_read"
         COMMAND "${TS_BUILDDIR}/test_conformance/buffers/test_buffers" buffer_read_int  buffer_read_uint  buffer_read_long  buffer_read_ulong  buffer_read_short  buffer_read_ushort  buffer_read_float  buffer_read_half  buffer_read_char  buffer_read_uchar  buffer_read_struct  buffer_read_random_size)
add_test(NAME "${TS_NAME}_buffers_write"
         COMMAND "${TS_BUILDDIR}/test_conformance/buffers/test_buffers" buffer_write_int  buffer_write_uint  buffer_write_short  buffer_write_ushort  buffer_write_char  buffer_write_uchar  buffer_write_float  buffer_write_half  buffer_write_long  buffer_write_ulong  buffer_write_struct)
add_test(NAME "${TS_NAME}_buffers_fill"
         COMMAND "${TS_BUILDDIR}/test_conformance/buffers/test_buffers"   buffer_fill_int  buffer_fill_uint  buffer_fill_short  buffer_fill_ushort  buffer_fill_char  buffer_fill_uchar  buffer_fill_long  buffer_fill_ulong  buffer_fill_float  buffer_fill_struct)

add_test(NAME "${TS_NAME}_buffers_var1"
         COMMAND "${TS_BUILDDIR}/test_conformance/buffers/test_buffers" buffer_copy  buffer_partial_copy  mem_read_write_flags  mem_write_only_flags  mem_read_only_flags  mem_copy_host_flags  mem_alloc_ref_flags  array_info_size)
add_test(NAME "${TS_NAME}_buffers_var2"
         COMMAND "${TS_BUILDDIR}/test_conformance/buffers/test_buffers" sub_buffers_read_write  sub_buffers_read_write_dual_devices  sub_buffers_overlapping  buffer_migrate  image_migrate)

add_test(NAME "${TS_NAME}_buffers_map_read"
         COMMAND "${TS_BUILDDIR}/test_conformance/buffers/test_buffers" buffer_map_read_int  buffer_map_read_uint  buffer_map_read_long  buffer_map_read_ulong  buffer_map_read_short  buffer_map_read_ushort  buffer_map_read_char  buffer_map_read_uchar  buffer_map_read_float  buffer_map_read_struct)
add_test(NAME "${TS_NAME}_buffers_map_write"
         COMMAND "${TS_BUILDDIR}/test_conformance/buffers/test_buffers" buffer_map_write_int  buffer_map_write_uint  buffer_map_write_long  buffer_map_write_ulong  buffer_map_write_short  buffer_map_write_ushort  buffer_map_write_char  buffer_map_write_uchar  buffer_map_write_float  buffer_map_write_struct)

add_test(NAME "${TS_NAME}_commonfns"
         COMMAND "${TS_BUILDDIR}/test_conformance/commonfns/test_commonfns" )

add_test(NAME "${TS_NAME}_compiler"
         COMMAND "${TS_BUILDDIR}/test_conformance/compiler/test_compiler" )

add_test(NAME "${TS_NAME}_computeinfo"
         COMMAND "${TS_BUILDDIR}/test_conformance/computeinfo/computeinfo" )

add_test(NAME "${TS_NAME}_contractions"
         COMMAND "${TS_BUILDDIR}/test_conformance/contractions/contractions" )

add_test(NAME "${TS_NAME}_conversion_uchar_uchar"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uchar_uchar uchar_sat_uchar uchar_rte_uchar uchar_sat_rte_uchar uchar_rtp_uchar uchar_sat_rtp_uchar uchar_rtn_uchar uchar_sat_rtn_uchar uchar_rtz_uchar uchar_sat_rtz_uchar )

add_test(NAME "${TS_NAME}_conversion_char_uchar"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" char_uchar char_sat_uchar char_rte_uchar char_sat_rte_uchar char_rtp_uchar char_sat_rtp_uchar char_rtn_uchar char_sat_rtn_uchar char_rtz_uchar char_sat_rtz_uchar )

add_test(NAME "${TS_NAME}_conversion_ushort_uchar"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ushort_uchar ushort_sat_uchar ushort_rte_uchar ushort_sat_rte_uchar ushort_rtp_uchar ushort_sat_rtp_uchar ushort_rtn_uchar ushort_sat_rtn_uchar ushort_rtz_uchar ushort_sat_rtz_uchar )

add_test(NAME "${TS_NAME}_conversion_short_uchar"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" short_uchar short_sat_uchar short_rte_uchar short_sat_rte_uchar short_rtp_uchar short_sat_rtp_uchar short_rtn_uchar short_sat_rtn_uchar short_rtz_uchar short_sat_rtz_uchar )

add_test(NAME "${TS_NAME}_conversion_uint_uchar"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uint_uchar uint_sat_uchar uint_rte_uchar uint_sat_rte_uchar uint_rtp_uchar uint_sat_rtp_uchar uint_rtn_uchar uint_sat_rtn_uchar uint_rtz_uchar uint_sat_rtz_uchar )

add_test(NAME "${TS_NAME}_conversion_int_uchar"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" int_uchar int_sat_uchar int_rte_uchar int_sat_rte_uchar int_rtp_uchar int_sat_rtp_uchar int_rtn_uchar int_sat_rtn_uchar int_rtz_uchar int_sat_rtz_uchar )

add_test(NAME "${TS_NAME}_conversion_float_uchar"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" float_uchar float_sat_uchar float_rte_uchar float_sat_rte_uchar float_rtp_uchar float_sat_rtp_uchar float_rtn_uchar float_sat_rtn_uchar float_rtz_uchar float_sat_rtz_uchar )

add_test(NAME "${TS_NAME}_conversion_double_uchar"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" double_uchar double_sat_uchar double_rte_uchar double_sat_rte_uchar double_rtp_uchar double_sat_rtp_uchar double_rtn_uchar double_sat_rtn_uchar double_rtz_uchar double_sat_rtz_uchar )

add_test(NAME "${TS_NAME}_conversion_ulong_uchar"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ulong_uchar ulong_sat_uchar ulong_rte_uchar ulong_sat_rte_uchar ulong_rtp_uchar ulong_sat_rtp_uchar ulong_rtn_uchar ulong_sat_rtn_uchar ulong_rtz_uchar ulong_sat_rtz_uchar )

add_test(NAME "${TS_NAME}_conversion_long_uchar"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" long_uchar long_sat_uchar long_rte_uchar long_sat_rte_uchar long_rtp_uchar long_sat_rtp_uchar long_rtn_uchar long_sat_rtn_uchar long_rtz_uchar long_sat_rtz_uchar )

add_test(NAME "${TS_NAME}_conversion_uchar_char"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uchar_char uchar_sat_char uchar_rte_char uchar_sat_rte_char uchar_rtp_char uchar_sat_rtp_char uchar_rtn_char uchar_sat_rtn_char uchar_rtz_char uchar_sat_rtz_char )

add_test(NAME "${TS_NAME}_conversion_char_char"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" char_char char_sat_char char_rte_char char_sat_rte_char char_rtp_char char_sat_rtp_char char_rtn_char char_sat_rtn_char char_rtz_char char_sat_rtz_char )

add_test(NAME "${TS_NAME}_conversion_ushort_char"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ushort_char ushort_sat_char ushort_rte_char ushort_sat_rte_char ushort_rtp_char ushort_sat_rtp_char ushort_rtn_char ushort_sat_rtn_char ushort_rtz_char ushort_sat_rtz_char )

add_test(NAME "${TS_NAME}_conversion_short_char"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" short_char short_sat_char short_rte_char short_sat_rte_char short_rtp_char short_sat_rtp_char short_rtn_char short_sat_rtn_char short_rtz_char short_sat_rtz_char )

add_test(NAME "${TS_NAME}_conversion_uint_char"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uint_char uint_sat_char uint_rte_char uint_sat_rte_char uint_rtp_char uint_sat_rtp_char uint_rtn_char uint_sat_rtn_char uint_rtz_char uint_sat_rtz_char )

add_test(NAME "${TS_NAME}_conversion_int_char"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" int_char int_sat_char int_rte_char int_sat_rte_char int_rtp_char int_sat_rtp_char int_rtn_char int_sat_rtn_char int_rtz_char int_sat_rtz_char )

add_test(NAME "${TS_NAME}_conversion_float_char"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" float_char float_sat_char float_rte_char float_sat_rte_char float_rtp_char float_sat_rtp_char float_rtn_char float_sat_rtn_char float_rtz_char float_sat_rtz_char )

add_test(NAME "${TS_NAME}_conversion_double_char"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" double_char double_sat_char double_rte_char double_sat_rte_char double_rtp_char double_sat_rtp_char double_rtn_char double_sat_rtn_char double_rtz_char double_sat_rtz_char )

add_test(NAME "${TS_NAME}_conversion_ulong_char"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ulong_char ulong_sat_char ulong_rte_char ulong_sat_rte_char ulong_rtp_char ulong_sat_rtp_char ulong_rtn_char ulong_sat_rtn_char ulong_rtz_char ulong_sat_rtz_char )

add_test(NAME "${TS_NAME}_conversion_long_char"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" long_char long_sat_char long_rte_char long_sat_rte_char long_rtp_char long_sat_rtp_char long_rtn_char long_sat_rtn_char long_rtz_char long_sat_rtz_char )

add_test(NAME "${TS_NAME}_conversion_uchar_ushort"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uchar_ushort uchar_sat_ushort uchar_rte_ushort uchar_sat_rte_ushort uchar_rtp_ushort uchar_sat_rtp_ushort uchar_rtn_ushort uchar_sat_rtn_ushort uchar_rtz_ushort uchar_sat_rtz_ushort )

add_test(NAME "${TS_NAME}_conversion_char_ushort"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" char_ushort char_sat_ushort char_rte_ushort char_sat_rte_ushort char_rtp_ushort char_sat_rtp_ushort char_rtn_ushort char_sat_rtn_ushort char_rtz_ushort char_sat_rtz_ushort )

add_test(NAME "${TS_NAME}_conversion_ushort_ushort"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ushort_ushort ushort_sat_ushort ushort_rte_ushort ushort_sat_rte_ushort ushort_rtp_ushort ushort_sat_rtp_ushort ushort_rtn_ushort ushort_sat_rtn_ushort ushort_rtz_ushort ushort_sat_rtz_ushort )

add_test(NAME "${TS_NAME}_conversion_short_ushort"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" short_ushort short_sat_ushort short_rte_ushort short_sat_rte_ushort short_rtp_ushort short_sat_rtp_ushort short_rtn_ushort short_sat_rtn_ushort short_rtz_ushort short_sat_rtz_ushort )

add_test(NAME "${TS_NAME}_conversion_uint_ushort"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uint_ushort uint_sat_ushort uint_rte_ushort uint_sat_rte_ushort uint_rtp_ushort uint_sat_rtp_ushort uint_rtn_ushort uint_sat_rtn_ushort uint_rtz_ushort uint_sat_rtz_ushort )

add_test(NAME "${TS_NAME}_conversion_int_ushort"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" int_ushort int_sat_ushort int_rte_ushort int_sat_rte_ushort int_rtp_ushort int_sat_rtp_ushort int_rtn_ushort int_sat_rtn_ushort int_rtz_ushort int_sat_rtz_ushort )

add_test(NAME "${TS_NAME}_conversion_float_ushort"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" float_ushort float_sat_ushort float_rte_ushort float_sat_rte_ushort float_rtp_ushort float_sat_rtp_ushort float_rtn_ushort float_sat_rtn_ushort float_rtz_ushort float_sat_rtz_ushort )

add_test(NAME "${TS_NAME}_conversion_double_ushort"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" double_ushort double_sat_ushort double_rte_ushort double_sat_rte_ushort double_rtp_ushort double_sat_rtp_ushort double_rtn_ushort double_sat_rtn_ushort double_rtz_ushort double_sat_rtz_ushort )

add_test(NAME "${TS_NAME}_conversion_ulong_ushort"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ulong_ushort ulong_sat_ushort ulong_rte_ushort ulong_sat_rte_ushort ulong_rtp_ushort ulong_sat_rtp_ushort ulong_rtn_ushort ulong_sat_rtn_ushort ulong_rtz_ushort ulong_sat_rtz_ushort )

add_test(NAME "${TS_NAME}_conversion_long_ushort"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" long_ushort long_sat_ushort long_rte_ushort long_sat_rte_ushort long_rtp_ushort long_sat_rtp_ushort long_rtn_ushort long_sat_rtn_ushort long_rtz_ushort long_sat_rtz_ushort )

add_test(NAME "${TS_NAME}_conversion_uchar_short"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uchar_short uchar_sat_short uchar_rte_short uchar_sat_rte_short uchar_rtp_short uchar_sat_rtp_short uchar_rtn_short uchar_sat_rtn_short uchar_rtz_short uchar_sat_rtz_short )

add_test(NAME "${TS_NAME}_conversion_char_short"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" char_short char_sat_short char_rte_short char_sat_rte_short char_rtp_short char_sat_rtp_short char_rtn_short char_sat_rtn_short char_rtz_short char_sat_rtz_short )

add_test(NAME "${TS_NAME}_conversion_ushort_short"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ushort_short ushort_sat_short ushort_rte_short ushort_sat_rte_short ushort_rtp_short ushort_sat_rtp_short ushort_rtn_short ushort_sat_rtn_short ushort_rtz_short ushort_sat_rtz_short )

add_test(NAME "${TS_NAME}_conversion_short_short"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" short_short short_sat_short short_rte_short short_sat_rte_short short_rtp_short short_sat_rtp_short short_rtn_short short_sat_rtn_short short_rtz_short short_sat_rtz_short )

add_test(NAME "${TS_NAME}_conversion_uint_short"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uint_short uint_sat_short uint_rte_short uint_sat_rte_short uint_rtp_short uint_sat_rtp_short uint_rtn_short uint_sat_rtn_short uint_rtz_short uint_sat_rtz_short )

add_test(NAME "${TS_NAME}_conversion_int_short"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" int_short int_sat_short int_rte_short int_sat_rte_short int_rtp_short int_sat_rtp_short int_rtn_short int_sat_rtn_short int_rtz_short int_sat_rtz_short )

add_test(NAME "${TS_NAME}_conversion_float_short"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" float_short float_sat_short float_rte_short float_sat_rte_short float_rtp_short float_sat_rtp_short float_rtn_short float_sat_rtn_short float_rtz_short float_sat_rtz_short )

add_test(NAME "${TS_NAME}_conversion_double_short"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" double_short double_sat_short double_rte_short double_sat_rte_short double_rtp_short double_sat_rtp_short double_rtn_short double_sat_rtn_short double_rtz_short double_sat_rtz_short )

add_test(NAME "${TS_NAME}_conversion_ulong_short"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ulong_short ulong_sat_short ulong_rte_short ulong_sat_rte_short ulong_rtp_short ulong_sat_rtp_short ulong_rtn_short ulong_sat_rtn_short ulong_rtz_short ulong_sat_rtz_short )

add_test(NAME "${TS_NAME}_conversion_long_short"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" long_short long_sat_short long_rte_short long_sat_rte_short long_rtp_short long_sat_rtp_short long_rtn_short long_sat_rtn_short long_rtz_short long_sat_rtz_short )

add_test(NAME "${TS_NAME}_conversion_uchar_uint"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uchar_uint uchar_sat_uint uchar_rte_uint uchar_sat_rte_uint uchar_rtp_uint uchar_sat_rtp_uint uchar_rtn_uint uchar_sat_rtn_uint uchar_rtz_uint uchar_sat_rtz_uint )

add_test(NAME "${TS_NAME}_conversion_char_uint"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" char_uint char_sat_uint char_rte_uint char_sat_rte_uint char_rtp_uint char_sat_rtp_uint char_rtn_uint char_sat_rtn_uint char_rtz_uint char_sat_rtz_uint )

add_test(NAME "${TS_NAME}_conversion_ushort_uint"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ushort_uint ushort_sat_uint ushort_rte_uint ushort_sat_rte_uint ushort_rtp_uint ushort_sat_rtp_uint ushort_rtn_uint ushort_sat_rtn_uint ushort_rtz_uint ushort_sat_rtz_uint )

add_test(NAME "${TS_NAME}_conversion_short_uint"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" short_uint short_sat_uint short_rte_uint short_sat_rte_uint short_rtp_uint short_sat_rtp_uint short_rtn_uint short_sat_rtn_uint short_rtz_uint short_sat_rtz_uint )

add_test(NAME "${TS_NAME}_conversion_uint_uint"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uint_uint uint_sat_uint uint_rte_uint uint_sat_rte_uint uint_rtp_uint uint_sat_rtp_uint uint_rtn_uint uint_sat_rtn_uint uint_rtz_uint uint_sat_rtz_uint )

add_test(NAME "${TS_NAME}_conversion_int_uint"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" int_uint int_sat_uint int_rte_uint int_sat_rte_uint int_rtp_uint int_sat_rtp_uint int_rtn_uint int_sat_rtn_uint int_rtz_uint int_sat_rtz_uint )

add_test(NAME "${TS_NAME}_conversion_float_uint"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" float_uint float_sat_uint float_rte_uint float_sat_rte_uint float_rtp_uint float_sat_rtp_uint float_rtn_uint float_sat_rtn_uint float_rtz_uint float_sat_rtz_uint )

add_test(NAME "${TS_NAME}_conversion_double_uint"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" double_uint double_sat_uint double_rte_uint double_sat_rte_uint double_rtp_uint double_sat_rtp_uint double_rtn_uint double_sat_rtn_uint double_rtz_uint double_sat_rtz_uint )

add_test(NAME "${TS_NAME}_conversion_ulong_uint"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ulong_uint ulong_sat_uint ulong_rte_uint ulong_sat_rte_uint ulong_rtp_uint ulong_sat_rtp_uint ulong_rtn_uint ulong_sat_rtn_uint ulong_rtz_uint ulong_sat_rtz_uint )

add_test(NAME "${TS_NAME}_conversion_long_uint"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" long_uint long_sat_uint long_rte_uint long_sat_rte_uint long_rtp_uint long_sat_rtp_uint long_rtn_uint long_sat_rtn_uint long_rtz_uint long_sat_rtz_uint )

add_test(NAME "${TS_NAME}_conversion_uchar_int"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uchar_int uchar_sat_int uchar_rte_int uchar_sat_rte_int uchar_rtp_int uchar_sat_rtp_int uchar_rtn_int uchar_sat_rtn_int uchar_rtz_int uchar_sat_rtz_int )

add_test(NAME "${TS_NAME}_conversion_char_int"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" char_int char_sat_int char_rte_int char_sat_rte_int char_rtp_int char_sat_rtp_int char_rtn_int char_sat_rtn_int char_rtz_int char_sat_rtz_int )

add_test(NAME "${TS_NAME}_conversion_ushort_int"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ushort_int ushort_sat_int ushort_rte_int ushort_sat_rte_int ushort_rtp_int ushort_sat_rtp_int ushort_rtn_int ushort_sat_rtn_int ushort_rtz_int ushort_sat_rtz_int )

add_test(NAME "${TS_NAME}_conversion_short_int"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" short_int short_sat_int short_rte_int short_sat_rte_int short_rtp_int short_sat_rtp_int short_rtn_int short_sat_rtn_int short_rtz_int short_sat_rtz_int )

add_test(NAME "${TS_NAME}_conversion_uint_int"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uint_int uint_sat_int uint_rte_int uint_sat_rte_int uint_rtp_int uint_sat_rtp_int uint_rtn_int uint_sat_rtn_int uint_rtz_int uint_sat_rtz_int )

add_test(NAME "${TS_NAME}_conversion_int_int"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" int_int int_sat_int int_rte_int int_sat_rte_int int_rtp_int int_sat_rtp_int int_rtn_int int_sat_rtn_int int_rtz_int int_sat_rtz_int )

add_test(NAME "${TS_NAME}_conversion_float_int"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" float_int float_sat_int float_rte_int float_sat_rte_int float_rtp_int float_sat_rtp_int float_rtn_int float_sat_rtn_int float_rtz_int float_sat_rtz_int )

add_test(NAME "${TS_NAME}_conversion_double_int"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" double_int double_sat_int double_rte_int double_sat_rte_int double_rtp_int double_sat_rtp_int double_rtn_int double_sat_rtn_int double_rtz_int double_sat_rtz_int )

add_test(NAME "${TS_NAME}_conversion_ulong_int"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ulong_int ulong_sat_int ulong_rte_int ulong_sat_rte_int ulong_rtp_int ulong_sat_rtp_int ulong_rtn_int ulong_sat_rtn_int ulong_rtz_int ulong_sat_rtz_int )

add_test(NAME "${TS_NAME}_conversion_long_int"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" long_int long_sat_int long_rte_int long_sat_rte_int long_rtp_int long_sat_rtp_int long_rtn_int long_sat_rtn_int long_rtz_int long_sat_rtz_int )

add_test(NAME "${TS_NAME}_conversion_uchar_float"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uchar_float uchar_sat_float uchar_rte_float uchar_sat_rte_float uchar_rtp_float uchar_sat_rtp_float uchar_rtn_float uchar_sat_rtn_float uchar_rtz_float uchar_sat_rtz_float )

add_test(NAME "${TS_NAME}_conversion_char_float"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" char_float char_sat_float char_rte_float char_sat_rte_float char_rtp_float char_sat_rtp_float char_rtn_float char_sat_rtn_float char_rtz_float char_sat_rtz_float )

add_test(NAME "${TS_NAME}_conversion_ushort_float"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ushort_float ushort_sat_float ushort_rte_float ushort_sat_rte_float ushort_rtp_float ushort_sat_rtp_float ushort_rtn_float ushort_sat_rtn_float ushort_rtz_float ushort_sat_rtz_float )

add_test(NAME "${TS_NAME}_conversion_short_float"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" short_float short_sat_float short_rte_float short_sat_rte_float short_rtp_float short_sat_rtp_float short_rtn_float short_sat_rtn_float short_rtz_float short_sat_rtz_float )

add_test(NAME "${TS_NAME}_conversion_uint_float"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uint_float uint_sat_float uint_rte_float uint_sat_rte_float uint_rtp_float uint_sat_rtp_float uint_rtn_float uint_sat_rtn_float uint_rtz_float uint_sat_rtz_float )

add_test(NAME "${TS_NAME}_conversion_int_float"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" int_float int_sat_float int_rte_float int_sat_rte_float int_rtp_float int_sat_rtp_float int_rtn_float int_sat_rtn_float int_rtz_float int_sat_rtz_float )

add_test(NAME "${TS_NAME}_conversion_float_float"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" float_float float_sat_float float_rte_float float_sat_rte_float float_rtp_float float_sat_rtp_float float_rtn_float float_sat_rtn_float float_rtz_float float_sat_rtz_float )

add_test(NAME "${TS_NAME}_conversion_double_float"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" double_float double_sat_float double_rte_float double_sat_rte_float double_rtp_float double_sat_rtp_float double_rtn_float double_sat_rtn_float double_rtz_float double_sat_rtz_float )

add_test(NAME "${TS_NAME}_conversion_ulong_float"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ulong_float ulong_sat_float ulong_rte_float ulong_sat_rte_float ulong_rtp_float ulong_sat_rtp_float ulong_rtn_float ulong_sat_rtn_float ulong_rtz_float ulong_sat_rtz_float )

add_test(NAME "${TS_NAME}_conversion_long_float"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" long_float long_sat_float long_rte_float long_sat_rte_float long_rtp_float long_sat_rtp_float long_rtn_float long_sat_rtn_float long_rtz_float long_sat_rtz_float )

add_test(NAME "${TS_NAME}_conversion_uchar_double"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uchar_double uchar_sat_double uchar_rte_double uchar_sat_rte_double uchar_rtp_double uchar_sat_rtp_double uchar_rtn_double uchar_sat_rtn_double uchar_rtz_double uchar_sat_rtz_double )

add_test(NAME "${TS_NAME}_conversion_char_double"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" char_double char_sat_double char_rte_double char_sat_rte_double char_rtp_double char_sat_rtp_double char_rtn_double char_sat_rtn_double char_rtz_double char_sat_rtz_double )

add_test(NAME "${TS_NAME}_conversion_ushort_double"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ushort_double ushort_sat_double ushort_rte_double ushort_sat_rte_double ushort_rtp_double ushort_sat_rtp_double ushort_rtn_double ushort_sat_rtn_double ushort_rtz_double ushort_sat_rtz_double )

add_test(NAME "${TS_NAME}_conversion_short_double"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" short_double short_sat_double short_rte_double short_sat_rte_double short_rtp_double short_sat_rtp_double short_rtn_double short_sat_rtn_double short_rtz_double short_sat_rtz_double )

add_test(NAME "${TS_NAME}_conversion_uint_double"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uint_double uint_sat_double uint_rte_double uint_sat_rte_double uint_rtp_double uint_sat_rtp_double uint_rtn_double uint_sat_rtn_double uint_rtz_double uint_sat_rtz_double )

add_test(NAME "${TS_NAME}_conversion_int_double"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" int_double int_sat_double int_rte_double int_sat_rte_double int_rtp_double int_sat_rtp_double int_rtn_double int_sat_rtn_double int_rtz_double int_sat_rtz_double )

add_test(NAME "${TS_NAME}_conversion_float_double"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" float_double float_sat_double float_rte_double float_sat_rte_double float_rtp_double float_sat_rtp_double float_rtn_double float_sat_rtn_double float_rtz_double float_sat_rtz_double )

add_test(NAME "${TS_NAME}_conversion_double_double"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" double_double double_sat_double double_rte_double double_sat_rte_double double_rtp_double double_sat_rtp_double double_rtn_double double_sat_rtn_double double_rtz_double double_sat_rtz_double )

add_test(NAME "${TS_NAME}_conversion_ulong_double"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ulong_double ulong_sat_double ulong_rte_double ulong_sat_rte_double ulong_rtp_double ulong_sat_rtp_double ulong_rtn_double ulong_sat_rtn_double ulong_rtz_double ulong_sat_rtz_double )

add_test(NAME "${TS_NAME}_conversion_long_double"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" long_double long_sat_double long_rte_double long_sat_rte_double long_rtp_double long_sat_rtp_double long_rtn_double long_sat_rtn_double long_rtz_double long_sat_rtz_double )

add_test(NAME "${TS_NAME}_conversion_uchar_ulong"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uchar_ulong uchar_sat_ulong uchar_rte_ulong uchar_sat_rte_ulong uchar_rtp_ulong uchar_sat_rtp_ulong uchar_rtn_ulong uchar_sat_rtn_ulong uchar_rtz_ulong uchar_sat_rtz_ulong )

add_test(NAME "${TS_NAME}_conversion_char_ulong"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" char_ulong char_sat_ulong char_rte_ulong char_sat_rte_ulong char_rtp_ulong char_sat_rtp_ulong char_rtn_ulong char_sat_rtn_ulong char_rtz_ulong char_sat_rtz_ulong )

add_test(NAME "${TS_NAME}_conversion_ushort_ulong"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ushort_ulong ushort_sat_ulong ushort_rte_ulong ushort_sat_rte_ulong ushort_rtp_ulong ushort_sat_rtp_ulong ushort_rtn_ulong ushort_sat_rtn_ulong ushort_rtz_ulong ushort_sat_rtz_ulong )

add_test(NAME "${TS_NAME}_conversion_short_ulong"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" short_ulong short_sat_ulong short_rte_ulong short_sat_rte_ulong short_rtp_ulong short_sat_rtp_ulong short_rtn_ulong short_sat_rtn_ulong short_rtz_ulong short_sat_rtz_ulong )

add_test(NAME "${TS_NAME}_conversion_uint_ulong"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uint_ulong uint_sat_ulong uint_rte_ulong uint_sat_rte_ulong uint_rtp_ulong uint_sat_rtp_ulong uint_rtn_ulong uint_sat_rtn_ulong uint_rtz_ulong uint_sat_rtz_ulong )

add_test(NAME "${TS_NAME}_conversion_int_ulong"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" int_ulong int_sat_ulong int_rte_ulong int_sat_rte_ulong int_rtp_ulong int_sat_rtp_ulong int_rtn_ulong int_sat_rtn_ulong int_rtz_ulong int_sat_rtz_ulong )

add_test(NAME "${TS_NAME}_conversion_float_ulong"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" float_ulong float_sat_ulong float_rte_ulong float_sat_rte_ulong float_rtp_ulong float_sat_rtp_ulong float_rtn_ulong float_sat_rtn_ulong float_rtz_ulong float_sat_rtz_ulong )

add_test(NAME "${TS_NAME}_conversion_double_ulong"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" double_ulong double_sat_ulong double_rte_ulong double_sat_rte_ulong double_rtp_ulong double_sat_rtp_ulong double_rtn_ulong double_sat_rtn_ulong double_rtz_ulong double_sat_rtz_ulong )

add_test(NAME "${TS_NAME}_conversion_ulong_ulong"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ulong_ulong ulong_sat_ulong ulong_rte_ulong ulong_sat_rte_ulong ulong_rtp_ulong ulong_sat_rtp_ulong ulong_rtn_ulong ulong_sat_rtn_ulong ulong_rtz_ulong ulong_sat_rtz_ulong )

add_test(NAME "${TS_NAME}_conversion_long_ulong"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" long_ulong long_sat_ulong long_rte_ulong long_sat_rte_ulong long_rtp_ulong long_sat_rtp_ulong long_rtn_ulong long_sat_rtn_ulong long_rtz_ulong long_sat_rtz_ulong )

add_test(NAME "${TS_NAME}_conversion_uchar_long"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uchar_long uchar_sat_long uchar_rte_long uchar_sat_rte_long uchar_rtp_long uchar_sat_rtp_long uchar_rtn_long uchar_sat_rtn_long uchar_rtz_long uchar_sat_rtz_long )

add_test(NAME "${TS_NAME}_conversion_char_long"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" char_long char_sat_long char_rte_long char_sat_rte_long char_rtp_long char_sat_rtp_long char_rtn_long char_sat_rtn_long char_rtz_long char_sat_rtz_long )

add_test(NAME "${TS_NAME}_conversion_ushort_long"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ushort_long ushort_sat_long ushort_rte_long ushort_sat_rte_long ushort_rtp_long ushort_sat_rtp_long ushort_rtn_long ushort_sat_rtn_long ushort_rtz_long ushort_sat_rtz_long )

add_test(NAME "${TS_NAME}_conversion_short_long"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" short_long short_sat_long short_rte_long short_sat_rte_long short_rtp_long short_sat_rtp_long short_rtn_long short_sat_rtn_long short_rtz_long short_sat_rtz_long )

add_test(NAME "${TS_NAME}_conversion_uint_long"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" uint_long uint_sat_long uint_rte_long uint_sat_rte_long uint_rtp_long uint_sat_rtp_long uint_rtn_long uint_sat_rtn_long uint_rtz_long uint_sat_rtz_long )

add_test(NAME "${TS_NAME}_conversion_int_long"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" int_long int_sat_long int_rte_long int_sat_rte_long int_rtp_long int_sat_rtp_long int_rtn_long int_sat_rtn_long int_rtz_long int_sat_rtz_long )

add_test(NAME "${TS_NAME}_conversion_float_long"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" float_long float_sat_long float_rte_long float_sat_rte_long float_rtp_long float_sat_rtp_long float_rtn_long float_sat_rtn_long float_rtz_long float_sat_rtz_long )

add_test(NAME "${TS_NAME}_conversion_double_long"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" double_long double_sat_long double_rte_long double_sat_rte_long double_rtp_long double_sat_rtp_long double_rtn_long double_sat_rtn_long double_rtz_long double_sat_rtz_long )

add_test(NAME "${TS_NAME}_conversion_ulong_long"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" ulong_long ulong_sat_long ulong_rte_long ulong_sat_rte_long ulong_rtp_long ulong_sat_rtp_long ulong_rtn_long ulong_sat_rtn_long ulong_rtz_long ulong_sat_rtz_long )

add_test(NAME "${TS_NAME}_conversion_long_long"
         COMMAND "${TS_BUILDDIR}/test_conformance/conversions/test_conversions" long_long long_sat_long long_rte_long long_sat_rte_long long_rtp_long long_sat_rtp_long long_rtn_long long_sat_rtn_long long_rtz_long long_sat_rtz_long )

add_test(NAME "${TS_NAME}_device_partition"
         COMMAND "${TS_BUILDDIR}/test_conformance/device_partition/test_device_partition" )

add_test(NAME "${TS_NAME}_events"
         COMMAND "${TS_BUILDDIR}/test_conformance/events/test_events" event_get_execute_status  event_get_write_array_status  event_get_read_array_status  event_get_info  event_wait_for_execute  event_wait_for_array  event_flush  event_finish_execute  event_finish_array  event_release_before_done  event_enqueue_marker  event_enqueue_marker_with_event_list  event_enqueue_barrier_with_event_list)
add_test(NAME "${TS_NAME}_events_ooo"
         COMMAND "${TS_BUILDDIR}/test_conformance/events/test_events" out_of_order_event_waitlist_single_queue  out_of_order_event_waitlist_multi_queue  out_of_order_event_waitlist_multi_queue_multi_device  out_of_order_event_enqueue_wait_for_events_single_queue  out_of_order_event_enqueue_wait_for_events_multi_queue  out_of_order_event_enqueue_wait_for_events_multi_queue_multi_device  out_of_order_event_enqueue_marker_single_queue  out_of_order_event_enqueue_marker_multi_queue  out_of_order_event_enqueue_marker_multi_queue_multi_device  out_of_order_event_enqueue_barrier_single_queue)
add_test(NAME "${TS_NAME}_events_other"
         COMMAND "${TS_BUILDDIR}/test_conformance/events/test_events" waitlists  test_userevents  callbacks  callbacks_simultaneous  userevents_multithreaded)

add_test(NAME "${TS_NAME}_geometrics"
         COMMAND "${TS_BUILDDIR}/test_conformance/geometrics/test_geometrics" )

add_test(NAME "${TS_NAME}_gl"
         COMMAND "${TS_BUILDDIR}/test_conformance/gl/test_gl" )

add_test(NAME "${TS_NAME}_half"
         COMMAND "${TS_BUILDDIR}/test_conformance/half/Test_half" )

add_test(NAME "${TS_NAME}_cl_gl_h"
         COMMAND "${TS_BUILDDIR}/test_conformance/headers/test_cl_gl_h" )

add_test(NAME "${TS_NAME}_cl_gl_h_c99"
         COMMAND "${TS_BUILDDIR}/test_conformance/headers/test_cl_gl_h_c99" )

add_test(NAME "${TS_NAME}_cl_h"
         COMMAND "${TS_BUILDDIR}/test_conformance/headers/test_cl_h" )

add_test(NAME "${TS_NAME}_cl_h_c99"
         COMMAND "${TS_BUILDDIR}/test_conformance/headers/test_cl_h_c99" )

add_test(NAME "${TS_NAME}_cl_platform_h"
         COMMAND "${TS_BUILDDIR}/test_conformance/headers/test_cl_platform_h" )

add_test(NAME "${TS_NAME}_cl_platform_h_c99"
         COMMAND "${TS_BUILDDIR}/test_conformance/headers/test_cl_platform_h_c99" )

add_test(NAME "${TS_NAME}_headers"
         COMMAND "${TS_BUILDDIR}/test_conformance/headers/test_headers" )

add_test(NAME "${TS_NAME}_opencl_h"
         COMMAND "${TS_BUILDDIR}/test_conformance/headers/test_opencl_h" )

add_test(NAME "${TS_NAME}_opencl_h_c99"
         COMMAND "${TS_BUILDDIR}/test_conformance/headers/test_opencl_h_c99" )

#***************************************************************************

add_test(NAME "${TS_NAME}_images_cl_copy_images"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clCopyImage/test_cl_copy_images" )
add_test(NAME "${TS_NAME}_images_cl_copy_images_max"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clCopyImage/test_cl_copy_images" max_images)
add_test(NAME "${TS_NAME}_images_cl_copy_images_min"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clCopyImage/test_cl_copy_images" small_images)
add_test(NAME "${TS_NAME}_images_cl_copy_images_rounding"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clCopyImage/test_cl_copy_images" rounding)

add_test(NAME "${TS_NAME}_images_cl_fill_images"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clFillImage/test_cl_fill_images" )
add_test(NAME "${TS_NAME}_images_cl_fill_images_min"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clFillImage/test_cl_fill_images" small_images)
add_test(NAME "${TS_NAME}_images_cl_fill_images_max"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clFillImage/test_cl_fill_images" max_images)
add_test(NAME "${TS_NAME}_images_cl_fill_images_pitch"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clFillImage/test_cl_fill_images" use_pitches)

add_test(NAME "${TS_NAME}_images_cl_get_info"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clGetInfo/test_cl_get_info" )
add_test(NAME "${TS_NAME}_images_cl_get_info_min"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clGetInfo/test_cl_get_info" small_images)
add_test(NAME "${TS_NAME}_images_cl_get_info_max"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clGetInfo/test_cl_get_info" max_images)
add_test(NAME "${TS_NAME}_images_cl_get_info_random"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clGetInfo/test_cl_get_info" randomize)

add_test(NAME "${TS_NAME}_images_cl_read_write_images"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clReadWriteImage/test_cl_read_write_images" )
add_test(NAME "${TS_NAME}_images_cl_read_write_images_min"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clReadWriteImage/test_cl_read_write_images" small_images)
add_test(NAME "${TS_NAME}_images_cl_read_write_images_max"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clReadWriteImage/test_cl_read_write_images" max_images)
add_test(NAME "${TS_NAME}_images_cl_read_write_images_round"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clReadWriteImage/test_cl_read_write_images" rounding)
add_test(NAME "${TS_NAME}_images_cl_read_write_images_pitch"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/clReadWriteImage/test_cl_read_write_images" use_pitches)

add_test(NAME "${TS_NAME}_images_kernel_image_methods_1D"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/kernel_image_methods/test_kernel_image_methods" 1D)
add_test(NAME "${TS_NAME}_images_kernel_image_methods_2D"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/kernel_image_methods/test_kernel_image_methods" 2D)
add_test(NAME "${TS_NAME}_images_kernel_image_methods_3D"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/kernel_image_methods/test_kernel_image_methods" 3D)
add_test(NAME "${TS_NAME}_images_kernel_image_methods_1Darray"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/kernel_image_methods/test_kernel_image_methods" 1Darray)
add_test(NAME "${TS_NAME}_images_kernel_image_methods_2Darray"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/kernel_image_methods/test_kernel_image_methods" 2Darray)

add_test(NAME "${TS_NAME}_images_image_streams_1D"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/kernel_read_write/test_image_streams" "1D" )
add_test(NAME "${TS_NAME}_images_image_streams_2D"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/kernel_read_write/test_image_streams" "2D" )
add_test(NAME "${TS_NAME}_images_image_streams_3D"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/kernel_read_write/test_image_streams" "3D")
add_test(NAME "${TS_NAME}_images_image_streams_1Darray"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/kernel_read_write/test_image_streams" "1Darray")
add_test(NAME "${TS_NAME}_images_image_streams_2Darray"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/kernel_read_write/test_image_streams" "2Darray")

add_test(NAME "${TS_NAME}_images_image_streams_1D_pitch"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/kernel_read_write/test_image_streams" "1D"  use_pitches)
add_test(NAME "${TS_NAME}_images_image_streams_2D_pitch"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/kernel_read_write/test_image_streams" "2D"  use_pitches)
add_test(NAME "${TS_NAME}_images_image_streams_3D_pitch"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/kernel_read_write/test_image_streams" "3D" use_pitches)
add_test(NAME "${TS_NAME}_images_image_streams_1Darray_pitch"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/kernel_read_write/test_image_streams" "1Darray" use_pitches)
add_test(NAME "${TS_NAME}_images_image_streams_2Darray_pitch"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/kernel_read_write/test_image_streams" "2Darray" use_pitches)

add_test(NAME "${TS_NAME}_images_samplerless_reads_1D"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/samplerlessReads/test_samplerless_reads" 1D)
add_test(NAME "${TS_NAME}_images_samplerless_reads_2D"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/samplerlessReads/test_samplerless_reads" 2D)
add_test(NAME "${TS_NAME}_images_samplerless_reads_3D"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/samplerlessReads/test_samplerless_reads" 3D)
add_test(NAME "${TS_NAME}_images_samplerless_reads_1Darray"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/samplerlessReads/test_samplerless_reads" 1Darray)
add_test(NAME "${TS_NAME}_images_samplerless_reads_2Darray"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/samplerlessReads/test_samplerless_reads" 2Darray)

add_test(NAME "${TS_NAME}_images_samplerless_reads_1D_pitch"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/samplerlessReads/test_samplerless_reads" 1D use_pitches)
add_test(NAME "${TS_NAME}_images_samplerless_reads_2D_pitch"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/samplerlessReads/test_samplerless_reads" 2D use_pitches)
add_test(NAME "${TS_NAME}_images_samplerless_reads_3D_pitch"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/samplerlessReads/test_samplerless_reads" 3D use_pitches)
add_test(NAME "${TS_NAME}_images_samplerless_reads_1Darray_pitch"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/samplerlessReads/test_samplerless_reads" 1Darray use_pitches)
add_test(NAME "${TS_NAME}_images_samplerless_reads_2Darray_pitch"
         COMMAND "${TS_BUILDDIR}/test_conformance/images/samplerlessReads/test_samplerless_reads" 2Darray use_pitches)

#***************************************************************************

add_test(NAME "${TS_NAME}_integer_ops_1"
         COMMAND "${TS_BUILDDIR}/test_conformance/integer_ops/test_integer_ops" integer_clz  integer_hadd  integer_rhadd  integer_mul_hi  integer_rotate  integer_clamp  integer_mad_sat  integer_mad_hi  integer_min  integer_max  integer_upsample  integer_abs  integer_abs_diff  integer_add_sat  integer_sub_sat  integer_addAssign  integer_subtractAssign  integer_multiplyAssign  integer_divideAssign  integer_moduloAssign  integer_andAssign  integer_orAssign  integer_exclusiveOrAssign)
add_test(NAME "${TS_NAME}_integer_ops_2"
         COMMAND "${TS_BUILDDIR}/test_conformance/integer_ops/test_integer_ops" unary_ops_increment  unary_ops_decrement  unary_ops_full  integer_mul24  integer_mad24  long_math  long_logic  long_shift  long_compare  ulong_math  ulong_logic  ulong_shift  ulong_compare  int_math  int_logic  int_shift  int_compare  uint_math  uint_logic  uint_shift  uint_compare  short_math  short_logic  short_shift  short_compare  ushort_math  ushort_logic  ushort_shift  ushort_compare  char_math  char_logic  char_shift  char_compare  uchar_math  uchar_logic  uchar_shift  uchar_compare  popcount  vector_scalar)
add_test(NAME "${TS_NAME}_integer_ops_3"
         COMMAND "${TS_BUILDDIR}/test_conformance/integer_ops/test_integer_ops" quick_long_math  quick_long_logic  quick_long_shift  quick_long_compare  quick_ulong_math  quick_ulong_logic  quick_ulong_shift  quick_ulong_compare  quick_int_math  quick_int_logic  quick_int_shift  quick_int_compare  quick_uint_math  quick_uint_logic  quick_uint_shift  quick_uint_compare  quick_short_math  quick_short_logic  quick_short_shift  quick_short_compare  quick_ushort_math  quick_ushort_logic  quick_ushort_shift  quick_ushort_compare  quick_char_math  quick_char_logic  quick_char_shift  quick_char_compare  quick_uchar_math  quick_uchar_logic  quick_uchar_shift  quick_uchar_compare)

add_test(NAME "${TS_NAME}_bruteforce"
         COMMAND "${TS_BUILDDIR}/test_conformance/math_brute_force/bruteforce" )

add_test(NAME "${TS_NAME}_mem_host_flags"
         COMMAND "${TS_BUILDDIR}/test_conformance/mem_host_flags/test_mem_host_flags" )

add_test(NAME "${TS_NAME}_multiples"
         COMMAND "${TS_BUILDDIR}/test_conformance/multiple_device_context/test_multiples" )

add_test(NAME "${TS_NAME}_printf"
         COMMAND "${TS_BUILDDIR}/test_conformance/printf/test_printf" )

add_test(NAME "${TS_NAME}_profiling"
         COMMAND "${TS_BUILDDIR}/test_conformance/profiling/test_profiling" )

add_test(NAME "${TS_NAME}_relationals"
         COMMAND "${TS_BUILDDIR}/test_conformance/relationals/test_relationals" relational_any  relational_all  relational_bitselect  relational_select_signed  relational_select_unsigned  relational_isequal  relational_isnotequal  relational_isgreater  relational_isgreaterequal  relational_isless  relational_islessequal  relational_islessgreater)
add_test(NAME "${TS_NAME}_relationals_shuffle"
         COMMAND "${TS_BUILDDIR}/test_conformance/relationals/test_relationals" shuffle_copy  shuffle_function_call  shuffle_array_cast  shuffle_built_in  shuffle_built_in_dual_input)

add_test(NAME "${TS_NAME}_select_uchar_uchar"
         COMMAND "${TS_BUILDDIR}/test_conformance/select/test_select" select_uchar_uchar)
add_test(NAME "${TS_NAME}_select_uchar_char"
         COMMAND "${TS_BUILDDIR}/test_conformance/select/test_select" select_uchar_char)
add_test(NAME "${TS_NAME}_select_char_uchar"
         COMMAND "${TS_BUILDDIR}/test_conformance/select/test_select" select_char_uchar)
add_test(NAME "${TS_NAME}_select_char_char"
         COMMAND "${TS_BUILDDIR}/test_conformance/select/test_select" select_char_char)

add_test(NAME "${TS_NAME}_select_ushort_ushort"
         COMMAND "${TS_BUILDDIR}/test_conformance/select/test_select" select_ushort_ushort)
add_test(NAME "${TS_NAME}_select_ushort_short"
         COMMAND "${TS_BUILDDIR}/test_conformance/select/test_select" select_ushort_short)
add_test(NAME "${TS_NAME}_select_short_ushort"
         COMMAND "${TS_BUILDDIR}/test_conformance/select/test_select" select_short_ushort)
add_test(NAME "${TS_NAME}_select_short_short"
         COMMAND "${TS_BUILDDIR}/test_conformance/select/test_select" select_short_short)

add_test(NAME "${TS_NAME}_select_uint_uint"
         COMMAND "${TS_BUILDDIR}/test_conformance/select/test_select" select_uint_uint)
add_test(NAME "${TS_NAME}_select_uint_int"
         COMMAND "${TS_BUILDDIR}/test_conformance/select/test_select" select_uint_int)
add_test(NAME "${TS_NAME}_select_int_uint"
         COMMAND "${TS_BUILDDIR}/test_conformance/select/test_select" select_int_uint)
add_test(NAME "${TS_NAME}_select_int_int"
         COMMAND "${TS_BUILDDIR}/test_conformance/select/test_select" select_int_int)

add_test(NAME "${TS_NAME}_select_ulong_ulong"
         COMMAND "${TS_BUILDDIR}/test_conformance/select/test_select" select_ulong_ulong)
add_test(NAME "${TS_NAME}_select_ulong_long"
         COMMAND "${TS_BUILDDIR}/test_conformance/select/test_select" select_ulong_long)
add_test(NAME "${TS_NAME}_select_long_ulong"
         COMMAND "${TS_BUILDDIR}/test_conformance/select/test_select" select_long_ulong)
add_test(NAME "${TS_NAME}_select_long_long"
         COMMAND "${TS_BUILDDIR}/test_conformance/select/test_select" select_long_long)

add_test(NAME "${TS_NAME}_thread_dimensions_quick"
         COMMAND "${TS_BUILDDIR}/test_conformance/thread_dimensions/test_thread_dimensions" quick_1d_explicit_local  quick_2d_explicit_local  quick_3d_explicit_local  quick_1d_implicit_local  quick_2d_implicit_local  quick_3d_implicit_local)
add_test(NAME "${TS_NAME}_thread_dimensions_full"
         COMMAND "${TS_BUILDDIR}/test_conformance/thread_dimensions/test_thread_dimensions" full_1d_explicit_local  full_2d_explicit_local  full_3d_explicit_local  full_1d_implicit_local  full_2d_implicit_local  full_3d_implicit_local)

add_test(NAME "${TS_NAME}_vecalign"
         COMMAND "${TS_BUILDDIR}/test_conformance/vec_align/test_vecalign" )

add_test(NAME "${TS_NAME}_vecstep"
         COMMAND "${TS_BUILDDIR}/test_conformance/vec_step/test_vecstep" )


set_tests_properties(
    "${TS_NAME}_allocations_buffer"
    "${TS_NAME}_allocations_image"
    "${TS_NAME}_api"
    "${TS_NAME}_atomics"
    "${TS_NAME}_basic_math"
    "${TS_NAME}_basic_s2v"
    "${TS_NAME}_basic_memory"
    "${TS_NAME}_basic_image"
    "${TS_NAME}_basic_other"
    "${TS_NAME}_buffers_read_async"
    "${TS_NAME}_buffers_write_async"
    "${TS_NAME}_buffers_read_array_barrier"
    "${TS_NAME}_buffers_read"
    "${TS_NAME}_buffers_write"
    "${TS_NAME}_buffers_fill"
    "${TS_NAME}_buffers_var1"
    "${TS_NAME}_buffers_var2"
    "${TS_NAME}_buffers_map_read"
    "${TS_NAME}_buffers_map_write"
    "${TS_NAME}_commonfns"
    "${TS_NAME}_compiler"
    "${TS_NAME}_computeinfo"
    "${TS_NAME}_contractions"
    "${TS_NAME}_conversion_uchar_uchar"
    "${TS_NAME}_conversion_char_uchar"
    "${TS_NAME}_conversion_ushort_uchar"
    "${TS_NAME}_conversion_short_uchar"
    "${TS_NAME}_conversion_uint_uchar"
    "${TS_NAME}_conversion_int_uchar"
    "${TS_NAME}_conversion_float_uchar"
    "${TS_NAME}_conversion_double_uchar"
    "${TS_NAME}_conversion_ulong_uchar"
    "${TS_NAME}_conversion_long_uchar"
    "${TS_NAME}_conversion_uchar_char"
    "${TS_NAME}_conversion_char_char"
    "${TS_NAME}_conversion_ushort_char"
    "${TS_NAME}_conversion_short_char"
    "${TS_NAME}_conversion_uint_char"
    "${TS_NAME}_conversion_int_char"
    "${TS_NAME}_conversion_float_char"
    "${TS_NAME}_conversion_double_char"
    "${TS_NAME}_conversion_ulong_char"
    "${TS_NAME}_conversion_long_char"
    "${TS_NAME}_conversion_uchar_ushort"
    "${TS_NAME}_conversion_char_ushort"
    "${TS_NAME}_conversion_ushort_ushort"
    "${TS_NAME}_conversion_short_ushort"
    "${TS_NAME}_conversion_uint_ushort"
    "${TS_NAME}_conversion_int_ushort"
    "${TS_NAME}_conversion_float_ushort"
    "${TS_NAME}_conversion_double_ushort"
    "${TS_NAME}_conversion_ulong_ushort"
    "${TS_NAME}_conversion_long_ushort"
    "${TS_NAME}_conversion_uchar_short"
    "${TS_NAME}_conversion_char_short"
    "${TS_NAME}_conversion_ushort_short"
    "${TS_NAME}_conversion_short_short"
    "${TS_NAME}_conversion_uint_short"
    "${TS_NAME}_conversion_int_short"
    "${TS_NAME}_conversion_float_short"
    "${TS_NAME}_conversion_double_short"
    "${TS_NAME}_conversion_ulong_short"
    "${TS_NAME}_conversion_long_short"
    "${TS_NAME}_conversion_uchar_uint"
    "${TS_NAME}_conversion_char_uint"
    "${TS_NAME}_conversion_ushort_uint"
    "${TS_NAME}_conversion_short_uint"
    "${TS_NAME}_conversion_uint_uint"
    "${TS_NAME}_conversion_int_uint"
    "${TS_NAME}_conversion_float_uint"
    "${TS_NAME}_conversion_double_uint"
    "${TS_NAME}_conversion_ulong_uint"
    "${TS_NAME}_conversion_long_uint"
    "${TS_NAME}_conversion_uchar_int"
    "${TS_NAME}_conversion_char_int"
    "${TS_NAME}_conversion_ushort_int"
    "${TS_NAME}_conversion_short_int"
    "${TS_NAME}_conversion_uint_int"
    "${TS_NAME}_conversion_int_int"
    "${TS_NAME}_conversion_float_int"
    "${TS_NAME}_conversion_double_int"
    "${TS_NAME}_conversion_ulong_int"
    "${TS_NAME}_conversion_long_int"
    "${TS_NAME}_conversion_uchar_float"
    "${TS_NAME}_conversion_char_float"
    "${TS_NAME}_conversion_ushort_float"
    "${TS_NAME}_conversion_short_float"
    "${TS_NAME}_conversion_uint_float"
    "${TS_NAME}_conversion_int_float"
    "${TS_NAME}_conversion_float_float"
    "${TS_NAME}_conversion_double_float"
    "${TS_NAME}_conversion_ulong_float"
    "${TS_NAME}_conversion_long_float"
    "${TS_NAME}_conversion_uchar_double"
    "${TS_NAME}_conversion_char_double"
    "${TS_NAME}_conversion_ushort_double"
    "${TS_NAME}_conversion_short_double"
    "${TS_NAME}_conversion_uint_double"
    "${TS_NAME}_conversion_int_double"
    "${TS_NAME}_conversion_float_double"
    "${TS_NAME}_conversion_double_double"
    "${TS_NAME}_conversion_ulong_double"
    "${TS_NAME}_conversion_long_double"
    "${TS_NAME}_conversion_uchar_ulong"
    "${TS_NAME}_conversion_char_ulong"
    "${TS_NAME}_conversion_ushort_ulong"
    "${TS_NAME}_conversion_short_ulong"
    "${TS_NAME}_conversion_uint_ulong"
    "${TS_NAME}_conversion_int_ulong"
    "${TS_NAME}_conversion_float_ulong"
    "${TS_NAME}_conversion_double_ulong"
    "${TS_NAME}_conversion_ulong_ulong"
    "${TS_NAME}_conversion_long_ulong"
    "${TS_NAME}_conversion_uchar_long"
    "${TS_NAME}_conversion_char_long"
    "${TS_NAME}_conversion_ushort_long"
    "${TS_NAME}_conversion_short_long"
    "${TS_NAME}_conversion_uint_long"
    "${TS_NAME}_conversion_int_long"
    "${TS_NAME}_conversion_float_long"
    "${TS_NAME}_conversion_double_long"
    "${TS_NAME}_conversion_ulong_long"
    "${TS_NAME}_conversion_long_long"
    "${TS_NAME}_device_partition"
    "${TS_NAME}_events"
    "${TS_NAME}_events_ooo"
    "${TS_NAME}_events_other"
    "${TS_NAME}_geometrics"
    "${TS_NAME}_gl"
    "${TS_NAME}_half"
    "${TS_NAME}_cl_gl_h"
    "${TS_NAME}_cl_gl_h_c99"
    "${TS_NAME}_cl_h"
    "${TS_NAME}_cl_h_c99"
    "${TS_NAME}_cl_platform_h"
    "${TS_NAME}_cl_platform_h_c99"
    "${TS_NAME}_headers"
    "${TS_NAME}_opencl_h"
    "${TS_NAME}_opencl_h_c99"
    "${TS_NAME}_images_cl_copy_images"
    "${TS_NAME}_images_cl_copy_images_max"
    "${TS_NAME}_images_cl_copy_images_min"
    "${TS_NAME}_images_cl_copy_images_rounding"
    "${TS_NAME}_images_cl_fill_images"
    "${TS_NAME}_images_cl_fill_images_min"
    "${TS_NAME}_images_cl_fill_images_max"
    "${TS_NAME}_images_cl_fill_images_pitch"
    "${TS_NAME}_images_cl_get_info"
    "${TS_NAME}_images_cl_get_info_min"
    "${TS_NAME}_images_cl_get_info_max"
    "${TS_NAME}_images_cl_get_info_random"
    "${TS_NAME}_images_cl_read_write_images"
    "${TS_NAME}_images_cl_read_write_images_min"
    "${TS_NAME}_images_cl_read_write_images_max"
    "${TS_NAME}_images_cl_read_write_images_round"
    "${TS_NAME}_images_cl_read_write_images_pitch"
    "${TS_NAME}_images_kernel_image_methods_1D"
    "${TS_NAME}_images_kernel_image_methods_2D"
    "${TS_NAME}_images_kernel_image_methods_3D"
    "${TS_NAME}_images_kernel_image_methods_1Darray"
    "${TS_NAME}_images_kernel_image_methods_2Darray"
    "${TS_NAME}_images_image_streams_1D"
    "${TS_NAME}_images_image_streams_2D"
    "${TS_NAME}_images_image_streams_3D"
    "${TS_NAME}_images_image_streams_1Darray"
    "${TS_NAME}_images_image_streams_2Darray"
    "${TS_NAME}_images_image_streams_1D_pitch"
    "${TS_NAME}_images_image_streams_2D_pitch"
    "${TS_NAME}_images_image_streams_3D_pitch"
    "${TS_NAME}_images_image_streams_1Darray_pitch"
    "${TS_NAME}_images_image_streams_2Darray_pitch"
    "${TS_NAME}_images_samplerless_reads_1D"
    "${TS_NAME}_images_samplerless_reads_2D"
    "${TS_NAME}_images_samplerless_reads_3D"
    "${TS_NAME}_images_samplerless_reads_1Darray"
    "${TS_NAME}_images_samplerless_reads_2Darray"
    "${TS_NAME}_images_samplerless_reads_1D_pitch"
    "${TS_NAME}_images_samplerless_reads_2D_pitch"
    "${TS_NAME}_images_samplerless_reads_3D_pitch"
    "${TS_NAME}_images_samplerless_reads_1Darray_pitch"
    "${TS_NAME}_images_samplerless_reads_2Darray_pitch"
    "${TS_NAME}_integer_ops_1"
    "${TS_NAME}_integer_ops_2"
    "${TS_NAME}_integer_ops_3"
    "${TS_NAME}_bruteforce"
    "${TS_NAME}_mem_host_flags"
    "${TS_NAME}_multiples"
    "${TS_NAME}_printf"
    "${TS_NAME}_profiling"
    "${TS_NAME}_relationals"
    "${TS_NAME}_relationals_shuffle"
    "${TS_NAME}_select_uchar_uchar"
    "${TS_NAME}_select_uchar_char"
    "${TS_NAME}_select_char_uchar"
    "${TS_NAME}_select_char_char"
    "${TS_NAME}_select_ushort_ushort"
    "${TS_NAME}_select_ushort_short"
    "${TS_NAME}_select_short_ushort"
    "${TS_NAME}_select_short_short"
    "${TS_NAME}_select_uint_uint"
    "${TS_NAME}_select_uint_int"
    "${TS_NAME}_select_int_uint"
    "${TS_NAME}_select_int_int"
    "${TS_NAME}_select_ulong_ulong"
    "${TS_NAME}_select_ulong_long"
    "${TS_NAME}_select_long_ulong"
    "${TS_NAME}_select_long_long"
    "${TS_NAME}_thread_dimensions_quick"
    "${TS_NAME}_thread_dimensions_full"
    "${TS_NAME}_vecalign"
    "${TS_NAME}_vecstep"
  PROPERTIES
    LABELS "conformance_suite"
    ENVIRONMENT "POCL_MEMORY_LIMIT=1")


set_tests_properties(
    "${TS_NAME}_allocations_buffer"
    "${TS_NAME}_allocations_image"
    "${TS_NAME}_api"
    "${TS_NAME}_atomics"
    "${TS_NAME}_basic_math"
    "${TS_NAME}_basic_s2v"
    "${TS_NAME}_basic_memory"
    "${TS_NAME}_basic_image"
    "${TS_NAME}_basic_other"
    "${TS_NAME}_buffers_read"
    "${TS_NAME}_buffers_write"
    "${TS_NAME}_buffers_fill"
    "${TS_NAME}_buffers_var1"
    "${TS_NAME}_buffers_var2"
    "${TS_NAME}_buffers_map_read"
    "${TS_NAME}_buffers_map_write"
    "${TS_NAME}_commonfns"
    "${TS_NAME}_compiler"
    "${TS_NAME}_computeinfo"
    "${TS_NAME}_contractions"

    "${TS_NAME}_conversion_uint_uchar"
    "${TS_NAME}_conversion_short_char"

    "${TS_NAME}_conversion_uchar_ushort"
    "${TS_NAME}_conversion_char_int"

    "${TS_NAME}_conversion_float_char"
    "${TS_NAME}_conversion_float_ushort"
    "${TS_NAME}_conversion_int_float"

    "${TS_NAME}_conversion_double_short"
    "${TS_NAME}_conversion_double_ulong"
    "${TS_NAME}_conversion_long_double"

    "${TS_NAME}_device_partition"
    "${TS_NAME}_events"
    "${TS_NAME}_events_ooo"
    "${TS_NAME}_events_other"
    "${TS_NAME}_geometrics"
    "${TS_NAME}_gl"
    "${TS_NAME}_half"
    "${TS_NAME}_headers"
    "${TS_NAME}_opencl_h"
    "${TS_NAME}_opencl_h_c99"
    "${TS_NAME}_images_cl_copy_images"
    "${TS_NAME}_images_cl_fill_images"
    "${TS_NAME}_images_cl_get_info"
    "${TS_NAME}_images_cl_read_write_images"
    "${TS_NAME}_images_kernel_image_methods_1D"
    "${TS_NAME}_images_kernel_image_methods_2D"
    "${TS_NAME}_images_kernel_image_methods_1Darray"
    "${TS_NAME}_images_image_streams_1D"
    "${TS_NAME}_images_image_streams_2D"
    "${TS_NAME}_images_image_streams_1Darray"
    "${TS_NAME}_images_samplerless_reads_1D"
    "${TS_NAME}_images_samplerless_reads_2D"
    "${TS_NAME}_images_samplerless_reads_1Darray"
    "${TS_NAME}_integer_ops_1"
    "${TS_NAME}_integer_ops_2"
    "${TS_NAME}_integer_ops_3"
    "${TS_NAME}_mem_host_flags"
    "${TS_NAME}_multiples"
    "${TS_NAME}_printf"
    "${TS_NAME}_profiling"
    "${TS_NAME}_relationals"
    "${TS_NAME}_relationals_shuffle"
    "${TS_NAME}_select_char_char"
    "${TS_NAME}_select_short_short"
    "${TS_NAME}_select_int_int"
    "${TS_NAME}_select_long_long"
    "${TS_NAME}_thread_dimensions_quick"
    "${TS_NAME}_thread_dimensions_full"
    "${TS_NAME}_vecalign"
    "${TS_NAME}_vecstep"
  PROPERTIES
    LABELS "conformance_suite;conformance_suite_short"
    ENVIRONMENT "POCL_MEMORY_LIMIT=1")

endif()
