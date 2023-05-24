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
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TS_BASEDIR}/src/${TS_NAME}")

if(NOT HAVE_GIT)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires git to checkout sources")
  return()
endif()

if(ENABLE_ICD AND (NOT OPENCL_LIBRARIES))
  message(STATUS "Disabling ${TS_NAME}, can't find OpenCL library")
  return()
endif()

# for SPIR-V tests
find_program(SPIRV_AS "spirv-as")
find_program(SPIRV_VAL "spirv-val")
if(SPIRV_AS AND SPIRV_VAL)
  set(HAVE_SPIRV_TOOLS 1)
endif()

# the CTS is compiled against:
#    OpenCL headers in <POCL>/include/CL, which are 3.0
#    OpenCL library, which might be libpocl or ICD; we need to check ICD is 3.0 compatible
if(ENABLE_ICD AND HAVE_OCL_ICD AND (NOT HAVE_OCL_ICD_30_COMPATIBLE))
  message(STATUS "OCL-ICD is not compatible with OpenCL-CTS version 3, disabling CTS")
  return()
endif()

# if ENABLE_CONFORMANCE is OFF (default), the CTS reports a lot of failures with CPU driver
if(ENABLE_HOST_CPU_DEVICES AND (NOT ENABLE_CONFORMANCE))
  message(WARNING "OpenCL-CTS + CPU driver need ENABLE_CONFORMANCE=ON, disabling CTS")
  return()
endif()

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

message(STATUS "enabling OpenCL-CTS version 3")

# new CTS (3.0)
if(ENABLE_ICD)
  if(OPENCL_LIBRARIES AND (NOT OPENCL_LIBDIR))
    get_filename_component(OPENCL_LIBDIR "${OPENCL_LIBRARIES}" DIRECTORY)
  endif()
  set(CTS_OPENCL_LIBDIR "-DCL_LIB_DIR=${OPENCL_LIBDIR}")
  set(CTS_OPENCL_LIBRARY "-DOPENCL_LIBRARIES=${OPENCL_LIBRARIES}")
else()
  set(CTS_OPENCL_LIBDIR "-DCL_LIB_DIR=$<TARGET_FILE_DIR:${POCL_LIBRARY_NAME}>")
  set(CTS_OPENCL_LIBRARY "-DOPENCL_LIBRARIES=$<TARGET_FILE_NAME:${POCL_LIBRARY_NAME}>")
endif()

if(CUSTOM_CTS_GIT_REPO)
  set(CTS_GIT_REPO "${CUSTOM_CTS_GIT_REPO}")
else()
  # Use PoCL's fork which has pending fixes.
  set(CTS_GIT_REPO "https://github.com/pocl/OpenCL-CTS")
#  set(CTS_GIT_REPO "https://github.com/KhronosGroup/OpenCL-CTS")
endif()
if(CUSTOM_CTS_GIT_TAG)
  set(CTS_GIT_TAG "${CUSTOM_CTS_GIT_TAG}")
else()
  # Use PoCL's fork which has pending fixes.
  set(CTS_GIT_TAG "cl30")
endif()

set(TS_BUILDDIR "${TS_BUILDDIR}/test_conformance")
set(TS_SRCDIR "${TS_SRCDIR}/test_conformance")

if(ENABLE_SPIRV AND HAVE_SPIRV_TOOLS)
  # build SPIR-V binary files required by the SPIR-V subtests
  set(CTS_INSTALL_COMMAND "${TS_SRCDIR}/spirv_new/assemble_spirv.py"
    "-s" "${TS_SRCDIR}/spirv_new/spirv_asm"
    "-o" "${TS_BUILDDIR}/spirv_new/spirv_bin"
    "-a" "${SPIRV_AS}"
    "-l" "${SPIRV_VAL}")
else()
  message(WARNING "spirv tools not found - disabling SPIRV tests in CTS")
  set(CTS_INSTALL_COMMAND echo "INSTALLED")
endif()

set(UPSTREAM_CMAKE_ARGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
if(CMAKE_C_COMPILER_LAUNCHER)
  list(APPEND UPSTREAM_CMAKE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}")
endif()
if(CMAKE_CXX_COMPILER_LAUNCHER)
  list(APPEND UPSTREAM_CMAKE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}")
endif()


ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  GIT_REPOSITORY "${CTS_GIT_REPO}"
        GIT_TAG "${CTS_GIT_TAG}"
  CONFIGURE_COMMAND ${CMAKE_COMMAND}
    ${UPSTREAM_CMAKE_ARGS}
    -DD3D10_IS_SUPPORTED=0 -DD3D11_IS_SUPPORTED=0
    -DGL_IS_SUPPORTED=0 -DGLES_IS_SUPPORTED=0
    -DCL_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/include
    ${CTS_OPENCL_LIBDIR}
    ${CTS_OPENCL_LIBRARY}
    -DENABLE_ASAN=${ENABLE_ASAN}
    -DENABLE_TSAN=${ENABLE_TSAN}
    "${TS_BASEDIR}/src/${TS_NAME}"
  INSTALL_COMMAND ${CTS_INSTALL_COMMAND}
)


set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

#################################################################################
#################################################################################
########################   FULL   ###############################################
#################################################################################
#################################################################################

function(add_cts_full_tests CTS_LABEL TS_SUFFIX)
set(EXTRA_ARGS ${ARGN})
add_test(NAME "${TS_NAME}${TS_SUFFIX}_allocations_buffer"
         COMMAND "${TS_BUILDDIR}/allocations/test_allocations" ${EXTRA_ARGS} buffer buffer_non_blocking)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_allocations_image"
         COMMAND "${TS_BUILDDIR}/allocations/test_allocations" ${EXTRA_ARGS} image2d_read  image2d_write  image2d_read_non_blocking  image2d_write_non_blocking)

# on recent LLVMs, run all tests except the get_kernel_arg_info.
# That test incorrectly assumes "const" type qualifiers are returned
# for non-pointer arguments too. The spec PDF says type qualifiers
# are only returned for pointer types, so the test is incorrect AFAICT
add_test(NAME "${TS_NAME}${TS_SUFFIX}_api"
           COMMAND "${TS_BUILDDIR}/api/test_api" ${EXTRA_ARGS} get_platform_info  get_sampler_info  get_command_queue_info  get_context_info  get_device_info  enqueue_task  binary_get  binary_create  kernel_required_group_size  release_kernel_order  release_during_execute  load_single_kernel  load_two_kernels  load_two_kernels_in_one  load_two_kernels_manually  get_program_info_kernel_names   create_kernels_in_program  get_kernel_info  execute_kernel_local_sizes  set_kernel_arg_by_index  set_kernel_arg_constant  set_kernel_arg_struct_array  kernel_global_constant  min_max_thread_dimensions  min_max_work_items_sizes  min_max_work_group_size  min_max_read_image_args  min_max_write_image_args  min_max_mem_alloc_size  min_max_image_2d_width  min_max_image_2d_height  min_max_image_3d_width  min_max_image_3d_height  min_max_image_3d_depth  min_max_image_array_size  min_max_image_buffer_size  min_max_parameter_size  min_max_samplers  min_max_constant_buffer_size  min_max_constant_args  min_max_compute_units  min_max_address_bits  min_max_single_fp_config  min_max_double_fp_config  min_max_local_mem_size  min_max_kernel_preferred_work_group_size_multiple  min_max_execution_capabilities  min_max_queue_properties  min_max_device_version  min_max_language_version  kernel_arg_changes  kernel_arg_multi_setup_random  native_kernel  create_context_from_type  platform_extensions  get_platform_ids  bool_type  repeated_setup_cleanup  retain_queue_single  retain_queue_multiple  retain_mem_object_single  retain_mem_object_multiple  min_data_type_align_size_alignment  mem_object_destructor_callback  null_buffer_arg  get_buffer_info  get_image2d_info  get_image3d_info  get_image1d_info  get_image1d_array_info  get_image2d_array_info)

if(LLVM_VERSION VERSION_GREATER_EQUAL 11.0)
  set(API_OCL_30_EXTRA_TEST kernel_attributes)
endif()

add_test(NAME "${TS_NAME}${TS_SUFFIX}_api_ocl_30"
           COMMAND "${TS_BUILDDIR}/api/test_api" ${EXTRA_ARGS} buffer_properties_queries   clone_kernel   consistency_2d_image_from_buffer   consistency_3d_image_writes   consistency_depth_images   consistency_device_and_host_timer   consistency_device_enqueue   consistency_il_programs   consistency_memory_model   consistency_non_uniform_work_group   consistency_pipes   consistency_prog_ctor_dtor   consistency_progvar   consistency_read_write_images   consistency_subgroups   consistency_svm   context_destructor_callback   get_command_queue_info_compatibility   get_sampler_info_compatibility   image_properties_queries   ${API_OCL_30_EXTRA_TEST}    kernel_private_memory_size   min_image_formats   negative_get_platform_ids   negative_get_platform_info   pipe_properties_queries   queue_flush_on_release   queue_hint   queue_properties   queue_properties_queries   retain_mem_object_set_kernel_arg   sub_group_dispatch   zero_sized_enqueue)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_atomics"
         COMMAND "${TS_BUILDDIR}/atomics/test_atomics" ${EXTRA_ARGS} )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_basic_math"
         COMMAND "${TS_BUILDDIR}/basic/test_basic" ${EXTRA_ARGS} fpmath_float  fpmath_float2  fpmath_float4  intmath_int  intmath_int2  intmath_int4  intmath_long  intmath_long2  intmath_long4  int2float  float2int)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_basic_s2v"
         COMMAND "${TS_BUILDDIR}/basic/test_basic" ${EXTRA_ARGS} explicit_s2v_char  explicit_s2v_uchar  explicit_s2v_short  explicit_s2v_ushort  explicit_s2v_int  explicit_s2v_uint  explicit_s2v_long  explicit_s2v_ulong  explicit_s2v_float  explicit_s2v_double)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_basic_memory"
         COMMAND "${TS_BUILDDIR}/basic/test_basic" ${EXTRA_ARGS} vload_global  vload_local  vload_constant  vload_private  vstore_global  vstore_local  vstore_private  bufferreadwriterect  arrayreadwrite  arraycopy  enqueue_map_buffer  async_copy_global_to_local  async_copy_local_to_global  async_strided_copy_global_to_local  async_strided_copy_local_to_global  kernel_memory_alignment_local  kernel_memory_alignment_global  kernel_memory_alignment_constant  kernel_memory_alignment_private)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_basic_image"
         COMMAND "${TS_BUILDDIR}/basic/test_basic" ${EXTRA_ARGS} readimage  readimage_int16  readimage_fp32  writeimage  writeimage_int16  writeimage_fp32  image_r8  imagereadwrite  imagereadwrite3d  readimage3d  readimage3d_int16  readimage3d_fp32  imagearraycopy  imagearraycopy3d  imagecopy  imagecopy3d  imagerandomcopy  arrayimagecopy  arrayimagecopy3d  imagenpot  imagedim_pow2  imagedim_non_pow2  image_param  image_multipass_integer_coord  image_multipass_float_coord  enqueue_map_image)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_basic_other"
         COMMAND "${TS_BUILDDIR}/basic/test_basic" ${EXTRA_ARGS} mri_one  mri_multiple  barrier  createkernelsinprogram  work_item_functions  astype  prefetch  kernel_call_kernel_function  host_numeric_constants  kernel_numeric_constants  kernel_limit_constants  kernel_preprocessor_macros  parameter_types  vector_creation  vec_type_hint  global_work_offsets  get_global_offset  hostptr  hiloeo  if  sizeof  loop  pointer_cast  local_arg_def  local_kernel_def  local_kernel_scope  constant  constant_source)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_basic_other_ocl_30"
         COMMAND "${TS_BUILDDIR}/basic/test_basic" ${EXTRA_ARGS} async_copy_global_to_local2D async_copy_global_to_local3D  async_copy_local_to_global2D async_copy_local_to_global3D
          async_work_group_copy_fence_export_after_import_aliased_global
          async_work_group_copy_fence_export_after_import_aliased_global_and_local
          async_work_group_copy_fence_export_after_import_aliased_local
          async_work_group_copy_fence_import_after_export_aliased_global
          async_work_group_copy_fence_import_after_export_aliased_global_and_local
          async_work_group_copy_fence_import_after_export_aliased_local
          enqueued_local_size   get_linear_ids  global_linear_id  local_linear_id
          progvar_func_scope  progvar_prog_scope_init  progvar_prog_scope_misc   progvar_prog_scope_uninit
          rw_image_access_qualifier   simple_read_image_pitch   simple_write_image_pitch    vector_swizzle    wg_barrier)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_read_async"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS} buffer_read_async_int  buffer_read_async_uint  buffer_read_async_long  buffer_read_async_ulong  buffer_read_async_short  buffer_read_async_ushort  buffer_read_async_char  buffer_read_async_uchar  buffer_read_async_float)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_write_async"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS}  buffer_write_async_int  buffer_write_async_uint  buffer_write_async_short  buffer_write_async_ushort  buffer_write_async_char  buffer_write_async_uchar  buffer_write_async_float  buffer_write_async_long  buffer_write_async_ulong)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_read_array_barrier"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS} buffer_read_array_barrier_int  buffer_read_array_barrier_uint  buffer_read_array_barrier_long  buffer_read_array_barrier_ulong  buffer_read_array_barrier_short  buffer_read_array_barrier_ushort  buffer_read_array_barrier_char  buffer_read_array_barrier_uchar  buffer_read_array_barrier_float)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_read"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS} buffer_read_int  buffer_read_uint  buffer_read_long  buffer_read_ulong  buffer_read_short  buffer_read_ushort  buffer_read_float  buffer_read_char  buffer_read_uchar  buffer_read_struct  buffer_read_random_size buffer_read_half)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_write"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS} buffer_write_int  buffer_write_uint  buffer_write_short  buffer_write_ushort  buffer_write_char  buffer_write_uchar  buffer_write_float  buffer_write_long  buffer_write_ulong  buffer_write_struct buffer_write_half)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_fill"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS}   buffer_fill_int  buffer_fill_uint  buffer_fill_short  buffer_fill_ushort  buffer_fill_char  buffer_fill_uchar  buffer_fill_long  buffer_fill_ulong  buffer_fill_float  buffer_fill_struct)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_var1"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS} buffer_copy  buffer_partial_copy  mem_read_write_flags  mem_write_only_flags  mem_read_only_flags  mem_copy_host_flags  mem_alloc_ref_flags  array_info_size)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_var2"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS} sub_buffers_read_write  sub_buffers_read_write_dual_devices  sub_buffers_overlapping  buffer_migrate  image_migrate)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_map_read"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS} buffer_map_read_int  buffer_map_read_uint  buffer_map_read_long  buffer_map_read_ulong  buffer_map_read_short  buffer_map_read_ushort  buffer_map_read_char  buffer_map_read_uchar  buffer_map_read_float  buffer_map_read_struct)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_map_write"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS} buffer_map_write_int  buffer_map_write_uint  buffer_map_write_long  buffer_map_write_ulong  buffer_map_write_short  buffer_map_write_ushort  buffer_map_write_char  buffer_map_write_uchar  buffer_map_write_float  buffer_map_write_struct)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_commonfns"
         COMMAND "${TS_BUILDDIR}/commonfns/test_commonfns" ${EXTRA_ARGS} )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_compiler"
         COMMAND "${TS_BUILDDIR}/compiler/test_compiler"
         WORKING_DIRECTORY "${TS_BASEDIR}/src/${TS_NAME}/test_conformance/compiler")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_1"
         COMMAND "${TS_BUILDDIR}/c11_atomics/test_c11_atomics" ${EXTRA_ARGS} atomic_store)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_2"
         COMMAND "${TS_BUILDDIR}/c11_atomics/test_c11_atomics" ${EXTRA_ARGS} atomic_load)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_3"
         COMMAND "${TS_BUILDDIR}/c11_atomics/test_c11_atomics" ${EXTRA_ARGS} atomic_fetch_max)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_4"
         COMMAND "${TS_BUILDDIR}/c11_atomics/test_c11_atomics" ${EXTRA_ARGS} atomic_fetch_xor)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_5"
         COMMAND "${TS_BUILDDIR}/c11_atomics/test_c11_atomics" ${EXTRA_ARGS} atomic_fetch_add)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_c11_atomics_init_store_load_exchange"
         COMMAND "${TS_BUILDDIR}/c11_atomics/test_c11_atomics" ${EXTRA_ARGS} atomic_init atomic_store atomic_load atomic_exchange atomic_compare_exchange_weak atomic_compare_exchange_strong)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_c11_atomics_add_sub_min_max"
         COMMAND "${TS_BUILDDIR}/c11_atomics/test_c11_atomics" ${EXTRA_ARGS} atomic_fetch_add atomic_fetch_sub atomic_fetch_min atomic_fetch_max)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_c11_atomics_flag_fence"
         COMMAND "${TS_BUILDDIR}/c11_atomics/test_c11_atomics" ${EXTRA_ARGS} atomic_flag atomic_fence)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_c11_atomics_or_xor_and"
         COMMAND "${TS_BUILDDIR}/c11_atomics/test_c11_atomics" ${EXTRA_ARGS} atomic_fetch_and atomic_fetch_or atomic_fetch_orand atomic_fetch_xor atomic_fetch_xor2)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_c11_atomics_svm_init_store_load_exchange"
         COMMAND "${TS_BUILDDIR}/c11_atomics/test_c11_atomics" ${EXTRA_ARGS} svm_atomic_init svm_atomic_store svm_atomic_load svm_atomic_exchange svm_atomic_compare_exchange_weak svm_atomic_compare_exchange_strong)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_c11_atomics_svm_add_sub_min_max"
         COMMAND "${TS_BUILDDIR}/c11_atomics/test_c11_atomics" ${EXTRA_ARGS} svm_atomic_fetch_add svm_atomic_fetch_sub svm_atomic_fetch_min svm_atomic_fetch_max)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_c11_atomics_svm_flag_fence"
         COMMAND "${TS_BUILDDIR}/c11_atomics/test_c11_atomics" ${EXTRA_ARGS} svm_atomic_flag svm_atomic_fence)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_c11_atomics_svm_or_xor_and"
         COMMAND "${TS_BUILDDIR}/c11_atomics/test_c11_atomics" ${EXTRA_ARGS} svm_atomic_fetch_and svm_atomic_fetch_or svm_atomic_fetch_orand svm_atomic_fetch_xor svm_atomic_fetch_xor2)


add_test(NAME "${TS_NAME}${TS_SUFFIX}_computeinfo"
         COMMAND "${TS_BUILDDIR}/computeinfo/test_computeinfo" ${EXTRA_ARGS} )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_contractions"
         COMMAND "${TS_BUILDDIR}/contractions/test_contractions" ${EXTRA_ARGS} )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uchar_uchar"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uchar_uchar uchar_sat_uchar )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_char_uchar"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} char_uchar char_sat_uchar )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ushort_uchar"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ushort_uchar ushort_sat_uchar )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_short_uchar"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} short_uchar short_sat_uchar )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uint_uchar"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uint_uchar uint_sat_uchar )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_int_uchar"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} int_uchar int_sat_uchar )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_float_uchar"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} float_uchar float_rte_uchar float_rtp_uchar float_rtn_uchar float_rtz_uchar )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_double_uchar"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} double_uchar double_rte_uchar double_rtp_uchar double_rtn_uchar double_rtz_uchar )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ulong_uchar"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ulong_uchar ulong_sat_uchar )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_long_uchar"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} long_uchar long_sat_uchar )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uchar_char"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uchar_char uchar_sat_char )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_char_char"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} char_char char_sat_char )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ushort_char"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ushort_char ushort_sat_char )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_short_char"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} short_char short_sat_char )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uint_char"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uint_char uint_sat_char )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_int_char"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} int_char int_sat_char )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_float_char"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} float_char float_rte_char float_rtp_char float_rtn_char float_rtz_char )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_double_char"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} double_char double_rte_char double_rtp_char double_rtn_char double_rtz_char )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ulong_char"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ulong_char ulong_sat_char )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_long_char"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} long_char long_sat_char )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uchar_ushort"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uchar_ushort uchar_sat_ushort )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_char_ushort"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} char_ushort char_sat_ushort )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ushort_ushort"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ushort_ushort ushort_sat_ushort )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_short_ushort"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} short_ushort short_sat_ushort )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uint_ushort"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uint_ushort uint_sat_ushort )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_int_ushort"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} int_ushort int_sat_ushort )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_float_ushort"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} float_ushort float_rte_ushort float_rtp_ushort float_rtn_ushort float_rtz_ushort )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_double_ushort"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} double_ushort double_rte_ushort double_rtp_ushort double_rtn_ushort double_rtz_ushort )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ulong_ushort"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ulong_ushort ulong_sat_ushort )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_long_ushort"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} long_ushort long_sat_ushort )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uchar_short"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uchar_short uchar_sat_short )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_char_short"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} char_short char_sat_short )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ushort_short"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ushort_short ushort_sat_short )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_short_short"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} short_short short_sat_short )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uint_short"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uint_short uint_sat_short )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_int_short"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} int_short int_sat_short )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_float_short"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} float_short float_rte_short float_rtp_short float_rtn_short float_rtz_short )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_double_short"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} double_short double_rte_short double_rtp_short double_rtn_short double_rtz_short )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ulong_short"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ulong_short ulong_sat_short )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_long_short"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} long_short long_sat_short )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uchar_uint"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uchar_uint uchar_sat_uint )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_char_uint"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} char_uint char_sat_uint )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ushort_uint"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ushort_uint ushort_sat_uint )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_short_uint"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} short_uint short_sat_uint )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uint_uint"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uint_uint uint_sat_uint )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_int_uint"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} int_uint int_sat_uint )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_float_uint"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} float_uint float_rte_uint float_rtp_uint float_rtn_uint float_rtz_uint )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_double_uint"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} double_uint double_rte_uint double_rtp_uint double_rtn_uint double_rtz_uint )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ulong_uint"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ulong_uint ulong_sat_uint )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_long_uint"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} long_uint long_sat_uint )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uchar_int"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uchar_int uchar_sat_int )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_char_int"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} char_int char_sat_int )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ushort_int"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ushort_int ushort_sat_int )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_short_int"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} short_int short_sat_int )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uint_int"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uint_int uint_sat_int )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_int_int"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} int_int int_sat_int )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_float_int"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} float_int float_rte_int float_rtp_int float_rtn_int float_rtz_int )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_double_int"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} double_int double_rte_int double_rtp_int double_rtn_int double_rtz_int )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ulong_int"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ulong_int ulong_sat_int )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_long_int"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} long_int long_sat_int )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uchar_float"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uchar_float uchar_sat_float uchar_rte_float uchar_sat_rte_float uchar_rtp_float uchar_sat_rtp_float uchar_rtn_float uchar_sat_rtn_float uchar_rtz_float uchar_sat_rtz_float )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_char_float"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} char_float char_sat_float char_rte_float char_sat_rte_float char_rtp_float char_sat_rtp_float char_rtn_float char_sat_rtn_float char_rtz_float char_sat_rtz_float )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ushort_float"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ushort_float ushort_sat_float ushort_rte_float ushort_sat_rte_float ushort_rtp_float ushort_sat_rtp_float ushort_rtn_float ushort_sat_rtn_float ushort_rtz_float ushort_sat_rtz_float )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_short_float"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} short_float short_sat_float short_rte_float short_sat_rte_float short_rtp_float short_sat_rtp_float short_rtn_float short_sat_rtn_float short_rtz_float short_sat_rtz_float )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uint_float"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uint_float uint_sat_float uint_rte_float uint_sat_rte_float uint_rtp_float uint_sat_rtp_float uint_rtn_float uint_sat_rtn_float uint_rtz_float uint_sat_rtz_float )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_int_float"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} int_float int_sat_float int_rte_float int_sat_rte_float int_rtp_float int_sat_rtp_float int_rtn_float int_sat_rtn_float int_rtz_float int_sat_rtz_float )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_float_float"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} float_float float_rte_float float_rtp_float float_rtn_float float_rtz_float )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_double_float"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} double_float double_rte_float double_rtp_float double_rtn_float double_rtz_float )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ulong_float"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ulong_float ulong_sat_float ulong_rte_float ulong_sat_rte_float ulong_rtp_float ulong_sat_rtp_float ulong_rtn_float ulong_sat_rtn_float ulong_rtz_float ulong_sat_rtz_float )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_long_float"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} long_float long_sat_float long_rte_float long_sat_rte_float long_rtp_float long_sat_rtp_float long_rtn_float long_sat_rtn_float long_rtz_float long_sat_rtz_float )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uchar_double"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uchar_double uchar_sat_double uchar_rte_double uchar_sat_rte_double uchar_rtp_double uchar_sat_rtp_double uchar_rtn_double uchar_sat_rtn_double uchar_rtz_double uchar_sat_rtz_double )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_char_double"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} char_double char_sat_double char_rte_double char_sat_rte_double char_rtp_double char_sat_rtp_double char_rtn_double char_sat_rtn_double char_rtz_double char_sat_rtz_double )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ushort_double"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ushort_double ushort_sat_double ushort_rte_double ushort_sat_rte_double ushort_rtp_double ushort_sat_rtp_double ushort_rtn_double ushort_sat_rtn_double ushort_rtz_double ushort_sat_rtz_double )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_short_double"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} short_double short_sat_double short_rte_double short_sat_rte_double short_rtp_double short_sat_rtp_double short_rtn_double short_sat_rtn_double short_rtz_double short_sat_rtz_double )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uint_double"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uint_double uint_sat_double uint_rte_double uint_sat_rte_double uint_rtp_double uint_sat_rtp_double uint_rtn_double uint_sat_rtn_double uint_rtz_double uint_sat_rtz_double )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_int_double"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} int_double int_sat_double int_rte_double int_sat_rte_double int_rtp_double int_sat_rtp_double int_rtn_double int_sat_rtn_double int_rtz_double int_sat_rtz_double )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_float_double"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} float_double float_rte_double float_rtp_double float_rtn_double float_rtz_double )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_double_double"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} double_double double_rte_double double_rtp_double double_rtn_double double_rtz_double )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ulong_double"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ulong_double ulong_sat_double ulong_rte_double ulong_sat_rte_double ulong_rtp_double ulong_sat_rtp_double ulong_rtn_double ulong_sat_rtn_double ulong_rtz_double ulong_sat_rtz_double )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_long_double"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} long_double long_sat_double long_rte_double long_sat_rte_double long_rtp_double long_sat_rtp_double long_rtn_double long_sat_rtn_double long_rtz_double long_sat_rtz_double )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uchar_ulong"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uchar_ulong uchar_sat_ulong )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_char_ulong"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} char_ulong char_sat_ulong )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ushort_ulong"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ushort_ulong ushort_sat_ulong )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_short_ulong"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} short_ulong short_sat_ulong )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uint_ulong"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uint_ulong uint_sat_ulong )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_int_ulong"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} int_ulong int_sat_ulong )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_float_ulong"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} float_ulong float_rte_ulong float_rtp_ulong float_rtn_ulong float_rtz_ulong )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_double_ulong"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} double_ulong double_rte_ulong double_rtp_ulong double_rtn_ulong double_rtz_ulong )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ulong_ulong"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ulong_ulong ulong_sat_ulong )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_long_ulong"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} long_ulong long_sat_ulong )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uchar_long"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uchar_long uchar_sat_long )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_char_long"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} char_long char_sat_long )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ushort_long"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ushort_long ushort_sat_long )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_short_long"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} short_long short_sat_long )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_uint_long"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} uint_long uint_sat_long )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_int_long"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} int_long int_sat_long )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_float_long"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} float_long float_rte_long float_rtp_long float_rtn_long float_rtz_long )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_double_long"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} double_long double_rte_long double_rtp_long double_rtn_long double_rtz_long )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_ulong_long"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} ulong_long ulong_sat_long )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_long_long"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} long_long long_sat_long )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_device_partition"
         COMMAND "${TS_BUILDDIR}/device_partition/test_device_partition")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_device_execution"
         COMMAND "${TS_BUILDDIR}/device_execution/test_device_execution")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_device_timer"
         COMMAND "${TS_BUILDDIR}/device_timer/test_device_timer")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_events"
         COMMAND "${TS_BUILDDIR}/events/test_events" ${EXTRA_ARGS} event_get_execute_status  event_get_write_array_status  event_get_read_array_status  event_get_info  event_wait_for_execute  event_wait_for_array  event_flush  event_finish_execute  event_finish_array  event_release_before_done  event_enqueue_marker  event_enqueue_marker_with_event_list  event_enqueue_barrier_with_event_list)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_events_ooo"
         COMMAND "${TS_BUILDDIR}/events/test_events" ${EXTRA_ARGS} out_of_order_event_waitlist_single_queue  out_of_order_event_waitlist_multi_queue  out_of_order_event_waitlist_multi_queue_multi_device  out_of_order_event_enqueue_wait_for_events_single_queue  out_of_order_event_enqueue_wait_for_events_multi_queue  out_of_order_event_enqueue_wait_for_events_multi_queue_multi_device  out_of_order_event_enqueue_marker_single_queue  out_of_order_event_enqueue_marker_multi_queue  out_of_order_event_enqueue_marker_multi_queue_multi_device  out_of_order_event_enqueue_barrier_single_queue)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_events_other"
         COMMAND "${TS_BUILDDIR}/events/test_events" ${EXTRA_ARGS} userevents  waitlists  callbacks  callbacks_simultaneous  userevents_multithreaded)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_generic_address_space"
         COMMAND "${TS_BUILDDIR}/generic_address_space/test_generic_address_space")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_geometrics"
         COMMAND "${TS_BUILDDIR}/geometrics/test_geometrics" ${EXTRA_ARGS} )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_half_1"
         COMMAND "${TS_BUILDDIR}/half/test_half" ${EXTRA_ARGS} vload_half vloada_half vstore_half)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_half_2"
         COMMAND "${TS_BUILDDIR}/half/test_half" ${EXTRA_ARGS} vstorea_half vstore_half_rte vstorea_half_rte)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_half_3"
         COMMAND "${TS_BUILDDIR}/half/test_half" ${EXTRA_ARGS} vstore_half_rtz vstore_half_rtn vstore_half_rtp)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_half_4"
         COMMAND "${TS_BUILDDIR}/half/test_half" ${EXTRA_ARGS} vstorea_half_rtp vstorea_half_rtz vstorea_half_rtn)


add_test(NAME "${TS_NAME}${TS_SUFFIX}_mini_half"
         COMMAND "${TS_BUILDDIR}/half/test_half" ${EXTRA_ARGS} -w)
set(OCL_FULL_TESTS_FP16
    "${TS_NAME}${TS_SUFFIX}_half_1"
    "${TS_NAME}${TS_SUFFIX}_half_2"
    "${TS_NAME}${TS_SUFFIX}_half_3"
    "${TS_NAME}${TS_SUFFIX}_half_4")
set(OCL_MINI_TESTS_FP16
    "${TS_NAME}${TS_SUFFIX}_mini_half")

#***************************************************************************

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_copy_images"
         COMMAND "${TS_BUILDDIR}/images/clCopyImage/test_cl_copy_images" ${EXTRA_ARGS} )
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_copy_images_max"
         COMMAND "${TS_BUILDDIR}/images/clCopyImage/test_cl_copy_images" ${EXTRA_ARGS} max_images)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_copy_images_min"
         COMMAND "${TS_BUILDDIR}/images/clCopyImage/test_cl_copy_images" ${EXTRA_ARGS} small_images)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_fill_images"
         COMMAND "${TS_BUILDDIR}/images/clFillImage/test_cl_fill_images" ${EXTRA_ARGS} )
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_fill_images_min"
         COMMAND "${TS_BUILDDIR}/images/clFillImage/test_cl_fill_images" ${EXTRA_ARGS} small_images)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_fill_images_max"
         COMMAND "${TS_BUILDDIR}/images/clFillImage/test_cl_fill_images" ${EXTRA_ARGS} max_images)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_fill_images_pitch"
         COMMAND "${TS_BUILDDIR}/images/clFillImage/test_cl_fill_images" ${EXTRA_ARGS} use_pitches)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_get_info"
         COMMAND "${TS_BUILDDIR}/images/clGetInfo/test_cl_get_info" ${EXTRA_ARGS} )
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_get_info_min"
         COMMAND "${TS_BUILDDIR}/images/clGetInfo/test_cl_get_info" ${EXTRA_ARGS} small_images)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_get_info_max"
         COMMAND "${TS_BUILDDIR}/images/clGetInfo/test_cl_get_info" ${EXTRA_ARGS} max_images)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_get_info_random"
         COMMAND "${TS_BUILDDIR}/images/clGetInfo/test_cl_get_info" ${EXTRA_ARGS} randomize)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_read_write_images"
         COMMAND "${TS_BUILDDIR}/images/clReadWriteImage/test_cl_read_write_images" ${EXTRA_ARGS} )
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_read_write_images_min"
         COMMAND "${TS_BUILDDIR}/images/clReadWriteImage/test_cl_read_write_images" ${EXTRA_ARGS} small_images)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_read_write_images_max"
         COMMAND "${TS_BUILDDIR}/images/clReadWriteImage/test_cl_read_write_images" ${EXTRA_ARGS} max_images)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_read_write_images_pitch"
         COMMAND "${TS_BUILDDIR}/images/clReadWriteImage/test_cl_read_write_images" ${EXTRA_ARGS} use_pitches)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_kernel_image_methods_1D"
         COMMAND "${TS_BUILDDIR}/images/kernel_image_methods/test_kernel_image_methods" ${EXTRA_ARGS} 1D)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_kernel_image_methods_2D"
         COMMAND "${TS_BUILDDIR}/images/kernel_image_methods/test_kernel_image_methods" ${EXTRA_ARGS} 2D)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_kernel_image_methods_3D"
         COMMAND "${TS_BUILDDIR}/images/kernel_image_methods/test_kernel_image_methods" ${EXTRA_ARGS} 3D)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_kernel_image_methods_1Darray"
         COMMAND "${TS_BUILDDIR}/images/kernel_image_methods/test_kernel_image_methods" ${EXTRA_ARGS} 1Darray)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_kernel_image_methods_2Darray"
         COMMAND "${TS_BUILDDIR}/images/kernel_image_methods/test_kernel_image_methods" ${EXTRA_ARGS} 2Darray)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_image_streams_1D"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" "1D" ${EXTRA_ARGS} )
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_image_streams_2D"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" "2D" ${EXTRA_ARGS} )
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_image_streams_3D"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" ${EXTRA_ARGS} "3D")
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_image_streams_1Darray"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" ${EXTRA_ARGS} "1Darray")
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_image_streams_2Darray"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" ${EXTRA_ARGS} "2Darray")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_image_streams_1D_pitch"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" "1D" ${EXTRA_ARGS}  use_pitches)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_image_streams_2D_pitch"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" "2D" ${EXTRA_ARGS}  use_pitches)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_image_streams_3D_pitch"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" "3D" ${EXTRA_ARGS} use_pitches)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_image_streams_1Darray_pitch"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" "1Darray" ${EXTRA_ARGS} use_pitches)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_image_streams_2Darray_pitch"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" "2Darray" ${EXTRA_ARGS} use_pitches)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_1D"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS} 1D)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_2D"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS} 2D)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_3D"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS} 3D)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_1Darray"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS} 1Darray)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_2Darray"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS} 2Darray)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_1D_pitch"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS} 1D use_pitches)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_2D_pitch"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS} 2D use_pitches)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_3D_pitch"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS} 3D use_pitches)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_1Darray_pitch"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS} 1Darray use_pitches)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_2Darray_pitch"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS} 2Darray use_pitches)

#***************************************************************************

add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops_basic"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${EXTRA_ARGS} integer_clz  integer_hadd  integer_rhadd  integer_mul_hi  integer_rotate  integer_clamp  integer_mad_sat  integer_mad_hi  integer_min  integer_max  integer_upsample  integer_abs  integer_abs_diff  integer_add_sat  integer_sub_sat  integer_addAssign  integer_subtractAssign  integer_multiplyAssign  integer_divideAssign  integer_moduloAssign  integer_andAssign  integer_orAssign  integer_exclusiveOrAssign)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops_unary"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${EXTRA_ARGS}  unary_ops_increment  unary_ops_decrement  unary_ops_full)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops_i24"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${EXTRA_ARGS}  integer_mul24  integer_mad24 )
add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops_long"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${EXTRA_ARGS}  long_math  long_logic long_shift  long_compare  )
add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops_ulong"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${EXTRA_ARGS}  ulong_math  ulong_logic  ulong_shift  ulong_compare )
add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops_int"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${EXTRA_ARGS}  int_math  int_logic  int_shift  int_compare )
add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops_uint"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${EXTRA_ARGS}  uint_math  uint_logic  uint_shift  uint_compare )
add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops_short"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${EXTRA_ARGS}  short_math  short_logic  short_shift  short_compare )
add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops_ushort"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${EXTRA_ARGS}  ushort_math  ushort_logic  ushort_shift  ushort_compare )
add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops_char"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${EXTRA_ARGS}  char_math  char_logic  char_shift  char_compare  )
add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops_uchar"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${EXTRA_ARGS}  uchar_math  uchar_logic  uchar_shift  uchar_compare )
add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops_other"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${EXTRA_ARGS}  popcount  vector_scalar )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops_ocl_30"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${EXTRA_ARGS}  integer_ctz integer_dot_product )

#***************************************************************************

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_add"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} add)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_assignment"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} assignment)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_cbrt"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} cbrt)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_ceil"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} ceil)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_copysign"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} copysign)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_cos"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} cos)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_cosh"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} cosh)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_cospi"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} cospi)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_divide"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} divide)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_exp"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} exp)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_exp10"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} exp10)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_exp2"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} exp2)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_expm1"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} expm1)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_fabs"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} fabs)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_fdim"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} fdim)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_floor"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} floor)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_fma"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} fma)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_fmax"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} fmax)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_fmin"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} fmin)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_fmod"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} fmod)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_fract"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} fract)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_frexp"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} frexp)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_hypot"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} hypot)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_ilogb"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} ilogb)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_isequal"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} isequal)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_isfinite"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} isfinite)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_isgreater"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} isgreater)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_isgreaterequal"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} isgreaterequal)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_isinf"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} isinf)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_isless"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} isless)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_islessequal"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} islessequal)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_islessgreater"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} islessgreater)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_isnan"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} isnan)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_isnormal"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} isnormal)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_isnotequal"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} isnotequal)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_isordered"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} isordered)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_isunordered"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} isunordered)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_ldexp"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} ldexp)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_log"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} log)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_log10"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} log10)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_log1p"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} log1p)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_log2"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} log2)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_logb"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} logb)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mad"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} mad)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_maxmag"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} maxmag)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_minmag"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} minmag)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_modf"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} modf)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_multiply"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} multiply)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_nan"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} nan)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_nextafter"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} nextafter)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_not"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} not)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_pow"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} pow)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_pown"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} pown)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_powr"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} powr)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_remainder"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} remainder)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_remquo"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} remquo)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_rint"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} rint)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_rootn"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} rootn)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_round"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} round)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_rsqrt"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} rsqrt)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_signbit"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} signbit)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_sin"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} sin)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_sincos"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} sincos)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_sinh"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} sinh)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_sinpi"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} sinpi)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_sqrt"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} sqrt)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_subtract"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} subtract)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_tan"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} tan)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_tanh"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} tanh)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_tanpi"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} tanpi)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_trunc"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS} trunc)

#***************************************************************************

add_test(NAME "${TS_NAME}${TS_SUFFIX}_mem_host_flags"
         COMMAND "${TS_BUILDDIR}/mem_host_flags/test_mem_host_flags" ${EXTRA_ARGS} )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_multiples"
         COMMAND "${TS_BUILDDIR}/multiple_device_context/test_multiples" ${EXTRA_ARGS} )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_non_uniform_work_group"
         COMMAND "${TS_BUILDDIR}/non_uniform_work_group/test_non_uniform_work_group" ${EXTRA_ARGS} )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_pipes"
         COMMAND "${TS_BUILDDIR}/pipes/test_pipes" ${EXTRA_ARGS} )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_printf"
         COMMAND "${TS_BUILDDIR}/printf/test_printf"
   WORKING_DIRECTORY "${TS_BASEDIR}/src/${TS_NAME}/test_conformance/printf" )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_profiling"
         COMMAND "${TS_BUILDDIR}/profiling/test_profiling" ${EXTRA_ARGS} )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_relationals"
         COMMAND "${TS_BUILDDIR}/relationals/test_relationals" ${EXTRA_ARGS} relational_any  relational_all  relational_bitselect  relational_select_signed  relational_select_unsigned  relational_isequal  relational_isnotequal  relational_isgreater  relational_isgreaterequal  relational_isless  relational_islessequal  relational_islessgreater)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_relationals_shuffle"
         COMMAND "${TS_BUILDDIR}/relationals/test_relationals" ${EXTRA_ARGS} shuffle_copy  shuffle_function_call  shuffle_array_cast  shuffle_built_in  shuffle_built_in_dual_input)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_uchar_uchar"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_uchar_uchar)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_uchar_char"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_uchar_char)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_char_uchar"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_char_uchar)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_char_char"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_char_char)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_ushort_ushort"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_ushort_ushort)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_ushort_short"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_ushort_short)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_short_ushort"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_short_ushort)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_short_short"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_short_short)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_uint_uint"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_uint_uint)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_uint_int"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_uint_int)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_int_uint"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_int_uint)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_int_int"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_int_int)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_float_uint"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_float_uint)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_float_int"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_float_int)


add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_ulong_ulong"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_ulong_ulong)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_ulong_long"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_ulong_long)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_long_ulong"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_long_ulong)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_long_long"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_long_long)

# we don't support the old (1.2 / 2.0) spir
#add_test(NAME "${TS_NAME}${TS_SUFFIX}_spir"
#         COMMAND "${TS_BUILDDIR}/spir/test_spir")

if(HAVE_SPIRV_TOOLS)

# this test requires a fix in llvm-spirv
# https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/1630
option(LLVM_SPIRV_SPEC_CONST "llvm-spirv has working --spec-const option" OFF)
unset(SPIRV_OCL_30_EXTRA_TEST)
if(LLVM_SPIRV_SPEC_CONST OR (LLVM_VERSION VERSION_GREATER_EQUAL 16.0))
  set(SPIRV_OCL_30_EXTRA_TEST op_spec_constant_double_simple)
endif()

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_1"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS} ext_cl_khr_spirv_no_integer_wrap_decoration_fadd_int  ext_cl_khr_spirv_no_integer_wrap_decoration_fsub_int  ext_cl_khr_spirv_no_integer_wrap_decoration_fmul_int  ext_cl_khr_spirv_no_integer_wrap_decoration_fshiftleft_int  ext_cl_khr_spirv_no_integer_wrap_decoration_fnegate_int  ext_cl_khr_spirv_no_integer_wrap_decoration_fadd_uint  ext_cl_khr_spirv_no_integer_wrap_decoration_fsub_uint  ext_cl_khr_spirv_no_integer_wrap_decoration_fmul_uint  ext_cl_khr_spirv_no_integer_wrap_decoration_fshiftleft_uint
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_2"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS} decorate_restrict  decorate_aliased  decorate_alignment  decorate_constant  decorate_cpacked  decorate_saturated_conversion_char  decorate_saturated_conversion_uchar  decorate_saturated_conversion_short  decorate_saturated_conversion_ushort  decorate_saturated_conversion_int  decorate_saturated_conversion_uint  decorate_fp_rounding_mode_rte_float_int  decorate_fp_rounding_mode_rtz_float_int  decorate_fp_rounding_mode_rtp_float_int  decorate_fp_rounding_mode_rtn_float_int  decorate_fp_rounding_mode_rte_double_long  decorate_fp_rounding_mode_rtz_double_long  decorate_fp_rounding_mode_rtp_double_long  decorate_fp_rounding_mode_rtn_double_long
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_3"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS} get_program_il  linkage_export_function_compile  linkage_import_function_compile  linkage_import_function_link  op_atomic_inc_global  op_atomic_dec_global  op_label_simple  op_branch_simple  op_unreachable_simple  op_branch_conditional  op_branch_conditional_weighted  op_composite_construct_int4  op_composite_construct_struct
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_4"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS} op_constant_true_simple  op_constant_false_simple  op_constant_int_simple  op_constant_uint_simple  op_constant_char_simple  op_constant_uchar_simple  op_constant_ushort_simple  op_constant_long_simple  op_constant_ulong_simple  op_constant_short_simple  op_constant_float_simple  op_constant_double_simple  op_constant_int4_simple  op_constant_int3_simple  op_constant_struct_int_float_simple  op_constant_struct_int_char_simple  op_constant_struct_struct_simple
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_5"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS} op_copy_int_simple  op_copy_uint_simple  op_copy_char_simple  op_copy_uchar_simple  op_copy_ushort_simple  op_copy_long_simple  op_copy_ulong_simple  op_copy_short_simple  op_copy_float_simple  op_copy_double_simple  op_copy_int4_simple  op_copy_int3_simple  op_copy_struct_int_float_simple  op_copy_struct_int_char_simple  op_copy_struct_struct_simple
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_6"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS} op_fadd_float_regular  op_fsub_float_regular  op_fmul_float_regular  op_fdiv_float_regular  op_frem_float_regular  op_fmod_float_regular  op_fadd_float_fast  op_fsub_float_fast  op_fmul_float_fast  op_fdiv_float_fast  op_frem_float_fast  op_fmod_float_fast  op_fadd_double_regular  op_fsub_double_regular  op_fmul_double_regular  op_fdiv_double_regular  op_frem_double_regular  op_fmod_double_regular  op_fadd_double_fast  op_fsub_double_fast  op_fmul_double_fast  op_fdiv_double_fast  op_frem_double_fast  op_fmod_double_fast  op_fadd_float4_regular  op_fsub_float4_regular  op_fmul_float4_regular  op_fdiv_float4_regular  op_frem_float4_regular  op_fmod_float4_regular  op_fadd_float4_fast  op_fsub_float4_fast  op_fmul_float4_fast  op_fdiv_float4_fast  op_frem_float4_fast  op_fmod_float4_fast  op_fadd_double2_regular  op_fsub_double2_regular  op_fmul_double2_regular  op_fdiv_double2_regular  op_frem_double2_regular  op_fmod_double2_regular  op_fadd_double2_fast  op_fsub_double2_fast  op_fmul_double2_fast  op_fdiv_double2_fast  op_frem_double2_fast  op_fmod_double2_fast
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_7"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS}  function_none  function_inline  function_noinline  function_pure  function_const  function_pure_ptr
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_8"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS}  op_lifetime_simple  op_loop_merge_branch_none  op_loop_merge_branch_unroll  op_loop_merge_branch_dont_unroll  op_loop_merge_branch_conditional_none  op_loop_merge_branch_conditional_unroll  op_loop_merge_branch_conditional_dont_unroll  op_neg_float  op_neg_double  op_neg_int  op_neg_long  op_not_int  op_not_long  op_neg_short  op_not_short  op_neg_float4  op_neg_int4  op_not_int4  op_type_opaque_simple  op_phi_2_blocks  op_phi_3_blocks  op_phi_4_blocks
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_9"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS}  op_selection_merge_if_none  op_selection_merge_if_flatten  op_selection_merge_if_dont_flatten  op_selection_merge_swith_none  op_selection_merge_swith_flatten  op_selection_merge_swith_dont_flatten  op_spec_constant_uint_simple  op_spec_constant_uchar_simple  op_spec_constant_ushort_simple  op_spec_constant_ulong_simple  op_spec_constant_float_simple   ${SPIRV_OCL_30_EXTRA_TEST}  op_spec_constant_true_simple  op_spec_constant_false_simple
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_0"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS}  op_undef_true_simple  op_undef_false_simple  op_undef_int_simple  op_undef_uint_simple  op_undef_char_simple  op_undef_uchar_simple  op_undef_ushort_simple  op_undef_long_simple  op_undef_ulong_simple  op_undef_short_simple  op_undef_float_simple  op_undef_double_simple  op_undef_int4_simple  op_undef_int3_simple  op_undef_struct_int_float_simple  op_undef_struct_int_char_simple  op_undef_struct_struct_simple   op_vector_int4_extract  op_vector_float4_extract  op_vector_long2_extract  op_vector_double2_extract  op_vector_char16_extract  op_vector_int4_insert  op_vector_float4_insert  op_vector_long2_insert  op_vector_double2_insert  op_vector_char16_insert  op_vector_times_scalar_float  op_vector_times_scalar_double
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_half"
        COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS}   op_fadd_half_regular  op_fsub_half_regular  op_fmul_half_regular  op_fdiv_half_regular  op_frem_half_regular  op_fmod_half_regular  op_fadd_half_fast  op_fsub_half_fast  op_fmul_half_fast  op_fdiv_half_fast  op_frem_half_fast  op_fmod_half_fast op_constant_half_simple op_copy_half_simple  op_spec_constant_half_simple op_undef_half_simple
        WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_micro_0"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS} decorate_restrict  decorate_aliased  decorate_alignment  decorate_constant  decorate_cpacked  decorate_saturated_conversion_char   decorate_saturated_conversion_ushort  decorate_saturated_conversion_int  decorate_fp_rounding_mode_rte_float_int  decorate_fp_rounding_mode_rtz_double_long get_program_il  linkage_export_function_compile  linkage_import_function_compile  linkage_import_function_link  op_atomic_inc_global  op_atomic_dec_global  op_label_simple  op_branch_simple  op_unreachable_simple  op_branch_conditional  op_branch_conditional_weighted  op_composite_construct_int4  op_composite_construct_struct
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_micro_1"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS} op_constant_false_simple  op_constant_int_simple  op_constant_uchar_simple op_constant_long_simple op_constant_float_simple         op_constant_struct_int_char_simple  op_constant_struct_struct_simple op_copy_int3_simple  op_copy_struct_int_float_simple  op_copy_struct_int_char_simple  op_copy_struct_struct_simple  op_copy_half_simple op_fadd_float_regular  op_fmul_float_regular  op_frem_float_regular  op_fadd_float_fast  op_fadd_double_regular  op_frem_double_regular op_fsub_double_fast  op_fdiv_double_fast  op_fadd_float4_regular  op_fmul_float4_regular  op_fsub_float4_fast  op_fadd_double2_regular  op_fsub_double2_fast  op_fmod_double2_fast
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_micro_2"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS} function_none  function_inline  function_noinline op_lifetime_simple  op_loop_merge_branch_none op_neg_float  op_neg_double op_not_int4  op_type_opaque_simple  op_phi_2_blocks op_selection_merge_if_none  op_selection_merge_if_flatten op_spec_constant_uchar_simple  op_spec_constant_ushort_simple ${SPIRV_OCL_30_EXTRA_TEST}  op_spec_constant_true_simple op_undef_true_simple op_undef_uint_simple  op_undef_char_simple op_undef_float_simple  op_undef_double_simple op_vector_int4_extract  op_vector_float4_extract op_vector_double2_extract  op_vector_char16_extract  op_vector_int4_insert op_vector_times_scalar_double
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

endif()

add_test(NAME "${TS_NAME}${TS_SUFFIX}_subgroups_1"
         COMMAND "${TS_BUILDDIR}/subgroups/test_subgroups" ${EXTRA_ARGS} sub_group_info_ext  sub_group_info_core  work_item_functions_ext  work_item_functions_core  subgroup_functions_ext  subgroup_functions_core  barrier_functions_ext  barrier_functions_core)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_subgroups_2"
         COMMAND "${TS_BUILDDIR}/subgroups/test_subgroups" ${EXTRA_ARGS} ifp_ext  ifp_core  subgroup_functions_extended_types  subgroup_functions_non_uniform_vote  subgroup_functions_non_uniform_arithmetic  subgroup_functions_ballot  subgroup_functions_clustered_reduce  subgroup_functions_shuffle  subgroup_functions_shuffle_relative)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_SVM_1"
         COMMAND "${TS_BUILDDIR}/SVM/test_svm" ${EXTRA_ARGS} svm_byte_granularity  svm_set_kernel_exec_info_svm_ptrs  svm_fine_grain_memory_consistency  svm_fine_grain_sync_buffers  svm_shared_address_space_fine_grain  svm_shared_sub_buffers  svm_shared_address_space_fine_grain_buffers)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_SVM_2"
         COMMAND "${TS_BUILDDIR}/SVM/test_svm" ${EXTRA_ARGS} svm_allocate_shared_buffer  svm_shared_address_space_coarse_grain_old_api  svm_shared_address_space_coarse_grain_new_api  svm_cross_buffer_pointers_coarse_grain  svm_pointer_passing  svm_enqueue_api  svm_migrate)


add_test(NAME "${TS_NAME}${TS_SUFFIX}_thread_dimensions_quick"
         COMMAND "${TS_BUILDDIR}/thread_dimensions/test_thread_dimensions" ${EXTRA_ARGS} quick_1d_explicit_local  quick_2d_explicit_local  quick_3d_explicit_local  quick_1d_implicit_local  quick_2d_implicit_local  quick_3d_implicit_local)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_thread_dimensions_full_12_im"
         COMMAND "${TS_BUILDDIR}/thread_dimensions/test_thread_dimensions" ${EXTRA_ARGS} full_1d_implicit_local  full_2d_implicit_local)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_thread_dimensions_full_3_im"
         COMMAND "${TS_BUILDDIR}/thread_dimensions/test_thread_dimensions" ${EXTRA_ARGS} full_3d_implicit_local)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_thread_dimensions_full_12_ex"
         COMMAND "${TS_BUILDDIR}/thread_dimensions/test_thread_dimensions" ${EXTRA_ARGS} full_1d_explicit_local  full_2d_explicit_local)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_thread_dimensions_full_3_ex"
         COMMAND "${TS_BUILDDIR}/thread_dimensions/test_thread_dimensions" ${EXTRA_ARGS} full_3d_explicit_local)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_vecalign"
         COMMAND "${TS_BUILDDIR}/vectors/test_vectors" ${EXTRA_ARGS} vec_align_array vec_align_struct vec_align_packed_struct vec_align_struct_arr vec_align_packed_struct_arr )
add_test(NAME "${TS_NAME}${TS_SUFFIX}_vecstep"
         COMMAND "${TS_BUILDDIR}/vectors/test_vectors" ${EXTRA_ARGS} step_type step_var step_typedef_type step_typedef_var )


add_test(NAME "${TS_NAME}${TS_SUFFIX}_workgroup"
         COMMAND "${TS_BUILDDIR}/workgroups/test_workgroups"
        work_group_all
        work_group_any)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_workgroup_reduce"
         COMMAND "${TS_BUILDDIR}/workgroups/test_workgroups"
        work_group_reduce_add
        work_group_reduce_min
        work_group_reduce_max)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_workgroup_scan"
         COMMAND "${TS_BUILDDIR}/workgroups/test_workgroups"
        work_group_scan_inclusive_add
        work_group_scan_inclusive_min
        work_group_scan_inclusive_max
        work_group_scan_exclusive_add
        work_group_scan_exclusive_min
        work_group_scan_exclusive_max)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_workgroup_broadcast"
         COMMAND "${TS_BUILDDIR}/workgroups/test_workgroups"
        work_group_broadcast_1D
        work_group_broadcast_2D
        work_group_broadcast_3D)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_workgroup_lsize"
         COMMAND "${TS_BUILDDIR}/workgroups/test_workgroups"
       work_group_suggested_local_size_1D
        work_group_suggested_local_size_2D
        work_group_suggested_local_size_3D)

set(OCL_30_TESTS_FULL
    "${TS_NAME}${TS_SUFFIX}_api_ocl_30"
    "${TS_NAME}${TS_SUFFIX}_basic_other_ocl_30"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_init_store_load_exchange"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_add_sub_min_max"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_flag_fence"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_or_xor_and"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_svm_init_store_load_exchange"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_svm_add_sub_min_max"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_svm_flag_fence"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_svm_or_xor_and"
    "${TS_NAME}${TS_SUFFIX}_device_execution"
    "${TS_NAME}${TS_SUFFIX}_device_timer"
    "${TS_NAME}${TS_SUFFIX}_generic_address_space"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_ocl_30"
    "${TS_NAME}${TS_SUFFIX}_non_uniform_work_group"
    "${TS_NAME}${TS_SUFFIX}_pipes"
    "${TS_NAME}${TS_SUFFIX}_select_float_uint"
    "${TS_NAME}${TS_SUFFIX}_select_float_int"
    "${TS_NAME}${TS_SUFFIX}_subgroups_1"
    "${TS_NAME}${TS_SUFFIX}_subgroups_2"
    "${TS_NAME}${TS_SUFFIX}_SVM_1"
    "${TS_NAME}${TS_SUFFIX}_SVM_2"
    "${TS_NAME}${TS_SUFFIX}_workgroup"
    "${TS_NAME}${TS_SUFFIX}_workgroup_reduce"
    "${TS_NAME}${TS_SUFFIX}_workgroup_scan"
    "${TS_NAME}${TS_SUFFIX}_workgroup_broadcast"
    "${TS_NAME}${TS_SUFFIX}_workgroup_lsize"
)

if(HAVE_SPIRV_TOOLS)
  list(APPEND OCL_30_TESTS_FULL
    "${TS_NAME}${TS_SUFFIX}_spirv_new_0"
    "${TS_NAME}${TS_SUFFIX}_spirv_new_1"
    "${TS_NAME}${TS_SUFFIX}_spirv_new_2"
    "${TS_NAME}${TS_SUFFIX}_spirv_new_3"
    "${TS_NAME}${TS_SUFFIX}_spirv_new_4"
    "${TS_NAME}${TS_SUFFIX}_spirv_new_5"
    "${TS_NAME}${TS_SUFFIX}_spirv_new_6"
    "${TS_NAME}${TS_SUFFIX}_spirv_new_7"
    "${TS_NAME}${TS_SUFFIX}_spirv_new_8"
    "${TS_NAME}${TS_SUFFIX}_spirv_new_9"
    "${TS_NAME}${TS_SUFFIX}_spirv_new_half")
endif()

set(CTS_FULL_LIST
    "${TS_NAME}${TS_SUFFIX}_allocations_buffer"
    "${TS_NAME}${TS_SUFFIX}_allocations_image"
    "${TS_NAME}${TS_SUFFIX}_api"
    "${TS_NAME}${TS_SUFFIX}_atomics"
    "${TS_NAME}${TS_SUFFIX}_basic_math"
    "${TS_NAME}${TS_SUFFIX}_basic_s2v"
    "${TS_NAME}${TS_SUFFIX}_basic_memory"
    "${TS_NAME}${TS_SUFFIX}_basic_image"
    "${TS_NAME}${TS_SUFFIX}_basic_other"
    "${TS_NAME}${TS_SUFFIX}_buffers_read_async"
    "${TS_NAME}${TS_SUFFIX}_buffers_write_async"
    "${TS_NAME}${TS_SUFFIX}_buffers_read_array_barrier"
    "${TS_NAME}${TS_SUFFIX}_buffers_read"
    "${TS_NAME}${TS_SUFFIX}_buffers_write"
    "${TS_NAME}${TS_SUFFIX}_buffers_fill"
    "${TS_NAME}${TS_SUFFIX}_buffers_var1"
    "${TS_NAME}${TS_SUFFIX}_buffers_var2"
    "${TS_NAME}${TS_SUFFIX}_buffers_map_read"
    "${TS_NAME}${TS_SUFFIX}_buffers_map_write"
    "${TS_NAME}${TS_SUFFIX}_commonfns"
    "${TS_NAME}${TS_SUFFIX}_compiler"
    "${TS_NAME}${TS_SUFFIX}_computeinfo"
    "${TS_NAME}${TS_SUFFIX}_contractions"
    "${TS_NAME}${TS_SUFFIX}_conversion_uchar_uchar"
    "${TS_NAME}${TS_SUFFIX}_conversion_char_uchar"
    "${TS_NAME}${TS_SUFFIX}_conversion_ushort_uchar"
    "${TS_NAME}${TS_SUFFIX}_conversion_short_uchar"
    "${TS_NAME}${TS_SUFFIX}_conversion_uint_uchar"
    "${TS_NAME}${TS_SUFFIX}_conversion_int_uchar"
    "${TS_NAME}${TS_SUFFIX}_conversion_float_uchar"
    "${TS_NAME}${TS_SUFFIX}_conversion_double_uchar"
    "${TS_NAME}${TS_SUFFIX}_conversion_ulong_uchar"
    "${TS_NAME}${TS_SUFFIX}_conversion_long_uchar"
    "${TS_NAME}${TS_SUFFIX}_conversion_uchar_char"
    "${TS_NAME}${TS_SUFFIX}_conversion_char_char"
    "${TS_NAME}${TS_SUFFIX}_conversion_ushort_char"
    "${TS_NAME}${TS_SUFFIX}_conversion_short_char"
    "${TS_NAME}${TS_SUFFIX}_conversion_uint_char"
    "${TS_NAME}${TS_SUFFIX}_conversion_int_char"
    "${TS_NAME}${TS_SUFFIX}_conversion_float_char"
    "${TS_NAME}${TS_SUFFIX}_conversion_double_char"
    "${TS_NAME}${TS_SUFFIX}_conversion_ulong_char"
    "${TS_NAME}${TS_SUFFIX}_conversion_long_char"
    "${TS_NAME}${TS_SUFFIX}_conversion_uchar_ushort"
    "${TS_NAME}${TS_SUFFIX}_conversion_char_ushort"
    "${TS_NAME}${TS_SUFFIX}_conversion_ushort_ushort"
    "${TS_NAME}${TS_SUFFIX}_conversion_short_ushort"
    "${TS_NAME}${TS_SUFFIX}_conversion_uint_ushort"
    "${TS_NAME}${TS_SUFFIX}_conversion_int_ushort"
    "${TS_NAME}${TS_SUFFIX}_conversion_float_ushort"
    "${TS_NAME}${TS_SUFFIX}_conversion_double_ushort"
    "${TS_NAME}${TS_SUFFIX}_conversion_ulong_ushort"
    "${TS_NAME}${TS_SUFFIX}_conversion_long_ushort"
    "${TS_NAME}${TS_SUFFIX}_conversion_uchar_short"
    "${TS_NAME}${TS_SUFFIX}_conversion_char_short"
    "${TS_NAME}${TS_SUFFIX}_conversion_ushort_short"
    "${TS_NAME}${TS_SUFFIX}_conversion_short_short"
    "${TS_NAME}${TS_SUFFIX}_conversion_uint_short"
    "${TS_NAME}${TS_SUFFIX}_conversion_int_short"
    "${TS_NAME}${TS_SUFFIX}_conversion_float_short"
    "${TS_NAME}${TS_SUFFIX}_conversion_double_short"
    "${TS_NAME}${TS_SUFFIX}_conversion_ulong_short"
    "${TS_NAME}${TS_SUFFIX}_conversion_long_short"
    "${TS_NAME}${TS_SUFFIX}_conversion_uchar_uint"
    "${TS_NAME}${TS_SUFFIX}_conversion_char_uint"
    "${TS_NAME}${TS_SUFFIX}_conversion_ushort_uint"
    "${TS_NAME}${TS_SUFFIX}_conversion_short_uint"
    "${TS_NAME}${TS_SUFFIX}_conversion_uint_uint"
    "${TS_NAME}${TS_SUFFIX}_conversion_int_uint"
    "${TS_NAME}${TS_SUFFIX}_conversion_float_uint"
    "${TS_NAME}${TS_SUFFIX}_conversion_double_uint"
    "${TS_NAME}${TS_SUFFIX}_conversion_ulong_uint"
    "${TS_NAME}${TS_SUFFIX}_conversion_long_uint"
    "${TS_NAME}${TS_SUFFIX}_conversion_uchar_int"
    "${TS_NAME}${TS_SUFFIX}_conversion_char_int"
    "${TS_NAME}${TS_SUFFIX}_conversion_ushort_int"
    "${TS_NAME}${TS_SUFFIX}_conversion_short_int"
    "${TS_NAME}${TS_SUFFIX}_conversion_uint_int"
    "${TS_NAME}${TS_SUFFIX}_conversion_int_int"
    "${TS_NAME}${TS_SUFFIX}_conversion_float_int"
    "${TS_NAME}${TS_SUFFIX}_conversion_double_int"
    "${TS_NAME}${TS_SUFFIX}_conversion_ulong_int"
    "${TS_NAME}${TS_SUFFIX}_conversion_long_int"
    "${TS_NAME}${TS_SUFFIX}_conversion_uchar_float"
    "${TS_NAME}${TS_SUFFIX}_conversion_char_float"
    "${TS_NAME}${TS_SUFFIX}_conversion_ushort_float"
    "${TS_NAME}${TS_SUFFIX}_conversion_short_float"
    "${TS_NAME}${TS_SUFFIX}_conversion_uint_float"
    "${TS_NAME}${TS_SUFFIX}_conversion_int_float"
    "${TS_NAME}${TS_SUFFIX}_conversion_float_float"
    "${TS_NAME}${TS_SUFFIX}_conversion_double_float"
    "${TS_NAME}${TS_SUFFIX}_conversion_ulong_float"
    "${TS_NAME}${TS_SUFFIX}_conversion_long_float"
    "${TS_NAME}${TS_SUFFIX}_conversion_uchar_double"
    "${TS_NAME}${TS_SUFFIX}_conversion_char_double"
    "${TS_NAME}${TS_SUFFIX}_conversion_ushort_double"
    "${TS_NAME}${TS_SUFFIX}_conversion_short_double"
    "${TS_NAME}${TS_SUFFIX}_conversion_uint_double"
    "${TS_NAME}${TS_SUFFIX}_conversion_int_double"
    "${TS_NAME}${TS_SUFFIX}_conversion_float_double"
    "${TS_NAME}${TS_SUFFIX}_conversion_double_double"
    "${TS_NAME}${TS_SUFFIX}_conversion_ulong_double"
    "${TS_NAME}${TS_SUFFIX}_conversion_long_double"
    "${TS_NAME}${TS_SUFFIX}_conversion_uchar_ulong"
    "${TS_NAME}${TS_SUFFIX}_conversion_char_ulong"
    "${TS_NAME}${TS_SUFFIX}_conversion_ushort_ulong"
    "${TS_NAME}${TS_SUFFIX}_conversion_short_ulong"
    "${TS_NAME}${TS_SUFFIX}_conversion_uint_ulong"
    "${TS_NAME}${TS_SUFFIX}_conversion_int_ulong"
    "${TS_NAME}${TS_SUFFIX}_conversion_float_ulong"
    "${TS_NAME}${TS_SUFFIX}_conversion_double_ulong"
    "${TS_NAME}${TS_SUFFIX}_conversion_ulong_ulong"
    "${TS_NAME}${TS_SUFFIX}_conversion_long_ulong"
    "${TS_NAME}${TS_SUFFIX}_conversion_uchar_long"
    "${TS_NAME}${TS_SUFFIX}_conversion_char_long"
    "${TS_NAME}${TS_SUFFIX}_conversion_ushort_long"
    "${TS_NAME}${TS_SUFFIX}_conversion_short_long"
    "${TS_NAME}${TS_SUFFIX}_conversion_uint_long"
    "${TS_NAME}${TS_SUFFIX}_conversion_int_long"
    "${TS_NAME}${TS_SUFFIX}_conversion_float_long"
    "${TS_NAME}${TS_SUFFIX}_conversion_double_long"
    "${TS_NAME}${TS_SUFFIX}_conversion_ulong_long"
    "${TS_NAME}${TS_SUFFIX}_conversion_long_long"
    "${TS_NAME}${TS_SUFFIX}_device_partition"
    "${TS_NAME}${TS_SUFFIX}_events"
    "${TS_NAME}${TS_SUFFIX}_events_ooo"
    "${TS_NAME}${TS_SUFFIX}_events_other"
    "${TS_NAME}${TS_SUFFIX}_geometrics"
    ${OCL_FULL_TESTS_FP16}


    "${TS_NAME}${TS_SUFFIX}_images_cl_copy_images"
    "${TS_NAME}${TS_SUFFIX}_images_cl_copy_images_max"
    "${TS_NAME}${TS_SUFFIX}_images_cl_copy_images_min"
    "${TS_NAME}${TS_SUFFIX}_images_cl_fill_images"
    "${TS_NAME}${TS_SUFFIX}_images_cl_fill_images_min"
    "${TS_NAME}${TS_SUFFIX}_images_cl_fill_images_max"
    "${TS_NAME}${TS_SUFFIX}_images_cl_fill_images_pitch"
    "${TS_NAME}${TS_SUFFIX}_images_cl_get_info"
    "${TS_NAME}${TS_SUFFIX}_images_cl_get_info_min"
    "${TS_NAME}${TS_SUFFIX}_images_cl_get_info_max"
    "${TS_NAME}${TS_SUFFIX}_images_cl_get_info_random"
    "${TS_NAME}${TS_SUFFIX}_images_cl_read_write_images"
    "${TS_NAME}${TS_SUFFIX}_images_cl_read_write_images_min"
    "${TS_NAME}${TS_SUFFIX}_images_cl_read_write_images_max"
    "${TS_NAME}${TS_SUFFIX}_images_cl_read_write_images_pitch"
    "${TS_NAME}${TS_SUFFIX}_images_kernel_image_methods_1D"
    "${TS_NAME}${TS_SUFFIX}_images_kernel_image_methods_2D"
    "${TS_NAME}${TS_SUFFIX}_images_kernel_image_methods_3D"
    "${TS_NAME}${TS_SUFFIX}_images_kernel_image_methods_1Darray"
    "${TS_NAME}${TS_SUFFIX}_images_kernel_image_methods_2Darray"
    "${TS_NAME}${TS_SUFFIX}_images_image_streams_1D"
    "${TS_NAME}${TS_SUFFIX}_images_image_streams_2D"
    "${TS_NAME}${TS_SUFFIX}_images_image_streams_3D"
    "${TS_NAME}${TS_SUFFIX}_images_image_streams_1Darray"
    "${TS_NAME}${TS_SUFFIX}_images_image_streams_2Darray"
    "${TS_NAME}${TS_SUFFIX}_images_image_streams_1D_pitch"
    "${TS_NAME}${TS_SUFFIX}_images_image_streams_2D_pitch"
    "${TS_NAME}${TS_SUFFIX}_images_image_streams_3D_pitch"
    "${TS_NAME}${TS_SUFFIX}_images_image_streams_1Darray_pitch"
    "${TS_NAME}${TS_SUFFIX}_images_image_streams_2Darray_pitch"
    "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_1D"
    "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_2D"
    "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_3D"
    "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_1Darray"
    "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_2Darray"
    "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_1D_pitch"
    "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_2D_pitch"
    "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_3D_pitch"
    "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_1Darray_pitch"
    "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_2Darray_pitch"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_basic"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_unary"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_i24"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_long"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_ulong"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_int"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_uint"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_short"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_ushort"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_char"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_uchar"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_other"

    "${TS_NAME}${TS_SUFFIX}_math_add"
    "${TS_NAME}${TS_SUFFIX}_math_assignment"
    "${TS_NAME}${TS_SUFFIX}_math_cbrt"
    "${TS_NAME}${TS_SUFFIX}_math_ceil"
    "${TS_NAME}${TS_SUFFIX}_math_copysign"
    "${TS_NAME}${TS_SUFFIX}_math_cos"
    "${TS_NAME}${TS_SUFFIX}_math_cosh"
    "${TS_NAME}${TS_SUFFIX}_math_cospi"
    "${TS_NAME}${TS_SUFFIX}_math_divide"
    "${TS_NAME}${TS_SUFFIX}_math_exp"
    "${TS_NAME}${TS_SUFFIX}_math_exp10"
    "${TS_NAME}${TS_SUFFIX}_math_exp2"
    "${TS_NAME}${TS_SUFFIX}_math_expm1"
    "${TS_NAME}${TS_SUFFIX}_math_fabs"
    "${TS_NAME}${TS_SUFFIX}_math_fdim"
    "${TS_NAME}${TS_SUFFIX}_math_floor"
    "${TS_NAME}${TS_SUFFIX}_math_fma"
    "${TS_NAME}${TS_SUFFIX}_math_fmax"
    "${TS_NAME}${TS_SUFFIX}_math_fmin"
    "${TS_NAME}${TS_SUFFIX}_math_fmod"
    "${TS_NAME}${TS_SUFFIX}_math_fract"
    "${TS_NAME}${TS_SUFFIX}_math_frexp"
    "${TS_NAME}${TS_SUFFIX}_math_hypot"
    "${TS_NAME}${TS_SUFFIX}_math_ilogb"
    "${TS_NAME}${TS_SUFFIX}_math_isequal"
    "${TS_NAME}${TS_SUFFIX}_math_isfinite"
    "${TS_NAME}${TS_SUFFIX}_math_isgreater"
    "${TS_NAME}${TS_SUFFIX}_math_isgreaterequal"
    "${TS_NAME}${TS_SUFFIX}_math_isinf"
    "${TS_NAME}${TS_SUFFIX}_math_isless"
    "${TS_NAME}${TS_SUFFIX}_math_islessequal"
    "${TS_NAME}${TS_SUFFIX}_math_islessgreater"
    "${TS_NAME}${TS_SUFFIX}_math_isnan"
    "${TS_NAME}${TS_SUFFIX}_math_isnormal"
    "${TS_NAME}${TS_SUFFIX}_math_isnotequal"
    "${TS_NAME}${TS_SUFFIX}_math_isordered"
    "${TS_NAME}${TS_SUFFIX}_math_isunordered"
    "${TS_NAME}${TS_SUFFIX}_math_ldexp"
    "${TS_NAME}${TS_SUFFIX}_math_log"
    "${TS_NAME}${TS_SUFFIX}_math_log10"
    "${TS_NAME}${TS_SUFFIX}_math_log1p"
    "${TS_NAME}${TS_SUFFIX}_math_log2"
    "${TS_NAME}${TS_SUFFIX}_math_logb"
    "${TS_NAME}${TS_SUFFIX}_math_mad"
    "${TS_NAME}${TS_SUFFIX}_math_maxmag"
    "${TS_NAME}${TS_SUFFIX}_math_minmag"
    "${TS_NAME}${TS_SUFFIX}_math_modf"
    "${TS_NAME}${TS_SUFFIX}_math_multiply"
    "${TS_NAME}${TS_SUFFIX}_math_nan"
    "${TS_NAME}${TS_SUFFIX}_math_nextafter"
    "${TS_NAME}${TS_SUFFIX}_math_not"
    "${TS_NAME}${TS_SUFFIX}_math_pow"
    "${TS_NAME}${TS_SUFFIX}_math_pown"
    "${TS_NAME}${TS_SUFFIX}_math_powr"
    "${TS_NAME}${TS_SUFFIX}_math_remquo"
    "${TS_NAME}${TS_SUFFIX}_math_remainder"
    "${TS_NAME}${TS_SUFFIX}_math_rint"
    "${TS_NAME}${TS_SUFFIX}_math_rootn"
    "${TS_NAME}${TS_SUFFIX}_math_round"
    "${TS_NAME}${TS_SUFFIX}_math_rsqrt"
    "${TS_NAME}${TS_SUFFIX}_math_signbit"
    "${TS_NAME}${TS_SUFFIX}_math_sin"
    "${TS_NAME}${TS_SUFFIX}_math_sincos"
    "${TS_NAME}${TS_SUFFIX}_math_sinh"
    "${TS_NAME}${TS_SUFFIX}_math_sinpi"
    "${TS_NAME}${TS_SUFFIX}_math_sqrt"
    "${TS_NAME}${TS_SUFFIX}_math_subtract"
    "${TS_NAME}${TS_SUFFIX}_math_tan"
    "${TS_NAME}${TS_SUFFIX}_math_tanh"
    "${TS_NAME}${TS_SUFFIX}_math_tanpi"
    "${TS_NAME}${TS_SUFFIX}_math_trunc"

    "${TS_NAME}${TS_SUFFIX}_mem_host_flags"
    "${TS_NAME}${TS_SUFFIX}_multiples"
    "${TS_NAME}${TS_SUFFIX}_printf"
    "${TS_NAME}${TS_SUFFIX}_profiling"
    "${TS_NAME}${TS_SUFFIX}_relationals"
    "${TS_NAME}${TS_SUFFIX}_relationals_shuffle"
    "${TS_NAME}${TS_SUFFIX}_select_uchar_uchar"
    "${TS_NAME}${TS_SUFFIX}_select_uchar_char"
    "${TS_NAME}${TS_SUFFIX}_select_char_uchar"
    "${TS_NAME}${TS_SUFFIX}_select_char_char"
    "${TS_NAME}${TS_SUFFIX}_select_ushort_ushort"
    "${TS_NAME}${TS_SUFFIX}_select_ushort_short"
    "${TS_NAME}${TS_SUFFIX}_select_short_ushort"
    "${TS_NAME}${TS_SUFFIX}_select_short_short"
    "${TS_NAME}${TS_SUFFIX}_select_uint_uint"
    "${TS_NAME}${TS_SUFFIX}_select_uint_int"
    "${TS_NAME}${TS_SUFFIX}_select_int_uint"
    "${TS_NAME}${TS_SUFFIX}_select_int_int"
    "${TS_NAME}${TS_SUFFIX}_select_ulong_ulong"
    "${TS_NAME}${TS_SUFFIX}_select_ulong_long"
    "${TS_NAME}${TS_SUFFIX}_select_long_ulong"
    "${TS_NAME}${TS_SUFFIX}_select_long_long"
    "${TS_NAME}${TS_SUFFIX}_thread_dimensions_quick"
    "${TS_NAME}${TS_SUFFIX}_thread_dimensions_full_12_im"
    "${TS_NAME}${TS_SUFFIX}_thread_dimensions_full_12_ex"
    "${TS_NAME}${TS_SUFFIX}_thread_dimensions_full_3_im"
    "${TS_NAME}${TS_SUFFIX}_thread_dimensions_full_3_ex"
    "${TS_NAME}${TS_SUFFIX}_vecalign"
    "${TS_NAME}${TS_SUFFIX}_vecstep"
    ${OCL_30_TESTS_FULL})

foreach(CTSTEST IN LISTS CTS_FULL_LIST)
  set_property(TEST ${CTSTEST}
    APPEND PROPERTY
    LABELS ${CTS_LABEL})
  set_property(TEST ${CTSTEST}
    PROPERTY
    ENVIRONMENT "POCL_MEMORY_LIMIT=1")
endforeach()

set_property(TEST ${OCL_30_TESTS_FULL}
             APPEND PROPERTY LABELS "conformance_30_only")
set_property(TEST "${TS_NAME}${TS_SUFFIX}_non_uniform_work_group"
             PROPERTY ENVIRONMENT "POCL_MEMORY_LIMIT=4")
endfunction()

add_cts_full_tests("conformance_suite_full_main" "_main")

if(HAVE_SPIRV_TOOLS)
set(CACHE_DIR "${CMAKE_BINARY_DIR}/CTS_FULL_CACHE")
file(MAKE_DIRECTORY "${CACHE_DIR}")

add_cts_full_tests("conformance_suite_full_spirv" "_spirv"
                   "--compilation-mode" "spir-v"
                   "--compilation-program" "${CMAKE_BINARY_DIR}/cl_offline_compiler.sh"
                   "--compilation-cache-path" "${CACHE_DIR}")
endif()

#################################################################################
#################################################################################
########################   MINI   ###############################################
#################################################################################
#################################################################################

# conversions, for short version of conformance suite.
# disables multithreading and only tests vectors of size 4

function(add_cts_mini_tests CTS_LABEL TS_SUFFIX)
set(EXTRA_ARGS ${ARGN})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_mini_uint_uchar"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -4 uint_uchar uint_sat_uchar )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_mini_short_char"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -4 short_char short_sat_char )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_mini_uchar_ushort"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -4 uchar_ushort uchar_sat_ushort )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_mini_char_int"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -4 char_int char_sat_int )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_mini_float_char"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -4 float_char float_rte_char float_rtp_char float_rtn_char float_rtz_char )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_mini_float_ushort"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -4 float_ushort float_rte_ushort float_rtp_ushort float_rtn_ushort float_rtz_ushort )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_mini_int_float"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -4 int_float int_sat_float int_rte_float int_sat_rte_float int_rtp_float int_sat_rtp_float int_rtn_float int_sat_rtn_float int_rtz_float int_sat_rtz_float )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_mini_double_short"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -4 double_short double_rte_short double_rtp_short double_rtn_short double_rtz_short )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_mini_double_ulong"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -4 double_ulong double_rte_ulong double_rtp_ulong double_rtn_ulong double_rtz_ulong )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_mini_long_double"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -4 long_double long_sat_double long_rte_double long_sat_rte_double long_rtp_double long_sat_rtp_double long_rtn_double long_sat_rtn_double long_rtz_double long_sat_rtz_double )

#***************************************************************************

# tests math on vectors of 4 floats.
# This is usually the fastest version on machines with AVX(2)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_add"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d add)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_assignment"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d assignment)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_cbrt"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d cbrt)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_ceil"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d ceil)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_copysign"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d copysign)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_cos"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d cos)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_cosh"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d cosh)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_cospi"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d cospi)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_divide"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d divide)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_exp"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d exp)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_expm1"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d expm1)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_fabs"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d fabs)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_fdim"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d fdim)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_floor"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d floor)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_fma"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d fma)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_fmax"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d fmax)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_fmin"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d fmin)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_fract"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d fract)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_frexp"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d frexp)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_hypot"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d hypot)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_ilogb"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d ilogb)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_isequal"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d isequal)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_isfinite"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d isfinite)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_isgreater"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d isgreater)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_isgreaterequal"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d isgreaterequal)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_isinf"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d isinf)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_isless"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d isless)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_islessequal"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d islessequal)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_islessgreater"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d islessgreater)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_isnan"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d isnan)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_isnormal"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d isnormal)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_isnotequal"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d isnotequal)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_isordered"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d isordered)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_isunordered"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d isunordered)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_ldexp"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d ldexp)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_log"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d log)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_log10"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d log10)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_log1p"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d log1p)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_log2"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d log2)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_logb"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d logb)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_mad"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d mad)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_maxmag"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d maxmag)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_minmag"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d minmag)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_modf"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d modf)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_multiply"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d multiply)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_nan"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d nan)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_nextafter"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d nextafter)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_not"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d not)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_pown"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d pown)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_rint"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d rint)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_rootn"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d rootn)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_round"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d round)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_rsqrt"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d rsqrt)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_signbit"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d signbit)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_sin"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d sin)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_sinh"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d sinh)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_sinpi"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d sinpi)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_sqrt"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d sqrt)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_subtract"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d subtract)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_tan"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d tan)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_tanh"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d tanh)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_tanpi"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d tanpi)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_mini_trunc"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -d trunc)

set(OCL_30_TESTS_MINI
    "${TS_NAME}${TS_SUFFIX}_api_ocl_30"
    "${TS_NAME}${TS_SUFFIX}_basic_other_ocl_30"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_1"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_2"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_3"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_4"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_5"
    "${TS_NAME}${TS_SUFFIX}_device_execution"
    "${TS_NAME}${TS_SUFFIX}_device_timer"
    "${TS_NAME}${TS_SUFFIX}_generic_address_space"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_ocl_30"
    "${TS_NAME}${TS_SUFFIX}_non_uniform_work_group"
    "${TS_NAME}${TS_SUFFIX}_pipes"
    "${TS_NAME}${TS_SUFFIX}_subgroups_1"
    "${TS_NAME}${TS_SUFFIX}_SVM_1"
    "${TS_NAME}${TS_SUFFIX}_SVM_2"
)


if(HAVE_SPIRV_TOOLS)
  list(APPEND OCL_30_TESTS_MINI
    "${TS_NAME}${TS_SUFFIX}_spirv_new_micro_0"
    "${TS_NAME}${TS_SUFFIX}_spirv_new_micro_1"
    "${TS_NAME}${TS_SUFFIX}_spirv_new_micro_2")
endif()


set(CTS_MINI_LIST
    "${TS_NAME}${TS_SUFFIX}_allocations_buffer"
    "${TS_NAME}${TS_SUFFIX}_allocations_image"
    "${TS_NAME}${TS_SUFFIX}_api"
    "${TS_NAME}${TS_SUFFIX}_atomics"
    "${TS_NAME}${TS_SUFFIX}_basic_math"
    "${TS_NAME}${TS_SUFFIX}_basic_s2v"
    "${TS_NAME}${TS_SUFFIX}_basic_memory"
    "${TS_NAME}${TS_SUFFIX}_basic_image"
    "${TS_NAME}${TS_SUFFIX}_basic_other"
    "${TS_NAME}${TS_SUFFIX}_buffers_read"
    "${TS_NAME}${TS_SUFFIX}_buffers_write"
    "${TS_NAME}${TS_SUFFIX}_buffers_fill"
    "${TS_NAME}${TS_SUFFIX}_buffers_var1"
    "${TS_NAME}${TS_SUFFIX}_buffers_var2"
    "${TS_NAME}${TS_SUFFIX}_buffers_map_read"
    "${TS_NAME}${TS_SUFFIX}_buffers_map_write"
    "${TS_NAME}${TS_SUFFIX}_commonfns"
    "${TS_NAME}${TS_SUFFIX}_compiler"
    "${TS_NAME}${TS_SUFFIX}_computeinfo"
    "${TS_NAME}${TS_SUFFIX}_contractions"
    "${TS_NAME}${TS_SUFFIX}_device_partition"
    "${TS_NAME}${TS_SUFFIX}_events"
    "${TS_NAME}${TS_SUFFIX}_events_ooo"
    "${TS_NAME}${TS_SUFFIX}_events_other"
    "${TS_NAME}${TS_SUFFIX}_geometrics"
    "${TS_NAME}${TS_SUFFIX}_images_cl_copy_images"
    "${TS_NAME}${TS_SUFFIX}_images_cl_fill_images"
    "${TS_NAME}${TS_SUFFIX}_images_cl_get_info"
    "${TS_NAME}${TS_SUFFIX}_images_cl_read_write_images"
    "${TS_NAME}${TS_SUFFIX}_images_kernel_image_methods_1D"
    "${TS_NAME}${TS_SUFFIX}_images_kernel_image_methods_2D"
    "${TS_NAME}${TS_SUFFIX}_images_kernel_image_methods_1Darray"
    "${TS_NAME}${TS_SUFFIX}_images_image_streams_1D"
    "${TS_NAME}${TS_SUFFIX}_images_image_streams_2D"
    "${TS_NAME}${TS_SUFFIX}_images_image_streams_1Darray"
    "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_1D"
    "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_2D"
    "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_1Darray"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_basic"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_unary"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_i24"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_long"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_ulong"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_int"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_uint"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_short"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_ushort"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_char"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_uchar"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_other"
    "${TS_NAME}${TS_SUFFIX}_mem_host_flags"
    "${TS_NAME}${TS_SUFFIX}_multiples"
    "${TS_NAME}${TS_SUFFIX}_printf"
    "${TS_NAME}${TS_SUFFIX}_profiling"
    "${TS_NAME}${TS_SUFFIX}_relationals"
    "${TS_NAME}${TS_SUFFIX}_relationals_shuffle"
    "${TS_NAME}${TS_SUFFIX}_select_char_char"
    "${TS_NAME}${TS_SUFFIX}_select_short_short"
    "${TS_NAME}${TS_SUFFIX}_select_int_int"
    "${TS_NAME}${TS_SUFFIX}_select_long_long"
    "${TS_NAME}${TS_SUFFIX}_thread_dimensions_quick"
    "${TS_NAME}${TS_SUFFIX}_thread_dimensions_full_12_im"
    "${TS_NAME}${TS_SUFFIX}_thread_dimensions_full_12_ex"
    "${TS_NAME}${TS_SUFFIX}_thread_dimensions_full_3_im"
    "${TS_NAME}${TS_SUFFIX}_thread_dimensions_full_3_ex"
    "${TS_NAME}${TS_SUFFIX}_vecstep"
    ${OCL_MINI_TESTS_FP16}
    ${OCL_30_TESTS_MINI}

    "${TS_NAME}${TS_SUFFIX}_math_mini_add"
    "${TS_NAME}${TS_SUFFIX}_math_mini_assignment"
    "${TS_NAME}${TS_SUFFIX}_math_mini_cbrt"
    "${TS_NAME}${TS_SUFFIX}_math_mini_ceil"
    "${TS_NAME}${TS_SUFFIX}_math_mini_copysign"
    "${TS_NAME}${TS_SUFFIX}_math_mini_cos"
    "${TS_NAME}${TS_SUFFIX}_math_mini_cosh"
    "${TS_NAME}${TS_SUFFIX}_math_mini_cospi"
    "${TS_NAME}${TS_SUFFIX}_math_mini_divide"
    "${TS_NAME}${TS_SUFFIX}_math_mini_exp"
    "${TS_NAME}${TS_SUFFIX}_math_mini_expm1"
    "${TS_NAME}${TS_SUFFIX}_math_mini_fabs"
    "${TS_NAME}${TS_SUFFIX}_math_mini_fdim"
    "${TS_NAME}${TS_SUFFIX}_math_mini_floor"
    "${TS_NAME}${TS_SUFFIX}_math_mini_fma"
    "${TS_NAME}${TS_SUFFIX}_math_mini_fmax"
    "${TS_NAME}${TS_SUFFIX}_math_mini_fmin"
    "${TS_NAME}${TS_SUFFIX}_math_mini_fract"
    "${TS_NAME}${TS_SUFFIX}_math_mini_frexp"
    "${TS_NAME}${TS_SUFFIX}_math_mini_hypot"
    "${TS_NAME}${TS_SUFFIX}_math_mini_ilogb"
    "${TS_NAME}${TS_SUFFIX}_math_mini_isequal"
    "${TS_NAME}${TS_SUFFIX}_math_mini_isfinite"
    "${TS_NAME}${TS_SUFFIX}_math_mini_isgreater"
    "${TS_NAME}${TS_SUFFIX}_math_mini_isgreaterequal"
    "${TS_NAME}${TS_SUFFIX}_math_mini_isinf"
    "${TS_NAME}${TS_SUFFIX}_math_mini_isless"
    "${TS_NAME}${TS_SUFFIX}_math_mini_islessequal"
    "${TS_NAME}${TS_SUFFIX}_math_mini_islessgreater"
    "${TS_NAME}${TS_SUFFIX}_math_mini_isnan"
    "${TS_NAME}${TS_SUFFIX}_math_mini_isnormal"
    "${TS_NAME}${TS_SUFFIX}_math_mini_isnotequal"
    "${TS_NAME}${TS_SUFFIX}_math_mini_isordered"
    "${TS_NAME}${TS_SUFFIX}_math_mini_isunordered"
    "${TS_NAME}${TS_SUFFIX}_math_mini_ldexp"
    "${TS_NAME}${TS_SUFFIX}_math_mini_log"
    "${TS_NAME}${TS_SUFFIX}_math_mini_log10"
    "${TS_NAME}${TS_SUFFIX}_math_mini_log1p"
    "${TS_NAME}${TS_SUFFIX}_math_mini_log2"
    "${TS_NAME}${TS_SUFFIX}_math_mini_logb"
    "${TS_NAME}${TS_SUFFIX}_math_mini_mad"
    "${TS_NAME}${TS_SUFFIX}_math_mini_maxmag"
    "${TS_NAME}${TS_SUFFIX}_math_mini_minmag"
    "${TS_NAME}${TS_SUFFIX}_math_mini_modf"
    "${TS_NAME}${TS_SUFFIX}_math_mini_multiply"
    "${TS_NAME}${TS_SUFFIX}_math_mini_nan"
    "${TS_NAME}${TS_SUFFIX}_math_mini_nextafter"
    "${TS_NAME}${TS_SUFFIX}_math_mini_not"
    "${TS_NAME}${TS_SUFFIX}_math_mini_pown"
    "${TS_NAME}${TS_SUFFIX}_math_mini_rint"
    "${TS_NAME}${TS_SUFFIX}_math_mini_rootn"
    "${TS_NAME}${TS_SUFFIX}_math_mini_round"
    "${TS_NAME}${TS_SUFFIX}_math_mini_rsqrt"
    "${TS_NAME}${TS_SUFFIX}_math_mini_signbit"
    "${TS_NAME}${TS_SUFFIX}_math_mini_sin"
    "${TS_NAME}${TS_SUFFIX}_math_mini_sinh"
    "${TS_NAME}${TS_SUFFIX}_math_mini_sinpi"
    "${TS_NAME}${TS_SUFFIX}_math_mini_sqrt"
    "${TS_NAME}${TS_SUFFIX}_math_mini_subtract"
    "${TS_NAME}${TS_SUFFIX}_math_mini_tan"
    "${TS_NAME}${TS_SUFFIX}_math_mini_tanh"
    "${TS_NAME}${TS_SUFFIX}_math_mini_tanpi"
    "${TS_NAME}${TS_SUFFIX}_math_mini_trunc"

    "${TS_NAME}${TS_SUFFIX}_conversion_mini_uint_uchar"
    "${TS_NAME}${TS_SUFFIX}_conversion_mini_short_char"
    "${TS_NAME}${TS_SUFFIX}_conversion_mini_uchar_ushort"
    "${TS_NAME}${TS_SUFFIX}_conversion_mini_char_int"
    "${TS_NAME}${TS_SUFFIX}_conversion_mini_float_char"
    "${TS_NAME}${TS_SUFFIX}_conversion_mini_float_ushort"
    "${TS_NAME}${TS_SUFFIX}_conversion_mini_int_float"
    "${TS_NAME}${TS_SUFFIX}_conversion_mini_double_short"
    "${TS_NAME}${TS_SUFFIX}_conversion_mini_double_ulong"
    "${TS_NAME}${TS_SUFFIX}_conversion_mini_long_double")

foreach(CTSTEST IN LISTS CTS_MINI_LIST)
  set_property(TEST ${CTSTEST}
    APPEND PROPERTY
    LABELS ${CTS_LABEL})
  set_property(TEST ${CTSTEST}
    PROPERTY
    ENVIRONMENT "POCL_MEMORY_LIMIT=1")
endforeach()

set_property(TEST "${TS_NAME}${TS_SUFFIX}_non_uniform_work_group"
             PROPERTY ENVIRONMENT "POCL_MEMORY_LIMIT=4")

endfunction()

add_cts_mini_tests("conformance_suite_mini_main" "_main")

if(HAVE_SPIRV_TOOLS)
set(CACHE_DIR "${CMAKE_BINARY_DIR}/CTS_MINI_CACHE")
file(MAKE_DIRECTORY "${CACHE_DIR}")

add_cts_mini_tests("conformance_suite_mini_spirv" "_spirv"
                   "--compilation-mode" "spir-v"
                   "--compilation-program" "${CMAKE_BINARY_DIR}/cl_offline_compiler.sh"
                   "--compilation-cache-path" "${CACHE_DIR}")
endif()

#################################################################################
#################################################################################
########################   MICRO   ##############################################
#################################################################################
#################################################################################

function(add_cts_micro_tests CTS_LABEL TS_SUFFIX)
set(EXTRA_ARGS ${ARGN})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_micro_kernel_image_methods_1D"
         COMMAND "${TS_BUILDDIR}/images/kernel_image_methods/test_kernel_image_methods" ${EXTRA_ARGS} 1D)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_micro_kernel_image_methods_2D"
         COMMAND "${TS_BUILDDIR}/images/kernel_image_methods/test_kernel_image_methods" ${EXTRA_ARGS} 2D)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_micro_kernel_image_methods_1Darray"
         COMMAND "${TS_BUILDDIR}/images/kernel_image_methods/test_kernel_image_methods" ${EXTRA_ARGS} 1Darray)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_micro_kernel_image_methods_2Darray"
         COMMAND "${TS_BUILDDIR}/images/kernel_image_methods/test_kernel_image_methods" ${EXTRA_ARGS} 2Darray)


add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_micro_image_streams_1D"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" ${EXTRA_ARGS} 1D CL_ARGB CL_SIGNED_INT8)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_micro_image_streams_2D"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" ${EXTRA_ARGS} 2D CL_A CL_UNSIGNED_INT8)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_micro_image_streams_3D"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" ${EXTRA_ARGS} 3D CL_RGBA CL_FLOAT)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_micro_image_streams_1Darray"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" ${EXTRA_ARGS} 1Darray CL_A CL_UNORM_INT8)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_micro_image_streams_2Darray"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" ${EXTRA_ARGS} 2Darray CL_RGBA CL_SNORM_INT16)


add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_micro_samplerless_reads_1D"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS} 1D CL_A CL_UNSIGNED_INT8)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_micro_samplerless_reads_2D"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS} 2D CL_RGBA CL_FLOAT)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_micro_samplerless_reads_3D"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS} 3D CL_A CL_UNORM_INT8)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_micro_samplerless_reads_1Darray"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS} 1Darray CL_RGBA CL_SNORM_INT16)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_micro_samplerless_reads_2Darray"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS} 2Darray CL_ARGB CL_SIGNED_INT8)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops_micro_1"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${EXTRA_ARGS} integer_clz  integer_hadd  integer_rhadd  integer_mul_hi  integer_rotate  integer_clamp  integer_mad_sat  integer_mad_hi  integer_min  integer_max  integer_addAssign  integer_subtractAssign  integer_multiplyAssign  integer_divideAssign  integer_moduloAssign)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops_micro_3"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${EXTRA_ARGS}  quick_long_math  quick_long_logic  quick_ulong_shift  quick_ulong_compare  quick_uint_math  quick_uint_logic  quick_int_shift  quick_int_compare  quick_short_math  quick_short_logic  quick_ushort_shift  quick_ushort_compare  quick_char_math  quick_char_logic  quick_uchar_shift  quick_uchar_compare)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_basic_micro_other"
         COMMAND "${TS_BUILDDIR}/basic/test_basic" ${EXTRA_ARGS} mri_one  mri_multiple  barrier  createkernelsinprogram  work_item_functions    kernel_call_kernel_function  host_numeric_constants  kernel_numeric_constants  kernel_limit_constants  kernel_preprocessor_macros  parameter_types  vector_creation  vec_type_hint  global_work_offsets  get_global_offset  hostptr    if  sizeof  loop  pointer_cast  local_arg_def  local_kernel_def  local_kernel_scope  constant  constant_source)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_basic_micro_memory"
         COMMAND "${TS_BUILDDIR}/basic/test_basic" ${EXTRA_ARGS} vload_global  vload_local  vstore_global  vstore_local   bufferreadwriterect  arrayreadwrite  arraycopy  enqueue_map_buffer   kernel_memory_alignment_local  kernel_memory_alignment_global)

# compiler_defines_for_extensions is DISABLED because of SPIR:
# FAIL: Extension cl_khr_spir is not in the list of approved Khronos Group extensions!
add_test(NAME "${TS_NAME}${TS_SUFFIX}_compiler_micro"
         COMMAND "${TS_BUILDDIR}/compiler/test_compiler" ${EXTRA_ARGS} load_program_source  load_multistring_source  load_two_kernel_source  load_null_terminated_source  load_null_terminated_multi_line_source  load_null_terminated_partial_multi_line_source  load_discreet_length_source  get_program_source  get_program_build_info  get_program_info   async_build  options_build_optimizations  options_build_macro  options_build_macro_existence  options_include_directory  options_denorm_cache  preprocessor_define_udef  preprocessor_include  preprocessor_line_error  preprocessor_pragma  image_macro  simple_compile_only  simple_static_compile_only  simple_extern_compile_only  simple_compile_with_callback  simple_embedded_header_compile  simple_link_only  two_file_regular_variable_access  two_file_regular_struct_access  two_file_regular_function_access  simple_link_with_callback  simple_embedded_header_link  execute_after_simple_compile_and_link  execute_after_simple_compile_and_link_no_device_info  execute_after_simple_compile_and_link_with_defines  execute_after_simple_compile_and_link_with_callbacks  execute_after_simple_library_with_link  execute_after_two_file_link  execute_after_embedded_header_link  execute_after_included_header_link  execute_after_serialize_reload_object  execute_after_serialize_reload_library  simple_library_only  simple_library_with_callback  simple_library_with_link  two_file_link     multiple_files_multiple_libraries  program_binary_type  compile_and_link_status_options_log
         WORKING_DIRECTORY "${TS_BASEDIR}/src/${TS_NAME}/test_conformance/compiler")

# with wimpy mode
add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_micro_char_char"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_char_char -w)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_micro_ushort_ushort"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_ushort_ushort -w)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_micro_int_int"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_int_int -w)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_select_micro_ulong_ulong"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${EXTRA_ARGS} select_ulong_ulong -w)

# conversions, for micro version of conformance suite.
# disables multithreading, enables Wimpy mode

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_micro_uint_uchar"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -w uint_uchar uint_sat_uchar )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_micro_short_char"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -w short_char short_sat_char )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_micro_uchar_ushort"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -w uchar_ushort uchar_sat_ushort )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_micro_char_int"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -w char_int char_sat_int )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_micro_float_char"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -w float_char float_rte_char float_rtp_char float_rtn_char float_rtz_char )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_micro_float_ushort"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -w float_ushort float_rte_ushort float_rtp_ushort float_rtn_ushort float_rtz_ushort )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_micro_int_float"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -w int_float int_sat_float int_rte_float int_sat_rte_float int_rtp_float int_sat_rtp_float int_rtn_float int_sat_rtn_float int_rtz_float int_sat_rtz_float )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_micro_double_short"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -w double_short double_rte_short double_rtp_short double_rtn_short double_rtz_short )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_micro_double_ulong"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -w double_ulong double_rte_ulong double_rtp_ulong double_rtn_ulong double_rtz_ulong )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversion_micro_long_double"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${EXTRA_ARGS} -m -w long_double long_sat_double long_rte_double long_sat_rte_double long_rtp_double long_sat_rtp_double long_rtn_double long_sat_rtn_double long_rtz_double long_sat_rtz_double )

# tests math on vectors of 4 floats for micro testsuite
# vectors of 4, wimpy mode

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_add"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w add)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_assignment"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w assignment)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_cbrt"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w cbrt)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_ceil"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w ceil)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_copysign"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w copysign)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_cos"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w cos)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_cosh"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w cosh)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_cospi"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w cospi)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_divide"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w divide)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_exp"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w exp)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_expm1"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w expm1)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_fabs"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w fabs)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_fdim"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w fdim)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_floor"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w floor)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_fma"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w fma)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_fmax"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w fmax)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_fmin"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w fmin)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_fract"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w fract)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_frexp"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w frexp)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_hypot"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w hypot)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_ilogb"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w ilogb)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_isequal"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w isequal)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_isfinite"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w isfinite)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_isgreater"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w isgreater)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_isgreaterequal"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w isgreaterequal)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_isinf"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w isinf)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_isless"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w isless)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_islessequal"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w islessequal)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_islessgreater"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w islessgreater)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_isnan"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w isnan)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_isnormal"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w isnormal)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_isnotequal"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w isnotequal)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_isordered"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w isordered)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_isunordered"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w isunordered)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_ldexp"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w ldexp)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_log"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w log)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_log10"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w log10)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_log1p"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w log1p)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_log2"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w log2)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_logb"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w logb)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_mad"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w mad)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_maxmag"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w maxmag)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_minmag"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w minmag)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_modf"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w modf)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_multiply"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w multiply)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_nan"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w nan)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_nextafter"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w nextafter)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_not"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w not)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_pown"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w pown)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_rint"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w rint)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_rootn"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w rootn)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_round"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w round)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_rsqrt"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w rsqrt)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_signbit"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w signbit)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_sin"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w sin)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_sinh"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w sinh)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_sinpi"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w sinpi)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_sqrt"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w sqrt)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_subtract"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w subtract)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_tan"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w tan)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_tanh"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w tanh)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_tanpi"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w tanpi)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_micro_trunc"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${EXTRA_ARGS}  -4 -w trunc)

set(OCL_30_TESTS_MICRO
    "${TS_NAME}${TS_SUFFIX}_api_ocl_30"
    "${TS_NAME}${TS_SUFFIX}_basic_other_ocl_30"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_1"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_2"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_3"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_4"
    "${TS_NAME}${TS_SUFFIX}_device_execution"
    "${TS_NAME}${TS_SUFFIX}_device_timer"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_ocl_30"
    "${TS_NAME}${TS_SUFFIX}_non_uniform_work_group"
    "${TS_NAME}${TS_SUFFIX}_subgroups_1"
    "${TS_NAME}${TS_SUFFIX}_SVM_1"
    "${TS_NAME}${TS_SUFFIX}_SVM_2"
)

if(HAVE_SPIRV_TOOLS)
  list(APPEND OCL_30_TESTS_MICRO
    "${TS_NAME}${TS_SUFFIX}_spirv_new_micro_0"
    "${TS_NAME}${TS_SUFFIX}_spirv_new_micro_1"
    "${TS_NAME}${TS_SUFFIX}_spirv_new_micro_2")
endif()


set(CTS_MICRO_LIST
    "${TS_NAME}${TS_SUFFIX}_allocations_buffer"
    "${TS_NAME}${TS_SUFFIX}_allocations_image"
    "${TS_NAME}${TS_SUFFIX}_api"
    "${TS_NAME}${TS_SUFFIX}_atomics"
    "${TS_NAME}${TS_SUFFIX}_basic_math"
    "${TS_NAME}${TS_SUFFIX}_basic_s2v"
    "${TS_NAME}${TS_SUFFIX}_basic_image"
    "${TS_NAME}${TS_SUFFIX}_buffers_read"
    "${TS_NAME}${TS_SUFFIX}_buffers_write"
    "${TS_NAME}${TS_SUFFIX}_buffers_fill"
    "${TS_NAME}${TS_SUFFIX}_buffers_var1"
    "${TS_NAME}${TS_SUFFIX}_buffers_var2"
    "${TS_NAME}${TS_SUFFIX}_buffers_map_read"
    "${TS_NAME}${TS_SUFFIX}_buffers_map_write"
    "${TS_NAME}${TS_SUFFIX}_commonfns"
    "${TS_NAME}${TS_SUFFIX}_compiler_micro"
    "${TS_NAME}${TS_SUFFIX}_computeinfo"
    "${TS_NAME}${TS_SUFFIX}_contractions"
    "${TS_NAME}${TS_SUFFIX}_device_partition"
    "${TS_NAME}${TS_SUFFIX}_events"
    "${TS_NAME}${TS_SUFFIX}_events_ooo"
    "${TS_NAME}${TS_SUFFIX}_events_other"
    "${TS_NAME}${TS_SUFFIX}_geometrics"
    "${TS_NAME}${TS_SUFFIX}_images_cl_copy_images"
    "${TS_NAME}${TS_SUFFIX}_images_cl_fill_images"
    "${TS_NAME}${TS_SUFFIX}_images_cl_get_info"
    "${TS_NAME}${TS_SUFFIX}_images_cl_read_write_images"

    "${TS_NAME}${TS_SUFFIX}_mem_host_flags"
    "${TS_NAME}${TS_SUFFIX}_multiples"
    "${TS_NAME}${TS_SUFFIX}_printf"
    "${TS_NAME}${TS_SUFFIX}_profiling"
    "${TS_NAME}${TS_SUFFIX}_relationals"
    "${TS_NAME}${TS_SUFFIX}_thread_dimensions_quick"
    "${TS_NAME}${TS_SUFFIX}_vecstep"

    ${OCL_30_TESTS_MICRO}

    "${TS_NAME}${TS_SUFFIX}_basic_micro_memory"
    "${TS_NAME}${TS_SUFFIX}_basic_micro_other"

    "${TS_NAME}${TS_SUFFIX}_math_micro_add"
    "${TS_NAME}${TS_SUFFIX}_math_micro_assignment"
    "${TS_NAME}${TS_SUFFIX}_math_micro_cbrt"
    "${TS_NAME}${TS_SUFFIX}_math_micro_ceil"
    "${TS_NAME}${TS_SUFFIX}_math_micro_copysign"
    "${TS_NAME}${TS_SUFFIX}_math_micro_cos"
    "${TS_NAME}${TS_SUFFIX}_math_micro_cosh"
    "${TS_NAME}${TS_SUFFIX}_math_micro_cospi"
    "${TS_NAME}${TS_SUFFIX}_math_micro_divide"
    "${TS_NAME}${TS_SUFFIX}_math_micro_exp"
    "${TS_NAME}${TS_SUFFIX}_math_micro_expm1"
    "${TS_NAME}${TS_SUFFIX}_math_micro_fabs"
    "${TS_NAME}${TS_SUFFIX}_math_micro_fdim"
    "${TS_NAME}${TS_SUFFIX}_math_micro_floor"
    "${TS_NAME}${TS_SUFFIX}_math_micro_fma"
    "${TS_NAME}${TS_SUFFIX}_math_micro_fmax"
    "${TS_NAME}${TS_SUFFIX}_math_micro_fmin"
    "${TS_NAME}${TS_SUFFIX}_math_micro_fract"
    "${TS_NAME}${TS_SUFFIX}_math_micro_frexp"
    "${TS_NAME}${TS_SUFFIX}_math_micro_hypot"
    "${TS_NAME}${TS_SUFFIX}_math_micro_ilogb"
    "${TS_NAME}${TS_SUFFIX}_math_micro_isequal"
    "${TS_NAME}${TS_SUFFIX}_math_micro_isfinite"
    "${TS_NAME}${TS_SUFFIX}_math_micro_isgreater"
    "${TS_NAME}${TS_SUFFIX}_math_micro_isgreaterequal"
    "${TS_NAME}${TS_SUFFIX}_math_micro_isinf"
    "${TS_NAME}${TS_SUFFIX}_math_micro_isless"
    "${TS_NAME}${TS_SUFFIX}_math_micro_islessequal"
    "${TS_NAME}${TS_SUFFIX}_math_micro_islessgreater"
    "${TS_NAME}${TS_SUFFIX}_math_micro_isnan"
    "${TS_NAME}${TS_SUFFIX}_math_micro_isnormal"
    "${TS_NAME}${TS_SUFFIX}_math_micro_isnotequal"
    "${TS_NAME}${TS_SUFFIX}_math_micro_isordered"
    "${TS_NAME}${TS_SUFFIX}_math_micro_isunordered"
    "${TS_NAME}${TS_SUFFIX}_math_micro_ldexp"
    "${TS_NAME}${TS_SUFFIX}_math_micro_log"
    "${TS_NAME}${TS_SUFFIX}_math_micro_log10"
    "${TS_NAME}${TS_SUFFIX}_math_micro_log1p"
    "${TS_NAME}${TS_SUFFIX}_math_micro_log2"
    "${TS_NAME}${TS_SUFFIX}_math_micro_logb"
    "${TS_NAME}${TS_SUFFIX}_math_micro_mad"
    "${TS_NAME}${TS_SUFFIX}_math_micro_maxmag"
    "${TS_NAME}${TS_SUFFIX}_math_micro_minmag"
    "${TS_NAME}${TS_SUFFIX}_math_micro_modf"
    "${TS_NAME}${TS_SUFFIX}_math_micro_multiply"
    "${TS_NAME}${TS_SUFFIX}_math_micro_nan"
    "${TS_NAME}${TS_SUFFIX}_math_micro_nextafter"
    "${TS_NAME}${TS_SUFFIX}_math_micro_not"
    "${TS_NAME}${TS_SUFFIX}_math_micro_pown"
    "${TS_NAME}${TS_SUFFIX}_math_micro_rint"
    "${TS_NAME}${TS_SUFFIX}_math_micro_rootn"
    "${TS_NAME}${TS_SUFFIX}_math_micro_round"
    "${TS_NAME}${TS_SUFFIX}_math_micro_rsqrt"
    "${TS_NAME}${TS_SUFFIX}_math_micro_signbit"
    "${TS_NAME}${TS_SUFFIX}_math_micro_sin"
    "${TS_NAME}${TS_SUFFIX}_math_micro_sinh"
    "${TS_NAME}${TS_SUFFIX}_math_micro_sinpi"
    "${TS_NAME}${TS_SUFFIX}_math_micro_sqrt"
    "${TS_NAME}${TS_SUFFIX}_math_micro_subtract"
    "${TS_NAME}${TS_SUFFIX}_math_micro_tan"
    "${TS_NAME}${TS_SUFFIX}_math_micro_tanh"
    "${TS_NAME}${TS_SUFFIX}_math_micro_tanpi"
    "${TS_NAME}${TS_SUFFIX}_math_micro_trunc"

    "${TS_NAME}${TS_SUFFIX}_conversion_micro_uint_uchar"
    "${TS_NAME}${TS_SUFFIX}_conversion_micro_short_char"
    "${TS_NAME}${TS_SUFFIX}_conversion_micro_uchar_ushort"
    "${TS_NAME}${TS_SUFFIX}_conversion_micro_char_int"
    "${TS_NAME}${TS_SUFFIX}_conversion_micro_float_char"
    "${TS_NAME}${TS_SUFFIX}_conversion_micro_float_ushort"
    "${TS_NAME}${TS_SUFFIX}_conversion_micro_int_float"
    "${TS_NAME}${TS_SUFFIX}_conversion_micro_double_short"
    "${TS_NAME}${TS_SUFFIX}_conversion_micro_double_ulong"
    "${TS_NAME}${TS_SUFFIX}_conversion_micro_long_double"

    "${TS_NAME}${TS_SUFFIX}_images_micro_kernel_image_methods_1D"
    "${TS_NAME}${TS_SUFFIX}_images_micro_kernel_image_methods_2D"
    "${TS_NAME}${TS_SUFFIX}_images_micro_kernel_image_methods_2D"
    "${TS_NAME}${TS_SUFFIX}_images_micro_kernel_image_methods_1Darray"
    "${TS_NAME}${TS_SUFFIX}_images_micro_kernel_image_methods_2Darray"

    "${TS_NAME}${TS_SUFFIX}_images_micro_image_streams_1D"
    "${TS_NAME}${TS_SUFFIX}_images_micro_image_streams_2D"
    "${TS_NAME}${TS_SUFFIX}_images_micro_image_streams_3D"
    "${TS_NAME}${TS_SUFFIX}_images_micro_image_streams_1Darray"
    "${TS_NAME}${TS_SUFFIX}_images_micro_image_streams_2Darray"

    "${TS_NAME}${TS_SUFFIX}_images_micro_samplerless_reads_1D"
    "${TS_NAME}${TS_SUFFIX}_images_micro_samplerless_reads_2D"
    "${TS_NAME}${TS_SUFFIX}_images_micro_samplerless_reads_3D"
    "${TS_NAME}${TS_SUFFIX}_images_micro_samplerless_reads_1Darray"
    "${TS_NAME}${TS_SUFFIX}_images_micro_samplerless_reads_2Darray"

    "${TS_NAME}${TS_SUFFIX}_integer_ops_micro_1"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_micro_3"

    "${TS_NAME}${TS_SUFFIX}_select_micro_char_char"
    "${TS_NAME}${TS_SUFFIX}_select_micro_ushort_ushort"
    "${TS_NAME}${TS_SUFFIX}_select_micro_int_int"
    "${TS_NAME}${TS_SUFFIX}_select_micro_ulong_ulong")

foreach(CTSTEST IN LISTS CTS_MICRO_LIST)
  set_property(TEST ${CTSTEST}
    APPEND PROPERTY
    LABELS ${CTS_LABEL})
  set_property(TEST ${CTSTEST}
    PROPERTY
    ENVIRONMENT "POCL_MEMORY_LIMIT=1")
endforeach()

set_property(TEST "${TS_NAME}${TS_SUFFIX}_non_uniform_work_group"
             PROPERTY ENVIRONMENT "POCL_MEMORY_LIMIT=4")

endfunction()

add_cts_micro_tests("conformance_suite_micro_main" "_main")

if(HAVE_SPIRV_TOOLS)
set(CACHE_DIR "${CMAKE_BINARY_DIR}/CTS_MICRO_CACHE")
file(MAKE_DIRECTORY "${CACHE_DIR}")
add_cts_micro_tests("conformance_suite_micro_spirv" "_spirv"
                   "--compilation-mode" "spir-v"
                   "--compilation-program" "${CMAKE_BINARY_DIR}/cl_offline_compiler.sh"
                   "--compilation-cache-path" "${CACHE_DIR}")
endif()

#################################################################################

