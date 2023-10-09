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

message(STATUS "Enabling testsuite ${TS_NAME} (OpenCL-CTS version 3)")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

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
add_test(NAME "${TS_NAME}${TS_SUFFIX}_allocations"
         COMMAND "${TS_BUILDDIR}/allocations/test_allocations" ${EXTRA_ARGS})

if(HOST_CPU_CORECOUNT GREATER 4)
  set(PAR_ARG "--num-worker-threads" ${HOST_CPU_CORECOUNT})
endif()

# TODO fix this in pocl's fork:::
# on recent LLVMs, run all tests except the get_kernel_arg_info.
# That test incorrectly assumes "const" type qualifiers are returned
# for non-pointer arguments too. The spec PDF says type qualifiers
# are only returned for pointer types, so the test is incorrect AFAICT
add_test(NAME "${TS_NAME}${TS_SUFFIX}_api"
           COMMAND "${TS_BUILDDIR}/api/test_api" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_atomics"
         COMMAND "${TS_BUILDDIR}/atomics/test_atomics" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_basic"
         COMMAND "${TS_BUILDDIR}/basic/test_basic" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_commonfns"
         COMMAND "${TS_BUILDDIR}/commonfns/test_commonfns" ${EXTRA_ARGS} )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_compiler"
         COMMAND "${TS_BUILDDIR}/compiler/test_compiler" ${EXTRA_ARGS}
         WORKING_DIRECTORY "${TS_BASEDIR}/src/${TS_NAME}/test_conformance/compiler")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_c11_atomics"
         COMMAND "${TS_BUILDDIR}/c11_atomics/test_c11_atomics" ${PAR_ARG} ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_computeinfo"
         COMMAND "${TS_BUILDDIR}/computeinfo/test_computeinfo" ${EXTRA_ARGS} )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_contractions"
         COMMAND "${TS_BUILDDIR}/contractions/test_contractions" ${EXTRA_ARGS} )

add_test(NAME "${TS_NAME}${TS_SUFFIX}_conversions"
         COMMAND "${TS_BUILDDIR}/conversions/test_conversions" ${PAR_ARG} ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_device_partition"
         COMMAND "${TS_BUILDDIR}/device_partition/test_device_partition" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_device_execution"
         COMMAND "${TS_BUILDDIR}/device_execution/test_device_execution" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_device_timer"
         COMMAND "${TS_BUILDDIR}/device_timer/test_device_timer" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_events"
         COMMAND "${TS_BUILDDIR}/events/test_events" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_generic_AS"
         COMMAND "${TS_BUILDDIR}/generic_address_space/test_generic_address_space" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_geometrics"
         COMMAND "${TS_BUILDDIR}/geometrics/test_geometrics" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_half"
         COMMAND "${TS_BUILDDIR}/half/test_half" ${EXTRA_ARGS})

#***************************************************************************

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_copy_images"
         COMMAND "${TS_BUILDDIR}/images/clCopyImage/test_cl_copy_images" ${EXTRA_ARGS})
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_copy_images_max"
         COMMAND "${TS_BUILDDIR}/images/clCopyImage/test_cl_copy_images" ${EXTRA_ARGS} max_images)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_copy_images_min"
         COMMAND "${TS_BUILDDIR}/images/clCopyImage/test_cl_copy_images" ${EXTRA_ARGS} small_images)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_cl_fill_images"
         COMMAND "${TS_BUILDDIR}/images/clFillImage/test_cl_fill_images" ${EXTRA_ARGS})
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

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_kernel_image_methods"
         COMMAND "${TS_BUILDDIR}/images/kernel_image_methods/test_kernel_image_methods" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_image_streams"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_image_streams_pitch"
         COMMAND "${TS_BUILDDIR}/images/kernel_read_write/test_image_streams" ${EXTRA_ARGS} use_pitches)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_pitch"
         COMMAND "${TS_BUILDDIR}/images/samplerlessReads/test_samplerless_reads" ${EXTRA_ARGS} use_pitches)

#***************************************************************************

add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${PAR_ARG} ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_math_bruteforce"
         COMMAND "${TS_BUILDDIR}/math_brute_force/test_bruteforce" ${PAR_ARG} ${EXTRA_ARGS})

#***************************************************************************

add_test(NAME "${TS_NAME}${TS_SUFFIX}_mem_host_flags"
         COMMAND "${TS_BUILDDIR}/mem_host_flags/test_mem_host_flags" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_multiples"
         COMMAND "${TS_BUILDDIR}/multiple_device_context/test_multiples" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_non_uniform_work_group"
         COMMAND "${TS_BUILDDIR}/non_uniform_work_group/test_non_uniform_work_group" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_pipes"
         COMMAND "${TS_BUILDDIR}/pipes/test_pipes" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_printf"
         COMMAND "${TS_BUILDDIR}/printf/test_printf" ${EXTRA_ARGS}
         WORKING_DIRECTORY "${TS_BASEDIR}/src/${TS_NAME}/test_conformance/printf")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_profiling"
         COMMAND "${TS_BUILDDIR}/profiling/test_profiling" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_relationals"
         COMMAND "${TS_BUILDDIR}/relationals/test_relationals" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_select"
         COMMAND "${TS_BUILDDIR}/select/test_select" ${PAR_ARG} ${EXTRA_ARGS})

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

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS}
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

endif()

add_test(NAME "${TS_NAME}${TS_SUFFIX}_subgroups"
         COMMAND "${TS_BUILDDIR}/subgroups/test_subgroups" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_SVM"
         COMMAND "${TS_BUILDDIR}/SVM/test_svm" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_thread_dimensions"
         COMMAND "${TS_BUILDDIR}/thread_dimensions/test_thread_dimensions" ${PAR_ARG} ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_vectors"
         COMMAND "${TS_BUILDDIR}/vectors/test_vectors" ${PAR_ARG} ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_workgroup"
         COMMAND "${TS_BUILDDIR}/workgroups/test_workgroups" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_cl_ext_cxx"
         COMMAND "${TS_BUILDDIR}/extensions/cl_ext_cxx_for_opencl/test_cl_ext_cxx_for_opencl" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_cl_khr_command_buffer"
         COMMAND "${TS_BUILDDIR}/extensions/cl_khr_command_buffer/test_cl_khr_command_buffer" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_cl_khr_command_buffer_mutable"
         COMMAND "${TS_BUILDDIR}/extensions/cl_khr_command_buffer/cl_khr_command_buffer_mutable_dispatch/test_cl_khr_mutable_dispatch" ${EXTRA_ARGS})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_cl_khr_semaphore"
         COMMAND "${TS_BUILDDIR}/extensions/cl_khr_semaphore/test_cl_khr_semaphore" ${EXTRA_ARGS})

set(OCL_30_TESTS_FULL
    "${TS_NAME}${TS_SUFFIX}_api"
    "${TS_NAME}${TS_SUFFIX}_basic"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics"
    "${TS_NAME}${TS_SUFFIX}_device_execution"
    "${TS_NAME}${TS_SUFFIX}_device_timer"
    "${TS_NAME}${TS_SUFFIX}_generic_AS"
    "${TS_NAME}${TS_SUFFIX}_integer_ops"
    "${TS_NAME}${TS_SUFFIX}_non_uniform_work_group"
    "${TS_NAME}${TS_SUFFIX}_pipes"
    "${TS_NAME}${TS_SUFFIX}_select"
    "${TS_NAME}${TS_SUFFIX}_select"
    "${TS_NAME}${TS_SUFFIX}_subgroups"
    "${TS_NAME}${TS_SUFFIX}_SVM"
    "${TS_NAME}${TS_SUFFIX}_workgroup"
    "${TS_NAME}${TS_SUFFIX}_cl_ext_cxx"
    "${TS_NAME}${TS_SUFFIX}_cl_khr_command_buffer"
    "${TS_NAME}${TS_SUFFIX}_cl_khr_command_buffer_mutable"
    "${TS_NAME}${TS_SUFFIX}_cl_khr_semaphore"
)

if(HAVE_SPIRV_TOOLS)
  list(APPEND OCL_30_TESTS_FULL
    "${TS_NAME}${TS_SUFFIX}_spirv_new")
endif()

set(CTS_FULL_LIST
    "${TS_NAME}${TS_SUFFIX}_allocations"
    "${TS_NAME}${TS_SUFFIX}_api"
    "${TS_NAME}${TS_SUFFIX}_atomics"
    "${TS_NAME}${TS_SUFFIX}_basic"
    "${TS_NAME}${TS_SUFFIX}_buffers"
    "${TS_NAME}${TS_SUFFIX}_commonfns"
    "${TS_NAME}${TS_SUFFIX}_compiler"
    "${TS_NAME}${TS_SUFFIX}_computeinfo"
    "${TS_NAME}${TS_SUFFIX}_contractions"
    "${TS_NAME}${TS_SUFFIX}_conversions"
    "${TS_NAME}${TS_SUFFIX}_device_partition"
    "${TS_NAME}${TS_SUFFIX}_events"
    "${TS_NAME}${TS_SUFFIX}_geometrics"
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
    "${TS_NAME}${TS_SUFFIX}_images_kernel_image_methods"
    "${TS_NAME}${TS_SUFFIX}_images_image_streams"
    "${TS_NAME}${TS_SUFFIX}_images_image_streams_pitch"
    "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads"
    "${TS_NAME}${TS_SUFFIX}_images_samplerless_reads_pitch"
    "${TS_NAME}${TS_SUFFIX}_integer_ops"
    "${TS_NAME}${TS_SUFFIX}_math_bruteforce"
    "${TS_NAME}${TS_SUFFIX}_mem_host_flags"
    "${TS_NAME}${TS_SUFFIX}_multiples"
    "${TS_NAME}${TS_SUFFIX}_printf"
    "${TS_NAME}${TS_SUFFIX}_profiling"
    "${TS_NAME}${TS_SUFFIX}_relationals"
    "${TS_NAME}${TS_SUFFIX}_select"
    "${TS_NAME}${TS_SUFFIX}_thread_dimensions"
    "${TS_NAME}${TS_SUFFIX}_vectors"
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

# add the full testsuite with online compilation from sources
add_cts_full_tests("conformance_suite_full_main" "_main")

if(HAVE_SPIRV_TOOLS)
set(CACHE_DIR "${CMAKE_BINARY_DIR}/CTS_FULL_CACHE")
file(MAKE_DIRECTORY "${CACHE_DIR}")

# add the full testsuite with offline compilation from SPIR-V
add_cts_full_tests("conformance_suite_full_spirv" "_spirv"
                   "--compilation-mode" "spir-v"
                   "--compilation-program" "${CMAKE_BINARY_DIR}/cl_offline_compiler.sh"
                   "--compilation-cache-path" "${CACHE_DIR}")

# these are only approximate costs from one particular machine.
# CTest launches the tests in order from most expensive to least,
# so this helps to minimize the testsuite runtime.
set_property(TEST conformance_main_buffers PROPERTY COST 76.70)
set_property(TEST conformance_main_generic_AS PROPERTY COST 100.44)
set_property(TEST conformance_main_spirv_new PROPERTY COST 63.91)
set_property(TEST conformance_main_relationals PROPERTY COST 127.25 PROCESSORS 2)
set_property(TEST conformance_main_compiler PROPERTY COST 166.07 PROCESSORS 2)
set_property(TEST conformance_main_basic PROPERTY COST 505.27 PROCESSORS 2)
set_property(TEST conformance_main_images_image_streams_pitch PROPERTY COST 736.54 PROCESSORS 2)
set_property(TEST conformance_main_images_image_streams PROPERTY COST 740.93 PROCESSORS 2)
set_property(TEST conformance_main_thread_dimensions PROPERTY COST 1351.20 PROCESSORS 2)
set_property(TEST conformance_main_vectors PROPERTY COST 1515.12 PROCESSORS 2)
set_property(TEST conformance_main_select PROPERTY COST 2107.24 PROCESSORS 2)
set_property(TEST conformance_main_integer_ops PROPERTY COST 2387.76 PROCESSORS 4)
set_property(TEST conformance_main_c11_atomics PROPERTY COST 6528.94 PROCESSORS 3)
set_property(TEST conformance_main_conversions PROPERTY COST 24581.43 PROCESSORS 3)
set_property(TEST conformance_main_math_bruteforce PROPERTY COST 60000 PROCESSORS ${PARALLELISM})


endif()

#################################################################################
#################################################################################
########################   MICRO   ##############################################
#################################################################################
#################################################################################

function(add_cts_micro_tests CTS_LABEL TS_SUFFIX)
set(EXTRA_ARGS ${ARGN})

add_test(NAME "${TS_NAME}${TS_SUFFIX}_api_micro_ocl_30"
         COMMAND "${TS_BUILDDIR}/api/test_api" ${EXTRA_ARGS} buffer_properties_queries   clone_kernel
         consistency_2d_image_from_buffer   consistency_3d_image_writes   consistency_depth_images
         consistency_device_and_host_timer   consistency_device_enqueue   consistency_il_programs
         consistency_memory_model   consistency_non_uniform_work_group   consistency_pipes
         consistency_prog_ctor_dtor   consistency_progvar   consistency_read_write_images
         consistency_subgroups   consistency_svm   context_destructor_callback  zero_sized_enqueue
         get_command_queue_info_compatibility   get_sampler_info_compatibility  sub_group_dispatch
         image_properties_queries   ${API_OCL_30_EXTRA_TEST}    kernel_private_memory_size
         min_image_formats   negative_get_platform_ids   negative_get_platform_info
         pipe_properties_queries   queue_flush_on_release   queue_hint   queue_properties
         queue_properties_queries   retain_mem_object_set_kernel_arg)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_subgroups_micro"
         COMMAND "${TS_BUILDDIR}/subgroups/test_subgroups" ${EXTRA_ARGS} sub_group_info_ext
         sub_group_info_core work_item_functions_ext  work_item_functions_core  subgroup_functions_ext
         subgroup_functions_core  barrier_functions_ext  barrier_functions_core)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_basic_micro_other_ocl_30"
         COMMAND "${TS_BUILDDIR}/basic/test_basic" ${EXTRA_ARGS} async_copy_global_to_local2D
          async_copy_global_to_local3D  async_copy_local_to_global2D async_copy_local_to_global3D
          async_work_group_copy_fence_export_after_import_aliased_global
          async_work_group_copy_fence_export_after_import_aliased_global_and_local
          async_work_group_copy_fence_export_after_import_aliased_local
          async_work_group_copy_fence_import_after_export_aliased_global
          async_work_group_copy_fence_import_after_export_aliased_global_and_local
          async_work_group_copy_fence_import_after_export_aliased_local wg_barrier
          enqueued_local_size   get_linear_ids  global_linear_id  local_linear_id
          progvar_func_scope  progvar_prog_scope_init  progvar_prog_scope_misc   progvar_prog_scope_uninit
          rw_image_access_qualifier   simple_read_image_pitch   simple_write_image_pitch    vector_swizzle)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_SVM_micro_1"
         COMMAND "${TS_BUILDDIR}/SVM/test_svm" ${EXTRA_ARGS} svm_byte_granularity  svm_set_kernel_exec_info_svm_ptrs  svm_fine_grain_memory_consistency  svm_fine_grain_sync_buffers  svm_shared_address_space_fine_grain  svm_shared_sub_buffers  svm_shared_address_space_fine_grain_buffers)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_SVM_micro_2"
         COMMAND "${TS_BUILDDIR}/SVM/test_svm" ${EXTRA_ARGS} svm_allocate_shared_buffer  svm_shared_address_space_coarse_grain_old_api  svm_shared_address_space_coarse_grain_new_api  svm_cross_buffer_pointers_coarse_grain  svm_pointer_passing  svm_enqueue_api  svm_migrate)

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

add_test(NAME "${TS_NAME}${TS_SUFFIX}_allocations_micro_buffer"
         COMMAND "${TS_BUILDDIR}/allocations/test_allocations" ${EXTRA_ARGS} buffer buffer_non_blocking)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_allocations_micro_image"
         COMMAND "${TS_BUILDDIR}/allocations/test_allocations" ${EXTRA_ARGS} image2d_read  image2d_write
         image2d_read_non_blocking  image2d_write_non_blocking)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_integer_ops_micro_ocl_30"
         COMMAND "${TS_BUILDDIR}/integer_ops/test_integer_ops" ${EXTRA_ARGS}  integer_ctz integer_dot_product)

# skips two tests that fail in SPIR-V mode:
# compare_pointers
# builtin_functions
add_test(NAME "${TS_NAME}${TS_SUFFIX}_generic_AS_micro"
         COMMAND "${TS_BUILDDIR}/generic_address_space/test_generic_address_space" ${EXTRA_ARGS}
         function_get_fence function_to_address_space variable_get_fence variable_to_address_space casting conditional_casting chain_casting ternary_operator_casting
         language_struct language_union multiple_calls_same_function library_function generic_variable_volatile generic_variable_const
         generic_variable_gentype generic_advanced_casting generic_ptr_to_host_mem generic_ptr_to_host_mem_svm max_number_of_params)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_basic_micro_s2v"
         COMMAND "${TS_BUILDDIR}/basic/test_basic" ${EXTRA_ARGS} explicit_s2v_char  explicit_s2v_uchar  explicit_s2v_short  explicit_s2v_ushort  explicit_s2v_int  explicit_s2v_uint  explicit_s2v_long  explicit_s2v_ulong  explicit_s2v_float  explicit_s2v_double)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_basic_micro_math"
         COMMAND "${TS_BUILDDIR}/basic/test_basic" ${EXTRA_ARGS} fpmath fp2int int2fp intmath_int intmath_int2 intmath_int4 intmath_long intmath_long2 intmath_long4)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_basic_micro_image"
         COMMAND "${TS_BUILDDIR}/basic/test_basic" ${EXTRA_ARGS} readimage  readimage_int16  readimage_fp32  writeimage  writeimage_int16  writeimage_fp32  image_r8  imagereadwrite  imagereadwrite3d  readimage3d  readimage3d_int16  readimage3d_fp32  imagearraycopy  imagearraycopy3d  imagecopy  imagecopy3d  imagerandomcopy  arrayimagecopy  arrayimagecopy3d  imagenpot  imagedim_pow2  imagedim_non_pow2  image_param  image_multipass_integer_coord  image_multipass_float_coord  enqueue_map_image)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_basic_micro_other"
         COMMAND "${TS_BUILDDIR}/basic/test_basic" ${EXTRA_ARGS} mri_one  mri_multiple  barrier  createkernelsinprogram  work_item_functions    kernel_call_kernel_function  host_numeric_constants  kernel_numeric_constants  kernel_limit_constants  kernel_preprocessor_macros  parameter_types  vector_creation  vec_type_hint  global_work_offsets  get_global_offset  hostptr    if  sizeof  loop  pointer_cast  local_arg_def  local_kernel_def  local_kernel_scope  constant  constant_source)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_basic_micro_memory"
         COMMAND "${TS_BUILDDIR}/basic/test_basic" ${EXTRA_ARGS} vload_global  vload_local  vstore_global  vstore_local   bufferreadwriterect  arrayreadwrite  arraycopy  enqueue_map_buffer   kernel_memory_alignment_local  kernel_memory_alignment_global)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_micro_read_async"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS} buffer_read_async_int  buffer_read_async_uint  buffer_read_async_long  buffer_read_async_ulong  buffer_read_async_short  buffer_read_async_ushort  buffer_read_async_char  buffer_read_async_uchar  buffer_read_async_float)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_micro_write_async"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS}  buffer_write_async_int  buffer_write_async_uint  buffer_write_async_short  buffer_write_async_ushort  buffer_write_async_char  buffer_write_async_uchar  buffer_write_async_float  buffer_write_async_long  buffer_write_async_ulong)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_micro_read_array_barrier"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS} buffer_read_array_barrier_int  buffer_read_array_barrier_uint  buffer_read_array_barrier_long  buffer_read_array_barrier_ulong  buffer_read_array_barrier_short  buffer_read_array_barrier_ushort  buffer_read_array_barrier_char  buffer_read_array_barrier_uchar  buffer_read_array_barrier_float)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_micro_read"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS} buffer_read_int  buffer_read_uint  buffer_read_long  buffer_read_ulong  buffer_read_short  buffer_read_ushort  buffer_read_float  buffer_read_char  buffer_read_uchar  buffer_read_struct  buffer_read_random_size buffer_read_half)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_micro_write"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS} buffer_write_int  buffer_write_uint  buffer_write_short  buffer_write_ushort  buffer_write_char  buffer_write_uchar  buffer_write_float  buffer_write_long  buffer_write_ulong  buffer_write_struct buffer_write_half)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_micro_fill"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS}   buffer_fill_int  buffer_fill_uint  buffer_fill_short  buffer_fill_ushort  buffer_fill_char  buffer_fill_uchar  buffer_fill_long  buffer_fill_ulong  buffer_fill_float  buffer_fill_struct)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_micro_var1"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS} buffer_copy  buffer_partial_copy  mem_read_write_flags  mem_write_only_flags  mem_read_only_flags  mem_copy_host_flags  mem_alloc_ref_flags  array_info_size)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_micro_var2"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS} sub_buffers_read_write  sub_buffers_read_write_dual_devices  sub_buffers_overlapping  buffer_migrate  image_migrate)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_micro_map_read"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS} buffer_map_read_int  buffer_map_read_uint  buffer_map_read_long  buffer_map_read_ulong  buffer_map_read_short  buffer_map_read_ushort  buffer_map_read_char  buffer_map_read_uchar  buffer_map_read_float  buffer_map_read_struct)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_buffers_micro_map_write"
         COMMAND "${TS_BUILDDIR}/buffers/test_buffers" ${EXTRA_ARGS} buffer_map_write_int  buffer_map_write_uint  buffer_map_write_long  buffer_map_write_ulong  buffer_map_write_short  buffer_map_write_ushort  buffer_map_write_char  buffer_map_write_uchar  buffer_map_write_float  buffer_map_write_struct)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_vectors_micro"
         COMMAND "${TS_BUILDDIR}/vectors/test_vectors" ${EXTRA_ARGS} vec_align_array vec_align_struct vec_align_struct_arr step_type step_var)


if(HAVE_SPIRV_TOOLS)

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_micro_0"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS} decorate_restrict  decorate_aliased  decorate_alignment  decorate_constant  decorate_cpacked  decorate_saturated_conversion_char   decorate_saturated_conversion_ushort  decorate_saturated_conversion_int  decorate_fp_rounding_mode_rte_float_int  decorate_fp_rounding_mode_rtz_double_long get_program_il  linkage_export_function_compile  linkage_import_function_compile  linkage_import_function_link  op_atomic_inc_global  op_atomic_dec_global  op_label_simple  op_branch_simple  op_unreachable_simple  op_branch_conditional  op_branch_conditional_weighted  op_composite_construct_int4  op_composite_construct_struct
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_micro_1"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS} op_constant_false_simple  op_constant_int_simple  op_constant_uchar_simple op_constant_long_simple op_constant_float_simple         op_constant_struct_int_char_simple  op_constant_struct_struct_simple op_copy_int3_simple  op_copy_struct_int_float_simple  op_copy_struct_int_char_simple  op_copy_struct_struct_simple  op_copy_half_simple op_fadd_float_regular  op_fmul_float_regular   op_fadd_float_fast  op_fadd_double_regular  op_fsub_double_fast  op_fdiv_double_fast  op_fadd_float4_regular  op_fmul_float4_regular  op_fsub_float4_fast  op_fadd_double2_regular  op_fsub_double2_fast
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")

add_test(NAME "${TS_NAME}${TS_SUFFIX}_spirv_new_micro_2"
         COMMAND "${TS_BUILDDIR}/spirv_new/test_spirv_new" ${EXTRA_ARGS} function_none  function_inline  function_noinline op_lifetime_simple  op_loop_merge_branch_none op_neg_float  op_neg_double op_not_int4  op_type_opaque_simple  op_phi_2_blocks op_selection_merge_if_none  op_selection_merge_if_flatten op_spec_constant_uchar_simple  op_spec_constant_ushort_simple ${SPIRV_OCL_30_EXTRA_TEST}  op_spec_constant_true_simple op_undef_true_simple op_undef_uint_simple  op_undef_char_simple op_undef_float_simple  op_undef_double_simple op_vector_int4_extract  op_vector_float4_extract op_vector_double2_extract  op_vector_char16_extract  op_vector_int4_insert op_vector_times_scalar_double
         WORKING_DIRECTORY "${TS_BUILDDIR}/spirv_new")
endif()

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

add_test(NAME "${TS_NAME}${TS_SUFFIX}_thread_dimensions_micro"
         COMMAND "${TS_BUILDDIR}/thread_dimensions/test_thread_dimensions" ${EXTRA_ARGS} quick_1d_explicit_local  quick_2d_explicit_local  quick_3d_explicit_local  quick_1d_implicit_local  quick_2d_implicit_local  quick_3d_implicit_local)

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

add_test(NAME "${TS_NAME}${TS_SUFFIX}_events_micro"
         COMMAND "${TS_BUILDDIR}/events/test_events" ${EXTRA_ARGS} event_get_execute_status  event_get_write_array_status  event_get_read_array_status  event_get_info  event_wait_for_execute  event_wait_for_array  event_flush  event_finish_execute  event_finish_array  event_release_before_done  event_enqueue_marker  event_enqueue_marker_with_event_list  event_enqueue_barrier_with_event_list)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_events_micro_ooo"
         COMMAND "${TS_BUILDDIR}/events/test_events" ${EXTRA_ARGS} out_of_order_event_waitlist_single_queue  out_of_order_event_waitlist_multi_queue  out_of_order_event_waitlist_multi_queue_multi_device  out_of_order_event_enqueue_wait_for_events_single_queue  out_of_order_event_enqueue_wait_for_events_multi_queue  out_of_order_event_enqueue_wait_for_events_multi_queue_multi_device  out_of_order_event_enqueue_marker_single_queue  out_of_order_event_enqueue_marker_multi_queue  out_of_order_event_enqueue_marker_multi_queue_multi_device  out_of_order_event_enqueue_barrier_single_queue)
add_test(NAME "${TS_NAME}${TS_SUFFIX}_events_micro_other"
         COMMAND "${TS_BUILDDIR}/events/test_events" ${EXTRA_ARGS} userevents  waitlists  callbacks  callbacks_simultaneous  userevents_multithreaded)

set(OCL_30_TESTS_MICRO
    "${TS_NAME}${TS_SUFFIX}_api_micro_ocl_30"
    "${TS_NAME}${TS_SUFFIX}_basic_micro_other_ocl_30"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_1"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_2"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_3"
    "${TS_NAME}${TS_SUFFIX}_c11_atomics_micro_4"
    "${TS_NAME}${TS_SUFFIX}_device_execution"
    "${TS_NAME}${TS_SUFFIX}_device_timer"
    "${TS_NAME}${TS_SUFFIX}_integer_ops_micro_ocl_30"
    "${TS_NAME}${TS_SUFFIX}_non_uniform_work_group"
    "${TS_NAME}${TS_SUFFIX}_generic_AS_micro"
    "${TS_NAME}${TS_SUFFIX}_subgroups_micro"
    "${TS_NAME}${TS_SUFFIX}_SVM_micro_1"
    "${TS_NAME}${TS_SUFFIX}_SVM_micro_2"
)

if(HAVE_SPIRV_TOOLS)
  list(APPEND OCL_30_TESTS_MICRO
    "${TS_NAME}${TS_SUFFIX}_spirv_new_micro_0"
    "${TS_NAME}${TS_SUFFIX}_spirv_new_micro_1"
    "${TS_NAME}${TS_SUFFIX}_spirv_new_micro_2")
endif()


set(CTS_MICRO_LIST
    "${TS_NAME}${TS_SUFFIX}_allocations_micro_buffer"
    "${TS_NAME}${TS_SUFFIX}_allocations_micro_image"
    "${TS_NAME}${TS_SUFFIX}_api"
    "${TS_NAME}${TS_SUFFIX}_atomics"
    "${TS_NAME}${TS_SUFFIX}_basic_micro_math"
    "${TS_NAME}${TS_SUFFIX}_basic_micro_s2v"
    "${TS_NAME}${TS_SUFFIX}_basic_micro_image"
    "${TS_NAME}${TS_SUFFIX}_buffers_micro_read"
    "${TS_NAME}${TS_SUFFIX}_buffers_micro_write"
    "${TS_NAME}${TS_SUFFIX}_buffers_micro_fill"
    "${TS_NAME}${TS_SUFFIX}_buffers_micro_var1"
    "${TS_NAME}${TS_SUFFIX}_buffers_micro_var2"
    "${TS_NAME}${TS_SUFFIX}_buffers_micro_map_read"
    "${TS_NAME}${TS_SUFFIX}_buffers_micro_map_write"
    "${TS_NAME}${TS_SUFFIX}_commonfns"
    "${TS_NAME}${TS_SUFFIX}_compiler_micro"
    "${TS_NAME}${TS_SUFFIX}_computeinfo"
    "${TS_NAME}${TS_SUFFIX}_contractions"
    "${TS_NAME}${TS_SUFFIX}_device_partition"
    "${TS_NAME}${TS_SUFFIX}_events_micro"
    "${TS_NAME}${TS_SUFFIX}_events_micro_ooo"
    "${TS_NAME}${TS_SUFFIX}_events_micro_other"
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
    "${TS_NAME}${TS_SUFFIX}_thread_dimensions_micro"
    "${TS_NAME}${TS_SUFFIX}_vectors_micro"

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

# add the micro testsuite with online compilation from sources
add_cts_micro_tests("conformance_suite_micro_main" "_main")

if(HAVE_SPIRV_TOOLS)
set(CACHE_DIR "${CMAKE_BINARY_DIR}/CTS_MICRO_CACHE")
file(MAKE_DIRECTORY "${CACHE_DIR}")

# add the micro testsuite with offline compilation from SPIR-V
add_cts_micro_tests("conformance_suite_micro_spirv" "_spirv"
                   "--compilation-mode" "spir-v"
                   "--compilation-program" "${CMAKE_BINARY_DIR}/cl_offline_compiler.sh"
                   "--compilation-cache-path" "${CACHE_DIR}")
endif()

#################################################################################

