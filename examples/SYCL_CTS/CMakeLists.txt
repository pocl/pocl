#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2025 Michal Babej / Intel Finland Oy
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

set(TS_NAME "SYCL_CTS")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TS_BASEDIR}/src/${TS_NAME}")

if(NOT SYCL_CTS_BINARIES_URL)
if((NOT SYCL_CXX_COMPILER) OR (NOT SYCL_LIBDIR))
  message(STATUS "Disabling testsuite ${TS_NAME}, requires a SYCL compiler (-DSYCL_CXX_COMPILER=...) and also SYCL_LIBDIR (path to libsycl.so)")
  return()
endif()

if(NOT HAVE_GIT)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires git to checkout sources")
  return()
endif()
endif()

set(SYCL_CTS_TEST_LIST
test_has_kernel_bundle_core_atomic64_ctx_only    test_has_kernel_bundle_core_atomic64_dev                                  test_math_builtin_api
test_address_space                               test_has_kernel_bundle_core_fp16_ctx_only                                 test_multi_ptr
test_atomic                                      test_has_kernel_bundle_core_fp16_dev                                      test_namespace
test_atomic_fence                                test_has_kernel_bundle_core_fp64_ctx_only                                 test_nd_item
test_atomic_ref                                  test_has_kernel_bundle_core_fp64_dev                                      test_nd_range
test_atomic_ref_stress                           test_has_kernel_bundle_core_like_support_reqd_sub_group_size_ctx_only     test_optional_kernel_features
test_bit_cast                                    test_has_kernel_bundle_core_like_support_reqd_sub_group_size_dev          test_platform
test_buffer                                      test_has_kernel_bundle_core_like_unsupport_reqd_sub_group_size_ctx_only   test_pointers
test_context                                     test_has_kernel_bundle_core_like_unsupport_reqd_sub_group_size_dev        test_property
test_device                                      test_has_kernel_bundle_core_reqd_like_support_work_group_size_ctx_only    test_queue
test_device_event                                test_has_kernel_bundle_core_reqd_like_support_work_group_size_dev         test_range
test_device_selector                             test_has_kernel_bundle_core_reqd_like_unsupport_work_group_size_ctx_only  test_reduction
test_error                                       test_has_kernel_bundle_core_reqd_like_unsupport_work_group_size_dev       test_sampler
test_event                                       test_has_kernel_bundle_multiple_kernels_ctx_only                          test_scalars
test_exception_handling                          test_has_kernel_bundle_multiple_kernels_dev                               test_spec_constants
test_exceptions                                  test_has_kernel_bundle_zero_kernels_ctx_only_and_ctx_and_dev              test_stream
test_full_feature_set                            test_header                                                               test_sub_group
test_function_objects                            test_hierarchical                                                         test_sycl_external
test_get_kernel_bundle_with_kernel_attr          test_h_item                                                               test_usm
test_get_kernel_bundle_without_kernel_attr       test_host_task                                                            test_vector_alias
test_get_kernel_bundle_zero_kernels              test_id                                                                   test_vector_api
test_get_kernel_ids_multiple_kernels             test_image                                                                test_vector_constructors
test_get_kernel_ids_no_kernels                   test_image_accessor                                                       test_vector_deduction_guides
test_get_kernel_ids_single_kernel                test_invoke                                                               test_vector_load_store
test_group                                       test_is_device_copyable                                                   test_vector_operators
test_group_functions                             test_item                                                                 test_vector_swizzle_assignment
test_handler                                     test_kernel                                                               test_vector_swizzles
test_has_kernel_bundle_core_all_states_ctx_only  test_kernel_args
test_has_kernel_bundle_core_all_states_dev       test_kernel_bundle
test_accessor_basic    test_accessor_generic    test_accessor_legacy    test_accessor_placeholder
test_marray_arithmetic_assignment   test_marray_arithmetic_binary   test_marray_basic   test_marray_bitwise   test_marray_pre_post   test_marray_relational
)

# TODO it seems upstream deleted branch "SYCL-2020"
if(EXAMPLES_USE_GIT_MASTER)
  set(REPO_TAG main)
else()
  set(REPO_TAG a0b35caa8441caa94cb01e14dc52244ebbac93fe)
endif()

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

unset(BUILD_OPTS)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
  set(BUILD_OPTS BUILD_JOB_SERVER_AWARE ON)
endif()

if(SYCL_CTS_BINARIES_URL)
  if(NOT CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    message(FATAL_ERROR "option SYCL_CTS_BINARIES_URL requires CMake >= 3.18")
  endif()
  file(MAKE_DIRECTORY "${TS_BUILDDIR}/download")
  get_filename_component(SYCL_ARCHIVE_NAME "${SYCL_CTS_BINARIES_URL}" NAME)
  file(DOWNLOAD "${SYCL_CTS_BINARIES_URL}" "${TS_BUILDDIR}/download/${SYCL_ARCHIVE_NAME}"
       SHOW_PROGRESS STATUS DOWNLD_STATUS)
  file(ARCHIVE_EXTRACT INPUT "${TS_BUILDDIR}/download/${SYCL_ARCHIVE_NAME}" DESTINATION  "${TS_BUILDDIR}")

else()
  ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  GIT_REPOSITORY "https://github.com/KhronosGroup/SYCL-CTS.git"
  GIT_TAG ${REPO_TAG}
  ${GIT_OPTIONS}
  CMAKE_ARGS
    -DCMAKE_CXX_COMPILER=${SYCL_CXX_COMPILER} -DCMAKE_BUILD_TYPE=Release -DSYCL_IMPLEMENTATION=DPCPP
    -DDPCPP_INSTALL_DIR=/opt/intel/oneapi/compiler/latest -DDPCPP_TARGET_TRIPLES=spirv64
    -DSYCL_CTS_ENABLE_HALF_TESTS=OFF -DSYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS=OFF
    "-DCMAKE_CXX_FLAGS_RELEASE=-O1 -march=native -DNDEBUG"
    "${TS_BASEDIR}/src/${TS_NAME}"
  # avoid building the default target, it includes a "test_all" binary which takes forever to build.
  BUILD_COMMAND ${CMAKE_COMMAND} --build ${TS_BUILDDIR} --target test_conformance
  ${BUILD_OPTS}
  INSTALL_COMMAND "")

  set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_dependencies(prepare_examples ${TS_NAME})
  set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build/bin")

endif()

foreach(TEST IN LISTS SYCL_CTS_TEST_LIST)
  add_test(NAME sycl_cts_${TEST}
         COMMAND "${TS_BUILDDIR}/${TEST}"
         WORKING_DIRECTORY "${TS_BUILDDIR}")

  set_tests_properties(sycl_cts_${TEST} PROPERTIES LABELS "${TS_NAME};cpu")

endforeach()
