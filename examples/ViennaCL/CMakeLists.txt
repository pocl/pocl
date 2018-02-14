#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2015 pocl developers
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

set(TS_NAME "ViennaCL")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")
set(VIENNA "ViennaCL-1.7.0")
set(ViennaCL_TGZ "${TS_SRCDIR}/${VIENNA}.tar.gz")

if(EXAMPLES_USE_GIT_MASTER)
  set(FETCH_SOURCE GIT_REPOSITORY "https://github.com/viennacl/viennacl-dev.git")
else()
  set(FETCH_SOURCE URL "http://sourceforge.net/projects/viennacl/files/1.7.x/ViennaCL-1.7.1.tar.gz/download")
endif()

if(EXISTS "${ViennaCL_TGZ}")

  message(STATUS "Enabling testsuite ${TS_NAME}")
  list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
  set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

  ExternalProject_Add(
    ${TS_NAME}
    PREFIX "${TS_BASEDIR}"
    SOURCE_DIR "${TS_BASEDIR}/src/${VIENNA}"
    DOWNLOAD_COMMAND
      test -d "${TS_BASEDIR}/src/${VIENNA}/ViennaCLConfig.cmake" || tar xzf "${ViennaCL_TGZ}"
    PATCH_COMMAND patch -N -r- -p1 -i
      ${CMAKE_CURRENT_SOURCE_DIR}/vienna_170.patch || /bin/true
    UPDATE_COMMAND /bin/true
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=RelWithDebInfo
      -DBUILD_TESTING:BOOL=ON
      -DBUILD_EXAMPLES:BOOL=ON
      -DENABLE_OPENCL:BOOL=ON
      -DENABLE_CUDA:BOOL=OFF
      -DENABLE_UBLAS:BOOL=OFF
      "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=-g -O0 -DCL_USE_DEPRECATED_OPENCL_1_1_APIS"
    INSTALL_COMMAND /bin/true
  )

  set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_dependencies(prepare_examples ${TS_NAME})

  add_test(NAME viennacl_examples_amg
          COMMAND "${TS_BUILDDIR}/examples/tutorial/amg"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
#  add_test(NAME viennacl_examples_bandwidth_reduction
#          COMMAND "${TS_BUILDDIR}/examples/tutorial/bandwidth-reduction"
#          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_examples_custom_kernels
          COMMAND "${TS_BUILDDIR}/examples/tutorial/custom-kernels"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_examples_blas1
          COMMAND "${TS_BUILDDIR}/examples/tutorial/blas1"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_examples_custom_context
          COMMAND "${TS_BUILDDIR}/examples/tutorial/custom-context"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_examples_bisect
          COMMAND "${TS_BUILDDIR}/examples/tutorial/bisect"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_examples_fft
          COMMAND "${TS_BUILDDIR}/examples/tutorial/fft"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_examples_iterative_custom
          COMMAND "${TS_BUILDDIR}/examples/tutorial/iterative-custom"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_examples_matrix_free
          COMMAND "${TS_BUILDDIR}/examples/tutorial/matrix-free"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_examples_nmf
          COMMAND "${TS_BUILDDIR}/examples/tutorial/nmf"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_examples_viennacl_info
          COMMAND "${TS_BUILDDIR}/examples/tutorial/viennacl-info"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_examples_wrap_host_buffer
          COMMAND "${TS_BUILDDIR}/examples/tutorial/wrap-host-buffer"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_examples_scheduler
          COMMAND "${TS_BUILDDIR}/examples/tutorial/scheduler"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_examples_libviennacl_tutorial
          COMMAND "${TS_BUILDDIR}/examples/tutorial/libviennacl-tutorial"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")

#  add_test(NAME viennacl_benchmarks_opencl_bench_opencl
#          COMMAND "${TS_BUILDDIR}/examples/benchmarks/opencl-bench-opencl"
#          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
#  add_test(NAME viennacl_benchmarks_dense_blas_bench_opencl
#          COMMAND "${TS_BUILDDIR}/examples/benchmarks/dense_blas-bench-opencl"
#          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")

#  add_test(NAME viennacl_tests_bisect_test_opencl
#          COMMAND "${TS_BUILDDIR}/tests/bisect-test-opencl"
#          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_fft_1d_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/fft_1d-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_external_linkage_opencl
          COMMAND "${TS_BUILDDIR}/tests/external_linkage-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_blas3_solve_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/blas3_solve-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_fft_2d_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/fft_2d-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_libviennacl_blas1_test
          COMMAND "${TS_BUILDDIR}/tests/libviennacl_blas1-test"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_iterators_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/iterators-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_global_variables_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/global_variables-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_libviennacl_blas2_test
          COMMAND "${TS_BUILDDIR}/tests/libviennacl_blas2-test"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_libviennacl_blas3_test
          COMMAND "${TS_BUILDDIR}/tests/libviennacl_blas3-test"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_matrix_col_double_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/matrix_col_double-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_matrix_col_float_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/matrix_col_float-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_matrix_col_int_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/matrix_col_int-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_matrix_product_double_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/matrix_product_double-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_matrix_convert_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/matrix_convert-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_matrix_product_float_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/matrix_product_float-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_matrix_row_double_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/matrix_row_double-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_matrix_row_float_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/matrix_row_float-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_matrix_row_int_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/matrix_row_int-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_matrix_vector_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/matrix_vector-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_nmf_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/nmf-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_scalar_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/scalar-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_matrix_vector_int_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/matrix_vector_int-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_qr_method_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/qr_method-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_qr_method_func_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/qr_method_func-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_scan_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/scan-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_self_assign_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/self_assign-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_sparse_prod_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/sparse_prod-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_sparse_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/sparse-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_structured_matrices_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/structured-matrices-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_spmdm_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/spmdm-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
#  add_test(NAME viennacl_tests_svd_test_opencl
#          COMMAND "${TS_BUILDDIR}/tests/svd-test-opencl"
#          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_tql_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/tql-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_vector_convert_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/vector_convert-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_vector_float_double_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/vector_float_double-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_vector_multi_inner_prod_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/vector_multi_inner_prod-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_vector_int_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/vector_int-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")
  add_test(NAME viennacl_tests_vector_uint_test_opencl
          COMMAND "${TS_BUILDDIR}/tests/vector_uint-test-opencl"
          WORKING_DIRECTORY "${TS_BUILDDIR}/tests")

#  viennacl_examples_bandwidth_reduction
  set_tests_properties(
  viennacl_examples_amg
  viennacl_examples_custom_kernels
  viennacl_examples_blas1
  viennacl_examples_custom_context
  viennacl_examples_bisect
  viennacl_examples_fft
  viennacl_examples_iterative_custom
  viennacl_examples_matrix_free
  viennacl_examples_nmf
  viennacl_examples_viennacl_info
  viennacl_examples_wrap_host_buffer
  viennacl_examples_scheduler
  viennacl_examples_libviennacl_tutorial
    PROPERTIES
      LABELS "ViennaCL ViennaCL_examples")

  # too slow
  # set_tests_properties(
  # viennacl_benchmarks_opencl_bench_opencl
  # viennacl_benchmarks_dense_blas_bench_opencl
  #  PROPERTIES
  #    LABELS "ViennaCL_benchmarks")

  set_tests_properties(
  # slow
  # viennacl_tests_bisect_test_opencl
  viennacl_tests_fft_1d_test_opencl
  viennacl_tests_external_linkage_opencl
  viennacl_tests_blas3_solve_test_opencl
  viennacl_tests_fft_2d_test_opencl
  viennacl_tests_libviennacl_blas1_test
  viennacl_tests_iterators_test_opencl
  viennacl_tests_global_variables_test_opencl
  viennacl_tests_libviennacl_blas2_test
  viennacl_tests_libviennacl_blas3_test
  viennacl_tests_matrix_col_double_test_opencl
  viennacl_tests_matrix_col_float_test_opencl
  viennacl_tests_matrix_col_int_test_opencl
  viennacl_tests_matrix_product_double_test_opencl
  viennacl_tests_matrix_convert_test_opencl
  viennacl_tests_matrix_product_float_test_opencl
  viennacl_tests_matrix_row_double_test_opencl
  viennacl_tests_matrix_row_float_test_opencl
  viennacl_tests_matrix_row_int_test_opencl
  viennacl_tests_matrix_vector_test_opencl
  viennacl_tests_nmf_test_opencl
  viennacl_tests_scalar_test_opencl
  viennacl_tests_matrix_vector_int_test_opencl
  viennacl_tests_qr_method_test_opencl
  viennacl_tests_qr_method_func_test_opencl
  viennacl_tests_scan_test_opencl
  viennacl_tests_self_assign_test_opencl
  viennacl_tests_sparse_prod_test_opencl
  viennacl_tests_sparse_test_opencl
  viennacl_tests_structured_matrices_test_opencl
  viennacl_tests_spmdm_test_opencl
  # slow
  # viennacl_tests_svd_test_opencl
  viennacl_tests_tql_test_opencl
  viennacl_tests_vector_convert_test_opencl
  viennacl_tests_vector_float_double_test_opencl
  viennacl_tests_vector_multi_inner_prod_test_opencl
  viennacl_tests_vector_int_test_opencl
  viennacl_tests_vector_uint_test_opencl
    PROPERTIES
      LABELS "ViennaCL ViennaCL_tests")


  # failing tests
  set_tests_properties(
    # Could not find a dominating alternative variable.
    viennacl_tests_sparse_prod_test_opencl
    viennacl_tests_spmdm_test_opencl
    viennacl_tests_sparse_test_opencl
    viennacl_examples_iterative_custom
    # used type 'float' where floating point type is not allowed
    viennacl_tests_self_assign_test_opencl
    # used type 'double' where floating point type is not allowed
    viennacl_examples_amg
    # Error! Difference: 1924149235
    viennacl_tests_vector_int_test_opencl
    viennacl_tests_vector_uint_test_opencl
    PROPERTIES
      WILL_FAIL 1)


else()

  message(STATUS "Disabling testsuite ${TS_NAME}, required files not found" )

endif()
