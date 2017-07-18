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

set(TS_NAME "arrayfire")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")

find_package(LAPACK QUIET)
find_package(Boost QUIET)

if(HAVE_GIT)

if(LAPACK_FOUND AND
   Boost_FOUND AND (NOT Boost_VERSION VERSION_LESS "1.48"))

  message(STATUS "Enabling testsuite ${TS_NAME}")
  list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
  set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

  ExternalProject_Add(
    ${TS_NAME}
    PREFIX "${TS_BASEDIR}"
    GIT_REPOSITORY "https://github.com/arrayfire/${TS_NAME}.git"
    #PATCH_COMMAND  pwd && sed -i "s/CL_DEVICE_TYPE_GPU/CL_DEVICE_TYPE_CPU/g" *.cpp
    #UPDATE_COMMAND /bin/true
    CMAKE_ARGS
      -DBUILD_GRAPHICS=OFF
      -DBUILD_CUDA=OFF
      -DBUILD_CPU=OFF
      -DBUILD_OPENCL=ON
      -DCMAKE_BUILD_TYPE=RelWithDebInfo
      "-DCMAKE_C_FLAGS_RELWITHDEBINFO=-O2 -g -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS"
      "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -g -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS"
    INSTALL_COMMAND /bin/true
  )

  set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_dependencies(prepare_examples ${TS_NAME})

  add_test(NAME arrayfire_tests_approx1_opencl
           COMMAND "${TS_BUILDDIR}/test/approx1_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_approx2_opencl
           COMMAND "${TS_BUILDDIR}/test/approx2_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_array_opencl
           COMMAND "${TS_BUILDDIR}/test/array_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_assign_opencl
           COMMAND "${TS_BUILDDIR}/test/assign_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_backend_opencl
           COMMAND "${TS_BUILDDIR}/test/backend_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_basic_c_opencl
           COMMAND "${TS_BUILDDIR}/test/basic_c_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_basic_opencl
           COMMAND "${TS_BUILDDIR}/test/basic_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_bilateral_opencl
           COMMAND "${TS_BUILDDIR}/test/bilateral_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_binary_opencl
           COMMAND "${TS_BUILDDIR}/test/binary_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_blas_opencl
           COMMAND "${TS_BUILDDIR}/test/blas_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_cholesky_dense_opencl
           COMMAND "${TS_BUILDDIR}/test/cholesky_dense_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_complex_opencl
           COMMAND "${TS_BUILDDIR}/test/complex_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_constant_opencl
           COMMAND "${TS_BUILDDIR}/test/constant_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_convolve_opencl
           COMMAND "${TS_BUILDDIR}/test/convolve_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_corrcoef_opencl
           COMMAND "${TS_BUILDDIR}/test/corrcoef_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_covariance_opencl
           COMMAND "${TS_BUILDDIR}/test/covariance_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_diagonal_opencl
           COMMAND "${TS_BUILDDIR}/test/diagonal_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_diff1_opencl
           COMMAND "${TS_BUILDDIR}/test/diff1_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_diff2_opencl
           COMMAND "${TS_BUILDDIR}/test/diff2_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_dog_opencl
           COMMAND "${TS_BUILDDIR}/test/dog_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_dot_opencl
           COMMAND "${TS_BUILDDIR}/test/dot_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_fast_opencl
           COMMAND "${TS_BUILDDIR}/test/fast_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_fft_large_opencl
           COMMAND "${TS_BUILDDIR}/test/fft_large_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_fft_opencl
           COMMAND "${TS_BUILDDIR}/test/fft_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_fft_real_opencl
           COMMAND "${TS_BUILDDIR}/test/fft_real_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_fftconvolve_opencl
           COMMAND "${TS_BUILDDIR}/test/fftconvolve_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_flat_opencl
           COMMAND "${TS_BUILDDIR}/test/flat_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_flip_opencl
           COMMAND "${TS_BUILDDIR}/test/flip_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_gaussiankernel_opencl
           COMMAND "${TS_BUILDDIR}/test/gaussiankernel_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_gen_assign_opencl
           COMMAND "${TS_BUILDDIR}/test/gen_assign_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_gen_index_opencl
           COMMAND "${TS_BUILDDIR}/test/gen_index_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_getting_started_opencl
           COMMAND "${TS_BUILDDIR}/test/getting_started_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_gfor_opencl
           COMMAND "${TS_BUILDDIR}/test/gfor_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_gloh_nonfree_opencl
           COMMAND "${TS_BUILDDIR}/test/gloh_nonfree_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_gradient_opencl
           COMMAND "${TS_BUILDDIR}/test/gradient_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_hamming_opencl
           COMMAND "${TS_BUILDDIR}/test/hamming_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_homography_opencl
           COMMAND "${TS_BUILDDIR}/test/homography_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_hsv_rgb_opencl
           COMMAND "${TS_BUILDDIR}/test/hsv_rgb_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_iir_opencl
           COMMAND "${TS_BUILDDIR}/test/iir_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_imageio_opencl
           COMMAND "${TS_BUILDDIR}/test/imageio_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_index_opencl
           COMMAND "${TS_BUILDDIR}/test/index_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_info_opencl
           COMMAND "${TS_BUILDDIR}/test/info_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_inverse_dense_opencl
           COMMAND "${TS_BUILDDIR}/test/inverse_dense_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_iota_opencl
           COMMAND "${TS_BUILDDIR}/test/iota_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_ireduce_opencl
           COMMAND "${TS_BUILDDIR}/test/ireduce_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_jit_opencl
           COMMAND "${TS_BUILDDIR}/test/jit_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_join_opencl
           COMMAND "${TS_BUILDDIR}/test/join_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_lu_dense_opencl
           COMMAND "${TS_BUILDDIR}/test/lu_dense_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_manual_memory_test_opencl
           COMMAND "${TS_BUILDDIR}/test/manual_memory_test_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_match_template_opencl
           COMMAND "${TS_BUILDDIR}/test/match_template_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_math_opencl
           COMMAND "${TS_BUILDDIR}/test/math_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_matrix_manipulation_opencl
           COMMAND "${TS_BUILDDIR}/test/matrix_manipulation_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_mean_opencl
           COMMAND "${TS_BUILDDIR}/test/mean_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_meanshift_opencl
           COMMAND "${TS_BUILDDIR}/test/meanshift_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_medfilt_opencl
           COMMAND "${TS_BUILDDIR}/test/medfilt_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
# takes too long
#  add_test(NAME arrayfire_tests_median_opencl
#           COMMAND "${TS_BUILDDIR}/test/median_opencl"
#           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_memory_lock_opencl
           COMMAND "${TS_BUILDDIR}/test/memory_lock_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_memory_opencl
           COMMAND "${TS_BUILDDIR}/test/memory_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_missing_opencl
           COMMAND "${TS_BUILDDIR}/test/missing_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_moddims_opencl
           COMMAND "${TS_BUILDDIR}/test/moddims_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_morph_opencl
           COMMAND "${TS_BUILDDIR}/test/morph_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_nearest_neighbour_opencl
           COMMAND "${TS_BUILDDIR}/test/nearest_neighbour_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_orb_opencl
           COMMAND "${TS_BUILDDIR}/test/orb_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_qr_dense_opencl
           COMMAND "${TS_BUILDDIR}/test/qr_dense_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_random_opencl
           COMMAND "${TS_BUILDDIR}/test/random_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_rank_dense_opencl
           COMMAND "${TS_BUILDDIR}/test/rank_dense_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_range_opencl
           COMMAND "${TS_BUILDDIR}/test/range_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_regions_opencl
           COMMAND "${TS_BUILDDIR}/test/regions_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_reduce_opencl
           COMMAND "${TS_BUILDDIR}/test/reduce_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_replace_opencl
           COMMAND "${TS_BUILDDIR}/test/replace_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_reorder_opencl
           COMMAND "${TS_BUILDDIR}/test/reorder_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_resize_opencl
           COMMAND "${TS_BUILDDIR}/test/resize_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_sat_opencl
           COMMAND "${TS_BUILDDIR}/test/sat_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_scan_opencl
           COMMAND "${TS_BUILDDIR}/test/scan_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_rotate_linear_opencl
           COMMAND "${TS_BUILDDIR}/test/rotate_linear_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_rotate_opencl
           COMMAND "${TS_BUILDDIR}/test/rotate_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_select_opencl
           COMMAND "${TS_BUILDDIR}/test/select_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_set_opencl
           COMMAND "${TS_BUILDDIR}/test/set_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_sift_nonfree_opencl
           COMMAND "${TS_BUILDDIR}/test/sift_nonfree_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_sobel_opencl
           COMMAND "${TS_BUILDDIR}/test/sobel_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_solve_dense_opencl
           COMMAND "${TS_BUILDDIR}/test/solve_dense_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_shift_opencl
           COMMAND "${TS_BUILDDIR}/test/shift_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_sort_by_key_opencl
           COMMAND "${TS_BUILDDIR}/test/sort_by_key_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_sort_index_opencl
           COMMAND "${TS_BUILDDIR}/test/sort_index_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_sort_opencl
           COMMAND "${TS_BUILDDIR}/test/sort_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_stdev_opencl
           COMMAND "${TS_BUILDDIR}/test/stdev_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_susan_opencl
           COMMAND "${TS_BUILDDIR}/test/susan_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_svd_dense_opencl
           COMMAND "${TS_BUILDDIR}/test/svd_dense_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_tile_opencl
           COMMAND "${TS_BUILDDIR}/test/tile_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_translate_opencl
           COMMAND "${TS_BUILDDIR}/test/translate_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_transpose_inplace_opencl
           COMMAND "${TS_BUILDDIR}/test/transpose_inplace_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_transpose_opencl
           COMMAND "${TS_BUILDDIR}/test/transpose_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_triangle_opencl
           COMMAND "${TS_BUILDDIR}/test/triangle_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_var_opencl
           COMMAND "${TS_BUILDDIR}/test/var_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_where_opencl
           COMMAND "${TS_BUILDDIR}/test/where_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_unwrap_opencl
           COMMAND "${TS_BUILDDIR}/test/unwrap_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_write_opencl
           COMMAND "${TS_BUILDDIR}/test/write_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_ycbcr_rgb_opencl
           COMMAND "${TS_BUILDDIR}/test/ycbcr_rgb_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  add_test(NAME arrayfire_tests_wrap_opencl
           COMMAND "${TS_BUILDDIR}/test/wrap_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/test")


  add_test(NAME arrayfire_examples_adaptive_thresholding_opencl
           COMMAND "${TS_BUILDDIR}/examples/image_processing/adaptive_thresholding_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/image_processing")
  add_test(NAME arrayfire_examples_binary_thresholding_opencl
           COMMAND "${TS_BUILDDIR}/examples/image_processing/binary_thresholding_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/image_processing")
  add_test(NAME arrayfire_examples_brain_segmentation_opencl
           COMMAND "${TS_BUILDDIR}/examples/image_processing/brain_segmentation_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/image_processing")
  add_test(NAME arrayfire_examples_filters_opencl
           COMMAND "${TS_BUILDDIR}/examples/image_processing/filters_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/image_processing")
# these tests are commented out b/c they take too much time on CPU.
#  add_test(NAME arrayfire_examples_deep_belief_net_opencl
#           COMMAND "${TS_BUILDDIR}/examples/machine_learning/deep_belief_net_opencl"
#           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/machine_learning")
#  add_test(NAME arrayfire_examples_kmeans_opencl
#           COMMAND "${TS_BUILDDIR}/examples/machine_learning/kmeans_opencl"
#           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/machine_learning")
  add_test(NAME arrayfire_examples_knn_opencl
           COMMAND "${TS_BUILDDIR}/examples/machine_learning/knn_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/machine_learning")
#  add_test(NAME arrayfire_examples_logistic_regression_opencl
#           COMMAND "${TS_BUILDDIR}/examples/machine_learning/logistic_regression_opencl"
#           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/machine_learning")
  add_test(NAME arrayfire_examples_naive_bayes_opencl
           COMMAND "${TS_BUILDDIR}/examples/machine_learning/naive_bayes_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/machine_learning")
#  add_test(NAME arrayfire_examples_neural_network_opencl
#           COMMAND "${TS_BUILDDIR}/examples/machine_learning/neural_network_opencl"
#           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/machine_learning")
#  add_test(NAME arrayfire_examples_perceptron_opencl
#           COMMAND "${TS_BUILDDIR}/examples/machine_learning/perceptron_opencl"
#           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/machine_learning")
#  add_test(NAME arrayfire_examples_rbm_opencl
#           COMMAND "${TS_BUILDDIR}/examples/machine_learning/rbm_opencl"
#           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/machine_learning")
#  add_test(NAME arrayfire_examples_softmax_regression_opencl
#           COMMAND "${TS_BUILDDIR}/examples/machine_learning/softmax_regression_opencl"
#           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/machine_learning")
  add_test(NAME arrayfire_examples_basic_opencl
           COMMAND "${TS_BUILDDIR}/examples/unified/basic_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/unified")
  add_test(NAME arrayfire_examples_black_scholes_options_opencl
           COMMAND "${TS_BUILDDIR}/examples/financial/black_scholes_options_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/financial")
  add_test(NAME arrayfire_examples_heston_model_opencl
           COMMAND "${TS_BUILDDIR}/examples/financial/heston_model_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/financial")
  add_test(NAME arrayfire_examples_monte_carlo_options_opencl
           COMMAND "${TS_BUILDDIR}/examples/financial/monte_carlo_options_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/financial")
  add_test(NAME arrayfire_examples_blas_opencl
           COMMAND "${TS_BUILDDIR}/examples/benchmarks/blas_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/benchmarks")
  add_test(NAME arrayfire_examples_fft_opencl
           COMMAND "${TS_BUILDDIR}/examples/benchmarks/fft_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/benchmarks")
  add_test(NAME arrayfire_examples_pi_opencl
           COMMAND "${TS_BUILDDIR}/examples/benchmarks/pi_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/benchmarks")
  add_test(NAME arrayfire_examples_cholesky_opencl
           COMMAND "${TS_BUILDDIR}/examples/lin_algebra/cholesky_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/lin_algebra")
  add_test(NAME arrayfire_examples_lu_opencl
           COMMAND "${TS_BUILDDIR}/examples/lin_algebra/lu_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/lin_algebra")
  add_test(NAME arrayfire_examples_qr_opencl
           COMMAND "${TS_BUILDDIR}/examples/lin_algebra/qr_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/lin_algebra")
  add_test(NAME arrayfire_examples_svd_opencl
           COMMAND "${TS_BUILDDIR}/examples/lin_algebra/svd_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/lin_algebra")
  add_test(NAME arrayfire_examples_convolve_opencl
           COMMAND "${TS_BUILDDIR}/examples/getting_started/convolve_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/getting_started")
  add_test(NAME arrayfire_examples_integer_opencl
           COMMAND "${TS_BUILDDIR}/examples/getting_started/integer_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/getting_started")
  add_test(NAME arrayfire_examples_rainfall_opencl
           COMMAND "${TS_BUILDDIR}/examples/getting_started/rainfall_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/getting_started")
  add_test(NAME arrayfire_examples_vectorize_opencl
           COMMAND "${TS_BUILDDIR}/examples/getting_started/vectorize_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/getting_started")
  add_test(NAME arrayfire_examples_histogram_opencl
           COMMAND "${TS_BUILDDIR}/examples/graphics/histogram_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/graphics")
  add_test(NAME arrayfire_examples_fast_opencl
           COMMAND "${TS_BUILDDIR}/examples/computer_vision/fast_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/computer_vision")
  add_test(NAME arrayfire_examples_matching_opencl
           COMMAND "${TS_BUILDDIR}/examples/computer_vision/matching_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/computer_vision")
  add_test(NAME arrayfire_examples_susan_opencl
           COMMAND "${TS_BUILDDIR}/examples/computer_vision/susan_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/computer_vision")
  add_test(NAME arrayfire_examples_helloworld_opencl
           COMMAND "${TS_BUILDDIR}/examples/helloworld/helloworld_opencl"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/helloworld")

  if (IMAGE_TESTS_ENABLED)
    set (IMAGE_TESTS_LIST
        arrayfire_examples_edge_opencl
        arrayfire_examples_plot2d_opencl
        arrayfire_examples_plot3_opencl
        arrayfire_examples_image_demo_opencl
        arrayfire_examples_image_editing_opencl
        arrayfire_examples_morphing_opencl
        arrayfire_examples_optical_flow_opencl
        arrayfire_examples_pyramids_opencl
        arrayfire_examples_bagging_opencl
        arrayfire_examples_conway_opencl
        arrayfire_examples_conway_pretty_opencl
        arrayfire_examples_fractal_opencl
        arrayfire_examples_histogram_opencl
        arrayfire_examples_surface_opencl
        arrayfire_examples_harris_opencl
        arrayfire_examples_swe_opencl
    )
    add_test(NAME arrayfire_examples_edge_opencl
             COMMAND "${TS_BUILDDIR}/examples/image_processing/edge_opencl"
             WORKING_DIRECTORY "${TS_BUILDDIR}/examples/image_processing")
    add_test(NAME arrayfire_examples_plot2d_opencl
             COMMAND "${TS_BUILDDIR}/examples/graphics/plot2d_opencl"
             WORKING_DIRECTORY "${TS_BUILDDIR}/examples/graphics")
    add_test(NAME arrayfire_examples_plot3_opencl
             COMMAND "${TS_BUILDDIR}/examples/graphics/plot3_opencl"
             WORKING_DIRECTORY "${TS_BUILDDIR}/examples/graphics")
    add_test(NAME arrayfire_examples_image_demo_opencl
             COMMAND "${TS_BUILDDIR}/examples/image_processing/image_demo_opencl"
             WORKING_DIRECTORY "${TS_BUILDDIR}/examples/image_processing")
    add_test(NAME arrayfire_examples_image_editing_opencl
             COMMAND "${TS_BUILDDIR}/examples/image_processing/image_editing_opencl"
             WORKING_DIRECTORY "${TS_BUILDDIR}/examples/image_processing")
    add_test(NAME arrayfire_examples_morphing_opencl
             COMMAND "${TS_BUILDDIR}/examples/image_processing/morphing_opencl"
             WORKING_DIRECTORY "${TS_BUILDDIR}/examples/image_processing")
    add_test(NAME arrayfire_examples_optical_flow_opencl
             COMMAND "${TS_BUILDDIR}/examples/image_processing/optical_flow_opencl"
             WORKING_DIRECTORY "${TS_BUILDDIR}/examples/image_processing")
    add_test(NAME arrayfire_examples_pyramids_opencl
             COMMAND "${TS_BUILDDIR}/examples/image_processing/pyramids_opencl"
             WORKING_DIRECTORY "${TS_BUILDDIR}/examples/image_processing")
    add_test(NAME arrayfire_examples_bagging_opencl
             COMMAND "${TS_BUILDDIR}/examples/machine_learning/bagging_opencl"
             WORKING_DIRECTORY "${TS_BUILDDIR}/examples/machine_learning")
    add_test(NAME arrayfire_examples_conway_opencl
             COMMAND "${TS_BUILDDIR}/examples/graphics/conway_opencl"
             WORKING_DIRECTORY "${TS_BUILDDIR}/examples/graphics")
    add_test(NAME arrayfire_examples_conway_pretty_opencl
             COMMAND "${TS_BUILDDIR}/examples/graphics/conway_pretty_opencl"
             WORKING_DIRECTORY "${TS_BUILDDIR}/examples/graphics")
    add_test(NAME arrayfire_examples_fractal_opencl
             COMMAND "${TS_BUILDDIR}/examples/graphics/fractal_opencl"
             WORKING_DIRECTORY "${TS_BUILDDIR}/examples/graphics")
    add_test(NAME arrayfire_tests_histogram_opencl
             COMMAND "${TS_BUILDDIR}/test/histogram_opencl"
             WORKING_DIRECTORY "${TS_BUILDDIR}/test")
    add_test(NAME arrayfire_tests_harris_opencl
             COMMAND "${TS_BUILDDIR}/test/harris_opencl"
             WORKING_DIRECTORY "${TS_BUILDDIR}/test")
    add_test(NAME arrayfire_examples_surface_opencl
             COMMAND "${TS_BUILDDIR}/examples/graphics/surface_opencl"
             WORKING_DIRECTORY "${TS_BUILDDIR}/examples/graphics")
    add_test(NAME arrayfire_examples_swe_opencl
             COMMAND "${TS_BUILDDIR}/examples/pde/swe_opencl"
             WORKING_DIRECTORY "${TS_BUILDDIR}/examples/pde")
  else()
    unset (IMAGE_TESTS_LIST)
  endif()

  set_tests_properties(
    arrayfire_tests_approx1_opencl
    arrayfire_tests_approx2_opencl
    arrayfire_tests_array_opencl
    arrayfire_tests_assign_opencl
    arrayfire_tests_backend_opencl
    arrayfire_tests_basic_c_opencl
    arrayfire_tests_basic_opencl
    arrayfire_tests_bilateral_opencl
    arrayfire_tests_binary_opencl
    arrayfire_tests_blas_opencl
    arrayfire_tests_cholesky_dense_opencl
    arrayfire_tests_complex_opencl
    arrayfire_tests_constant_opencl
    arrayfire_tests_convolve_opencl
    arrayfire_tests_corrcoef_opencl
    arrayfire_tests_covariance_opencl
    arrayfire_tests_diagonal_opencl
    arrayfire_tests_diff1_opencl
    arrayfire_tests_diff2_opencl
    arrayfire_tests_dog_opencl
    arrayfire_tests_dot_opencl
    arrayfire_tests_fast_opencl
    arrayfire_tests_fft_large_opencl
    arrayfire_tests_fft_opencl
    arrayfire_tests_fft_real_opencl
    arrayfire_tests_fftconvolve_opencl
    arrayfire_tests_flat_opencl
    arrayfire_tests_flip_opencl
    arrayfire_tests_gaussiankernel_opencl
    arrayfire_tests_gen_assign_opencl
    arrayfire_tests_gen_index_opencl
    arrayfire_tests_getting_started_opencl
    arrayfire_tests_gfor_opencl
    arrayfire_tests_gloh_nonfree_opencl
    arrayfire_tests_gradient_opencl
    arrayfire_tests_hamming_opencl
    arrayfire_tests_homography_opencl
    arrayfire_tests_hsv_rgb_opencl
    arrayfire_tests_iir_opencl
    arrayfire_tests_imageio_opencl
    arrayfire_tests_index_opencl
    arrayfire_tests_info_opencl
    arrayfire_tests_inverse_dense_opencl
    arrayfire_tests_iota_opencl
    arrayfire_tests_ireduce_opencl
    arrayfire_tests_jit_opencl
    arrayfire_tests_join_opencl
    arrayfire_tests_lu_dense_opencl
    arrayfire_tests_manual_memory_test_opencl
    arrayfire_tests_match_template_opencl
    arrayfire_tests_math_opencl
    arrayfire_tests_matrix_manipulation_opencl
    arrayfire_tests_mean_opencl
    arrayfire_tests_meanshift_opencl
    arrayfire_tests_medfilt_opencl
#    arrayfire_tests_median_opencl
    arrayfire_tests_memory_lock_opencl
    arrayfire_tests_memory_opencl
    arrayfire_tests_missing_opencl
    arrayfire_tests_moddims_opencl
    arrayfire_tests_morph_opencl
    arrayfire_tests_nearest_neighbour_opencl
#    arrayfire_tests_ocl_ext_context_opencl
    arrayfire_tests_orb_opencl
    arrayfire_tests_qr_dense_opencl
    arrayfire_tests_random_opencl
    arrayfire_tests_rank_dense_opencl
    arrayfire_tests_range_opencl
    arrayfire_tests_regions_opencl
    arrayfire_tests_reduce_opencl
    arrayfire_tests_replace_opencl
    arrayfire_tests_reorder_opencl
    arrayfire_tests_resize_opencl
    arrayfire_tests_sat_opencl
    arrayfire_tests_scan_opencl
    arrayfire_tests_rotate_linear_opencl
    arrayfire_tests_rotate_opencl
    arrayfire_tests_select_opencl
    arrayfire_tests_set_opencl
    arrayfire_tests_sift_nonfree_opencl
    arrayfire_tests_sobel_opencl
    arrayfire_tests_solve_dense_opencl
    arrayfire_tests_shift_opencl
    arrayfire_tests_sort_by_key_opencl
    arrayfire_tests_sort_index_opencl
    arrayfire_tests_sort_opencl
    arrayfire_tests_stdev_opencl
    arrayfire_tests_susan_opencl
    arrayfire_tests_svd_dense_opencl
#    arrayfire_tests_transform_coordinates_opencl
#    arrayfire_tests_transform_opencl
    arrayfire_tests_tile_opencl
    arrayfire_tests_translate_opencl
    arrayfire_tests_transpose_inplace_opencl
    arrayfire_tests_transpose_opencl
    arrayfire_tests_triangle_opencl
    arrayfire_tests_var_opencl
    arrayfire_tests_where_opencl
    arrayfire_tests_unwrap_opencl
    arrayfire_tests_write_opencl
    arrayfire_tests_ycbcr_rgb_opencl
    arrayfire_tests_wrap_opencl
    PROPERTIES
      LABELS "${TS_NAME} ${TS_NAME}_tests")


  set_tests_properties(
    ${IMAGE_TESTS_LIST}
    arrayfire_examples_adaptive_thresholding_opencl
    arrayfire_examples_binary_thresholding_opencl
    arrayfire_examples_brain_segmentation_opencl
    arrayfire_examples_filters_opencl
#    arrayfire_examples_deep_belief_net_opencl
#    arrayfire_examples_kmeans_opencl
    arrayfire_examples_knn_opencl
#    arrayfire_examples_logistic_regression_opencl
    arrayfire_examples_naive_bayes_opencl
#    arrayfire_examples_neural_network_opencl
#    arrayfire_examples_perceptron_opencl
#    arrayfire_examples_rbm_opencl
#    arrayfire_examples_softmax_regression_opencl
    arrayfire_examples_basic_opencl
    arrayfire_examples_black_scholes_options_opencl
    arrayfire_examples_heston_model_opencl
    arrayfire_examples_monte_carlo_options_opencl
    arrayfire_examples_blas_opencl
    arrayfire_examples_fft_opencl
    arrayfire_examples_pi_opencl
    arrayfire_examples_cholesky_opencl
    arrayfire_examples_lu_opencl
    arrayfire_examples_qr_opencl
    arrayfire_examples_svd_opencl
    arrayfire_examples_convolve_opencl
    arrayfire_examples_integer_opencl
    arrayfire_examples_rainfall_opencl
    arrayfire_examples_vectorize_opencl
#    arrayfire_examples_gravity_sim_opencl
    arrayfire_examples_fast_opencl
    arrayfire_examples_matching_opencl
    arrayfire_examples_susan_opencl
    arrayfire_examples_helloworld_opencl
    PROPERTIES
      LABELS "${TS_NAME} ${TS_NAME}_examples")


else()
  message(STATUS "Disabling testsuite ${TS_NAME}, required files (Boost/LAPACK) not found" )
endif()
else()
  message(STATUS "Disabling testsuite ${TS_NAME}, requires git to checkout sources")
endif()

# disabled tests / examples

  #~ add_test(NAME arrayfire_tests_ocl_ext_context_opencl
           #~ COMMAND "${TS_BUILDDIR}/test/ocl_ext_context_opencl"
           #~ WORKING_DIRECTORY "${TS_BUILDDIR}/test")

  #~ add_test(NAME arrayfire_tests_transform_opencl
           #~ COMMAND "${TS_BUILDDIR}/test/transform_opencl"
           #~ WORKING_DIRECTORY "${TS_BUILDDIR}/test")
  #~ add_test(NAME arrayfire_tests_transform_coordinates_opencl
           #~ COMMAND "${TS_BUILDDIR}/test/transform_coordinates_opencl"
           #~ WORKING_DIRECTORY "${TS_BUILDDIR}/test")


  #~ add_test(NAME arrayfire_examples_gravity_sim_opencl
           #~ COMMAND "${TS_BUILDDIR}/examples/graphics/gravity_sim_opencl"
           #~ WORKING_DIRECTORY "${TS_BUILDDIR}/examples/graphics")
