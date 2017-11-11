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

set(TS_NAME Halide)
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")

# Halide disabled for now, need to find out which tests actually use OpenCL
if(0)

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

if(EXAMPLES_USE_GIT_MASTER)
  set(FETCH_SOURCE GIT_REPOSITORY "https://github.com/Halide/Halide.git")
else()
  set(FETCH_SOURCE URL "https://github.com/halide/Halide/archive/release_2017_10_30.tar.gz")
endif()

ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  ${FETCH_SOURCE}
  #PATCH_COMMAND /bin/sh "${AMD_APP_SDK_TGZ}" --noexec --keep --target AMD-APP-SDK-3.0 &&
  #     patch -p1 -i ${CMAKE_CURRENT_SOURCE_DIR}/amdsdk3_0.patch
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DTARGET_ARM=OFF
    -DTARGET_AARCH64=OFF
    -DTARGET_MIPS=OFF
    -DTARGET_METAL=OFF
    -DTARGET_OPENCL=ON
    -DTARGET_OPENGL=OFF
    -DTARGET_OPENGLCOMPUTE=OFF
    -DTARGET_PTX=OFF
    -DTARGET_X86=ON
  CMAKE_CACHE_ARGS
    "-DLLVM_BIN:STRING=${LLVM_BINDIR}"
    "-DLLVM_LIB:STRING=${LLVM_LIBDIR}"
    "-DLLVM_INCLUDE:STRING=${LLVM_INCLUDEDIR}"
    "-DLLVM_VERSION:STRING=${LLVM_MAJOR}${LLVM_MINOR}"
  INSTALL_COMMAND "/bin/true"
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

# TODO Probably not tests
# acquire_release.generator argvcall.generator cleanup_on_error.generator
# embed_image.generator example.generator extended_buffer_t.generator
# error_codes.generator gpu_object_lifetime.generator gpu_only.generator
# mandelbrot.generator matlab.generator metadata_tester.generator
# nested_externs.generator paramtest.generator pyramid.generator
# runtime.generator tiled_blur.generator tiled_blur_blur.generator
# user_context.generator user_context_insanity.generator

add_test(NAME halide_bilateral_grid
         COMMAND "${TS_BUILDDIR}/bin/bilateral_grid")
add_test(NAME halide_bitcode2cpp
         COMMAND "${TS_BUILDDIR}/bin/bitcode2cpp")
add_test(NAME halide_blur_test
         COMMAND "${TS_BUILDDIR}/bin/blur_test")
add_test(NAME halide_build_halide_h
         COMMAND "${TS_BUILDDIR}/bin/build_halide_h")
add_test(NAME halide_camera_pipe
         COMMAND "${TS_BUILDDIR}/bin/camera_pipe")
add_test(NAME halide_correctness_argmax
         COMMAND "${TS_BUILDDIR}/bin/correctness_argmax")
add_test(NAME halide_correctness_assertion_failure_in_parallel_for
         COMMAND "${TS_BUILDDIR}/bin/correctness_assertion_failure_in_parallel_for")
add_test(NAME halide_correctness_autotune_bug
         COMMAND "${TS_BUILDDIR}/bin/correctness_autotune_bug")
add_test(NAME halide_correctness_autotune_bug_2
         COMMAND "${TS_BUILDDIR}/bin/correctness_autotune_bug_2")
add_test(NAME halide_correctness_autotune_bug_3
         COMMAND "${TS_BUILDDIR}/bin/correctness_autotune_bug_3")
add_test(NAME halide_correctness_autotune_bug_4
         COMMAND "${TS_BUILDDIR}/bin/correctness_autotune_bug_4")
add_test(NAME halide_correctness_autotune_bug_5
         COMMAND "${TS_BUILDDIR}/bin/correctness_autotune_bug_5")
add_test(NAME halide_correctness_bad_likely
         COMMAND "${TS_BUILDDIR}/bin/correctness_bad_likely")
add_test(NAME halide_correctness_bit_counting
         COMMAND "${TS_BUILDDIR}/bin/correctness_bit_counting")
add_test(NAME halide_correctness_bitwise_ops
         COMMAND "${TS_BUILDDIR}/bin/correctness_bitwise_ops")
add_test(NAME halide_correctness_bool_compute_root_vectorize
         COMMAND "${TS_BUILDDIR}/bin/correctness_bool_compute_root_vectorize")
add_test(NAME halide_correctness_bound
         COMMAND "${TS_BUILDDIR}/bin/correctness_bound")
add_test(NAME halide_correctness_boundary_conditions
         COMMAND "${TS_BUILDDIR}/bin/correctness_boundary_conditions")
add_test(NAME halide_correctness_bounds
         COMMAND "${TS_BUILDDIR}/bin/correctness_bounds")
add_test(NAME halide_correctness_bounds_inference
         COMMAND "${TS_BUILDDIR}/bin/correctness_bounds_inference")
add_test(NAME halide_correctness_bounds_inference_chunk
         COMMAND "${TS_BUILDDIR}/bin/correctness_bounds_inference_chunk")
add_test(NAME halide_correctness_bounds_inference_complex
         COMMAND "${TS_BUILDDIR}/bin/correctness_bounds_inference_complex")
add_test(NAME halide_correctness_bounds_of_abs
         COMMAND "${TS_BUILDDIR}/bin/correctness_bounds_of_abs")
add_test(NAME halide_correctness_bounds_of_cast
         COMMAND "${TS_BUILDDIR}/bin/correctness_bounds_of_cast")
add_test(NAME halide_correctness_bounds_of_func
         COMMAND "${TS_BUILDDIR}/bin/correctness_bounds_of_func")
add_test(NAME halide_correctness_bounds_of_monotonic_math
         COMMAND "${TS_BUILDDIR}/bin/correctness_bounds_of_monotonic_math")
add_test(NAME halide_correctness_bounds_query
         COMMAND "${TS_BUILDDIR}/bin/correctness_bounds_query")
add_test(NAME halide_correctness_buffer_t
         COMMAND "${TS_BUILDDIR}/bin/correctness_buffer_t")
add_test(NAME halide_correctness_cascaded_filters
         COMMAND "${TS_BUILDDIR}/bin/correctness_cascaded_filters")
add_test(NAME halide_correctness_cast
         COMMAND "${TS_BUILDDIR}/bin/correctness_cast")
add_test(NAME halide_correctness_cast_handle
         COMMAND "${TS_BUILDDIR}/bin/correctness_cast_handle")
add_test(NAME halide_correctness_c_function
         COMMAND "${TS_BUILDDIR}/bin/correctness_c_function")
add_test(NAME halide_correctness_chunk
         COMMAND "${TS_BUILDDIR}/bin/correctness_chunk")
add_test(NAME halide_correctness_chunk_sharing
         COMMAND "${TS_BUILDDIR}/bin/correctness_chunk_sharing")
add_test(NAME halide_correctness_circular_reference_leak
         COMMAND "${TS_BUILDDIR}/bin/correctness_circular_reference_leak")
add_test(NAME halide_correctness_code_explosion
         COMMAND "${TS_BUILDDIR}/bin/correctness_code_explosion")
add_test(NAME halide_correctness_compare_vars
         COMMAND "${TS_BUILDDIR}/bin/correctness_compare_vars")
add_test(NAME halide_correctness_compile_to
         COMMAND "${TS_BUILDDIR}/bin/correctness_compile_to")
add_test(NAME halide_correctness_compile_to_bitcode
         COMMAND "${TS_BUILDDIR}/bin/correctness_compile_to_bitcode")
add_test(NAME halide_correctness_compile_to_lowered_stmt
         COMMAND "${TS_BUILDDIR}/bin/correctness_compile_to_lowered_stmt")
add_test(NAME halide_correctness_compute_at_split_rvar
         COMMAND "${TS_BUILDDIR}/bin/correctness_compute_at_split_rvar")
add_test(NAME halide_correctness_computed_index
         COMMAND "${TS_BUILDDIR}/bin/correctness_computed_index")
add_test(NAME halide_correctness_compute_outermost
         COMMAND "${TS_BUILDDIR}/bin/correctness_compute_outermost")
add_test(NAME halide_correctness_constant_expr
         COMMAND "${TS_BUILDDIR}/bin/correctness_constant_expr")
add_test(NAME halide_correctness_constant_type
         COMMAND "${TS_BUILDDIR}/bin/correctness_constant_type")
add_test(NAME halide_correctness_constraints
         COMMAND "${TS_BUILDDIR}/bin/correctness_constraints")
add_test(NAME halide_correctness_convolution
         COMMAND "${TS_BUILDDIR}/bin/correctness_convolution")
add_test(NAME halide_correctness_convolution_multiple_kernels
         COMMAND "${TS_BUILDDIR}/bin/correctness_convolution_multiple_kernels")
add_test(NAME halide_correctness_cross_compilation
         COMMAND "${TS_BUILDDIR}/bin/correctness_cross_compilation")
add_test(NAME halide_correctness_custom_allocator
         COMMAND "${TS_BUILDDIR}/bin/correctness_custom_allocator")
add_test(NAME halide_correctness_custom_error_reporter
         COMMAND "${TS_BUILDDIR}/bin/correctness_custom_error_reporter")
add_test(NAME halide_correctness_custom_lowering_pass
         COMMAND "${TS_BUILDDIR}/bin/correctness_custom_lowering_pass")
add_test(NAME halide_correctness_debug_to_file
         COMMAND "${TS_BUILDDIR}/bin/correctness_debug_to_file")
add_test(NAME halide_correctness_deinterleave4
         COMMAND "${TS_BUILDDIR}/bin/correctness_deinterleave4")
add_test(NAME halide_correctness_div_mod
         COMMAND "${TS_BUILDDIR}/bin/correctness_div_mod")
add_test(NAME halide_correctness_dynamic_reduction_bounds
         COMMAND "${TS_BUILDDIR}/bin/correctness_dynamic_reduction_bounds")
add_test(NAME halide_correctness_erf
         COMMAND "${TS_BUILDDIR}/bin/correctness_erf")
add_test(NAME halide_correctness_exception
         COMMAND "${TS_BUILDDIR}/bin/correctness_exception")
add_test(NAME halide_correctness_explicit_inline_reductions
         COMMAND "${TS_BUILDDIR}/bin/correctness_explicit_inline_reductions")
add_test(NAME halide_correctness_extern_bounds_inference
         COMMAND "${TS_BUILDDIR}/bin/correctness_extern_bounds_inference")
add_test(NAME halide_correctness_extern_consumer
         COMMAND "${TS_BUILDDIR}/bin/correctness_extern_consumer")
add_test(NAME halide_correctness_extern_error
         COMMAND "${TS_BUILDDIR}/bin/correctness_extern_error")
add_test(NAME halide_correctness_extern_output_expansion
         COMMAND "${TS_BUILDDIR}/bin/correctness_extern_output_expansion")
add_test(NAME halide_correctness_extern_producer
         COMMAND "${TS_BUILDDIR}/bin/correctness_extern_producer")
add_test(NAME halide_correctness_extern_sort
         COMMAND "${TS_BUILDDIR}/bin/correctness_extern_sort")
add_test(NAME halide_correctness_extern_stage
         COMMAND "${TS_BUILDDIR}/bin/correctness_extern_stage")
add_test(NAME halide_correctness_fibonacci
         COMMAND "${TS_BUILDDIR}/bin/correctness_fibonacci")
add_test(NAME halide_correctness_float16_t_comparison
         COMMAND "${TS_BUILDDIR}/bin/correctness_float16_t_comparison")
add_test(NAME halide_correctness_float16_t_constants
         COMMAND "${TS_BUILDDIR}/bin/correctness_float16_t_constants")
add_test(NAME halide_correctness_float16_t_image_type
         COMMAND "${TS_BUILDDIR}/bin/correctness_float16_t_image_type")
add_test(NAME halide_correctness_float16_t_implicit_upcast
         COMMAND "${TS_BUILDDIR}/bin/correctness_float16_t_implicit_upcast")
add_test(NAME halide_correctness_float16_t_realize_constant
         COMMAND "${TS_BUILDDIR}/bin/correctness_float16_t_realize_constant")
add_test(NAME halide_correctness_func_lifetime
         COMMAND "${TS_BUILDDIR}/bin/correctness_func_lifetime")
add_test(NAME halide_correctness_func_lifetime_2
         COMMAND "${TS_BUILDDIR}/bin/correctness_func_lifetime_2")
add_test(NAME halide_correctness_fuse
         COMMAND "${TS_BUILDDIR}/bin/correctness_fuse")
add_test(NAME halide_correctness_fused_where_inner_extent_is_zero
         COMMAND "${TS_BUILDDIR}/bin/correctness_fused_where_inner_extent_is_zero")
add_test(NAME halide_correctness_fuzz_simplify
         COMMAND "${TS_BUILDDIR}/bin/correctness_fuzz_simplify")
add_test(NAME halide_correctness_gameoflife
         COMMAND "${TS_BUILDDIR}/bin/correctness_gameoflife")
add_test(NAME halide_correctness_gpu_data_flows
         COMMAND "${TS_BUILDDIR}/bin/correctness_gpu_data_flows")
add_test(NAME halide_correctness_gpu_dynamic_shared
         COMMAND "${TS_BUILDDIR}/bin/correctness_gpu_dynamic_shared")
add_test(NAME halide_correctness_gpu_free_sync
         COMMAND "${TS_BUILDDIR}/bin/correctness_gpu_free_sync")
add_test(NAME halide_correctness_gpu_large_alloc
         COMMAND "${TS_BUILDDIR}/bin/correctness_gpu_large_alloc")
add_test(NAME halide_correctness_gpu_mixed_dimensionality
         COMMAND "${TS_BUILDDIR}/bin/correctness_gpu_mixed_dimensionality")
add_test(NAME halide_correctness_gpu_mixed_shared_mem_types
         COMMAND "${TS_BUILDDIR}/bin/correctness_gpu_mixed_shared_mem_types")
add_test(NAME halide_correctness_gpu_multi_device
         COMMAND "${TS_BUILDDIR}/bin/correctness_gpu_multi_device")
add_test(NAME halide_correctness_gpu_multi_kernel
         COMMAND "${TS_BUILDDIR}/bin/correctness_gpu_multi_kernel")
add_test(NAME halide_correctness_gpu_non_contiguous_copy
         COMMAND "${TS_BUILDDIR}/bin/correctness_gpu_non_contiguous_copy")
add_test(NAME halide_correctness_gpu_object_lifetime
         COMMAND "${TS_BUILDDIR}/bin/correctness_gpu_object_lifetime")
add_test(NAME halide_correctness_gpu_specialize
         COMMAND "${TS_BUILDDIR}/bin/correctness_gpu_specialize")
add_test(NAME halide_correctness_gpu_sum_scan
         COMMAND "${TS_BUILDDIR}/bin/correctness_gpu_sum_scan")
add_test(NAME halide_correctness_gpu_thread_barrier
         COMMAND "${TS_BUILDDIR}/bin/correctness_gpu_thread_barrier")
add_test(NAME halide_correctness_gpu_transpose
         COMMAND "${TS_BUILDDIR}/bin/correctness_gpu_transpose")
add_test(NAME halide_correctness_gpu_vectorize_div_mod
         COMMAND "${TS_BUILDDIR}/bin/correctness_gpu_vectorize_div_mod")
add_test(NAME halide_correctness_gpu_vectorized_shared_memory
         COMMAND "${TS_BUILDDIR}/bin/correctness_gpu_vectorized_shared_memory")
add_test(NAME halide_correctness_handle
         COMMAND "${TS_BUILDDIR}/bin/correctness_handle")
add_test(NAME halide_correctness_heap_cleanup
         COMMAND "${TS_BUILDDIR}/bin/correctness_heap_cleanup")
add_test(NAME halide_correctness_hello_gpu
         COMMAND "${TS_BUILDDIR}/bin/correctness_hello_gpu")
add_test(NAME halide_correctness_histogram
         COMMAND "${TS_BUILDDIR}/bin/correctness_histogram")
add_test(NAME halide_correctness_histogram_equalize
         COMMAND "${TS_BUILDDIR}/bin/correctness_histogram_equalize")
add_test(NAME halide_correctness_image_of_lists
         COMMAND "${TS_BUILDDIR}/bin/correctness_image_of_lists")
add_test(NAME halide_correctness_implicit_args
         COMMAND "${TS_BUILDDIR}/bin/correctness_implicit_args")
add_test(NAME halide_correctness_infer_arguments
         COMMAND "${TS_BUILDDIR}/bin/correctness_infer_arguments")
add_test(NAME halide_correctness_inline_reduction
         COMMAND "${TS_BUILDDIR}/bin/correctness_inline_reduction")
add_test(NAME halide_correctness_in_place
         COMMAND "${TS_BUILDDIR}/bin/correctness_in_place")
add_test(NAME halide_correctness_input_image_bounds_check
         COMMAND "${TS_BUILDDIR}/bin/correctness_input_image_bounds_check")
add_test(NAME halide_correctness_input_larger_than_two_gigs
         COMMAND "${TS_BUILDDIR}/bin/correctness_input_larger_than_two_gigs")
add_test(NAME halide_correctness_integer_powers
         COMMAND "${TS_BUILDDIR}/bin/correctness_integer_powers")
add_test(NAME halide_correctness_interleave
         COMMAND "${TS_BUILDDIR}/bin/correctness_interleave")
add_test(NAME halide_correctness_introspection
         COMMAND "${TS_BUILDDIR}/bin/correctness_introspection")
add_test(NAME halide_correctness_inverse
         COMMAND "${TS_BUILDDIR}/bin/correctness_inverse")
add_test(NAME halide_correctness_isnan
         COMMAND "${TS_BUILDDIR}/bin/correctness_isnan")
add_test(NAME halide_correctness_iterate_over_circle
         COMMAND "${TS_BUILDDIR}/bin/correctness_iterate_over_circle")
add_test(NAME halide_correctness_lambda
         COMMAND "${TS_BUILDDIR}/bin/correctness_lambda")
add_test(NAME halide_correctness_lazy_convolution
         COMMAND "${TS_BUILDDIR}/bin/correctness_lazy_convolution")
add_test(NAME halide_correctness_legal_race_condition
         COMMAND "${TS_BUILDDIR}/bin/correctness_legal_race_condition")
add_test(NAME halide_correctness_lerp
         COMMAND "${TS_BUILDDIR}/bin/correctness_lerp")
add_test(NAME halide_correctness_likely
         COMMAND "${TS_BUILDDIR}/bin/correctness_likely")
add_test(NAME halide_correctness_logical
         COMMAND "${TS_BUILDDIR}/bin/correctness_logical")
add_test(NAME halide_correctness_loop_invariant_extern_calls
         COMMAND "${TS_BUILDDIR}/bin/correctness_loop_invariant_extern_calls")
add_test(NAME halide_correctness_make_struct
         COMMAND "${TS_BUILDDIR}/bin/correctness_make_struct")
add_test(NAME halide_correctness_many_dimensions
         COMMAND "${TS_BUILDDIR}/bin/correctness_many_dimensions")
add_test(NAME halide_correctness_many_small_extern_stages
         COMMAND "${TS_BUILDDIR}/bin/correctness_many_small_extern_stages")
add_test(NAME halide_correctness_many_updates
         COMMAND "${TS_BUILDDIR}/bin/correctness_many_updates")
add_test(NAME halide_correctness_math
         COMMAND "${TS_BUILDDIR}/bin/correctness_math")
add_test(NAME halide_correctness_memoize
         COMMAND "${TS_BUILDDIR}/bin/correctness_memoize")
add_test(NAME halide_correctness_min_extent
         COMMAND "${TS_BUILDDIR}/bin/correctness_min_extent")
add_test(NAME halide_correctness_mod
         COMMAND "${TS_BUILDDIR}/bin/correctness_mod")
add_test(NAME halide_correctness_multi_output_pipeline_with_bad_sizes
         COMMAND "${TS_BUILDDIR}/bin/correctness_multi_output_pipeline_with_bad_sizes")
add_test(NAME halide_correctness_multipass_constraints
         COMMAND "${TS_BUILDDIR}/bin/correctness_multipass_constraints")
add_test(NAME halide_correctness_multi_pass_reduction
         COMMAND "${TS_BUILDDIR}/bin/correctness_multi_pass_reduction")
add_test(NAME halide_correctness_multiple_outputs
         COMMAND "${TS_BUILDDIR}/bin/correctness_multiple_outputs")
add_test(NAME halide_correctness_multi_way_select
         COMMAND "${TS_BUILDDIR}/bin/correctness_multi_way_select")
add_test(NAME halide_correctness_named_updates
         COMMAND "${TS_BUILDDIR}/bin/correctness_named_updates")
add_test(NAME halide_correctness_newtons_method
         COMMAND "${TS_BUILDDIR}/bin/correctness_newtons_method")
add_test(NAME halide_correctness_obscure_image_references
         COMMAND "${TS_BUILDDIR}/bin/correctness_obscure_image_references")
add_test(NAME halide_correctness_oddly_sized_output
         COMMAND "${TS_BUILDDIR}/bin/correctness_oddly_sized_output")
add_test(NAME halide_correctness_out_of_memory
         COMMAND "${TS_BUILDDIR}/bin/correctness_out_of_memory")
add_test(NAME halide_correctness_output_larger_than_two_gigs
         COMMAND "${TS_BUILDDIR}/bin/correctness_output_larger_than_two_gigs")
add_test(NAME halide_correctness_parallel
         COMMAND "${TS_BUILDDIR}/bin/correctness_parallel")
add_test(NAME halide_correctness_parallel_alloc
         COMMAND "${TS_BUILDDIR}/bin/correctness_parallel_alloc")
add_test(NAME halide_correctness_parallel_gpu_nested
         COMMAND "${TS_BUILDDIR}/bin/correctness_parallel_gpu_nested")
add_test(NAME halide_correctness_parallel_nested
         COMMAND "${TS_BUILDDIR}/bin/correctness_parallel_nested")
add_test(NAME halide_correctness_parallel_reductions
         COMMAND "${TS_BUILDDIR}/bin/correctness_parallel_reductions")
add_test(NAME halide_correctness_parallel_rvar
         COMMAND "${TS_BUILDDIR}/bin/correctness_parallel_rvar")
add_test(NAME halide_correctness_param
         COMMAND "${TS_BUILDDIR}/bin/correctness_param")
add_test(NAME halide_correctness_parameter_constraints
         COMMAND "${TS_BUILDDIR}/bin/correctness_parameter_constraints")
add_test(NAME halide_correctness_partial_application
         COMMAND "${TS_BUILDDIR}/bin/correctness_partial_application")
add_test(NAME halide_correctness_partition_loops_bug
         COMMAND "${TS_BUILDDIR}/bin/correctness_partition_loops_bug")
add_test(NAME halide_correctness_pipeline_set_jit_externs_func
         COMMAND "${TS_BUILDDIR}/bin/correctness_pipeline_set_jit_externs_func")
add_test(NAME halide_correctness_print
         COMMAND "${TS_BUILDDIR}/bin/correctness_print")
add_test(NAME halide_correctness_process_some_tiles
         COMMAND "${TS_BUILDDIR}/bin/correctness_process_some_tiles")
add_test(NAME halide_correctness_random
         COMMAND "${TS_BUILDDIR}/bin/correctness_random")
add_test(NAME halide_correctness_realize_larger_than_two_gigs
         COMMAND "${TS_BUILDDIR}/bin/correctness_realize_larger_than_two_gigs")
add_test(NAME halide_correctness_realize_over_shifted_domain
         COMMAND "${TS_BUILDDIR}/bin/correctness_realize_over_shifted_domain")
add_test(NAME halide_correctness_reduction_chain
         COMMAND "${TS_BUILDDIR}/bin/correctness_reduction_chain")
add_test(NAME halide_correctness_reduction_schedule
         COMMAND "${TS_BUILDDIR}/bin/correctness_reduction_schedule")
add_test(NAME halide_correctness_reduction_subregion
         COMMAND "${TS_BUILDDIR}/bin/correctness_reduction_subregion")
add_test(NAME halide_correctness_reorder_rvars
         COMMAND "${TS_BUILDDIR}/bin/correctness_reorder_rvars")
add_test(NAME halide_correctness_reorder_storage
         COMMAND "${TS_BUILDDIR}/bin/correctness_reorder_storage")
add_test(NAME halide_correctness_reschedule
         COMMAND "${TS_BUILDDIR}/bin/correctness_reschedule")
add_test(NAME halide_correctness_reuse_stack_alloc
         COMMAND "${TS_BUILDDIR}/bin/correctness_reuse_stack_alloc")
add_test(NAME halide_correctness_round
         COMMAND "${TS_BUILDDIR}/bin/correctness_round")
add_test(NAME halide_correctness_runtime_float16_t_upcast
         COMMAND "${TS_BUILDDIR}/bin/correctness_runtime_float16_t_upcast")
add_test(NAME halide_correctness_scatter
         COMMAND "${TS_BUILDDIR}/bin/correctness_scatter")
add_test(NAME halide_correctness_shared_self_references
         COMMAND "${TS_BUILDDIR}/bin/correctness_shared_self_references")
add_test(NAME halide_correctness_shifted_image
         COMMAND "${TS_BUILDDIR}/bin/correctness_shifted_image")
add_test(NAME halide_correctness_side_effects
         COMMAND "${TS_BUILDDIR}/bin/correctness_side_effects")
add_test(NAME halide_correctness_simd_op_check
         COMMAND "${TS_BUILDDIR}/bin/correctness_simd_op_check")
add_test(NAME halide_correctness_simplified_away_embedded_image
         COMMAND "${TS_BUILDDIR}/bin/correctness_simplified_away_embedded_image")
add_test(NAME halide_correctness_skip_stages
         COMMAND "${TS_BUILDDIR}/bin/correctness_skip_stages")
add_test(NAME halide_correctness_skip_stages_external_array_functions
         COMMAND "${TS_BUILDDIR}/bin/correctness_skip_stages_external_array_functions")
add_test(NAME halide_correctness_sliding_backwards
         COMMAND "${TS_BUILDDIR}/bin/correctness_sliding_backwards")
add_test(NAME halide_correctness_sliding_reduction
         COMMAND "${TS_BUILDDIR}/bin/correctness_sliding_reduction")
add_test(NAME halide_correctness_sliding_window
         COMMAND "${TS_BUILDDIR}/bin/correctness_sliding_window")
add_test(NAME halide_correctness_sort_exprs
         COMMAND "${TS_BUILDDIR}/bin/correctness_sort_exprs")
add_test(NAME halide_correctness_specialize
         COMMAND "${TS_BUILDDIR}/bin/correctness_specialize")
add_test(NAME halide_correctness_specialize_to_gpu
         COMMAND "${TS_BUILDDIR}/bin/correctness_specialize_to_gpu")
add_test(NAME halide_correctness_split_fuse_rvar
         COMMAND "${TS_BUILDDIR}/bin/correctness_split_fuse_rvar")
add_test(NAME halide_correctness_split_reuse_inner_name_bug
         COMMAND "${TS_BUILDDIR}/bin/correctness_split_reuse_inner_name_bug")
add_test(NAME halide_correctness_split_store_compute
         COMMAND "${TS_BUILDDIR}/bin/correctness_split_store_compute")
add_test(NAME halide_correctness_stack_allocations
         COMMAND "${TS_BUILDDIR}/bin/correctness_stack_allocations")
add_test(NAME halide_correctness_stmt_to_html
         COMMAND "${TS_BUILDDIR}/bin/correctness_stmt_to_html")
add_test(NAME halide_correctness_storage_folding
         COMMAND "${TS_BUILDDIR}/bin/correctness_storage_folding")
add_test(NAME halide_correctness_stream_compaction
         COMMAND "${TS_BUILDDIR}/bin/correctness_stream_compaction")
add_test(NAME halide_correctness_strided_load
         COMMAND "${TS_BUILDDIR}/bin/correctness_strided_load")
add_test(NAME halide_correctness_target
         COMMAND "${TS_BUILDDIR}/bin/correctness_target")
add_test(NAME halide_correctness_tracing
         COMMAND "${TS_BUILDDIR}/bin/correctness_tracing")
add_test(NAME halide_correctness_tracing_bounds
         COMMAND "${TS_BUILDDIR}/bin/correctness_tracing_bounds")
add_test(NAME halide_correctness_tracing_stack
         COMMAND "${TS_BUILDDIR}/bin/correctness_tracing_stack")
add_test(NAME halide_correctness_transitive_bounds
         COMMAND "${TS_BUILDDIR}/bin/correctness_transitive_bounds")
add_test(NAME halide_correctness_tuple_reduction
         COMMAND "${TS_BUILDDIR}/bin/correctness_tuple_reduction")
add_test(NAME halide_correctness_two_vector_args
         COMMAND "${TS_BUILDDIR}/bin/correctness_two_vector_args")
add_test(NAME halide_correctness_undef
         COMMAND "${TS_BUILDDIR}/bin/correctness_undef")
add_test(NAME halide_correctness_uninitialized_read
         COMMAND "${TS_BUILDDIR}/bin/correctness_uninitialized_read")
add_test(NAME halide_correctness_unique_func_image
         COMMAND "${TS_BUILDDIR}/bin/correctness_unique_func_image")
add_test(NAME halide_correctness_unrolled_reduction
         COMMAND "${TS_BUILDDIR}/bin/correctness_unrolled_reduction")
add_test(NAME halide_correctness_update_chunk
         COMMAND "${TS_BUILDDIR}/bin/correctness_update_chunk")
add_test(NAME halide_correctness_vector_bounds_inference
         COMMAND "${TS_BUILDDIR}/bin/correctness_vector_bounds_inference")
add_test(NAME halide_correctness_vector_cast
         COMMAND "${TS_BUILDDIR}/bin/correctness_vector_cast")
add_test(NAME halide_correctness_vector_extern
         COMMAND "${TS_BUILDDIR}/bin/correctness_vector_extern")
add_test(NAME halide_correctness_vectorized_initialization
         COMMAND "${TS_BUILDDIR}/bin/correctness_vectorized_initialization")
add_test(NAME halide_correctness_vectorized_reduction_bug
         COMMAND "${TS_BUILDDIR}/bin/correctness_vectorized_reduction_bug")
add_test(NAME halide_correctness_vectorize_mixed_widths
         COMMAND "${TS_BUILDDIR}/bin/correctness_vectorize_mixed_widths")
add_test(NAME halide_correctness_vector_math
         COMMAND "${TS_BUILDDIR}/bin/correctness_vector_math")
add_test(NAME halide_error_ambiguous_inline_reductions
         COMMAND "${TS_BUILDDIR}/bin/error_ambiguous_inline_reductions")
add_test(NAME halide_error_bad_bound
         COMMAND "${TS_BUILDDIR}/bin/error_bad_bound")
add_test(NAME halide_error_bad_compute_at
         COMMAND "${TS_BUILDDIR}/bin/error_bad_compute_at")
add_test(NAME halide_error_bad_const_cast
         COMMAND "${TS_BUILDDIR}/bin/error_bad_const_cast")
add_test(NAME halide_error_bad_rvar_order
         COMMAND "${TS_BUILDDIR}/bin/error_bad_rvar_order")
add_test(NAME halide_error_bad_schedule
         COMMAND "${TS_BUILDDIR}/bin/error_bad_schedule")
add_test(NAME halide_error_bad_store_at
         COMMAND "${TS_BUILDDIR}/bin/error_bad_store_at")
add_test(NAME halide_error_buffer_larger_than_two_gigs
         COMMAND "${TS_BUILDDIR}/bin/error_buffer_larger_than_two_gigs")
add_test(NAME halide_error_constrain_wrong_output_buffer
         COMMAND "${TS_BUILDDIR}/bin/error_constrain_wrong_output_buffer")
add_test(NAME halide_error_define_after_realize
         COMMAND "${TS_BUILDDIR}/bin/error_define_after_realize")
add_test(NAME halide_error_define_after_use
         COMMAND "${TS_BUILDDIR}/bin/error_define_after_use")
add_test(NAME halide_error_expanding_reduction
         COMMAND "${TS_BUILDDIR}/bin/error_expanding_reduction")
add_test(NAME halide_error_five_d_gpu_buffer
         COMMAND "${TS_BUILDDIR}/bin/error_five_d_gpu_buffer")
add_test(NAME halide_error_float16_t_implicit_downcast
         COMMAND "${TS_BUILDDIR}/bin/error_float16_t_implicit_downcast")
add_test(NAME halide_error_float16_t_overflow
         COMMAND "${TS_BUILDDIR}/bin/error_float16_t_overflow")
add_test(NAME halide_error_float16_t_overflow_int_conv
         COMMAND "${TS_BUILDDIR}/bin/error_float16_t_overflow_int_conv")
add_test(NAME halide_error_float_arg
         COMMAND "${TS_BUILDDIR}/bin/error_float_arg")
add_test(NAME halide_error_impossible_constraints
         COMMAND "${TS_BUILDDIR}/bin/error_impossible_constraints")
add_test(NAME halide_error_lerp_float_weight_out_of_range
         COMMAND "${TS_BUILDDIR}/bin/error_lerp_float_weight_out_of_range")
add_test(NAME halide_error_lerp_mismatch
         COMMAND "${TS_BUILDDIR}/bin/error_lerp_mismatch")
add_test(NAME halide_error_lerp_signed_weight
         COMMAND "${TS_BUILDDIR}/bin/error_lerp_signed_weight")
add_test(NAME halide_error_memoize_different_compute_store
         COMMAND "${TS_BUILDDIR}/bin/error_memoize_different_compute_store")
add_test(NAME halide_error_missing_args
         COMMAND "${TS_BUILDDIR}/bin/error_missing_args")
add_test(NAME halide_error_modulo_constant_zero
         COMMAND "${TS_BUILDDIR}/bin/error_modulo_constant_zero")
add_test(NAME halide_error_nonexistent_update_stage
         COMMAND "${TS_BUILDDIR}/bin/error_nonexistent_update_stage")
add_test(NAME halide_error_old_implicit_args
         COMMAND "${TS_BUILDDIR}/bin/error_old_implicit_args")
add_test(NAME halide_error_pointer_arithmetic
         COMMAND "${TS_BUILDDIR}/bin/error_pointer_arithmetic")
add_test(NAME halide_error_race_condition
         COMMAND "${TS_BUILDDIR}/bin/error_race_condition")
add_test(NAME halide_error_realize_constantly_larger_than_two_gigs
         COMMAND "${TS_BUILDDIR}/bin/error_realize_constantly_larger_than_two_gigs")
add_test(NAME halide_error_reduction_bounds
         COMMAND "${TS_BUILDDIR}/bin/error_reduction_bounds")
add_test(NAME halide_error_reduction_type_mismatch
         COMMAND "${TS_BUILDDIR}/bin/error_reduction_type_mismatch")
add_test(NAME halide_error_reused_args
         COMMAND "${TS_BUILDDIR}/bin/error_reused_args")
add_test(NAME halide_error_reuse_var_in_schedule
         COMMAND "${TS_BUILDDIR}/bin/error_reuse_var_in_schedule")
add_test(NAME halide_error_thread_id_outside_block_id
         COMMAND "${TS_BUILDDIR}/bin/error_thread_id_outside_block_id")
add_test(NAME halide_error_too_many_args
         COMMAND "${TS_BUILDDIR}/bin/error_too_many_args")
add_test(NAME halide_error_unbounded_input
         COMMAND "${TS_BUILDDIR}/bin/error_unbounded_input")
add_test(NAME halide_error_unbounded_output
         COMMAND "${TS_BUILDDIR}/bin/error_unbounded_output")
add_test(NAME halide_error_undefined_rdom_dimension
         COMMAND "${TS_BUILDDIR}/bin/error_undefined_rdom_dimension")
add_test(NAME halide_error_vectorize_dynamic
         COMMAND "${TS_BUILDDIR}/bin/error_vectorize_dynamic")
add_test(NAME halide_error_vectorize_too_little
         COMMAND "${TS_BUILDDIR}/bin/error_vectorize_too_little")
add_test(NAME halide_error_vectorize_too_much
         COMMAND "${TS_BUILDDIR}/bin/error_vectorize_too_much")
add_test(NAME halide_error_wrong_type
         COMMAND "${TS_BUILDDIR}/bin/error_wrong_type")
add_test(NAME halide_filter
         COMMAND "${TS_BUILDDIR}/bin/filter")
add_test(NAME halide_generator_aot_acquire_release
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_acquire_release")
add_test(NAME halide_generator_aot_argvcall
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_argvcall")
add_test(NAME halide_generator_aot_cleanup_on_error
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_cleanup_on_error")
add_test(NAME halide_generator_aot_embed_image
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_embed_image")
add_test(NAME halide_generator_aot_error_codes
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_error_codes")
add_test(NAME halide_generator_aot_example
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_example")
add_test(NAME halide_generator_aot_extended_buffer_t
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_extended_buffer_t")
add_test(NAME halide_generator_aot_gpu_object_lifetime
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_gpu_object_lifetime")
add_test(NAME halide_generator_aot_gpu_only
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_gpu_only")
add_test(NAME halide_generator_aot_mandelbrot
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_mandelbrot")
add_test(NAME halide_generator_aot_matlab
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_matlab")
add_test(NAME halide_generator_aot_metadata_tester
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_metadata_tester")
add_test(NAME halide_generator_aot_nested_externs
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_nested_externs")
add_test(NAME halide_generator_aot_pyramid
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_pyramid")
add_test(NAME halide_generator_aot_tiled_blur
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_tiled_blur")
add_test(NAME halide_generator_aot_tiled_blur_interleaved
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_tiled_blur_interleaved")
add_test(NAME halide_generator_aot_user_context
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_user_context")
add_test(NAME halide_generator_aot_user_context_insanity
         COMMAND "${TS_BUILDDIR}/bin/generator_aot_user_context_insanity")
add_test(NAME halide_generator_jit_example
         COMMAND "${TS_BUILDDIR}/bin/generator_jit_example")
add_test(NAME halide_generator_jit_paramtest
         COMMAND "${TS_BUILDDIR}/bin/generator_jit_paramtest")
add_test(NAME halide_halide_blur
         COMMAND "${TS_BUILDDIR}/bin/halide_blur")
add_test(NAME halide_HalideTraceViz
         COMMAND "${TS_BUILDDIR}/bin/HalideTraceViz")
add_test(NAME halide_lesson_01_basics
         COMMAND "${TS_BUILDDIR}/bin/lesson_01_basics")
add_test(NAME halide_lesson_02_input_image
         COMMAND "${TS_BUILDDIR}/bin/lesson_02_input_image")
add_test(NAME halide_lesson_03_debugging_1
         COMMAND "${TS_BUILDDIR}/bin/lesson_03_debugging_1")
add_test(NAME halide_lesson_04_debugging_2
         COMMAND "${TS_BUILDDIR}/bin/lesson_04_debugging_2")
add_test(NAME halide_lesson_05_scheduling_1
         COMMAND "${TS_BUILDDIR}/bin/lesson_05_scheduling_1")
add_test(NAME halide_lesson_06_realizing_over_shifted_domains
         COMMAND "${TS_BUILDDIR}/bin/lesson_06_realizing_over_shifted_domains")
add_test(NAME halide_lesson_07_multi_stage_pipelines
         COMMAND "${TS_BUILDDIR}/bin/lesson_07_multi_stage_pipelines")
add_test(NAME halide_lesson_08_scheduling_2
         COMMAND "${TS_BUILDDIR}/bin/lesson_08_scheduling_2")
add_test(NAME halide_lesson_09_update_definitions
         COMMAND "${TS_BUILDDIR}/bin/lesson_09_update_definitions")
add_test(NAME halide_lesson_10_aot_compilation_generate
         COMMAND "${TS_BUILDDIR}/bin/lesson_10_aot_compilation_generate")
add_test(NAME halide_lesson_10_aot_compilation_run
         COMMAND "${TS_BUILDDIR}/bin/lesson_10_aot_compilation_run")
add_test(NAME halide_lesson_11_cross_compilation
         COMMAND "${TS_BUILDDIR}/bin/lesson_11_cross_compilation")
add_test(NAME halide_lesson_12_using_the_gpu
         COMMAND "${TS_BUILDDIR}/bin/lesson_12_using_the_gpu")
add_test(NAME halide_lesson_13_tuples
         COMMAND "${TS_BUILDDIR}/bin/lesson_13_tuples")
add_test(NAME halide_lesson_14_types
         COMMAND "${TS_BUILDDIR}/bin/lesson_14_types")
add_test(NAME halide_ll_process
         COMMAND "${TS_BUILDDIR}/bin/ll_process")
add_test(NAME halide_local_laplacian_gen
         COMMAND "${TS_BUILDDIR}/bin/local_laplacian_gen")
add_test(NAME halide_performance_block_transpose
         COMMAND "${TS_BUILDDIR}/bin/performance_block_transpose")
add_test(NAME halide_performance_boundary_conditions
         COMMAND "${TS_BUILDDIR}/bin/performance_boundary_conditions")
add_test(NAME halide_performance_clamped_vector_load
         COMMAND "${TS_BUILDDIR}/bin/performance_clamped_vector_load")
add_test(NAME halide_performance_const_division
         COMMAND "${TS_BUILDDIR}/bin/performance_const_division")
add_test(NAME halide_performance_fast_inverse
         COMMAND "${TS_BUILDDIR}/bin/performance_fast_inverse")
add_test(NAME halide_performance_fast_pow
         COMMAND "${TS_BUILDDIR}/bin/performance_fast_pow")
add_test(NAME halide_performance_inner_loop_parallel
         COMMAND "${TS_BUILDDIR}/bin/performance_inner_loop_parallel")
add_test(NAME halide_performance_jit_stress
         COMMAND "${TS_BUILDDIR}/bin/performance_jit_stress")
add_test(NAME halide_performance_matrix_multiplication
         COMMAND "${TS_BUILDDIR}/bin/performance_matrix_multiplication")
add_test(NAME halide_performance_memcpy
         COMMAND "${TS_BUILDDIR}/bin/performance_memcpy")
add_test(NAME halide_performance_packed_planar_fusion
         COMMAND "${TS_BUILDDIR}/bin/performance_packed_planar_fusion")
add_test(NAME halide_performance_parallel_performance
         COMMAND "${TS_BUILDDIR}/bin/performance_parallel_performance")
add_test(NAME halide_performance_profiler
         COMMAND "${TS_BUILDDIR}/bin/performance_profiler")
add_test(NAME halide_performance_rgb_interleaved
         COMMAND "${TS_BUILDDIR}/bin/performance_rgb_interleaved")
add_test(NAME halide_performance_sort
         COMMAND "${TS_BUILDDIR}/bin/performance_sort")
add_test(NAME halide_performance_vectorize
         COMMAND "${TS_BUILDDIR}/bin/performance_vectorize")
add_test(NAME halide_pipeline
         COMMAND "${TS_BUILDDIR}/bin/pipeline")
add_test(NAME halide_process
         COMMAND "${TS_BUILDDIR}/bin/process")
add_test(NAME halide_renderscript_aot_copy
         COMMAND "${TS_BUILDDIR}/bin/renderscript_aot_copy")
add_test(NAME halide_renderscript_aot_copy_error
         COMMAND "${TS_BUILDDIR}/bin/renderscript_aot_copy_error")
add_test(NAME halide_renderscript_jit_copy
         COMMAND "${TS_BUILDDIR}/bin/renderscript_jit_copy")
add_test(NAME halide_run_c_backend_and_native
         COMMAND "${TS_BUILDDIR}/bin/run_c_backend_and_native")
add_test(NAME halide_test_internal
         COMMAND "${TS_BUILDDIR}/bin/test_internal")
add_test(NAME halide_warning_double_vectorize
         COMMAND "${TS_BUILDDIR}/bin/warning_double_vectorize")
add_test(NAME halide_warning_float16_t_underflow
         COMMAND "${TS_BUILDDIR}/bin/warning_float16_t_underflow")
add_test(NAME halide_warning_hidden_pure_definition
         COMMAND "${TS_BUILDDIR}/bin/warning_hidden_pure_definition")
add_test(NAME halide_warning_parallel_size_one
         COMMAND "${TS_BUILDDIR}/bin/warning_parallel_size_one")
add_test(NAME halide_warning_vectorize_size_one
         COMMAND "${TS_BUILDDIR}/bin/warning_vectorize_size_one")



set_tests_properties(
  halide_bilateral_grid
  halide_bitcode2cpp
  halide_blur_test
  halide_build_halide_h
  halide_camera_pipe
  halide_correctness_argmax
  halide_correctness_assertion_failure_in_parallel_for
  halide_correctness_autotune_bug
  halide_correctness_autotune_bug_2
  halide_correctness_autotune_bug_3
  halide_correctness_autotune_bug_4
  halide_correctness_autotune_bug_5
  halide_correctness_bad_likely
  halide_correctness_bit_counting
  halide_correctness_bitwise_ops
  halide_correctness_bool_compute_root_vectorize
  halide_correctness_bound
  halide_correctness_boundary_conditions
  halide_correctness_bounds
  halide_correctness_bounds_inference
  halide_correctness_bounds_inference_chunk
  halide_correctness_bounds_inference_complex
  halide_correctness_bounds_of_abs
  halide_correctness_bounds_of_cast
  halide_correctness_bounds_of_func
  halide_correctness_bounds_of_monotonic_math
  halide_correctness_bounds_query
  halide_correctness_buffer_t
  halide_correctness_cascaded_filters
  halide_correctness_cast
  halide_correctness_cast_handle
  halide_correctness_c_function
  halide_correctness_chunk
  halide_correctness_chunk_sharing
  halide_correctness_circular_reference_leak
  halide_correctness_code_explosion
  halide_correctness_compare_vars
  halide_correctness_compile_to
  halide_correctness_compile_to_bitcode
  halide_correctness_compile_to_lowered_stmt
  halide_correctness_compute_at_split_rvar
  halide_correctness_computed_index
  halide_correctness_compute_outermost
  halide_correctness_constant_expr
  halide_correctness_constant_type
  halide_correctness_constraints
  halide_correctness_convolution
  halide_correctness_convolution_multiple_kernels
  halide_correctness_cross_compilation
  halide_correctness_custom_allocator
  halide_correctness_custom_error_reporter
  halide_correctness_custom_lowering_pass
  halide_correctness_debug_to_file
  halide_correctness_deinterleave4
  halide_correctness_div_mod
  halide_correctness_dynamic_reduction_bounds
  halide_correctness_erf
  halide_correctness_exception
  halide_correctness_explicit_inline_reductions
  halide_correctness_extern_bounds_inference
  halide_correctness_extern_consumer
  halide_correctness_extern_error
  halide_correctness_extern_output_expansion
  halide_correctness_extern_producer
  halide_correctness_extern_sort
  halide_correctness_extern_stage
  halide_correctness_fibonacci
  halide_correctness_float16_t_comparison
  halide_correctness_float16_t_constants
  halide_correctness_float16_t_image_type
  halide_correctness_float16_t_implicit_upcast
  halide_correctness_float16_t_realize_constant
  halide_correctness_func_lifetime
  halide_correctness_func_lifetime_2
  halide_correctness_fuse
  halide_correctness_fused_where_inner_extent_is_zero
  halide_correctness_fuzz_simplify
  halide_correctness_gameoflife
  halide_correctness_gpu_data_flows
  halide_correctness_gpu_dynamic_shared
  halide_correctness_gpu_free_sync
  halide_correctness_gpu_large_alloc
  halide_correctness_gpu_mixed_dimensionality
  halide_correctness_gpu_mixed_shared_mem_types
  halide_correctness_gpu_multi_device
  halide_correctness_gpu_multi_kernel
  halide_correctness_gpu_non_contiguous_copy
  halide_correctness_gpu_object_lifetime
  halide_correctness_gpu_specialize
  halide_correctness_gpu_sum_scan
  halide_correctness_gpu_thread_barrier
  halide_correctness_gpu_transpose
  halide_correctness_gpu_vectorize_div_mod
  halide_correctness_gpu_vectorized_shared_memory
  halide_correctness_handle
  halide_correctness_heap_cleanup
  halide_correctness_hello_gpu
  halide_correctness_histogram
  halide_correctness_histogram_equalize
  halide_correctness_image_of_lists
  halide_correctness_implicit_args
  halide_correctness_infer_arguments
  halide_correctness_inline_reduction
  halide_correctness_in_place
  halide_correctness_input_image_bounds_check
  halide_correctness_input_larger_than_two_gigs
  halide_correctness_integer_powers
  halide_correctness_interleave
  halide_correctness_introspection
  halide_correctness_inverse
  halide_correctness_isnan
  halide_correctness_iterate_over_circle
  halide_correctness_lambda
  halide_correctness_lazy_convolution
  halide_correctness_legal_race_condition
  halide_correctness_lerp
  halide_correctness_likely
  halide_correctness_logical
  halide_correctness_loop_invariant_extern_calls
  halide_correctness_make_struct
  halide_correctness_many_dimensions
  halide_correctness_many_small_extern_stages
  halide_correctness_many_updates
  halide_correctness_math
  halide_correctness_memoize
  halide_correctness_min_extent
  halide_correctness_mod
  halide_correctness_multi_output_pipeline_with_bad_sizes
  halide_correctness_multipass_constraints
  halide_correctness_multi_pass_reduction
  halide_correctness_multiple_outputs
  halide_correctness_multi_way_select
  halide_correctness_named_updates
  halide_correctness_newtons_method
  halide_correctness_obscure_image_references
  halide_correctness_oddly_sized_output
  halide_correctness_out_of_memory
  halide_correctness_output_larger_than_two_gigs
  halide_correctness_parallel
  halide_correctness_parallel_alloc
  halide_correctness_parallel_gpu_nested
  halide_correctness_parallel_nested
  halide_correctness_parallel_reductions
  halide_correctness_parallel_rvar
  halide_correctness_param
  halide_correctness_parameter_constraints
  halide_correctness_partial_application
  halide_correctness_partition_loops_bug
  halide_correctness_pipeline_set_jit_externs_func
  halide_correctness_print
  halide_correctness_process_some_tiles
  halide_correctness_random
  halide_correctness_realize_larger_than_two_gigs
  halide_correctness_realize_over_shifted_domain
  halide_correctness_reduction_chain
  halide_correctness_reduction_schedule
  halide_correctness_reduction_subregion
  halide_correctness_reorder_rvars
  halide_correctness_reorder_storage
  halide_correctness_reschedule
  halide_correctness_reuse_stack_alloc
  halide_correctness_round
  halide_correctness_runtime_float16_t_upcast
  halide_correctness_scatter
  halide_correctness_shared_self_references
  halide_correctness_shifted_image
  halide_correctness_side_effects
  halide_correctness_simd_op_check
  halide_correctness_simplified_away_embedded_image
  halide_correctness_skip_stages
  halide_correctness_skip_stages_external_array_functions
  halide_correctness_sliding_backwards
  halide_correctness_sliding_reduction
  halide_correctness_sliding_window
  halide_correctness_sort_exprs
  halide_correctness_specialize
  halide_correctness_specialize_to_gpu
  halide_correctness_split_fuse_rvar
  halide_correctness_split_reuse_inner_name_bug
  halide_correctness_split_store_compute
  halide_correctness_stack_allocations
  halide_correctness_stmt_to_html
  halide_correctness_storage_folding
  halide_correctness_stream_compaction
  halide_correctness_strided_load
  halide_correctness_target
  halide_correctness_tracing
  halide_correctness_tracing_bounds
  halide_correctness_tracing_stack
  halide_correctness_transitive_bounds
  halide_correctness_tuple_reduction
  halide_correctness_two_vector_args
  halide_correctness_undef
  halide_correctness_uninitialized_read
  halide_correctness_unique_func_image
  halide_correctness_unrolled_reduction
  halide_correctness_update_chunk
  halide_correctness_vector_bounds_inference
  halide_correctness_vector_cast
  halide_correctness_vector_extern
  halide_correctness_vectorized_initialization
  halide_correctness_vectorized_reduction_bug
  halide_correctness_vectorize_mixed_widths
  halide_correctness_vector_math
  halide_error_ambiguous_inline_reductions
  halide_error_bad_bound
  halide_error_bad_compute_at
  halide_error_bad_const_cast
  halide_error_bad_rvar_order
  halide_error_bad_schedule
  halide_error_bad_store_at
  halide_error_buffer_larger_than_two_gigs
  halide_error_constrain_wrong_output_buffer
  halide_error_define_after_realize
  halide_error_define_after_use
  halide_error_expanding_reduction
  halide_error_five_d_gpu_buffer
  halide_error_float16_t_implicit_downcast
  halide_error_float16_t_overflow
  halide_error_float16_t_overflow_int_conv
  halide_error_float_arg
  halide_error_impossible_constraints
  halide_error_lerp_float_weight_out_of_range
  halide_error_lerp_mismatch
  halide_error_lerp_signed_weight
  halide_error_memoize_different_compute_store
  halide_error_missing_args
  halide_error_modulo_constant_zero
  halide_error_nonexistent_update_stage
  halide_error_old_implicit_args
  halide_error_pointer_arithmetic
  halide_error_race_condition
  halide_error_realize_constantly_larger_than_two_gigs
  halide_error_reduction_bounds
  halide_error_reduction_type_mismatch
  halide_error_reused_args
  halide_error_reuse_var_in_schedule
  halide_error_thread_id_outside_block_id
  halide_error_too_many_args
  halide_error_unbounded_input
  halide_error_unbounded_output
  halide_error_undefined_rdom_dimension
  halide_error_vectorize_dynamic
  halide_error_vectorize_too_little
  halide_error_vectorize_too_much
  halide_error_wrong_type
  halide_filter
  halide_generator_aot_acquire_release
  halide_generator_aot_argvcall
  halide_generator_aot_cleanup_on_error
  halide_generator_aot_embed_image
  halide_generator_aot_error_codes
  halide_generator_aot_example
  halide_generator_aot_extended_buffer_t
  halide_generator_aot_gpu_object_lifetime
  halide_generator_aot_gpu_only
  halide_generator_aot_mandelbrot
  halide_generator_aot_matlab
  halide_generator_aot_metadata_tester
  halide_generator_aot_nested_externs
  halide_generator_aot_pyramid
  halide_generator_aot_tiled_blur
  halide_generator_aot_tiled_blur_interleaved
  halide_generator_aot_user_context
  halide_generator_aot_user_context_insanity
  halide_generator_jit_example
  halide_generator_jit_paramtest
  halide_halide_blur
  halide_HalideTraceViz
  halide_lesson_01_basics
  halide_lesson_02_input_image
  halide_lesson_03_debugging_1
  halide_lesson_04_debugging_2
  halide_lesson_05_scheduling_1
  halide_lesson_06_realizing_over_shifted_domains
  halide_lesson_07_multi_stage_pipelines
  halide_lesson_08_scheduling_2
  halide_lesson_09_update_definitions
  halide_lesson_10_aot_compilation_generate
  halide_lesson_10_aot_compilation_run
  halide_lesson_11_cross_compilation
  halide_lesson_12_using_the_gpu
  halide_lesson_13_tuples
  halide_lesson_14_types
  halide_ll_process
  halide_local_laplacian_gen
  halide_performance_block_transpose
  halide_performance_boundary_conditions
  halide_performance_clamped_vector_load
  halide_performance_const_division
  halide_performance_fast_inverse
  halide_performance_fast_pow
  halide_performance_inner_loop_parallel
  halide_performance_jit_stress
  halide_performance_matrix_multiplication
  halide_performance_memcpy
  halide_performance_packed_planar_fusion
  halide_performance_parallel_performance
  halide_performance_profiler
  halide_performance_rgb_interleaved
  halide_performance_sort
  halide_performance_vectorize
  halide_pipeline
  halide_process
  halide_renderscript_aot_copy
  halide_renderscript_aot_copy_error
  halide_renderscript_jit_copy
  halide_run_c_backend_and_native
  halide_test_internal
  halide_warning_double_vectorize
  halide_warning_float16_t_underflow
  halide_warning_hidden_pure_definition
  halide_warning_parallel_size_one
  halide_warning_vectorize_size_one
  PROPERTIES
    LABELS "Halide"
    ENVIRONMENT "HL_JIT_TARGET=opencl" "HL_TARGET=opencl")

# disabled OpenGL tests

# halide_opengl_copy_pixels
# halide_opengl_copy_to_device
# halide_opengl_copy_to_host
# halide_opengl_float_texture
# halide_opengl_internal
# halide_opengl_lut
# halide_opengl_produce
# halide_opengl_select
# halide_opengl_set_pixels
# halide_opengl_shifted_domains
# halide_opengl_special_funcs
# halide_opengl_test
# halide_opengl_varying
# halide_glsl_halide_blur
# halide_glsl_halide_ycc
# add_test(NAME halide_opengl_copy_pixels
         # COMMAND "${TS_BUILDDIR}/bin/opengl_copy_pixels")
# add_test(NAME halide_opengl_copy_to_device
         # COMMAND "${TS_BUILDDIR}/bin/opengl_copy_to_device")
# add_test(NAME halide_opengl_copy_to_host
         # COMMAND "${TS_BUILDDIR}/bin/opengl_copy_to_host")
# add_test(NAME halide_opengl_float_texture
         # COMMAND "${TS_BUILDDIR}/bin/opengl_float_texture")
# add_test(NAME halide_opengl_internal
         # COMMAND "${TS_BUILDDIR}/bin/opengl_internal")
# add_test(NAME halide_opengl_lut
         # COMMAND "${TS_BUILDDIR}/bin/opengl_lut")
# add_test(NAME halide_opengl_produce
         # COMMAND "${TS_BUILDDIR}/bin/opengl_produce")
# add_test(NAME halide_opengl_select
         # COMMAND "${TS_BUILDDIR}/bin/opengl_select")
# add_test(NAME halide_opengl_set_pixels
         # COMMAND "${TS_BUILDDIR}/bin/opengl_set_pixels")
# add_test(NAME halide_opengl_shifted_domains
         # COMMAND "${TS_BUILDDIR}/bin/opengl_shifted_domains")
# add_test(NAME halide_opengl_special_funcs
         # COMMAND "${TS_BUILDDIR}/bin/opengl_special_funcs")
# add_test(NAME halide_opengl_test
         # COMMAND "${TS_BUILDDIR}/bin/opengl_test")
# add_test(NAME halide_opengl_varying
         # COMMAND "${TS_BUILDDIR}/bin/opengl_varying")
# add_test(NAME halide_glsl_halide_blur
         # COMMAND "${TS_BUILDDIR}/bin/glsl_halide_blur")
# add_test(NAME halide_glsl_halide_ycc
         # COMMAND "${TS_BUILDDIR}/bin/glsl_halide_ycc")



endif()
