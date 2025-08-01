#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2017 pocl developers
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


set(C_PROGRAMS_TO_BUILD test_assign_loop_variable_to_privvar_makes_it_local
     test_program_from_binary_with_local_1_1_1
     test_assign_loop_variable_to_privvar_makes_it_local_2
     test_llvm_segfault_issue_889
     test_issue_1711 test_issue_1794 test_issue_1747
     test_rematerialized_alloca_load_with_outside_pr_users
)
foreach(PROG ${C_PROGRAMS_TO_BUILD})
  add_executable("${PROG}" "${PROG}.c")
  target_link_libraries("${PROG}" ${POCLU_LINK_OPTIONS})
  add_symlink_to_built_opencl_dynlib("${PROG}")
endforeach()


set(PROGRAMS_TO_BUILD test_barrier_between_for_loops test_early_return
  test_for_with_var_iteration_count test_id_dependent_computation
  test_locals test_multi_level_loops_with_barriers
  test_simple_for_with_a_barrier test_structs_as_args test_vectors_as_args
  test_barrier_before_return test_infinite_loop test_constant_array
  test_undominated_variable test_setargs test_null_arg
  test_fors_with_var_iteration_counts test_issue_231 test_issue_445
  test_autolocals_in_constexprs test_issue_553 test_issue_577 test_issue_757
  test_flatten_barrier_subs test_alignment_with_dynamic_wg
  test_alignment_with_dynamic_wg2 test_alignment_with_dynamic_wg3
  test_issue_893 test_issue_1435 test_builtin_args test_issue_1390
  test_workitem_func_outside_kernel test_program_scope_vars test_issue_1548
  test_issue_1525 test_issue_1608 test_issue_1826 test_builtin_regression
  test_mem_host_ptr_issue test_issue_1962
)

if (MSVC)
  add_compile_options(${OPENCL_CFLAGS})
else ()
  add_compile_options("-Wno-deprecated" "-Wno-deprecated-declarations" ${OPENCL_CFLAGS})
endif ()
 
add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
include_directories("${CMAKE_SOURCE_DIR}/lib/CL")

foreach(PROG ${PROGRAMS_TO_BUILD})
  add_executable("${PROG}" "${PROG}.cpp")
  target_link_libraries("${PROG}" ${POCLU_LINK_OPTIONS})
  add_symlink_to_built_opencl_dynlib("${PROG}")
endforeach()

if (ENABLE_SPIRV)
  set(SPIRV_LABELS "")
else()
  set(SPIRV_LABELS "cpu_fail")
endif()


######################################################################

add_test_pocl(NAME "regression/test_issue_231" COMMAND "test_issue_231")

add_test_pocl(NAME "regression/test_issue_445" COMMAND "test_issue_445")

add_test_pocl(NAME "regression/test_issue_553" COMMAND "test_issue_553")

add_test_pocl(NAME "regression/test_issue_577" COMMAND "test_issue_577")

add_test_pocl(NAME "regression/test_issue_757" COMMAND "test_issue_757")

add_test_pocl(NAME "regression/test_issue_1435" COMMAND "test_issue_1435")

# fails on windows/MinGW with "access violation"
add_test_pocl(NAME "regression/test_issue_1525" LABELS "mingw_fail;win_fail"
  COMMAND "test_issue_1525"
  LLVM_FILECHECK "test_issue_1525.fc"
  WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "regression/test_issue_1608" COMMAND "test_issue_1608")

add_test_pocl(NAME "regression/test_issue_1826" COMMAND "test_issue_1826")

add_test_pocl(NAME "regression/test_builtin_regression"
  COMMAND "test_builtin_regression")


add_test_pocl(NAME "regression/test_mem_host_ptr_issue"
  COMMAND "test_mem_host_ptr_issue")

add_test_pocl(NAME "regression/test_issue_1962" COMMAND "test_issue_1962")

add_test_pocl(NAME "regression/test_workitem_func_outside_kernel" COMMAND "test_workitem_func_outside_kernel")

add_test(NAME "regression/test_program_scope_vars" COMMAND "test_program_scope_vars")
set_tests_properties("regression/test_program_scope_vars"
  PROPERTIES
    LABELS "level0"
    SKIP_RETURN_CODE 77)

# this test requires two USM devices
if(ENABLE_HOST_CPU_DEVICES AND ENABLE_LEVEL0)
  add_test(NAME "regression/test_issue_1548" COMMAND "test_issue_1548")
  set_tests_properties("regression/test_issue_1548" PROPERTIES
    LABELS "level0"
    ENVIRONMENT "POCL_DEVICES=level0 cpu"
    SKIP_RETURN_CODE 77)
endif()

add_test_pocl(NAME "regression/test_llvm_segfault_issue_889" COMMAND "test_llvm_segfault_issue_889")

# Also smoke test cq-tracing with this case.
add_test_pocl(NAME "regression/test_issue_1711" COMMAND "test_issue_1711" ENVIRONMENT "POCL_TRACING=cq")

add_test_pocl(NAME "regression/test_issue_1794a" COMMAND "test_issue_1794" "0" "${CMAKE_CURRENT_SOURCE_DIR}/test_issue_1794a.spv" LABELS ${SPIRV_LABELS})

add_test_pocl(NAME "regression/test_issue_1794b" COMMAND "test_issue_1794" "0" "${CMAKE_CURRENT_SOURCE_DIR}/test_issue_1794b.spv" LABELS ${SPIRV_LABELS})

add_test_pocl(NAME "regression/test_issue_1747" COMMAND "test_issue_1747" "0" "${CMAKE_CURRENT_SOURCE_DIR}/test_issue_1747.spv" LABELS ${SPIRV_LABELS})

add_test_pocl(NAME "regression/test_issue_893" COMMAND "test_issue_893")

add_test_pocl(NAME "regression/test_flatten_barrier_subs" COMMAND "test_flatten_barrier_subs" EXPECTED_OUTPUT "test_flatten_barrier_subs.output")

add_test_pocl(NAME "regression/issues_with_local_pointers" WORKITEM_HANDLER "loopvec;cbs;" COMMAND "test_locals")

add_test_pocl(NAME "regression/barrier_between_two_for_loops" WORKITEM_HANDLER "loopvec;cbs;" COMMAND "test_barrier_between_for_loops")

add_test_pocl(NAME "regression/simple_for-loop_with_a_barrier_inside" WORKITEM_HANDLER "loopvec;cbs;" COMMAND "test_simple_for_with_a_barrier")

add_test_pocl(NAME "regression/for-loop_with_computation_after_the_brexit" WORKITEM_HANDLER "loopvec;cbs;" COMMAND "test_multi_level_loops_with_barriers")

add_test_pocl(NAME "regression/for-loop_with_a_variable_iteration_count" WORKITEM_HANDLER "loopvec;cbs;" COMMAND "test_for_with_var_iteration_count")

add_test_pocl(NAME "regression/early_return_before_a_barrier_region" WORKITEM_HANDLER "loopvec;cbs;" COMMAND "test_early_return")

add_test_pocl(NAME "regression/id-dependent_computation_before_kernel_exit" WORKITEM_HANDLER "loopvec;cbs;" COMMAND "test_id_dependent_computation")

add_test_pocl(NAME "regression/barrier_just_before_return" WORKITEM_HANDLER "loopvec;cbs;" COMMAND "test_barrier_before_return")

add_test_pocl(NAME "regression/infinite_loop" WORKITEM_HANDLER "loopvec;cbs;" COMMAND "test_infinite_loop")

add_test_pocl(NAME "regression/undominated_variable_from_conditional_barrier_handling"
  WORKITEM_HANDLER "loopvec;cbs;" COMMAND "test_undominated_variable")

add_test_pocl(NAME "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local"
  WORKITEM_HANDLER "loopvec;cbs;" COMMAND "test_assign_loop_variable_to_privvar_makes_it_local")

add_test_pocl(NAME "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_2"
  WORKITEM_HANDLER "loopvec;cbs;" COMMAND "test_assign_loop_variable_to_privvar_makes_it_local_2")

add_test_pocl(NAME "regression/test_program_from_binary_with_local_1_1_1" WORKITEM_HANDLER "loopvec;cbs;"
  COMMAND "test_program_from_binary_with_local_1_1_1")

add_test_pocl(NAME "regression/test_rematerialized_alloca_load_with_outside_pr_users"
  WORKITEM_HANDLER "loopvec"
  COMMAND "test_rematerialized_alloca_load_with_outside_pr_users"
  EXPECTED_OUTPUT "test_rematerialized_alloca_load_with_outside_pr_users.output"
  LABELS "${SPIRV_LABELS}")

set(VARIANTS "loopvec;cbs;")
foreach(VARIANT ${VARIANTS})
set_tests_properties(
  "regression/issues_with_local_pointers_${VARIANT}"
  "regression/barrier_between_two_for_loops_${VARIANT}"
  "regression/simple_for-loop_with_a_barrier_inside_${VARIANT}"
  "regression/for-loop_with_computation_after_the_brexit_${VARIANT}"
  "regression/for-loop_with_a_variable_iteration_count_${VARIANT}"
  "regression/early_return_before_a_barrier_region_${VARIANT}"
  "regression/id-dependent_computation_before_kernel_exit_${VARIANT}"
  "regression/barrier_just_before_return_${VARIANT}"
  "regression/infinite_loop_${VARIANT}"
  "regression/undominated_variable_from_conditional_barrier_handling_${VARIANT}"
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_${VARIANT}"
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_2_${VARIANT}"
  "regression/test_program_from_binary_with_local_1_1_1_${VARIANT}"
  "regression/test_issue_1711_${VARIANT}"
  "regression/test_issue_1747_${VARIANT}"
  "regression/test_issue_1794a_${VARIANT}"
  "regression/test_issue_1794b_${VARIANT}"
  "regression/test_issue_1826_${VARIANT}"
  "regression/test_builtin_regression_${VARIANT}"
  "regression/test_mem_host_ptr_issue_${VARIANT}"
  "regression/test_issue_1962_${VARIANT}"
  PROPERTIES
    COST 1.5
    PROCESSORS 1
    DEPENDS "pocl_version_check"
    SKIP_RETURN_CODE 77
    SKIP_REGULAR_EXPRESSION "SKIP")

  set_tests_properties(
    "regression/test_program_from_binary_with_local_1_1_1_${VARIANT}"
    PROPERTIES
      ENVIRONMENT "POCL_WORK_GROUP_SPECIALIZATION=0"
   )
endforeach()


add_test_pocl(NAME "regression/test_alignment_with_dynamic_wg_114" COMMAND "test_alignment_with_dynamic_wg" 1 1 4)

add_test_pocl(NAME "regression/test_alignment_with_dynamic_wg_117" COMMAND "test_alignment_with_dynamic_wg" 1 1 7)

add_test_pocl(NAME "regression/test_alignment_with_dynamic_wg_225" COMMAND "test_alignment_with_dynamic_wg" 2 2 5)

add_test_pocl(NAME "regression/test_alignment_with_dynamic_wg_173" COMMAND "test_alignment_with_dynamic_wg" 1 7 3)

add_test_pocl(NAME "regression/test_alignment_with_dynamic_wg_183" COMMAND "test_alignment_with_dynamic_wg" 1 8 3)

add_test_pocl(NAME "regression/test_alignment_with_dynamic_wg_283" COMMAND "test_alignment_with_dynamic_wg" 2 8 3)

add_test_pocl(NAME "regression/test_alignment_with_dynamic_wg_332" COMMAND "test_alignment_with_dynamic_wg" 3 3 2)

add_test_pocl(NAME "regression/test_alignment_with_dynamic_wg_323" COMMAND "test_alignment_with_dynamic_wg" 3 2 3)

add_test_pocl(NAME "regression/test_alignment_with_dynamic_wg2" COMMAND "test_alignment_with_dynamic_wg2")

add_test_pocl(NAME "regression/test_alignment_with_dynamic_wg3" COMMAND "test_alignment_with_dynamic_wg3")

set(VARIANTS "loopvec;cbs")
foreach(VARIANT ${VARIANTS})
set_tests_properties(
  "regression/test_alignment_with_dynamic_wg_114_${VARIANT}"
  "regression/test_alignment_with_dynamic_wg_117_${VARIANT}"
  "regression/test_alignment_with_dynamic_wg_225_${VARIANT}"
  "regression/test_alignment_with_dynamic_wg_173_${VARIANT}"
  "regression/test_alignment_with_dynamic_wg_183_${VARIANT}"
  "regression/test_alignment_with_dynamic_wg_283_${VARIANT}"
  "regression/test_alignment_with_dynamic_wg_332_${VARIANT}"
  "regression/test_alignment_with_dynamic_wg_323_${VARIANT}"
  "regression/test_alignment_with_dynamic_wg2_${VARIANT}"
  "regression/test_alignment_with_dynamic_wg3_${VARIANT}"

  PROPERTIES
    COST 1.5
    PROCESSORS 1
    DEPENDS "pocl_version_check")
endforeach()

# other

add_test_pocl(NAME "regression/setting_a_buffer_argument_to_NULL_causes_a_segfault" COMMAND "test_null_arg")

add_test_pocl(NAME "regression/clSetKernelArg_overwriting_the_previous_kernel's_args" COMMAND "test_setargs")

add_test_pocl(NAME "regression/passing_a_constant_array_as_an_arg" COMMAND "test_constant_array")

add_test_pocl(NAME "regression/case_with_multiple_variable_length_loops_and_a_barrier_in_one" COMMAND "test_fors_with_var_iteration_counts")

add_test_pocl(NAME "regression/autolocals_in_constexprs" COMMAND "test_autolocals_in_constexprs")

add_test_pocl(NAME "regression/struct_kernel_arguments" COMMAND "test_structs_as_args")

add_test_pocl(NAME "regression/vector_kernel_arguments" COMMAND "test_vectors_as_args")

# Label tests that also work with TCE
set(TCE_TESTS "regression/barrier_between_two_for_loops_loopvec"
  "regression/for-loop_with_computation_after_the_brexit_loopvec"
  "regression/early_return_before_a_barrier_region_loopvec")

# TODO fails with TCE + LLVM 6, issue #609
# it seems to have worked with LLVM up to 16/17; fails again with PR #1613
list(REMOVE_ITEM TCE_TESTS "regression/barrier_between_two_for_loops_loopvec" )

set_property(TEST ${TCE_TESTS} APPEND PROPERTY LABELS "tce")

set(VARIANTS "loopvec;cbs")
foreach(VARIANT ${VARIANTS})
  set_tests_properties("regression/setting_a_buffer_argument_to_NULL_causes_a_segfault_${VARIANT}"
    "regression/clSetKernelArg_overwriting_the_previous_kernel's_args_${VARIANT}"
    "regression/passing_a_constant_array_as_an_arg_${VARIANT}"
    "regression/case_with_multiple_variable_length_loops_and_a_barrier_in_one_${VARIANT}"
    "regression/struct_kernel_arguments_${VARIANT}" "regression/vector_kernel_arguments_${VARIANT}"
    "regression/autolocals_in_constexprs_${VARIANT}" "regression/test_issue_231_${VARIANT}"
    "regression/test_issue_445_${VARIANT}" "regression/test_issue_553_${VARIANT}"
    "regression/test_issue_577_${VARIANT}" "regression/test_issue_757_${VARIANT}"
    "regression/test_llvm_segfault_issue_889_${VARIANT}"
    "regression/test_issue_893_${VARIANT}" "regression/test_issue_1435_${VARIANT}"
    "regression/test_flatten_barrier_subs_${VARIANT}"
    "regression/test_workitem_func_outside_kernel_${VARIANT}"
    "regression/test_program_scope_vars"
    ${TCE_TESTS}
    PROPERTIES
      COST 1.5
      PROCESSORS 1
      SKIP_RETURN_CODE 77
      SKIP_REGULAR_EXPRESSION "SKIP"
      DEPENDS "pocl_version_check")
endforeach()


# Label tests that fail with MinGW build
set_property(TEST
  "regression/test_flatten_barrier_subs_loopvec"
  "regression/test_flatten_barrier_subs_cbs"
  "regression/infinite_loop_loopvec"
  "regression/infinite_loop_cbs"
  APPEND PROPERTY LABELS "mingw_fail")

# Label tests that work with CUDA backend
set_property(TEST
  "regression/issues_with_local_pointers_loopvec"
  "regression/barrier_between_two_for_loops_loopvec"
  "regression/simple_for-loop_with_a_barrier_inside_loopvec"
  "regression/for-loop_with_computation_after_the_brexit_loopvec"
  "regression/for-loop_with_a_variable_iteration_count_loopvec"
  "regression/early_return_before_a_barrier_region_loopvec"
  "regression/barrier_just_before_return_loopvec"
  "regression/undominated_variable_from_conditional_barrier_handling_loopvec"
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_loopvec"
  "regression/test_program_from_binary_with_local_1_1_1_loopvec"
  "regression/setting_a_buffer_argument_to_NULL_causes_a_segfault_loopvec"
  "regression/clSetKernelArg_overwriting_the_previous_kernel's_args_loopvec"
  "regression/vector_kernel_arguments_loopvec"
  APPEND PROPERTY LABELS "cuda")

set_property(TEST
   "regression/case_with_multiple_variable_length_loops_and_a_barrier_in_one_loopvec"
  APPEND PROPERTY LABELS "cuda")

# Label tests that work with HSA runtime with native kernel compilation
set_property(TEST
  "regression/issues_with_local_pointers_loopvec"
  "regression/barrier_between_two_for_loops_loopvec"
  "regression/simple_for-loop_with_a_barrier_inside_loopvec"
  "regression/for-loop_with_computation_after_the_brexit_loopvec"
  "regression/for-loop_with_a_variable_iteration_count_loopvec"
  "regression/id-dependent_computation_before_kernel_exit_loopvec"
  "regression/barrier_just_before_return_loopvec"
  "regression/undominated_variable_from_conditional_barrier_handling_loopvec"
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_loopvec"
  "regression/setting_a_buffer_argument_to_NULL_causes_a_segfault_loopvec"
  "regression/early_return_before_a_barrier_region_loopvec"
  "regression/autolocals_in_constexprs_loopvec"
  "regression/test_issue_445_loopvec"
  "regression/test_issue_553_loopvec"
  "regression/test_issue_577_loopvec"
  APPEND PROPERTY LABELS "hsa-native")

# Label tests that work with proxy driver
set_property(TEST
  "regression/barrier_between_two_for_loops_loopvec"
  "regression/simple_for-loop_with_a_barrier_inside_loopvec"
  "regression/for-loop_with_computation_after_the_brexit_loopvec"
  "regression/for-loop_with_a_variable_iteration_count_loopvec"
  "regression/id-dependent_computation_before_kernel_exit_loopvec"
  "regression/barrier_just_before_return_loopvec"
  "regression/undominated_variable_from_conditional_barrier_handling_loopvec"
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_loopvec"
  "regression/setting_a_buffer_argument_to_NULL_causes_a_segfault_loopvec"
  "regression/early_return_before_a_barrier_region_loopvec"
  "regression/autolocals_in_constexprs_loopvec"
  "regression/test_issue_445_loopvec"
  "regression/test_issue_553_loopvec"
  APPEND PROPERTY LABELS "proxy")

# Label tests that work with Vulkan
set_property(TEST
  "regression/issues_with_local_pointers_loopvec"
  "regression/for-loop_with_computation_after_the_brexit_loopvec"
  "regression/early_return_before_a_barrier_region_loopvec"
  "regression/test_alignment_with_dynamic_wg3_loopvec"
  "regression/clSetKernelArg_overwriting_the_previous_kernel's_args_loopvec"
  APPEND PROPERTY LABELS "vulkan")

if(ENABLE_LEVEL0)
set_property(TEST
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_loopvec"
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_2_loopvec"
  "regression/barrier_between_two_for_loops_loopvec"
  "regression/barrier_just_before_return_loopvec"
  "regression/early_return_before_a_barrier_region_loopvec"
  "regression/for-loop_with_computation_after_the_brexit_loopvec"
  "regression/for-loop_with_a_variable_iteration_count_loopvec"
  "regression/id-dependent_computation_before_kernel_exit_loopvec"
  "regression/issues_with_local_pointers_loopvec"
  "regression/simple_for-loop_with_a_barrier_inside_loopvec"
  "regression/test_program_from_binary_with_local_1_1_1_loopvec"
  "regression/undominated_variable_from_conditional_barrier_handling_loopvec"


  "regression/autolocals_in_constexprs_loopvec"
  "regression/case_with_multiple_variable_length_loops_and_a_barrier_in_one_loopvec"
  "regression/clSetKernelArg_overwriting_the_previous_kernel's_args_loopvec"
  "regression/setting_a_buffer_argument_to_NULL_causes_a_segfault_loopvec"
  "regression/struct_kernel_arguments_loopvec"
  "regression/vector_kernel_arguments_loopvec"

  "regression/test_workitem_func_outside_kernel_loopvec"
  "regression/test_issue_893_loopvec"
  "regression/test_flatten_barrier_subs_loopvec"

  "regression/test_alignment_with_dynamic_wg_114_loopvec"
  "regression/test_alignment_with_dynamic_wg_117_loopvec"
  "regression/test_alignment_with_dynamic_wg_225_loopvec"
  "regression/test_alignment_with_dynamic_wg_173_loopvec"
  "regression/test_alignment_with_dynamic_wg_183_loopvec"
  "regression/test_alignment_with_dynamic_wg_283_loopvec"
  "regression/test_alignment_with_dynamic_wg_332_loopvec"
  "regression/test_alignment_with_dynamic_wg_323_loopvec"
  "regression/test_alignment_with_dynamic_wg2_loopvec"
  "regression/test_alignment_with_dynamic_wg3_loopvec"


  "regression/test_issue_231_loopvec"
  "regression/test_issue_445_loopvec"
  "regression/test_issue_553_loopvec"
  "regression/test_issue_577_loopvec"

  APPEND PROPERTY LABELS "level0")

if(NOT ENABLE_CONFORMANCE)
set_property(TEST
  "regression/infinite_loop_loopvec"
  "regression/test_issue_757_loopvec"
  "regression/passing_a_constant_array_as_an_arg_loopvec"
  APPEND PROPERTY LABELS "level0")
endif()

endif()

###################################################################

# TODO infinite loop test: AT_SKIP_IF([ env | grep -q POCL_IMPLICIT_FINISH])

set_tests_properties(
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_2_loopvec"
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_2_cbs"
    PROPERTIES PASS_REGULAR_EXPRESSION
"changing the value at global_id: 6, local_id 2, group_id 1, to: 3
value is changed at global_id: 6, local_id 2, group_id 1, to: 3
")

# Produced wrong results with LLVM 14. Works with all newer versions.
# No point putting time for debugging.
if(LLVM_VERSION_MAJOR EQUAL 14)
  set_tests_properties(
    "regression/test_alignment_with_dynamic_wg_225_cbs"
    PROPERTIES DISABLED 1)
endif()

