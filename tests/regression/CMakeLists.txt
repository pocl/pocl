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


# Mac OS X currently can't digest cl2.hpp, which all reg tests include
if(NOT APPLE)

set(C_PROGRAMS_TO_BUILD test_assign_loop_variable_to_privvar_makes_it_local
     test_program_from_binary_with_local_1_1_1
     test_assign_loop_variable_to_privvar_makes_it_local_2)
foreach(PROG ${C_PROGRAMS_TO_BUILD})
  if(MSVC)
    set_source_files_properties( "${PROG}.c" PROPERTIES LANGUAGE CXX )
  endif(MSVC)
  add_executable("${PROG}" "${PROG}.c")
  target_link_libraries("${PROG}" ${POCLU_LINK_OPTIONS})
endforeach()


set(PROGRAMS_TO_BUILD test_barrier_between_for_loops test_early_return
  test_for_with_var_iteration_count test_id_dependent_computation
  test_locals test_loop_phi_replication test_multi_level_loops_with_barriers
  test_simple_for_with_a_barrier test_structs_as_args test_vectors_as_args
  test_barrier_before_return test_infinite_loop test_constant_array
  test_undominated_variable test_setargs test_null_arg
  test_fors_with_var_iteration_counts test_issue_231 test_issue_445
  test_autolocals_in_constexprs test_issue_553 test_issue_577)

if (MSVC)
  add_compile_options(${OPENCL_CFLAGS})
else ()
  add_compile_options("-std=c++11" "-Wno-deprecated" "-Wno-deprecated-declarations" ${OPENCL_CFLAGS})
endif ()
 
add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
include_directories("${CMAKE_SOURCE_DIR}/lib/CL")

foreach(PROG ${PROGRAMS_TO_BUILD})
  add_executable("${PROG}" "${PROG}.cpp")
  target_link_libraries("${PROG}" ${POCLU_LINK_OPTIONS})
endforeach()


######################################################################

add_test_pocl(NAME "regression/test_issue_231" COMMAND "test_issue_231")
add_test_pocl(NAME "regression/test_issue_445" COMMAND "test_issue_445")
add_test_pocl(NAME "regression/test_issue_553" COMMAND "test_issue_553")
add_test_pocl(NAME "regression/test_issue_577" COMMAND "test_issue_577")


# repl

add_test_pocl(NAME "regression/phi_nodes_not_replicated_REPL" COMMAND "test_loop_phi_replication")

add_test_pocl(NAME "regression/issues_with_local_pointers_REPL" COMMAND "test_locals")

add_test_pocl(NAME "regression/barrier_between_two_for_loops_REPL" COMMAND "test_barrier_between_for_loops")

add_test_pocl(NAME "regression/simple_for-loop_with_a_barrier_inside_REPL" COMMAND "test_simple_for_with_a_barrier")

add_test_pocl(NAME "regression/for-loop_with_computation_after_the_brexit_REPL" COMMAND "test_multi_level_loops_with_barriers")

add_test_pocl(NAME "regression/for-loop_with_a_variable_iteration_count_REPL" COMMAND "test_for_with_var_iteration_count")

add_test_pocl(NAME "regression/early_return_before_a_barrier_region_REPL" COMMAND "test_early_return")

add_test_pocl(NAME "regression/id-dependent_computation_before_kernel_exit_REPL" COMMAND "test_id_dependent_computation")

add_test_pocl(NAME "regression/barrier_just_before_return_REPL" COMMAND "test_barrier_before_return")

add_test_pocl(NAME "regression/infinite_loop_REPL" COMMAND "test_infinite_loop")

add_test_pocl(NAME "regression/undominated_variable_from_conditional_barrier_handling_REPL" COMMAND "test_undominated_variable")

add_test_pocl(NAME "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_REPL"
              COMMAND "test_assign_loop_variable_to_privvar_makes_it_local")

add_test_pocl(NAME "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_2_REPL"
              COMMAND "test_assign_loop_variable_to_privvar_makes_it_local_2")

add_test_pocl(NAME "regression/test_program_from_binary_with_local_1_1_1_REPL"
              COMMAND "test_program_from_binary_with_local_1_1_1")

set_tests_properties("regression/phi_nodes_not_replicated_REPL"
  "regression/issues_with_local_pointers_REPL"
  "regression/barrier_between_two_for_loops_REPL"
  "regression/simple_for-loop_with_a_barrier_inside_REPL"
  "regression/for-loop_with_computation_after_the_brexit_REPL"
  "regression/for-loop_with_a_variable_iteration_count_REPL"
  "regression/early_return_before_a_barrier_region_REPL"
  "regression/id-dependent_computation_before_kernel_exit_REPL"
  "regression/barrier_just_before_return_REPL"
  "regression/infinite_loop_REPL"
  "regression/undominated_variable_from_conditional_barrier_handling_REPL"
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_REPL"
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_2_REPL"
  "regression/test_program_from_binary_with_local_1_1_1_REPL"
  PROPERTIES
    ENVIRONMENT "POCL_WORK_GROUP_METHOD=workitemrepl"
    COST 1.5
    PROCESSORS 1
    DEPENDS "pocl_version_check"
    LABELS "internal;regression")


# loops

add_test_pocl(NAME "regression/phi_nodes_not_replicated_LOOPS" COMMAND "test_loop_phi_replication")

add_test_pocl(NAME "regression/issues_with_local_pointers_LOOPS" COMMAND "test_locals")

add_test_pocl(NAME "regression/barrier_between_two_for_loops_LOOPS" COMMAND "test_barrier_between_for_loops")

add_test_pocl(NAME "regression/simple_for-loop_with_a_barrier_inside_LOOPS" COMMAND "test_simple_for_with_a_barrier")

add_test_pocl(NAME "regression/for-loop_with_computation_after_the_brexit_LOOPS" COMMAND "test_multi_level_loops_with_barriers")

add_test_pocl(NAME "regression/for-loop_with_a_variable_iteration_count_LOOPS" COMMAND "test_for_with_var_iteration_count")

add_test_pocl(NAME "regression/early_return_before_a_barrier_region_LOOPS" COMMAND "test_early_return")

add_test_pocl(NAME "regression/id-dependent_computation_before_kernel_exit_LOOPS" COMMAND "test_id_dependent_computation")

add_test_pocl(NAME "regression/barrier_just_before_return_LOOPS" COMMAND "test_barrier_before_return")

add_test_pocl(NAME "regression/infinite_loop_LOOPS" COMMAND "test_infinite_loop")

add_test_pocl(NAME "regression/undominated_variable_from_conditional_barrier_handling_LOOPS" COMMAND "test_undominated_variable")

add_test_pocl(NAME "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_LOOPS"
              COMMAND "test_assign_loop_variable_to_privvar_makes_it_local")

add_test_pocl(NAME "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_2_LOOPS"
              COMMAND "test_assign_loop_variable_to_privvar_makes_it_local_2")

add_test_pocl(NAME "regression/test_program_from_binary_with_local_1_1_1_LOOPS"
              COMMAND "test_program_from_binary_with_local_1_1_1")

set_tests_properties("regression/phi_nodes_not_replicated_LOOPS"
  "regression/issues_with_local_pointers_LOOPS"
  "regression/barrier_between_two_for_loops_LOOPS"
  "regression/simple_for-loop_with_a_barrier_inside_LOOPS"
  "regression/for-loop_with_computation_after_the_brexit_LOOPS"
  "regression/for-loop_with_a_variable_iteration_count_LOOPS"
  "regression/early_return_before_a_barrier_region_LOOPS"
  "regression/id-dependent_computation_before_kernel_exit_LOOPS"
  "regression/barrier_just_before_return_LOOPS"
  "regression/infinite_loop_LOOPS"
  "regression/undominated_variable_from_conditional_barrier_handling_LOOPS"
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_LOOPS"
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_2_LOOPS"
  "regression/test_program_from_binary_with_local_1_1_1_LOOPS"
  PROPERTIES
    ENVIRONMENT "POCL_WORK_GROUP_METHOD=workitemloops"
    COST 1.5
    PROCESSORS 1
    DEPENDS "pocl_version_check"
    LABELS "internal;regression")


# other

add_test_pocl(NAME "regression/setting_a_buffer_argument_to_NULL_causes_a_segfault" COMMAND "test_null_arg")

add_test_pocl(NAME "regression/clSetKernelArg_overwriting_the_previous_kernel's_args" COMMAND "test_setargs")

add_test_pocl(NAME "regression/passing_a_constant_array_as_an_arg" COMMAND "test_constant_array")

add_test_pocl(NAME "regression/case_with_multiple_variable_length_loops_and_a_barrier_in_one" COMMAND "test_fors_with_var_iteration_counts")

add_test_pocl(NAME "regression/autolocals_in_constexprs" COMMAND "test_autolocals_in_constexprs")

add_test_pocl(NAME "regression/struct_kernel_arguments" COMMAND "test_structs_as_args")

add_test_pocl(NAME "regression/vector_kernel_arguments" COMMAND "test_vectors_as_args")

set(TCE_TESTS "regression/barrier_between_two_for_loops_LOOPS"
  "regression/barrier_between_two_for_loops_REPL"
  "regression/for-loop_with_computation_after_the_brexit_REPL"
  "regression/for-loop_with_computation_after_the_brexit_LOOPS"
  "regression/early_return_before_a_barrier_region_REPL"
  "regression/early_return_before_a_barrier_region_LOOPS")

set_tests_properties("regression/setting_a_buffer_argument_to_NULL_causes_a_segfault"
  "regression/clSetKernelArg_overwriting_the_previous_kernel's_args"
  "regression/passing_a_constant_array_as_an_arg"
  "regression/case_with_multiple_variable_length_loops_and_a_barrier_in_one"
  "regression/struct_kernel_arguments" "regression/vector_kernel_arguments"
  "regression/autolocals_in_constexprs"
  ${TCE_TESTS}
  PROPERTIES
    COST 1.5
    PROCESSORS 1
    DEPENDS "pocl_version_check"
    LABELS "internal;regression")

# Label tests that also work with TCE

# TODO fails with TCE + LLVM 6, issue #609
if(NOT LLVM_OLDER_THAN_6_0)
list(REMOVE_ITEM TCE_TESTS
  "regression/barrier_between_two_for_loops_LOOPS" )
endif()

set_property(TEST ${TCE_TESTS} APPEND PROPERTY LABELS "tce")

# Label tests that work with CUDA backend
set_property(TEST
  "regression/phi_nodes_not_replicated_REPL"
  "regression/issues_with_local_pointers_REPL"
  "regression/barrier_between_two_for_loops_REPL"
  "regression/simple_for-loop_with_a_barrier_inside_REPL"
  "regression/for-loop_with_computation_after_the_brexit_REPL"
  "regression/for-loop_with_a_variable_iteration_count_REPL"
  "regression/early_return_before_a_barrier_region_REPL"
  "regression/id-dependent_computation_before_kernel_exit_REPL"
  "regression/barrier_just_before_return_REPL"
  "regression/undominated_variable_from_conditional_barrier_handling_REPL"
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_REPL"
  "regression/test_program_from_binary_with_local_1_1_1_REPL"
  "regression/phi_nodes_not_replicated_LOOPS"
  "regression/issues_with_local_pointers_LOOPS"
  "regression/barrier_between_two_for_loops_LOOPS"
  "regression/simple_for-loop_with_a_barrier_inside_LOOPS"
  "regression/for-loop_with_computation_after_the_brexit_LOOPS"
  "regression/for-loop_with_a_variable_iteration_count_LOOPS"
  "regression/early_return_before_a_barrier_region_LOOPS"
  "regression/id-dependent_computation_before_kernel_exit_LOOPS"
  "regression/barrier_just_before_return_LOOPS"
  "regression/undominated_variable_from_conditional_barrier_handling_LOOPS"
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_LOOPS"
  "regression/test_program_from_binary_with_local_1_1_1_LOOPS"
  "regression/setting_a_buffer_argument_to_NULL_causes_a_segfault"
  "regression/clSetKernelArg_overwriting_the_previous_kernel's_args"
  "regression/case_with_multiple_variable_length_loops_and_a_barrier_in_one"
  "regression/vector_kernel_arguments"
  "regression/autolocals_in_constexprs"
  APPEND PROPERTY LABELS "cuda")

if (NOT CLANG_IS_PATCHED_FOR_SPIR_CC)
set_tests_properties("regression/struct_kernel_arguments"
                      PROPERTIES  WILL_FAIL 1)

if(ARM64)
  set_tests_properties("regression/vector_kernel_arguments"
    PROPERTIES  WILL_FAIL 1)
endif()

endif()

###################################################################

# TODO infinite loop test: AT_SKIP_IF([ env | grep -q POCL_IMPLICIT_FINISH])

set_tests_properties(
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_2_REPL"
  "regression/assigning_a_loop_iterator_variable_to_a_private_makes_it_local_2_LOOPS"
    PROPERTIES PASS_REGULAR_EXPRESSION
"changing the value at global_id: 6, local_id 2, group_id 1, to: 3
value is changed at global_id: 6, local_id 2, group_id 1, to: 3
")

endif()
