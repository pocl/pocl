#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2024 pocl developers
#                 2024-2025 Pekka Jääskeläinen / Intel Finland Oy
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


add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
include_directories("${CMAKE_SOURCE_DIR}/lib/CL")
add_compile_options(${OPENCL_CFLAGS})

add_executable("run_kernel" "run_kernel.c")
target_link_libraries("run_kernel" ${POCLU_LINK_OPTIONS})

add_test_pocl(NAME "workgroup/different_implicit_barrier_injection_scenarios"
              EXPECTED_OUTPUT "implicit_barriers_1_2_1_1.stdout"
              COMMAND "run_kernel" "implicit_barriers.cl" 1 2 1 1
              WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "workgroup/unbarriered_for_loops"
              EXPECTED_OUTPUT "forloops_2_2_1_1.stdout"
              COMMAND "run_kernel" "forloops.cl" 2 2 1 1)

add_test_pocl(NAME "workgroup/barriered_for_loops"
              EXPECTED_OUTPUT "loopbarriers_1_4_1_1.stdout"
              COMMAND "run_kernel" "loopbarriers.cl" 1 4 1 1)

add_test_pocl(NAME "workgroup/switch_case"
              EXPECTED_OUTPUT "switch_case_1_4_1_1.stdout"
              COMMAND "run_kernel" "switch_case.cl" 1 4 1 1)

add_test_pocl(NAME "workgroup/b_loop_with_none_of_the_WIs_reaching_the_barrier"
              EXPECTED_OUTPUT "tricky_for_1_2_1_1.stdout"
              COMMAND "run_kernel" "tricky_for.cl" 1 2 1 1)

add_test_pocl(NAME "workgroup/for_with_divergent_return"
              EXPECTED_OUTPUT "for_with_divergent_return_1_6_1_1.stdout"
              COMMAND "run_kernel" "for_with_divergent_return.cl" 1 6 1 1)

add_test_pocl(NAME "workgroup/cond_barriers_in_for"
              EXPECTED_OUTPUT "cond_barriers_in_for_2_4_1_1.stdout"
              COMMAND "run_kernel" "cond_barriers_in_for.cl" 2 4 1 1)

add_test_pocl(NAME "workgroup/cond_barrier_in_var_for"
              EXPECTED_OUTPUT "cond_barrier_in_var_for_2_4_1_1.stdout"
              COMMAND "run_kernel" "cond_barrier_in_var_for.cl" 2 4 1 1
              WORKITEM_HANDLER "cbs")

# Cases which are not dependent on the work-group or work-item
# execution (printout) order or the method (use the default method
# for the device).
set(VARIANTS "loopvec;cbs")
foreach(VARIANT ${VARIANTS})
  set_tests_properties(
    "workgroup/unbarriered_for_loops_${VARIANT}"
    "workgroup/barriered_for_loops_${VARIANT}"
    "workgroup/switch_case_${VARIANT}"
    "workgroup/b_loop_with_none_of_the_WIs_reaching_the_barrier_${VARIANT}"
    "workgroup/for_with_divergent_return_${VARIANT}"
    "workgroup/cond_barriers_in_for_${VARIANT}"
    PROPERTIES
      COST 2.0
      PROCESSORS 1
      DEPENDS "pocl_version_check")
endforeach()

set_tests_properties(
  "workgroup/different_implicit_barrier_injection_scenarios"
  "workgroup/cond_barrier_in_var_for"
  PROPERTIES
    COST 2.0
    PROCESSORS 1
    DEPENDS "pocl_version_check")

add_test_pocl(NAME "workgroup/unconditional_barriers"
              EXPECTED_OUTPUT "basic_barriers_2_2_2_2.stdout"
              COMMAND "run_kernel" "basic_barriers.cl" 2 2 2 2)

add_test_pocl(NAME "workgroup/conditional_barrier_loopvec"
              EXPECTED_OUTPUT "cond_barriers_1_2_1_1_loopvec.stdout"
              COMMAND "run_kernel" "conditional_barriers.cl" 1 2 1 1
              WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "workgroup/conditional_barrier_cbs"
              EXPECTED_OUTPUT "cond_barriers_1_2_1_1_cbs.stdout"
              COMMAND "run_kernel" "conditional_barriers.cl" 1 2 1 1
              WORKITEM_HANDLER "cbs")

add_test_pocl(NAME "workgroup/conditional_barrier_dynamic"
              EXPECTED_OUTPUT "cond_barriers_dynamic_1_16_1_1_loopvec.stdout"
              COMMAND "run_kernel" "conditional_barrier_dynamic.cl" 1 16 1 1
              WORKITEM_HANDLER "loopvec"
              LABELS "cpu_fail")

add_test_pocl(NAME "workgroup/forcing_horizontal_parallelization_to_some_outer_loopvec"
              EXPECTED_OUTPUT "outerlooppar_2_2_1_1.stdout"
              COMMAND "run_kernel" "outerlooppar.cl" 2 2 1 1
              WORKITEM_HANDLER "loopvec")

add_test_pocl(NAME "workgroup/loop_with_two_paths_to_the_latch"
              EXPECTED_OUTPUT "for_bug_1_2_1_1.stdout"
              COMMAND "run_kernel" "for_bug.cl" 1 2 1 1)

add_test_pocl(NAME "workgroup/b_loop_with_two_latches"
              EXPECTED_OUTPUT "multilatch_bloop_1_3_1_1.stdout"
              COMMAND "run_kernel" "multilatch_bloop.cl" 1 3 1 1)

add_test_pocl(NAME "workgroup/workgroup_sizes_work_items_get_wrong_ids"
              SORT_OUTPUT
              EXPECTED_OUTPUT "print_all_ids_114114.txt"
              COMMAND "run_kernel" "print_all_ids.cl" 1 1 1 4)

add_test_pocl(NAME "workgroup/issue_548_convergent_propagation"
              EXPECTED_OUTPUT "issue_548_1_2_1_1.stdout"
              COMMAND "run_kernel" "issue_548.cl" "1" "2" "1" "1")

add_test_pocl(NAME "workgroup/issue_1747_wg_divergent_barrier"
              EXPECTED_OUTPUT "issue_1747.stdout"
              COMMAND "run_kernel" "issue_1747.cl" "2" "2" "1" "1")

add_test_pocl(NAME "workgroup/range_md_small_grid"
              EXPECTED_OUTPUT "range_md_small_grid.stdout"
              COMMAND "run_kernel" "range_md.cl" "1" "13" "1" "1")

add_test_pocl(NAME "workgroup/range_md_large_grid"
              EXPECTED_OUTPUT ""
              COMMAND "run_kernel" "range_md.cl" "1000" "128" "1" "1")

if(ENABLE_CONFORMANCE)
  set(BALLOT_LABELS "cpu_fail")
else()
  set(BALLOT_LABELS "")
endif()

add_test_pocl(NAME "workgroup/ballot"
              EXPECTED_OUTPUT "ballot_1_32_1_1.stdout"
              COMMAND "run_kernel" "ballot.cl" "1" "32" "1" "1"
              LABELS ${BALLOT_LABELS})

# These tests are now always ran with the basic device with a predefined
# work-group execution order. Their printout verification depends
# on it.
set(VARIANTS "loopvec;cbs")
foreach(VARIANT ${VARIANTS})
  set(TEST_LIST     "workgroup/unconditional_barriers_${VARIANT}"
  "workgroup/conditional_barrier_${VARIANT}"
  "workgroup/forcing_horizontal_parallelization_to_some_outer_loopvec"
  "workgroup/loop_with_two_paths_to_the_latch_${VARIANT}"
  "workgroup/b_loop_with_two_latches_${VARIANT}"
  "workgroup/workgroup_sizes_work_items_get_wrong_ids_${VARIANT}"
  "workgroup/issue_548_convergent_propagation_${VARIANT}"
  "workgroup/range_md_small_grid_${VARIANT}"
  "workgroup/range_md_large_grid_${VARIANT}"
  "workgroup/issue_1747_wg_divergent_barrier_${VARIANT}"
  )
  set_tests_properties(
    ${TEST_LIST}
    PROPERTIES
      COST 2.0
      PROCESSORS 1
      DEPENDS "pocl_version_check")
  set_property(TEST ${TEST_LIST}
    APPEND PROPERTY ENVIRONMENT "POCL_MAX_COMPUTE_UNITS=1")
endforeach()

set_property(TEST
  "workgroup/issue_548_convergent_propagation_loopvec"
  "workgroup/issue_548_convergent_propagation_cbs"
  APPEND PROPERTY LABELS "mingw_fail")

set_property(TEST
  "workgroup/unconditional_barriers_loopvec"
  "workgroup/unbarriered_for_loops_loopvec"
  "workgroup/barriered_for_loops_loopvec"
  "workgroup/conditional_barrier_loopvec"
  "workgroup/b_loop_with_none_of_the_WIs_reaching_the_barrier_loopvec"
  "workgroup/forcing_horizontal_parallelization_to_some_outer_loopvec"
  "workgroup/loop_with_two_paths_to_the_latch_loopvec"
  "workgroup/b_loop_with_two_latches_loopvec"
  "workgroup/workgroup_sizes_work_items_get_wrong_ids_loopvec"
  "workgroup/issue_548_convergent_propagation_loopvec"
  "workgroup/different_implicit_barrier_injection_scenarios"
  "workgroup/cond_barrier_in_var_for"
  APPEND PROPERTY LABELS "hsa-native")

set_property(TEST
  "workgroup/unbarriered_for_loops_loopvec"
  "workgroup/barriered_for_loops_loopvec"
  "workgroup/b_loop_with_none_of_the_WIs_reaching_the_barrier_loopvec"
  APPEND PROPERTY LABELS "proxy")

set_property(TEST
  "workgroup/unbarriered_for_loops_loopvec"
  "workgroup/barriered_for_loops_loopvec"
  "workgroup/switch_case_loopvec"
  "workgroup/b_loop_with_none_of_the_WIs_reaching_the_barrier_loopvec"
  APPEND PROPERTY LABELS "vulkan")

# broken:  "workgroup/for_with_divergent_return"
set_property(TEST
  "workgroup/unbarriered_for_loops_loopvec"
  "workgroup/barriered_for_loops_loopvec"
  "workgroup/switch_case_loopvec"
  "workgroup/b_loop_with_none_of_the_WIs_reaching_the_barrier_loopvec"
  "workgroup/cond_barriers_in_for_loopvec"
  APPEND PROPERTY LABELS "level0")

add_symlink_to_built_opencl_dynlib(run_kernel)
