#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2019 pocl developers
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

if(MSVC)
  set_source_files_properties( run_kernel.c PROPERTIES LANGUAGE CXX )
endif(MSVC)
add_executable("run_kernel" "run_kernel.c")
target_link_libraries("run_kernel" ${POCLU_LINK_OPTIONS})

add_test_pocl(NAME "workgroup/different_implicit_barrier_injection_scenarios"
              EXPECTED_OUTPUT "implicit_barriers_1_2_1_1.stdout"
              COMMAND "run_kernel" "implicit_barriers.cl" 1 2 1 1)

add_test_pocl(NAME "workgroup/unconditional_barriers"
              EXPECTED_OUTPUT "basic_barriers_2_2_2_2.stdout"
              COMMAND "run_kernel" "basic_barriers.cl" 2 2 2 2)

add_test_pocl(NAME "workgroup/unbarriered_for_loops"
              EXPECTED_OUTPUT "forloops_2_2_1_1.stdout"
              COMMAND "run_kernel" "forloops.cl" 2 2 1 1)

add_test_pocl(NAME "workgroup/barriered_for_loops"
              EXPECTED_OUTPUT "loopbarriers_1_4_1_1.stdout"
              COMMAND "run_kernel" "loopbarriers.cl" 1 4 1 1)

add_test_pocl(NAME "workgroup/switch_case"
              EXPECTED_OUTPUT "switch_case_1_4_1_1.stdout"
              COMMAND "run_kernel" "switch_case.cl" 1 4 1 1)

add_test_pocl(NAME "workgroup/conditional_barrier"
              EXPECTED_OUTPUT "cond_barriers_1_2_1_1.stdout"
              COMMAND "run_kernel" "conditional_barriers.cl" 1 2 1 1)

add_test_pocl(NAME "workgroup/b_loop_with_none_of_the_WIs_reaching_the_barrier"
              EXPECTED_OUTPUT "tricky_for_1_2_1_1.stdout"
              COMMAND "run_kernel" "tricky_for.cl" 1 2 1 1)

add_test_pocl(NAME "workgroup/for_with_divergent_return"
              EXPECTED_OUTPUT "for_with_divergent_return_1_6_1_1.stdout"
              COMMAND "run_kernel" "for_with_divergent_return.cl" 1 6 1 1)

add_test_pocl(NAME "workgroup/forcing_horizontal_parallelization_to_some_outer_loops"
              EXPECTED_OUTPUT "outerlooppar_2_2_1_1.stdout"
              COMMAND "run_kernel" "outerlooppar.cl" 2 2 1 1)

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

add_test_pocl(NAME "workgroup/range_md_small_grid"
              EXPECTED_OUTPUT "range_md_small_grid.stdout"
              COMMAND "run_kernel" "range_md.cl" "1" "10" "1" "1")

add_test_pocl(NAME "workgroup/range_md_large_grid"
              EXPECTED_OUTPUT ""
              COMMAND "run_kernel" "range_md.cl" "1000" "128" "1" "1")

# These tests are now always ran with the basic device with a predefined
# work-group execution order. Their printout verification depends
# on it.
set_tests_properties( "workgroup/unconditional_barriers"
  "workgroup/conditional_barrier"
  "workgroup/forcing_horizontal_parallelization_to_some_outer_loops"
  "workgroup/loop_with_two_paths_to_the_latch"
  "workgroup/b_loop_with_two_latches"
  "workgroup/workgroup_sizes_work_items_get_wrong_ids"
  "workgroup/issue_548_convergent_propagation"
  "workgroup/range_md_small_grid"
  "workgroup/range_md_large_grid"
  PROPERTIES
    COST 2.0
    PROCESSORS 1
    LABELS "workgroup"
    ENVIRONMENT "POCL_DEVICES=basic;POCL_WORK_GROUP_METHOD=workitemloops"
    DEPENDS "pocl_version_check"
    LABELS "internal;workgroup")

# Cases which are not dependent on the work-group or work-item
# execution (printout) order or the method (use the default method
# for the device).
set_tests_properties("workgroup/unbarriered_for_loops"
  "workgroup/barriered_for_loops"
  "workgroup/switch_case"
  "workgroup/b_loop_with_none_of_the_WIs_reaching_the_barrier"
  PROPERTIES
    COST 2.0
    PROCESSORS 1
    LABELS "workgroup"
    DEPENDS "pocl_version_check"
    LABELS "internal;workgroup")
