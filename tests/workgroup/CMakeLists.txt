#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014 pocl developers
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

function(add_test_workgroup TEST_NAME RESULT_FILE CL_FILE)
  add_test_custom("${CMAKE_CURRENT_BINARY_DIR}/run_kernel" "${TEST_NAME}"
                  "${RESULT_FILE}" "${CL_FILE}" ${ARGN})
endfunction()

function(add_test_workgroup_sorted TEST_NAME RESULT_FILE CL_FILE)
  set(RUN_CMD "${CMAKE_CURRENT_BINARY_DIR}/run_kernel")
  foreach(LOOPVAR  "${CL_FILE}" ${ARGN})
    set(RUN_CMD "${RUN_CMD}####${LOOPVAR}")
  endforeach()

  add_test("${TEST_NAME}"
    "${CMAKE_COMMAND}"
    -Dtest_cmd=${RUN_CMD}
    -Dsort_output=1
    -Doutput_blessed=${CMAKE_CURRENT_SOURCE_DIR}/${RESULT_FILE}
    -P "${CMAKE_SOURCE_DIR}/cmake/run_test.cmake"
  )
endfunction()


#AM_LDFLAGS = ../../lib/poclu/libpoclu.la @OPENCL_LIBS@
#POCLU_LINK_OPTIONS

#AM_CPPFLAGS = -I$(top_srcdir)/fix-include -I$(top_srcdir)/include -I$(top_srcdir)/lib/CL -DSRCDIR='"$(abs_srcdir)"' @OPENCL_CFLAGS@
add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
include_directories("${CMAKE_SOURCE_DIR}/lib/CL")
add_compile_options(${OPENCL_CFLAGS})

#EXTRA_DIST = basic_barriers.cl conditional_barriers.cl forloops.cl forloops_2_2_1_1.stdout loopbarriers.cl basic_barriers_2_2_2_2.stdout tricky_for.cl outerlooppar.cl outerlooppar_2_2_1_1.stdout for_bug.cl for_bug_1_2_1_1.stdout multilatch_bloop.cl multilatch_bloop_1_3_1_1.stdout print_all_ids.cl print_all_ids_114114.txt implicit_barriers.cl implicit_barriers_1_2_1_1.stdout

#noinst_PROGRAMS = run_kernel
if(MSVC)
  set_source_files_properties( run_kernel.c PROPERTIES LANGUAGE CXX )
endif(MSVC)
add_executable("run_kernel" "run_kernel.c")
target_link_libraries("run_kernel" ${POCLU_LINK_OPTIONS})



# repl
add_test_workgroup("\"workgroup/unconditional barriers (full replication)\"" "basic_barriers_2_2_2_2.stdout" "basic_barriers.cl" 2 2 2 2 )

add_test_workgroup("\"workgroup/unbarriered for loops (full replication)\"" "forloops_2_2_1_1.stdout" "forloops.cl" 2 2 1 1)

add_test_workgroup("\"workgroup/barriered for loops (full replication)\"" "loopbarriers_2_2_1_1.stdout" "loopbarriers.cl" 2 2 1 1 )

add_test_workgroup("\"workgroup/conditional barrier (full replication)\"" "cond_barriers_1_2_1_1.stdout" "conditional_barriers.cl" 1 2 1 1)

add_test_workgroup("\"workgroup/b-loop with none of the WIs reaching the barrier (full replication)\"" "tricky_for_1_2_1_1.stdout" "tricky_for.cl" 1 2 1 1)

add_test_workgroup("\"workgroup/forcing horizontal parallelization to some outer loops (full replication)\"" "outerlooppar_2_2_1_1.stdout" "outerlooppar.cl" 2 2 1 1)

add_test_workgroup("\"workgroup/loop with two paths to the latch (full replication)\"" "for_bug_1_2_1_1.stdout" "for_bug.cl" 1 2 1 1)

add_test_workgroup("\"workgroup/b-loop with two latches (full replication)\"" "multilatch_bloop_1_3_1_1.stdout" "multilatch_bloop.cl" 1 3 1 1)

add_test_workgroup_sorted("\"workgroup/workgroup_sizes: work-items get wrong ids (full replication)\"" "print_all_ids_114114.txt" "print_all_ids.cl" 1 1 1 4)

set_tests_properties( "\"workgroup/unconditional barriers (full replication)\""
  "\"workgroup/unbarriered for loops (full replication)\""
  "\"workgroup/barriered for loops (full replication)\""
  "\"workgroup/conditional barrier (full replication)\""
  "\"workgroup/b-loop with none of the WIs reaching the barrier (full replication)\""
  "\"workgroup/forcing horizontal parallelization to some outer loops (full replication)\""
  "\"workgroup/loop with two paths to the latch (full replication)\""
  "\"workgroup/b-loop with two latches (full replication)\""
  "\"workgroup/workgroup_sizes: work-items get wrong ids (full replication)\""
  PROPERTIES
    COST 2.0
    PROCESSORS 1
    LABELS "workgroup"
    ENVIRONMENT "POCL_DEVICES=basic;POCL_WORK_GROUP_METHOD=workitemrepl"
    DEPENDS "pocl_version_check")

# loops
add_test_workgroup("\"workgroup/different implicit barrier injection scenarios (loops)\"" "implicit_barriers_1_2_1_1.stdout" "implicit_barriers.cl" 1 2 1 1)

add_test_workgroup("\"workgroup/unconditional barriers (loops)\"" "basic_barriers_2_2_2_2.stdout" "basic_barriers.cl" 2 2 2 2 )

add_test_workgroup("\"workgroup/unbarriered for loops (loops)\"" "forloops_2_2_1_1.stdout" "forloops.cl" 2 2 1 1)

add_test_workgroup("\"workgroup/barriered for loops (loops)\"" "loopbarriers_2_2_1_1.stdout" "loopbarriers.cl" 2 2 1 1 )

add_test_workgroup("\"workgroup/conditional barrier (loops)\"" "cond_barriers_1_2_1_1.stdout" "conditional_barriers.cl" 1 2 1 1)

add_test_workgroup("\"workgroup/b-loop with none of the WIs reaching the barrier (loops)\"" "tricky_for_1_2_1_1.stdout" "tricky_for.cl" 1 2 1 1)

add_test_workgroup("\"workgroup/forcing horizontal parallelization to some outer loops (loops)\"" "outerlooppar_2_2_1_1.stdout" "outerlooppar.cl" 2 2 1 1)

add_test_workgroup("\"workgroup/loop with two paths to the latch (loops)\"" "for_bug_1_2_1_1.stdout" "for_bug.cl" 1 2 1 1)

add_test_workgroup("\"workgroup/b-loop with two latches (loops)\"" "multilatch_bloop_1_3_1_1.stdout" "multilatch_bloop.cl" 1 3 1 1)

add_test_workgroup_sorted("\"workgroup/workgroup_sizes: work-items get wrong ids (loops)\"" "print_all_ids_114114.txt" "print_all_ids.cl" 1 1 1 4)


set_tests_properties( "\"workgroup/unconditional barriers (loops)\""
  "\"workgroup/unbarriered for loops (loops)\""
  "\"workgroup/barriered for loops (loops)\""
  "\"workgroup/conditional barrier (loops)\""
  "\"workgroup/b-loop with none of the WIs reaching the barrier (loops)\""
  "\"workgroup/forcing horizontal parallelization to some outer loops (loops)\""
  "\"workgroup/loop with two paths to the latch (loops)\""
  "\"workgroup/b-loop with two latches (loops)\""
  "\"workgroup/workgroup_sizes: work-items get wrong ids (loops)\""
  PROPERTIES
    COST 2.0
    PROCESSORS 1
    LABELS "workgroup"
    ENVIRONMENT "POCL_DEVICES=basic;POCL_WORK_GROUP_METHOD=workitemloops"
    DEPENDS "pocl_version_check")
