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

set(PROGRAMS_TO_BUILD test_barrier_between_for_loops test_early_return
  test_for_with_var_iteration_count test_id_dependent_computation
  test_locals test_loop_phi_replication test_multi_level_loops_with_barriers
  test_simple_for_with_a_barrier test_structs_as_args test_vectors_as_args
  test_barrier_before_return test_infinite_loop test_constant_array
  test_undominated_variable test_setargs test_null_arg
  test_fors_with_var_iteration_counts)

#AM_LDFLAGS = ../../lib/poclu/libpoclu.la @OPENCL_LIBS@
# POCLU_LINK_OPTIONS

#AM_CXXFLAGS = @OPENCL_CFLAGS@ -Wno-deprecated -Wno-deprecated-declarations
if (MSVC)
  add_compile_options(${OPENCL_CFLAGS})
else ()
  add_compile_options("-std=c++11" "-Wno-deprecated" "-Wno-deprecated-declarations" ${OPENCL_CFLAGS})
endif ()
 

#AM_CPPFLAGS = -I$(top_srcdir)/fix-include -I$(top_builddir)/include -I$(top_srcdir)/include -I$(top_srcdir)/lib/CL -DSRCDIR='"$(abs_srcdir)"'
add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
include_directories("${CMAKE_SOURCE_DIR}/lib/CL")

foreach(PROG ${PROGRAMS_TO_BUILD})
  add_executable("${PROG}" "${PROG}.cpp")
  target_link_libraries("${PROG}" ${POCLU_LINK_OPTIONS})
endforeach()

set(C_PROGRAMS_TO_BUILD test_assign_loop_variable_to_privvar_makes_it_local
     test_assign_loop_variable_to_privvar_makes_it_local_2)
foreach(PROG ${C_PROGRAMS_TO_BUILD})
  if(MSVC)
    set_source_files_properties( "${PROG}.c" PROPERTIES LANGUAGE CXX )
  endif(MSVC)
  add_executable("${PROG}" "${PROG}.c")
  target_link_libraries("${PROG}" ${POCLU_LINK_OPTIONS})
endforeach()


######################################################################

#AT_SETUP([phi nodes not replicated (repl) - lp:927573])

# repl

add_test("\"regression/phi nodes not replicated (repl)\"" "test_loop_phi_replication")

add_test("\"regression/issues with local pointers (repl)\"" "test_locals")

add_test("\"regression/barrier between two for loops (repl)\"" "test_barrier_between_for_loops")

add_test("\"regression/simple for-loop with a barrier inside (repl)\"" "test_simple_for_with_a_barrier")

add_test("\"regression/for-loop with computation after the brexit (repl)\"" "test_multi_level_loops_with_barriers")

add_test("\"regression/for-loop with a variable iteration count (repl)\"" "test_for_with_var_iteration_count")

add_test("\"regression/early return before a barrier region (repl)\"" "test_early_return")

add_test("\"regression/id-dependent computation before kernel exit (repl)\"" "test_id_dependent_computation")

add_test("\"regression/barrier just before return (repl)\"" "test_barrier_before_return")

add_test("\"regression/infinite loop (repl)\"" "test_infinite_loop")

add_test("\"regression/undominated variable from conditional barrier handling (repl)\"" "test_undominated_variable")

add_test("\"regression/assigning a loop iterator variable to a private makes it local (repl)\""
                           "test_assign_loop_variable_to_privvar_makes_it_local")

add_test("\"regression/assigning a loop iterator variable to a private makes it local 2 (repl)\""
                           "test_assign_loop_variable_to_privvar_makes_it_local_2")

set_tests_properties("\"regression/phi nodes not replicated (repl)\""
  "\"regression/issues with local pointers (repl)\""
  "\"regression/barrier between two for loops (repl)\""
  "\"regression/simple for-loop with a barrier inside (repl)\""
  "\"regression/for-loop with computation after the brexit (repl)\""
  "\"regression/for-loop with a variable iteration count (repl)\""
  "\"regression/early return before a barrier region (repl)\""
  "\"regression/id-dependent computation before kernel exit (repl)\""
  "\"regression/barrier just before return (repl)\""
  "\"regression/infinite loop (repl)\""
  "\"regression/undominated variable from conditional barrier handling (repl)\""
  "\"regression/assigning a loop iterator variable to a private makes it local (repl)\""
  "\"regression/assigning a loop iterator variable to a private makes it local 2 (repl)\""
  PROPERTIES
    ENVIRONMENT "POCL_WORK_GROUP_METHOD=workitemrepl"
    COST 1.5
    PROCESSORS 1
    DEPENDS "pocl_version_check")


# loops

add_test("\"regression/phi nodes not replicated (loops)\"" "test_loop_phi_replication")

add_test("\"regression/issues with local pointers (loops)\"" "test_locals")

add_test("\"regression/barrier between two for loops (loops)\"" "test_barrier_between_for_loops")

add_test("\"regression/simple for-loop with a barrier inside (loops)\"" "test_simple_for_with_a_barrier")

add_test("\"regression/for-loop with computation after the brexit (loops)\"" "test_multi_level_loops_with_barriers")

add_test("\"regression/for-loop with a variable iteration count (loops)\"" "test_for_with_var_iteration_count")

add_test("\"regression/early return before a barrier region (loops)\"" "test_early_return")

add_test("\"regression/id-dependent computation before kernel exit (loops)\"" "test_id_dependent_computation")

add_test("\"regression/barrier just before return (loops)\"" "test_barrier_before_return")

add_test("\"regression/infinite loop (loops)\"" "test_infinite_loop")

add_test("\"regression/undominated variable from conditional barrier handling (loops)\"" "test_undominated_variable")

add_test("\"regression/assigning a loop iterator variable to a private makes it local (loops)\""
                           "test_assign_loop_variable_to_privvar_makes_it_local")

add_test("\"regression/assigning a loop iterator variable to a private makes it local 2 (loops)\""
                           "test_assign_loop_variable_to_privvar_makes_it_local_2")

set_tests_properties("\"regression/phi nodes not replicated (loops)\""
  "\"regression/issues with local pointers (loops)\""
  "\"regression/barrier between two for loops (loops)\""
  "\"regression/simple for-loop with a barrier inside (loops)\""
  "\"regression/for-loop with computation after the brexit (loops)\""
  "\"regression/for-loop with a variable iteration count (loops)\""
  "\"regression/early return before a barrier region (loops)\""
  "\"regression/id-dependent computation before kernel exit (loops)\""
  "\"regression/barrier just before return (loops)\""
  "\"regression/infinite loop (loops)\""
  "\"regression/undominated variable from conditional barrier handling (loops)\""
  "\"regression/assigning a loop iterator variable to a private makes it local (loops)\""
  "\"regression/assigning a loop iterator variable to a private makes it local 2 (loops)\""
  PROPERTIES
    ENVIRONMENT "POCL_WORK_GROUP_METHOD=workitemloops"
    COST 1.5
    PROCESSORS 1
    DEPENDS "pocl_version_check")


# other

add_test("\"regression/setting a buffer argument to NULL causes a segfault\"" "test_null_arg")

add_test("\"regression/clSetKernelArg overwriting the previous kernel's args\"" "test_setargs")

add_test("\"regression/passing a constant array as an arg\"" "test_constant_array")

add_test("\"regression/case with multiple variable length loops and a barrier in one\"" "test_fors_with_var_iteration_counts")

#add_test("\"regression/struct kernel arguments\"" "test_structs_as_args")

#add_test("\"regression/vector kernel arguments\"" "test_vectors_as_args")

set_tests_properties("\"regression/setting a buffer argument to NULL causes a segfault\""
  "\"regression/clSetKernelArg overwriting the previous kernel's args\""
  "\"regression/passing a constant array as an arg\""
  "\"regression/case with multiple variable length loops and a barrier in one\""
  PROPERTIES
    COST 1.5
    PROCESSORS 1
    DEPENDS "pocl_version_check")

#  "\"regression/struct kernel arguments\""
#  "\"regression/vector kernel arguments\""

###################################################################

if(POWERPC)
  set_tests_properties("\"regression/for-loop with a variable iteration count (loops)\""
    PROPERTIES  WILL_FAIL 1)
  if(LLVM_3_2)
    set_tests_properties("\"regression/vector kernel arguments\""
      PROPERTIES  WILL_FAIL 1)
  endif()
endif()

# The vector/struct kernel arguments are known to be flaky and
# work by luck sometimes. Disable them for now.
#
#if((LLVM_CXXFLAGS MATCHES "_DEBUG") OR (NOT LLVM_CXXFLAGS MATCHES "DNDEBUG"))
#  set_tests_properties("\"regression/vector kernel arguments\""
#    PROPERTIES  WILL_FAIL 1)
#endif()

# TODO infinite loop test: AT_SKIP_IF([ env | grep -q POCL_IMPLICIT_FINISH])

#set_tests_properties("\"regression/struct kernel arguments\""
#  PROPERTIES  WILL_FAIL 1)

set_tests_properties(
  "\"regression/assigning a loop iterator variable to a private makes it local 2 (repl)\""
  "\"regression/assigning a loop iterator variable to a private makes it local 2 (loops)\""
    PROPERTIES PASS_REGULAR_EXPRESSION
"changing the value at global_id: 6, local_id 2, group_id 1, to: 3
value is changed at global_id: 6, local_id 2, group_id 1, to: 3
")
