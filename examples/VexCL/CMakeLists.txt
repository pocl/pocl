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

set(TS_NAME "VexCL")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")

if(NOT HAVE_GIT)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires git to checkout sources")
elseif(NOT TESTS_USE_ICD)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires ocl-icd")

else()
message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  GIT_REPOSITORY "https://github.com/ddemidov/vexcl.git"
  CMAKE_ARGS
    -DVEXCL_CACHE_KERNELS=OFF
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DVEXCL_BACKEND=OpenCL
    -DVEXCL_BUILD_EXAMPLES=ON
    -DVEXCL_BUILD_TESTS=ON
    -DVEXCL_CACHE_KERNELS=OFF
    -DVEXCL_HAVE_BOOST_COMPUTE=ON
  INSTALL_COMMAND /bin/true
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

add_test(NAME vexcl_boost_version
         COMMAND "${TS_BUILDDIR}/tests/boost_version")
add_test(NAME vexcl_types
         COMMAND "${TS_BUILDDIR}/tests/types")
add_test(NAME vexcl_deduce
         COMMAND "${TS_BUILDDIR}/tests/deduce")
add_test(NAME vexcl_context
         COMMAND "${TS_BUILDDIR}/tests/context")
add_test(NAME vexcl_vector_create
         COMMAND "${TS_BUILDDIR}/tests/vector_create")
add_test(NAME vexcl_vector_copy
         COMMAND "${TS_BUILDDIR}/tests/vector_copy")
add_test(NAME vexcl_vector_arithmetics
         COMMAND "${TS_BUILDDIR}/tests/vector_arithmetics")
add_test(NAME vexcl_vector_view
         COMMAND "${TS_BUILDDIR}/tests/vector_view")
add_test(NAME vexcl_tensordot
         COMMAND "${TS_BUILDDIR}/tests/tensordot")
add_test(NAME vexcl_vector_pointer
         COMMAND "${TS_BUILDDIR}/tests/vector_pointer")
add_test(NAME vexcl_tagged_terminal
         COMMAND "${TS_BUILDDIR}/tests/tagged_terminal")
add_test(NAME vexcl_temporary
         COMMAND "${TS_BUILDDIR}/tests/temporary")
add_test(NAME vexcl_cast
         COMMAND "${TS_BUILDDIR}/tests/cast")
add_test(NAME vexcl_multivector_create
         COMMAND "${TS_BUILDDIR}/tests/multivector_create")
add_test(NAME vexcl_multivector_arithmetics
         COMMAND "${TS_BUILDDIR}/tests/multivector_arithmetics")
add_test(NAME vexcl_multi_array
         COMMAND "${TS_BUILDDIR}/tests/multi_array")
add_test(NAME vexcl_spmv
         COMMAND "${TS_BUILDDIR}/tests/spmv")
add_test(NAME vexcl_stencil
         COMMAND "${TS_BUILDDIR}/tests/stencil")
add_test(NAME vexcl_generator
         COMMAND "${TS_BUILDDIR}/tests/generator")
add_test(NAME vexcl_mba
         COMMAND "${TS_BUILDDIR}/tests/mba")
add_test(NAME vexcl_random
         COMMAND "${TS_BUILDDIR}/tests/random")
add_test(NAME vexcl_sort
         COMMAND "${TS_BUILDDIR}/tests/sort")
add_test(NAME vexcl_scan
         COMMAND "${TS_BUILDDIR}/tests/scan")
add_test(NAME vexcl_scan_by_key
         COMMAND "${TS_BUILDDIR}/tests/scan_by_key")
add_test(NAME vexcl_reduce_by_key
         COMMAND "${TS_BUILDDIR}/tests/reduce_by_key")
add_test(NAME vexcl_logical
         COMMAND "${TS_BUILDDIR}/tests/logical")
add_test(NAME vexcl_threads
         COMMAND "${TS_BUILDDIR}/tests/threads")

set_tests_properties(vexcl_boost_version vexcl_types vexcl_deduce
  vexcl_context vexcl_vector_create vexcl_vector_copy
  vexcl_vector_arithmetics vexcl_vector_view vexcl_tensordot
  vexcl_vector_pointer vexcl_tagged_terminal vexcl_temporary
  vexcl_cast vexcl_multivector_create vexcl_multivector_arithmetics
  vexcl_multi_array vexcl_spmv vexcl_stencil vexcl_generator
  vexcl_mba vexcl_random vexcl_sort vexcl_scan vexcl_scan_by_key
  vexcl_reduce_by_key vexcl_logical vexcl_threads
  PROPERTIES
    LABELS "VexCL")

endif()
