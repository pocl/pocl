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

set(TS_NAME IntelSVM)
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")
set(INTEL_ZIP "${TS_SRCDIR}/intel_ocl_svm_basic_win.zip")

if (EXISTS "${INTEL_ZIP}")
  message(STATUS "Enabling testsuite ${TS_NAME}")
  list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
  set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

  ExternalProject_Add(
    ${TS_NAME}
    PREFIX "${TS_BASEDIR}"
    DOWNLOAD_COMMAND /bin/true
    PATCH_COMMAND pwd && unzip "${INTEL_ZIP}" &&
      cp "${CMAKE_CURRENT_SOURCE_DIR}/intelsvm_CMakeLists.txt" ./CMakeLists.txt &&
      patch -p0 -d SVMBasicCoarseGrained -i ${CMAKE_CURRENT_SOURCE_DIR}/intelsvm.patch &&
      patch -p0 -d SVMBasicFineGrained -i ${CMAKE_CURRENT_SOURCE_DIR}/intelsvm.patch
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=RelWithDebInfo
      "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -g -DCL_USE_DEPRECATED_OPENCL_1_2_APIS"
      -DOPENCL_LIBRARIES:STRING=OpenCL
      "-DPOCL_OPENCL_HEADERS=${CMAKE_SOURCE_DIR}/include"
    INSTALL_COMMAND "/bin/true"
  )

  set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_dependencies(prepare_examples ${TS_NAME})

  # LLVM 3.8 fails with
  # error: Explicit load/store type does not match pointee type of pointer operand
  # This seems to occur only with some LLVM built setups, so let's just skip
  # the tests completely.

  if(NOT LLVM_3_8)
    add_test(NAME intel_svm_coarse
      COMMAND "${TS_BUILDDIR}/coarse" -p 0 -t default
      WORKING_DIRECTORY "${TS_BASEDIR}/src/${TS_NAME}/SVMBasicCoarseGrained")
    add_test(NAME intel_svm_fine
      COMMAND "${TS_BUILDDIR}/fine" -p 0 -t default
      WORKING_DIRECTORY "${TS_BASEDIR}/src/${TS_NAME}/SVMBasicFineGrained")

    set_tests_properties(intel_svm_coarse intel_svm_fine
      PROPERTIES  LABELS "IntelSVM;hsa")
    set_tests_properties(intel_svm_coarse intel_svm_fine
      PROPERTIES  PASS_REGULAR_EXPRESSION "PASSED")

  endif()

else()

  message(STATUS "Disabling testsuite ${TS_NAME}, required files not found" )

endif()
