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

set(TS_NAME "clFFT")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")

find_package(Boost 1.44 QUIET)

if(EXAMPLES_USE_GIT_MASTER)
  set(FETCH_SOURCE GIT_REPOSITORY "https://github.com/clMathLibraries/clFFT.git")
else()
  set(FETCH_SOURCE URL "https://github.com/clMathLibraries/clFFT/archive/v2.12.2.tar.gz")
endif()

if(NOT TESTS_USE_ICD)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires ocl-icd")
elseif(NOT Boost_FOUND)
  message(STATUS "Disabling testsuite ${TS_NAME}, required Boost version not found" )

else()
  message(STATUS "Enabling testsuite ${TS_NAME}")
  list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
  set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

  ExternalProject_Add(
    ${TS_NAME}
    PREFIX "${TS_BASEDIR}"
    ${FETCH_SOURCE}
    #PATCH_COMMAND  pwd && sed -i "s/CL_DEVICE_TYPE_GPU/CL_DEVICE_TYPE_CPU/g" *.cpp

    # -fpermissive required due to https://github.com/clMathLibraries/clFFT/issues/237
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
      -DBUILD_RUNTIME=ON
      -DBUILD_CLIENT=ON
      -DBUILD_TEST=OFF
      -DBUILD_SHARED_LIBS=ON
      -DBUILD_EXAMPLES=ON
      -DCMAKE_BUILD_TYPE=RelWithDebInfo
      "-DCMAKE_C_FLAGS_RELWITHDEBINFO=-O2 -g -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS"
      "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -g -fpermissive -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS"
      "${TS_BASEDIR}/src/${TS_NAME}/src"
    INSTALL_COMMAND /bin/true
  )

  set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_dependencies(prepare_examples ${TS_NAME})


  add_test(NAME clFFT_samples_fft1d
           COMMAND "${TS_BUILDDIR}/examples/examples/fft1d"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/examples")
  add_test(NAME clFFT_samples_fft3d
           COMMAND "${TS_BUILDDIR}/examples/examples/fft3d"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/examples")
  add_test(NAME clFFT_samples_fft2d
           COMMAND "${TS_BUILDDIR}/examples/examples/fft2d"
           WORKING_DIRECTORY "${TS_BUILDDIR}/examples/examples")

  # TODO
  #./staging/clFFT-client-2.5.0


  set_tests_properties(
    clFFT_samples_fft1d
    clFFT_samples_fft2d
    clFFT_samples_fft3d
    PROPERTIES
      LABELS "${TS_NAME}")

endif()
