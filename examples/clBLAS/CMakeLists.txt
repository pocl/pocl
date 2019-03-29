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

set(TS_NAME "clBLAS")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")

find_package(Boost 1.44 QUIET)

if(EXAMPLES_USE_GIT_MASTER)
  set(FETCH_SOURCE GIT_REPOSITORY "https://github.com/clMathLibraries/clBLAS.git")
else()
  set(FETCH_SOURCE URL "https://github.com/clMathLibraries/clBLAS/archive/v2.12.tar.gz")
endif()

if(NOT Boost_FOUND)
  message(STATUS "Disabling testsuite ${TS_NAME}, required Boost version not found" )
elseif(NOT TESTS_USE_ICD)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires ocl-icd" )

else()
  message(STATUS "Enabling testsuite ${TS_NAME}")
  list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
  set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

  ExternalProject_Add(
    ${TS_NAME}
    PREFIX "${TS_BASEDIR}"
    ${FETCH_SOURCE}
    PATCH_COMMAND cd src && pwd && patch -p1 -i ${CMAKE_SOURCE_DIR}/examples/clBLAS/clBLAS_float_error.patch && patch -p1 -i ${CMAKE_SOURCE_DIR}/examples/clBLAS/clBLAS_link.patch
    UPDATE_COMMAND pwd && find ./src/samples/ -type f -name *.c -o -name *.cpp  | xargs sed -i "s/CL_DEVICE_TYPE_GPU/CL_DEVICE_TYPE_CPU/g"
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
      -DBUILD_RUNTIME=ON
      -DBUILD_TEST=OFF
      -DBUILD_PERFORMANCE=ON
      -DBUILD_SAMPLE=ON
      -DBUILD_CLIENT=ON
      -DBUILD_KTEST=ON
      -DBUILD_SHARED_LIBS=ON
      -DCMAKE_BUILD_TYPE=RelWithDebInfo
      "-DCMAKE_SHARED_LINKER_FLAGS=-pthread"
      "-DCMAKE_C_FLAGS_RELWITHDEBINFO=-O2 -g -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS"
      "${TS_BASEDIR}/src/${TS_NAME}/src"
    INSTALL_COMMAND /bin/true
  )

  set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_dependencies(prepare_examples ${TS_NAME})

  add_test(NAME clBLAS_samples_example_chbmv
           COMMAND "${TS_BUILDDIR}/samples/example_chbmv"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_chemm
           COMMAND "${TS_BUILDDIR}/samples/example_chemm"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_cher
           COMMAND "${TS_BUILDDIR}/samples/example_cher"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_cher2k
           COMMAND "${TS_BUILDDIR}/samples/example_cher2k"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_cherk
           COMMAND "${TS_BUILDDIR}/samples/example_cherk"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_chpmv
           COMMAND "${TS_BUILDDIR}/samples/example_chpmv"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_chpr
           COMMAND "${TS_BUILDDIR}/samples/example_chpr"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_csscal
           COMMAND "${TS_BUILDDIR}/samples/example_csscal"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_dtrmv
           COMMAND "${TS_BUILDDIR}/samples/example_dtrmv"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_isamax
           COMMAND "${TS_BUILDDIR}/samples/example_isamax"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_sasum
           COMMAND "${TS_BUILDDIR}/samples/example_sasum"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_saxpy
           COMMAND "${TS_BUILDDIR}/samples/example_saxpy"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_scopy
           COMMAND "${TS_BUILDDIR}/samples/example_scopy"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_sdot
           COMMAND "${TS_BUILDDIR}/samples/example_sdot"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_sgbmv
           COMMAND "${TS_BUILDDIR}/samples/example_sgbmv"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_sgemm
           COMMAND "${TS_BUILDDIR}/samples/example_sgemm"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_sgemv
           COMMAND "${TS_BUILDDIR}/samples/example_sgemv"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_sger
           COMMAND "${TS_BUILDDIR}/samples/example_sger"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_snrm2
           COMMAND "${TS_BUILDDIR}/samples/example_snrm2"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_srot
           COMMAND "${TS_BUILDDIR}/samples/example_srot"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_srotg
           COMMAND "${TS_BUILDDIR}/samples/example_srotg"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_srotm
           COMMAND "${TS_BUILDDIR}/samples/example_srotm"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_srotmg
           COMMAND "${TS_BUILDDIR}/samples/example_srotmg"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_ssbmv
           COMMAND "${TS_BUILDDIR}/samples/example_ssbmv"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_sscal
           COMMAND "${TS_BUILDDIR}/samples/example_sscal"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_sspmv
           COMMAND "${TS_BUILDDIR}/samples/example_sspmv"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_sspr
           COMMAND "${TS_BUILDDIR}/samples/example_sspr"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_sspr2
           COMMAND "${TS_BUILDDIR}/samples/example_sspr2"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_sswap
           COMMAND "${TS_BUILDDIR}/samples/example_sswap"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_ssymm
           COMMAND "${TS_BUILDDIR}/samples/example_ssymm"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_ssymv
           COMMAND "${TS_BUILDDIR}/samples/example_ssymv"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_ssyr
           COMMAND "${TS_BUILDDIR}/samples/example_ssyr"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_ssyr2
           COMMAND "${TS_BUILDDIR}/samples/example_ssyr2"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_ssyr2k
           COMMAND "${TS_BUILDDIR}/samples/example_ssyr2k"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_ssyrk
           COMMAND "${TS_BUILDDIR}/samples/example_ssyrk"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_stbmv
           COMMAND "${TS_BUILDDIR}/samples/example_stbmv"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_stbsv
           COMMAND "${TS_BUILDDIR}/samples/example_stbsv"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_stpmv
           COMMAND "${TS_BUILDDIR}/samples/example_stpmv"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_stpsv
           COMMAND "${TS_BUILDDIR}/samples/example_stpsv"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_strmm
           COMMAND "${TS_BUILDDIR}/samples/example_strmm"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_strmv
           COMMAND "${TS_BUILDDIR}/samples/example_strmv"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_strsm
           COMMAND "${TS_BUILDDIR}/samples/example_strsm"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_strsv
           COMMAND "${TS_BUILDDIR}/samples/example_strsv"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_zhemv
           COMMAND "${TS_BUILDDIR}/samples/example_zhemv"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_zher2
           COMMAND "${TS_BUILDDIR}/samples/example_zher2"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")
  add_test(NAME clBLAS_samples_example_zhpr2
           COMMAND "${TS_BUILDDIR}/samples/example_zhpr2"
           WORKING_DIRECTORY "${TS_BUILDDIR}/samples")

  # TODO
  #./staging/testPerfWrapper
  #./staging/client


  set_tests_properties(

    clBLAS_samples_example_chbmv
    clBLAS_samples_example_chemm
    clBLAS_samples_example_cher
    clBLAS_samples_example_cher2k
    clBLAS_samples_example_cherk
    clBLAS_samples_example_chpmv
    clBLAS_samples_example_chpr
    clBLAS_samples_example_csscal
    clBLAS_samples_example_dtrmv
    clBLAS_samples_example_isamax
    clBLAS_samples_example_sasum
    clBLAS_samples_example_saxpy
    clBLAS_samples_example_scopy
    clBLAS_samples_example_sdot
    clBLAS_samples_example_sgbmv
    clBLAS_samples_example_sgemm
    clBLAS_samples_example_sgemv
    clBLAS_samples_example_sger
    clBLAS_samples_example_snrm2
    clBLAS_samples_example_srot
    clBLAS_samples_example_srotg
    clBLAS_samples_example_srotm
    clBLAS_samples_example_srotmg
    clBLAS_samples_example_ssbmv
    clBLAS_samples_example_sscal
    clBLAS_samples_example_sspmv
    clBLAS_samples_example_sspr
    clBLAS_samples_example_sspr2
    clBLAS_samples_example_sswap
    clBLAS_samples_example_ssymm
    clBLAS_samples_example_ssymv
    clBLAS_samples_example_ssyr
    clBLAS_samples_example_ssyr2
    clBLAS_samples_example_ssyr2k
    clBLAS_samples_example_ssyrk
    clBLAS_samples_example_stbmv
    clBLAS_samples_example_stbsv
    clBLAS_samples_example_stpmv
    clBLAS_samples_example_stpsv
    clBLAS_samples_example_strmm
    clBLAS_samples_example_strmv
    clBLAS_samples_example_strsm
    clBLAS_samples_example_strsv
    clBLAS_samples_example_zhemv
    clBLAS_samples_example_zher2
    clBLAS_samples_example_zhpr2

    PROPERTIES
      LABELS "${TS_NAME}"
      ENVIRONMENT "LD_LIBRARY_PATH=${TS_BUILDDIR}/library")

  # with LLVM: 6.0+
  # error: variables in the local address space can only be declared in the outermost scope of a kernel function
  set_tests_properties(
    clBLAS_samples_example_snrm2
    clBLAS_samples_example_chpmv
    clBLAS_samples_example_dtrmv
    clBLAS_samples_example_sspmv
    clBLAS_samples_example_stpmv
    clBLAS_samples_example_stpsv
    clBLAS_samples_example_strmv
    clBLAS_samples_example_strsv
    clBLAS_samples_example_zhemv
    PROPERTIES
    WILL_FAIL 1)

endif()
