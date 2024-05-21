#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2023 Michal Babej / Intel Finland Oy
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

set(TS_NAME "intel-compute-samples")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TS_BASEDIR}/src/${TS_NAME}")

if(NOT HAVE_GIT)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires git to checkout sources")
  return()
endif()

find_package(Boost 1.71)
if(NOT Boost_FOUND)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires Boost 1.71")
  return()
endif()

find_package(PNG 1.6.37)
if(NOT PNG_FOUND)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires libpng 1.6")
  return()
endif()

find_program(OCLOC ocloc)
if(NOT OCLOC)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires ocloc")
  return()
endif()


#find_package(GTest 1.10)
#if(NOT GTest_FOUND)
#  message(STATUS "Disabling testsuite ${TS_NAME}, requires GoogleTest 1.10 (libgtest-dev)")
#  return()
#endif()

if(PKG_CONFIG_EXECUTABLE)
  pkg_check_modules(LIBVA libva>=1.10)
else()
  find_path(LIBVA_FOUND NAMES va_x11.h PATH_SUFFIXES va)
endif()
if(NOT LIBVA_FOUND)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires libva-dev >= 1.10")
  return()
endif()

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  GIT_REPOSITORY "https://github.com/intel/compute-samples.git"
  PATCH_COMMAND ${BASH} -c "if [ ! -e ${TS_SRCDIR}/mediadata/external/yuv/goal_1280x720.yuv ] $<SEMICOLON> then ${TS_SRCDIR}/scripts/install/install_googletest.sh && ${TS_SRCDIR}/scripts/install/download_mediadata.sh $<SEMICOLON> fi"
  CONFIGURE_COMMAND ${CMAKE_COMMAND}
    -DWITH_L0=OFF
    -DOpenCL_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/include
    -DOPENCL_ROOT=${OPENCL_LIBDIR}
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    "${TS_BASEDIR}/src/${TS_NAME}"

  INSTALL_COMMAND /bin/true
)


set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

set(SKIP_TEST_EXPR "subgroups_imagecopy_tests_hw|subgroups_visualization_tests_hw|vme_search_tests_hw|vme_hme_tests_hw|vme_intra_tests_hw|vme_wpp_tests_hw|vme_interlaced_tests_hw|vme_interop_tests_hw|ocl_utils_tests_hw|test_cl_unified_shared_memory_hw|test_cl_visa_injection_rt_basic_hw|test_cl_command_queue_families_hw")

add_test(NAME all_intel_compute_tests
         COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
         -E "${SKIP_TEST_EXPR}"
         WORKING_DIRECTORY "${TS_BUILDDIR}")

set_tests_properties(all_intel_compute_tests PROPERTIES LABELS "${TS_NAME}")
