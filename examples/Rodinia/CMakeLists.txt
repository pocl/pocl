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

set(TS_NAME "Rodinia")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")
set(RODINIA "rodinia_3.1")
set(RODINIA_TGZ "${TS_SRCDIR}/${RODINIA}.tar.bz2")

if(NOT MAKE_COMMAND)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires make to build")
elseif (EXISTS "${RODINIA_TGZ}")

  message(STATUS "Enabling testsuite ${TS_NAME}")
  list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
  set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

  ExternalProject_Add(
    ${TS_NAME}
    PREFIX "${TS_BASEDIR}"
    DOWNLOAD_COMMAND test -d ${RODINIA} || /bin/bash -c "pwd && echo 'extracting rodinia tgz' && tar xjf '${RODINIA_TGZ}' && patch -p1 -r- -N -i '${CMAKE_CURRENT_SOURCE_DIR}/Rodinia.patch' && rmdir Rodinia && mv ${RODINIA} Rodinia"
    CONFIGURE_COMMAND /bin/true
    BUILD_IN_SOURCE 1
    BUILD_COMMAND pwd && ${MAKE_PROGRAM} OPENCL
      "CFLAGS=-Wno-unused-result -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS"
      "OPENCL_INC=${CMAKE_SOURCE_DIR}/include"
    INSTALL_COMMAND /bin/true
  )

  set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_dependencies(prepare_examples ${TS_NAME})

  add_test(NAME rodinia_backprop                          COMMAND ./run
           WORKING_DIRECTORY "${TS_BUILDDIR}/opencl/backprop")
  add_test(NAME rodinia_bfs                               COMMAND ./run
           WORKING_DIRECTORY "${TS_BUILDDIR}/opencl/bfs")
  add_test(NAME rodinia_euler3d                           COMMAND ./run
           WORKING_DIRECTORY "${TS_BUILDDIR}/opencl/cfd")
  add_test(NAME rodinia_gaussian                          COMMAND ./run
           WORKING_DIRECTORY "${TS_BUILDDIR}/opencl/gaussian")
  add_test(NAME rodinia_heartwall                         COMMAND ./run
           WORKING_DIRECTORY "${TS_BUILDDIR}/opencl/heartwall")
  add_test(NAME rodinia_hotspot                           COMMAND ./run
           WORKING_DIRECTORY "${TS_BUILDDIR}/opencl/hotspot")
  add_test(NAME rodinia_kmeans                            COMMAND ./run
           WORKING_DIRECTORY "${TS_BUILDDIR}/opencl/kmeans")
  add_test(NAME rodinia_lavaMD                            COMMAND ./run
           WORKING_DIRECTORY "${TS_BUILDDIR}/opencl/lavaMD")
  add_test(NAME rodinia_leukocyte                         COMMAND ./run
           WORKING_DIRECTORY "${TS_BUILDDIR}/opencl/leukocyte/OpenCL")
  add_test(NAME rodinia_lud                               COMMAND ./run
           WORKING_DIRECTORY "${TS_BUILDDIR}/opencl/lud/ocl")
  add_test(NAME rodinia_nn                                COMMAND ./run
           WORKING_DIRECTORY "${TS_BUILDDIR}/opencl/nn")
  add_test(NAME rodinia_nw                                COMMAND ./run
           WORKING_DIRECTORY "${TS_BUILDDIR}/opencl/nw")
  add_test(NAME rodinia_particlefilter                    COMMAND ./run
           WORKING_DIRECTORY "${TS_BUILDDIR}/opencl/particlefilter")
  add_test(NAME rodinia_pathfinder                        COMMAND ./run
           WORKING_DIRECTORY "${TS_BUILDDIR}/opencl/pathfinder")
  add_test(NAME rodinia_srad                              COMMAND ./run
           WORKING_DIRECTORY "${TS_BUILDDIR}/opencl/srad")
  add_test(NAME rodinia_streamcluster                     COMMAND ./run
           WORKING_DIRECTORY "${TS_BUILDDIR}/opencl/streamcluster")

  set_tests_properties(
    rodinia_backprop
    rodinia_bfs
    rodinia_euler3d
    rodinia_gaussian
    rodinia_heartwall
    rodinia_hotspot
    rodinia_kmeans
    rodinia_lavaMD
    rodinia_leukocyte
    rodinia_lud
    rodinia_nn
    rodinia_nw
    rodinia_particlefilter
    rodinia_pathfinder
    rodinia_srad
    rodinia_streamcluster

    PROPERTIES
      LABELS "Rodinia")

else()

  message(STATUS "Disabling testsuite ${TS_NAME}, required files not found" )

endif()
