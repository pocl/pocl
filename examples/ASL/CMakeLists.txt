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

set(TS_NAME "ASL")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")
set(ASL_TGZ "${TS_SRCDIR}/v0.1.7.tar.gz")

# find required libs - boost & vtk
find_package(Boost 1.55 QUIET)
find_package(VTK QUIET)
set(VTK_VER "${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}")

if(EXAMPLES_USE_GIT_MASTER)
  set(FETCH_SOURCE GIT_REPOSITORY "https://github.com/AvtechScientific/ASL")
else()
  set(FETCH_SOURCE URL "https://github.com/AvtechScientific/ASL/archive/v0.1.7.tar.gz")
endif()

if ((NOT "${CMAKE_VERSION}" VERSION_LESS "3.0.2")
    AND VTK_FOUND AND (NOT VTK_VER VERSION_LESS "6.1")
    AND Boost_FOUND AND (NOT Boost_VERSION VERSION_LESS "1.55"))

  message(STATUS "Enabling testsuite ${TS_NAME}")
  list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
  set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

  ExternalProject_Add(
    ${TS_NAME}
    PREFIX "${TS_BASEDIR}"
    ${FETCH_SOURCE}
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=RelWithDebInfo
      -DWITH_API_DOC:BOOL=OFF
      -DWITH_EXAMPLES:BOOL=ON
      -DWITH_MATIO:BOOL=OFF
      -DWITH_TESTS:BOOL=ON
      "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -g -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS"
      "-DCMAKE_C_FLAGS_RELWITHDEBINFO=-O2 -g -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS"
    INSTALL_COMMAND /bin/true
  )

  set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_dependencies(prepare_examples ${TS_NAME})

  add_test(NAME ASL_testABDFormat
           COMMAND "${TS_BUILDDIR}/test/testABD/testABDFormat")
  add_test(NAME ASL_testVectorOfElements
           COMMAND "${TS_BUILDDIR}/test/testACL/testVectorOfElements")
  add_test(NAME ASL_testMatrixOfElements
           COMMAND "${TS_BUILDDIR}/test/testACL/testMatrixOfElements")
  add_test(NAME ASL_testKernel
           COMMAND "${TS_BUILDDIR}/test/testACL/testKernel")
  add_test(NAME ASL_testOperators
           COMMAND "${TS_BUILDDIR}/test/testACL/testOperators")
  add_test(NAME ASL_testKernelMerger
           COMMAND "${TS_BUILDDIR}/test/testACL/testKernelMerger")
# takes too long
#  add_test(NAME ASL_testPrivateVar
#           COMMAND "${TS_BUILDDIR}/test/testACL/testPrivateVar")
  add_test(NAME ASL_testASLData
           COMMAND "${TS_BUILDDIR}/test/testMath/testASLData")
  add_test(NAME ASL_testDistanceFunction
           COMMAND "${TS_BUILDDIR}/test/testMath/testDistanceFunction")
  add_test(NAME ASL_testReductionFunction
           COMMAND "${TS_BUILDDIR}/test/testMath/testReductionFunction")

# Tests are disabled until ASL issue #31 is resolved.
# https://github.com/AvtechScientific/ASL/issues/31
#
#
#  add_test(NAME ASL_example_bus_wind
#           COMMAND "${TS_BUILDDIR}/examples/flow/bus_wind/asl-bus_wind")
#  add_test(NAME ASL_example_compressor
#           COMMAND "${TS_BUILDDIR}/examples/flow/compressor/asl-compressor")
#  add_test(NAME ASL_example_flow
#           COMMAND "${TS_BUILDDIR}/examples/flow/flow/asl-flow")
#  add_test(NAME ASL_example_flow2
#           COMMAND "${TS_BUILDDIR}/examples/flow/flow2/asl-flow2")
#  add_test(NAME ASL_example_flow3
#           COMMAND "${TS_BUILDDIR}/examples/flow/flow3/asl-flow3")
#  add_test(NAME ASL_example_flowKDPGrowth
#           COMMAND "${TS_BUILDDIR}/examples/flow/flowKDPGrowth/asl-flowKDPGrowth")
#  add_test(NAME ASL_example_flowRotatingCylinders
#           COMMAND "${TS_BUILDDIR}/examples/flow/flowRotatingCylinders/asl-flowRotatingCylinders")
#  add_test(NAME ASL_example_locomotive
#           COMMAND "${TS_BUILDDIR}/examples/flow/locomotive/asl-locomotive")
#  add_test(NAME ASL_example_locomotive_laminar
#           COMMAND "${TS_BUILDDIR}/examples/flow/locomotive_laminar/asl-locomotive_laminar")
#  add_test(NAME ASL_example_locomotive_stability
#           COMMAND "${TS_BUILDDIR}/examples/flow/locomotive_stability/asl-locomotive_stability")
#  add_test(NAME ASL_example_multicomponent_flow
#           COMMAND "${TS_BUILDDIR}/examples/flow/multicomponent_flow/asl-multicomponent_flow")
#  add_test(NAME ASL_example_multiphase_flow
#           COMMAND "${TS_BUILDDIR}/examples/flow/multiphase_flow/asl-multiphase_flow")
#  add_test(NAME ASL_example_pitot_tube_ice
#           COMMAND "${TS_BUILDDIR}/examples/flow/pitot_tube_ice/asl-pitot_tube_ice")
#  add_test(NAME ASL_example_acousticWaves
#           COMMAND "${TS_BUILDDIR}/examples/elastic/acousticWaves/asl-acousticWaves")
#  add_test(NAME ASL_example_cubeGravity
#           COMMAND "${TS_BUILDDIR}/examples/elastic/cubeGravity/asl-cubeGravity")
#  add_test(NAME ASL_example_cubeIncompressibleGravity
#           COMMAND "${TS_BUILDDIR}/examples/elastic/cubeIncompressibleGravity/asl-cubeIncompressibleGravity")
#  add_test(NAME ASL_example_cubePoroelasticGravity
#           COMMAND "${TS_BUILDDIR}/examples/elastic/cubePoroelasticGravity/asl-cubePoroelasticGravity")
#  add_test(NAME ASL_example_poroelastic
#           COMMAND "${TS_BUILDDIR}/examples/elastic/poroelastic/asl-poroelastic")
#  add_test(NAME ASL_example_levelSetBasic
#           COMMAND "${TS_BUILDDIR}/examples/levelSet/levelSetBasic/asl-levelSetBasic")
#  add_test(NAME ASL_example_levelSetFacetedGrowth
#           COMMAND "${TS_BUILDDIR}/examples/levelSet/levelSetFacetedGrowth/asl-levelSetFacetedGrowth")
#  add_test(NAME ASL_example_levelSetNormalGrowth
#           COMMAND "${TS_BUILDDIR}/examples/levelSet/levelSetNormalGrowth/asl-levelSetNormalGrowth")
#  add_test(NAME ASL_example_jumpingBox
#           COMMAND "${TS_BUILDDIR}/examples/jumpingObjects/jumpingBox/asl-jumpingBox")
#  add_test(NAME ASL_example_surfaceFlux
#           COMMAND "${TS_BUILDDIR}/examples/heatTransfer/surfaceFlux/asl-surfaceFlux")
#  add_test(NAME ASL_example_testSMDiff
#           COMMAND "${TS_BUILDDIR}/examples/massTransferSM/testSMDiff/asl-testSMDiff")
#  add_test(NAME ASL_example_testSMDiff3C
#           COMMAND "${TS_BUILDDIR}/examples/massTransferSM/testSMDiff3C/asl-testSMDiff3C")
#  add_test(NAME ASL_example_testSMPhi
#           COMMAND "${TS_BUILDDIR}/examples/massTransferSM/testSMPhi/asl-testSMPhi")
#  add_test(NAME ASL_example_testSMPhiBV
#           COMMAND "${TS_BUILDDIR}/examples/massTransferSM/testSMPhiBV/asl-testSMPhiBV")


  set_tests_properties(
    ASL_testABDFormat
    ASL_testVectorOfElements
    ASL_testMatrixOfElements
    ASL_testKernel
    ASL_testOperators
    ASL_testKernelMerger
#    ASL_testPrivateVar
    ASL_testASLData
    ASL_testDistanceFunction
    ASL_testReductionFunction
#    ASL_example_bus_wind
#    ASL_example_compressor
#    ASL_example_flow
#    ASL_example_flow2
#    ASL_example_flow3
#    ASL_example_flowKDPGrowth
#    ASL_example_flowRotatingCylinders
#    ASL_example_locomotive
#    ASL_example_locomotive_laminar
#    ASL_example_locomotive_stability
#    ASL_example_multicomponent_flow
#    ASL_example_multiphase_flow
#    ASL_example_pitot_tube_ice
#    ASL_example_acousticWaves
#    ASL_example_cubeGravity
#    ASL_example_cubeIncompressibleGravity
#    ASL_example_cubePoroelasticGravity
#    ASL_example_poroelastic
#    ASL_example_levelSetBasic
#    ASL_example_levelSetFacetedGrowth
#    ASL_example_levelSetNormalGrowth
#    ASL_example_jumpingBox
#    ASL_example_surfaceFlux
#    ASL_example_testSMDiff
#    ASL_example_testSMDiff3C
#    ASL_example_testSMPhi
#    ASL_example_testSMPhiBV

    PROPERTIES
      LABELS "ASL")

else()
  message(STATUS "Disabling testsuite ${TS_NAME}, required files (Boost or VTK) not found" )
endif()
