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


if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" AND X86_64)

  set(TS_NAME AMDSDK2.9)
  set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
  set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")
  set(AMD_APP_SDK_TGZ "${TS_SRCDIR}/AMD-APP-SDK-v2.9-RC-lnx64.tgz")

  if(EXISTS "${AMD_APP_SDK_TGZ}")

    message(STATUS "Enabling testsuite ${TS_NAME}")
    list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
    set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

    ExternalProject_Add(
      ${TS_NAME}
      PREFIX "${TS_BASEDIR}"
      DOWNLOAD_COMMAND "/bin/true"
      PATCH_COMMAND tar -xzf "${AMD_APP_SDK_TGZ}" && pwd && patch -p1 -i ${CMAKE_CURRENT_SOURCE_DIR}/AMDSDK2_9.patch
      CONFIGURE_COMMAND "/bin/true"
      BUILD_IN_SOURCE 1
      BUILD_COMMAND "/bin/true"
      INSTALL_COMMAND "/bin/true"
    )

    set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    add_dependencies(prepare_examples ${TS_NAME})

    set(AMD_SAMPLES
    BinarySearch
    BinomialOption
    BinomialOptionMultiGPU
    BitonicSort
    BlackScholes
    BlackScholesDP
    DCT
    DeviceFission
    DwtHaar1D
    DynamicOpenCLDetection
    FastWalshTransform
    FloydWarshall
    GlobalMemoryBandwidth
    HelloWorld
    Histogram
    HistogramAtomics
    ImageOverlap
    LUDecomposition
    Mandelbrot
    MatrixMulImage
    MatrixMultiplication
    MatrixTranspose
    MemoryModel
    MonteCarloAsianMultiGPU
    NBody
    PrefixSum
    QuasiRandomSequence
    RadixSort
    RecursiveGaussian
    Reduction
    ScanLargeArrays
    SimpleConvolution
    SimpleImage
    SimpleMultiDevice
    SobelFilter
    StringSearch
    Template
    TransferOverlap
    URNG
    )

    # disabled c++ tests: DwtHaar1DCPPKernel EigenValue FFT IntroStaticCPPKernel
    # MatrixMultiplicationCPPKernel MersenneTwister SoAversusAoS TransferOverlapCPP
    #
    # disabled but fixable: CplusplusWrapper FluidSimulation2D GaussianNoise HDRToneMapping
    # ImageBandwidth KernelLaunch MatrixMulDouble MonteCarloAsian
    # MonteCarloAsianDP SobelFilterImage UnsharpMask
    #
    # disabled tests: NBody AtomicCounters BasicDebug DeviceFission11Ext
    #
    # disabled graphics tests: KmeansAutoclustering GaussianNoiseGL SimpleGL
    #
    # very slow: LDSBandwidth ConstantBandwidth MemoryOptimizations

    foreach(SAMPLE IN LISTS AMD_SAMPLES)
      add_test(NAME "AMD_29_${SAMPLE}"
        COMMAND "${TS_BASEDIR}/src/${TS_NAME}/AMD-APP-SDK-v2.9-RC-lnx64/samples/opencl/bin/x86_64/${SAMPLE}" "-q" "-t"
        WORKING_DIRECTORY "${TS_BASEDIR}/src/${TS_NAME}/AMD-APP-SDK-v2.9-RC-lnx64/samples/opencl/bin/x86_64")
      set_tests_properties("AMD_29_${SAMPLE}" PROPERTIES LABELS "amdsdk_29")
    endforeach()

  # TODO
  set_tests_properties(
    AMD_29_BitonicSort AMD_29_BinarySearch AMD_29_BinomialOption AMD_29_DCT
    AMD_29_BlackScholes AMD_29_FastWalshTransform AMD_29_FloydWarshall
    AMD_29_HelloWorld AMD_29_Histogram AMD_29_MatrixMultiplication
    AMD_29_MatrixTranspose AMD_29_PrefixSum AMD_29_QuasiRandomSequence
    AMD_29_ScanLargeArrays AMD_29_SimpleConvolution AMD_29_URNG
    PROPERTIES
      LABELS "hsa")


  else()
    message(STATUS "Disabling testsuite ${TS_NAME} - tarball not found")
  endif()

else()
  message(STATUS "AMD APP SDK testsuite is only enabled for x86_64 Linux systems currently")
endif()
