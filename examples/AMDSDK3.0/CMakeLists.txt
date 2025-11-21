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

set(TS_NAME AMDSDK3.0)
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")
set(TS_ROOT "${TS_BASEDIR}/src/${TS_NAME}")

if(NOT (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" AND X86_64))
  message(STATUS "Disabling testsuite ${TS_NAME}, requires Linux x86-64")
  return()
endif()

# https://archive.org/download/AMDAPPSDK has this version available:
set(AMD_APP_SDK_TGZ "${TS_SRCDIR}/AMD-APP-SDK-v3.0.130.136-GA-linux64.sh")
if(NOT EXISTS "${AMD_APP_SDK_TGZ}")
  # The previous version hangs around in the webz too.
  set(AMD_APP_SDK_TGZ "${TS_SRCDIR}/AMD-APP-SDK-v3.0.130.135-GA-linux64.sh")
endif()

if(NOT EXISTS "${AMD_APP_SDK_TGZ}")
  message(STATUS "Disabling testsuite ${TS_NAME}, could not find source files")
  return()
endif()

if(NOT Python3_EXECUTABLE)
  message(STATUS "Disabling testsuite ${TS_NAME}, can't find suitable python3")
  return()
else()
  set(PYTHON_INTERP "${Python3_EXECUTABLE}")
endif()

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"

  SOURCE_DIR "${TS_ROOT}"
  SOURCE_SUBDIR "AMD-APP-SDK-3.0/samples"

  DOWNLOAD_COMMAND ""

  PATCH_COMMAND  ${BASH} -c "if [ ! -e ${TS_ROOT}/AMD-APP-SDK-3.0/install.sh ] $<SEMICOLON> then \
         ${AMD_APP_SDK_TGZ} --nox11 --noexec --keep --target AMD-APP-SDK-3.0 && \
         patch -p1 -i ${CMAKE_CURRENT_SOURCE_DIR}/amdsdk3_0.patch && \
         rm ${TS_ROOT}/AMD-APP-SDK-3.0/lib/x86_64/libOpenCL* \
         $<SEMICOLON> fi"

  CMAKE_ARGS ${UPSTREAM_CMAKE_ARGS}
    "-DCMAKE_CXX_FLAGS_DEBUG=-O0 -ggdb -Wno-deprecated-declarations -Wno-narrowing"
    "-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG -Wno-deprecated-declarations -Wno-narrowing"
    "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -ggdb -Wno-deprecated-declarations -Wno-narrowing"
    -DBUILD_OPENCL=ON -DBUILD_OPENCV=OFF -DBUILD_BOLT=OFF -DBUILD_AMP=OFF
    INSTALL_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/install.sh "${TS_ROOT}" "${CMAKE_BUILD_TYPE}"
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

set(AMD_SDK_BINARY_DIR "${TS_ROOT}/AMD-APP-SDK-3.0/samples/opencl/bin/x86_64")

set(AMD_SAMPLES
    AdvancedConvolution
    BasicDebug
    BinomialOption
    BinomialOptionMultiGPU
    BitonicSort
    BlackScholes
    BlackScholesDP
    BoxFilter
    BuiltInScan
    ConcurrentKernel
    DCT
    DeviceFission11Ext
    DwtHaar1D
    FastWalshTransform
    FloydWarshall
    FluidSimulation2D
    GaussianNoise
    HDRToneMapping
    HelloWorld
    Histogram
    HistogramAtomics
    ImageBinarization
    ImageOverlap
    KernelLaunch
    LUDecomposition
    Mandelbrot
    MatrixMulImage
    MatrixMultiplication
    MatrixTranspose
    MemoryModel
    MonteCarloAsian
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
    UnsharpMask
    URNG)

    # freeze (requires async queue): FineGrainSVM
    # disabled c++ tests: DwtHaar1DCPPKernel EigenValue
    #     IntroStaticCPPKernel FFT MatrixMultiplicationCPPKernel
    #     MersenneTwister SoAversusAoS TransferOverlapCPP
    # disabled but fixable: BuiltInScan CalcPie CplusplusWrapper
    #     GaussianNoise HDRToneMapping
    #     HeatPDE ImageBandwidth KernelLaunch MatrixMulDouble
    #     MonteCarloAsianDP SobelFilterImage UnsharpMask DynamicOpenCLDetection
    #     FineGrainSVMCAS RecursiveGaussian_ProgramScope SobelFilterImage
    #     SVMAtomicsBinaryTreeInsert SVMBinaryTreeSearch
    # disabled tests: AsyncDataTransfer AtomicCounters BasicDebug
    #     DeviceFission11Ext RangeMinimumQuery SimpleDepthImage
    #     SimpleGenericAddressSpace
    # disabled graphics tests: KmeansAutoclustering
    #     GaussianNoiseGL SimpleGL BoxFilterGL URNGNoiseGL
    # very slow: BufferBandwidth LDSBandwidth ConstantBandwidth
    #     MemoryOptimizations TransferOverlap
    # disabled - requires dev queue: BinarySearchDeviceSideEnqueue
    #     DeviceEnqueueBFS ExtractPrimes GlobalMemoryBandwidth
    #     RegionGrowingSegmentation SimpleSPIR
    # disabled - requires pipe: PipeProducerConsumerKernels SimplePipe
    # requires OpenCL 2.0: BufferImageInterop
    # requires subdevices:
    # DeviceFission

set(HSA_LABELED_TESTS
    AMD_30_BinomialOption AMD_30_DCT AMD_30_BlackScholes
    AMD_30_FastWalshTransform AMD_30_FloydWarshall AMD_30_HelloWorld
    AMD_30_Histogram AMD_30_MatrixMultiplication AMD_30_MatrixTranspose
    AMD_30_PrefixSum AMD_30_QuasiRandomSequence AMD_30_ScanLargeArrays
    AMD_30_SimpleConvolution AMD_30_URNG)

if(ENABLE_HSA)
  list(APPEND AMD_SAMPLES CalcPie)
  list(APPEND HSA_LABELED_TESTS AMD_30_CalcPie)
endif()

foreach(SAMPLE IN LISTS AMD_SAMPLES)
  add_test(NAME "AMD_30_${SAMPLE}"
    COMMAND "${AMD_SDK_BINARY_DIR}/${SAMPLE}" "-q" "-t" "-e"
    WORKING_DIRECTORY "${AMD_SDK_BINARY_DIR}")
  set_tests_properties("AMD_30_${SAMPLE}" PROPERTIES LABELS "amdsdk_30")
endforeach()

# disabled because of stack size requirement & not checking max workgroup size
if(HOST_CPU_ENABLE_STACK_SIZE_CHECK)
  foreach(SAMPLE IN ImageOverlap MonteCarloAsianMultiGPU MatrixMulImage SimpleImage)
    set_tests_properties("AMD_30_${SAMPLE}" PROPERTIES DISABLED 1)
  endforeach()
endif()

set_tests_properties(
  ${HSA_LABELED_TESTS}
  PROPERTIES
    LABELS "hsa")

# XFAILs as of 2016-05-17:
# AMD_30_BuiltInScan: undefined symbol: work_group_scan_inclusive_add
# AMD_30_GaussianNoise, AMD_30_HDRToneMapping: Invalid platform.
# AMD_30_ImageBinarization: work_group_barrier missing
# AMD_30_HDRToneMapping: no output
# AMD_30_UnsharpMask: clCreateContextFromType failed
set_tests_properties(AMD_30_BuiltInScan
  AMD_30_GaussianNoise AMD_30_HDRToneMapping
  AMD_30_ImageBinarization AMD_30_KernelLaunch
  AMD_30_UnsharpMask
  PROPERTIES DISABLED 1)

## Benchmarks. Unlike the tests above, these run longer and disable verification
# Mandelbrot & MatrixMult are disabled for benchmarking, because the resulting times vary too much

set(TEST_LAUNCHER_SCRIPT ${PYTHON_INTERP} "${CMAKE_CURRENT_SOURCE_DIR}/amdsdk_bench_runner.py" "--output=${CMAKE_BINARY_DIR}/amdsdk_bench_result.json")

add_test(NAME "AMDSDK_Bench_BitonicSort"
         COMMAND ${TEST_LAUNCHER_SCRIPT} --field 3 "${AMD_SDK_BINARY_DIR}/BitonicSort" -q -t -x 1048576 -i 400
         WORKING_DIRECTORY "${TS_BUILDDIR}")

add_test(NAME "AMDSDK_Bench_BinomialOption"
         COMMAND ${TEST_LAUNCHER_SCRIPT} --field 3 "${AMD_SDK_BINARY_DIR}/BinomialOption"  -q -t -x 12288 -i 400
         WORKING_DIRECTORY "${TS_BUILDDIR}")

add_test(NAME "AMDSDK_Bench_BlackScholes"
        COMMAND ${TEST_LAUNCHER_SCRIPT} --field 3 "${AMD_SDK_BINARY_DIR}/BlackScholes" -q -t -x 16777216 -i 100
        WORKING_DIRECTORY "${TS_BUILDDIR}")

add_test(NAME "AMDSDK_Bench_DCT"
       COMMAND ${TEST_LAUNCHER_SCRIPT} --field 4 "${AMD_SDK_BINARY_DIR}/DCT" -q -t -x 4096 -y 4096 -i 50
       WORKING_DIRECTORY "${TS_BUILDDIR}")

add_test(NAME "AMDSDK_Bench_FastWalshTransform"
        COMMAND ${TEST_LAUNCHER_SCRIPT} --field 3 "${AMD_SDK_BINARY_DIR}/FastWalshTransform" -q -t -x 134217728 -i 10
        WORKING_DIRECTORY "${TS_BUILDDIR}")

add_test(NAME "AMDSDK_Bench_FloydWarshall"
         COMMAND ${TEST_LAUNCHER_SCRIPT} --field 3 "${AMD_SDK_BINARY_DIR}/FloydWarshall" -q -t -x 512 -i 200
         WORKING_DIRECTORY "${TS_BUILDDIR}")

add_test(NAME "AMDSDK_Bench_Histogram"
         COMMAND ${TEST_LAUNCHER_SCRIPT} --field 4 "${AMD_SDK_BINARY_DIR}/Histogram" -t -x 12288 -y 12288 -q -i 50
         WORKING_DIRECTORY "${TS_BUILDDIR}")

#add_test(NAME "AMDSDK_Bench_Mandelbrot"
#         COMMAND ${TEST_LAUNCHER_SCRIPT} --field 4 "${AMD_SDK_BINARY_DIR}/Mandelbrot" -t -H 8192 -W 8192 -q
#         WORKING_DIRECTORY "${TS_BUILDDIR}")

add_test(NAME "AMDSDK_Bench_MatrixTranspose"
         COMMAND ${TEST_LAUNCHER_SCRIPT} --field 3 "${AMD_SDK_BINARY_DIR}/MatrixTranspose" -t -x 12288 -q -i 50
         WORKING_DIRECTORY "${TS_BUILDDIR}")

#add_test(NAME "AMDSDK_Bench_MatrixMultiplication"
#         COMMAND ${TEST_LAUNCHER_SCRIPT} --field 4 "${AMD_SDK_BINARY_DIR}/MatrixMultiplication" -q -t -x 2048 -y 2048 -z 2048 -i 40
#         WORKING_DIRECTORY "${TS_BUILDDIR}")

add_test(NAME "AMDSDK_Bench_NBody"
         COMMAND ${TEST_LAUNCHER_SCRIPT} --field 3 "${AMD_SDK_BINARY_DIR}/NBody" -t -x 16384 -q -i 50
         WORKING_DIRECTORY "${TS_BUILDDIR}")

add_test(NAME "AMDSDK_Bench_QuasiRandomSequence"
         COMMAND ${TEST_LAUNCHER_SCRIPT} --field 3 "${AMD_SDK_BINARY_DIR}/QuasiRandomSequence" -q -t -y 8192 -x 8192 -i 200
         WORKING_DIRECTORY "${TS_BUILDDIR}")

add_test(NAME "AMDSDK_Bench_RadixSort"
         COMMAND ${TEST_LAUNCHER_SCRIPT} --field 3 "${AMD_SDK_BINARY_DIR}/RadixSort" -q -t -x 65536 -i 200
         WORKING_DIRECTORY "${TS_BUILDDIR}")

add_test(NAME "AMDSDK_Bench_Reduction"
         COMMAND ${TEST_LAUNCHER_SCRIPT} --field 3 "${AMD_SDK_BINARY_DIR}/Reduction" -q -t -x 122880000 -i 50
         WORKING_DIRECTORY "${TS_BUILDDIR}")

add_test(NAME "AMDSDK_Bench_SimpleConvolution"
         COMMAND ${TEST_LAUNCHER_SCRIPT} --field 4 "${AMD_SDK_BINARY_DIR}/SimpleConvolution" -q -t -x 256000 -i 40
         WORKING_DIRECTORY "${TS_BUILDDIR}")

# Mandelbrot MatrixMultiplication
foreach(BENCH IN ITEMS BitonicSort BinomialOption BlackScholes DCT FastWalshTransform FloydWarshall Histogram
        MatrixTranspose NBody QuasiRandomSequence RadixSort Reduction SimpleConvolution)
  # RUN_SERIAL to avoid running anything else at the same time
  set_tests_properties("AMDSDK_Bench_${BENCH}" PROPERTIES LABELS "amdsdk_bench" RUN_SERIAL ON)
endforeach()
