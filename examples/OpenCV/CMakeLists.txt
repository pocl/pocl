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

set(TS_NAME "OpenCV")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TESTSUITE_SOURCE_BASEDIR}/${TS_NAME}")

set(OPENCV "opencv-3.4.1")
set(OPENCV_ZIP "${TS_SRCDIR}/${OPENCV}.zip")

if(EXISTS "${OPENCV_ZIP}")

  message(STATUS "Enabling testsuite ${TS_NAME}")
  list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
  set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)

  ExternalProject_Add(
    ${TS_NAME}
    PREFIX "${TS_BASEDIR}"
    # Download URL is "https://github.com/Itseez/opencv/archive/${OPENCV_ZIP}"
    # but for some reason it doesn't work from cmake.
    DOWNLOAD_COMMAND pwd && unzip "${OPENCV_ZIP}"
      && rmdir OpenCV && mv ${OPENCV} OpenCV

    # TODO this should be optimized to only build
    # the stuff required for testing OpenCL
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=RelWithDebInfo
      -DWITH_CUDA=OFF
      -DWITH_OPENCL=ON
      -DWITH_FFMPEG=OFF
      -DBUILD_TESTS=ON
      -DBUILD_PERF_TESTS=ON
      -DBUILD_EXAMPLES=ON
      -DBUILD_DOCS=OFF
      -DWITH_IPP=OFF
    INSTALL_COMMAND /bin/true
  )

  set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_dependencies(prepare_examples ${TS_NAME})

  add_test(NAME test_UMat
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=UMat.*")

  add_test(NAME test_Core_UMat
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=Core_UMat.*")

  add_test(NAME test_Image2D
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=Image2D.*")

  add_test(NAME test_UMat_UMatBasicTests
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=UMat/UMatBasicTests.*")

  add_test(NAME test_UMat_UMatTestReshape
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=UMat/UMatTestReshape.*")

  add_test(NAME test_UMat_UMatTestRoi
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=UMat/UMatTestRoi.*")

  add_test(NAME test_UMat_UMatTestSizeOperations
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=UMat/UMatTestSizeOperations.*")

  add_test(NAME test_UMat_UMatTestUMatOperations
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=UMat/UMatTestUMatOperations.*")

  add_test(NAME test_OCL_MeanStdDev_
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_MeanStdDev_.*")

  add_test(NAME test_OCL_Channels_Merge
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Channels/Merge.*")

  add_test(NAME test_OCL_Channels_Split
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Channels/Split.*")

  add_test(NAME test_OCL_Channels_MixChannels
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Channels/MixChannels.*")

  add_test(NAME test_OCL_Channels_InsertChannels
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Channels/InsertChannels.*")

  add_test(NAME test_OCL_Channels_ExtractChannels
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Channels/ExtractChannels.*")

  add_test(NAME test_OCL_Arithm_Lut
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Lut.*")

  add_test(NAME test_OCL_Arithm_Add
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Add.*")

  add_test(NAME test_OCL_Arithm_Subtract
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Substract.*")

  add_test(NAME test_OCL_Arithm_Mul
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Mul.*")

  add_test(NAME test_OCL_Arithm_Div
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Div.*")

  add_test(NAME test_OCL_Arithm_AddWeighted
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/AddWeighted.*")

  add_test(NAME test_OCL_Arithm_Min
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Min.*")

  add_test(NAME test_OCL_Arithm_Max
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Max.*")

  add_test(NAME test_OCL_Arithm_Absdiff
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Absdiff.*")

  add_test(NAME test_OCL_Arithm_CartToPolar
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/CartToPolar.*")

  add_test(NAME test_OCL_Arithm_PolarToCart
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/PolarToCart.*")

  add_test(NAME test_OCL_Arithm_Transpose
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Transpose.*")

  add_test(NAME test_OCL_Arithm_Bitwise_and
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/BitWise_and.*")

  add_test(NAME test_OCL_Arithm_Bitwise_or
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Bitwise_or.*")

  add_test(NAME test_OCL_Arithm_Bitwise_xor
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Bitwise_xor.*")

  add_test(NAME test_OCL_Arithm_Bitwise_not
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Bitwise_not.*")

  add_test(NAME test_OCL_Arithm_Compare
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Compare.*")

  add_test(NAME test_OCL_Arithm_Pow
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Pow.*")

  add_test(NAME test_OCL_Arithm_SetIdentity
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/SetIdentity.*")

  add_test(NAME test_OCL_Arithm_Repeat
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Repeat.*")

  add_test(NAME test_OCL_Arithm_CountNonZero
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/CountNonZero.*")

  add_test(NAME test_OCL_Arithm_Sum
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Sum.*")

  add_test(NAME test_OCL_Arithm_MeanStdDev
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/MeanStdDev.*")

  add_test(NAME test_OCL_Arithm_Log
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Log.*")

  add_test(NAME test_OCL_Arithm_Exp
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Exp.*")

  add_test(NAME test_OCL_Arithm_Phase
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Phase.*")

  add_test(NAME test_OCL_Arithm_Magnitude
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Magnitude.*")

  add_test(NAME test_OCL_Arithm_Flip
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Flip.*")

  add_test(NAME test_OCL_Arithm_MinMaxIdx
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/MinMaxIdx.*")

  add_test(NAME test_OCL_Arithm_MinMaxIdx_Mask
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/MinMaxIdx_Mask.*")

  add_test(NAME test_OCL_Arithm_Norm
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Norm.*")

  add_test(NAME test_OCL_Arithm_UMatDot
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/UMatDot.*")

  add_test(NAME test_OCL_Arithm_Sqrt
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Sqrt.*")

  add_test(NAME test_OCL_Arithm_Normalize
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Normalize.*")

  add_test(NAME test_OCL_Arithm_InRange
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/InRange.*")

  add_test(NAME test_OCL_Arithm_ConvertScaleAbs
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/ConvertScaleAbs.*")

  add_test(NAME test_OCL_Arithm_ScaleAdd
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/ScaleAdd.*")

  add_test(NAME test_OCL_Arithm_PatchNaNs
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/PatchNaNs.*")

  add_test(NAME test_OCL_Arithm_Psnr
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/Psnr.*")

  add_test(NAME test_OCL_Arithm_ReduceSum
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/ReduceSum.*")

  add_test(NAME test_OCL_Arithm_ReduceMax
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/ReduceMax.*")

  add_test(NAME test_OCL_Arithm_ReduceMin
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/ReduceMin.*")

  add_test(NAME test_OCL_Arithm_ReduceAvg
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Arithm/ReduceAvg.*")

  add_test(NAME test_OCL_Core_Gemm
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Core/Gemm.*")

  add_test(NAME test_OCL_Core_Dft
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_Core/Dft.*")

  add_test(NAME test_OCL_OCL_ImgProc_MultiSpectrums
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_OCL_ImgProc/MultiSpectrums.*")

  add_test(NAME test_OCL_MatrixOperation_ConvertTo
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_MatrixOperation/ConvertTo.*")

  add_test(NAME test_OCL_MatrixOperation_CopyTo
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_MatrixOperation/CopyTo.*")

  add_test(NAME test_OCL_MatrixOperation_SetTo
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_MatrixOperation/SetTo.*")

  add_test(NAME test_OCL_MatrixOperation_UMatExpr
     COMMAND "${TS_BUILDDIR}/bin/opencv_test_core" "--gtest_filter=OCL_MatrixOperation/UMatExpr.*")

  set_tests_properties(
    test_UMat
    test_Core_UMat
    test_Image2D
    test_UMat_UMatBasicTests
    test_UMat_UMatTestReshape
    test_UMat_UMatTestRoi
    test_UMat_UMatTestSizeOperations
    test_UMat_UMatTestUMatOperations
    test_OCL_MeanStdDev_
    test_OCL_Channels_Merge
    test_OCL_Channels_Split
    test_OCL_Channels_MixChannels
    test_OCL_Channels_InsertChannels
    test_OCL_Channels_ExtractChannels
    test_OCL_Arithm_Lut
    test_OCL_Arithm_Add
    test_OCL_Arithm_Subtract
    test_OCL_Arithm_Mul
    test_OCL_Arithm_Div
    test_OCL_Arithm_AddWeighted
    test_OCL_Arithm_Min
    test_OCL_Arithm_Max
    test_OCL_Arithm_Absdiff
    test_OCL_Arithm_CartToPolar
    test_OCL_Arithm_PolarToCart
    test_OCL_Arithm_Transpose
    test_OCL_Arithm_Bitwise_and
    test_OCL_Arithm_Bitwise_or
    test_OCL_Arithm_Bitwise_xor
    test_OCL_Arithm_Bitwise_not
    test_OCL_Arithm_Compare
    test_OCL_Arithm_Pow
    test_OCL_Arithm_SetIdentity
    test_OCL_Arithm_Repeat
    test_OCL_Arithm_CountNonZero
    test_OCL_Arithm_Sum
    test_OCL_Arithm_MeanStdDev
    test_OCL_Arithm_Log
    test_OCL_Arithm_Exp
    test_OCL_Arithm_Phase
    test_OCL_Arithm_Magnitude
    test_OCL_Arithm_Flip
    test_OCL_Arithm_MinMaxIdx
    test_OCL_Arithm_MinMaxIdx_Mask
    test_OCL_Arithm_Norm
    test_OCL_Arithm_UMatDot
    test_OCL_Arithm_Sqrt
    test_OCL_Arithm_Normalize
    test_OCL_Arithm_InRange
    test_OCL_Arithm_ConvertScaleAbs
    test_OCL_Arithm_ScaleAdd
    test_OCL_Arithm_PatchNaNs
    test_OCL_Arithm_Psnr
    test_OCL_Arithm_ReduceSum
    test_OCL_Arithm_ReduceMax
    test_OCL_Arithm_ReduceMin
    test_OCL_Arithm_ReduceAvg
    test_OCL_Core_Gemm
    test_OCL_Core_Dft
    test_OCL_OCL_ImgProc_MultiSpectrums
    test_OCL_MatrixOperation_ConvertTo
    test_OCL_MatrixOperation_CopyTo
    test_OCL_MatrixOperation_SetTo
    test_OCL_MatrixOperation_UMatExpr

    PROPERTIES
      ENVIRONMENT "OPENCV_OPENCL_DEVICE=:CPU:0"
      LABELS "OpenCV")


else()

  message(STATUS "Disabling testsuite ${TS_NAME}, required files not found" )

endif()
