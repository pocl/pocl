# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/examples/OpenCV.
# 
# Copyright (c) 2012 Lassi Koskinen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

if TEST_SUITE_OPENCV
opencv_local_src_dir=opencv-3.0.0-beta
opencv_zip=opencv-3.0.0-beta.zip
testsuite_pocl_dir=${abs_top_srcdir}/examples/OpenCV
testsuite_src_dir=${testsuite_pocl_dir}/${opencv_local_src_dir}

.PHONY: prepare-examples

$(srcdir)/$(opencv_zip):
	cd $(srcdir) && \
	wget https://github.com/Itseez/opencv/archive/$(opencv_zip)

EXTRA_DIST = opencv.patch

$(opencv_local_src_dir):
	$(MAKE) $(srcdir)/$(opencv_zip) 
	cd $(testsuite_pocl_dir) && \
	unzip $(opencv_zip)

test_UMat:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=UMat.*

test_Core_UMat:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=Core_UMat.*

test_Image2D:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=Image2D.*

test_UMat/UMatBasicTests:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=UMat/UMatBasicTests.*

test_UMat/UMatTestReshape:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=UMat/UMatTestReshape.*

test_UMat/UMatTestRoi:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=UMat/UMatTestRoi.*

test_UMat/UMatTestSizeOperations:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=UMat/UMatTestSizeOperations.*

test_UMat/UMatTestUMatOperations:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=UMat/UMatTestUMatOperations.*

test_OCL_MeanStdDev_:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_MeanStdDev_.*

test_OCL_Channels/Merge:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Channels/Merge.*

test_OCL_Channels/Split:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Channels/Split.*

test_OCL_Channels/MixChannels:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Channels/MixChannels.*

test_OCL_Channels/InsertChannels:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Channels/InsertChannels.*

test_OCL_Channels/ExtractChannels:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Channels/ExtractChannels.*

test_OCL_Arithm/Lut:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Lut.*

test_OCL_Arithm/Add:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Add.*

test_OCL_Arithm/Subtract:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Substract.*

test_OCL_Arithm/Mul:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Mul.*

test_OCL_Arithm/Div:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Div.*

test_OCL_Arithm/AddWeighted:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/AddWeighted.*

test_OCL_Arithm/Min:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Min.*

test_OCL_Arithm/Max:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Max.*

test_OCL_Arithm/Absdiff:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Absdiff.*

test_OCL_Arithm/CartToPolar:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/CartToPolar.*

test_OCL_Arithm/PolarToCart:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/PolarToCart.*

test_OCL_Arithm/Transpose:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Transpose.*

test_OCL_Arithm/Bitwise_and:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/BitWise_and.*

test_OCL_Arithm/Bitwise_or:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Bitwise_or.*

test_OCL_Arithm/Bitwise_xor:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Bitwise_xor.*

test_OCL_Arithm/Bitwise_not:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Bitwise_not.*

test_OCL_Arithm/Compare:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Compare.*

test_OCL_Arithm/Pow:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Pow.*

test_OCL_Arithm/SetIdentity:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/SetIdentity.*

test_OCL_Arithm/Repeat:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Repeat.*

test_OCL_Arithm/CountNonZero:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/CountNonZero.*

test_OCL_Arithm/Sum:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Sum.*

test_OCL_Arithm/MeanStdDev:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/MeanStdDev.*

test_OCL_Arithm/Log:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Log.*

test_OCL_Arithm/Exp:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Exp.*

test_OCL_Arithm/Phase:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Phase.*

test_OCL_Arithm/Magnitude:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Magnitude.*

test_OCL_Arithm/Flip:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Flip.*

test_OCL_Arithm/MinMaxIdx:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/MinMaxIdx.*

test_OCL_Arithm/MinMaxIdx_Mask:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/MinMaxIdx_Mask.*

test_OCL_Arithm/Norm:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Norm.*

test_OCL_Arithm/UMatDot:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/UMatDot.*

test_OCL_Arithm/Sqrt:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Sqrt.*

test_OCL_Arithm/Normalize:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Normalize.*

test_OCL_Arithm/InRange:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/InRange.*

test_OCL_Arithm/ConvertScaleAbs:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/ConvertScaleAbs.*

test_OCL_Arithm/ScaleAdd:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/ScaleAdd.*

test_OCL_Arithm/PatchNaNs:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/PatchNaNs.*

test_OCL_Arithm/Psnr:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/Psnr.*

test_OCL_Arithm/ReduceSum:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/ReduceSum.*

test_OCL_Arithm/ReduceMax:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/ReduceMax.*

test_OCL_Arithm/ReduceMin:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/ReduceMin.*

test_OCL_Arithm/ReduceAvg:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Arithm/ReduceAvg.*

test_OCL_Core/Gemm:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Core/Gemm.*

test_OCL_Core/Dft:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_Core/Dft.*

test_OCL_OCL_ImgProc/MultiSpectrums:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_OCL_ImgProc/MultiSpectrums.*

test_OCL_MatrixOperation/ConvertTo:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_MatrixOperation/ConvertTo.*

test_OCL_MatrixOperation/CopyTo:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_MatrixOperation/CopyTo.*

test_OCL_MatrixOperation/SetTo:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_MatrixOperation/SetTo.*

test_OCL_MatrixOperation/UMatExpr:
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors $(opencv_local_src_dir)/build/bin/opencv_test_core --gtest_filter=OCL_MatrixOperation/UMatExpr.*

# OpenCV must be built in the subdirectory build/ of the extracted source
# (it does not work elsewhere) and its source tree is modified.
# So, copying the source tree in our build tree if required
# (ie if $(srcdir) is not $(CURDIR) )
build:
	if [ ! -d $(opencv_local_src_dir) ]; then cp -aL $(testsuite_src_dir) . ; fi
	( patch -sNp1 < $(testsuite_pocl_dir)/opencv.patch || true )
	mkdir $(opencv_local_src_dir)/build; \
	cd $(opencv_local_src_dir)/build; \
	cmake \
		-D CMAKE_BUILD_TYPE=RELEASE \
		-D WITH_CUDA=OFF \
		-D WITH_OPENCL=ON \
		-D WITH_FFMPEG=OFF \
		-D BUILD_TESTS=ON \
		-D BUILD_PERF_TESTS=ON \
                -D BUILD_EXAMPLES=ON \
		-D BUILD_DOCS=ON \
		.. && \
	$(MAKE) -s

prepare-examples: $(opencv_local_src_dir) build

clean-examples:
	rm -fr $(opencv_local_src_dir)

endif

