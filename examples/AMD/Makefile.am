# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/examples/AMD.
# 
# Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
# Copyright (c) 2012 Vincent Danjean <Vincent.Danjean@ens-lyon.org>
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

AMD-APP-SDK-v2.8-RC-lnx64.tgz:
	cd $(srcdir) && \
	wget http://developer.amd.com.php53-23.ord1-1.websitetestlink.com/wordpress/media/2012/11/AMD-APP-SDK-v2.8-lnx64.tgz && \
	tar xzf AMD-APP-SDK-v2.8-lnx64.tgz

EXTRA_DIST = $(srcdir)/*.patch

if TEST_SUITE_AMD
testsuite_pocl_dir=${abs_top_srcdir}/examples/AMD
testsuite_src_dir=${testsuite_pocl_dir}/AMD-APP-SDK-v2.8-RC-lnx64
amd_app_dir=$(testsuite_src_dir)/samples/opencl/cl/app

.PHONY: build

prepare-examples: $(testsuite_src_dir) build

$(testsuite_src_dir): AMD-APP-SDK-v2.8-RC-lnx64.tgz
	test -d $(testsuite_src_dir) || (cd $(testsuite_pocl_dir) && tar xzf AMD-APP-SDK-v2.8-RC-lnx64.tgz)

build: $(testsuite_src_dir) 
	cd $(testsuite_pocl_dir) && \
	( patch -sNp1 < $(testsuite_pocl_dir)/AMDSDK.patch || true ) && \
	cd $(testsuite_src_dir)/samples/opencl/ && \
	find . -name "*.cl" | xargs perl -pi -e 's/cl_amd_fp64/cl_khr_fp64/g' && \
	LDFLAGS="-lX11 -pthread ${LDFLAGS}" $(MAKE) -s 2>&1 | grep -v "No LSB modules are available" || true

test_AESEncryptDecrypt: 
	cd $(amd_app_dir)/AESEncryptDecrypt && \
	ln -sf $(amd_app_dir)/AESEncryptDecrypt/AESEncryptDecrypt_Kernels.cl $(amd_app_dir)/AESEncryptDecrypt/build/debug/x86_64/AESEncryptDecrypt_Kernels.cl && \
	ln -sf $(amd_app_dir)/AESEncryptDecrypt/input512.bmp $(amd_app_dir)/AESEncryptDecrypt/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/AESEncryptDecrypt -e || true

test_AtomicCounters: 
	cd $(amd_app_dir)/AtomicCounters && \
	ln -sf $(amd_app_dir)/AtomicCounters/AtomicCounters_Kernels.cl $(amd_app_dir)/AtomicCounters/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/AtomicCounters -e 

test_BitonicSort: 
	cd $(amd_app_dir)/BitonicSort && \
	ln -sf $(amd_app_dir)/BitonicSort/BitonicSort_Kernels.cl $(amd_app_dir)/BitonicSort/build/debug/x86_64/BitonicSort_Kernels.cl && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/BitonicSort -e

test_BlackScholes: 
	cd $(amd_app_dir)/BlackScholes && \
	ln -sf $(amd_app_dir)/BlackScholes/BlackScholes_Kernels.cl $(amd_app_dir)/BlackScholes/build/debug/x86_64/BlackScholes_Kernels.cl && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/BlackScholes -e

test_BlackScholesDP: 
	cd $(amd_app_dir)/BlackScholesDP && \
	ln -sf $(amd_app_dir)/BlackScholesDP/BlackScholesDP_Kernels.cl $(amd_app_dir)/BlackScholesDP/build/debug/x86_64/BlackScholesDP_Kernels.cl && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/BlackScholesDP -e

test_BinarySearch: 
	cd $(amd_app_dir)/BinarySearch && \
	ln -sf $(amd_app_dir)/BinarySearch/BinarySearch_Kernels.cl $(amd_app_dir)/BinarySearch/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/BinarySearch -e

test_BinomialOption: 
	cd $(amd_app_dir)/BinomialOption && \
	ln -sf $(amd_app_dir)/BinomialOption/BinomialOption_Kernels.cl $(amd_app_dir)/BinomialOption/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/BinomialOption -e

test_BoxFilter: 
	cd $(amd_app_dir)/BoxFilter && \
	ln -sf $(amd_app_dir)/BoxFilter/BoxFilter_Kernels.cl $(amd_app_dir)/BoxFilter/build/debug/x86_64/ && \
	ln -sf $(amd_app_dir)/BoxFilter/BoxFilter_Input.bmp $(amd_app_dir)/BoxFilter/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/BoxFilter -e

test_DCT: 
	cd $(amd_app_dir)/DCT && \
	ln -sf $(amd_app_dir)/DCT/DCT_Kernels.cl $(amd_app_dir)/DCT/build/debug/x86_64/DCT_Kernels.cl && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/DCT -e

test_DeviceFission: 
	cd $(amd_app_dir)/DeviceFission && \
	ln -sf $(amd_app_dir)/DeviceFission/DeviceFission_Kernels.cl $(amd_app_dir)/DeviceFission/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/DeviceFission -e

test_DwtHaar1D: 
	cd $(amd_app_dir)/DwtHaar1D && \
	ln -sf $(amd_app_dir)/DwtHaar1D/DwtHaar1D_Kernels.cl $(amd_app_dir)/DwtHaar1D/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/DwtHaar1D -e

test_FastWalshTransform: 
	cd $(amd_app_dir)/FastWalshTransform && \
	ln -sf $(amd_app_dir)/FastWalshTransform/FastWalshTransform_Kernels.cl $(amd_app_dir)/FastWalshTransform/build/debug/x86_64/FastWalshTransform_Kernels.cl && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/FastWalshTransform -e

test_FloydWarshall: 
	cd $(amd_app_dir)/FloydWarshall && \
	ln -sf $(amd_app_dir)/FloydWarshall/FloydWarshall_Kernels.cl $(amd_app_dir)/FloydWarshall/build/debug/x86_64/FloydWarshall_Kernels.cl && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/FloydWarshall -e

test_FluidSimulation2D: 
	cd $(amd_app_dir)/FluidSimulation2D && \
	ln -sf $(amd_app_dir)/FluidSimulation2D/FluidSimulation2D_Kernels.cl $(amd_app_dir)/FluidSimulation2D/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/FluidSimulation2D -e

test_Histogram: 
	cd $(amd_app_dir)/Histogram && \
	ln -sf $(amd_app_dir)/Histogram/Histogram_Kernels.cl $(amd_app_dir)/Histogram/build/debug/x86_64/Histogram_Kernels.cl && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/Histogram -e

test_HelloWorld: 
	cd $(amd_app_dir)/HelloWorld && \
	ln -sf $(amd_app_dir)/HelloWorld/HelloWorld_Kernels.cl $(amd_app_dir)/HelloWorld/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/HelloWorld -e

test_ImageOverlap: 
	cd $(amd_app_dir)/ImageOverlap && \
	ln -sf $(amd_app_dir)/ImageOverlap/ImageOverlap_Kernels.cl $(amd_app_dir)/ImageOverlap/build/debug/x86_64/ && \
	ln -sf $(amd_app_dir)/ImageOverlap/ImageOverlap_map.bmp $(amd_app_dir)/ImageOverlap/build/debug/x86_64/ && \
	ln -sf $(amd_app_dir)/ImageOverlap/ImageOverlap_verify_map.bmp $(amd_app_dir)/ImageOverlap/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/ImageOverlap -e

test_LUDecomposition: 
	cd $(amd_app_dir)/LUDecomposition && \
	ln -sf $(amd_app_dir)/LUDecomposition/LUDecomposition_Kernels.cl $(amd_app_dir)/LUDecomposition/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/LUDecomposition -e

test_Mandelbrot: 
	cd $(amd_app_dir)/Mandelbrot && \
	ln -sf $(amd_app_dir)/Mandelbrot/Mandelbrot_Kernels.cl $(amd_app_dir)/Mandelbrot/build/debug/x86_64/Mandelbrot_Kernels.cl && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/Mandelbrot -e

test_MatrixMultiplication: 
	cd $(amd_app_dir)/MatrixMultiplication && \
	ln -sf $(amd_app_dir)/MatrixMultiplication/MatrixMultiplication_Kernels.cl $(amd_app_dir)/MatrixMultiplication/build/debug/x86_64/MatrixMultiplication_Kernels.cl && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/MatrixMultiplication -e

test_MatrixMulImage: 
	cd $(amd_app_dir)/MatrixMulImage && \
	ln -sf $(amd_app_dir)/MatrixMulImage/MatrixMulImage_Kernels.cl $(amd_app_dir)/MatrixMulImage/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/MatrixMulImage -e

test_MatrixTranspose: 
	cd $(amd_app_dir)/MatrixTranspose && \
	ln -sf $(amd_app_dir)/MatrixTranspose/MatrixTranspose_Kernels.cl $(amd_app_dir)/MatrixTranspose/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/MatrixTranspose -e

test_MemoryModel: 
	cd $(amd_app_dir)/MemoryModel && \
	ln -sf $(amd_app_dir)/MemoryModel/MemoryModel_Kernels.cl $(amd_app_dir)/MemoryModel/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/MemoryModel -e

test_MonteCarloAsian: 
	cd $(amd_app_dir)/MonteCarloAsian && \
	ln -sf $(amd_app_dir)/MonteCarloAsian/MonteCarloAsian_Kernels.cl $(amd_app_dir)/MonteCarloAsian/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/MonteCarloAsian -e

test_MonteCarloAsianDP: 
	cd $(amd_app_dir)/MonteCarloAsianDP && \
	ln -sf $(amd_app_dir)/MonteCarloAsianDP/MonteCarloAsianDP_Kernels.cl $(amd_app_dir)/MonteCarloAsianDP/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/MonteCarloAsianDP -e

test_NBody: 
	cd $(amd_app_dir)/NBody && \
	ln -sf $(amd_app_dir)/NBody/NBody_Kernels.cl $(amd_app_dir)/NBody/build/debug/x86_64/NBody_Kernels.cl && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/NBody -e

test_PrefixSum: 
	cd $(amd_app_dir)/PrefixSum && \
	ln -sf $(amd_app_dir)/PrefixSum/PrefixSum_Kernels.cl $(amd_app_dir)/PrefixSum/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/PrefixSum -e

test_QuasiRandomSequence: 
	cd $(amd_app_dir)/QuasiRandomSequence && \
	ln -sf $(amd_app_dir)/QuasiRandomSequence/QuasiRandomSequence_Kernels.cl $(amd_app_dir)/QuasiRandomSequence/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/QuasiRandomSequence -e

test_RadixSort: 
	cd $(amd_app_dir)/RadixSort && \
	ln -sf $(amd_app_dir)/RadixSort/RadixSort_Kernels.cl $(amd_app_dir)/RadixSort/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/RadixSort -e

test_RecursiveGaussian: 
	cd $(amd_app_dir)/RecursiveGaussian && \
	ln -sf $(amd_app_dir)/RecursiveGaussian/RecursiveGaussian_Kernels.cl $(amd_app_dir)/RecursiveGaussian/build/debug/x86_64/ && \
	ln -sf $(amd_app_dir)/RecursiveGaussian/RecursiveGaussian_Input.bmp $(amd_app_dir)/RecursiveGaussian/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/RecursiveGaussian -e

test_Reduction: 
	cd $(amd_app_dir)/Reduction && \
	ln -sf $(amd_app_dir)/Reduction/Reduction_Kernels.cl $(amd_app_dir)/Reduction/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/Reduction -e

test_ScanLargeArrays: 
	cd $(amd_app_dir)/ScanLargeArrays && \
	ln -sf $(amd_app_dir)/ScanLargeArrays/ScanLargeArrays_Kernels.cl $(amd_app_dir)/ScanLargeArrays/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/ScanLargeArrays -e

test_SimpleConvolution: 
	cd $(amd_app_dir)/SimpleConvolution && \
	ln -sf $(amd_app_dir)/SimpleConvolution/SimpleConvolution_Kernels.cl $(amd_app_dir)/SimpleConvolution/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/SimpleConvolution -e

test_SimpleImage: 
	cd $(amd_app_dir)/SimpleImage && \
	ln -sf $(amd_app_dir)/SimpleImage/SimpleImage_Kernels.cl $(amd_app_dir)/SimpleImage/build/debug/x86_64/ && \
	ln -sf $(amd_app_dir)/SimpleImage/SimpleImage_Input.bmp $(amd_app_dir)/SimpleImage/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/SimpleImage -e

test_SobelFilter: 
	cd $(amd_app_dir)/SobelFilter && \
	ln -sf $(amd_app_dir)/SobelFilter/SobelFilter_Kernels.cl $(amd_app_dir)/SobelFilter/build/debug/x86_64/ && \
	ln -sf $(amd_app_dir)/SobelFilter/SobelFilter_Input.bmp $(amd_app_dir)/SobelFilter/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/SobelFilter -e

test_Template: 
	cd $(amd_app_dir)/Template && \
	ln -sf $(amd_app_dir)/Template/Template_Kernels.cl $(amd_app_dir)/Template/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/Template -e

test_TemplateC: 
	cd $(amd_app_dir)/TemplateC && \
	ln -sf $(amd_app_dir)/TemplateC/TemplateC_Kernels.cl $(amd_app_dir)/TemplateC/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/TemplateC -e

test_TransferOverlap: 
	cd $(amd_app_dir)/TransferOverlap && \
	ln -sf $(amd_app_dir)/TransferOverlap/TransferOverlap_Kernels.cl $(amd_app_dir)/TransferOverlap/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/TransferOverlap -e

test_URNG: 
	cd $(amd_app_dir)/URNG && \
	ln -sf $(amd_app_dir)/URNG/URNG_Kernels.cl $(amd_app_dir)/URNG/build/debug/x86_64/ && \
	ln -sf $(amd_app_dir)/URNG/URNG_Input.bmp $(amd_app_dir)/URNG/build/debug/x86_64/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/URNG -e

clean-examples:
	rm -fr $(testsuite_src_dir) Install-AMD-APP.sh default-install_lnx_64.pl ReadMe.txt

endif
