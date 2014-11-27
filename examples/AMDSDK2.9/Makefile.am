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

EXTRA_DIST = $(srcdir)/*.patch

if TEST_SUITE_AMDSDK2_9
testsuite_pocl_dir=${abs_top_srcdir}/examples/AMDSDK2.9
testsuite_src_dir=${testsuite_pocl_dir}/AMD-APP-SDK-v2.9-RC-lnx64
amd_app_dir=$(testsuite_src_dir)/samples/opencl/

.PHONY: build prepare-examples

prepare-examples: $(testsuite_src_dir) build

$(testsuite_src_dir): AMD-APP-SDK-v2.9-RC-lnx64.tgz
	test -d $(testsuite_src_dir) || (cd $(testsuite_pocl_dir) && tar xzf AMD-APP-SDK-v2.9-RC-lnx64.tgz)

# Some of the tests do not build with OpenGL 3.0. Just skip them by using make -k.
build: $(testsuite_src_dir) 
	-cd $(testsuite_pocl_dir) && \
	( patch -sNp1 < $(testsuite_pocl_dir)/AMDSDK2_9.patch || true ) && \
	cd $(testsuite_src_dir)/samples/ && \
	find . -name "CMakeLists.txt" -exec sed -i \
		-e 's/mark_as_advanced( OPENCL_LIBRARIES )/set(OPENCL_LIBRARIES "-lOpenCL")/g' \
		-e 's/mark_as_advanced(OPENCL_LIBRARIES)/set(OPENCL_LIBRARIES "-lOpenCL")/g' "{}" \; && \
	LDFLAGS="-pthread ${LDFLAGS}" cmake \
		-DBUILD_OPENCL=ON \
		-DBUILD_OPENCV=OFF \
		-DBUILD_BOLT=OFF .;\
	make -sk

test_AsyncDataTransfer:
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cpp_cl/AsyncDataTransfer/bin/x86_64/Release/AsyncDataTransfer -e 

test_AtomicCounters: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/AtomicCounters/bin/x86_64/Release/AtomicCounters -e 

test_BasicDebug: 
	cd $(amd_app_dir)cl/BasicDebug/bin/x86_64/Release && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./BasicDebug -e

test_BinarySearch: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/BinarySearch/bin/x86_64/Release/BinarySearch -e

test_BinomialOption: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/BinomialOption/bin/x86_64/Release/BinomialOption -e

test_BinomialOptionMultiGPU: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/BinomialOptionMultiGPU/bin/x86_64/Release/BinomialOptionMultiGPU -e

test_BitonicSort: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/BitonicSort/bin/x86_64/Release/BitonicSort -e

test_BlackScholes: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/BlackScholes/bin/x86_64/Release/BlackScholes -e

test_BlackScholesDP: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/BlackScholesDP/bin/x86_64/Release/BlackScholesDP -e

test_BoxFilter: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/BoxFilter/bin/x86_64/Release/BoxFilter -e

test_BoxFilterGL: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/BoxFilterGL/bin/x86_64/Release/BoxFilterGL -e

test_BufferBandwidth: 
	cd $(amd_app_dir)cl/BufferBandwidth/bin/x86_64/Release && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./BufferBandwidth -e

test_BufferImageInterop: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/BufferImageInterop/bin/x86_64/Release/BufferImageInterop -e

test_ConcurrentKernel:
	cd $(amd_app_dir)cpp_cl/ConcurrentKernel/bin/x86_64/Release && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./ConcurrentKernel -e

test_ConstantBandwidth:
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors benchmark/ConstantBandwidth/bin/x86_64/Release/ConstantBandwidth -e

test_CplusplusWrapper:
	cd $(amd_app_dir)cpp_cl/CplusplusWrapper/bin/x86_64/Release && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./CplusplusWrapper -e

test_DCT: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/DCT/bin/x86_64/Release/DCT -e

test_DeviceFission: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/DeviceFission/bin/x86_64/Release/DeviceFission -e

test_DeviceFission11Ext: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/DeviceFission11Ext/bin/x86_64/Release/DeviceFission11Ext -e

test_DwtHaar1D: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/DwtHaar1D/bin/x86_64/Release/DwtHaar1D -e

test_DwtHaar1DCPPKernel: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/DwtHaar1DCPPKernel/bin/x86_64/Release/DwtHaar1DCPPKernel -e

test_Eigenvalue:
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/Eigenvalue/bin/x86_64/Release/Eigenvalue -e

test_FastWalshTransform: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/FastWalshTransform/bin/x86_64/Release/FastWalshTransform -e

test_FFT:
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/FFT/bin/x86_64/Release/FFT -e

test_FloydWarshall: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/FloydWarshall/bin/x86_64/Release/FloydWarshall -e

test_FluidSimulation2D: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/FluidSimulation/bin/x86_64/Release/FluidSimulation2D -e

test_GaussianNoise:
	cd $(amd_app_dir)/cpp_cl/GaussianNoise/bin/x86_64/Release && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./GaussianNoise -e --platformId 0

test_GaussianNoiseGL:
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/GaussianNoiseGL/bin/x86_64/Release/GaussianNoiseGL -e

test_GlobalMemoryBandwidth:
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/GlobalMemoryBandwidth/bin/x86_64/Release/GlobalMemoryBandwidth -e

test_HDRToneMapping:
	cd $(amd_app_dir)/cpp_cl/HDRToneMapping/bin/x86_64/Release && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./HDRToneMapping -e --platformId 0

test_HelloWorld: 
	cd $(amd_app_dir)/cl/HelloWorld/bin/x86_64/Release/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./HelloWorld -e

test_Histogram: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/Histogram/bin/x86_64/Release/Histogram -e

test_HistogramAtomics:
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cpp_cl/HistogramAtomics/bin/x86_64/Release/HistogramAtomics -e

test_ImageBandwidth: 
	cd $(amd_app_dir)cl/ImageBandwidth/bin/x86_64/Release && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./ImageBandwidth -e

test_ImageOverlap: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/ImageOverlap/bin/x86_64/Release/ImageOverlap -e

test_IntroStaticCPPKernel:
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cpp_cl/IntroStaticCPPKernel/bin/x86_64/Release/IntroStaticCPPKernel -e

test_KernelLaunch:
	cd $(amd_app_dir)cl/KernelLaunch/bin/x86_64/Release && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./KernelLaunch -e

test_KmeansAutoclustering:
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/KmeansAutoclustering/bin/x86_64/Release/KmeansAutoclustering -e

test_LDSBandwidth:
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors benchmark/LDSBandwidth/bin/x86_64/Release/LDSBandwidth -e

test_LUDecomposition: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/LUDecomposition/bin/x86_64/Release/LUDecomposition -e

test_Mandelbrot: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors bin/x86_64/Mandelbrot -e

test_MatrixMulDouble: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cpp_cl/MatrixMulDouble/bin/x86_64/Release/MatrixMulDouble -e --platformId 0

test_MatrixMulImage: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/MatrixMulImage/bin/x86_64/Release/MatrixMulImage -e

test_MatrixMultiplication: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/MatrixMultiplication/bin/x86_64/Release/MatrixMultiplication -e

test_MatrixTranspose: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/MatrixTranspose/bin/x86_64/Release/MatrixTranspose -e

test_MemoryModel: 
	cd $(amd_app_dir)cl/MemoryModel/bin/x86_64/Release && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./MemoryModel -e

test_MemoryOptimizations: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors benchmark/MemoryOptimizations/bin/x86_64/Release/MemoryOptimizations -e

test_MersenneTwister: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cpp_cl/MerzenneTwister/bin/x86_64/Release/MersenneTwister -e

test_MonteCarloAsian: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/MonteCarloAsian/bin/x86_64/Release/MonteCarloAsian -e

test_MonteCarloAsianDP: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/MonteCarloAsianDP/bin/x86_64/Release/MonteCarloAsianDP -e

test_MonteCarloAsianMultiGPU: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/MonteCarloAsianMultiGPU/bin/x86_64/Release/MonteCarloAsianMultiGPU -e

test_NBody: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/NBody/bin/x86_64/Release/NBody -e

test_PrefixSum: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/PrefixSum/bin/x86_64/Release/PrefixSum -e

test_QuasiRandomSequence: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/QuasiRandomSequence/bin/x86_64/Release/QuasiRandomSequence -e

test_RadixSort: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/RadixSort/bin/x86_64/Release/RadixSort -e

test_RecursiveGaussian: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/RecursiveGaussian/bin/x86_64/Release/RecursiveGaussian -e

test_Reduction: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/Reduction/bin/x86_64/Release/Reduction -e

test_ScanLargeArrays: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/ScanLargeArrays/bin/x86_64/Release/ScanLargeArrays -e

test_SimpleConvolution: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/SimpleConvolution/bin/x86_64/Release/SimpleConvolution -e

test_SimpleGL: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/SimpleGL/bin/x86_64/Release/SimpleGL -e

test_SimpleImage: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/SimpleImage/bin/x86_64/Release/SimpleImage -e

test_SoAversusAoS:
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cpp_cl/SoAversusAoS/bin/x86_64/Release/SoAversusAoS -e

test_SobelFilter: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/SobelFilter/bin/x86_64/Release/SobelFilter -e

test_SobelFilterImage: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cpp_cl/SobelFilterImage/bin/x86_64/Release/SobelFilterImage -e

test_StringSearch:
	cd $(amd_app_dir)cl/StringSearch/bin/x86_64/Release/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./StringSearch -e

test_Template: 
	cd $(amd_app_dir)cl/Template/bin/x86_64/Release/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./Template -e

test_TransferOverlap: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/TransferOverlap/bin/x86_64/Release/TransferOverlap -e

test_TransferOverlapCPP: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/TransferOverlapCPP/bin/x86_64/Release/TransferOverlapCPP -e

test_UnsharpMask:
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cpp_cl/UnsharpMask/bin/x86_64/Release/UnsharpMask -e

test_URNG: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/URNG/bin/x86_64/Release/URNG -e

test_URNGNoiseGL: 
	cd $(amd_app_dir) && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors cl/URNGNoiseGL/bin/x86_64/Release/URNGNoiseGL -e

clean-examples:
	rm -fr $(testsuite_src_dir) Install-AMD-APP.sh default-install_lnx_64.pl ReadMe.txt

endif
