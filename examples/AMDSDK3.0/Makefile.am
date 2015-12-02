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

if TEST_SUITE_AMDSDK3_0
testsuite_pocl_dir=${abs_top_srcdir}/examples/AMDSDK3.0
testsuite_src_dir=${testsuite_pocl_dir}/AMD-APP-SDK-3.0
ocl_sample_dir=${testsuite_pocl_dir}/AMD-APP-SDK-3.0/samples/opencl
tar_bz=AMD-APP-SDKInstaller-v3.0.130.135-GA-linux64.tar.bz2
tar_sh=AMD-APP-SDK-v3.0.130.135-GA-linux64.sh
build_type=RelWithDebInfo

.PHONY: build prepare-examples

prepare-examples: $(testsuite_src_dir) build

$(tar_sh): $(tar_bz)
	test -f $(tar_sh) || (cd $(testsuite_pocl_dir) && tar xjf $(tar_bz) )

$(testsuite_src_dir): $(tar_sh)
	test -d $(testsuite_src_dir) || (cd $(testsuite_pocl_dir) && ./$(tar_sh) --noexec --keep --target AMD-APP-SDK-3.0 && \
	( patch -sNp1 < $(testsuite_pocl_dir)/amdsdk3_0.patch || true ) && \
	find $(testsuite_src_dir)/samples -name "CMakeLists.txt" -exec sed -i \
		-e 's/mark_as_advanced( OPENCL_LIBRARIES )/set(OPENCL_LIBRARIES "-lOpenCL")/g' \
		-e 's/mark_as_advanced(OPENCL_LIBRARIES)/set(OPENCL_LIBRARIES "-lOpenCL")/g' "{}" \;)

# Some of the tests do not build with OpenGL 3.0. Just skip them by using make -k.
build: $(testsuite_src_dir)
	cd $(testsuite_src_dir)/samples && \
	LDFLAGS="-pthread ${LDFLAGS}" cmake -DCMAKE_BUILD_TYPE=$(build_type) -DBUILD_OPENCL=ON -DBUILD_OPENCV=OFF \
	-DBUILD_BOLT=OFF . && cmake --build . -- -k -j`getconf _NPROCESSORS_ONLN`

test_AsyncDataTransfer:
	cd $(ocl_sample_dir)/cpp_cl/1.x/AsyncDataTransfer/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./AsyncDataTransfer -e

test_AtomicCounters:
	cd $(ocl_sample_dir)/cl/1.x/AtomicCounters/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./AtomicCounters -e

test_BasicDebug:
	cd $(ocl_sample_dir)/cl/1.x/BasicDebug/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./BasicDebug -e

test_BinomialOption:
	cd $(ocl_sample_dir)/cl/1.x/BinomialOption/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./BinomialOption -e

test_BinomialOptionMultiGPU:
	cd $(ocl_sample_dir)/cl/1.x/BinomialOptionMultiGPU/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./BinomialOptionMultiGPU -e

test_BitonicSort:
	cd $(ocl_sample_dir)/cl/1.x/BitonicSort/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./BitonicSort -e

test_BlackScholes:
	cd $(ocl_sample_dir)/cl/1.x/BlackScholes/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./BlackScholes -e

test_BlackScholesDP:
	cd $(ocl_sample_dir)/cl/1.x/BlackScholesDP/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./BlackScholesDP -e

test_BoxFilter:
	cd $(ocl_sample_dir)/cl/1.x/BoxFilter/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./BoxFilter -e

test_BoxFilterGL:
	cd $(ocl_sample_dir)/cl/1.x/BoxFilterGL/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./BoxFilterGL -e

test_BufferBandwidth:
	cd $(ocl_sample_dir)/benchmark/BufferBandwidth/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./BufferBandwidth -e

test_ConcurrentKernel:
	cd $(ocl_sample_dir)/cpp_cl/1.x/ConcurrentKernel/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./ConcurrentKernel -e

test_ConstantBandwidth:
	cd $(ocl_sample_dir)/benchmark/ConstantBandwidth/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./ConstantBandwidth -e

test_CplusplusWrapper:
	cd $(ocl_sample_dir)/cpp_cl/1.x/CplusplusWrapper/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./CplusplusWrapper -e

test_DCT:
	cd $(ocl_sample_dir)/cl/1.x/DCT/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./DCT -e

test_DeviceFission:
	cd $(ocl_sample_dir)/cl/1.x/DeviceFission/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./DeviceFission -e

test_DeviceFission11Ext:
	cd $(ocl_sample_dir)/cl/1.x/DeviceFission11Ext/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./DeviceFission11Ext -e

test_DwtHaar1D:
	cd $(ocl_sample_dir)/cl/1.x/DwtHaar1D/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./DwtHaar1D -e

test_DwtHaar1DCPPKernel:
	cd $(ocl_sample_dir)/cpp_cl/1.x/DwtHaar1DCPPKernel/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./DwtHaar1DCPPKernel -e

test_Eigenvalue:
	cd $(ocl_sample_dir)/cpp_cl/1.x/Eigenvalue/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./Eigenvalue -e

test_FastWalshTransform:
	cd $(ocl_sample_dir)/cl/1.x/FastWalshTransform/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./FastWalshTransform -e

test_FFT:
	cd $(ocl_sample_dir)/cpp_cl/1.x/FFT/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./FFT -e

test_FloydWarshall:
	cd $(ocl_sample_dir)/cl/1.x/FloydWarshall/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./FloydWarshall -e

test_FluidSimulation2D:
	cd $(ocl_sample_dir)/cl/1.x/FluidSimulation/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./FluidSimulation2D -e

test_GaussianNoise:
	cd $(ocl_sample_dir)/cpp_cl/1.x/GaussianNoise/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./GaussianNoise -e --platformId 0

test_GaussianNoiseGL:
	cd $(ocl_sample_dir)/cl/1.x/GaussianNoiseGL/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./GaussianNoiseGL -e

test_GlobalMemoryBandwidth:
	cd $(ocl_sample_dir)/benchmark/GlobalMemoryBandwidth/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./GlobalMemoryBandwidth -e

test_HDRToneMapping:
	cd $(ocl_sample_dir)/cpp_cl/1.x/HDRToneMapping/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./HDRToneMapping -e --platformId 0

test_HelloWorld:
	cd $(ocl_sample_dir)/cl/1.x/HelloWorld/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./HelloWorld -e

test_Histogram:
	cd $(ocl_sample_dir)/cl/1.x/Histogram/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./Histogram -e

test_HistogramAtomics:
	cd $(ocl_sample_dir)/cl/1.x/HistogramAtomics/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./HistogramAtomics -e

test_ImageBandwidth:
	cd $(ocl_sample_dir)/cl/1.x/ImageBandwidth/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./ImageBandwidth -e

test_ImageOverlap:
	cd $(ocl_sample_dir)/cl/1.x/ImageOverlap/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./ImageOverlap -e

test_IntroStaticCPPKernel:
	cd $(ocl_sample_dir)/cpp_cl/1.x/IntroStaticCPPKernel/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./IntroStaticCPPKernel -e

test_KernelLaunch:
	cd $(ocl_sample_dir)/cl/1.x/KernelLaunch/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./KernelLaunch -e

test_KmeansAutoclustering:
	cd $(ocl_sample_dir)/cl/1.x/KmeansAutoclustering/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./KmeansAutoclustering -e

test_LDSBandwidth:
	cd $(ocl_sample_dir)/benchmark/LDSBandwidth/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./LDSBandwidth -e

test_LUDecomposition:
	cd $(ocl_sample_dir)/cl/1.x/LUDecomposition/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./LUDecomposition -e

test_Mandelbrot:
	cd $(ocl_sample_dir)/cl/1.x/Mandelbrot/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./Mandelbrot -e

test_MatrixMulDouble:
	cd $(ocl_sample_dir)/cpp_cl/1.x/MatrixMulDouble/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./MatrixMulDouble -e --platformId 0

test_MatrixMulImage:
	cd $(ocl_sample_dir)/cl/1.x/MatrixMulImage/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./MatrixMulImage -e

test_MatrixMultiplication:
	cd $(ocl_sample_dir)/cl/1.x/MatrixMultiplication/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./MatrixMultiplication -e

test_MatrixTranspose:
	cd $(ocl_sample_dir)/cl/1.x/MatrixTranspose/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./MatrixTranspose -e

test_MemoryModel:
	cd $(ocl_sample_dir)/cl/1.x/MemoryModel/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./MemoryModel -e

test_MemoryOptimizations:
	cd $(ocl_sample_dir)/benchmark/MemoryOptimizations/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./MemoryOptimizations -e

test_MersenneTwister:
	cd $(ocl_sample_dir)/cpp_cl/1.x/MerzenneTwister/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./MersenneTwister -e

test_MonteCarloAsian:
	cd $(ocl_sample_dir)/cl/1.x/MonteCarloAsian/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./MonteCarloAsian -e

test_MonteCarloAsianDP:
	cd $(ocl_sample_dir)/cl/1.x/MonteCarloAsianDP/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./MonteCarloAsianDP -e

test_MonteCarloAsianMultiGPU:
	cd $(ocl_sample_dir)/cl/1.x/MonteCarloAsianMultiGPU/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./MonteCarloAsianMultiGPU -e

test_NBody:
	cd $(ocl_sample_dir)/cl/1.x/NBody/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./NBody -e

test_PrefixSum:
	cd $(ocl_sample_dir)/cl/1.x/PrefixSum/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./PrefixSum -e

test_QuasiRandomSequence:
	cd $(ocl_sample_dir)/cl/1.x/QuasiRandomSequence/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./QuasiRandomSequence -e

test_RadixSort:
	cd $(ocl_sample_dir)/cl/1.x/RadixSort/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./RadixSort -e

test_RecursiveGaussian:
	cd $(ocl_sample_dir)/cl/1.x/RecursiveGaussian/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./RecursiveGaussian -e

test_Reduction:
	cd $(ocl_sample_dir)/cl/1.x/Reduction/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./Reduction -e

test_ScanLargeArrays:
	cd $(ocl_sample_dir)/cl/1.x/ScanLargeArrays/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./ScanLargeArrays -e

test_SimpleConvolution:
	cd $(ocl_sample_dir)/cl/1.x/SimpleConvolution/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./SimpleConvolution -e

test_SimpleGL:
	cd $(ocl_sample_dir)/cl/1.x/SimpleGL/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./SimpleGL -e

test_SimpleImage:
	cd $(ocl_sample_dir)/cl/1.x/SimpleImage/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./SimpleImage -e

test_SimpleSPIR:
	cd $(ocl_sample_dir)/cl/1.x/SimpleSPIR/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./SimpleSPIR -e

test_SoAversusAoS:
	cd $(ocl_sample_dir)/cpp_cl/1.x/SoAversusAoS/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./SoAversusAoS -e

test_SobelFilter:
	cd $(ocl_sample_dir)/cl/1.x/SobelFilter/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./SobelFilter -e

test_SobelFilterImage:
	cd $(ocl_sample_dir)/cpp_cl/1.x/SobelFilterImage/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./SobelFilterImage -e

test_StringSearch:
	cd $(ocl_sample_dir)/cl/1.x/StringSearch/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./StringSearch -e

test_Template:
	cd $(ocl_sample_dir)/cl/1.x/Template/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./Template -e

test_TransferOverlap:
	cd $(ocl_sample_dir)/cl/1.x/TransferOverlap/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./TransferOverlap -e

test_TransferOverlapCPP:
	cd $(ocl_sample_dir)/cpp_cl/1.x/TransferOverlapCPP/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./TransferOverlapCPP -e

test_UnsharpMask:
	cd $(ocl_sample_dir)/cpp_cl/1.x/UnsharpMask/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./UnsharpMask -e

test_URNG:
	cd $(ocl_sample_dir)/cl/1.x/URNG/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./URNG -e

test_URNGNoiseGL:
	cd $(ocl_sample_dir)/cpp_cl/1.x/URNGNoiseGL/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./URNGNoiseGL -e

clean-examples:
	rm -fr $(testsuite_src_dir) Install-AMD-APP.sh default-install_lnx_64.pl ReadMe.txt

# 2.0 samples

test_BinarySearchDeviceSideEnqueue:
	cd $(ocl_sample_dir)/cl/2.0/BinarySearchDeviceSideEnqueue/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./BinarySearchDeviceSideEnqueue -e

test_BufferImageInterop:
	cd $(ocl_sample_dir)/cl/2.0/BufferImageInterop/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./BufferImageInterop -e

test_BuiltInScan:
	cd $(ocl_sample_dir)/cl/2.0/BuiltInScan/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./BuiltInScan -e

test_CalcPie:
	cd $(ocl_sample_dir)/cl/2.0/CalcPie/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./CalcPie -e

test_DeviceEnqueueBFS:
	cd $(ocl_sample_dir)/cl/2.0/DeviceEnqueueBFS/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./DeviceEnqueueBFS -e

test_ExtractPrimes:
	cd $(ocl_sample_dir)/cl/2.0/ExtractPrimes/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./ExtractPrimes -e

test_FineGrainSVM:
	cd $(ocl_sample_dir)/cl/2.0/FineGrainSVM/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./FineGrainSVM -e

test_FineGrainSVMCAS:
	cd $(ocl_sample_dir)/cl/2.0/FineGrainSVMCAS/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./FineGrainSVMCAS -e

test_HeatPDE:
	cd $(ocl_sample_dir)/cl/2.0/HeatPDE/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./HeatPDE -e

test_ImageBinarization:
	cd $(ocl_sample_dir)/cl/2.0/ImageBinarization/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./ImageBinarization -e

test_PipeProducerConsumerKernels:
	cd $(ocl_sample_dir)/cl/2.0/PipeProducerConsumerKernels/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./PipeProducerConsumerKernels -e

test_RangeMinimumQuery:
	cd $(ocl_sample_dir)/cl/2.0/RangeMinimumQuery/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./RangeMinimumQuery -e

test_RecursiveGaussian_ProgramScope:
	cd $(ocl_sample_dir)/cl/2.0/RecursiveGaussian_ProgramScope/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./RecursiveGaussian_ProgramScope -e

test_RegionGrowingSegmentation:
	cd $(ocl_sample_dir)/cl/2.0/RegionGrowingSegmentation/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./RegionGrowingSegmentation -e

test_SimpleDepthImage:
	cd $(ocl_sample_dir)/cl/2.0/SimpleDepthImage/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./SimpleDepthImage -e

test_SimpleGenericAddressSpace:
	cd $(ocl_sample_dir)/cl/2.0/SimpleGenericAddressSpace/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./SimpleGenericAddressSpace -e

test_SimplePipe:
	cd $(ocl_sample_dir)/cl/2.0/SimplePipe/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./SimplePipe -e

test_SVMAtomicsBinaryTreeInsert:
	cd $(ocl_sample_dir)/cl/2.0/SVMAtomicsBinaryTreeInsert/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./SVMAtomicsBinaryTreeInsert -e

test_SVMBinaryTreeSearch:
	cd $(ocl_sample_dir)/cl/2.0/SVMBinaryTreeSearch/bin/x86_64/$(build_type)/ && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./SVMBinaryTreeSearch -e

endif
