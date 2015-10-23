# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/examples/example1.
# 
# Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
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

if TEST_SUITE_SAMPLES
book_samples_base_dir=${abs_top_srcdir}/examples/opencl-book-samples
book_samples_dir=${book_samples_base_dir}/checkout

.PHONY: build run_all run_working check_dir run_HelloWorld run_OpenCLInfo run_OpenCLConvolutionChap3
.PHONY: run_SimpleBufferSubBuffer run_spmv run_VectorAdd run_Dijkstra

all: $(book_samples_base_dir)/stamp-patched

$(book_samples_dir):
	cd $(book_samples_base_dir) && \
	svn checkout http://opencl-book-samples.googlecode.com/svn/trunk/ checkout

$(book_samples_base_dir)/stamp-patched: $(book_samples_dir)
	(cd ${book_samples_dir}; \
	sed -i 's/bool doCPU = false/bool doCPU = true/g' src/Chapter_16/Dijkstra/oclDijkstra.cpp; \
	sed -i 's/size_t localWorkSize = maxWorkGroupSize/size_t localWorkSize = 2/g' src/Chapter_16/Dijkstra/oclDijkstraKernel.cpp; \
	sed -i 's/device.j..type == CL_DEVICE_TYPE_GPU/device[j].type \& CL_DEVICE_TYPE_GPU/g' src/Chapter_22/spmv.c; \
	sed -i 's/context.CL_DEVICE_TYPE_GPU/context\(CL_DEVICE_TYPE_CPU/g' src/Chapter_12/VectorAdd/vecadd.cpp; \
	rm -f src/Chapter_12/Sinewave/CMakeLists.txt)
	touch $@

build: $(book_samples_base_dir)/stamp-patched

build: check_dir
	mkdir -p build; cd build; \
	cmake \
		-D CMAKE_CXX_FLAGS:STRING='-g -O3 -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -lGL -lglut ${PTHREAD_CFLAGS}' \
		$(OPENCL_CMAKE) \
		${book_samples_dir}
	$(MAKE) clean -sk -C build
	$(MAKE) all -sk -C build

run_HelloWorld: 
	@(cd build/src/Chapter_2/HelloWorld; ./HelloWorld)

run_OpenCLInfo: 
	@echo
	@echo "###### Chapter 3: OpenCLInfo"
	@echo
	@(cd build/src/Chapter_3/OpenCLInfo; ./OpenCLInfo)

run_OpenCLConvolutionChap3: 
	@(cd build/src/Chapter_3/OpenCLConvolution; \
	ln -sf ${book_samples_dir}/src/Chapter_3/OpenCLConvolution/Convolution.cl; \
	./OpenCLConvolutionChap3)

run_HelloBinaryWorld: 
	@(cd build/src/Chapter_6/HelloBinaryWorld; \
	rm -f HelloWorld.cl.bin; \
	./HelloBinaryWorld; \
	./HelloBinaryWorld)

run_SimpleBufferSubBuffer: 
	@(cd build/src/Chapter_7/SimpleBufferSubBuffer; \
	./SimpleBufferSubBuffer --useMap)

run_VectorAdd: 
	@(cd build/src/Chapter_12/VectorAdd; \
	./vecadd)

# This failed because --cpu is not parsed correctly
# and it tries to run the --cpugpu case. We
# force 'doCPU = true' in oclDijkstra.cpp to make
# it run. This case uses the maximum work group size
# thus the llc takes a lot of memory and time.
# The latter sed patch forces the localWorkSize to 2 to make it feasible for now.
run_Dijkstra: 
	@echo
	@echo "###### Chapter 16: Dijkstra"
	@echo
	@(cd build/src/Chapter_16/Dijkstra; \
	./Dijkstra --cpu)

run_spmv: 
	@echo
	@echo "###### Chapter 22: spmv"
	@echo
	@(cd build/src/Chapter_22; \
	ln -sf ${book_samples_dir}/src/Chapter_22/spmv.cl; \
	./spmv)

run_working: run_HelloWorld run_OpenCLConvolutionChap3 run_HelloBinaryWorld run_SimpleBufferSubBuffer run_VectorAdd

run_all: run_working run_OpenCLInfo run_Dijkstra run_spmv

clean-local:
	rm -fr build

distclean-local: clean-local
	$(RM) -r $(book_samples_dir)
	$(RM) $(book_samples_base_dir)/stamp-patched
endif


EXTRA_DIST = HelloBinaryWorld.stdout OpenCLConvolutionChap3.stdout ImageFilter2D.cl \
             VectorAdd.stdout HelloWorld.stdout SimpleBufferSubBuffer.stdout
