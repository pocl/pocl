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

# Usage:
#
# mkdir -p ~/src && cd ~/src  (or wherever you want) 
# svn checkout http://opencl-book-samples.googlecode.com/svn/trunk/ opencl-book-samples-read-only
# make book_samples_dir=~/src/opencl-book-samples-read-only

abs_builddir=${PWD}/$(top_builddir)
abs_srcdir=${PWD}/$(top_srcdir)

.PHONY: build run_all run_working check_dir run_HelloWorld run_OpenCLInfo run_OpenCLConvolutionChap3
.PHONY: run_SimpleBufferSubBuffer run_spmv

build: check_dir
	(cd ${book_samples_dir}; mkdir -p build; cd build; \
	cmake \
	-D CMAKE_CXX_FLAGS:STRING='-g -O3 -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -lGL -lglut' \
	-D OPENCL_LIBRARIES:FILEPATH=${abs_builddir}/lib/CL/.libs/libOpenCL.so \
	-D OPENCL_INCLUDE_DIRS:PATH=${abs_srcdir}/include .. && \
	make clean all -sk)

check_dir: 
	@(if test "x${book_samples_dir}" == "x";  \
		then \
	echo Usage: make book_samples_dir=BOOK-SAMPLES-CHECKOUT;  \
	exit 1; \
	fi)

run_HelloWorld: check_dir
	@echo
	@echo "###### Chapter 2: Hello World"
	@echo
	@(cd ${book_samples_dir}; cd build/src/Chapter_2/HelloWorld; ./HelloWorld)

run_OpenCLInfo: check_dir
	@echo
	@echo "###### Chapter 3: OpenCLInfo"
	@echo
	@(cd ${book_samples_dir}; cd build/src/Chapter_3/OpenCLInfo; ./OpenCLInfo)

run_OpenCLConvolutionChap3: check_dir
	@echo
	@echo "###### Chapter 3: OpenCLConvolution"
	@echo
	@(cd ${book_samples_dir}; \
	cd build/src/Chapter_3/OpenCLConvolution; \
	ln -sf ../../../../src/Chapter_3/OpenCLConvolution/Convolution.cl; \
	./OpenCLConvolutionChap3)

run_HelloBinaryWorld: check_dir
	@echo
	@echo "###### Chapter 6: HelloBinaryWorld"
	@echo
	@(cd ${book_samples_dir}; \
	cd build/src/Chapter_6/HelloBinaryWorld; \
	rm -f HelloWorld.cl.bin; \
	./HelloBinaryWorld; \
	./HelloBinaryWorld)

run_SimpleBufferSubBuffer: check_dir
	@echo
	@echo "###### Chapter 7: SimpleBufferSubBuffer"
	@echo
	@(cd ${book_samples_dir}; \
	cd build/src/Chapter_7/SimpleBufferSubBuffer; \
	./SimpleBufferSubBuffer --useMap)

# This failed because --cpu is not parsed correctly
# and it tries to run the --cpugpu case. We
# force 'doCPU = true' in oclDijkstra.cpp to make
# it run. This case uses the maximum work group size
# thus the llc takes a lot of memory and time.
run_Dijkstra: check_dir
	@echo
	@echo "###### Chapter 16: Dijkstra"
	@echo
	@(cd ${book_samples_dir}; \
	sed -i 's/bool doCPU = false/bool doCPU = true/g' src/Chapter_16/Dijkstra/oclDijkstra.cpp; \
	cd build/src/Chapter_16/Dijkstra; \
	./Dijkstra --cpu)

run_spmv: check_dir
	@echo
	@echo "###### Chapter 22: spmv"
	@echo
	@(cd ${book_samples_dir}; \
	cd build/src/Chapter_22; \
	ln -sf ../../../src/Chapter_22/spmv.cl; \
	./spmv)

run_working: run_HelloWorld run_OpenCLConvolutionChap3 run_HelloBinaryWorld run_SimpleBufferSubBuffer

run_all: run_working run_OpenCLInfo run_Dijkstra run_spmv


