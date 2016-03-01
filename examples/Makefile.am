# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/examples.
# 
# Copyright (c) 2011 Universidad Rey Juan Carlos
#               2011-2014 Pekka Jääskeläinen / Tampere University of Tech.
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

# The examples under test suite are both internal ("smoke test") to pocl
# and benchmarks/test suites that are from external SDKs. The build
# systems of the latter should not be invoked every time 'make all' or
# 'make check' is executed because they are not affected by pocl changes nor
# should itself change.

SUBDIRS = example1 example1-spir32 example1-spir64 example2 example2a trig \
	scalarwave poclcc opencl-book-samples VexCL ViennaCL Rodinia Parboil \
	AMD AMDSDK2.9 AMDSDK3.0 EinsteinToolkit piglit Halide OpenCV CloverLeaf IntelSVM

BASIC_EXAMPLES = example1 example1-spir32 example1-spir64 example2 example2a trig \
	scalarwave poclcc opencl-book-samples EinsteinToolkit

EXTRA_DIST = CMakeLists.txt PyOpenCL/README

# The external examples which should be prepared only
# once by calling "make prepare-examples".
EXAMPLES_TO_PREPARE = 

if TEST_SUITE_VEXCL
EXAMPLES_TO_PREPARE += "VexCL "
endif
if TEST_SUITE_VIENNACL
EXAMPLES_TO_PREPARE += "ViennaCL "
endif
if TEST_SUITE_RODINIA
EXAMPLES_TO_PREPARE += "Rodinia "
endif
if TEST_SUITE_PARBOIL
EXAMPLES_TO_PREPARE += "Parboil "
endif
if TEST_SUITE_AMD
EXAMPLES_TO_PREPARE += "AMD "
endif
if TEST_SUITE_AMDSDK2_9
EXAMPLES_TO_PREPARE += "AMDSDK2.9 "
endif
if TEST_SUITE_AMDSDK3_0
EXAMPLES_TO_PREPARE += "AMDSDK3.0 "
endif
if TEST_SUITE_PIGLIT
EXAMPLES_TO_PREPARE += "piglit "
endif
if TEST_SUITE_HALIDE
EXAMPLES_TO_PREPARE += "Halide "
endif
if TEST_SUITE_CLOVERLEAF
EXAMPLES_TO_PREPARE += "CloverLeaf "
endif
if TEST_SUITE_INTELSVM
EXAMPLES_TO_PREPARE += "IntelSVM "
endif
if TEST_SUITE_OPENCV
EXAMPLES_TO_PREPARE += "OpenCV "
endif

prepare-examples:
	for dir in $(EXAMPLES_TO_PREPARE); do \
	   $(MAKE) -C $$dir prepare-examples; \
	done

clean-examples:
	for dir in $(EXAMPLES_TO_PREPARE); do \
	   $(MAKE) -C $$dir clean-examples; \
	done

clean:
	for dir in $(BASIC_EXAMPLES); do \
	   $(MAKE) -C $$dir clean; \
	done
	@echo Call 'make clean-examples' to clean up the examples.

