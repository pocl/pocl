# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/examples.
# 
# Copyright (c) 2011 Universidad Rey Juan Carlos
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

SUBDIRS = example1 example1-spir32 example1-spir64 example2 example2a trig \
	scalarwave standalone opencl-book-samples VexCL ViennaCL Rodinia Parboil \
	AMD AMDSDK2.9 EinsteinToolkit piglit

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
if TEST_SUITE_PIGLIT
EXAMPLES_TO_PREPARE += "piglit "
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
	@echo Call 'make clean-examples' to clean up the examples.

