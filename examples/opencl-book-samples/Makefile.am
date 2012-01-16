# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/examples/example1.
# 
# Copyright (c) 2012 Tampere University of Technology
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

all:
	@(if test "x${book_samples_dir}" == "x";  \
		then \
	echo Usage: make book_samples_dir=BOOK-SAMPLES-CHECKOUT;  \
	exit 1; \
	fi)
	(cd ${book_samples_dir}; mkdir -p build; cd build; \
	cmake \
	-D CMAKE_CXX_FLAGS:STRING='-g -O3' \
	-D OPENCL_LIBRARIES:FILEPATH=${abs_builddir}/lib/CL/.libs/libOpenCL.so \
	-D OPENCL_INCLUDE_DIRS:PATH=${abs_srcdir}/include .. && \
	make clean all -sk)
