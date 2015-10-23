# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/examples/VexCL
# 
# Copyright (c) 2013 Ville Korhonen / Tampere University of Technology
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

if TEST_SUITE_PIGLIT

.PHONY: build

prepare-examples:
	cd piglit ; \
	cmake \
		-DPIGLIT_BUILD_GL_TESTS=OFF \
		-DPIGLIT_BUILD_GLES1_TESTS=OFF \
		-DPIGLIT_BUILD_GLES2_TESTS=OFF \
		-DPIGLIT_BUILD_GLES3_TESTS=OFF \
		-DPIGLIT_BUILD_CL_TESTS=ON \
		-DPIGLIT_USE_WAFFLE=OFF \
		-DPIGLIT_BUILD_GLX_TESTS=OFF \
		-DOPENCL_INCLUDE_PATH="../../../include/" \
		-DOPENCL_opencl_LIBRARY:STRING="-l OpenCL" .; \
	make

clean-examples:
	cd piglit; make clean

endif

EXTRA_DIST = produce_results.sh
