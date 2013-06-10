# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/include/CL.
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

#force the patching of cl.hpp.in - 
BUILT_SOURCES = cl.hpp
CLEANFILES = ${BUILT_SOURCES}
#note - this gets run on every build, so don't fail if patch detects a
#previously applied patch. The copy is a workaround for OSX's 'patch'
cl.hpp: $(top_srcdir)/include/CL/cl.hpp.in $(top_srcdir)/tools/patches/khronos_cl.hpp.patch
	cp $< __tmp_header && \
	(patch -sNp2 __tmp_header $(top_srcdir)/tools/patches/khronos_cl.hpp.patch -o cl.hpp || true) && \
	rm __tmp_header

if INSTALL_OPENCL_HEADERS
library_includedir = $(includedir)/CL
library_include_HEADERS = cl.h cl_ext.h			\
                          cl_gl.h cl_gl_ext.h		\
                          cl_platform.h opencl.h
nodist_library_include_HEADERS = cl.hpp
endif

EXTRA_DIST = cl.hpp.in $(top_srcdir)/tools/patches/khronos_cl.hpp.patch
