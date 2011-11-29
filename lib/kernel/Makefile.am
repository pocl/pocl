# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/lib/kernel.
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

LEX =

SUBDIRS = tce x86_64 arm # ppc

pkglib_LIBRARIES = libkernel.a

include sources.mk

RANLIB = `@LLVM_CONFIG@ --bindir`/llvm-ranlib
AR = `@LLVM_CONFIG@ --bindir`/llvm-ar

.cl.o:
	$(CLANG) $(AM_CPPFLAGS) $(CLANGFLAGS) -c -emit-llvm -I$(top_builddir) -include $(top_srcdir)/include/types.h -include $(top_srcdir)/include/_kernel.h -o $@ $<

.c.o:
	$(CLANG) $(AM_CPPFLAGS) $(CLANGFLAGS) -c -emit-llvm -I$(top_builddir) -include $(top_srcdir)/include/types.h -include $(top_srcdir)/include/_kernel.h -o $@ $<

.ll.o:
	$(LLVM_AS) -o $@ $<

$(libkernel_a_SOURCES:.c=.o):  $(top_srcdir)/include/_kernel.h
$(libkernel_a_SOURCES:.cl=.o):  $(top_srcdir)/include/_kernel.h templates.h
