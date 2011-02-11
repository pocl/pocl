# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for llvmopencl/scripts.
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

bin_SCRIPTS = locl-build locl-kernel locl-workgroup
CLEANFILES = locl-build locl-kernel locl-workgroup
EXTRA_DIST = locl-build.in locl-kernel.in locl-workgroup.in

do_subst = sed -e 's,[@]abs_top_srcdir[@],$(abs_top_srcdir),g'		\
               -e 's,[@]abs_top_builddir[@],$(abs_top_builddir),g'	\
               -e 's,[@]pkgincludedir[@],$(pkgincludedir),g'		\
               -e 's,[@]pkglibdir[@],$(pkglibdir),g'			\
               -e 's,[@]CLANG[@],$(CLANG),g'				\
               -e 's,[@]OPT[@],$(OPT),g'				\
               -e 's,[@]LLVM_LD[@],$(LLVM_LD),g'			\
               -e 's,[@]LLC[@],$(LLC),g'				\
               -e 's,[@]SHARED[@],$(SHARED),g'

locl-build: locl-build.in
	$(do_subst) < $(srcdir)/$< > $@
	chmod +x $@

locl-kernel: locl-kernel.in
	$(do_subst) < $(srcdir)/$< > $@
	chmod +x $@

locl-workgroup: locl-workgroup.in
	$(do_subst) < $(srcdir)/$< > $@
	chmod +x $@