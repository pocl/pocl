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

SUBDIRS = @OCL_TARGETS@

# The source files in this dir are not built directly
# but are used with the target specific subdirs as
# fallback implementations. Include them in the
# dist once here.
EXTRA_DIST = $(srcdir)/*.c $(srcdir)/*.cl $(srcdir)/*.h $(srcdir)/*.ll			\
	$(srcdir)/vecmathlib/LICENCE			\
	$(srcdir)/vecmathlib/README			\
	$(srcdir)/vecmathlib/*.h				\
	$(srcdir)/vecmathlib-pocl/*.cc			\
	$(srcdir)/vecmathlib-pocl/*.cl			\
	$(srcdir)/vecmathlib-pocl/*.h			\
	$(srcdir)/vecmathlib-pocl/*.py  \
	$(srcdir)/cellspu/* \
	$(srcdir)/tce/* \
	CMakeLists.txt


# Distclean the dirs regardless if the target was enabled or not
# to cleanup the Makefiles.
distclean-local:
	for dir in host tce cellspu; do \
		make -C $$dir distclean;\
	done;
