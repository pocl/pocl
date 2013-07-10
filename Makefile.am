# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile for the top-level directory of pocl.
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

SUBDIRS = include lib scripts examples tests

ACLOCAL_AMFLAGS = -I m4

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = pocl.pc

CLEANFILES = install-paths.h
PHONIES = install-paths.h

if BUILD_ICD
icddir = $(sysconfdir)/OpenCL/vendors/
icd_DATA = pocl.icd
do_subst = sed -e 's,[@]libdir[@],$(libdir),g'

if FREEBSD_HOST_OS
do_subst += -e 's,so.VER,so.2,g'
else
do_subst += -e 's,so.VER,so.1.1.0,g'
endif

pocl.icd:
	$(do_subst) < $(srcdir)/pocl.icd.in > $(builddir)/pocl.icd

CLEANFILES += pocl.icd
PHONIES += pocl.icd

endif

.PHONY: ${PHONIES}

EXTRA_DIST = config/xclang doc/envs.txt tools/data/*.adf \
  CHANGES fix-include pocl.icd.in README README.ARM README.Cell \
  README.OSX README.ppc64

BUILT_SOURCES = install-paths.h

# Always rebuild install-paths.h as paths can be changed at each make invocation
.PHONY: install-paths.h
install-paths.h:
	echo "#define PKGDATADIR \"${pkgdatadir}\"" > $@.new
	echo "#define PKGLIBEXECDIR \"${pkglibexecdir}\"" >> $@.new
	echo "#define PKGINCLUDEDIR \"${pkgincludedir}\"" >> $@.new
	@if cmp --quiet $@.new $@ ; then \
		: "Paths not changed" ;\
		$(RM) $@.new ;\
	else \
		echo "New paths, updating $@" ;\
		mv $@.new $@ ;\
	fi


