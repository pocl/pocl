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

SUBDIRS = include lib

if !POCL_ANDROID
SUBDIRS += bin examples tests
endif

ACLOCAL_AMFLAGS = -I m4

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = pocl.pc

CLEANFILES = install-paths.h
PHONIES = install-paths.h

if BUILD_ICD
icddir = $(sysconfdir)/OpenCL/vendors/
icd_DATA = pocl.icd
do_subst = sed -e 's,[@]libdir[@],$(libdir),g'

#TODO: this naming scheme is used on other platforms (primarly BSDs?) too
if FREEBSD_HOST_OS
do_subst += -e 's,so.VER,so.$(LIB_CURRENT_VERSION),g'
else
do_subst += -e 's,so.VER,so.$(LIB_FIRST_VERSION).$(LIB_AGE_VERSION).$(LIB_REVISION_VERSION),g'
endif

pocl.icd:
	$(do_subst) < $(srcdir)/pocl.icd.in > $(builddir)/pocl.icd

CLEANFILES += pocl.icd
PHONIES += pocl.icd

endif

.PHONY: ${PHONIES} prepare-examples clean-examples

EXTRA_DIST = config/xclang tools/data/test_machine.adf tools/data/test_machine_fp16.adf \
  doc/build-envs.txt CHANGES fix-include pocl.icd.in README README.ARM README.mips \
  README.Windows README.OSX README.powerpc README.packaging CREDITS LICENSE \
  tools/uncrustify_cxx.cfg windows/setup_and_build_win64.sh README.mipsel

# CMake
EXTRA_DIST += CMakeLists.txt cmake/FindHwloc.cmake cmake/FindPthreadsWin32.cmake \
  cmake/LLVM.cmake cmake/Sphinx.cmake cmake/bitcode_rules.cmake cmake/run_test.cmake \
	android/androideabi.cmake config.h.in.cmake pocl.icd.in.cmake pocl.pc.in.cmake \
	cmake/kernellib_hash.cmake

# Sphinx dist:
EXTRA_DIST += doc/sphinx/source/conf.py doc/sphinx/source/design.rst doc/sphinx/source/development.rst doc/sphinx/source/env_variables.rst doc/sphinx/source/faq.rst doc/sphinx/source/features.rst doc/sphinx/source/host_library.rst doc/sphinx/source/hsa.rst doc/sphinx/source/index.rst doc/sphinx/source/kernel_compiler.rst doc/sphinx/source/memory_management.rst doc/sphinx/source/releasing.rst doc/sphinx/source/using.rst doc/sphinx/Makefile

# Tools, patches etc
EXTRA_DIST += tools/patches/khronos-icd-loader.patch tools/patches/clang-3.7-hsail-branch.patch tools/patches/llvm-3.7-hsail-branch.patch tools/scripts/devel-envs.sh tools/scripts/run_hsa_tests


BUILT_SOURCES = install-paths.h

prepare-examples:
	$(MAKE) -C examples prepare-examples

clean-examples:
	$(MAKE) -C examples clean-examples

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
