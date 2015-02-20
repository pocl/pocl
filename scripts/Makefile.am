# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/scripts.
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

bin_SCRIPTS = pocl-standalone
pkgdataincludedir = $(pkgdatadir)/include

CLEANFILES = pocl-standalone
EXTRA_DIST = pocl-standalone.in CMakeLists.txt

do_subst = sed -e 's|[@]abs_top_srcdir[@]|$(abs_top_srcdir)|g'		\
               -e 's|[@]abs_top_builddir[@]|$(abs_top_builddir)|g'	\
               -e 's|[@]pkgdataincludedir[@]|$(pkgdataincludedir)|g'		\
               -e 's|[@]pkglibdir[@]|$(pkglibdir)|g'			\
               -e 's|[@]pkgdatadir[@]|$(pkgdatadir)|g'			\
               -e 's|[@]pkglibexecdir[@]|$(pkglibexecdir)|g'		\
               -e 's|[@]CLANG[@]|$(CLANG)|g'				\
               -e 's|[@]CLFLAGS[@]|$(CLFLAGS)|g'			\
               -e 's|[@]FORCED_CLFLAGS[@]|$(FORCED_CLFLAGS)|g'		\
               -e 's|[@]LD_SHARED[@]|$(LD_SHARED)|g'			\
               -e 's|[@]OCL_KERNEL_TARGET[@]|$(OCL_KERNEL_TARGET)|g'	\
               -e 's|[@]TARGET[@]|$(TARGET)|g'				\
               -e 's|[@]TARGET_CLANG_FLAGS[@]|$(TARGET_CLANG_FLAGS)|g'	\
               -e 's|[@]TARGET_LLC_FLAGS[@]|$(TARGET_LLC_FLAGS)|g'	\
               -e 's|[@]TARGET_LD_FLAGS[@]|$(TARGET_LD_FLAGS)|g'	\
               -e 's|[@]HOST[@]|$(HOST)|g'				\
               -e 's|[@]HOST_CLANG_FLAGS[@]|$(HOST_CLANG_FLAGS)|g'	\
               -e 's|[@]HOST_LLC_FLAGS[@]|$(HOST_LLC_FLAGS)|g'		\
               -e 's|[@]HOST_LD_FLAGS[@]|$(HOST_LD_FLAGS)|g'		\
               -e 's|[@]OPT[@]|$(OPT)|g'				\
               -e 's|[@]LLVM_LINK[@]|$(LLVM_LINK)|g'			\
               -e 's|[@]LLC[@]|$(LLC)|g' 				\
               -e 's|[@]LLVM_VERSION[@]|$(LLVM_VERSION)|g' 

pocl-standalone: %: %.in Makefile ../install-paths.h
	$(do_subst) < $< > $@
	chmod +x $@

install-exec-hook: $(bin_SCRIPTS) 
	for f in $(bin_SCRIPTS) ; do \
		sed -e '/^# BEGIN REMOVE ONCE INSTALLED$$/,/^# END REMOVE ONCE INSTALLED$$/d' \
			> "$(DESTDIR)$(bindir)/$$f" \
			< "$$f" && \
		chmod +x "$(DESTDIR)$(bindir)/$$f" ; \
	done
