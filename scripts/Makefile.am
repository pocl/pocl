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
pkgdata_SCRIPTS = pocl-build pocl-kernel pocl-workgroup
CLEANFILES = pocl-build pocl-kernel pocl-workgroup pocl-standalone
EXTRA_DIST = pocl-build.in pocl-kernel.in pocl-workgroup.in pocl-standalone.in

do_subst = sed -e 's,[@]abs_top_srcdir[@],$(abs_top_srcdir),g'		\
               -e 's,[@]abs_top_builddir[@],$(abs_top_builddir),g'	\
               -e 's,[@]pkgincludedir[@],$(pkgincludedir),g'		\
               -e 's,[@]pkglibdir[@],$(pkglibdir),g'			\
               -e 's,[@]pkglibexecdir[@],$(pkglibexecdir),g'		\
               -e 's,[@]CLANG[@],$(CLANG),g'				\
               -e 's,[@]CLFLAGS[@],$(CLFLAGS),g'			\
               -e 's,[@]TARGET[@],$(TARGET),g'	\
               -e 's,[@]TARGET_CLANG_FLAGS[@],$(TARGET_CLANG_FLAGS),g'	\
               -e 's,[@]HOST[@],$(HOST),g'	\
               -e 's,[@]HOST_CLANG_FLAGS[@],$(HOST_CLANG_FLAGS),g'	\
               -e 's,[@]HOST_LD_FLAGS[@],$(HOST_LD_FLAGS),g'		\
               -e 's,[@]OPT[@],$(OPT),g'				\
               -e 's,[@]LLVM_LINK[@],$(LLVM_LINK),g'			\
               -e 's,[@]LLC[@],$(LLC),g' \
               -e 's,[@]LLVM_VERSION[@],$(LLVM_VERSION),g' 

pocl-build pocl-kernel pocl-workgroup pocl-standalone: %: %.in Makefile ../install-paths.h
	$(do_subst) < $< > $@
	chmod +x $@

#pocl-kernel: pocl-kernel.in Makefile
#	$(do_subst) < $< > $@
#	chmod +x $@
#
#pocl-workgroup: pocl-workgroup.in Makefile
#	$(do_subst) < $< > $@
#	chmod +x $@
#
#pocl-standalone: pocl-standalone.in Makefile
#	$(do_subst) < $< > $@
#	chmod +x $@

install-exec-hook: $(bin_SCRIPTS) 
	for f in $(bin_SCRIPTS) ; do \
		sed -e '/^# BEGIN REMOVE ONCE INSTALLED$$/,/^# END REMOVE ONCE INSTALLED$$/d' \
			> "$(DESTDIR)$(bindir)/$$f" \
			< "$$f" && \
		chmod +x "$(DESTDIR)$(bindir)/$$f" ; \
	done

install-data-hook: $(pkgdata_SCRIPTS)
	for f in $(pkgdata_SCRIPTS) ; do \
		sed -e '/^# BEGIN REMOVE ONCE INSTALLED$$/,/^# END REMOVE ONCE INSTALLED$$/d' \
			> "$(DESTDIR)$(pkgdatadir)/$$f" \
			< "$$f" && \
		chmod +x "$(DESTDIR)$(pkgdatadir)/$$f" ; \
	done
