# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/tests.
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

# The `:;' works around a Bash 3.2 bug when the output is not writeable.
$(srcdir)/package.m4: $(top_srcdir)/configure.ac
	:;{ \
	  echo '# Signature of the current package.' && \
	  echo 'm4_define([AT_PACKAGE_NAME],' && \
	  echo '  [$(PACKAGE_NAME)])' && \
	  echo 'm4_define([AT_PACKAGE_TARNAME],' && \
	  echo '  [$(PACKAGE_TARNAME)])' && \
	  echo 'm4_define([AT_PACKAGE_VERSION],' && \
	  echo '  [$(PACKAGE_VERSION)])' && \
	  echo 'm4_define([AT_PACKAGE_STRING],' && \
	  echo '  [$(PACKAGE_STRING)])' && \
	  echo 'm4_define([AT_PACKAGE_BUGREPORT],' && \
	  echo '  [$(PACKAGE_BUGREPORT)])'; \
	  echo 'm4_define([AT_PACKAGE_URL],' && \
	  echo '  [$(PACKAGE_URL)])'; \
	} >'$(srcdir)/package.m4'

EXTRA_DIST = testsuite.at $(srcdir)/package.m4 $(TESTSUITE) atlocal.in
TESTSUITE = $(srcdir)/testsuite

check-local: atconfig atlocal $(TESTSUITE)
	$(SHELL) '$(TESTSUITE)' $(TESTSUITEFLAGS)

installcheck-local: atconfig atlocal $(TESTSUITE)
	$(SHELL) '$(TESTSUITE)' AUTOTEST_PATH='$(bindir)' \
	  $(TESTSUITEFLAGS)

clean-local:
	test ! -f '$(TESTSUITE)' || \
	 $(SHELL) '$(TESTSUITE)' --clean

AUTOM4TE = $(SHELL) $(top_srcdir)/config/missing --run autom4te
AUTOTEST = $(AUTOM4TE) --language=autotest
$(TESTSUITE): $(srcdir)/testsuite.at $(srcdir)/package.m4
	$(AUTOTEST) -I '$(srcdir)' -o $@.tmp $@.at
	mv $@.tmp $@

GRAPH_TEST_OUTPUTS = nobarrier1_workitem.dot					\
                     barrier1_barriers.dot					\
                     barrier1_barriers_workitem.dot				\
                     ifbarrier1_barriers.dot					\
                     ifbarrier1_barriers_btr.dot				\
                     ifbarrier1_barriers_btr_workitem.dot			\
                     ifbarrier2_btr.dot						\
                     ifbarrier3_btr.dot						\
                     ifbarrier4_btr.dot						\
                     forbarrier1_barriers.dot					\
                     forbarrier1_barriers_btr.dot				\
                     forbarrier1_barriers_btr_workitem.dot			\
                     ifforbarrier1_btr.dot					\
                     forifbarrier1_btr.dot

GRAPH_TEST_INPUTS = $(addsuffix .dot,					\
                    $(subst .,_, $(basename				\
                    $(subst _,., $(basename $(GRAPH_TEST_OUTPUTS))))))

GRAPH_TEST_SOURCES = $(GRAPH_TEST_INPUTS:.dot=.ll)

EXTRA_DIST += $(GRAPH_TEST_OUTPUTS) $(GRAPH_TEST_INPUTS) $(GRAPH_TEST_SOURCES)

sources: $(addprefix $(srcdir)/, $(GRAPH_TEST_SOURCES))

dots: $(addprefix $(srcdir)/, $(GRAPH_TEST_OUTPUTS) $(GRAPH_TEST_INPUTS) $(GRAPH_TEST_SOURCES))

%_workitem.ll: %.ll
	@OPT@ -load=@abs_top_builddir@/lib/llvmopencl/.libs/llvmopencl.so -workitem -local-size=2 2 2 -S -o $@ $<

%_loops.ll: %.ll
	@OPT@ -loop-simplify -S -o $@ $<

%_barriers.ll: %.ll
	@OPT@ -load=@abs_top_builddir@/lib/llvmopencl/.libs/llvmopencl.so -barriers -S -o $@ $<

%_btr.ll: %.ll
	@OPT@ -load=@abs_top_builddir@/lib/llvmopencl/.libs/llvmopencl.so -barriertails -S -o $@ $<

%.dot: %.ll
	@OPT@ -disable-output -dot-cfg-only $<
	mv cfg.*.dot $@
