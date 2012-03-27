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

SUBDIRS = regression

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

EXTRA_DIST = testsuite.at testsuite-samples.at testsuite-viennacl.at testsuite-regression.at $(srcdir)/package.m4 $(TESTSUITE) atlocal.in
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
$(TESTSUITE): $(srcdir)/testsuite.at $(srcdir)/testsuite-samples.at $(srcdir)/testsuite-regression.at $(srcdir)/testsuite-viennacl.at $(srcdir)/testsuite-rodinia.at $(srcdir)/package.m4
	$(AUTOTEST) -I '$(srcdir)' -o $@.tmp $@.at
	mv $@.tmp $@

LOOPBARRIERS_TESTS_INPUTS = loopbarriers1_input.ll	\
                            loopbarriers2_input.ll
LOOPBARRIERS_TESTS_GOLDEN = $(subst input,golden,		\
                            $(LOOPBARRIERS_TESTS_INPUTS))

EXTRA_DIST += $(LOOPBARRIERS_TESTS_INPUTS) $(LOOPBARRIERS_TESTS_OUTPUTS)

loopbarriers_golden: $(addprefix $(srcdir)/, $(LOOPBARRIERS_TESTS_GOLDEN))

loopbarriers%_golden.ll: loopbarriers%_input.ll
	cd $(dir $@) && @OPT@ -load=@abs_top_builddir@/lib/llvmopencl/.libs/llvmopencl.so -loop-barriers -S -o $(notdir $@) $(notdir $<)

BARRIERS_TESTS_INPUTS = barriers1_input.ll	\
                        barriers2_input.ll
BARRIERS_TESTS_GOLDEN = $(subst input,golden,		\
                        $(BARRIERS_TESTS_INPUTS))

EXTRA_DIST += $(BARRIERS_TESTS_INPUTS) $(BARRIERS_TESTS_OUTPUTS)

barriers_golden: $(addprefix $(srcdir)/, $(BARRIERS_TESTS_GOLDEN))

barriers%_golden.ll: barriers%_input.ll
	cd $(dir $@) && @OPT@ -load=@abs_top_builddir@/lib/llvmopencl/.libs/llvmopencl.so -barriers -S -o $(notdir $@) $(notdir $<)

BARRIERTAILS_TESTS_INPUTS = barriertails1_input.ll	\
                            barriertails2_input.ll
BARRIERTAILS_TESTS_GOLDEN = $(subst input,golden,		\
                            $(BARRIERTAILS_TESTS_INPUTS))

EXTRA_DIST += $(BARRIERTAILS_TESTS_INPUTS) $(BARRIERTAILS_TESTS_OUTPUTS)

barriertails_golden: $(addprefix $(srcdir)/, $(BARRIERTAILS_TESTS_GOLDEN))

barriertails%_golden.ll: barriertails%_input.ll
	cd $(dir $@) && @OPT@ -load=@abs_top_builddir@/lib/llvmopencl/.libs/llvmopencl.so -barriertails -S -o $(notdir $@) $(notdir $<)

COMPOSITE_TEST_GOLDEN = nobarrier1_workitem.ll				\
                        barrier1_barriers.ll				\
                        barrier1_barriers_workitem.ll			\
                        ifbarrier1_btr.ll				\
                        ifbarrier1_btr_barriers.ll			\
                        ifbarrier1_btr_barriers_workitem.ll		\
                        ifbarrier2_btr.ll				\
                        ifbarrier3_btr.ll				\
                        ifbarrier4_btr.ll				\
                        forbarrier1_loops.ll				\
                        forbarrier1_loops_btr.ll			\
                        forbarrier1_loops_btr_barriers.ll		\
                        forbarrier1_loops_btr_barriers_workitem.ll	\
                        forbarrier2_loops.ll				\
                        forbarrier2_loops_btr.ll			\
                        forbarrier2_loops_btr_barriers.ll		\
                        forbarrier2_loops_btr_barriers_workitem.ll	\
                        ifforbarrier1_loops.ll                          \
                        ifforbarrier1_loops_btr.ll                      \
                        ifforbarrier1_loops_btr_barriers.ll             \
                        ifforbarrier1_loops_btr_barriers_workitem.ll    \
                        forifbarrier1_loops.ll                          \
                        forifbarrier1_loops_btr.ll                      \
                        forifbarrier1_loops_btr_barriers.ll             \
                        forifbarrier1_loops_btr_barriers_workitem.ll

COMPOSITE_TEST_INPUTS = $(addsuffix .ll,					\
                        $(subst .,_, $(basename					\
                        $(subst _,., $(basename $(COMPOSITE_TEST_OUTPUTS))))))

EXTRA_DIST += $(COMPOSITE_TEST_GOLDEN) $(COMPOSITE_TEST_INPUTS)

composite_golden: $(addprefix $(srcdir)/, $(COMPOSITE_TEST_GOLDEN))

%_workitem.ll: %.ll
	cd $(dir $@) && @OPT@ -load=@abs_top_builddir@/lib/llvmopencl/.libs/llvmopencl.so -verify-dom-info -verify-loop-info -workitem -local-size=2 2 2 -S -o $(notdir $@) $(notdir $<)

%_barriers.ll: %.ll
	cd $(dir $@) && @OPT@ -load=@abs_top_builddir@/lib/llvmopencl/.libs/llvmopencl.so -verify-dom-info -verify-loop-info -barriers -S -o $(notdir $@) $(notdir $<)

%_btr.ll: %.ll
	cd $(dir $@) && @OPT@ -load=@abs_top_builddir@/lib/llvmopencl/.libs/llvmopencl.so -verify-dom-info -verify-loop-info -barriertails -S -o $(notdir $@) $(notdir $<)

%_loops.ll: %.ll
	cd $(dir $@) && @OPT@ -load=@abs_top_builddir@/lib/llvmopencl/.libs/llvmopencl.so -verify-dom-info -verify-loop-info -loop-simplify -loop-barriers -S -o $(notdir $@) $(notdir $<)
