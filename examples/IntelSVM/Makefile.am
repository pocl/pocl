# examples / Intel OpenCL 2.0 Shared Virtual Memory Code Sample
#
#   Copyright (c) 2015 Michal Babej / Tampere University of Technology
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

if TEST_SUITE_INTELSVM
testsuite_pocl_dir=${abs_top_srcdir}/examples/IntelSVM
testsuite_src_dir=${testsuite_pocl_dir}/source

.PHONY: build prepare-examples

prepare-examples: $(testsuite_src_dir) build

$(testsuite_src_dir):
	test -d $(testsuite_src_dir) || (mkdir $(testsuite_src_dir) && \
	cd $(testsuite_src_dir) && unzip ../intel_ocl_svm_basic_win.zip)

build: $(testsuite_src_dir) common coarse fine

common: common/basic.o common/cmdparser.o  common/oclobject.o

common/basic.o:
	cd $(testsuite_src_dir) && $(CXX) $(CFLAGS) -std=c++11 -Icommon -o common/basic.o -c common/basic.cpp

common/cmdparser.o:
	cd $(testsuite_src_dir) && $(CXX) $(CFLAGS) -std=c++11 -Icommon -o common/cmdparser.o -c common/cmdparser.cpp

common/oclobject.o:
	cd $(testsuite_src_dir) && $(CXX) $(CFLAGS) -std=c++11 -Icommon -o common/oclobject.o -c common/oclobject.cpp

coarse: common
	cd $(testsuite_src_dir) && $(CXX) $(CFLAGS) -std=c++11 -Icommon -o SVMBasicCoarseGrained/svmbasic.o -c SVMBasicCoarseGrained/svmbasic.cpp
	cd $(testsuite_src_dir) && $(CXX) $(LDFLAGS) -o SVMBasicCoarseGrained/svmbasic SVMBasicCoarseGrained/svmbasic.o \
	common/basic.o  common/cmdparser.o  common/oclobject.o -lOpenCL

fine: common
	cd $(testsuite_src_dir) && $(CXX) $(CFLAGS) -std=c++11 -Icommon -o SVMBasicFineGrained/svmbasic.o -c SVMBasicFineGrained/svmbasic.cpp
	cd $(testsuite_src_dir) && $(CXX) $(LDFLAGS) -o SVMBasicFineGrained/svmbasic SVMBasicFineGrained/svmbasic.o \
	common/basic.o  common/cmdparser.o  common/oclobject.o -lOpenCL


test_CoarseGrained:
	cd $(testsuite_src_dir)/SVMBasicCoarseGrained && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./svmbasic -p 0 -t all

test_FineGrained:
	cd $(testsuite_src_dir)/SVMBasicFineGrained && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors ./svmbasic -p 0 -t all

clean-examples:
	rm -fr $(testsuite_src_dir)

endif
