# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/examples/AMD.
# 
# Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
# Copyright (c) 2012 Vincent Danjean <Vincent.Danjean@ens-lyon.org>
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

AMD-APP-SDK-v2.8-RC-lnx64.tgz:
	cd $(srcdir) && \
	wget http://developer.amd.com.php53-23.ord1-1.websitetestlink.com/wordpress/media/2012/11/AMD-APP-SDK-v2.8-lnx64.tgz && \
	tar xzf AMD-APP-SDK-v2.8-lnx64.tgz

if TEST_SUITE_AMD
testsuite_pocl_dir=${abs_top_srcdir}/examples/AMD
testsuite_src_dir=${testsuite_pocl_dir}/AMD-APP-SDK-v2.8-RC-lnx64
amd_app_dir=$(testsuite_src_dir)/samples/opencl/cl/app

.PHONY: build

all: $(testsuite_src_dir) 

$(testsuite_src_dir): AMD-APP-SDK-v2.8-RC-lnx64.tgz
	test -d $(testsuite_src_dir) || tar xzf AMD-APP-SDK-v2.8-RC-lnx64.tgz

build:
	cd $(testsuite_src_dir)/samples/opencl && \
	$(MAKE) -s 2>&1 | grep -v "No LSB modules are available" || true

test_MatrixMultiplication: build
	cd $(amd_app_dir)/MatrixMultiplication && \
	ln -sf $(amd_app_dir)/MatrixMultiplication/MatrixMultiplication_Kernels.cl $(amd_app_dir)/MatrixMultiplication/build/debug/x86_64/MatrixMultiplication_Kernels.cl && \
	OCL_ICD_VENDORS=$(abs_top_builddir)/ocl-vendors build/debug/x86_64/MatrixMultiplication -e

test_all: test_MatrixMultiplication

clean-local:
	rm -fr $(testsuite_src_dir) Install-AMD-APP.sh default-install_lnx_64.pl ReadMe.txt

distclean-local: clean-local
	$(RM) -r $(testsuite_src_dir)

endif
