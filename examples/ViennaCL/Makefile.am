# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/examples/ViennaCL.
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

testsuite_pocl_dir=${abs_top_srcdir}/examples/ViennaCL
testsuite_src_dir=${testsuite_pocl_dir}/ViennaCL-1.2.0-src

.PHONY: build

if TEST_SUITE_VIENNACL
all: $(testsuite_src_dir)
endif

$(srcdir)/ViennaCL-1.2.0-src.tar.gz:
	cd $(srcdir) && \
	wget http://sourceforge.net/projects/viennacl/files/1.2.x/ViennaCL-1.2.0-src.tar.gz

$(testsuite_src_dir):
if TEST_SUITE_VIENNACL
	$(MAKE) $(srcdir)/ViennaCL-1.2.0-src.tar.gz
	cd $(testsuite_pocl_dir) && \
	tar xvzf ViennaCL-1.2.0-src.tar.gz
else
	@echo 'use --enable-test-viennacl in configure options'
	@echo 'or get and extract sources yourself'
	@echo 'Aborting'
	@false
endif

# ViennaCL must be built in the subdirectory build/ of the extracted source
# (it does not work elsewhere) and its source tree is modified.
# So, copying the source tree in our build tree if required
# (ie if $(srcdir) is not $(CURDIR) )
build:
	if [ ! -d ViennaCL-1.2.0-src ]; then cp -aL $(testsuite_src_dir) . ; fi
	cd ViennaCL-1.2.0-src/build; \
 	cmake \
		-D BUILD_DOXYGEN_DOCS:BOOL=OFF \
		-D BUILD_MANUAL:BOOL=OFF \
		-D CMAKE_CXX_FLAGS:STRING='-g -O0 -DCL_USE_DEPRECATED_OPENCL_1_1_APIS' \
		$(OPENCL_CMAKE) \
		..

clean-local:
	rm -fr ViennaCL-1.2.0-src

if TEST_SUITE_VIENNACL
distclean-local: clean-local
	$(RM) -r $(testsuite_src_dir)
	$(RM) $(testsuite_pocl_dir)/ViennaCL-1.2.0-src.tar.gz
endif
