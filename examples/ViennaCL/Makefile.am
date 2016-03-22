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

if TEST_SUITE_VIENNACL
# VIENNACL_VERSION comes from configure.ac
viennacl_local_src_dir=ViennaCL-1.5.1
viennacl_tarball=ViennaCL-1.5.1.tar.gz
testsuite_pocl_dir=${abs_top_srcdir}/examples/ViennaCL
testsuite_src_dir=${testsuite_pocl_dir}/${viennacl_local_src_dir}

.PHONY: prepare-examples

$(srcdir)/$(viennacl_tarball):
	cd $(srcdir) && \
	wget http://sourceforge.net/projects/viennacl/files/1.5.x/$(viennacl_tarball)

$(viennacl_local_src_dir):
	$(MAKE) $(srcdir)/$(viennacl_tarball) 
	cd $(testsuite_pocl_dir) && \
	tar xzf $(viennacl_tarball)

# ViennaCL must be built in the subdirectory build/ of the extracted source
# (it does not work elsewhere) and its source tree is modified.
# So, copying the source tree in our build tree if required
# (ie if $(srcdir) is not $(CURDIR) )
build:
	if [ ! -d $(viennacl_local_src_dir) ]; then cp -aL $(testsuite_src_dir) . ; fi
	cd $(viennacl_local_src_dir)/build; \
	cmake \
		-D OPENCL_LIBRARIES="-lOpenCL" \
		-D BUILD_TESTING:BOOL=ON \
		-D BUILD_DOXYGEN_DOCS:BOOL=OFF \
		-D BUILD_MANUAL:BOOL=OFF \
		-D CMAKE_CXX_FLAGS:STRING='-g -O0 -DCL_USE_DEPRECATED_OPENCL_1_1_APIS' \
		$(OPENCL_CMAKE) \
		.. && \
	$(MAKE) -s

prepare-examples: $(viennacl_local_src_dir) build

clean-examples:
	rm -fr $(viennacl_local_src_dir)

endif

EXTRA_DIST=                                  \
  fft.stdout                                 \
  custom-context.stdout                      \
  custom-kernels.stdout                      \
  scheduler.stdout                           \
  bandwidth-reduction.stdout                 \
  blas3_solve_double-test-opencl.stdout      \
  blas3_solve_float-test-opencl.stdout       \
  external_linkage-opencl.stdout             \
  global_variables-test-opencl.stdout        \
  iterators-test-opencl.stdout               \
  matrix_col_double-test-opencl.stdout       \
  matrix_col_float-test-opencl.stdout        \
  matrix_col_int-test-opencl.stdout          \
  matrix_row_double-test-opencl.stdout       \
  matrix_row_float-test-opencl.stdout        \
  matrix_row_int-test-opencl.stdout          \
  matrix_vector_int-test-opencl.stdout       \
  matrix_vector-test-opencl.stdout           \
  nmf-test-opencl.stdout                     \
  scalar-test-opencl.stdout                  \
  structured-matrices-test-opencl.stdout     \
  vector_double-test-opencl.stdout


