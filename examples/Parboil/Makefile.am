# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/examples/Parboil.
# 
# Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
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

EXTRA_DIST = *.patch

if TEST_SUITE_PARBOIL

testsuite_pocl_dir=${abs_top_srcdir}/examples/Parboil
testsuite_src_dir=${testsuite_pocl_dir}/parboil

driver_tarfile=pb2.5driver.tgz
benchmarks_tarfile=pb2.5benchmarks.tgz
datasets_tarfile=pb2.5datasets_standard.tgz

.PHONY: build

all: build-prepare

${srcdir}/${driver_tarfile}:
	@echo 'You need to download $@ to "$(srcdir)"'
	@echo 'from the http://impact.crhc.illinois.edu/parboil.aspx website'
	exit 1

${srcdir}/${benchmarks_tarfile}:
	@echo 'You need to download $@ to "$(srcdir)"'
	@echo 'from the http://impact.crhc.illinois.edu/parboil.aspx website'
	exit 1

${srcdir}/${datasets_tarfile}:
	@echo 'You need to download $@ to "$(srcdir)"'
	@echo 'from the http://impact.crhc.illinois.edu/parboil.aspx website'
	exit 1

$(testsuite_src_dir): 
	$(MAKE) ${srcdir}/$(driver_tarfile) 
	$(MAKE) ${srcdir}/$(benchmarks_tarfile)
	$(MAKE) ${srcdir}/$(datasets_tarfile)
	cd $(testsuite_pocl_dir) && \
	tar xvzf $(driver_tarfile) && cd parboil && \
	tar xvzf ../${benchmarks_tarfile} && \
	tar xvzf ../${datasets_tarfile}

build-prepare: $(testsuite_src_dir)
# The OpenCL benchmarks want GPU devices. Change to CPU devices using
# sed.
	@cd $(testsuite_src_dir) && \
	find . -name "*.c" -o -name "*.cc" -o -name "*.cpp" | \
	grep opencl | \
	xargs perl -pi -e 's/CL_DEVICE_TYPE_GPU/CL_DEVICE_TYPE_CPU/g'
	@cd ${testsuite_pocl_dir} && patch -sNp0 < opencl.mk.patch > /dev/null || true
	@cd ${testsuite_pocl_dir} && patch -sNp0 < OpenCL_common.cpp.patch > /dev/null || true
	@cd ${testsuite_src_dir} && patch -sNp0 < ${testsuite_pocl_dir}/benchmarks.patch > /dev/null || true
	@echo -e "OPENCL_PATH=${abs_top_srcdir}/\nOPENCL_LIB_PATH=${abs_top_builddir}/lib/CL/.libs" > ${testsuite_src_dir}/common/Makefile.conf
	@echo -e "APP_CFLAGS=-DCL_USE_DEPRECATED_OPENCL_1_1_APIS" >> ${testsuite_src_dir}/common/Makefile.conf
	@echo -e "APP_CXXFLAGS=-DCL_USE_DEPRECATED_OPENCL_1_1_APIS" >> ${testsuite_src_dir}/common/Makefile.conf
src=rodinia_2.0.1/opencl

build-spmv: build-prepare
	cd ${testsuite_src_dir} ; ./parboil compile spmv opencl_base 

run-spmv: 
	cd ${testsuite_src_dir} ; ./parboil run spmv opencl_base small

build-stencil: build-prepare
	cd ${testsuite_src_dir} ; ./parboil compile stencil opencl_base 

run-stencil: 
	cd ${testsuite_src_dir} ; ./parboil run stencil opencl_base small

build-tpacf: build-prepare
	cd ${testsuite_src_dir} ; ./parboil compile tpacf opencl_base 

run-tpacf: 
	cd ${testsuite_src_dir} ; ./parboil run tpacf opencl_base small

build-cutcp: build-prepare
	cd ${testsuite_src_dir} ; ./parboil compile cutcp opencl_base 

run-cutcp: 
	cd ${testsuite_src_dir} ; ./parboil run cutcp opencl_base small

build-mri-gridding: build-prepare
	cd ${testsuite_src_dir} ; ./parboil compile mri-gridding opencl_base 

run-mri-gridding: 
	cd ${testsuite_src_dir} ; ./parboil run mri-gridding opencl_base small

build-sad: build-prepare
	cd ${testsuite_src_dir} ; ./parboil compile sad opencl_base 

run-sad: 
	cd ${testsuite_src_dir} ; ./parboil run sad opencl_base default

build-bfs: build-prepare parboil/benchmarks/bfs/src/opencl_base/OpenCL_common.cpp
	cd ${testsuite_src_dir} ; ./parboil compile bfs opencl_base 

run-bfs: 
	cd ${testsuite_src_dir} ; ./parboil run bfs opencl_base NY

build-histo: build-prepare
	cd ${testsuite_src_dir} ; ./parboil compile histo opencl_base 

run-histo: 
	cd ${testsuite_src_dir} ; ./parboil run histo opencl_base default

build-sgemm: build-prepare
	cd ${testsuite_src_dir} ; ./parboil compile sgemm opencl_base 

run-sgemm: 
	cd ${testsuite_src_dir} ; ./parboil run sgemm opencl_base medium

build-mri-q: build-prepare
	cd ${testsuite_src_dir} ; ./parboil compile mri-q opencl

run-mri-q: 
	cd ${testsuite_src_dir} ; ./parboil run mri-q opencl small

build-lbm: build-prepare
	cd ${testsuite_src_dir} ; ./parboil compile lbm opencl_base

run-lbm: 
	cd ${testsuite_src_dir} ; ./parboil run lbm opencl_base short

build: build-spmv build-stencil build-tpacf build-cutcp build-mri-gridding build-sad \
	build-bfs build-histo build-sgemm build-mri-q build-lbm

clean-local:
	$(RM) -r parboil

check-local: build

distclean-local: clean-local
	$(RM) -r $(testsuite_src_dir)
	$(RM) ${driver_tarfile} ${benchmarks_tarfile} ${datasets_tarfile}

endif
