# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/examples/Rodinia.
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

if TEST_SUITE_RODINIA
testsuite_pocl_dir=${abs_top_srcdir}/examples/Rodinia
testsuite_src_dir=${testsuite_pocl_dir}/rodinia_2.0.1
tarfile=rodinia_2.0.1.tar.gz

.PHONY: build

all: $(testsuite_src_dir)

$(srcdir)/$(tarfile):
	@echo 'You need to go download $(tarfile) in $(srcfile)'
	@echo 'from the http://lava.cs.virginia.edu/Rodinia/download.htm website'
	exit 1

$(testsuite_src_dir):
	$(MAKE) $(srcdir)/$(tarfile)
	cd $(testsuite_pocl_dir) && \
	tar xzf $(tarfile)

# Rodinia are build in place and its source tree is modified.
# So, copying the source tree in our build tree if required
# (ie if $(srcdir) is not $(CURDIR) )
build-prepare: $(testsuite_src_dir)
	if [ ! -d rodinia_2.0.1 ]; then cp -aL $(testsuite_src_dir) . ; fi
	: Rodinia Makefiles are a mess. Trying to fix them a little.
	sed -i \
		-e 's/-lOpenCL/\$$(OPENCL_LIBS)/g' \
		-e 's/-I\$$(OPENCL_INC)/\$$(OPENCL_CFLAGS)/g' \
		-e 's/-L\$$(OPENCL_LIB)/\$$(OPENCL_CFLAGS)/g' \
		-e 's/	clear/	: clear/g' \
		rodinia_2.0.1/opencl/*/Makefile \
		rodinia_2.0.1/opencl/*/makefile \
		rodinia_2.0.1/opencl/*/*/Makefile \
		rodinia_2.0.1/opencl/*/*/makefile

MAKE_OPTS= OPENCL_LIBS="$(OPENCL_EXTLIBS)" OPENCL_CFLAGS="-I${abs_top_srcdir}/include $(OPENCL_CFLAGS)"

src=rodinia_2.0.1/opencl

build-backprop: build-prepare
	@echo "  [Building Rodinia backprop]"
	sed -i "s/log2(HEIGHT)/log2((float)HEIGHT)/g" $(src)/backprop/backprop_kernel.cl
	sed -i 's/pow(2, i)/pow(2.0f, i)/g' $(src)/backprop/backprop_kernel.cl
	sed -i 's/ use_gpu = 1/ use_gpu = 0/g' $(src)/backprop/backprop_ocl.cpp
	$(MAKE) clean $(MAKE_OPTS) -C $(src)/backprop
	$(MAKE) backprop $(MAKE_OPTS) -C $(src)/backprop

build-bfs: build-prepare
	@echo "  [Building Rodinia bfs]"
	sed -i 's/CL_DEVICE_TYPE_GPU/CL_DEVICE_TYPE_CPU/g' $(src)/bfs/CLHelper.h
	$(MAKE) clean $(MAKE_OPTS) -C $(src)/bfs
	$(MAKE) release $(MAKE_OPTS) -C $(src)/bfs

build-cfd: build-prepare
	@echo "  [Building Rodinia cfd]"
	sed  -i 's/t gpu/t cpu/g' $(src)/cfd/run
	$(MAKE) clean $(MAKE_OPTS) -C $(src)/cfd
	$(MAKE) release $(MAKE_OPTS) -C $(src)/cfd

build-lud: build-prepare
	@echo "  [Building Rodinia lud]"
	sed -i 's/ use_gpu = 1/ use_gpu = 0/g' $(src)/lud/ocl/lud.cpp
	$(MAKE) clean $(MAKE_OPTS) -C $(src)/lud
	$(MAKE) lud $(MAKE_OPTS) -C $(src)/lud

build-hotspot: build-prepare
	@echo "  [Building Rodinia hotspot]"
	sed -i 's/CL_DEVICE_TYPE_GPU/CL_DEVICE_TYPE_CPU/g' $(src)/hotspot/hotspot.c
	$(MAKE) clean $(MAKE_OPTS) -C $(src)/hotspot
	$(MAKE) hotspot $(MAKE_OPTS) -C $(src)/hotspot

build-kmeans: build-prepare
	@echo "  [Building Rodinia kmeans]"
	sed -i 's/ use_gpu = 1/ use_gpu = 0/g' $(src)/kmeans/kmeans.cpp
	$(MAKE) clean $(MAKE_OPTS) -C $(src)/kmeans
	$(MAKE) kmeans $(MAKE_OPTS) -C $(src)/kmeans

build-lavaMD: build-prepare
	@echo "  [Building Rodinia lavaMD]"
	sed -i 's/CL_DEVICE_TYPE_GPU/CL_DEVICE_TYPE_CPU/g' $(src)/lavaMD/kernel/kernel_gpu_opencl_wrapper.c
	-$(MAKE) clean $(MAKE_OPTS) -C $(src)/lavaMD
	$(MAKE) $(MAKE_OPTS) -C $(src)/lavaMD

build-pathfinder: build-prepare
	@echo "  [Building Rodinia pathfinder]"
	sed -i 's/cl.init(1)/cl.init(0)/g' $(src)/pathfinder/main.cpp
	$(MAKE) clean $(MAKE_OPTS) -C $(src)/pathfinder
	$(MAKE) $(MAKE_OPTS) -C $(src)/pathfinder

build-srad: build-prepare
	@echo "  [Building Rodinia srad]"
	sed -i 's/CL_DEVICE_TYPE_GPU/CL_DEVICE_TYPE_CPU/g' $(src)/srad/main.cpp
	$(MAKE) clean $(MAKE_OPTS) -C $(src)/srad
	$(MAKE) $(MAKE_OPTS) -C $(src)/srad

build: build-backprop build-bfs build-cfd build-lud build-hotspot build-kmeans\
	build-lavaMD build-pathfinder build-srad

clean-local:
	$(RM) -r rodinia_2.0.1

distclean-local: clean-local
	$(RM) -r $(testsuite_src_dir)
	$(RM) $(testsuite_pocl_dir)/rodinia_2.0.1.tar.gz
endif
