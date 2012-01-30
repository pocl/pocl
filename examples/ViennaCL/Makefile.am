# Usage:
# 
# Download and unpack the ViennaCL package into $pocl_source_dir/examples/ViennaCL/:
# http://sourceforge.net/projects/viennacl/files/1.2.x/ViennaCL-1.2.0-src.tar.gz
#
# Then, in $pocl_sourc_dir/, run 'make check' or 'make TESTUITE=viennacl check' 
# to run just the ViennaCL tests.
# 


viennacl_dir=${top_srcdir}/examples/ViennaCL/ViennaCL-1.2.0-src

#.PHONY: build run_all run_working check_dir run_HelloWorld run_OpenCLInfo run_OpenCLConvolutionChap3
#.PHONY: run_SimpleBufferSubBuffer run_spmv run_VectorAdd run_Dijkstra

build:
	(cd ${viennacl_dir}; mkdir -p build; cd build; \
	cmake \
  -D BUILD_DOXYGEN_DOCS:BOOL=OFF -D BUILD_MANUAL:BOOL=OFF \
	-D CMAKE_CXX_FLAGS:STRING='-g -O0 -DCL_USE_DEPRECATED_OPENCL_1_1_APIS  ${PTHREAD_CFLAGS}' \
	-D OPENCL_LIBRARIES:FILEPATH=${abs_top_builddir}/lib/CL/.libs/libOpenCL.so \
	-D CMAKE_EXE_LINKER_FLAGS:STRING='${PTHREAD_CFLAGS}' \
	-D OPENCL_INCLUDE_DIRS:PATH=${abs_top_srcdir}/include .. && \
	make clean all -sk)

run_vector: 
	@echo
	@echo "###### vector-test"
	@echo
	@(cd ${viennacl_dir}; cd build/test; ./vector-test)

run_working: run_vector

run_all: run_working

clean:
	rm -fr ${viennacl_dir}/build
