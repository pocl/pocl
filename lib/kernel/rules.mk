# rules.mk - the make rules for building the kernel library
# 
# Copyright (c) 2013 Erik Schnetter
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



# The caller (the Makefile which includes this file) needs to set the
# following variables:
# 
# KERNEL_TARGET
# CLANG_FLAGS
# LLC_FLAGS
# LD_FLAGS



all: kernel-${KERNEL_TARGET}.bc



# These may be necessary for powerpc64. If so: why? Is this maybe only
# system-specific? Can this be enabled for all architectures?
# RANLIB = @LLVM_RANLIB@
# AR = @LLVM_AR@


# The standard list of kernel sources can be modified with
# LKERNEL_SRCS_EXCLUDE, which removes files from the standard list,
# and LKERNEL_SRCS_EXTRA, which adds extra files to the source list.
LKERNEL_SRCS =								\
	$(filter-out ${LKERNEL_SRCS_EXCLUDE}, ${LKERNEL_SRCS_DEFAULT})	\
	${LKERNEL_SRCS_EXTRA}

OBJ = $(LKERNEL_SRCS:%=%.bc)



vpath %.c  @top_srcdir@/lib/kernel
vpath %.cc @top_srcdir@/lib/kernel
vpath %.cl @top_srcdir@/lib/kernel
vpath %.ll @top_srcdir@/lib/kernel

# Rules to compile the different kernel library source file types into
# LLVM bitcode
%.c.bc: %.c ${abs_top_srcdir}/include/pocl_types.h ${abs_top_srcdir}/include/pocl_features.h
	mkdir -p ${dir $@}
	@CLANG@ ${CLANG_FLAGS} ${CLFLAGS} -c -o $@ $< -include ${abs_top_srcdir}/include/pocl_types.h
%.cc.bc: %.cc ${abs_top_srcdir}/include/pocl_features.h
	mkdir -p ${dir $@}
	@CLANGXX@ ${CLANG_FLAGS} ${CLANGXX_FLAGS} -c -o $@ $< -include ${abs_top_srcdir}/include/pocl_features.h
%.cl.bc: %.cl ${abs_top_srcdir}/include/_kernel.h ${abs_top_srcdir}/include/_kernel_c.h ${abs_top_srcdir}/include/pocl_types.h ${abs_top_srcdir}/include/pocl_features.h
	mkdir -p ${dir $@}
	@CLANG@ ${CLANG_FLAGS} -x cl ${CLFLAGS} -fsigned-char -c -o $@ $< -include ${abs_top_srcdir}/include/_kernel.h
%.ll.bc: %.ll
	mkdir -p ${dir $@}
	@LLVM_AS@ -o $@ $<

CLEANFILES = kernel-${KERNEL_TARGET}.bc ${OBJ}

# Optimize the bitcode library to speed up optimization times for the
# OpenCL kernels
kernel-${KERNEL_TARGET}.bc: ${OBJ}
	@LLVM_LINK@ $^ -o - | @LLVM_OPT@ ${LLC_FLAGS} ${KERNEL_LIB_OPT_FLAGS} -O3 -fp-contract=off -o $@
