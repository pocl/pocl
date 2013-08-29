# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/lib/kernel.
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



KERNEL_TARGET = @TARGET@

all: kernel-${KERNEL_TARGET}.bc



# These may be necessary for powerpc64. If so: why? Is this maybe only
# system-specific? Can this be enabled for all architectures?
# RANLIB = @LLVM_RANLIB@
# AR = @LLVM_AR@


include sources.mk

# The standard list of kernel sources can be modified with
# LKERNEL_SRCS_EXCLUDE, which removes files from the standard list,
# and LKERNEL_SRCS_EXTRA, which adds extra files to the source list.
LKERNEL_SRCS =								\
	$(filter-out ${LKERNEL_SRCS_EXCLUDE}, ${LKERNEL_SRCS_DEFAULT})	\
	${LKERNEL_SRCS_EXTRA}

OBJ = $(LKERNEL_SRCS:%=%.bc)



# Rules to compile the different kernel library source file types into
# LLVM bitcode
%.c.bc: %.c @top_srcdir@/include/types.h
	@CLANG@ -Xclang -ffake-address-space-map -emit-llvm ${CLFLAGS} -ffp-contract=off -c -target ${KERNEL_TARGET} -o ${notdir $@} -x c $< -include ${abs_top_srcdir}/include/types.h
%.cc.bc: %.cc @top_srcdir@/include/features.h
	@CLANGXX@ -Xclang -ffake-address-space-map -fno-exceptions -emit-llvm -ffp-contract=off ${CLANGXX_FLAGS} -c -target ${KERNEL_TARGET} -o ${notdir $@} $< -include ${abs_top_srcdir}/include/features.h
%.cl.bc: %.cl @top_srcdir@/include/_kernel.h
	@CLANG@ -Xclang -ffake-address-space-map -emit-llvm ${CLFLAGS} -ffp-contract=off -fsigned-char -c -target ${KERNEL_TARGET} -o ${notdir $@} -x cl $< -include ${abs_top_srcdir}/include/_kernel.h
%.ll.bc: %.ll
	@LLVM_AS@ -o $@ $<

CLEANFILES = kernel-${KERNEL_TARGET}.bc ${notdir ${OBJ}}

# Optimize the bitcode library to speed up optimization times for the
# OpenCL kernels
kernel-${KERNEL_TARGET}.bc: ${OBJ}
	@LLVM_LINK@ ${notdir $^} -o - | @LLVM_OPT@ ${KERNEL_LIB_OPT_FLAGS} -O3 -o $@



# The source files in this dir are not built directly
# but are used with the target specific subdirs as
# fallback implementations. Include them in the
# dist once here.
EXTRA_DIST = *.c *.cl *.h *.ll			\
	vecmathlib/LICENCE			\
	vecmathlib/README			\
	vecmathlib/*.h				\
	vecmathlib-pocl/*.cc			\
	vecmathlib-pocl/*.cl			\
	vecmathlib-pocl/*.h			\
	vecmathlib-pocl/*.py

