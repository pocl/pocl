# sources.mk - a template for the target overridden kernel library makefiles
# 
# Copyright (c) 2011-2013 Universidad Rey Juan Carlos
#                         Pekka Jääskeläinen / Tampere University of Technology
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

CLANGFLAGS = -emit-llvm

# Search the .cl, .c, and .ll sources first from this (target
# specific) directory, then the one-up (generic) directory. This
# allows to override the generic implementation simply by adding a
# similarly named file in the target specific directory

# NOTE: GNU make always searches from '.' first, which can be
# problematic in case one wants to override the files in the
# device-specific directory. To circumvent this, one can override a
# default file with a more accurate path, e.g.,
# vecmathlib/pocl/fmax.cl. This overrides a possible fmax.cl found in
# the current directory by one from the VML. TODO: dependency tracking
# does not work in that case because the .bc is not created in the
# directory where the source is.
vpath %.h @top_srcdir@/include:@srcdir@:${SECONDARY_VPATH}:@srcdir@/..
vpath %.c @srcdir@:${SECONDARY_VPATH}:@srcdir@/..
vpath %.cc @srcdir@:${SECONDARY_VPATH}:@srcdir@/..
vpath %.cl @srcdir@:${SECONDARY_VPATH}:@srcdir@/..
vpath %.ll @srcdir@:${SECONDARY_VPATH}:@srcdir@/..

LKERNEL_HDRS = image.h templates.h
LKERNEL_SRCS_DEFAULT=				\
	barrier.ll				\
	get_global_id.c				\
	get_global_offset.c			\
	get_global_size.c			\
	get_group_id.c				\
	get_local_id.c				\
	get_local_size.c			\
	get_num_groups.c			\
	get_work_dim.c				\
	abs.cl					\
	abs_diff.cl				\
	acos.cl					\
	acosh.cl				\
	acospi.cl				\
	add_sat.cl				\
	all.cl					\
	any.cl					\
	as_type.cl				\
	asin.cl					\
	asinh.cl				\
	asinpi.cl				\
	async_work_group_copy.cl		\
	atan.cl					\
	atan2.cl				\
	atan2pi.cl				\
	atanh.cl				\
	atanpi.cl				\
	atomics.cl				\
	bitselect.cl				\
	cbrt.cl					\
	ceil.cl					\
	clamp.cl				\
	clamp_int.cl				\
	clz.cl					\
	convert_type.cl				\
	copysign.cl				\
	cos.cl					\
	cosh.cl					\
	cospi.cl				\
	cross.cl				\
	degrees.cl				\
	distance.cl				\
	divide.cl				\
	dot.cl					\
	erf.cl					\
	erfc.cl					\
	exp.cl					\
	exp10.cl				\
	exp2.cl					\
	expm1.cl				\
	fabs.cl					\
	fast_distance.cl			\
	fast_length.cl				\
	fast_normalize.cl			\
	fdim.cl					\
	floor.cl				\
	fma.cl					\
	fmax.cl					\
	fmin.cl					\
	fmod.cl					\
	fract.cl				\
	get_image_height.cl			\
	get_image_width.cl			\
	hadd.cl					\
	hypot.cl				\
	ilogb.cl				\
	isequal.cl				\
	isfinite.cl				\
	isgreater.cl				\
	isgreaterequal.cl			\
	isinf.cl				\
	isless.cl				\
	islessequal.cl				\
	islessgreater.cl			\
	isnan.cl				\
	isnormal.cl				\
	isnotequal.cl				\
	isordered.cl				\
	isunordered.cl				\
	ldexp.cl				\
	length.cl				\
	lgamma.cl				\
	log.cl					\
	log10.cl				\
	log1p.cl				\
	log2.cl					\
	logb.cl					\
	mad.cl					\
	mad24.cl				\
	mad_hi.cl				\
	mad_sat.cl				\
	max.cl					\
	max_i.cl				\
	maxmag.cl				\
	min.cl					\
	min_i.cl				\
	minmag.cl				\
	mix.cl					\
	mul24.cl				\
	mul_hi.cl				\
	nan.cl					\
	native_cos.cl				\
	native_log2.cl				\
	nextafter.cl				\
	normalize.cl				\
	popcount.cl				\
	pow.cl					\
	pown.cl					\
	powr.cl					\
	radians.cl				\
	read_image.cl				\
	recip.cl				\
	remainder.cl				\
	rhadd.cl				\
	rint.cl					\
	rootn.cl				\
	rotate.cl				\
	round.cl				\
	rsqrt.cl				\
	select.cl				\
	sign.cl					\
	signbit.cl				\
	sin.cl					\
	sincos.cl				\
	sinh.cl					\
	sinpi.cl				\
	smoothstep.cl				\
	sqrt.cl					\
	step.cl					\
	sub_sat.cl				\
	tan.cl					\
	tanh.cl					\
	tanpi.cl				\
	tgamma.cl				\
	trunc.cl				\
	upsample.cl				\
	vload.cl				\
	vload_half.cl				\
	vstore.cl				\
	vstore_half.cl				\
	wait_group_events.cl			\
	write_image.cl

# The standard list of kernel sources can be modified with
# LKERNEL_SRCS_EXCLUDE, which removes files from the standard list,
# and LKERNEL_SRCS_EXTRA, which adds extra files to the source list.
LKERNEL_SRCS =								\
	$(filter-out ${LKERNEL_SRCS_EXCLUDE}, ${LKERNEL_SRCS_DEFAULT})	\
	${LKERNEL_SRCS_EXTRA}

OBJ = $(LKERNEL_SRCS:%=%.bc)

# Rules to compile the different kernel library source file types into
# LLVM bitcode
%.c.bc: %.c @top_builddir@/include/${TARGET_DIR}/types.h
	@CLANG@ -Xclang -ffake-address-space-map -emit-llvm ${CLFLAGS} ${EXTRA_CLANGFLAGS} -c -target ${KERNEL_TARGET} -o ${notdir $@} -x c $< -include ../../../include/${TARGET_DIR}/types.h
%.cc.bc: %.cc @top_builddir@/include/${TARGET_DIR}/types.h
	@CLANGXX@ -Xclang -ffake-address-space-map -fno-exceptions -emit-llvm ${EXTRA_CLANGFLAGS} ${CLANGXX_FLAGS} -c -target ${KERNEL_TARGET} -o ${notdir $@} $< -include ../../../include/${TARGET_DIR}/types.h
%.cl.bc: %.cl @top_builddir@/include/${TARGET_DIR}/types.h @top_srcdir@/include/_kernel.h
	@CLANG@ -Xclang -ffake-address-space-map -emit-llvm ${CLFLAGS} ${EXTRA_CLANGFLAGS} -fsigned-char -c -target ${KERNEL_TARGET} -o ${notdir $@} -x cl $< -include ../../../include/${TARGET_DIR}/types.h -include ${abs_top_srcdir}/include/_kernel.h
%.ll.bc: %.ll
	@LLVM_AS@ -o $@ $<

CLEANFILES = kernel-${KERNEL_TARGET}.bc ${notdir ${OBJ}}

# Optimize the bitcode library to speed up optimization times for the
# OpenCL kernels
kernel-${KERNEL_TARGET}.bc: ${OBJ}
	@LLVM_LINK@ ${notdir $^} -o - | @LLVM_OPT@ ${KERNEL_LIB_OPT_FLAGS} -O3 -o $@
