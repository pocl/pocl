CLANGFLAGS = -emit-llvm

#Search the .cl,.c and .ll sources first from this (target specific) directory, then
#the one-up (generic) directory. This allows to override the generic implementation 
#simply by adding a similarlily named file in the target specific directory
vpath %.cl @srcdir@:@srcdir@/..
vpath %.c @srcdir@:@srcdir@/..
vpath %.ll @srcdir@:@srcdir@/..
vpath %.h @top_srcdir@/include:@srcdir@:@srcdir@/..

LKERNEL_HDRS=templates.h image.h
# Nodist here because these files should be included
# to the distribution only once, from the root kernel
# makefile.
LKERNEL_SRCS= \
	barrier.ll				\
	get_work_dim.c				\
	get_global_size.c			\
	get_global_id.c				\
	get_local_size.c			\
	get_local_id.c				\
	get_num_groups.c			\
	get_group_id.c				\
	get_global_offset.c			\
	as_type.cl				\
	atomics.cl				\
	acos.cl					\
	acosh.cl				\
	acospi.cl				\
	asin.cl					\
	asinh.cl				\
	asinpi.cl				\
	atan.cl					\
	atan2.cl				\
	atan2pi.cl				\
	atanh.cl				\
	atanpi.cl				\
	cbrt.cl					\
	ceil.cl					\
	convert_type.cl				\
	copysign.cl				\
	cos.cl					\
	cosh.cl					\
	cospi.cl				\
	erfc.cl					\
	erf.cl					\
	exp.cl					\
	exp2.cl					\
	exp10.cl				\
	expm1.cl				\
	fabs.cl					\
	fdim.cl					\
	floor.cl				\
	fma.cl					\
	fmax.cl					\
	fmin.cl					\
	fmod.cl					\
	fract.cl				\
	hypot.cl				\
	ilogb.cl				\
	ldexp.cl				\
	lgamma.cl				\
	log.cl					\
	log2.cl					\
	log10.cl				\
	log1p.cl				\
	logb.cl					\
	mad.cl					\
	maxmag.cl				\
	minmag.cl				\
	nan.cl					\
	nextafter.cl				\
	pow.cl					\
	pown.cl					\
	powr.cl					\
	remainder.cl				\
	rint.cl					\
	rootn.cl				\
	round.cl				\
	rsqrt.cl				\
	sin.cl					\
	sincos.cl				\
	sinh.cl					\
	sinpi.cl				\
	sqrt.cl					\
	tan.cl					\
	tanh.cl					\
	tanpi.cl				\
	tgamma.cl				\
	trunc.cl				\
	divide.cl				\
	recip.cl				\
	abs.cl					\
	abs_diff.cl				\
	add_sat.cl				\
	hadd.cl					\
	rhadd.cl				\
	clamp.cl				\
	clz.cl					\
	mad_hi.cl				\
	mad_sat.cl				\
	max.cl					\
	min.cl					\
	mul_hi.cl				\
	rotate.cl				\
	sub_sat.cl				\
	upsample.cl				\
	popcount.cl				\
	mad24.cl				\
	mul24.cl				\
	degrees.cl				\
	mix.cl					\
	radians.cl				\
	step.cl					\
	smoothstep.cl				\
	sign.cl					\
	cross.cl				\
	dot.cl					\
	distance.cl				\
	length.cl				\
	normalize.cl				\
	fast_distance.cl			\
	fast_length.cl				\
	fast_normalize.cl			\
	isequal.cl				\
	isnotequal.cl				\
	isgreater.cl				\
	isgreaterequal.cl			\
	isless.cl				\
	islessequal.cl				\
	islessgreater.cl			\
	isfinite.cl				\
	isinf.cl				\
	isnan.cl				\
	isnormal.cl				\
	isordered.cl				\
	isunordered.cl				\
	signbit.cl				\
	any.cl					\
	all.cl					\
	bitselect.cl				\
	select.cl				\
	vload.cl				\
	vstore.cl				\
	vload_half.cl				\
	vstore_half.cl				\
	async_work_group_copy.cl		\
	wait_group_events.cl			\
	read_image.cl				\
	write_image.cl				\
	get_image_width.cl			\
	get_image_height.cl     

OBJ_L=$(LKERNEL_SRCS:.cl=.bc)
OBJ_C=$(OBJ_L:.ll=.bc)
OBJ=$(OBJ_C:.c=.bc)

OBJ:LKERNEL_SRCS

#libkernel_SRCS = $LIBKERNEL_SOURCES

#rules to compile the different kernel library source file types into LLVM bitcode
%.bc: %.c @top_builddir@/include/${TARGET_DIR}/types.h
	@CLANG@ -emit-llvm -c -target ${KERNEL_TARGET} -o $@ -x c $< -include ../../../include/${TARGET_DIR}/types.h
%.bc: %.cl @top_builddir@/include/${TARGET_DIR}/types.h @top_srcdir@/include/_kernel.h
	@CLANG@ -emit-llvm -c -target ${KERNEL_TARGET} -o $@ -x cl $< -include ../../../include/${TARGET_DIR}/types.h \
		-include ${abs_top_srcdir}/include/_kernel.h

CLEANFILES = kernel-${KERNEL_TARGET}.bc ${OBJ}

kernel-${KERNEL_TARGET}.bc: ${OBJ}
	llvm-link -o $@ $^


# We need an explicitly rule to overwrite automake guess about LEX file :-(
barrier.bc: barrier.ll
	@LLVM_AS@ -o $@ $<
