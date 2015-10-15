# sources.mk - a list of all kernel source files
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

LKERNEL_HDRS = image.h pocl_image_rw_utils.h templates.h


LKERNEL_SRCS_DEFAULT =				\
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
	barrier.ll				\
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
	get_global_id.c				\
	get_global_offset.c			\
	get_global_size.c			\
	get_group_id.c				\
	get_image_depth.cl			\
	get_image_height.cl			\
	get_image_width.cl			\
	get_image_dim.cl			\
	get_local_id.c				\
	get_local_size.c			\
	get_num_groups.c			\
	get_work_dim.c				\
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
	native_exp.cl				\
	native_exp10.cl				\
	native_exp2.cl				\
	native_log.cl				\
	native_log10.cl				\
	native_log2.cl				\
	native_powr.cl				\
	native_recip.cl				\
	native_rsqrt.cl				\
	native_sin.cl				\
	native_sqrt.cl				\
	native_tan.cl				\
	nextafter.cl				\
	normalize.cl				\
	popcount.cl				\
	pow.cl					\
	pown.cl					\
	powr.cl					\
	printf.c                                \
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
	shuffle.cl				\
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

# vim: set noexpandtab ts=8:
