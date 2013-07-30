# sources.mk - Makefile definitions for the including the vecmathlib implementations
# 
# Copyright (c) 2013 Pekka Jääskeläinen / Tampere University of Technology
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
#
# This file should be included from the Makefile.am of the target kernel
# library in case vecmathlib versions of the builtins are wanted.

SECONDARY_VPATH = @srcdir@/../vecmathlib/pocl

LKERNEL_SRCS_EXCLUDE =				\
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
	clamp.cl				\
	copysign.cl				\
	cos.cl					\
	cosh.cl					\
	cospi.cl				\
	cross.cl				\
	degrees.cl				\
	distance.cl				\
	divide.cl				\
	dot.cl					\
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
	fmax.cl					\
	fmin.cl					\
	fmin.cl					\
	fmod.cl					\
	fract.cl				\
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
	log.cl					\
	log10.cl				\
	log1p.cl				\
	log2.cl					\
	logb.cl					\
	mad.cl					\
	max.cl					\
	maxmag.cl				\
	min.cl					\
	minmag.cl				\
	mix.cl					\
	nan.cl					\
	native_cos.cl				\
	native_log2.cl				\
	normalize.cl				\
	pow.cl					\
	pown.cl					\
	powr.cl					\
	radians.cl				\
	recip.cl				\
	remainder.cl				\
	rint.cl					\
	rootn.cl				\
	round.cl				\
	rsqrt.cl				\
	sign.cl					\
	signbit.cl				\
	sin.cl					\
	sincos.cl				\
	sinh.cl					\
	sinpi.cl				\
	smoothstep.cl				\
	sqrt.cl					\
	step.cl					\
	tan.cl					\
	tanh.cl					\
	tanpi.cl				\
	trunc.cl

LKERNEL_SRCS_EXTRA =				\
	acos.cc					\
	acosh.cc				\
	acospi.cl				\
	asin.cc					\
	asinh.cc				\
	asinpi.cl				\
	atan.cc					\
	atan2.cl				\
	atan2pi.cl				\
	atanh.cc				\
	atanpi.cl				\
	cbrt.cc					\
	ceil.cc					\
	clamp.cl				\
	copysign.cc				\
	cos.cc					\
	cosh.cc					\
	cospi.cl				\
	cross.cl				\
	degrees.cl				\
	distance.cl				\
	dot.cl					\
	exp.cc					\
	exp10.cc				\
	exp2.cc					\
	expm1.cc				\
	fabs.cc					\
	fast_distance.cl			\
	fast_length.cl				\
	fast_normalize.cl			\
	fdim.cc					\
	floor.cc				\
	fma.cc					\
	fmax.cc					\
	fmax.cl					\
	fmin.cc					\
	fmin.cl					\
	fmod.cc					\
	fract.cl				\
	frexp.cl				\
	half_cos.cl				\
	half_divide.cl				\
	half_exp.cl				\
	half_exp10.cl				\
	half_exp2.cl				\
	half_log.cl				\
	half_log10.cl				\
	half_log2.cl				\
	half_powr.cl				\
	half_recip.cl				\
	half_rsqrt.cl				\
	half_sin.cl				\
	half_sqrt.cl				\
	half_tan.cl				\
	hypot.cc				\
	ilogb.cl				\
	ilogb_.cc				\
	isequal.cl				\
	isfinite.cc				\
	isgreater.cl				\
	isgreaterequal.cl			\
	isinf.cc				\
	isless.cl				\
	islessequal.cl				\
	islessgreater.cl			\
	isnan.cc				\
	isnormal.cc				\
	isnotequal.cl				\
	isordered.cl				\
	isunordered.cl				\
	ldexp.cl				\
	ldexp_.cc				\
	length.cl				\
	log.cc					\
	log10.cc				\
	log1p.cc				\
	log2.cc					\
	logb.cl					\
	mad.cl					\
	max.cl					\
	maxmag.cl				\
	min.cl					\
	minmag.cl				\
	mix.cl					\
	modf.cl					\
	nan.cl					\
	native_cos.cl				\
	native_divide.cl			\
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
	normalize.cl				\
	pow.cc					\
	pown.cl					\
	powr.cl					\
	radians.cl				\
	remainder.cc				\
	remquo.cl				\
	rint.cc					\
	rootn.cl				\
	round.cc				\
	rsqrt.cc				\
	sign.cl					\
	signbit.cc				\
	sin.cc					\
	sincos.cl				\
	sinh.cc					\
	sinpi.cl				\
	smoothstep.cl				\
	sqrt.cc					\
	step.cl					\
	tan.cc					\
	tanh.cc					\
	tanpi.cl				\
	trunc.cc

include ../sources.mk
