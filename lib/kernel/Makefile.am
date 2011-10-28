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

LEX=

SUBDIRS = dummy

pkglib_LIBRARIES = libkernel.a

libkernel_a_SOURCES = get_global_size.c		\
                      get_global_id.c		\
                      get_local_id.c		\
                      get_num_groups.c		\
                      get_group_id.c		\
                      as_type.cl		\
                      convert_type.cl		\
                      acos.cl			\
                      acosh.cl			\
                      acospi.cl			\
                      asin.cl			\
                      asinh.cl			\
                      asinpi.cl			\
                      atan.cl			\
                      atan2.cl			\
                      atan2pi.cl		\
                      atanh.cl			\
                      atanpi.cl			\
                      cbrt.cl			\
                      ceil.cl			\
                      copysign.cl		\
                      cos.cl			\
                      cosh.cl			\
                      cospi.cl			\
                      erfc.cl			\
                      erf.cl			\
                      exp.cl			\
                      exp2.cl			\
                      exp10.cl			\
                      expm1.cl			\
                      fabs.cl			\
                      fdim.cl			\
                      floor.cl			\
                      fma.cl			\
                      fmax.cl			\
                      fmin.cl			\
                      fmod.cl			\
                      fract.cl			\
                      hypot.cl			\
                      ilogb.cl			\
                      ldexp.cl			\
                      lgamma.cl			\
                      log.cl			\
                      log2.cl			\
                      log10.cl			\
                      log1p.cl			\
                      logb.cl			\
                      mad.cl			\
                      maxmag.cl			\
                      minmag.cl			\
                      nan.cl			\
                      nextafter.cl		\
                      pow.cl			\
                      pown.cl			\
                      powr.cl			\
                      remainder.cl		\
                      rint.cl			\
                      rootn.cl			\
                      round.cl			\
                      rsqrt.cl			\
                      sin.cl			\
                      sinh.cl			\
                      sinpi.cl			\
                      sqrt.cl			\
                      tan.cl			\
                      tanh.cl			\
                      tanpi.cl			\
                      tgamma.cl			\
                      trunc.cl			\
                      abs.cl			\
                      abs_diff.cl		\
                      add_sat.cl		\
                      hadd.cl			\
                      rhadd.cl			\
                      clamp.cl			\
                      clz.cl			\
                      mad_hi.cl			\
                      mad_sat.cl		\
                      max.cl			\
                      min.cl			\
                      mul_hi.cl			\
                      rotate.cl			\
                      sub_sat.cl		\
                      upsample.cl		\
                      mad24.cl			\
                      mul24.cl			\
                      degrees.cl		\
                      mix.cl			\
                      radians.cl		\
                      step.cl			\
                      smoothstep.cl		\
                      sign.cl			\
                      cross.cl			\
                      dot.cl			\
                      distance.cl		\
                      length.cl			\
                      normalize.cl		\
                      fast_distance.cl		\
                      fast_length.cl		\
                      fast_normalize.cl		\
                      isequal.cl		\
                      isnotequal.cl		\
                      isgreater.cl		\
                      isgreaterequal.cl		\
                      isless.cl			\
                      islessequal.cl		\
                      islessgreater.cl		\
                      isfinite.cl		\
                      isinf.cl			\
                      isnan.cl			\
                      isnormal.cl		\
                      isordered.cl		\
                      isunordered.cl		\
                      signbit.cl		\
                      any.cl			\
                      all.cl			\
                      bitselect.cl		\
                      select.cl

libkernel_a_LIBADD = barrier.o
EXTRA_DIST = barrier.ll

RANLIB = `@LLVM_CONFIG@ --bindir`/llvm-ranlib
AR = `@LLVM_CONFIG@ --bindir`/llvm-ar

.cl.o:
	$(CLANG) $(AM_CPPFLAGS) $(CLANGFLAGS) -c -emit-llvm -include $(top_srcdir)/include/_kernel.h -o $@ $<

.c.o:
	$(CLANG) $(AM_CPPFLAGS) $(CLANGFLAGS) -c -emit-llvm -include $(top_srcdir)/include/_kernel.h -o $@ $<

barrier.o: barrier.ll
	$(LLVM_AS) -o $@ $<

$(libkernel_a_SOURCES:.c=.o):  $(top_srcdir)/include/_kernel.h
$(libkernel_a_SOURCES:.cl=.o):  $(top_srcdir)/include/_kernel.h
