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

pkglib_LIBRARIES = libkernel.a

libkernel_a_SOURCES = get_global_id.c		\
                      get_group_id.cl		\
                      get_local_id.cl		\
                      as_type.cl		\
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
                      copysign.cl		\
                      cos.cl			\
                      cosh.cl			\
                      cospi.cl			\
                      dot.cl			\
                      erfc.cl			\
                      erf.cl			\
                      exp.cl			\
                      exp2.cl			\
                      exp10.cl			\
                      expm1.cl			\
                      fabs.cl			\
                      fdim.cl			\
                      fma.cl			\
                      fmax.cl			\
                      fmin.cl			\
                      fmod.cl			\
                      gamma.cl			\
                      hypot.cl			\
                      lgamma.cl			\
                      log.cl			\
                      log2.cl			\
                      log10.cl			\
                      log1p.cl			\
                      logb.cl			\
                      mad.cl			\
                      mad24.cl			\
                      maxmag.cl			\
                      minmag.cl			\
                      nextafter.cl		\
                      pow.cl			\
                      remainder.cl		\
                      rint.cl			\
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
                      trunc.cl
libkernel_a_LIBADD = barrier.o
EXTRA_DIST = barrier.ll

RANLIB = `@LLVM_CONFIG@ --bindir`/llvm-ranlib
AR = `@LLVM_CONFIG@ --bindir`/llvm-ar

.cl.o:
	$(CLANG) $(AM_CPPFLAGS) $(CLANGFLAGS) -c -emit-llvm -include $(top_srcdir)/include/_kernel.h -o $@ $<

get_global_id.o: get_global_id.c
	$(CLANG) $(AM_CPPFLAGS) $(CLANGFLAGS) -c -emit-llvm -include $(top_srcdir)/include/_kernel.h -o $@ $<

barrier.o: barrier.ll
	$(LLVM_AS) -o $@ $<
