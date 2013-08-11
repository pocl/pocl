# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/examples/kernel.
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

noinst_PROGRAMS = kernel

kernel_SOURCES = kernel.c test_as_type.cl test_bitselect.cl test_convert_sat_regression.cl test_convert_type.cl test_fabs.cl test_fmin_fmax_fma.cl test_hadd.cl test_min_max.cl test_rotate.cl test_sizeof.cl test_block.cl
kernel_LDADD = -lm @OPENCL_LIBS@ ../../lib/poclu/libpoclu.la
kernel_CFLAGS = -std=c99 @OPENCL_CFLAGS@

AM_CPPFLAGS = -I$(top_srcdir)/fix-include -I$(top_srcdir)/include -DSRCDIR='"$(abs_srcdir)"'
