# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for tests/workgroup.
# 
# Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
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

noinst_PROGRAMS = run_kernel

AM_LDFLAGS = ../../lib/poclu/libpoclu.la @OPENCL_LIBS@
AM_CPPFLAGS = -I$(top_srcdir)/fix-include -I$(top_srcdir)/include -I$(top_srcdir)/lib/CL -DSRCDIR='"$(abs_srcdir)"' @OPENCL_CFLAGS@

EXTRA_DIST = basic_barriers.cl conditional_barriers.cl forloops.cl forloops_2_2_1_1.stdout loopbarriers.cl basic_barriers_2_2_2_2.stdout tricky_for.cl outerlooppar.cl outerlooppar_2_2_1_1.stdout for_bug.cl for_bug_1_2_1_1.stdout multilatch_bloop.cl multilatch_bloop_1_3_1_1.stdout print_all_ids.cl print_all_ids_114114.txt

