# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for tests/regression.
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

noinst_PROGRAMS = test_barrier_between_for_loops test_early_return \
	test_for_with_var_iteration_count test_id_dependent_computation \
	test_locals test_loop_phi_replication test_multi_level_loops_with_barriers \
	test_simple_for_with_a_barrier test_structs_as_args test_vectors_as_args \
	test_barrier_before_return test_infinite_loop test_constant_array \
	test_undominated_variable test_setargs test_null_arg

AM_DEFAULT_SOURCE_EXT = .cpp

AM_LDFLAGS = ../../lib/poclu/libpoclu.la @OPENCL_LIBS@
AM_CXXFLAGS = @OPENCL_CFLAGS@ -Wno-deprecated -Wno-deprecated-declarations
AM_CPPFLAGS = -I$(top_srcdir)/fix-include -I$(top_builddir)/include -I$(top_srcdir)/include -I$(top_srcdir)/lib/CL -DSRCDIR='"$(abs_srcdir)"'




