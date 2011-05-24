#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
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

AC_PREREQ([2.61])
AC_INIT([llvmopencl], [20110324], [carlos.delalama@urjc.es])
AC_CONFIG_SRCDIR([config.h.in])
AC_CONFIG_HEADER([config.h])
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign])
LT_INIT

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX
AC_PROG_LIBTOOL

AC_PATH_PROG([CLANG], [clang])
AC_PATH_PROG([LLVM_CONFIG], [llvm-config])
AC_PATH_PROG([OPT], [opt])
AC_PATH_PROG([LLVM_LD], [llvm-ld])
AC_PATH_PROG([LLC], [llc])
AC_PATH_PROG([LLVM_AS], [llvm-as])

old_LDFLAGS=$LDFLAGS
LDFLAGS=-bundle
AC_LINK_IFELSE([AC_LANG_SOURCE([[
int
test(int a, int b)
{
  return a + b;
}
]])],[SHARED=-bundle], [SHARED=-shared])
LDFLAGS=$old_LDFLAGS
AC_SUBST([SHARED])

AC_DEFINE_UNQUOTED([SHARED], ["$SHARED"], [Parameter to compiler to generate loadable module.])
AC_DEFINE_UNQUOTED([BUILDDIR], ["`pwd`"], [Absolute path of build directory.])
AC_DEFINE_UNQUOTED([LLVM_LD], ["$LLVM_LD"], [Path to LLVM linker.])
AC_DEFINE_UNQUOTED([LLC], ["$LLC"], [Path to LLVM compiler.])

# Checks for libraries.
AC_SUBST([LIBS_TRANSFORMUTILS], [`$LLVM_CONFIG --libs transformutils`])

# Checks for header files.
AC_CHECK_HEADERS([stddef.h stdint.h stdlib.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_INT16_T
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_INT8_T
AC_TYPE_SIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T

# Checks for library functions.
#AC_FUNC_MALLOC

AC_CONFIG_FILES([Makefile
                 include/Makefile
                 include/CL/Makefile
                 lib/Makefile
                 lib/CL/Makefile
		 lib/llvmopencl/Makefile
		 lib/kernel/Makefile
		 examples/Makefile
		 examples/example1/Makefile
		 examples/barriers/Makefile
		 examples/example2/Makefile
                 scripts/Makefile])

AC_OUTPUT
