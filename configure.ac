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
AC_INIT([pocl], [20111014], [carlos.delalama@urjc.es])
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

AC_ARG_VAR([CLANG], [Program to compile kernels to bytecode])
AC_ARG_VAR([CLANGFLAGS], [Flags to be passed to CLANG])

AC_CHECK_TOOL([CLANG], [clang])
test -z "$CLANG" && AC_MSG_FAILURE([no clang found in \$PATH])

AC_CHECK_TOOL([LLVM_CONFIG], [llvm-config])
AC_CHECK_TOOL([OPT], [opt])
AC_CHECK_TOOL([LLVM_LD], [llvm-ld])
AC_CHECK_TOOL([LLC], [llc])
AC_CHECK_TOOL([LLVM_AS], [llvm-as])
test -z "$LLVM_CONFIG" || \
test -z "$OPT" || \
test -z "$LLVM_LD" || \
test -z "$LLC" || \
test -z "$LLVM_AS" && AC_MSG_FAILURE([no llvm required binaries found in \$PATH])

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
AC_DEFINE_UNQUOTED([LLVM_LD], ["$LLVM_LD"], [LLVM linker executable.])
AC_DEFINE_UNQUOTED([LLC], ["$LLC"], [LLVM compiler executable.])

AC_LANG([C++])
ACX_PTHREAD()

# Checks for libraries.
old_LDFLAGS=$LDFLAGS
LDFLAGS+=`$LLVM_CONFIG --ldflags`
old_LIBS=$LIBS
LIBS=
AC_SEARCH_LIBS([LLVMInitializeTransformUtils],
               [LLVM-3.0 LLVMTransformUtils],
	       , ,
               [`$LLVM_CONFIG --libs transformutils`])
AC_SUBST([LIBS_LLVMTRANSFORMUTILS], ["$LIBS"])
test "$ac_cv_search_LLVMInitializeTransformUtils" = "no" && \
  AC_MSG_FAILURE([Unable to link with LLVM libraries])
LIBS=$old_LIBS
LDFLAGS=$old_LDFLAGS

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
		 examples/example2/Makefile
		 examples/barriers/Makefile
		 examples/forloops/Makefile
		 examples/standalone/Makefile
                 scripts/Makefile])

AC_OUTPUT
