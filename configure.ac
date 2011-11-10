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
AC_CONFIG_TESTDIR([tests])
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

old_LDFLAGS="$LDFLAGS"
LDFLAGS=-bundle
AC_LINK_IFELSE([AC_LANG_SOURCE([[
int
test(int a, int b)
{
  return a + b;
}
]])],[SHARED_LD_FLAGS=-bundle], [SHARED_LD_FLAGS=-shared])
LDFLAGS="$old_LDFLAGS"

AC_DEFINE_UNQUOTED([BUILDDIR], ["`pwd`"], [Absolute path of build directory.])
AC_DEFINE_UNQUOTED([LLVM_LD], ["$LLVM_LD"], [LLVM linker executable.])
AC_DEFINE_UNQUOTED([LLC], ["$LLC"], [LLVM compiler executable.])

AC_LANG([C++])
ACX_PTHREAD()

# Checks for libraries.
old_LDFLAGS="$LDFLAGS"
LDFLAGS="$LDFLAGS `$LLVM_CONFIG --ldflags`"
old_LIBS="$LIBS"
LIBS=
AC_SEARCH_LIBS([LLVMInitializeTransformUtils],
               [LLVM-3.0 LLVMTransformUtils],
	       , ,
               [`$LLVM_CONFIG --libs transformutils`])
AC_SUBST([LIBS_LLVMTRANSFORMUTILS], ["$LIBS"])
test "$ac_cv_search_LLVMInitializeTransformUtils" = "no" && \
  AC_MSG_FAILURE([Unable to link with LLVM libraries])
LIBS="$old_LIBS"
LDFLAGS="$old_LDFLAGS"

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

# The HOST_ARCH variable is used to choose an optimized
# kernel library for the "default target" which is assumed
# to be the same as the one where the OpenCL host code runs
# in.
HOST_ARCH=`llvm-config --host-target | cut -d '-' -f 1`

ARM_CLANG_FLAGS="-ccc-host-triple armv4t-unknown-linux-gnueabihf"
X86_64_CLANG_FLAGS="-ccc-host-triple x86_64-unknown-linux-gnu"

case $HOST_ARCH in
     arm)
     AC_MSG_NOTICE([using arm optimized kernel lib for the native device])
     # Assume -lm should not be used. TODO: add an embedded math lib
     # for compiling kernels for ARM envs without usable math libs.
     SHARED_LD_FLAGS="$SHARED_LD_FLAGS"
     # Use the default clang flags
     ARM_CLANG_FLAGS=""
     ;;

     x86_64)
     AC_MSG_NOTICE([using x86_64 optimized kernel lib for the native device])
     SHARED_LD_FLAGS="$SHARED_LD_FLAGS -lm"
     X86_64_CLANG_FLAGS=""
     ;;

     *)
     AC_MSG_NOTICE([unknown host, using the generic kernel lib])
     SHARED_LD_FLAGS="$SHARED_LD_FLAGS -lm"
     HOST_ARCH=""
     ;;
esac

AC_DEFINE_UNQUOTED([HOST_ARCH], ["$HOST_ARCH"], 
["The native arch (for selecting the default kernel library version)."])
AC_SUBST([HOST_ARCH])
AC_SUBST([X86_64_CLANG_FLAGS])
AC_SUBST([ARM_CLANG_FLAGS])

AC_CANONICAL_HOST
# Check for which host we are on and setup a few things
# specifically based on the host
AM_CONDITIONAL([DARWIN_HOST_OS], false)
case $host_os in
  darwin* )
    AM_CONDITIONAL([DARWIN_HOST_OS], true)
    AC_SUBST([LIBRARY_SUFFIX], [".dylib"] )
    # MacOS ld outputs useless warnings like
    # ld: warning: -macosx_version_min not specificed, assuming 10.7
    # suppress them with -w.
    SHARED_LD_FLAGS="$SHARED_LD_FLAGS -w"  
         ;;
  *)
    #Default Case
    AC_SUBST([LIBRARY_SUFFIX], [".so"] )
         ;;
esac

AC_SUBST([SHARED_LD_FLAGS])
AC_DEFINE_UNQUOTED([SHARED_LD_FLAGS], ["$SHARED_LD_FLAGS"], [Parameter to compiler to generate loadable module.])

AC_CONFIG_FILES([Makefile
                 include/Makefile
                 include/CL/Makefile
                 lib/Makefile
                 lib/CL/Makefile
		 lib/llvmopencl/Makefile
		 lib/kernel/Makefile
		 lib/kernel/tce/Makefile
		 lib/kernel/x86_64/Makefile
		 lib/kernel/arm/Makefile
		 examples/Makefile
		 examples/example1/Makefile
		 examples/example2/Makefile
		 examples/barriers/Makefile
		 examples/forloops/Makefile
		 examples/standalone/Makefile
		 examples/trig/Makefile
                 scripts/Makefile
                 tests/Makefile
                 tests/atlocal])
# lib/kernel/bgp/Makefile
# lib/kernel/ppc/Makefile

AC_OUTPUT
