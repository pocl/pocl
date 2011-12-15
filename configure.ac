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
AC_INIT([pocl], [0.9], [carlos.delalama@urjc.es])
AC_CONFIG_SRCDIR([config.h.in])
AC_CONFIG_HEADER([config.h clconfig.h])
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_TESTDIR([tests])
AM_INIT_AUTOMAKE([foreign])
LT_INIT

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX

AC_CHECK_PROGS([LLVM_CONFIG], [llvm-config])
test -z "$LLVM_CONFIG" && AC_MSG_FAILURE([no llvm-config found in \$PATH])

LLVM_VERSION=`$LLVM_CONFIG --version`

case "$LLVM_VERSION" in
     3.0*)
     AC_DEFINE([LLVM_3_0], [], "Using LLVM 3.0")
   ;;
     *svn)
     AC_DEFINE([LLVM_3_1], [], "Using LLVM svn (upcoming 3.1)")     
     AC_DEFINE([LLVM_SVN], [], "Using LLVM svn (upcoming 3.1)")     
   ;;   
     *)
   AC_MSG_NOTICE(
   [
Unsupported LLVM version. 
   ])
   LLVM_VERSION=
   ;;
esac


#AC_LANG([C++])
ACX_PTHREAD()

# Checks for libraries.
#old_LDFLAGS="$LDFLAGS"
#LDFLAGS="$LDFLAGS `$LLVM_CONFIG --ldflags`"
#old_LIBS="$LIBS"
#LIBS=
#AC_SEARCH_LIBS([LLVMInitializeTransformUtils],
#               [LLVM-3.0 LLVMTransformUtils],
#	       , ,
#               [`$LLVM_CONFIG --libs transformutils`])
#AC_SUBST([LIBS_LLVMTRANSFORMUTILS], ["$LIBS"])
#test "$ac_cv_search_LLVMInitializeTransformUtils" = "no" && \
#  AC_MSG_FAILURE([Unable to link with LLVM libraries])
#LIBS=
#AC_SEARCH_LIBS([LLVMInitializeAnalysis],
#               [LLVM-3.0 LLVMAnalysis],
#	       , ,
#               [`$LLVM_CONFIG --libs analysis`])
#AC_SUBST([LIBS_LLVMANALYSIS], ["$LIBS"])
#test "$ac_cv_search_LLVMInitializeAnalysis" = "no" && \
#  AC_MSG_FAILURE([Unable to link with LLVM libraries])
#LIBS="$old_LIBS"
#LDFLAGS="$old_LDFLAGS"

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.
#AC_FUNC_MALLOC

# Following checks are related to the build system for devices (clang and so),
# and not for building the host code.
AC_MSG_NOTICE([Starting OpenCL runtime build system checks])

AC_ARG_ENABLE([debug],
AS_HELP_STRING([--enable-debug], [enable debug build of pocl (default is no)]),
[debug=$enableval],
[debug=no])
if test "x$debug" = "xno"
then
	CXXFLAGS="$CXXFLAGS"
else
	CFLAGS="-O0 -g"
	CXXFLAGS="-O0 -g"
fi

# Checks for programs.
AC_ARG_VAR([CLANG], [Program to compile kernels to bytecode])
AC_ARG_VAR([CLANGFLAGS], [Flags to be passed to CLANG regardless of the target])

AC_CHECK_PROGS([CLANG], [clang])
test -z "$CLANG" && AC_MSG_FAILURE([no clang found in \$PATH])

AC_DEFINE_UNQUOTED([CLANG], ["$CLANG"], [clang executable.])
AC_DEFINE_UNQUOTED([CLANGFLAGS], ["$CLANGFLAGS"], [clang flags.])

AC_LANG([C])
old_CC="$CC"
CC="$CLANG"

AC_CHECK_PROGS([OPT], [opt])
AC_CHECK_PROGS([LLVM_LD], [llvm-ld])
AC_CHECK_PROGS([LLC], [llc])
AC_CHECK_PROGS([LLVM_AS], [llvm-as])
test -z "$OPT" || \
test -z "$LLVM_LD" || \
test -z "$LLC" || \
test -z "$LLVM_AS" && AC_MSG_FAILURE([no llvm required binaries found in \$PATH])

AC_DEFINE_UNQUOTED([LLVM_LD], ["$LLVM_LD"], [LLVM linker executable.])
AC_DEFINE_UNQUOTED([LLC], ["$LLC"], [LLVM compiler executable.])

ARM_CLANG_FLAGS="-D__ARM__DEVICE__"
X86_64_CLANG_FLAGS="-D__X86_64_DEVICE__"
TCE_CLANG_FLAGS="-D__TCE_DEVICE__"

AC_ARG_ENABLE([arm-cortex-a8],
        [AS_HELP_STRING([--enable-arm-cortex-a8],
         [Optimize the ARM OpenCL kernel library for the Cortex A8 instruction set.])],
        [
         ARM_CLANG_FLAGS="$ARM_CLANG_FLAGS -mcpu=cortex-a8"
         ARM_LLC_FLAGS="-relocation-model=pic -mcpu=cortex-a8 -mattr=+neon"
         AC_MSG_NOTICE([optimizing the ARM kernel library for the Cortex A8 instruction set.])
        ]
)

NATIVE_CLANG_FLAGS="$CLANGFLAGS"
NATIVE_LLC_FLAGS="-relocation-model=pic"

AC_CANONICAL_HOST
case $host_os in
  darwin* )
    # MacOS ld outputs useless warnings like
    # ld: warning: -macosx_version_min not specificed, assuming 10.7
    # suppress them with -w.
    SHARED_LD_FLAGS="-bundle -w"  
         ;;
  *)
    SHARED_LD_FLAGS="-shared"
         ;;
esac
case $host_cpu in
  arm*)
    AC_MSG_NOTICE([using the ARM optimized kernel lib for the native device])
    # Assume -lm should not be used. TODO: add an embedded math lib
    # for compiling kernels for ARM envs without usable math libs.
    HOST_ARCH=arm
    # Use the Scratchbox default Clang ARM target by default.
    NATIVE_CLANG_FLAGS="$ARM_CLANG_FLAGS"
    NATIVE_LLC_FLAGS="$ARM_LLC_FLAGS"
    ;;
  x86_64)
    AC_MSG_NOTICE([using the x86_64 optimized kernel lib for the native device])
    HOST_ARCH=x86_64
    NATIVE_CLANG_FLAGS="$X86_64_CLANG_FLAGS"
    SHARED_LD_FLAGS="$SHARED_LD_FLAGS -lm"
    ;;

  tce)
    AC_MSG_NOTICE([using the TCE optimized kernel lib for the native device])
    HOST_ARCH=tce
    NATIVE_CLANG_FLAGS="$TCE_CLANG_FLAGS"
    SHARED_LD_FLAGS="$SHARED_LD_FLAGS -lm"
    ;;

  *)
    AC_MSG_NOTICE([unknown host, using the generic kernel lib])
    SHARED_LD_FLAGS="$SHARED_LD_FLAGS -lm"
    HOST_ARCH=""
    ;;
esac
AC_SUBST([HOST_ARCH])
AC_DEFINE_UNQUOTED([NATIVE_CLANG_FLAGS], ["$NATIVE_CLANG_FLAGS"], 
["Additional flags to pass to the Clang when compiling code for the 'native device'."])
AC_SUBST([NATIVE_CLANG_FLAGS])
AC_DEFINE_UNQUOTED([NATIVE_LLC_FLAGS], ["$NATIVE_LLC_FLAGS"], 
["Additional flags to pass to the llc call of the code generation of the 'native device'."])
AC_SUBST([SHARED_LD_FLAGS])
AC_DEFINE_UNQUOTED([SHARED_LD_FLAGS], ["$SHARED_LD_FLAGS"], [Parameter to compiler to generate loadable module.])

AM_CONDITIONAL([NATIVE_ARM], [test x$HOST_ARCH = xarm])
AM_CONDITIONAL([NATIVE_X86_64], [test x$HOST_ARCH = xx86_64])
AM_CONDITIONAL([NATIVE_TCE], [test x$HOST_ARCH = xtce])

# Use the Scratchbox default Clang ARM target by default.
ARM_CLANG_FLAGS="-ccc-host-triple armv4t-unknown-linux-gnueabihf $ARM_CLANG_FLAGS"
X86_64_CLANG_FLAGS="-ccc-host-triple x86_64-unknown-linux-gnu $X86_64_CLANG_FLAGS"
TCE_CLANG_FLAGS="-ccc-host-triple tce-tut-llvm $TCE_CLANG_FLAGS"

AC_SUBST([ARM_CLANG_FLAGS])
AC_SUBST([X86_64_CLANG_FLAGS])
AC_SUBST([TCE_CLANG_FLAGS])

AC_DEFINE_UNQUOTED([BUILDDIR], ["`pwd`"], [Absolute path of build directory.])

# Checks for libraries.

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_SIZEOF([long])
AC_CHECK_SIZEOF([__fp16])
AC_CHECK_SIZEOF([double])
AC_CHECK_SIZEOF([void *])
AC_CHECK_ALIGNOF([float16], [typedef float float16  __attribute__((__ext_vector_type__(16)));])

CC="$old_CC"

AC_CONFIG_FILES([Makefile
                 pocl.pc
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
                 examples/example2a/Makefile
                 examples/barriers/Makefile
                 examples/forloops/Makefile
                 examples/standalone/Makefile
                 examples/trig/Makefile
                 examples/scalarwave/Makefile
                 examples/loopbarriers/Makefile
                 examples/kernel/Makefile
                 scripts/Makefile
                 tests/Makefile
                 tests/atlocal
                 ])
# lib/kernel/bgp/Makefile
# lib/kernel/ppc/Makefile

AC_OUTPUT
