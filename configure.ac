#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
# 
# Copyright (c) 2011-2012 Universidad Rey Juan Carlos
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
AC_INIT([pocl], [0.7-pre], [pocl-devel@lists.sourceforge.net])
AC_CONFIG_SRCDIR([config.h.in])
AC_CONFIG_HEADER([config.h])
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_TESTDIR([tests])
AC_CANONICAL_TARGET
AC_CANONICAL_HOST
AM_INIT_AUTOMAKE([foreign])
LT_INIT

AC_SEARCH_LIBS([hwloc_topology_init], [hwloc], [], [AC_MSG_ERROR([hwloc is required])])

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX
AC_PROG_LN_S

# The extra LDFLAGS that must be given to link final binaries against pocl
LD_FLAGS_BIN=""

AC_ARG_VAR([LLVM_CONFIG], [Program used to retrieve LLVM options and binaries])
AC_PATH_PROG([LLVM_CONFIG], [llvm-config])
test -z "$LLVM_CONFIG" && AC_MSG_FAILURE([no llvm-config found in \$PATH])

LLVM_VERSION=`$LLVM_CONFIG --version`
LLVM_BINDIR=`$LLVM_CONFIG --bindir`

case "$LLVM_VERSION" in
     3.1*)
     AC_DEFINE([LLVM_3_1], [], "Using LLVM 3.1")
   ;;
     3.2svn)
     AC_DEFINE([LLVM_3_2], [], "Using LLVM svn (upcoming 3.2)")     
     AC_DEFINE([LLVM_SVN], [], "Using LLVM svn (upcoming 3.2)")     
   ;;   
     *)
   AC_MSG_ERROR(
   [
Unsupported LLVM version. Please use LLVM 3.1.
   ])
   LLVM_VERSION=
   ;;
esac

AC_ARG_ENABLE([region-allocator],
[AS_HELP_STRING([--disable-region-allocator], 
 [Do not use a custom OpenCL optimized region-based memory allocator but allocate buffers with malloc directly.])],
[],
[AC_DEFINE([CUSTOM_BUFFER_ALLOCATOR], [], "Use a custom buffer allocator")])

# Option to build as an ICD
AC_ARG_ENABLE([icd],
              [AS_HELP_STRING([--disable-icd], [Do not build pocl with the ICD extensions])],
              [icd="$enableval"],
              [icd="yes"])
if test x"$icd" != xno; then :
  AC_DEFINE([BUILD_ICD], [], "Build with ICD")
fi

AC_SUBST(icd)
AM_CONDITIONAL(INSTALL_ICD, test "$icd" != "no")

#AC_LANG([C++])
ACX_PTHREAD()

LD_FLAGS_BIN="$LD_FLAGS_BIN $PTHREAD_LDFLAGS"

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

# look for the TCE library (needed for the ttasim device driver)

AC_PATH_PROG([TCE_CONFIG], [tce-config])

if test ! -z "$TCE_CONFIG"; then

AC_PATH_PROG([TCECC], [tcecc])

AC_LANG([C++])

AC_LANG_CONFTEST(
[AC_LANG_PROGRAM(
  [[#include <Application.hh>]],
  [Application::initialize()])
])

  if $CXX conftest.cpp `$TCE_CONFIG --libs --cxxflags --includes` -o /dev/null;
  then
        TCE_LIBRARY=ok
  fi

fi

if test -z "$TCECC" || \
   test -z "$TCE_CONFIG" || \
   test -z "$TCE_LIBRARY"; then

AC_MSG_NOTICE([TCE installation not found. The ttasim device won't be built.])

else

AC_MSG_NOTICE([TCE installation found. The ttasim device will be built.])
AC_DEFINE([TCE_AVAILABLE], [1], [Defined to 1 if TCE libraries and tools are available])
AC_SUBST([TCE_AVAILABLE], ["1"])
LD_FLAGS_BIN="$LD_FLAGS_BIN `$TCE_CONFIG --libs`"
fi

AM_CONDITIONAL([TCE_AVAILABLE], test ! -z $TCE_AVAILABLE)

# libtool dynamic library handling library
AC_SEARCH_LIBS([lt_dlsym], [ltdl], [], [
AC_MSG_ERROR([unable to find the libtool dl library (usually libltdl-dev)])
])

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

AC_DEFINE([HAVE_GL_GL_H], [], [OpenGL])
AC_DEFINE([HAVE_OPENGL_OPENGL_H], [], [OpenGL on Apple])

AC_CHECK_HEADERS([GL/gl.h OpenGL/OpenGL.h], [opengl_found="ok"; break])

test -z "$opengl_found" && AC_MSG_FAILURE([Please install an OpenGL implementation with gl.h or OpenGL.h. It is required by the C++ bindings.])

# Checks for programs.
AC_ARG_VAR([CLANG], [Program to compile kernels to bytecode])
AC_PATH_PROG([CLANG], [clang], , [$LLVM_BINDIR:$PATH])
test -z "$CLANG" && AC_MSG_FAILURE([no clang found in \$PATH])

AC_ARG_VAR([CLFLAGS], [Flags to be used when compiling CL sources])

AC_DEFINE_UNQUOTED([CLANG], ["$CLANG"], [clang executable.])
AC_DEFINE_UNQUOTED([CLFLAGS], ["$CLFLAGS"], [CL compiler flags.])

AC_ARG_VAR([OPT], [LLVM optimizer])
AC_PATH_PROG([OPT], [opt], , ["$LLVM_BINDIR:$PATH"])
AC_ARG_VAR([LLVM_LD], [LLVM linker])
AC_PATH_PROG([LLVM_LD], [llvm-ld], , ["$LLVM_BINDIR:$PATH"])
AC_ARG_VAR([LLC], [LLVM static compiler])
AC_PATH_PROG([LLC], [llc], , ["$LLVM_BINDIR:$PATH"])
AC_ARG_VAR([LLVM_AS], [LLVM assembler])
AC_PATH_PROG([LLVM_AS], [llvm-as], , ["$LLVM_BINDIR:$PATH"])
test -z "$OPT" || \
test -z "$LLVM_LD" || \
test -z "$LLC" || \
test -z "$LLVM_AS" && AC_MSG_FAILURE([no llvm required binaries found in $LLVM_BINDIR nor \$PATH])
# $CLANG and $LLC are needed by config/xclang
export CLANG
export LLC

AC_DEFINE_UNQUOTED([LLVM_LD], ["$LLVM_LD"], [LLVM linker executable.])
AC_DEFINE_UNQUOTED([LLC], ["$LLC"], [LLVM compiler executable.])

AC_CONFIG_FILES([lib/kernel/arm/Makefile
                 lib/kernel/tce/Makefile
                 lib/kernel/x86_64/Makefile])

AC_ARG_ENABLE([arm-cortex-a8],
        [AS_HELP_STRING([--enable-arm-cortex-a8],
         [Optimize the ARM OpenCL kernel library for the Cortex A8 instruction set.])],
        [
         ARM_CLANG_FLAGS="$ARM_CLANG_FLAGS -mcpu=cortex-a8"
         ARM_LLC_FLAGS="-relocation-model=pic -mcpu=cortex-a8 -mattr=+neon"
         AC_MSG_NOTICE([optimizing the ARM kernel library for the Cortex A8 instruction set.])
        ]
)

case $target_cpu in
  arm*)   
    TARGET_CLANG_FLAGS="$TARGET_CLANG_FLAGS $ARM_CLANG_FLAGS"
    TARGET_LLC_FLAGS="$TARGET_LLC_FLAGS $ARM_LLC_FLAGS"
    ;;
esac

HOST_LLC_FLAGS="-relocation-model=pic"
HOST_LD_FLAGS="-shared"

case $host_os in
  darwin* )
    # MacOS ld outputs useless warnings like
    # ld: warning: -macosx_version_min not specificed, assuming 10.7
    # suppress them with -w.
    HOST_LD_FLAGS="-bundle -w"  
    ICD_LD_FLAGS="-single_module"
    ;;
  *gnu )
    ICD_LD_FLAGS="-Wl,-Bsymbolic"
    ;;  
esac
AC_SUBST([ICD_LD_FLAGS])
case $host_cpu in
  arm*)
    AC_MSG_NOTICE([using the ARM optimized kernel lib for the native device])
    # Assume -lm should not be used. TODO: add an embedded math lib
    # for compiling kernels for ARM envs without usable math libs.
    HOST_CLANG_FLAGS="$HOST_CLANG_FLAGS $ARM_CLANG_FLAGS"
    HOST_LLC_FLAGS="$ARM_LLC_FLAGS"
    ;;
  x86_64)
    AC_MSG_NOTICE([using the x86_64 optimized kernel lib for the native device])
    HOST_LD_FLAGS="$HOST_LD_FLAGS -lm"
    ;;
  tce)
    AC_MSG_NOTICE([using the TCE optimized kernel lib for the native device])
    HOST_LD_FLAGS="$HOST_LD_FLAGS -lm"
    ;;
  *)
    AC_MSG_NOTICE([unknown host, using the generic kernel lib])
    HOST_LD_FLAGS="$HOST_LD_FLAGS -lm"
    ;;
esac

AC_SUBST([TARGET], [$target])
AC_SUBST([TARGET_CPU], [$target_cpu])
AC_ARG_VAR([TARGET_CLANG_FLAGS],
           [Parameters to for target compilation.])
AC_DEFINE_UNQUOTED([TARGET_CLANG_FLAGS],
                   ["$TARGET_CLANG_FLAGS"],
                   [Parameters to for target compilation.])
AC_SUBST([TARGET_LLC_FLAGS])
AC_DEFINE_UNQUOTED([TARGET_LLC_FLAGS],
                   ["$TARGET_LLC_FLAGS"], 
                   [Parameters to llc for code generation in the target.])

AC_SUBST([HOST], [$host])
AC_SUBST([HOST_CPU], [$host_cpu])
AC_ARG_VAR([HOST_CLANG_FLAGS],
           [Parameters to for host compilation.])
AC_DEFINE_UNQUOTED([HOST_CLANG_FLAGS],
                   ["$HOST_CLANG_FLAGS"],
                   [Parameters to for host compilation.])
AC_SUBST([HOST_LLC_FLAGS])
AC_DEFINE_UNQUOTED([HOST_LLC_FLAGS],
                   ["$HOST_LLC_FLAGS"], 
                   [Parameters to llc for code generation in the host.])

AC_SUBST([HOST_LD_FLAGS])
AC_DEFINE_UNQUOTED([HOST_LD_FLAGS],
                   ["$HOST_LD_FLAGS"],
                   [Parameter to compiler to generate loadable module.])

AC_CONFIG_FILES(lib/kernel/${TARGET}/Makefile:lib/kernel/Makefile.target.in)
AC_CONFIG_FILES([lib/kernel/${HOST}/Makefile:lib/kernel/Makefile.host.in])

AC_DEFINE_UNQUOTED([BUILDDIR], ["`pwd`"], [Absolute path of build directory.])
AC_DEFINE_UNQUOTED([SRCDIR], ["`cd $srcdir && pwd`"], [Root for the project sources.])

if test x"$prefix" = xNONE; then
  prefix=/usr/local
fi
AC_DEFINE_UNQUOTED([POCL_INSTALLATION_ROOT],["$prefix"],[The installation directory])
AC_DEFINE_UNQUOTED([POCL_INSTALLED_DATA],["$prefix/share/pocl"],[The installation directory])
AC_SUBST([POCL_INSTALLED_DATA], $prefix/share/pocl)

# Checks for libraries.

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.
AC_LANG([C])

old_CC="$CC"
old_CPPFLAGS="$CPPFLAGS"
old_CFLAGS="$CFLAGS"
old_LDFLAGS="$LDFLAGS"
CC=$srcdir/config/xclang
CPPFLAGS=""
LDFLAGS=""

AC_CONFIG_FILES([include/arm/types.h
                 include/tce/types.h
                 include/x86_64/types.h])

AC_MSG_NOTICE([Checking target compiler characteristics])
CFLAGS="-h $target"
AC_CHECK_SIZEOF([long])
AC_SUBST([TARGET_SIZEOF_LONG], [$ac_cv_sizeof_long])
AC_CHECK_SIZEOF([__fp16])
AC_SUBST([TARGET_SIZEOF_HALF], [$ac_cv_sizeof___fp16])
AC_CHECK_SIZEOF([double])
AC_SUBST([TARGET_SIZEOF_DOUBLE], [$ac_cv_sizeof_double])
AC_CHECK_SIZEOF([void *])
AC_SUBST([TARGET_SIZEOF_VOID_P], [$ac_cv_sizeof_void_p])

AC_CONFIG_FILES([include/${TARGET}/types.h:include/types.target.in])

AS_UNSET([ac_cv_sizeof_long])
AS_UNSET([ac_cv_sizeof___fp16])
AS_UNSET([ac_cv_sizeof_double])
AS_UNSET([ac_cv_sizeof_void_p])

AC_MSG_NOTICE([Checking host compiler characteristics])
CFLAGS="-h $host"
AC_CHECK_SIZEOF([long])
AC_SUBST([HOST_SIZEOF_LONG], [$ac_cv_sizeof_long])
AC_CHECK_SIZEOF([__fp16])
AC_SUBST([HOST_SIZEOF_HALF], [$ac_cv_sizeof___fp16])
AC_CHECK_SIZEOF([double])
AC_SUBST([HOST_SIZEOF_DOUBLE], [$ac_cv_sizeof_double])
AC_CHECK_SIZEOF([void *])
AC_SUBST([HOST_SIZEOF_VOID_P], [$ac_cv_sizeof_void_p])

AC_CONFIG_FILES([include/${HOST}/types.h:include/types.host.in])

AC_CHECK_ALIGNOF([float16], [typedef float float16  __attribute__((__ext_vector_type__(16)));])
AC_CHECK_ALIGNOF([double16], [typedef double double16  __attribute__((__ext_vector_type__(16)));])

# Seems to be unused
#AC_MSG_CHECKING([whether __fp16 can be converted to float])
#AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[__fp16 x; float y;]],
#                                   [[y=x]])],
#                  [
#                    AC_MSG_RESULT([yes])
#                    AC_DEFINE_UNQUOTED([CONVERT___FP16_FLOAT], 1, [Define to 1 if __fp16 can be converted to float.])
#                  ],
#                  [
#                    AC_MSG_RESULT([no])
#                    AC_DEFINE_UNQUOTED([CONVERT___FP16_FLOAT], 0, [Define to 1 if __fp16 can be converted to float.])
#                  ]
#                 )

CC="$old_CC"
CPPFLAGS="$old_CPPFLAGS"
CFLAGS="$old_CFLAGS"
LDFLAGS="$old_LDFLAGS"

AC_C_BIGENDIAN

# The libtool library version string to use (passed to --version-info).
# 0:0:0 == 0.6
# 1:0:0 == 0.7
AC_SUBST([LIB_VERSION], ["1:0:0"])
AC_SUBST([LD_FLAGS_BIN])

AC_CONFIG_FILES([Makefile
                 pocl.pc
                 pocl.icd
                 include/Makefile
                 include/CL/Makefile
                 include/OpenCL/Makefile
                 lib/Makefile
                 lib/CL/Makefile
                 lib/CL/devices/Makefile
                 lib/CL/devices/pthread/Makefile
                 lib/CL/devices/basic/Makefile
                 lib/CL/devices/tce/Makefile
                 lib/CL/devices/tce/ttasim/Makefile
                 lib/CL/devices/topology/Makefile
                 lib/llvmopencl/Makefile
                 lib/kernel/Makefile
                 lib/poclu/Makefile
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
                 examples/opencl-book-samples/Makefile
                 scripts/Makefile
                 tests/Makefile
                 tests/atlocal
                 tests/regression/Makefile
                 tests/regression/loop-phi-replication/Makefile
                 tests/regression/locals/Makefile
                 tests/regression/barrier-between-for-loops/Makefile
                 tests/regression/simple-for-with-a-barrier/Makefile
                 tests/regression/multi-level-loops-with-barriers/Makefile
                 tests/regression/for-with-var-iteration-count/Makefile
                 tests/regression/early-return/Makefile
                 tests/regression/id-dependent-computation/Makefile
                 tests/regression/structs-as-args/Makefile
                 tests/runtime/Makefile
                 tests/tce/Makefile
                 tests/tce/ttasim/Makefile
                 ])

AC_OUTPUT

