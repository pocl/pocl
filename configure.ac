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
AM_INIT_AUTOMAKE([foreign silent-rules])
LT_INIT

# check for tools
PKG_PROG_PKG_CONFIG

# check: hwloc version must be 1.0+
PKG_CHECK_MODULES([HWLOC], [hwloc >= 1.0], [
  :
], [
  AC_MSG_ERROR([hwloc 1.0+ is required])
])

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

####################################################################
# Manage optional testsuites

# Option to enable the OpenCL book samples testsuite
AC_ARG_ENABLE([testsuite-opencl-book-samples],
              [AS_HELP_STRING([--enable-testsuite-opencl-book-samples], [checkout OpenCL book sample programs for tests (default no)])],
              [], [enable_testsuite_opencl_book_samples="no"])
if test "$enable_testsuite_opencl_book_samples" = "yes"; then
   AC_CHECK_LIB([glut], [glutInit], [:], [
     AC_MSG_WARN([disabling tests from OpenCL book as glut library is missing])
     enable_testsuite_opencl_book_samples="no"
   ])
  :
fi
if test "$enable_testsuite_opencl_book_samples" = "yes"; then
  AC_CHECK_HEADER([FreeImage.h], [:], [
    AC_MSG_WARN([disabling tests from OpenCL book as FreeImage.h header is missing])
    enable_testsuite_opencl_book_samples="no"
  ])
fi
AM_CONDITIONAL([TEST_SUITE_SAMPLES], [test "$enable_testsuite_opencl_book_samples" = "yes"])

# Option to enable the ViennaCL testsuite
AC_ARG_ENABLE([testsuite-viennacl],
              [AS_HELP_STRING([--enable-testsuite-viennacl], [fetch ViennaCL software for tests (default no)])],
              [], [enable_testsuite_viennacl="no"])

AM_CONDITIONAL([TEST_SUITE_VIENNACL], [test "$enable_testsuite_viennacl" = "yes"])

# Some information for the user
AC_MSG_NOTICE([internal tests are enabled])
if test "$enable_testsuite_opencl_book_samples" = "yes"; then
  AC_MSG_NOTICE([tests from OpenCL book are enabled])
fi
if test "$enable_testsuite_viennacl" = "yes"; then
  AC_MSG_NOTICE([tests from ViennaCL are enabled])
fi
####################################################################

# Option to build as an ICD
AC_ARG_ENABLE([icd],
              [AS_HELP_STRING([--enable-icd], [Build pocl with the ICD extensions (default check)])],
              [icd="$enableval"],
              [icd="check"])
OCL_ICD_available=no
if test x"$icd" != xno; then :
  PKG_CHECK_MODULES([OCL_ICD], [ocl-icd >= 1.3], [
    OCL_ICD_available=yes ], [:])
fi

AS_CASE(["$icd,$OCL_ICD_available"],
  [no,*], [ AC_MSG_NOTICE([disabling ICD extension build as requested]) ],
  [*,yes], [ icd=yes
    AC_MSG_NOTICE([enabling ICD extension build]) ],
  [check,no], [ icd=no
    AC_MSG_NOTICE([disabling ICD extension build (ocd-icd not available)]) ],
  [yes,no], [ AC_MSG_ERROR([cannot build ICD extension without ocd-icd])],
  [AC_MSG_ERROR([Invalid value for --enable-icd])]) 

if test x"$icd" = xyes; then
  AC_DEFINE([BUILD_ICD], [], "Build with ICD")
fi

AM_CONDITIONAL(BUILD_ICD, test "$icd" != "no")

# Option to allow to link a OpenCL program directly against libpocl
# instead of going through an ICD Loader
AC_ARG_ENABLE([direct-linkage],
              [AS_HELP_STRING([--enable-direct-linkage], [Allow OpenCL programs to be linked directly against the pocl library instead of using a ICD Loader (OpenCL library)])],
              [oclsym="$enableval"],
              [oclsym="default"])

AS_CASE(["$oclsym,$icd"],
  [no,no], [ AC_MSG_ERROR([at least, one of ICD or direct-linkage must be enabled]) ],
  [no,yes], [ AC_MSG_NOTICE([disabling direct linkage as requested]) ],
  [default,yes], [ oclsym=no
    AC_MSG_NOTICE([disabling direct linkage as ICD extension is built]) ],
  [default,no], [ oclsym=yes
    AC_MSG_NOTICE([enabling direct linkage as ICD extension is not build]) ],
  [yes,*], [ 
    AC_MSG_NOTICE([enabling direct linkage as requested]) ],
  [AC_MSG_ERROR([Invalid value for --enable-direct-linkage])]) 

if test x"$oclsym" = xyes; then
  AC_DEFINE([DIRECT_LINKAGE], [], "Export OpenCL symbols")
fi

AM_CONDITIONAL(EXPORT_OPENCL_SYMBOLS, test "$oclsym" != "no")

if test x"$oclsym" = xno; then
  PKG_CHECK_MODULES([OPENCL], [OpenCL >= 1.2], [
    OCL_ICD_available=yes
  ], [
    if test -z "$OPENCL_LIBS"; then
      OPENCL_LIBS="-lOpenCL"
    fi
  ])
  OPENCL_CMAKE=''
else
  OPENCL_LIBS='$(top_builddir)/lib/CL/libpocl.la $(PTHREAD_LIBS) $(LD_FLAGS_BIN)'
  OPENCL_CFLAGS='$(PTHREAD_CFLAGS)'
  OPENCL_CMAKE=' \
	-D OPENCL_LIBRARIES:FILEPATH=${abs_top_builddir}/lib/CL/.libs/libpocl.so \
	-D OPENCL_LIBRARY:FILEPATH=${abs_top_builddir}/lib/CL/.libs/libpocl.so \
	-D CMAKE_EXE_LINKER_FLAGS:STRING='${PTHREAD_CFLAGS}' \
	-D OPENCL_INCLUDE_DIRS:PATH=${abs_top_srcdir}/include \
	-D OPENCL_INCLUDE_DIR:PATH=${abs_top_srcdir}/include
  '
fi
AC_SUBST([OPENCL_CMAKE])

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

# Option to install OpenCL header files
# By default, do not install them unless none are found in the system include directories
AC_ARG_ENABLE([install-opencl-headers],
              [AS_HELP_STRING([--enable-install-opencl-headers], [install OpenCL headers. Ones from Kronos should be installed instead.])],
              [ioh="$enableval"],
              [ioh="default"])

if test x"$ioh" = xdefault; then
  AC_CHECK_HEADER([CL/opencl.h], [ioh=no], [ioh=yes])
fi

AM_CONDITIONAL(INSTALL_OPENCL_HEADERS, test "$ioh" = "yes")

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
AC_DEFUN([LLVM_PROG], [
  AC_ARG_VAR([$1], [$3])
  AC_PATH_PROG([$1], [$2], , [$LLVM_BINDIR:$PATH])
  AS_IF([test "x$1" = x], [
    AC_MSG_FAILURE([cannot find LLVM binary '$2' in '$LLVM_BINDIR' nor \$PATH])
  ])
])

LLVM_PROG([CLANG], [clang], [Program to compile kernels to bytecode])

AC_ARG_VAR([CLFLAGS], [Flags to be used when compiling CL sources])

AC_DEFINE_UNQUOTED([CLANG], ["$CLANG"], [clang executable.])
AC_DEFINE_UNQUOTED([CLFLAGS], ["$CLFLAGS"], [CL compiler flags.])

LLVM_PROG([OPT], [opt], [LLVM optimizer])
LLVM_PROG([LLC], [llc], [LLVM static compiler])
LLVM_PROG([LLVM_AS], [llvm-as], [LLVM assembler])
LLVM_PROG([LLVM_RANLIB], [llvm-ranlib], [index generator for LLVM archive])
LLVM_PROG([LLVM_AR], [llvm-ar], [LLVM archiver])
# $CLANG and $LLC are needed by config/xclang
export CLANG
export LLC

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
AC_DEFINE_UNQUOTED([LLVM_LD], ["pocl-llvm-ld"],[LLVM linker binary])
AC_SUBST([LLVM_LD], [pocl-llvm-ld])

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
                 examples/ViennaCL/custom-context.stdout
                 tests/atlocal
                 ocl-vendors/pocl-tests.icd

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
                 examples/standalone/Makefile
                 examples/trig/Makefile
                 examples/scalarwave/Makefile
                 examples/kernel/Makefile
                 examples/opencl-book-samples/Makefile
                 examples/ViennaCL/Makefile
                 scripts/Makefile
                 tests/Makefile
                 tests/regression/Makefile
                 tests/workgroup/Makefile
                 tests/runtime/Makefile
                 tests/tce/Makefile
                 tests/tce/ttasim/Makefile
                 tools/llvm-ld/Makefile
                 ])

AC_OUTPUT

