##                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
# 
# Copyright (c) 2011-2013 Universidad Rey Juan Carlos and
#                         Pekka Jääskeläinen / Tampere University of Technology
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

AC_PREREQ([2.64])
AC_INIT([pocl], [0.9-pre], [pocl-devel@lists.sourceforge.net])
AC_CONFIG_SRCDIR([config.h.in])
AC_CONFIG_HEADER([config.h])
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_TESTDIR([tests])
AC_CANONICAL_TARGET
AC_CANONICAL_HOST
AM_INIT_AUTOMAKE([foreign])
AM_SILENT_RULES([AM_DEFAULT_VERBOSITY=0])

AM_CONDITIONAL([DARWIN_HOST_OS], false)
case $host_os in
  darwin* )
    AM_CONDITIONAL([DARWIN_HOST_OS], true)
    AC_SUBST([LIBRARY_SUFFIX], [".dylib"] )
         ;;
  *)
    AC_SUBST([LIBRARY_SUFFIX], [".so"] )
         ;;
esac

# check for tools
PKG_PROG_PKG_CONFIG

#Exit if pkg-config is not found.
PKG_CHECK_EXISTS([pkg-config],[],[AC_MSG_ERROR([pkg-config not found])])

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

m4_ifdef([AM_PROG_AR], [AM_PROG_AR])

LT_INIT

# The extra LDFLAGS that must be given to link final binaries against pocl
LD_FLAGS_BIN=""

AC_ARG_VAR([LLVM_CONFIG], [Program used to retrieve LLVM options and binaries])
AC_PATH_PROGS([LLVM_CONFIG], [llvm-config llvm-config-3.2 llvm-config-3.3])
test -z "$LLVM_CONFIG" && AC_MSG_FAILURE([no llvm-config found in \$PATH])

LLVM_VERSION=`$LLVM_CONFIG --version`
LLVM_BINDIR=`$LLVM_CONFIG --bindir`
LLVM_LIBDIR=`$LLVM_CONFIG --libdir`

AC_SUBST([LLVM_VERSION], [$LLVM_VERSION])

case "$LLVM_VERSION" in
     3.2*)
     AC_DEFINE([LLVM_3_2], [], "Using LLVM 3.2")
   ;;
     3.3*)
     AC_DEFINE([LLVM_3_3], [], "Using LLVM 3.3 RC")     
   ;; 
     3.4*)
     AC_DEFINE([LLVM_3_4], [], "Using LLVM svn (upcoming 3.4)")     
     AC_DEFINE([LLVM_SVN], [], "Using LLVM svn (upcoming 3.4)")     
   ;;   
     *)
   AC_MSG_ERROR(
   [
Unsupported LLVM version. Please use LLVM 3.2 or later.
   ])
   LLVM_VERSION=
   ;;
esac

LLVM_SHARED_LIB_FILE=$LLVM_LIBDIR/libLLVM-$LLVM_VERSION$LIBRARY_SUFFIX

if test ! -e $LLVM_SHARED_LIB_FILE;
then
        AC_MSG_WARN([
$LLVM_SHARED_LIB_FILE not found. 
Recompile and install LLVM after ./configure --enable-shared.
Read the INSTALL file for details.])
fi

LLVM_RTTI_CHECK=$($LLVM_CONFIG --cxxflags | grep -i "\-fno-rtti")

if test x"$LLVM_RTTI_CHECK" != x; then
AC_MSG_WARN([Your LLVM was not built with RTTI. 
You should rebuild LLVM with 'make REQUIRES_RTTI=1'. 
See the INSTALL file for more information.])
fi

AC_ARG_ENABLE([region-allocator],
[AS_HELP_STRING([--disable-region-allocator], 
 [Do not use a custom OpenCL optimized region-based memory allocator but allocate buffers with malloc directly.])],
[],
[AC_DEFINE([CUSTOM_BUFFER_ALLOCATOR], [], "Use a custom buffer allocator")])

####################################################################
# Manage optional testsuites

# Option to choose external testsuites
AC_ARG_ENABLE([testsuites],
	      [AS_HELP_STRING([--enable-testsuites=suite1,suite2,...],
          [choose enabled external project testsuites (all,opencl-book-samples,ViennaCL,Rodinia,Parboil,amd,VexCL])],
	      [],[enable_testsuites=check])

enable_testsuite_opencl_book_samples=no
enable_testsuite_viennacl=no
enable_testsuite_rodinia=no
enable_testsuite_parboil=no
enable_testsuite_amd=no
enable_testsuite_vexcl=no
if test x"$enable_testsuites" = xcheck ; then
  if test -d "$srcdir/examples/opencl-book-samples/checkout" ; then
    enable_testsuite_opencl_book_samples=yes
  fi
  if test -f "$srcdir/examples/ViennaCL/ViennaCL-1.3.1-src.tar.gz" ; then
    enable_testsuite_viennacl=yes
  fi
  if test -f "$srcdir/examples/Rodinia/rodinia_2.0.1.tar.gz" ; then
    enable_testsuite_rodinia=yes
  fi
  if test -f "$srcdir/examples/Parboil/pb2.5benchmarks.tgz" -a \
          -f "$srcdir/examples/Parboil/pb2.5datasets_standard.tgz" -a \
          -f "$srcdir/examples/Parboil/pb2.5driver.tgz"; then
    enable_testsuite_parboil=yes
  fi
  if test -f "$srcdir/examples/AMD/AMD-APP-SDK-v2.8-RC-lnx64.tgz" ; then
    enable_testsuite_amd=yes
  fi
  if test -f "$srcdir/examples/VexCL/vexcl/README.md" ; then
    enable_testsuite_vexcl=yes
  fi
fi
AS_CASE([,"$enable_testsuites",],
  [*,all,*|*,yes,*], [enable_testsuites="opencl-book-samples,ViennaCL,Rodinia,Parboil,amd,VexCL"],
  [*,no,*], [enable_testsuites=""]
)
AS_CASE([,"$enable_testsuites",],
  [*,opencl-book-samples,*|*,samples,*], [
    enable_testsuite_opencl_book_samples=yes
  ])
AS_CASE([,"$enable_testsuites",],
  [*,ViennaCL,*|*,viennacl,*], [
    enable_testsuite_viennacl=yes
  ])
AS_CASE([,"$enable_testsuites",],
  [*,Rodinia,*|*,rodinia,*], [
    enable_testsuite_rodinia=yes
  ])
AS_CASE([,"$enable_testsuites",],
  [*,Parboil,*|*,parboil,*], [
    enable_testsuite_parboil=yes
  ])
AS_CASE([,"$enable_testsuites",],
  [*,AMD,*|*,amd,*], [
    enable_testsuite_amd=yes
  ])
AS_CASE([,"$enable_testsuites",],
  [*,VexCL,*|*,vexcl,*], [
    enable_testsuite_vexcl=yes
  ])

if test "$enable_testsuite_opencl_book_samples" = "yes"; then
  # Check for software required for opencl book samples
  AC_CHECK_LIB([glut], [glutInit], [:], [
    AC_MSG_WARN([disabling tests from OpenCL book as glut library is missing])
    enable_testsuite_opencl_book_samples="no"
  ])
  :
  AC_CHECK_HEADER([FreeImage.h], [:], [
    AC_MSG_WARN([disabling tests from OpenCL book as FreeImage.h header is missing])
    enable_testsuite_opencl_book_samples="no"
  ])
fi

if test "$enable_testsuite_rodinia" = "yes" ; then
  # Check for software and sources for Rodinia
  if ! test -f "$srcdir/examples/Rodinia/rodinia_2.0.1.tar.gz" ; then
    AC_MSG_WARN([disabling Rodinia testsuite as rodinia_2.0.1.tar.gz is not in '$srcdir/examples/Rodinia'. You must download it yourself from http://lava.cs.virginia.edu/Rodinia/download.htm after registration.])
    enable_testsuite_rodinia=no
  fi
fi

if test "$enable_testsuite_parboil" = "yes" ; then
  # Check for software and sources for Parboil
  if ! test -f "$srcdir/examples/Parboil/pb2.5benchmarks.tgz" -a \
          -f "$srcdir/examples/Parboil/pb2.5datasets_standard.tgz" -a \
          -f "$srcdir/examples/Parboil/pb2.5driver.tgz"; then
    AC_MSG_WARN([disabling Parboil testsuite as one of the required packages not in '$srcdir/examples/Parboil'. You must download it yourself from http://impact.crhc.illinois.edu/parboil.aspx.])
    enable_testsuite_parboil=no
  fi
fi

if test "$enable_testsuite_amd" = "yes" ; then
  # Check for software and sources for AMD APP SDK
  if ! test -f "$srcdir/examples/AMD/AMD-APP-SDK-v2.8-RC-lnx64.tgz"; then
    AC_MSG_WARN([disabling AMD APP SDK testsuite as the 2.8 lnx64 package not in '$srcdir/examples/AMD'.])
    enable_testsuite_amd=no
  fi
  PKG_CHECK_MODULES([SDL], [sdl >= 1.2], [:], 
    [ AC_MSG_WARN([pkg-config could not find libSDL, AMD APP SDK testsuite is not used ]) 
      enable_testsuite_amd=no
    ])
fi

if test "$enable_testsuite_vexcl" = "yes" ; then
  # Check for software and sources for VexCL
  if ! test -f "$srcdir/examples/VexCL/vexcl/README.md"; then
    AC_MSG_WARN([disabling VexCL testsuite as the git branch is not in '$srcdir/examples/VexCL/vexcl'.])
    enable_testsuite_vexcl=no
  fi
fi

#All ViennaCL tests use doubles, thus they fail on these platforms
case $host_cpu in
  arm* | powerpc* | x86 | i?86 )
    AC_MSG_WARN([Looks like this host doesn't support doubles. Disabling all ViennaCL tests])
    enable_testsuite_viennacl=no
    ;;
esac

AM_CONDITIONAL([TEST_SUITE_SAMPLES], [test "$enable_testsuite_opencl_book_samples" = "yes"])
AM_CONDITIONAL([TEST_SUITE_VIENNACL], [test "$enable_testsuite_viennacl" = "yes"])
AM_CONDITIONAL([TEST_SUITE_RODINIA], [test "$enable_testsuite_rodinia" = "yes"])
AM_CONDITIONAL([TEST_SUITE_PARBOIL], [test "$enable_testsuite_parboil" = "yes"])
AM_CONDITIONAL([TEST_SUITE_AMD], [test "$enable_testsuite_amd" = "yes"])
AM_CONDITIONAL([TEST_SUITE_VEXCL], [test "$enable_testsuite_vexcl" = "yes"])

# Some information for the user
AC_MSG_NOTICE([internal tests are enabled])
POAT_TESTSUITES=""
if test "$enable_testsuite_opencl_book_samples" = "yes"; then
  AC_MSG_NOTICE([tests from OpenCL book are enabled])
  POAT_TESTSUITES="$POAT_TESTSUITES samples"
else
  AC_MSG_NOTICE([tests from OpenCL book are disabled])
fi
if test "$enable_testsuite_viennacl" = "yes"; then
  AC_MSG_NOTICE([tests from ViennaCL are enabled])
  POAT_TESTSUITES="$POAT_TESTSUITES viennacl"
else
  AC_MSG_NOTICE([tests from ViennaCL are disabled])
fi
if test "$enable_testsuite_rodinia" = "yes"; then
  AC_MSG_NOTICE([tests from Rodinia are enabled])
  POAT_TESTSUITES="$POAT_TESTSUITES rodinia"
else
  AC_MSG_NOTICE([tests from Rodinia are disabled])
fi
if test "$enable_testsuite_parboil" = "yes"; then
  AC_MSG_NOTICE([tests from Parboil are enabled])
  POAT_TESTSUITES="$POAT_TESTSUITES parboil"
else
  AC_MSG_NOTICE([tests from Parboil are disabled])
fi
if test "$enable_testsuite_amd" = "yes"; then
  AC_MSG_NOTICE([tests from AMD APP SDK are enabled])
  POAT_TESTSUITES="$POAT_TESTSUITES amd"
else
  AC_MSG_NOTICE([tests from AMD APP SDK are disabled])
fi
if test "$enable_testsuite_vexcl" = "yes"; then
  AC_MSG_NOTICE([tests from VexCL are enabled])
  POAT_TESTSUITES="$POAT_TESTSUITES vexcl"
else
  AC_MSG_NOTICE([tests from VexCL are disabled])
fi



AC_SUBST([POAT_TESTSUITES])
####################################################################

# Option to build as an ICD
AC_ARG_ENABLE([icd],
              [AS_HELP_STRING([--enable-icd], [Build pocl with the ICD extensions (default yes)])],
              [icd="$enableval"],
              [icd="yes"])
OCL_ICD_available=no
if test x"$icd" != xno; then :
  # In fact, ocl-icd >= 1.4 is required. However, Debian wheezy will
  # have ocl-icd 1.3 *with the required fixes*
  # So, lets hope Debian wheezy adds a minor version number to the bugfixed ocl-icd.
  PKG_CHECK_MODULES([OCL_ICD], [ocl-icd >= 1.3], 
    [ OCL_ICD_available=yes 
      AC_DEFINE([HAVE_OCL_ICD], [1], [Defined to 1 if ocl-icd is found on system]) ],
    [:])
  # In case pkg-config could not find ocl-icd above, try the system default.
  # This test will fail if e.g. ocl-icd is installed to a non-standard location,
  # so skip it if pkg-config succeeded. There doesn't seem to be
  # a symbol that is available only in the ICD loaders. Thus,
  # this check is passed also in case libOpenCL is the implementation,
  # not the loader.
  if test x"$OCL_ICD_available" != xyes; then :
    AC_CHECK_LIB([OpenCL], [clGetPlatformIDs], [:], [
      AC_MSG_ERROR([Could not find an ICD loader. Please install one or add --disable-icd (see README for details).])
      ])
  fi
fi

AS_CASE(["$icd,$OCL_ICD_available"],
  [no,*], [ AC_MSG_NOTICE([disabling ICD extension build as requested]) ],
  [*,yes], [ icd=yes
    AC_MSG_NOTICE([enabling ICD extension build]) ],
  [check,no], [ icd=no
    AC_MSG_NOTICE([disabling ICD extension build (ocl-icd not available)]) ],
  [yes,no], [AC_MSG_WARN([enabling ICD, but the testsuite might not work! (ocl-icd not available)])],
  [AC_MSG_ERROR([Invalid value for --enable-icd])]) 

if test x"$icd" = xyes; then
  AC_DEFINE([BUILD_ICD], [], "Build with ICD")
fi

AM_CONDITIONAL(BUILD_ICD, test "$icd" = "yes")

# Option to allow to link a OpenCL program directly against libpocl
# instead of going through an ICD Loader
AC_ARG_ENABLE([direct-linkage],
              [AS_HELP_STRING([--enable-direct-linkage], [Allow OpenCL programs to be linked directly against the pocl library instead of using a ICD Loader (OpenCL library)])],
              [oclsym="$enableval"],
              [oclsym="default"])

AS_CASE(["$oclsym,$icd"],
  [no,no], [ AC_MSG_ERROR([at least, one of ICD or direct-linkage must be enabled]) ],
  [no,yes], [ AC_MSG_NOTICE([disabling direct linkage as requested]) ],
  [default,*], [ oclsym=yes
    AC_MSG_NOTICE([enabling direct linkage in pocl library]) ],
  [yes,*], [ 
    AC_MSG_NOTICE([enabling direct linkage as requested]) ],
  [AC_MSG_ERROR([Invalid value for --enable-direct-linkage])]) 

if test x"$oclsym" = xyes; then
  AC_DEFINE([DIRECT_LINKAGE], [], "Export OpenCL symbols")
fi

AM_CONDITIONAL(EXPORT_OPENCL_SYMBOLS, test "$oclsym" = "yes")

# Option to run tests through the OpenCL ICD loader
# Note that running test *in tree* requires the ocl-icd loader, not any ICD
# loader
AC_ARG_ENABLE([tests-with-icd],
              [AS_HELP_STRING([--enable-tests-with-icd], [run tests through the ICD Loader (requires the ocl-icd loader)])],
              [tests_with_icd="$enableval"],
              [tests_with_icd="default"])
AS_CASE(["$tests_with_icd,$icd,$oclsym"],
  [yes,no,*], [ AC_MSG_ERROR([Cannot use ICD for tests when not building it]) ],
  [no,*,no], [ AC_MSG_ERROR([Cannot avoid ICD for tests when direct-linkage is disabled]) ],
  [default,yes,*], [ tests_with_icd=yes
    AC_MSG_NOTICE([tests will be run through the ICD]) ],
  [default,*,yes], [ tests_with_icd=no
    AC_MSG_NOTICE([tests will use POCL library directly]) ],
  [AC_MSG_ERROR([Invalid value for --enable-tests-with-icd])]) 

if test x"$tests_with_icd" = xyes; then
  PKG_CHECK_MODULES([OPENCL], [OpenCL >= 1.2], 
  [], 
  [
    AC_MSG_WARN([Tests through ICD loader enabled, but OpenCL.pc file installed by ocl-icd seems missing])
    if test -z "$OPENCL_LIBS"; then
      OPENCL_LIBS="-lOpenCL"
    fi
  ])
  OPENCL_CMAKE=''
  OPENCL_EXTLIBS="$OPENCL_LIBS"
else
  OPENCL_LIBS='$(abs_top_builddir)/lib/CL/libpocl.la $(PTHREAD_LIBS) $(LD_FLAGS_BIN)'
  OPENCL_EXTLIBS='-L$(abs_top_builddir)/lib/CL/.libs -lpocl -Xlinker -rpath=${abs_top_builddir}/lib/CL/.libs $(PTHREAD_LIBS) $(LD_FLAGS_BIN)'
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
AC_SUBST([OPENCL_EXTLIBS])

####################################################################
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
  ioh=yes
  case $host_os in
    darwin* )
      AC_CHECK_HEADER([OpenCL/opencl.h],[ioh=no])
      ;;
    * )
      AC_CHECK_HEADER([CL/opencl.h],[ioh=no])
      ;;
  esac
fi

AM_CONDITIONAL(INSTALL_OPENCL_HEADERS, test "$ioh" = "yes")

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.
#AC_FUNC_MALLOC

# libtool dynamic library handling library
old_LIBS="$LIBS"
LIBS=
AC_SEARCH_LIBS([lt_dlsym], [ltdl], [], [
AC_MSG_ERROR([unable to find the libtool dl library (usually libltdl-dev)])
])
LTDL_LIBS="$LIBS"
AC_SUBST([LTDL_LIBS])
LIBS="$old_LIBS"

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
	CFLAGS="$CXXFLAGS -O0 -g"
	CXXFLAGS="$CXXFLAGS -O0 -g"
fi

AC_DEFINE([HAVE_GL_GL_H], [], [OpenGL])
AC_DEFINE([HAVE_OPENGL_OPENGL_H], [], [OpenGL on Apple])

AC_CHECK_HEADERS([GL/gl.h OpenGL/OpenGL.h], [opengl_found="ok"; break])

test -z "$opengl_found" && AC_MSG_FAILURE([Please install an OpenGL implementation with gl.h or OpenGL.h. It is required by the C++ bindings.])

PKG_CHECK_MODULES([GLEW], [glew], 
  [
    AC_DEFINE([HAVE_GLEW], [],
        [Defined if The OpenGL Extension Wrangler library is found])
  ],
  [AC_MSG_NOTICE([libGLEW not found. A few tests will not work])])

# Checks for programs.
AC_DEFUN([LLVM_PROG], [
  AC_ARG_VAR([$1], [$3])
  AC_PATH_PROGS([$1], [$2 $2-$LLVM_VERSION], , [$LLVM_BINDIR:$PATH])
  AS_IF([test -z "$$1"], [
    AC_MSG_FAILURE([cannot find LLVM binary '$2' in '$LLVM_BINDIR' nor \$PATH])
  ])
])

AC_DEFUN([LLVM_PROG_WARN], [
  AC_ARG_VAR([$1], [$3])
  AC_PATH_PROGS([$1], [$2 $2-$LLVM_VERSION], , [$LLVM_BINDIR:$PATH])
  AS_IF([test -z "$$1"], [
    AC_MSG_WARN([cannot find LLVM binary '$2' in '$LLVM_BINDIR' nor \$PATH])
  ])
])

LLVM_PROG([CLANG], [clang], [Program to compile kernels to bytecode])

# Check for a functioning clang++. It is needed to compile faster math 
# builtins from vecmathlib.
LLVM_PROG_WARN([CLANGXX], [clang++])

if test ! -z "$CLANGXX"; then
# Test that clang++ can compile a simple C++ program.

: ${CLANGXX_FLAGS:="-std=c++11"}

AC_LANG([C++])

old_CXX=$CXX
CXX=$CLANGXX
old_CXXFLAGS=$CXXFLAGS
CXXFLAGS=$CLANGXX_FLAGS

# The namespace std { class type_info; } is needed for older libstdc++
# versions such as the 4.4 in Debian 6.0.
# http://lists.cs.uiuc.edu/pipermail/cfe-dev/2011-February/013207.html
AC_MSG_CHECKING([that clang++ works])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM(
    [[ namespace std { class type_info; }
       #include <iostream> ]],
    [[ std::cout << "Hello clang++ world!" << std::endl ]]
  )],
  [AC_MSG_RESULT(yes)
   # The vecmathlib repo should be checked out under lib/kernel
   # and the pocl directory should have the generated files in place.
   if test -r $srcdir/lib/kernel/vecmathlib/pocl/pow.cc; then
      use_vecmathlib=yes
   fi
   AC_DEFINE_UNQUOTED([CLANGXX], ["$CLANGXX"], [clang++ executable])
   AC_DEFINE_UNQUOTED([CLANGXX_FLAGS], ["$CLANGXX_FLAGS"], [clang++ compiler flags])
  ],
  [AC_MSG_RESULT(no)
   CLANGXX=""
   CLANGXX_FLAGS=""])

CXX=$old_CXX
CXXFLAGS=$old_CXXFLAGS

fi

AM_CONDITIONAL([CLANGXX_AVAILABLE], test ! -z $CLANGXX)
AM_CONDITIONAL([USE_VECMATHLIB], test ! -z "$use_vecmathlib")

if test x"$use_vecmathlib" = xyes; then
  AC_DEFINE([USE_VECMATHLIB], [], "Use vecmathlib if available for the target.")
fi

AC_ARG_VAR([CLFLAGS], [Flags to be used when compiling CL sources])

AC_DEFINE_UNQUOTED([CLANG], ["$CLANG"], [clang executable.])
AC_DEFINE_UNQUOTED([CLFLAGS], ["$CLFLAGS"], [CL compiler flags.])

LLVM_PROG([OPT], [opt], [LLVM optimizer])
LLVM_PROG([LLVM_OPT], [opt], [LLVM optimizer])
LLVM_PROG([LLC], [llc], [LLVM static compiler])
LLVM_PROG([LLVM_AS], [llvm-as], [LLVM assembler])
LLVM_PROG([LLVM_RANLIB], [llvm-ranlib], [index generator for LLVM archive])
LLVM_PROG([LLVM_AR], [llvm-ar], [LLVM archiver])
LLVM_PROG([LLVM_LINK], [llvm-link], [LLVM IR linker])


# $CLANG and $LLC are needed by config/xclang
export CLANG
export LLC

AC_DEFINE_UNQUOTED([LLC], ["$LLC"], [LLVM compiler executable.])

# Targets.
# This is the target architecture of the basic and pthreads devices.
# These always are the same arch as host (in current pocl implementation).
# Also, this is the autoconf "host".
# We have: 
# target triplet: the OCL device target triple
# kernel dir: where to search for the ocl lib kernel
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
  armv7l)
    AC_MSG_NOTICE([using the ARM optimized kernel lib for the native device])
    # check for gnueabihf, and set clang target accordingly
    case $host in
      armv7l*gnueabihf)
         OCL_KERNEL_TARGET="armv7l-unknown-linux-gnueabihf"
         HOST_LLC_FLAGS="$HOST_LLC_FLAGS -float-abi=hard";;
      *)
         #armv7l seems a popular host_cpu in distros. Clang uses "armv7", though
         OCL_KERNEL_TARGET="armv7";;
    esac

    # Assume -lm should not be used. TODO: add an embedded math lib
    # for compiling kernels for ARM envs without usable math libs.
    HOST_CLANG_FLAGS="$HOST_CLANG_FLAGS -target $OCL_KERNEL_TARGET -mfloat-abi=hard"
    kernel_dir="arm"
    CL_DEVICE_ADDRESS_BITS=32
    ;;
  amd64|x86_64)
    AC_MSG_NOTICE([using the x86_64 optimized kernel lib for the native device])
    HOST_LD_FLAGS="$HOST_LD_FLAGS -lm"
    OCL_KERNEL_TARGET=$host
    kernel_dir="x86_64"
    CL_DEVICE_ADDRESS_BITS=64
    ;;
  i?86)
    AC_MSG_NOTICE([using the x86 32bit optimized kernel lib for the native device])
    HOST_LD_FLAGS="$HOST_LD_FLAGS -lm"
    OCL_KERNEL_TARGET=$host
    kernel_dir="x86_64"
    CL_DEVICE_ADDRESS_BITS=32
    ;;
 
  tce)
    AC_MSG_NOTICE([using the TCE optimized kernel lib for the native device])
    HOST_LD_FLAGS="$HOST_LD_FLAGS -lm"
    kernel_dir="tce"
    ;;

  powerpc*)
    #on all tested operating systems with PowerPC, the processor is 64bit,
    #but the operating system is 32 bit. Therefore, when on powerpc, always build
    #for 32 bit ppc. If this does not work for you ... patches welcome.
    AC_MSG_NOTICE([using the ppc64 optimized kernel lib for the native device])
    HOST_LD_FLAGS="$HOST_LD_FLAGS -lm"
    kernel_dir="powerpc"
    OCL_KERNEL_TARGET="powerpc-unknown-linux-gnu"
    CL_DEVICE_ADDRESS_BITS=32
    #TODO: not all PowerPCs have Altivec?
    HOST_LLC_FLAGS="$HOST_LLC_FLAGS -mattr=altivec"
    ;;
  *)
    AC_MSG_ERROR([unknown/unsupported host $host])
    ;;
esac
OCL_TARGETS=$kernel_dir


AC_DEFINE_UNQUOTED([KERNEL_DIR], "$kernel_dir", [Use the libkernel from lib/kernel/$KERNEL_DIR/])

AC_SUBST([TARGET], [$target])
AC_DEFINE_UNQUOTED([TARGET], "$target", [The target triple.])
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
AC_DEFINE_UNQUOTED([HOST], "$host", [The host triple.])
AC_SUBST([HOST_CPU], [$host_cpu])
AC_DEFINE_UNQUOTED([HOST_CPU], ["$host_cpu"], [The host CPU type.])
AC_SUBST([OCL_KERNEL_TARGET],[$OCL_KERNEL_TARGET])
AC_DEFINE_UNQUOTED([OCL_KERNEL_TARGET], ["$OCL_KERNEL_TARGET"], [The kernel target triplet.])
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

AC_SUBST([POCL_DEVICE_ADDRESS_BITS], [$CL_DEVICE_ADDRESS_BITS])
AC_DEFINE_UNQUOTED([POCL_DEVICE_ADDRESS_BITS],
                   [$CL_DEVICE_ADDRESS_BITS],
                   [Value based on host processor, for basic and pthreads devices])


####################################################################
# Determine which device drivers to build.

OCL_DRIVERS="basic pthreads"

# look for the TCE library (needed for the ttasim device driver)
AC_PATH_PROG([TCE_CONFIG], [tce-config])

AC_ARG_ENABLE([tce],
              [AS_HELP_STRING([--disable-tce], 
                              [Do not build TCE driver, even if TCE is found.])],
              [try_tce="$enableval"],
              [try_tce="yes"])
            
if test ! -z "$TCE_CONFIG" -a x"$try_tce" = xyes; then

AC_PATH_PROG([TCECC], [tcecc])

AC_LANG([C++])

old_CXXFLAGS=$CXXFLAGS
CXXFLAGS=`$TCE_CONFIG --libs --cxxflags --includes`

AC_MSG_CHECKING([for TCE])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM(
    [[ #include <Application.hh> ]],
    [[ Application::initialize() ]]
  )],
  [AC_MSG_RESULT(yes)
  [TCE_LIBRARY=ok]],
  [AC_MSG_RESULT(no)])

CXXFLAGS=$old_CXXFLAGS

fi

if test -z "$TCECC" || \
   test -z "$TCE_CONFIG" || \
   test -z "$TCE_LIBRARY"; then

AC_MSG_NOTICE([TCE installation not found. The ttasim device will not be built.])

else

AC_MSG_NOTICE([TCE installation found. The ttasim device will be built.])
AC_DEFINE([TCE_AVAILABLE], [1], [Defined to 1 if TCE libraries and tools are available])
AC_SUBST([TCE_AVAILABLE], ["1"])
LD_FLAGS_BIN="$LD_FLAGS_BIN `$TCE_CONFIG --libs`"
OCL_DRIVERS="$OCL_DRIVERS tce"
OCL_TARGETS="$OCL_TARGETS tce"
fi

AM_CONDITIONAL([TCE_AVAILABLE], test ! -z $TCE_AVAILABLE)

# Check if CellSPU support is found
# The libspe version requirement is not strict. This is the only one tested.
PKG_CHECK_MODULES([LIBSPE], [libspe2 >= 2.2.80],
  [
    AC_MSG_NOTICE([libspe2 found.])
    LIBSPE_AVAILABLE=yes
    OCL_DRIVERS="$OCL_DRIVERS spu"
    OCL_TARGETS="$OCL_TARGETS cellspu"
    AC_DEFINE([BUILD_SPU], [], [Defined when CellSPU tools are found])
  ],
  [
    AC_MSG_NOTICE([libspe2 not found (by pkg-config). Spu driver not built.])
    LIBSPE_AVAILABLE=no
  ])

AM_CONDITIONAL([BUILD_SPU],[echo $OCL_DRIVERS | grep spu])

AC_MSG_NOTICE([Building the following device drivers: $OCL_DRIVERS])

AC_SUBST([OCL_TARGETS])

AC_CONFIG_FILES([lib/kernel/arm/Makefile
                 lib/kernel/tce/Makefile
                 lib/kernel/cellspu/Makefile
                 lib/kernel/powerpc/Makefile
                 lib/kernel/powerpc64/Makefile
                 lib/kernel/x86_64/Makefile])


AC_DEFINE_UNQUOTED([BUILDDIR], ["`pwd`"], [Absolute path of build directory.])
AC_DEFINE_UNQUOTED([SRCDIR], ["`cd $srcdir && pwd`"], [Root for the project sources.])

if test x"$prefix" = xNONE; then
  prefix=/usr/local
  #this check is "if user did not override sysconfdir"
  if test x"$sysconfdir" = x'${prefix}/etc'; then
    sysconfdir=/etc
  fi
fi

# Checks for header files.

# Checks for library features.
AC_LANG([C++])

AC_MSG_CHECKING([for aligned_alloc])
AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM(
        [[ #include <stdlib.h> ]],
        [[ void* p = aligned_alloc(0, 0); ]]
    )],
    [AC_MSG_RESULT(yes)
    AC_DEFINE([HAVE_ALIGNED_ALLOC], [], [Defined if aligned_alloc is available.])],
    [AC_MSG_RESULT(no)])

AC_MSG_CHECKING([for posix_memalign])
AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM(
        [[ #include <stdlib.h> ]],
        [[ int r = posix_memalign(NULL, 0, 0); ]]
    )],
    [AC_MSG_RESULT(yes)
    AC_DEFINE([HAVE_POSIX_MEMALIGN], [], [Defined if posix_memalign is available.])],
    [AC_MSG_RESULT(no)])

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
                 include/powerpc/types.h
                 include/powerpc64/types.h
                 include/cellspu/types.h
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

AC_CHECK_ALIGNOF([float16], [typedef float float16  __attribute__((__ext_vector_type__(16)));])
AC_CHECK_ALIGNOF([double16], [typedef double double16  __attribute__((__ext_vector_type__(16)));])
AS_IF([test $ac_cv_alignof_float16 -gt $ac_cv_alignof_double16],
      [AC_DEFINE([MAX_EXTENDED_ALIGNMENT], [ALIGNOF_FLOAT16], [Defined to greatest expected alignment for extended types, in bytes.])],
      [AC_DEFINE([MAX_EXTENDED_ALIGNMENT], [ALIGNOF_DOUBLE16], [Defined to greatest expected alignment for extended types, in bytes.])])

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

# The libtool library version string to use (passed to -version-info).
# See: http://www.nondot.org/sabre/Mirrored/libtool-2.1a/libtool_6.html
# libpocl.so should get only API additions as we are implementing a standard.
# 0:0:0 == 0.6
# 1:0:0 == 0.7 (not backwards compatible with 0:0:0 due to the ICD)
# 2:0:1 == 0.8 (currently backwards compatible with 0.7, thus age = 1).
AC_SUBST([LIB_VERSION], ["2:0:1"])

# The kernel compiler opt plugin shared library, however, changes more 
# drastically. Let's try to follow the similar 'current' numbering as
# pocl and perhaps tune the 'revision' and 'age' later.
AC_SUBST([KERNEL_COMPILER_LIB_VERSION], ["2:0:1"])

AC_SUBST([LD_FLAGS_BIN])

AC_CONFIG_FILES([Makefile
                 pocl.pc
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
                 lib/CL/devices/cellspu/Makefile
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
                 examples/opencl-book-samples/Makefile
                 examples/Rodinia/Makefile
                 examples/Parboil/Makefile
                 examples/ViennaCL/Makefile
                 examples/AMD/Makefile
                 examples/EinsteinToolkit/Makefile
                 examples/VexCL/Makefile
                 scripts/Makefile
                 tests/Makefile
                 tests/kernel/Makefile
                 tests/regression/Makefile
                 tests/workgroup/Makefile
                 tests/runtime/Makefile
                 tests/tce/Makefile
                 tests/tce/ttasim/Makefile
                 tests/tce/fp16/Makefile
                 tests/cell/Makefile
                 tests/cell/hello/Makefile
                 ])

AC_OUTPUT


#Print a summary of selected options:
echo
echo =======================================================================
echo
echo Using LLVM version $LLVM_VERSION from $($LLVM_CONFIG --prefix)
echo
echo -n ICD extension:
if test x"$icd" == xno;
then
echo " disabled"
else
 echo " enabled"
 if test x"$OCL_ICD_available" != xyes; 
 then
  echo  Warning: ocl-icd not found.
  if test x"$sysconfdir" != x/etc;
  then
   echo "Not installing pocl.icd to /etc/OpenCL/vendors. This means that"
   echo "your ICD loader probably will not find pocl. See README for details."
  else
   echo "'make check' might fail if it is run before 'make install' unless" 
   echo "your ICD loader supports the OCL_ICD_VENDORS environment variable"
   echo "please consult README.devel for details."
  fi
 else
  echo "Using ocl-icd to run checks inside build tree."
 fi
fi
echo

if test x"$oclsym" == xyes;
then
 echo "Enabling direct linking with libpocl."
else
 echo "Linking directly with libpocl will not work."
fi
echo

if test x"$use_vecmathlib" == xyes;
then
 echo "Using vecmathlib math builtins."
fi

echo Enabled device drivers: $OCL_DRIVERS
echo Enabled testsuites: $POAT_TESTSUITES
echo

