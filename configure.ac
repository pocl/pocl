##                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
# 
# Copyright (c) 2011-2013 Universidad Rey Juan Carlos and
#               2011-2015 Pekka Jääskeläinen / Tampere University of Technology
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
AC_INIT([pocl], [0.12-pre], [pocl-devel@lists.sourceforge.net])
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
AM_CONDITIONAL([FREEBSD_HOST_OS], false)
case $host_os in
  darwin* )
    AM_CONDITIONAL([DARWIN_HOST_OS], true)
    AC_SUBST([LIBRARY_SUFFIX], [".dylib"] )
         ;;
  freebsd* )
    AM_CONDITIONAL([FREEBSD_HOST_OS], true)
    AC_SUBST([LIBRARY_SUFFIX], [".so"] )
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

###################################################
# LLVM configuration
#
AC_ARG_VAR([LLVM_CONFIG], [Program used to retrieve LLVM options and binaries])
AC_PATH_PROGS([LLVM_CONFIG], [llvm-config llvm-config-mp-3.3 llvm-config-3.3 llvm-config33 llvm-config-mp-3.4 llvm-config-3.4 llvm-config34 llvm-config-mp-3.2 llvm-config-3.2 llvm-config32 ])
test -z "$LLVM_CONFIG" && AC_MSG_FAILURE([no llvm-config found in \$PATH])

LLVM_VERSION=`$LLVM_CONFIG --version`
LLVM_BINDIR=`$LLVM_CONFIG --bindir`
LLVM_LIBDIR=`$LLVM_CONFIG --libdir`
LLVM_LDFLAGS=`$LLVM_CONFIG --ldflags`
# Whether we can use our own printf implementation
NEW_PRINTF_WORKS=true
LLVM_3_6=false

AC_SUBST([LLVM_VERSION], [$LLVM_VERSION])
AC_DEFINE_UNQUOTED([LLVM_VERSION], ["$LLVM_VERSION"], "LLVM version as a string.")

case "$LLVM_VERSION" in
     3.2*)
     AC_DEFINE([LLVM_3_2], [], "Using LLVM 3.2")
     NEW_PRINTF_WORKS=false
   ;;
     3.3*)
     AC_DEFINE([LLVM_3_3], [], "Using LLVM 3.3")
     NEW_PRINTF_WORKS=false
   ;;
     3.4*)
     AC_DEFINE([LLVM_3_4], [], "Using LLVM 3.4")
   ;;
     3.5*)
     AC_DEFINE([LLVM_3_5], [], "Using LLVM 3.5")
     LLVM_LDFLAGS="$LLVM_LDFLAGS `$LLVM_CONFIG --system-libs`"
   ;;
     3.6*)
     AC_DEFINE([LLVM_3_6], [], "Using LLVM 3.6")
     LLVM_3_6=true
     LLVM_LDFLAGS="$LLVM_LDFLAGS `$LLVM_CONFIG --system-libs`"
   ;;
     3.7*)
     AC_DEFINE([LLVM_3_7], [], "Using LLVM svn, upcoming 3.7")
     LLVM_LDFLAGS="$LLVM_LDFLAGS `$LLVM_CONFIG --system-libs`"
   ;;
     *)
   AC_MSG_ERROR(
   [
Unsupported LLVM version. Please use LLVM version 3.2, 3.3, 3.4, 3.5, 3.6, 3.7.
   ])
   LLVM_VERSION=
   ;;
esac

AM_CONDITIONAL([LLVM_3_6], [test "x$LLVM_3_6" = "xtrue"])

AC_SUBST([LLVM_LDFLAGS], [$LLVM_LDFLAGS], [llvm-config returned ldflags])
AM_CONDITIONAL([NEW_PRINTF_WORKS], $NEW_PRINTF_WORKS)

# When building with API linking, clang is always linked statically, so user might want to link llvm static to libpocl too
# or risk causing version mismatches. Also useful when other platform libraries use LLVM too, see issue #46.
# Also note, when building shared LLVM libraries with cmake, there is no libLLVM-<versionnumber>.so
link_llvm_static="no"
AC_ARG_ENABLE([static-llvm],
              [AS_HELP_STRING([--enable-static-llvm],
                              [Link LLVM statically. Default is dynamic.])],
              [link_llvm_static="yes"])

LLVM_SHARED_LIB_FILE=$LLVM_LIBDIR/libLLVM-$LLVM_VERSION$LIBRARY_SUFFIX

if test "$link_llvm_static" = "yes"; then
  LLVM_LIBS=$($LLVM_CONFIG --libfiles)
else
  LLVM_LIBS=-lLLVM-$LLVM_VERSION
  if test ! -e $LLVM_SHARED_LIB_FILE;
  then
    # Cmake's -DLLVM_BUILD_LLVM_DYLIB=on builds a dynlib
    # named libLLVM.so.VERSION. Let's see if we can find it.
    LLVM_SHARED_LIB_FILE=$LLVM_LIBDIR/libLLVM$LIBRARY_SUFFIX.$LLVM_VERSION
    if test ! -e $LLVM_SHARED_LIB_FILE;
    then
      LLVM_LIBS=$($LLVM_CONFIG --libs)
      AC_MSG_NOTICE([LLVM shared library $LLVM_SHARED_LIB_FILE not found.])
    else
      LLVM_LIBS=-lLLVM
      AC_MSG_NOTICE([LLVM shared library $LLVM_SHARED_LIB_FILE built with cmake found.])
    fi
  fi
fi

AC_SUBST([LLVM_LIBS])

if `$LLVM_CONFIG --build-mode | grep -q Asserts`;
then
AC_DEFINE([LLVM_BUILT_WITH_ASSERTS], [], "LLVM was built with Assertions on.")
fi

LLVM_RTTI_CHECK=$($LLVM_CONFIG --cxxflags | grep -i "\-fno-rtti")

if test x"$LLVM_RTTI_CHECK" != x; then
AC_MSG_WARN([Your LLVM was not built with RTTI. 
You should rebuild LLVM with 'make REQUIRES_RTTI=1'. 
See the INSTALL file for more information.])
fi

AC_ARG_ENABLE([region-allocator],
[AS_HELP_STRING([--enable-region-allocator], 
 [Use a custom OpenCL optimized region-based memory allocator for the CPU devices instead of allocating buffers directly with malloc (experimental with known issues!).])],
[
AC_DEFINE([CUSTOM_BUFFER_ALLOCATOR], [], "Use a custom buffer allocator")
],
[])

####################################################################
# Manage optional testsuites

# Option to choose external testsuites
AC_ARG_ENABLE([testsuites],
	      [AS_HELP_STRING([--enable-testsuites=suite1,suite2,...],
          [choose enabled external project testsuites (all,opencl-book-samples,ViennaCL,Rodinia,Parboil,amd,amdsdk2_9,VexCL,Piglit,Halide,OpenCV,CloverLeaf,hsa])],
	      [],[enable_testsuites=check])

enable_testsuite_opencl_book_samples=no
enable_testsuite_viennacl=no
enable_testsuite_rodinia=no
enable_testsuite_parboil=no
enable_testsuite_amd=no
enable_testsuite_amdsdk2_9=no
enable_testsuite_vexcl=no
enable_testsuite_piglit=no
enable_testsuite_halide=no
enable_testsuite_opencv=no
enable_testsuite_hsa=no
if test x"$enable_testsuites" = xcheck ; then
  if test -d "$srcdir/examples/opencl-book-samples/checkout" ; then
    enable_testsuite_opencl_book_samples=yes
  fi
  if test -f "$srcdir/examples/ViennaCL/ViennaCL-1.5.1.tar.gz" ; then
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
  if test -f "$srcdir/examples/AMDSDK2.9/AMD-APP-SDK-v2.9-RC-lnx64.tgz" ; then
     enable_testsuite_amdsdk2_9=yes
  fi
  if test -f "$srcdir/examples/VexCL/vexcl/README.md" ; then
    enable_testsuite_vexcl=yes
  fi
  if test -f "$srcdir/examples/piglit/piglit/piglit-run.py" ; then
    enable_testsuite_piglit=yes
  fi  
  if test -f "$srcdir/examples/Halide/Halide/Makefile" ; then
    enable_testsuite_halide=yes
  fi
  if test -f "$srcdir/examples/OpenCV/opencv-3.0.0-beta.zip" ; then
    enable_testsuite_opencv=yes
  fi  
  if test -f "$srcdir/examples/CloverLeaf/CloverLeaf_OpenCL/Makefile" ; then
    enable_testsuite_cloverleaf=yes
  fi  
fi
AS_CASE([,"$enable_testsuites",],
  [*,all,*|*,yes,*], [enable_testsuites="opencl-book-samples,ViennaCL,Rodinia,Parboil,amd,amdsdk2_9,VexCL,piglit,Halide,OpenCV,CloverLeaf"],
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
    enable_testsuite_amdsdk2_9=yes
  ])
AS_CASE([,"$enable_testsuites",],
  [*,VexCL,*|*,vexcl,*], [
    enable_testsuite_vexcl=yes
  ])
AS_CASE([,"$enable_testsuites",],
  [*,piglit,*|*,Piglit,*], [
    enable_testsuite_piglit=yes
  ])
AS_CASE([,"$enable_testsuites",],
  [*,halide,*|*,Halide,*], [
    enable_testsuite_halide=yes
  ])
AS_CASE([,"$enable_testsuites",],
  [*,opencv,*|*,OpenCV,*], [
    enable_testsuite_opencv=yes
  ])
AS_CASE([,"$enable_testsuites",],
  [*,cloverleaf,*|*,CloverLeaf,*], [
    enable_testsuite_cloverleaf=yes
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

if test "$enable_testsuite_amdsdk2_9" = "yes" ; then
   # Check for software and sources for AMD APP SDK
  if ! test -f "$srcdir/examples/AMDSDK2.9/AMD-APP-SDK-v2.9-RC-lnx64.tgz"; then
    AC_MSG_WARN([disabling AMD APP SDK testsuite as the 2.9 lnx64 package not in '$srcdir/examples/AMDSDK2.9'.])
    enable_testsuite_amdsdk2_9=no
  fi
  PKG_CHECK_MODULES([SDL], [sdl >= 1.2], [:], 
    [ AC_MSG_WARN([pkg-config could not find libSDL, AMD APP SDK 2.9 testsuite is not used ]) 
      enable_testsuite_amdsdk2_9=no
    ])
fi

if test "$enable_testsuite_vexcl" = "yes" ; then
  # Check for software and sources for VexCL
  if ! test -f "$srcdir/examples/VexCL/vexcl/README.md"; then
    AC_MSG_WARN([disabling VexCL testsuite as the git branch is not in '$srcdir/examples/VexCL/vexcl'.])
    enable_testsuite_vexcl=no
  fi

  AX_BOOST_BASE([1.52.0], [], [
    AC_MSG_WARN(
    [boost library v1.52+ not found. VexCL tests disabled])
    enable_testsuite_vexcl=no
  ])
fi

if test "$enable_testsuite_viennacl" = "yes" ; then
  # Check for software and sources for ViennaCL
  AX_BOOST_BASE([1.42.0], [], [
    AC_MSG_WARN(
    [boost library v1.42+ not found. ViennaCL tests disabled])
    enable_testsuite_viennacl=no
  ])
fi

if test "$enable_testsuite_piglit" = "yes" ; then
  if ! test -f "$srcdir/examples/piglit/piglit/piglit-run.py" ; then
    AC_MSG_WARN([disabling Piglit testsuite: piglit-run.py not found in '$srcdir/examples/piglit/piglit/'])
    enable_testsuite_piglit=no
  fi
fi

if test "$enable_testsuite_opencv" = "yes" ; then
  if ! test -f "$srcdir/examples/OpenCV/opencv-3.0.0-beta.zip" ; then
    AC_MSG_WARN([disabling OpenCV testsuite: opencv-3.0.0-beta.zip not found in '$srcdir/examples/OpenCV'])
    enable_testsuite_opencv=no
  fi
fi      

AM_CONDITIONAL([TEST_SUITE_SAMPLES], [test "$enable_testsuite_opencl_book_samples" = "yes"])
AM_CONDITIONAL([TEST_SUITE_VIENNACL], [test "$enable_testsuite_viennacl" = "yes"])
AM_CONDITIONAL([TEST_SUITE_RODINIA], [test "$enable_testsuite_rodinia" = "yes"])
AM_CONDITIONAL([TEST_SUITE_PARBOIL], [test "$enable_testsuite_parboil" = "yes"])
AM_CONDITIONAL([TEST_SUITE_AMD], [test "$enable_testsuite_amd" = "yes"])
AM_CONDITIONAL([TEST_SUITE_AMDSDK2_9], [test "$enable_testsuite_amdsdk2_9" = "yes"])
AM_CONDITIONAL([TEST_SUITE_VEXCL], [test "$enable_testsuite_vexcl" = "yes"])
AM_CONDITIONAL([TEST_SUITE_PIGLIT], [test "$enable_testsuite_piglit" = "yes"])
AM_CONDITIONAL([TEST_SUITE_HALIDE], [test "$enable_testsuite_halide" = "yes"])
AM_CONDITIONAL([TEST_SUITE_OPENCV], [test "$enable_testsuite_opencv" = "yes"])
AM_CONDITIONAL([TEST_SUITE_CLOVERLEAF], [test "$enable_testsuite_cloverleaf" = "yes"])

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
if test "$enable_testsuite_amdsdk2_9" = "yes"; then
  AC_MSG_NOTICE([tests from AMD APP SDK 2.9 are enabled])
  POAT_TESTSUITES="$POAT_TESTSUITES amdsdk2_9"
else
  AC_MSG_NOTICE([tests from AMD APP SDK 2.9 are disabled])
fi
if test "$enable_testsuite_vexcl" = "yes"; then
  AC_MSG_NOTICE([tests from VexCL are enabled])
  POAT_TESTSUITES="$POAT_TESTSUITES vexcl"
else
  AC_MSG_NOTICE([tests from VexCL are disabled])
fi
if test "$enable_testsuite_piglit" = "yes"; then
  AC_MSG_NOTICE([tests from Piglit are enabled])
  POAT_TESTSUITES="$POAT_TESTSUITES piglit"
else
  AC_MSG_NOTICE([tests from piglit are disabled])
fi
if test "$enable_testsuite_halide" = "yes"; then
  AC_MSG_NOTICE([tests from Halide are enabled])
  POAT_TESTSUITES="$POAT_TESTSUITES Halide"
else
  AC_MSG_NOTICE([tests from Halide are disabled])
fi
if test "$enable_testsuite_opencv" = "yes"; then
  AC_MSG_NOTICE([tests from OpenCV are enabled])
  POAT_TESTSUITES="$POAT_TESTSUITES OpenCV"
else
  AC_MSG_NOTICE([tests from OpenCV are disabled])
fi
if test "$enable_testsuite_cloverleaf" = "yes"; then
  AC_MSG_NOTICE([tests from CloverLeaf are enabled])
  POAT_TESTSUITES="$POAT_TESTSUITES CloverLeaf"
else
  AC_MSG_NOTICE([tests from CloverLeaf are disabled])
fi


####################################################################
# Pthread Library

old_CFLAGS="$CFLAGS"
old_CPPFLAGS="$CPPFLAGS"
CFLAGS=`echo "$CFLAGS" | sed 's/-Werror[[-A-Za-z0-9=]]*//g'`
CPPFLAGS=`echo "$CPPFLAGS" | sed 's/-Werror[[-A-Za-z0-9=]]*//g'`

ACX_PTHREAD()

CFLAGS="$old_CFLAGS"
CPPFLAGS="$old_CPPFLAGS"

LD_FLAGS_BIN="$LD_FLAGS_BIN $PTHREAD_LDFLAGS"


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
# OpenCL headers

# Option to install OpenCL header files
# By default, do not install them unless none are found in the system include directories
AC_ARG_ENABLE([install-opencl-headers],
              [AS_HELP_STRING([--enable-install-opencl-headers],
              [install OpenCL headers. Ones from Khronos should be installed instead.])],
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

if test x"$ioh" = "yes"; then
# In case we install pocl's headers, ensure we build against them
# by adding the pocl's include as the first include path.
   CPPFLAGS="-Iinclude $CPPFLAGS"
fi

# Check for OpenGL.h headers as they are required to compile
# test cases that use cl.hpp (which requires it).
AC_CHECK_HEADERS([GL/gl.h OpenGL/OpenGL.h], [opengl_found="ok"; break])
test -z "$opengl_found" && \
AC_MSG_FAILURE([Please install an OpenGL implementation with gl.h or OpenGL.h. It is required by the C++ bindings.])
AM_CONDITIONAL([HAVE_OPENCL_HPP], true)
AC_DEFINE([HAVE_OPENCL_HPP], [] ,[Defined if system cl.hpp found to be working])

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
	AC_DEFINE([POCL_DEBUG_BUILD], [], "Build pocl in debug mode")
fi

AC_LANG([C])
AC_MSG_CHECKING([for clock_gettime availability])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM(
    [ #include <time.h> ],
    [ struct timespec pocl_debug_timespec; clock_gettime(CLOCK_REALTIME, &pocl_debug_timespec); ]
  )],
  [AC_MSG_RESULT(yes)
   AC_DEFINE([HAVE_CLOCK_GETTIME], [1], [Have clock_gettime])
   AM_CONDITIONAL([HAVE_CLOCK_GETTIME], true)],
  [AC_MSG_RESULT(no)
   AM_CONDITIONAL([HAVE_CLOCK_GETTIME], false)]
)

# Enable debug message output when the env POCL_DEBUG=1 is set.
AC_DEFINE([POCL_DEBUG_MESSAGES], [1], [Printout debug messages in case POCL_DEBUG env is set])

be_pedantic="no"
AC_ARG_ENABLE([pedantic],
              [AS_HELP_STRING([--enable-pedantic], [Compile host library with stricter compiler flags.])],
              [be_pedantic="yes"])
if test "$be_pedantic" = "yes"
then
	CXXFLAGS="$CXXFLAGS -Wno-unused-result -Werror"
	CFLAGS="$CFLAGS -Wno-unused-result -Werror"
fi

cache_kernel="yes"
AC_ARG_ENABLE([kernel-cache],
              [AS_HELP_STRING([--disable-kernel-cache], [Disable persistent kernel cache])],
              [cache_kernel="no"])
if test "$cache_kernel" = "yes"
then
  AC_DEFINE([POCL_BUILD_KERNEL_CACHE], [1], "Enabled kernel cache feature")
else
  AC_DEFINE([POCL_BUILD_KERNEL_CACHE], [0], "Disabled kernel cache feature")
fi


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

if `$CLANG --version|grep -q SPIR`; 
then 
AC_DEFINE([CLANG_SPIR], [], "Using a SPIR generator Clang from Khronos.")
fi

# Targets.
# This is the target architecture of the basic and pthreads devices.
# These always are the same arch as host (in current pocl implementation).
# Also, this is the autoconf "host".
# We have:
# target triplet: the OCL device target triple
# kernel dir: where to search for the ocl lib kernel

# The option for specifying the target changed; try the modern syntax
# first, and fall back to the old-style syntax if this failed
AC_MSG_CHECKING([Clang option to specify the target])
if [ $CLANG --target=$host -x c /dev/null -S -o - >/dev/null 2>&1 ]; then
  CLANG_TARGET_OPTION='--target='
elif [ $CLANG -target $host -x c /dev/null -S -o - >/dev/null 2>&1 ]; then
  CLANG_TARGET_OPTION='-target '
else
  AC_MSG_ERROR([Cannot determine Clang option to specify the target])
fi
AC_MSG_RESULT(${CLANG_TARGET_OPTION}TARGET)

# Check for a functioning clang++. It is needed to compile faster math 
# builtins from vecmathlib.
LLVM_PROG_WARN([CLANGXX], [clang++])

# Test features of clang
# -can it compile a simple program
# -was LLVM built with NDEBUG or not
if test ! -z "$CLANGXX"; then

AC_LANG([C++])

CLANGXX_FLAGS="${CLANG_TARGET_OPTION}$host ${CLANGXX_FLAGS}"

old_CXX=$CXX
CXX=$CLANGXX
old_CXXFLAGS=$CXXFLAGS
CXXFLAGS=$CLANGXX_FLAGS
old_LDFLAGS=$LDFLAGS
old_LIBS=$LIBS

# The namespace std { class type_info; } is needed for older libstdc++
# versions such as the 4.4 in Debian 6.0.
# http://lists.cs.uiuc.edu/pipermail/cfe-dev/2011-February/013207.html
AC_MSG_CHECKING([whether clang++ works])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM(
    [[ namespace std { class type_info; }
       #include <iostream> ]],
    [[ std::cout << "Hello clang++ world!" << std::endl ]]
  )],
  [AC_MSG_RESULT(yes)
   # The vecmathlib repo should be checked out under lib/kernel
   # and the pocl directory should have the generated files in place.
   if test -r $srcdir/lib/kernel/vecmathlib/vecmathlib.h; then
      use_vecmathlib=yes
   fi
   AC_DEFINE_UNQUOTED([CLANGXX], ["$CLANGXX"], [clang++ executable])
   AC_DEFINE_UNQUOTED([CLANGXX_FLAGS], ["$CLANGXX_FLAGS"], [clang++ compiler flags])
  ],
  [AC_MSG_RESULT(no)
   CLANGXX=""
   CLANGXX_FLAGS=""])

#
# A few work-arounds for llvm-config issues
# - '-fno-rtti' is a work-around for llvm bug 14200
# - '-DNDEBUG' is a work-around for llvm bug 18253
# - pocl doesn't compile with '-pedantic'

LLVM_CXX_FLAGS=$($LLVM_CONFIG --cxxflags | sed -e 's/ -pedantic / /g')
LLVM_CXX_FLAGS="$LLVM_CXX_FLAGS -fno-rtti"

# llvm-config does not always report the "-DNDEBUG" flag correctly
# (see LLVM bug 18253). If LLVM and the pocl passes are built with 
# different NDEBUG settings, problems arise 
AC_MSG_CHECKING([if LLVM is built with assertions])
CXXFLAGS="$CXXFLAGS $($LLVM_CONFIG --cxxflags) -UNDEBUG"
LIBS="$LIBS $($LLVM_CONFIG --libs)"
LDFLAGS="$LDFLAGS $LLVM_LDFLAGS"

AC_LINK_IFELSE(
  [AC_LANG_PROGRAM(
    [[ #include <llvm/Support/Debug.h> ]],
    [[ llvm::DebugFlag=true; ]]
  )],
  [#No linking error. The llvm::DebugFlags was found in the LLVM libraries 
   AC_MSG_RESULT(yes)
  ],
  [AC_MSG_RESULT(no)
    LLVM_CXX_FLAGS="$LLVM_CXX_FLAGS -DNDEBUG"
  ])

CXX=$old_CXX
CXXFLAGS=$old_CXXFLAGS
LDFLAGS=$old_LDFLAGS
LIBS=$old_LIBS
fi
AC_SUBST(LLVM_CXX_FLAGS)


# Define the OpenCL version when compiling OpenCL C code. This also
# serves as indicator for the OpenCL C language, similar to
# __cplusplus for C++.
CLFLAGS="$CLFLAGS -D__OPENCL_VERSION__=120"

AM_CONDITIONAL([CLANGXX_AVAILABLE], test ! -z $CLANGXX)

AC_ARG_VAR([CLFLAGS], [Flags to be used when compiling CL sources])
AC_ARG_VAR([CLANGXX_FLAGS], [Flags to be used when compiling C++ sources with clang])

use_vecmathlib_builtins="no"
AC_ARG_ENABLE([vecmathlib-builtins],
              [AS_HELP_STRING([--enable-vecmathlib-builtins], [Use only __builtin_* functions in the kernel library.])],
              [use_vecmathlib_builtins="yes"])
if test "$use_vecmathlib_builtins" = "yes"
then
	CLFLAGS="$CLFLAGS -DPOCL_VECMATHLIB_BUILTIN"
	CLANGXX_FLAGS="$CLANGXX_FLAGS -DPOCL_VECMATHLIB_BUILTIN"
fi

AC_DEFINE_UNQUOTED([CLANG], ["$CLANG"], [clang executable.])
AC_DEFINE_UNQUOTED([CLFLAGS], ["$CLFLAGS"], [Additional CL compiler flags.])

# The Clang flags that are used always when compiling OpenCL C code.
# Note: We use -Wno-format to avoid warnings for printf, where we
# currently have to use %lld for long arguments (to handle 32-bit
# architectures).
FORCED_CLFLAGS="-Xclang -ffake-address-space-map -fno-math-errno -fblocks -fno-builtin -fasm -Wno-format"
AC_DEFINE_UNQUOTED([FORCED_CLFLAGS], 
["$FORCED_CLFLAGS"], 
[Always used flags for Clang when compiling OpenCL C code.])
AC_SUBST([FORCED_CLFLAGS], ["$FORCED_CLFLAGS"])

# TODO: this is duplicated
LLVM_PROG([OPT], [opt], [LLVM optimizer])
LLVM_PROG([LLVM_OPT], [opt], [LLVM optimizer])
LLVM_PROG([LLC], [llc], [LLVM static compiler])
LLVM_PROG([LLVM_AS], [llvm-as], [LLVM assembler])
LLVM_PROG([LLVM_LINK], [llvm-link], [LLVM IR linker])


# $CLANG and $LLC are needed by config/xclang
export CLANG
export LLC

AC_DEFINE_UNQUOTED([LLC], ["$LLC"], [LLVM compiler executable.])

AC_MSG_CHECKING([Linker option to build a shared library])
if echo $host | grep -q darwin; then
  LD_SHARED="-dylib"
else
  LD_SHARED="-shared"
fi
AC_MSG_RESULT($LD_SHARED)

HOST_CLANG_FLAGS=""
HOST_LLC_FLAGS="-relocation-model=pic"
HOST_AS_FLAGS=""
HOST_LD_FLAGS="$LD_SHARED"



# TODO: We need to set both target-triple and cpu-type when
# building, since the ABI depends on both. We can either add flags
# to all the scripts, or set the respective flags here in
# *_CLANG_FLAGS and *_LLC_FLAGS. Note that clang and llc use
# different option names to set these. Note that clang calls the
# triple "target" and the cpu "architecture", which is different
# from llc.

# Normalise the triple. Otherwise, clang normalises it when
# passing it to llc, which is then different from the triple we
# pass to llc. This would lead to inconsistent bytecode files,
# depending on whether they are generated via clang or directly
# via llc.

AC_MSG_CHECKING([LLC target triple (for host $host)])
dnl The quadrigraphs @<:@ and @:>@ turn into [ and ].
llc_triple=$($CLANG ${CLANG_TARGET_OPTION}$host -x c /dev/null -S -emit-llvm -o - | grep -e 'target triple' | perl -p -e 's/^.*target triple = "+(@<:@^ @:>@+).*"$/\1/')
AC_MSG_RESULT($llc_triple)
test -z "$llc_triple" && AC_MSG_ERROR([Cannot determine LLC target triple])
AC_MSG_CHECKING([LLC host CPU])
llc_host_cpu=$($LLC --version | grep 'Host CPU' | sed -e 's/^ *Host CPU: *//')

test -n "$LLC_HOST_CPU" && llc_host_cpu=$LLC_HOST_CPU 
test -z "$llc_host_cpu" && AC_MSG_ERROR([Cannot determine LLC host CPU])
AC_MSG_RESULT($llc_host_cpu)
#X86 has -march and -mcpu reversed, for clang
CLANG_MARCH_FLAG=march

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
  armv6l)
    AC_MSG_NOTICE([using the ARM optimized kernel lib for the native device])
    HOST_LD_FLAGS="$HOST_LD_FLAGS -lm"
    CL_DEVICE_ADDRESS_BITS=32
    case $host in
      *gnueabihf)
      case $llc_host_cpu in
        arm1176jz-s) 
          llc_host_cpu=arm1176jzf-s;;
      esac
      HOST_CLANG_FLAGS="$HOST_CLANG_FLAGS -mfloat-abi=hard"
      HOST_LLC_FLAGS="$HOST_LLC_FLAGS -float-abi=hard"
      HOST_AS_FLAGS="$HOST_AS_FLAGS -mfloat-abi=hard";;
    esac
    ;;
  arm*)
    AC_MSG_NOTICE([using the ARM optimized kernel lib for the native device])
    # check for gnueabihf, and set clang target accordingly
    case $host in
      #armv7l seems a popular host_cpu in distros. Clang uses "armv7", though
      arm*android*)
         llc_triple="armv7-unknown-linux-gnueabi"
         AC_DEFINE([HOST_FLOAT_SOFT_ABI], [], ["Use soft-float ABI to support most of android devices"])
         HOST_CLANG_FLAGS="$HOST_CLANG_FLAGS -mfloat-abi=soft -I"$SYSROOTDIR"/usr/include "
         HOST_LLC_FLAGS="$HOST_LLC_FLAGS -float-abi=soft"
         HOST_AS_FLAGS="$HOST_AS_FLAGS -mcpu=$llc_host_cpu -mfloat-abi=soft"
         HOST_LD_FLAGS="$HOST_LD_FLAGS -L/system/lib/ -shared -ldl -lc -lm /system/lib/crtbegin_so.o /system/lib/crtend_so.o";;
      armv7l*gnueabihf)
         llc_triple="armv7-unknown-linux-gnueabihf"
         HOST_CLANG_FLAGS="$HOST_CLANG_FLAGS -mfloat-abi=hard"
         HOST_LLC_FLAGS="$HOST_LLC_FLAGS -float-abi=hard"
         HOST_AS_FLAGS="$HOST_AS_FLAGS -mfloat-abi=hard";;
      armv7l*gnueabi)
         llc_triple="armv7-unknown-linux-gnueabi"
         AC_DEFINE([HOST_FLOAT_SOFT_ABI], [], ["basic and pthreads devices use soft-float ABI"])
         HOST_CLANG_FLAGS="$HOST_CLANG_FLAGS -mfloat-abi=soft"
         HOST_LLC_FLAGS="$HOST_LLC_FLAGS -float-abi=soft"
         HOST_AS_FLAGS="$HOST_AS_FLAGS -mcpu=$llc_host_cpu"
         HOST_AS_FLAGS="$HOST_AS_FLAGS -mfloat-abi=soft";;
      armv7hl-suse-linux-gnueabi)
         llc_triple="armv7hl-suse-linux-gnueabi"
         HOST_CLANG_FLAGS="$HOST_CLANG_FLAGS -mfloat-abi=hard"
         HOST_LLC_FLAGS="$HOST_LLC_FLAGS -float-abi=hard"
         HOST_AS_FLAGS="$HOST_AS_FLAGS -mcpu=$llc_host_cpu"
         HOST_AS_FLAGS="$HOST_AS_FLAGS -mfloat-abi=hard";;
      armv7hl-redhat-linux-gnu)
         llc_triple="armv7hl-redhat-linux-gnueabi"
         HOST_CLANG_FLAGS="$HOST_CLANG_FLAGS -mfloat-abi=hard"
         HOST_LLC_FLAGS="$HOST_LLC_FLAGS -float-abi=hard"
         HOST_AS_FLAGS="$HOST_AS_FLAGS -mcpu=$llc_host_cpu"
         HOST_AS_FLAGS="$HOST_AS_FLAGS -mfloat-abi=hard";;
       *)
         #this is for old arm distros (pre-hardfloat),
         #probably not used, and kept here "just in case"
         llc_triple="armv7";;
    esac
    CLANG_MARCH_FLAG=mcpu

    # Assume -lm should not be used. TODO: add an embedded math lib
    # for compiling kernels for ARM envs without usable math libs.
    CL_DEVICE_ADDRESS_BITS=32
    ;;
  amd64|x86_64)
    AC_MSG_NOTICE([using the x86_64 optimized kernel lib for the native device])
    HOST_LD_FLAGS="$HOST_LD_FLAGS -lm"
    CL_DEVICE_ADDRESS_BITS=64
    ;;
  i?86)
    AC_MSG_NOTICE([using the x86 32bit optimized kernel lib for the native device])
    HOST_LD_FLAGS="$HOST_LD_FLAGS -lm"
    CL_DEVICE_ADDRESS_BITS=32
    ;;
 
  tce)
    AC_MSG_NOTICE([using the TCE optimized kernel lib for the native device])
    HOST_LD_FLAGS="$HOST_LD_FLAGS -lm"
    llc_triple=tce-tut-llvm
    ;;

  powerpc64*)
    test ${LLVM_VERSION:0:3} == "3.4" && AC_MSG_ERROR([LLVM 3.4 fails on PowerPC, use 3.3])
    AC_MSG_NOTICE([using the ppc64 optimized kernel lib for the native device])
    HOST_LD_FLAGS="$HOST_LD_FLAGS -lm"
    llc_triple="powerpc64-unknown-linux-gnu"
    CL_DEVICE_ADDRESS_BITS=32
    #TODO: not all PowerPCs have Altivec?
    HOST_LLC_FLAGS="$HOST_LLC_FLAGS -mattr=altivec"
    CLANG_MARCH_FLAG="mcpu"
    ;;

  powerpc*)
    test ${LLVM_VERSION:0:3} == "3.4" && AC_MSG_ERROR([LLVM 3.4 fails on PowerPC, use 3.3])
    #on all tested operating systems with PowerPC, the processor is 64bit,
    #but the operating system is 32 bit. 
    AC_MSG_NOTICE([using the ppc64 optimized kernel lib for the native device])
    HOST_LD_FLAGS="$HOST_LD_FLAGS -lm"
    llc_triple="powerpc-unknown-linux-gnu"
    CL_DEVICE_ADDRESS_BITS=32
    #Altivec on PPC32 seems to have a few codegen issues
    use_vecmathlib=""
    HOST_AS_FLAGS="$HOST_AS_FLAGS ${CLANG_TARGET_OPTION}$llc_triple"
    CLANG_MARCH_FLAG="mcpu"
    ;;

  mipsel*)
    llc_triple="mipsel-unknown-linux-gnu"
    CLANG_MARCH_FLAG=mcpu
    CL_DEVICE_ADDRESS_BITS=32
    ;;

  mips*)
    llc_triple="mips-unknown-linux-gnu"
    CLANG_MARCH_FLAG=mcpu
    CL_DEVICE_ADDRESS_BITS=32
    ;;

  *)
    AC_MSG_ERROR([unknown/unsupported host $host])
    ;;
esac


kernel_dir="host"
OCL_TARGETS=$kernel_dir
HOST_CLANG_FLAGS="$HOST_CLANG_FLAGS ${CLANG_TARGET_OPTION}$llc_triple"
HOST_LLC_FLAGS="$HOST_LLC_FLAGS -mtriple=$llc_triple"
if test "$llc_host_cpu" != "(unknown)"
then
  HOST_CLANG_FLAGS="$HOST_CLANG_FLAGS -$CLANG_MARCH_FLAG=$llc_host_cpu"
  HOST_LLC_FLAGS="$HOST_LLC_FLAGS -mcpu=$llc_host_cpu"
fi

# Work-around a clang bug in LLVM 3.3: On 32-bit platforms, the size
# of Open CL C long is not 8 bytes 
if [echo "constant int test[sizeof(long)==8?1:-1]={1};" | $CLANG $HOST_CLANG_FLAGS -S -x cl - ]
then
    : # OpenCL C long is supported
else
    # AC_DEFINE needed for host code, HOST_CLANG_FLGAS for kernel code.
    AC_DEFINE_UNQUOTED([_CL_DISABLE_LONG],[],[Disable cl_khr_int64 on host based devices.])
    HOST_CLANG_FLAGS="$HOST_CLANG_FLAGS -D_CL_DISABLE_LONG"
fi
rm -f ./-.s
	
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
AC_SUBST([OCL_KERNEL_TARGET],[$llc_triple])
AC_DEFINE_UNQUOTED([OCL_KERNEL_TARGET], ["$llc_triple"], [The kernel target triplet.])
AC_SUBST([OCL_KERNEL_TARGET_CPU],[$llc_host_cpu])
AC_DEFINE_UNQUOTED([OCL_KERNEL_TARGET_CPU], ["$llc_host_cpu"], [The kernel target CPU variant.])
AC_SUBST([OCL_KERNEL_ARCH],[$OCL_KERNEL_ARCH])
AC_DEFINE_UNQUOTED([OCL_KERNEL_ARCH], ["$OCL_KERNEL_ARCH"], [The kernel architecture.])

AC_ARG_VAR([HOST_CLANG_FLAGS],
           [Parameters to for host compilation.])
AC_DEFINE_UNQUOTED([HOST_CLANG_FLAGS],
                   ["$HOST_CLANG_FLAGS"],
                   [Parameters to for host compilation.])
AC_ARG_VAR([HOST_LLC_FLAGS])
AC_DEFINE_UNQUOTED([HOST_LLC_FLAGS],
                   ["$HOST_LLC_FLAGS"], 
                   [Parameters to llc for code generation in the host.])
AC_ARG_VAR([HOST_AS_FLAGS])
AC_DEFINE_UNQUOTED([HOST_AS_FLAGS],
                   ["$HOST_AS_FLAGS"], 
                   [Parameters to as for code generation in the host.])
AC_SUBST([HOST_LD_FLAGS])
AC_DEFINE_UNQUOTED([HOST_LD_FLAGS],
                   ["$HOST_LD_FLAGS"],
                   [Parameter to compiler to generate loadable module.])

AC_SUBST([POCL_DEVICE_ADDRESS_BITS], [$CL_DEVICE_ADDRESS_BITS])
AC_DEFINE_UNQUOTED([POCL_DEVICE_ADDRESS_BITS],
                   [$CL_DEVICE_ADDRESS_BITS],
                   [Value based on host processor, for basic and pthreads devices])

AM_CONDITIONAL([USE_VECMATHLIB], test ! -z "$use_vecmathlib")
if test x"$use_vecmathlib" = xyes; then
  AC_DEFINE([USE_VECMATHLIB], [], "Use vecmathlib if available for the target.")
  CLANGXX_FLAGS="$CLANGXX_FLAGS -DVML_NO_IOSTREAM"
fi



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

LD_FLAGS_BIN="$LD_FLAGS_BIN `$TCE_CONFIG --libs`"

if `$TCE_CONFIG --version|grep -q trunk`
then
 AC_MSG_NOTICE([TCEMC installation found. The ttasim device will be built.])
 AC_DEFINE([TCEMC_AVAILABLE], [1], [Defined to 1 if TCEMC libraries and tools are available])
 AC_SUBST([TCEMC_AVAILABLE], ["1"])
 AC_DEFINE([TCE_AVAILABLE], [1], [Defined to 1 if TCE libraries and tools are available])
 AC_SUBST([TCE_AVAILABLE], ["1"])
 OCL_DRIVERS="$OCL_DRIVERS tcemc"
 TCEMC_AVAILABLE=1
 # TCEMC is a "superset" of TCE (lp:tce) features.
 TCE_AVAILABLE=1
 else
 AC_MSG_NOTICE([TCE installation found. The ttasim device will be built.])
 AC_DEFINE([TCE_AVAILABLE], [1], [Defined to 1 if TCE libraries and tools are available])
 AC_SUBST([TCE_AVAILABLE], ["1"])
 OCL_DRIVERS="$OCL_DRIVERS tce"
 TCE_AVAILABLE=1
fi

OCL_TARGETS="$OCL_TARGETS tce"
fi

AM_CONDITIONAL([TCE_AVAILABLE], test ! -z $TCE_AVAILABLE)
AM_CONDITIONAL([TCEMC_AVAILABLE], test ! -z $TCEMC_AVAILABLE)

# Checking for android cross-compile
ANDROID_PRESENT="";
AC_MSG_CHECKING([for Android])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM(
    [[ #include <android/log.h> ]]
  )],
  [AC_MSG_RESULT(yes)
  [ANDROID_PRESENT=ok]],
  [AC_MSG_RESULT(no)])

if test "$ANDROID_PRESENT" == ok;
then
AC_DEFINE([POCL_ANDROID], [1], [Defined to 1 if compiling for Android])
AC_DEFINE([POCL_ANDROID_PREFIX],
                   ["/data/data/org.pocl.libs/files"],
                   [Directory  where pocl files are installed in android])

fi
AM_CONDITIONAL([POCL_ANDROID], [test "$ANDROID_PRESENT" = ok])

# Check if CellSPU support is found
# The libspe version requirement is not strict. This is the only one tested.
# SPU backend was removed in LLVM 3.3 (and we don't accept LLVM 3.1 anymore in pocl)
# so compile the spu backend only when LLVM 3.2 is found
if test "$LLVM_VERSION" == 3.2 -o "$LLVM_VERSION" == 3.2svn
then
PKG_CHECK_MODULES([LIBSPE], [libspe2 >= 2.2.80],
  [
    AC_MSG_NOTICE([libspe2 found.])
    LIBSPE_AVAILABLE=yes
    OCL_DRIVERS="$OCL_DRIVERS spu"
    OCL_TARGETS="$OCL_TARGETS cellspu"
    AC_DEFINE([BUILD_SPU], [], [Defined when CellSPU tools are found])
    BUILD_SPU=1
  ],
  [
    AC_MSG_NOTICE([libspe2 not found (by pkg-config). Spu driver not built.])
    LIBSPE_AVAILABLE=no
    BUILD_SPU=0
  ])
fi
AM_CONDITIONAL([BUILD_SPU],[echo $OCL_DRIVERS | grep spu])

# Enable HSA by default in case the dependencies are met.
AC_ARG_ENABLE([hsa-amdgcn],
              [AS_HELP_STRING([--disable-hsa-amdgcn],
              [Disable the HSA device driver for AMD GCN devices.])],
              [enable_hsa="$enableval"], [enable_hsa="yes"])

AC_ARG_WITH(hsa-runtime-headers,
[AS_HELP_STRING([--with-hsa-runtime-headers],[absolute path where HSA runtime include files are found])],
[
  hsa_cpp_flags="-I${withval}"
  AC_SUBST([HSA_INCLUDES], ["$hsa_cpp_flags"] )
],
[
  hsa_cpp_flags="-I/opt/hsa/include"
  AC_SUBST([HSA_INCLUDES], ["$hsa_cpp_flags"] )
]
)

# Search for the HSA runtime lib.
AC_MSG_CHECKING([for HSA runtime])
old_LIBS="$LIBS"
old_LDFLAGS="$LDFLAGS"
LDFLAGS="$LDFLAGS -L/opt/hsa/lib"
LIBS=""
AC_SEARCH_LIBS([hsa_executable_get_symbol], [hsa-runtime64], [
HSA_LIBS="$LIBS"
AC_SUBST([HSA_LIBS])
],
[
enable_hsa=no
LDFLAGS="$old_LDFLAGS"
])
LIBS=$old_LIBS

# test that Clang supports the amdgcn--amdhsa target
AC_LANG([C])
 AC_LANG_CONFTEST(
 [AC_LANG_PROGRAM(
       [])
       ])

LLVM_AMDGCN_TARGET=no

AC_MSG_CHECKING([if the amdgcn--amdhsa target is supported in Clang/LLVM])
if $LLVM_BINDIR/clang conftest.c -target amdgcn--amdhsa -emit-llvm -S -c -o - 2>/dev/null | grep -cq 'target triple = "amdgcn--amdhsa"';
then
   AC_MSG_RESULT([yes])
   LLVM_AMDGCN_TARGET=yes
else
   AC_MSG_RESULT([no])
   enable_hsa=no
fi

tempCPPFLAGS=$CPPFLAGS
CPPFLAGS="$CPPFLAGS -I/opt/hsa/include"

AC_MSG_CHECKING([for HSA runtime headers])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM(
    [[ #include <hsa.h> ]]
  )],
[AC_MSG_RESULT(yes)],
[AC_MSG_RESULT(no)
enable_hsa="no"
])

CPPFLAGS=$tempCPPFLAGS


if test "$enable_hsa" = "yes";
then
  OCL_DRIVERS="$OCL_DRIVERS hsa"
  OCL_TARGETS="$OCL_TARGETS amdgcn hsail64"
  AC_DEFINE([AMDGCN_ENABLED], [1], [Defined in case AMDGCN device support enabled.])
  AC_DEFINE([BUILD_HSA], [1], [Defined to 1 in case the HSA driver should be built.])
  enable_testsuite_hsa=yes
  POAT_TESTSUITES="$POAT_TESTSUITES hsa"

  # -lhsail is missing from llvm-config --system-libs
  # -lHSAILUtil is missing from llvm-config --libfiles
  LLVM_LIBS="$LLVM_LIBS -lLLVMHSAILUtil -lhsail"
fi

# TODO: move above

# Enable the HSA test suite in case HSA support enabled.
AM_CONDITIONAL([TEST_SUITE_HSA], [test "$enable_testsuite_hsa" = "yes"])
AC_SUBST([POAT_TESTSUITES])

AM_CONDITIONAL([BUILD_AMDGCN],[echo $OCL_TARGETS | grep amdgcn])
AM_CONDITIONAL([BUILD_HSA],[echo $OCL_DRIVERS | grep hsa])

AC_MSG_NOTICE([Building the following device drivers: $OCL_DRIVERS])

AC_SUBST([OCL_TARGETS])

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
AC_LANG([C])

old_CC="$CC"
old_CPPFLAGS="$CPPFLAGS"
old_CFLAGS="$CFLAGS"
old_LDFLAGS="$LDFLAGS"
CC=$srcdir/config/xclang
if test ! -z "$host_alias"; then
    CC="$CC -h $host_alias"
fi
if test ! -z "$build_alias"; then
    CC="$CC -b $build_alias"
fi

CFLAGS=""
CPPFLAGS=""
LDFLAGS=""
for flag in $old_CPPFLAGS; do
    if echo " $flag" | grep -q -e -Werror; then
       CPPFLAGS="$CPPFLAGS $flag"
    fi
done
for flag in $old_CFLAGS; do
    if echo " $flag" | grep -q -e -Werror; then
       CFLAGS="$CFLAGS $flag"
    fi
done

AC_MSG_CHECKING([for posix_memalign])
AC_LINK_IFELSE(
    [AC_LANG_PROGRAM(
        [[ #include <stdlib.h> ]],
        [[ int r = posix_memalign(NULL, 0, 0); ]]
    )],
    [AC_MSG_RESULT(yes)
    AC_DEFINE([HAVE_POSIX_MEMALIGN], [], [Defined if posix_memalign is available.])],
    [AC_MSG_RESULT(no)])

# Checks for typedefs, structures, and compiler characteristics.
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

if test "$ANDROID_PRESENT" == ok;
then
AC_MSG_NOTICE([Setting float16 & double16 alignments manually in cross-compile mode])
AC_DEFINE([ALIGNOF_FLOAT16], 64)
AC_DEFINE([ALIGNOF_DOUBLE16], 128)
fi

AS_IF([test $ac_cv_alignof_float16 -gt $ac_cv_alignof_double16],
      [AC_DEFINE([MAX_EXTENDED_ALIGNMENT], [ALIGNOF_FLOAT16], [Defined to greatest expected alignment for extended types, in bytes.])],
      [AC_DEFINE([MAX_EXTENDED_ALIGNMENT], [ALIGNOF_DOUBLE16], [Defined to greatest expected alignment for extended types, in bytes.])])

AC_MSG_CHECKING([whether __fp16 supports arithmetic operations])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[__fp16 callfp16(__fp16 a) { return a * (__fp16)1.8; };]],
                                   [[__fp16 x=callfp16((__fp16)1.643);]])],
                  [
                    AC_MSG_RESULT([yes])
                    AC_DEFINE_UNQUOTED([HAVE_WORKING_HALF], 1, [Define to 1 if __fp16 supports arithmetic operations float.])
                  ],
                  [
                    AC_MSG_RESULT([no])
                    AC_DEFINE_UNQUOTED([HAVE_WORKING_HALF], 0, [Define to 1 if __fp16 supports arithmetic operations float.])
                    AC_DEFINE_UNQUOTED([_CL_DISABLE_HALF],[],[Disable cl_khr_fp16 on host based devices.])
                    HOST_CLANG_FLAGS="$HOST_CLANG_FLAGS -D_CL_DISABLE_HALF"
                  ]
                 )

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

AC_C_BIGENDIAN(
  [ AC_DEFINE([WORDS_BIGENDIAN], [1], [Endianness]) ],
  [ AC_DEFINE([WORDS_BIGENDIAN], [0], [Endianness]) ]
)


# The libtool library version string to use (passed to -version-info).
# See: http://www.nondot.org/sabre/Mirrored/libtool-2.1a/libtool_6.html
# libpocl.so should get only API additions as we are implementing a standard.
#
# The library version encodings into the library file name are platform
# dependant. Therefore we need to be a bit verbose here for the pocl.icd file 
# creation to succeed (see Makefile.am).
# Chiefly, GNU differs from BSD, and others are untested. See e.g.
# http://en.opensuse.org/openSUSE%3aShared_library_packaging_policy#Versioning_schemes
#
# 0:0:0 == 0.6
# 1:0:0 == 0.7 (not backwards compatible with 0:0:0 due to the ICD)
# 2:0:1 == 0.8 (currently backwards compatible with 0.7, thus age = 1).
# 3:0:2 == 0.9 (currently backwards compatible with 0.7, thus age = 2).
# 4:0:3 == 0.10 (currently backwards compatible with 0.7, thus age = 3).
# 5:0:4 == 0.11 (currently backwards compatible with 0.7, thus age = 4).
# 6:0:5 == 0.12 (currently backwards compatible with 0.7, thus age = 5).
LIB_CURRENT_VERSION=6
LIB_REVISION_VERSION=0
LIB_AGE_VERSION=5
LIB_FIRST_VERSION=$(($LIB_CURRENT_VERSION - $LIB_AGE_VERSION))
BUILD_TIMESTAMP=""
AC_SUBST([LIB_VERSION], ["$LIB_CURRENT_VERSION:$LIB_REVISION_VERSION:$LIB_AGE_VERSION"])
AC_SUBST([LIB_CURRENT_VERSION], [$LIB_CURRENT_VERSION])
AC_SUBST([LIB_REVISION_VERSION], [$LIB_REVISION_VERSION])
AC_SUBST([LIB_AGE_VERSION], [$LIB_AGE_VERSION])
AC_SUBST([LIB_FIRST_VERSION], [$LIB_FIRST_VERSION])
AC_SUBST([BUILD_TIMESTAMP], [$( date +%G%m%d%H%M%N )])

AC_DEFINE_UNQUOTED([POCL_BUILD_TIMESTAMP], ["$BUILD_TIMESTAMP"], [Timestamp of build])

# The kernel compiler opt plugin shared library, however, changes more 
# drastically. Let's try to follow the similar 'current' numbering as
# the pocl host API library and perhaps tune the 'revision' and 'age' later.
AC_SUBST([KERNEL_COMPILER_LIB_VERSION], ["$LIB_CURRENT_VERSION:0:0"])

AC_SUBST([LD_FLAGS_BIN])

AC_CONFIG_FILES([Makefile
                 pocl.pc
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
                 lib/CL/devices/hsa/Makefile
                 lib/llvmopencl/Makefile
                 lib/kernel/Makefile
                 lib/kernel/cellspu/Makefile
                 lib/kernel/host/Makefile
                 lib/kernel/tce/Makefile
                 lib/kernel/amdgcn/Makefile
                 lib/kernel/hsail64/Makefile
                 lib/poclu/Makefile
                 examples/Makefile
                 examples/example1/Makefile
                 examples/example1-spir64/Makefile
                 examples/example1-spir32/Makefile
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
                 examples/AMDSDK2.9/Makefile
                 examples/EinsteinToolkit/Makefile
                 examples/VexCL/Makefile
                 examples/piglit/Makefile
                 examples/Halide/Makefile
                 examples/OpenCV/Makefile
                 examples/CloverLeaf/Makefile
                 scripts/Makefile
                 tests/Makefile
                 tests/kernel/Makefile
                 tests/regression/Makefile
                 tests/workgroup/Makefile
                 tests/runtime/Makefile
                 tests/tce/Makefile
                 tests/tce/ttasim/Makefile
                 tests/tce/tcemc/Makefile
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
   echo "please consult http://pocl.sourceforge.net/docs/html/development.html#test-suite"
   echo "for details."
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
echo NOTE: Initialize the examples used by the testsuites by running 
echo \"make prepare-examples\" once.
echo
echo Basic and pthreads devices optimized for:
echo architecture: $host_cpu
echo cpu variant: $llc_host_cpu
