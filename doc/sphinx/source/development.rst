General information for pocl developers
=======================================

configuring
-----------

If you checked out a development version of pocl, the configuration
scripts need to be regenerated. This is achieved by issuing the
command::

    ./autogen.sh

in the root of the source tree. You will need a decent version of GNU
autotools, usually installable from distribution packages 'automake',
'autoconf', and 'libtool'.

Once that is done, the usual GNU build commands build pocl. Builds out
of source directory are supported. We recommend using::

    ./configure --enable-debug 
    make

This will build pocl without optimization, which simplifies debugging.
(This does not influence whether pocl will optimize the code that it
generates from OpenCL source files.)

The configure script will use special environment variables, if
present or passed in the command line such as:

==================== ===========
Environment variable Description
==================== ===========
CLANG                Program to compile kernels to bytecode 
CLFLAGS              Flags to be used when compiling CL sources 
TARGET_CLANG_FLAGS   Parameters to for target compilation.  
HOST_CLANG_FLAGS     Parameters to for host compilation.  
==================== ===========

All such special environment variables can be seen with the --help
option of the configure script.

using LLVM svn
--------------

It's highly recommended to use the latest development version of LLVM
when developing pocl.

However, as llvm-svn is a moving target, new revisions might break
pocl compilation. The latest LLVM trunk revision pocl has been tested
successfully with is:

**180999**

The test suite (make check) should pass with this revision with xfails
used to mark known-broken tests.

test suite
----------

Before changes are committed to the mainline, all tests in the 'make
check' suite should pass. As a minimum requirement the short test
suite should be executed before committing as follows::

   make check TESTSUITEFLAGS="-k \!long"

Under the 'examples' directory there are placeholder directories for
external OpenCL application projects which are used as test suites for
pocl (e.g. ViennaCL). These test suites can be enabled at configure
time with --enable-testsuites (you can specify a list of test suites
if you do not want to enabled all of them, see configure help for the
available list).  Note that these additionnal test suites require
additionnal software (tools and libraries). The configure script check
some of them but the check is not exhautive (patch welcome). Test
suites are disabled if their requirement is not available.

The pocl OpenCL implementation can be used directly or through an ICD
loader.  The --enable-tests-with-icd configure option allows to choose
how tests are linked to pocl when running the 'make check' target. By
default, if the ICD is built, tests are done through the ICD loader.

IMPORTANT: using the ICD for in tree 'make check' requires an icd
loader that allows overriding the icd search path. Other ICD loaders
wont be able to work in tree (they require the ICD config file to be
installed in the system).  There are now two options for such a loder:
the open source ocl-icd loader and the Khronos supplied loader with a
patch applied.

ocl-icd
-------

ocl-icd can be downloaded from
https://forge.imag.fr/projects/ocl-icd/. It allows overriding the path
from which the icd files are searched which is used to select only the
OpenCL library in the build tree of pocl for the make check. Note,
however, if you run the tests or examples manually this overriding is
not done automatically. To direct the ocl-icd to use only the pocl in
the build tree, export the following environment variable in your
shell::

  export OCL_ICD_VENDORS="PATH_TO_THE_POCL_BUILD_TREE/ocl-vendors"

Inside the 'ocl-vendors' directory there's a single .icd file which is
generated to point to the pocl library in the build tree.

khronos ICD loader
------------------

The ICD loader supplied by Khronos can be used for pocl development by
applying a minor patch that enables overriding the ICD search path as
explained above (OCL-ICD).

The steps to build and install the Khronos ICD loader so it can be
used to run the pocl test suite:

#. Download the loader from http://www.khronos.org/registry/cl Unpack
   it. Copy the OpenCL headers to inc/CL like instructed in
   inc/README.txt.
#. Apply a patch from the pocl checkout::
     cd icd

     patch -p1 < ~/pocl/tools/patches/khronos-icd-loader.patch

#. Build it with 'make'.
#. Copy the loader to a library search path: sudo cp bin/libOpenCL* /usr/lib

Now it should use the Khronos loader for ICD dispatching and you (and
the pocl build system) should be able to override the icd search path
with OCL_ICD_VENDORS environment variable.

Using pocl from the build tree
------------------------------

If you want use the pocl from the build tree, you must export
POCL_BUILDING=1 so pocl searches for its utility scripts from the
build tree first, then the installation location. The "make check"
testsuite does this automatically.

To test as much as possible link options, it is recommended to
configure pocl two times and run "make check" with both. One should be
configurated with::

  $src/configure --enable-icd --disable-direct-linkage ...

And the second one with::

  $src/configure --disable-icd --enable-direct-linkage ...

Target and host CPU architectures for 'basic' and 'pthread' devices
-------------------------------------------------------------------

By default, pocl build system compiles the kernel libraries for
the host CPU architecture, to be used by 'basic' and 'pthread' devices.

LLVM is used to detect the CPU variant to be used as target. This 
can be overridden by passing LLC_HOST_CPU to './configure'.
Valid options are best documented in the output of::

  llvm-as /dev/null | llc -mcpu=help

Cross-compilation where 'build' is different from 'host' has not been
tested.
Cross-compilation where 'host' is a different architecture from 'target'
has not been tested for 'basic' and 'pthread' devices. 

Writing documentation
---------------------

The documentation is written using the `Sphinx documentation generator 
<http://sphinx-doc.org/>`_ and
the reStructuredText markup.

This Sphinx documentation can be built by::

  cd doc/sphinx
  make html

This builds the html version of the documents under the 'build/html' directory.
