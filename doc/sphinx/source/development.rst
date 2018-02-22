Information for Developers
==========================

Using cmake to build & install pocl
-----------------------------------

Most of the important stuff on using cmake is in the install document,
see :ref:pocl-install

A few additional items:

The

     export OCL_ICD_VENDORS="PATH_TO_THE_POCL_BUILD_TREE/ocl-vendors"

command must point to ocl-vendors in the  cmake *build* directory, not the
pocl source directory.

You can run the tests or built examples using "ctest" directly;
``ctest --print-labels`` prints the available labels (testsuites);
Invoke ctest with -jX option to run X tests in parallel.

"make check_tier1" will invoke ctest with tier-1 testsuites.
See :ref:`maintenance-policy` for details.

Testsuite
----------

Before changes are committed to the mainline, all tests in the 'make
check' tier-1 suite should pass::

   make check_tier1

Under the 'examples' directory there are placeholder directories for
external OpenCL application projects which are used as test suites for
pocl (e.g. ViennaCL). These test suites can be enabled for cmake
with -DENABLE_TESTSUITES (you can specify a list of test suites
if you do not want to enabled all of them, see configure help for the
available list).  Note that these additional test suites require
additional software (tools and libraries). The configure script checks
some of them but the check is not exhautive. Test suites are disabled if
their requirement files are not available.

In order to prepare the external OpenCL examples for the testsuite, you
need to run the following build command once::

   make prepare_examples

IMPORTANT: using the ICD for in tree 'make check' requires an icd
loader that allows overriding the icd search path. Other ICD loaders
wont be able to work in tree (they require the ICD config file to be
installed in the system).  There are now two options for such a loder:
the open source ocl-icd loader and the Khronos supplied loader with a
patch applied.

Debugging a Failed Test
^^^^^^^^^^^^^^^^^^^^^^^

If there are failing tests in the suite, the usual way to start
debugging is to look what was printed to the logs for the failing
cases. After running the test suite, the logs are stored under
``Testing/Temporary/*.log`` Or one could re-run the test with more
verbose output. Useful ctest options are "-V" and "--output-on-failure";
to make pocl more chatty, use the POCL_DEBUG env variable.

Ocl-icd
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

Coding Style
------------

The code base of pocl consists most of pure C sources and C++ sources (mostly
the kernel compiler).

1) In the C sources, follow the GNU style.

   http://www.gnu.org/prep/standards/html_node/Writing-C.html

2) In the C++ sources (mostly the LLVM passes), follow the LLVM coding
   guidelines so it is easier to upstream general code to the LLVM project
   at any point.

   http://llvm.org/docs/CodingStandards.html

It's acknowledged that the pocl code base does fully not adhere to these
principles at the moment, but the aim is to gradually fix the style and any
new code merged to master should adhere to them.

Khronos ICD Loader
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

Using pocl from the Build Tree
------------------------------

If you want use the pocl from the build tree, you must export
POCL_BUILDING=1 so pocl searches for its utility scripts from the
build tree first, then the installation location. The "make check"
testsuite does this automatically.

There's a helper script that, when sourced, in addition to setting
POCL_BUILDING setups the OCL_ICD_VENDORS path to point to the pocl in
the build tree. This removes the need to install pocl to test the
built version. It should be executed in the build root, typically::

  . ../tools/scripts/devel-envs.sh

Target and Host CPU Architectures for 'basic' and 'pthread' Devices
-------------------------------------------------------------------

By default, pocl build system compiles the kernel libraries for
the host CPU architecture, to be used by 'basic' and 'pthread' devices.

LLVM is used to detect the CPU variant to be used as target. This 
can be overridden by passing -DLLC_HOST_CPU=... to CMake. See the
documentation for LLC_HOST_CPU build option.

Cross-compilation where 'build' is different from 'host' has not been
tested.
Cross-compilation where 'host' is a different architecture from 'target'
has not been tested for 'basic' and 'pthread' devices. 

Writing Documentation
---------------------

The documentation is written using the `Sphinx documentation generator 
<http://sphinx-doc.org/>`_ and
the reStructuredText markup.

This Sphinx documentation can be built by::

  cd doc/sphinx
  make html

This builds the html version of the documents under the 'build/html' directory.
