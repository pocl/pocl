Information for Pocl developers
===================================

Testsuite
----------

Before changes are committed to the mainline, all tests in the 'make
check' tier-1 suite should pass::

   make check_tier1

"make check_tier1" will invoke ctest with tier-1 testsuites. See
 `maintenance-policy`_ for list of what's included in tier-1.

Under the 'examples' directory there are placeholder directories for
external OpenCL application projects which are used as test suites for
pocl (e.g. ViennaCL). These test suites can be enabled for cmake
with -DENABLE_TESTSUITES (you can specify a list of test suites
if you do not want to enabled all of them, see configure help for the
available list).  Note that these additional test suites require
additional software (tools and libraries). The configure script checks
some of them but the check is not exhaustive. Test suites are disabled if
their requirement files are not available.

You can run the tests or built examples using "ctest" directly;
``ctest --print-labels`` prints the available labels (testsuites);
Invoke ctest with -jX option to run X tests in parallel.

In order to prepare the external OpenCL examples for the testsuite, you
need to run the following build command once::

   make prepare_examples

IMPORTANT: using the ICD for in tree 'make check' requires an icd
loader that allows overriding the icd search path. Other ICD loaders
wont be able to work in tree (they require the ICD config file to be
installed in the system).  There are now two options for such a loader:
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

Ocl-icd is packaged for most popular linux distributions,
but can also be downloaded from:

https://forge.imag.fr/projects/ocl-icd/.

It allows overriding the path from which the icd files
are searched which is used to select only the
OpenCL library in the build tree of pocl for the make check. Note,
however, if you run the tests or examples manually this overriding is
not done automatically. To direct the ocl-icd to use only the pocl *in
the build tree*, export the following environment variable in your
shell::

  export OCL_ICD_VENDORS="PATH_TO_THE_POCL_BUILD_TREE/ocl-vendors"

Inside the 'ocl-vendors' directory there's a single .icd file which is
generated to point to the pocl library in the build tree.

Coding Style
------------

The code base of pocl consists most of pure C sources and C++ sources.

1) In the C sources, follow the GNU C style, but with spaces for indent.

   The GNU C style guide is here: http://www.gnu.org/prep/standards/html_node/Writing-C.html

   This guide should be followed except please use 2 spaces instead of the
   confusing "smart" mix of tabs and spaces for indentation.

2) In the C++ sources (mostly the LLVM passes), follow the LLVM coding
   guidelines so it is easier to upstream general code to the LLVM project
   at any point.

   http://llvm.org/docs/CodingStandards.html

It's acknowledged that the pocl code base does not fully adhere to these
principles at the moment, but the aim is to gradually fix the style with
every new commit improving the style.

There are clang-format scripts to help in getting the style gradually
improved. Running ``tools/scripts/format-branch.sh`` in the root of
the repository diffs against a ``master`` branch and formats the difference,
and leaves the diff uncommitted in the working tree.
``tools/scripts/format-last-commit.sh`` formats only the last commit and can be
used in an interactive rebase session.

An example emacs configuration to help get the pocl code style correct::

  (setq default-tab-width 2)
  (setq-default indent-tabs-mode nil)
  (setq-default show-trailing-whitespace t)
  
  (defun my-c-mode-common-hook ()
    (c-set-style "gnu")
    (setq tab-width 2)
    (setq c-basic-offset 2)
  )
  (add-hook 'c-mode-common-hook 'my-c-mode-common-hook)
  
  (defun my-cpp-mode-common-hook ()
    (c-set-style "stroustrup")
    (setq tab-width 4)
    (setq c-basic-offset 4)
    )
  (add-hook 'c++-mode-hook 'my-cpp-mode-common-hook)
  
  (add-to-list 'auto-mode-alist '("\\.cl$" . c-mode))
  (add-to-list 'auto-mode-alist '("\\.icc$" . c++-mode))
  (add-to-list 'auto-mode-alist '("\\.cc$" . c++-mode))

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


.. _maintenance-policy:

Maintenance Policy
-------------------

pocl development is currently managed mostly by researchers and
research assistants of the `Customized Parallel Computing <https://tuni.fi/cpc>`_
group of Tampere University. We provide general maintenance for pocl
on the side of our research projects (which on the other hand might use
and/or extend it) because we consider it an important project that helps the
"heterogeneous parallel programming cause". However, doing maintenance "on the
side" unfortunately means that there is limited time to respond to external
support requests due to other activities.

To make pocl maintenance feasible within our limited time, we have set the following
policy regarding releases:
**External projects using OpenCL that have a test suite included in "regularly
tested suites" (we later call 'tier-1' test suites) will be kept regression free,
but for the rest we cannot make any promises.**

Tier-1 tests will be executed successfully before the lead developer pushes
new pull requests (PR) to the master branch, and some of them are additionally
executed with multiple continuous integration (buildbot) servers on
different platforms. Active developers are also assumed to run them locally
before submitting PRs. Thus, regressions on these suites should be detected
early. The required testsuites can be enabled at buildtime with
``-DENABLE_TESTSUITES=tier1`` cmake option.

Currently (2022-11-07) the following are included in the tier-1 test suites:

* The standard test suite of pocl.
* PyOpenCL test suite
* piglit test suite
* conformance_suite_micro test suite
* OpenASIP short smoke test suite (against the latest OpenASIP open source release)

Please note that not necessarily all the tests currently pass in the suites,
we just ensure the currently passing ones do not regress with new
commits (expected failing ones are marked as XFAILs or skipped).
The primary test platform is x86-64.

The latest LLVM release is given priority when testing, and we cannot
guarantee older LLVM versions keep working over pocl releases due to
the constantly changing library API.

If you would like get your favourite OpenCL-using project's test
suite included in the tier-1 suite, please send a pull request that
adds the suite under the 'examples' dir and the main CMakeLists.txt along with
instructions (a README will do) on how to setup it so it is included in
the 'make check' run. Please make the test suite short enough to be suitable for
frequent "smoke testing" (under 5 minutes per typical run preferred).
If your favourite project is already under 'example', but not listed as a tier-1
test suite, please update its status so that 'make check' passes with the current
HEAD of pocl and let us know, and we do our best to add it.

Naturally this policy/support promise concerns only the lead developers
(the CPC group). Any community involvement to provide a wider support/maintenance
level will be heartily welcomed.

.. _releasing:

Release management
----------------------------------

We aim to make a new release according to the Clang/LLVM release schedule.


For each release, a release manager is assigned. Release manager is responsible
for creating and uploading new release candidate tar balls and requesting for
testers from different platforms. After a release candidate round with
success reports and no failure reports, a release is published.

See the `maintenance-policy`_ for the current release criteria.

A checklist and hints for testing and making a release successfully:

* Check that CHANGES has the most interesting updates done during the release
  cycle. Add missing notable changes from git log.

* Update the release notes in *doc/notes-VERNUM.txt*.

* Create a single commit in master branch: change the version to the
  release one (without -pre), in all relevant places (CHANGES, docs,
  CMakeLists.txt, etc); update the .so version (if required);
  check that supported LLVM versions in cmake/LLVM.cmake are correct.
  Create the release branch from this commit and push it to github.

* In the master branch, create a new commit: increase version
  number (with -pre) in all relevant places; update the .so version;
  increase the supported LLVM versions in cmake/LLVM.cmake.
  Commit, push master to github. Now development can go on in master
  while the release branch is being stabilized.

* The previous two steps ensure that merge-base of release & master is
  the start of release branch, which ensures that merging release
  to the master will not screw up the version numbers in the master.
  Bugs which need to be fixed in both branches, should be committed to
  the release branch, then release branch merged to master.

* Create a new release on Github. Mark it as pre-release. This should
  create both a tarball and a git tag.

* Upload the package to portablecl.org/downloads via SFTP or to the
  sourceforge file listing for the pocl project.

* Request for testers in Twitter and/or mailing list. Point the testers to
  send their test reports to you privately or by adding them to the wiki.
  A good way is to create a wiki page for the release schedule and a test
  log. See https://github.com/pocl/pocl/wiki/pocl-0.10-release-testing for
  an example.

* To publish a release, create a new release on Github without the
  checking the pre-release checkbox.
  Upload the tar ball to the sourceforge download page and
  to http://portablecl.org/downloads.
* Update the CHANGES and ANNOUNCEMENT text files in these directories.
  ANNOUNCEMENT is a copy of the latest release notes. A direct link to it can
  be easily circulated in IRC, for example.
* Update the http://portablecl.org web page with the release information.
* Advertise everywhere you can. At least in Twitter and the mailing list.

In case of any problems, ask any previous release manager for help.
Previous releases were managed by the following pocl developers:

* 0.14: Pekka Jääskeläinen
* 0.11: Michal Babej
* 0.10: Pekka Jääskeläinen
* 0.9: Kalle Raiskila
* 0.8: Erik Schnetter
* 0.6 and 0.7: Pekka Jääskeläinen
