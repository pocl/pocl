.. _maintenance-policy:

Maintenance Policy
==================

pocl development is currently managed mostly by researchers and
research assistants of the `Customized Parallel Computing <http://cpc.cs.tut.fi>`_
group of Tampere University of Technology. We provide general maintenance
for pocl on the side of our research projects (which on the other hand might use
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

Currently (2017-03-16) the following are included in the tier-1 test suites:

* The standard test suite of pocl.
* AMD SDK 3.0 test suite
* PyOpenCL test suite
* piglit test suite
* conformance_suite_micro test suite
* CLBlast tests (excluding the longest running ones)
* HSA test suite (uses the LLVM 3.7 with an HSAIL backend and targets an AMD Kaveri GPU)
* TCE short smoke test suite (against the latest TCE open source release)

Please note that not neccessarily all the tests currently pass in the suites,
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
(the CPC group). Any community involvement to provide a wider support/maintance
level will be heartily welcomed.

