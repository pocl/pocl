===
HSA
===

Note: pocl's HSA support is currently in experimental stage.

The experimental HSA driver works only with an AMD Kaveri or Carrizo APUs
using the HSAIL-supported LLVM and Clang. Other than that, you will need
a recent linux (4.0+) and some software.

Installing prerequisite software
---------------------------------

1) Install the HSA AMD runtime library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Pre-built binaries can be found here:

  https://github.com/HSAFoundation/HSA-Runtime-AMD

  This usually installs into /opt/hsa. Make sure to read Q&A in README.md, it
  lists some common issues (like /dev/kfd permissions) and run sample/vector_copy
  to verify you have a working runtime.

2) Build HSAIL-Tools
~~~~~~~~~~~~~~~~~~~~~

   `git clone https://github.com/HSAFoundation/HSAIL-Tools`

   In particular **HSAILasm** executable will be required by pocl.

3) Build & install the LLVM with HSAIL support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  `git clone https://github.com/HSAFoundation/HLC-HSAIL-Development-LLVM/`

  Use the branch hsail-stable-3.7; before build, patch it with

  `pocl/tools/patches/llvm-3.7-hsail-branch.patch`

  Build it with a Clang 3.7 (branch release_37)

  `cd tools; svn co http://llvm.org/svn/llvm-project/cfe/branches/release_37 clang`

  patched with

  `pocl/tools/patches/clang-3.7-hsail-branch.patch`

  to get the HSAIL Clang support.

  An LLVM cmake configuration command like this worked for me:

  `mkdir build; cd build; cmake .. -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=HSAIL
  -DBUILD_SHARED_LIBS=off -DCMAKE_INSTALL_PREFIX=INSTALL_DIR -DLLVM_ENABLE_RTTI=on
  -DLLVM_BUILD_LLVM_DYLIB=on -DLLVM_ENABLE_EH=ON`

  Note that these are **required** :

  `-DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=HSAIL`

  Also, if you don't want to build all the default targets, you'll need AMDGPU.


4) Build pocl.
~~~~~~~~~~~~~~~
  Using autotools:

    `./configure --with-hsa-runtime-dir=\</opt/hsa\>
    LLVM_CONFIG=<hsail-built-llvm-dir>/bin/llvm-config
    HSAILASM=\<path/to/HSAILasm\>`

  Or using cmake:
    `cmake -DENABLE_HSA=ON -DWITH_HSA_RUNTIME_DIR=\</opt/hsa\>
    -DWITH_HSAILASM_PATH=\<path/to/HSAILasm\>`

  Both should result in "hsa" appearing in pocl's targets to build ("OCL_TARGETS"
  in cmake output, "Enabled device drivers:" in autoconf output)

5) Run tests & play around
~~~~~~~~~~~~~~~~~~~~~~~~~~~

  After building pocl, you can smoke test the HSA driver by executing the HSA
  tests of the pocl testsuite:

  `make check TESTSUITEFLAGS="-k hsa"`


HSA Support notes
------------------
Note that the support is still experimental and very much unfinished. You're
welcome to try it out and report any issues, though.

What's implemented:
 * global/local/private memory
 * atomics, barriers
 * most of the OpenCL kernel library builtins

What's missing
 * printf() is not implemented
 * several builtins are not implemented yet (erf(c), lgamma, tgamma,
   logb, remainder, nextafter) and some are suboptimal or may give incorrect
   results with under/overflows (e.g. hypot, length, distance). We're working on
   this, if you find any problem  please let us know)
 * image support is not implemented
 * OpenCL 2.0 features (SVM) are unimplemented
 * Performance is suboptimal in many cases
