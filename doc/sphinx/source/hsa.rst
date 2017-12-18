===
HSA
===

Note: pocl's HSA support is currently in experimental stage.

The experimental HSA driver works with AMD Kaveri or Carrizo APUs using
an AMD's HSA Runtime implementation using the HSAIL-supported LLVM and Clang.
Also, generic HSA Agent support (e.g. for your CPU) can be enabled using
the phsa project.

Installing prerequisite software
---------------------------------

1) Install an HSA AMD runtime library implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  For AMD devices, pre-built binaries can be found here:

  https://github.com/HSAFoundation/HSA-Runtime-AMD

  This usually installs into /opt/hsa. Make sure to read Q&A in README.md (it
  lists some common issues (like /dev/kfd permissions) and run sample/vector_copy
  to verify you have a working runtime.

  Alternatively, you can use *phsa* to add generic HSA support on your gcc-supported
  CPU. Its installation instructions are here:

  https://github.com/HSAFoundation/phsa

2) Build & install the LLVM with HSAIL support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  Fetch the HSAIL branch of LLVM 3.7::

    git clone https://github.com/HSAFoundation/HLC-HSAIL-Development-LLVM/ -b hsail-stable-3.7

  Fetch the upstream Clang 3.7 branch::

    cd HLC-HSAIL-Development-LLVM/tools
    svn co http://llvm.org/svn/llvm-project/cfe/branches/release_37 clang

  Patch it::

    cd clang; patch -p0 < PATHTO-POCL/tools/patches/clang-3.7-hsail-branch.patch

  An LLVM cmake configuration command like this worked for me::

    cd ../../
    mkdir build
    cd build
    cmake .. -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=HSAIL \
    -DBUILD_SHARED_LIBS=off -DCMAKE_INSTALL_PREFIX=INSTALL_DIR \
    -DLLVM_ENABLE_RTTI=on -DLLVM_BUILD_LLVM_DYLIB=on -DLLVM_ENABLE_EH=ON -DHSAIL_USE_LIBHSAIL=OFF

  ``-DHSAIL_USE_LIBHSAIL=OFF`` is only for safety. If you accidentally build clang with libHSAIL,
  it will cause mysterious link errors later when building pocl.

  Change the INSTALL_DIR to your installation location of choice. Note that these are **required**::

    -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=HSAIL

  Also, if you don't want to build all the default targets, you'll need AMDGPU.

  Then build and install the Clang/LLVM::

    make -j4 && make install


3) Get HSAIL-Tools
~~~~~~~~~~~~~~~~~~~~~

   Clone the repo::

     git clone https://github.com/HSAFoundation/HSAIL-Tools

   Then either copy ``HSAILasm`` executable to /opt/hsa/bin, or give
   the path to ``HSAILasm`` on the build command line (see below)

4) Build pocl
~~~~~~~~~~~~~

  Using cmake::

    mkdir build ; cd build
    cmake -DENABLE_HSA=ON -DWITH_HSA_RUNTIME_DIR=\</opt/hsa\> \
    -DWITH_HSAILASM_PATH=\<path/to/HSAILasm\> -DSINGLE_LLVM_LIB=off ..

  It should result in "hsa" appearing in pocl's targets to build. ``-DSINGLE_LLVM_LIB=off``
  workarounds an LLVM 3.7 build system issue.

5) Run tests & play around
~~~~~~~~~~~~~~~~~~~~~~~~~~~

  After building pocl, you can smoke test the HSA driver by executing the HSA
  tests of the pocl testsuite::

    ../tools/scripts/run_hsa_tests

HSA Support notes
------------------

Note that the support is still experimental and very much unfinished. You're
welcome to try it out and report any issues, though.

For more details, see :ref:`hsa-status`
