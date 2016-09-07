===
HSA
===

Note: pocl's HSA support is currently in experimental stage.

The experimental HSA driver can work in three ways:

1) with AMD Kaveri or Carrizo APUs, using AMD's original HSA Runtime (version 1.0) with the HSAIL-enabled LLVM+Clang

2) with AMD Kaveri or Carrizo APUs, using AMD's ROCm HSA Runtime (1.1 / 1.2) with AMD's LLVM+Clang and AMD's libclc

3) generic HSA Agent support (e.g. for your CPU) using the phsa project

Installing prerequisite software for AMD
------------------------------------------

1) Install an HSA AMD runtime library implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  1) **For AMD + original HSA**, pre-built binaries can be found here:

      https://github.com/HSAFoundation/HSA-Runtime-AMD

    This usually installs into ``/opt/hsa``. Make sure to read Q&A in README.md (it
    lists some common issues (like ``/dev/kfd`` permissions) and run sample/vector_copy
    to verify you have a working runtime.

  2) **For AMD + ROCm runtime**, pre-built binaries for Fedora & Ubuntu are here:

      https://github.com/RadeonOpenCompute/ROCm

    This usually installs into ``/opt/rocm/hsa``.

  3) **For phsa** to add generic HSA support on gcc-supported CPU,
     installation instructions are here:

      https://github.com/HSAFoundation/phsa

2) Build & install the LLVM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  1) **For AMD + original HSA**, you'll need to build LLVM with HSAIL support

    Fetch the HSAIL branch of LLVM 3.7::

      git clone https://github.com/HSAFoundation/HLC-HSAIL-Development-LLVM/ -b hsail-stable-3.7

    Patch it a bit with::

      cd HLC-HSAIL-Development-LLVM; patch -p1 < PATHTO-POCL/tools/patches/llvm-3.7-hsail-branch.patch

    Fetch the upstream Clang 3.7 branch::

      cd tools; svn co http://llvm.org/svn/llvm-project/cfe/branches/release_37 clang

    Patch it also::

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

  2) **For AMD + ROCm runtime**, pre-built llvm binaries for Ubuntu may not work
     (i got some undefined LLVM symbols in libpocl.so when i linked against
     AMD's LLVM on Ubuntu 16.04; YMMV) so you might have to rebuild it.

     Fetch one of roc-x.x.x branches of AMD llvm:

      https://github.com/RadeonOpenCompute/llvm

    then ``cd`` into ``tools`` directory and fetch corresponding clang branch:

      https://github.com/RadeonOpenCompute/clang

    and build LLVM with (at least) amdgpu and x86 targets.

  3) **For phsa**, follow the steps in 1) to build LLVM with HSAIL support

3) Get HSAIL-Tools
~~~~~~~~~~~~~~~~~~~~~

  1) This step is **only required for AMD + original HSA, and PHSA**.
     pocl's ROCm use does not require HSAIL-Tools.

    Clone the repo::

      git clone https://github.com/HSAFoundation/HSAIL-Tools

    Then either copy ``HSAILasm`` executable to /opt/hsa/bin, or give
    the path to ``HSAILasm`` on the build command line (see below)

4) Build pocl
~~~~~~~~~~~~~

  1) **For AMD + original HSA**, build using cmake::

      mkdir build ; cd build
      cmake -DENABLE_HSA=ON -DWITH_HSA_RUNTIME_DIR=</opt/hsa> \
      -DWITH_HSAILASM_PATH=<path/to/HSAILasm> -DSINGLE_LLVM_LIB=off ..

    It should result in "hsa" appearing in pocl's targets to build. ``-DSINGLE_LLVM_LIB=off``
    workarounds an LLVM 3.7 build system issue.

  2) **For AMD + ROCm HSA**, build using cmake::

      mkdir build ; cd build
      cmake -DHSA_RUNTIME_IS_ROCM=ON -DENABLE_HSA=ON \
      -DWITH_HSA_RUNTIME_DIR=</opt/rocm/hsa> -DWITH_LLVM_CONFIG=<path-to-amd-llvm> ..

    It should result in "hsa" appearing in pocl's targets to build.

  3) **For phsa**, same as 1)

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
