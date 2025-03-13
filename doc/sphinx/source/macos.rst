.. _pocl-macos-setup:

Setting up and running PoCL on MacOS
------------------------------------

Note about the kernel compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clang/LLVM is included with Xcode, but at least the default
installation lacks development headers/libraries and llvm-config.
As a result, this version cannot be used as a kernel compiler for PoCL.

The simplest way to install llvm is through Homebrew::

    brew install llvm
    export PATH=/opt/homebrew/opt/llvm/bin:$PATH

Then, ensure that LLVM is correctly set up for PoCL::

    which clang
    /opt/homebrew/opt/llvm/bin/clang
    llvm-config --version
    19.1.7

Alternatively, you can compile LLVM from source (Example is for an ARM Mac)::

    git clone https://github.com/llvm/llvm-project
    cd llvm-project
    mkdir build && cd build
    cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_ENABLE_PROJECTS="clang;llvm"
    -DCMAKE_INSTALL_PREFIX=<path-to-installation-directory> -DLLVM_TARGETS_TO_BUILD="AArch64" ../llvm

    ninja install

Installing PoCL on MacOS using pre-built binaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Homebrew**

PoCL with the CPU driver supports Intel and Apple Silicon chips can be found on
homebrew and can be installed with::

    brew install pocl

Note that this installs an ICD loader from KhronoGroup and the built-in OpenCL
implementation will be invisible when your application is linked to this loader.

**Conda**

PoCL with the CPU driver supports Intel and Apple Silicon chips can be found on
conda-forge distribution and can be installed with::

    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash Mambaforge-$(uname)-$(uname -m).sh

To install the CPU driver::

    mamba install pocl

Note that this installs an ICD loader from KhronosGroup and the builtin OpenCL implementation will be invisible when your application is linked to this loader. To make both pocl and the builtin OpenCL implementaiton visible, do::

    mamba install pocl ocl_icd_wrapper_apple


Building PoCL from source on MacOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure that all required dependencies are installed.
Clang/LLVM must be properly set up (see above).

Get the sources::

    git clone git@github.com:pocl/pocl.git
    cd pocl
    mkdir build && cd build

For a standard build without the ICD loader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    ::

        cmake .. -G Ninja -DENABLE_ICD=OFF -DCMAKE_INSTALL_PREFIX=<path-to-installation-directory>
        ninja install

    This will install ``libOpenCL.dylib`` to::

        <path-to-installation-directory>/lib

    **Usage:**

    To override the MacOS OpenCL framework::

        export LIBRARY_PATH=<path-to-installation-directory>/lib:$LIBRARY_PATH
        export DYLD_LIBRARY_PATH=<path-to-installation-directory>/lib:$DYLD_LIBRARY_PATH
        clang <program-source>.c -lOpenCL

        // Use PoCL's debugging functionality to ensure it runs through PoCL.
        POCL_DEBUG=all ./a.out

For a build with the ICD loader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    ::

        // If not installed:
        brew install ocl-icd
        brew install opencl-headers

        // These should enable PoCL to automatically detect the ICD loader.
        export PKG_CONFIG_PATH="/opt/homebrew/opt/opencl-headers/share/pkgconfig":$PKG_CONFIG_PATH
        export PKG_CONFIG_PATH="/opt/homebrew/opt/ocl-icd/lib/pkgconfig":$PKG_CONFIG_PATH
        export CPATH=/opt/homebrew/opt/ocl-icd/include:$CPATH

        cmake .. -G Ninja -DCMAKE_INSTALL_PREFIX=<path-to-installation-directory>
        ninja install


    This will install ``libpocl.dylib`` to::

        <path-to-installation-directory>/lib

    Make it visible to the ICD loader by setting::

        export OCL_ICD_VENDORS=<path-to-installation-directory>/etc/OpenCL/vendors

    **Usage:**

    To override the MacOS OpenCL framework::

        export LIBRARY_PATH=/opt/homebrew/opt/ocl-icd/lib:$LIBRARY_PATH
        clang <program-source>.c -lOpenCL

        // Use PoCL's debugging functionality to ensure it runs through PoCL.
        POCL_DEBUG=all ./a.out


