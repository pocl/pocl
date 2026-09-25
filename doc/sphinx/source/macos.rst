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

To make both the Apple GPU ICD and the PoCL CPU ICD visible—i.e., to enable dual devices—follow these steps (using an Apple Silicon device as an example). Run the following commands in a terminal::

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    xcode-select --install
    brew install autoconf automake libtool git
    brew install ocl-icd pocl

After installing `ocl-icd`, create a symbolic link so that `/opt/homebrew/lib/libOpenCL.dylib` exists (needed for linking)::

    # Find the actual library version (e.g., 2.3.5)
    ls /opt/homebrew/Cellar/ocl-icd/*/lib/libOpenCL*.dylib
    # Create the symlink (adjust the version number if necessary)
    sudo ln -sf /opt/homebrew/Cellar/ocl-icd/2.3.5/lib/libOpenCL.1.dylib /opt/homebrew/lib/libOpenCL.dylib

Build and Install the Apple Wrapper::

    git clone https://github.com/octaveoclx/ocl_icd_wrapper.git
    cd ocl_icd_wrapper
    autoreconf -ivf
    ./configure
    make LDFLAGS+="-framework OpenCL"

If `/usr/local/lib` does not exist, create it::

    sudo mkdir -p /usr/local/lib
    sudo chmod 755 /usr/local/lib

Then copy the wrapper libraries::

    sudo cp .libs/libocl_icd_wrapper.0.dylib /usr/local/lib/
    sudo cp .libs/libocl_icd_wrapper.dylib /usr/local/lib/

Configure ICD Vendor Files (Critical Step), where the ICD loader (`ocl-icd`) searches `/opt/homebrew/etc/OpenCL/vendors/` by default (not `/etc/OpenCL/vendors/`)::

    sudo mkdir -p /opt/homebrew/etc/OpenCL/vendors
    echo "/usr/local/lib/libocl_icd_wrapper.dylib" | sudo tee /opt/homebrew/etc/OpenCL/vendors/apple.icd
    echo "/opt/homebrew/lib/libpocl.2.dylib" | sudo tee /opt/homebrew/etc/OpenCL/vendors/pocl.icd
    sudo chmod 644 /opt/homebrew/etc/OpenCL/vendors/*.icd

Recompile `clinfo` to an ICD‑aware Version (Optional but Recommended): Clone the `clinfo` repository and compile it to link against the ICD loader::

    git clone https://github.com/Oblomov/clinfo.git
    cd clinfo
    clang -I/opt/homebrew/include -o clinfo src/clinfo.c /opt/homebrew/lib/libOpenCL.dylib

Now test that both platforms are visible::

    ./clinfo -l

Expected output::

    Platform #0: Apple
     `-- Device #0: Apple M2 Max
    Platform #1: Portable Computing Language
     `-- Device #0: cpu

Set Environment Variables (Required for Every Session): Add the following lines to your `~/.zshrc` (or `~/.bash_profile`) to make them permanent::

    export DYLD_LIBRARY_PATH=/opt/homebrew/lib
    export OCL_ICD_VENDORS=/opt/homebrew/etc/OpenCL/vendors

Then apply::

    source ~/.zshrc

Now any OpenCL program that links against `libOpenCL.dylib` will see both the Apple GPU and the PoCL CPU.

Fix `ld: library 'System' not found` When Using Homebrew GCC: If you have previously used Homebrew’s `gcc` and `g++` (e.g., `gcc@11`), you may encounter the linker error::

    ld: library 'System' not found

This happens because Homebrew’s GCC cannot find the macOS system libraries by default. To solve it, **force the use of Apple’s Clang** and explicitly set the SDK path. This is especially important when compiling PoCL from source or any other OpenCL project that uses `-lOpenCL`.

Temporary fix (for one terminal session)::

    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export LDFLAGS="-L/usr/lib -lSystem"
    export CPATH="/usr/include"
    export LIBRARY_PATH="/usr/lib"

Permanent fix (add to `~/.zshrc`)::

    echo 'export CC=/usr/bin/clang' >> ~/.zshrc
    echo 'export CXX=/usr/bin/clang++' >> ~/.zshrc
    echo 'export LDFLAGS="-L/usr/lib -lSystem"' >> ~/.zshrc
    echo 'export CPATH="/usr/include"' >> ~/.zshrc
    echo 'export LIBRARY_PATH="/usr/lib"' >> ~/.zshrc
    source ~/.zshrc

After applying these environment variables, you can reinstall PoCL from source (if needed) or any other software that requires system libraries::

    brew reinstall --build-from-source pocl

All regular `brew install`, `brew reinstall`, and other commands will automatically inherit these settings.

Verify the Dual‑Device Setup in your own OpenCL program (Optional): If you intend to compile your own OpenCL program, make sure to link against the ICD loader by using::

    -L/opt/homebrew/lib -lOpenCL

and **remove** any occurrence of `-framework OpenCL` from your build flags.

For example, a minimal compilation command for a source file `my_program.c` would be::

    clang -I/opt/homebrew/include -L/opt/homebrew/lib -lOpenCL my_program.c -o my_program

Then run the program with the required environment variables::

    export DYLD_LIBRARY_PATH=/opt/homebrew/lib
    export OCL_ICD_VENDORS=/opt/homebrew/etc/OpenCL/vendors
    ./my_program

Your program should detect both the Apple GPU and the PoCL CPU as separate OpenCL platforms/devices.

If you are using a build system (Makefile, CMake, etc.), ensure that `-L/opt/homebrew/lib -lOpenCL` is used instead of `-framework OpenCL`.

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

        # Use PoCL's debugging functionality to ensure it runs through PoCL.
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

        # Use PoCL's debugging functionality to ensure it runs through PoCL.
        POCL_DEBUG=all ./a.out


