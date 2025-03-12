==========================================
Using PoCL as the OpenCL backend for DPC++
==========================================

SYCL is programming model that enables single-source C++ development for
heterogenous computing. Compared to OpenCL, SYCL operates at a higher level
of abstraction, relying on lower-level implementations like OpenCL for device
offloading.

If the OpenCL backend is used, the SYCL runtime translates the SYCL API calls
into corresponding OpenCL API calls and forwards them to the OpenCL runtime.

DPC++ is Intel's implementation of SYCL, and the toolchain flow is as follows:

- The DPC++ Clang++ frontend compiles the SYCL kernel into LLVM IR.
- ``llvm-spirv`` is used to translate LLVM IR to SPIR-V.
- SPIR-V is ingested by PoCL, where it is translated back into LLVM IR.
- PoCL applies additional transformations to the LLVM IR.
- If using CPU driver, PoCL leverages ``llc`` (LLVM backend) to lower the kernel to machine code.

This page covers the following steps:

- How to obtain, install, and set up DPC++ (Proprietary or open-source version)
- How to build PoCL to support DPC++.
- Verification with an example program.


Propietary oneAPI DPC++ installation
------------------------------------

DPC++ is available with various bundles, but the oneAPI Base Toolkit is a safe bet:

https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

There are multiple ways to obtain the installer. For the offline GUI installer (check the website above for the latest version)::

    wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/dfc4a434-838c-4450-a6fe-2fa903b75aa7/intel-oneapi-base-toolkit-2025.0.1.46_offline.sh


The installer can be run with different options, either via the GUI or command line.

To install with GUI::

    sh ./intel-oneapi-base-toolkit-2025.0.1.46_offline.sh

The oneAPI Base Toolkit includes various components, some of which have dependencies.

For a minimal setup, pick:

- Intel oneAPI DPC++ Library
- Intel oneAPI DPC++/C++ compiler
- Intel Distribution for GDB (Required by the compiler)
- Intel oneAPI Threading Building Blocks (Required by the compiler)
- Intel oneAPI Math Kernel Library (Useful, but not required)

After installation, run the initialization script to set environment variables (This must be done in every new shell session unless added to ``.bashrc``)::

    source <path-to-oneapi-installation>/setvars.sh

Now, DPC++ should be set up. This can be verified by checking the available SYCL backends (In this example, Intel OpenCL was detected). ::

    sycl-ls
    [opencl:cpu][opencl:0] Intel(R) OpenCL, AMD Ryzen Threadripper 2990WX 32-Core Processor OpenCL 3.0 (Build 0) [2024.18.12.0.05_160000]
    [opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [23.43.027642]


The initialization script also adds the compiler to the ``PATH``::

    icpx --version
    Intel(R) oneAPI DPC++/C++ Compiler 2025.0.4 (2025.0.4.20241205)


Open-source DPC++ installation
------------------------------
The sources for the open-source DPC++ compiler can be obtained from `DPC++ repository <https://github.com/intel/llvm>`__.

Official detailed instructions can be found `here <https://intel.github.io/llvm-docs/GetStartedGuide.html#build-dpc-toolchain>`__.
The build process is managed using two Python scripts: ``configure.py`` and ``compile.py``, which handle most of the heavy lifting.
The ``configure.py`` is essentially a wrapper for **CMake**, so checking its contents can provide further details.

For a basic setup, run::

    git clone git@github.com:intel/llvm.
    cd llvm
    python3 ./buildbot/configure.py -o <path-to-dpcpp-installation>
    python3 ./buildbot/compile.py -o <path-to-dpcpp-installation> -j <number-of-threads>

After building, export the compiler and SYCL runtime library paths::

    export PATH=<path-to-dpcpp-installation>/bin:$PATH
    export LD_LIBRARY_PATH=<path-to-dpcpp-installation>/lib:$LD_LIBRARY_PATH

**Note:** The open-source DPC++ compiler driver is ``clang++``, not ``icpx``.

::

    which clang++
    <path-to-dpcpp-installation>/bin/clang++


Building PoCL for DPC++
-----------------------
PoCL doesn't normally require ``llvm-spirv``, but in this case, it is a strict
dependency because PoCL needs to convert SPIR-V back to LLVM IR.

Pay attention to LLVM and ``llvm-spirv`` versions:

- DPC++ includes its own ``llvm-spirv`` which is typically the latest version (Do not use it with PoCL - it is meant for internal use by DPC++).
- You need a separate build of ``llvm-spirv`` checked out from the branch that corresponds to the LLVM version PoCL uses as its kernel compiler.
- These two versions of ``llvm-spirv`` should be reasonably close to each other, but they do not have to be an exact match.

For example, if DPC++ ``llvm-spirv`` is based on **LLVM 21**, and the PoCL kernel
compiler uses **LLVM 18**, then ``llvm-spirv`` should be checked out from the 'llvm_release_180' branch.


Example PoCL build::

    git clone git@github.com:pocl/pocl.git
    cd pocl
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=<path-to-installation-directory> -DLLVM_SPIRV=<path-to-llvm-spirv> -DWITH_LLVM_CONFIG=<path-to-llvm-config>

The final step is to make PoCL visible to ICD loader::

    export OCL_ICD_VENDORS=<path-to-pocl-installation>/etc/OpenCL/vendors


Compiling with DPC++ using PoCL as the backend
----------------------------------------------
If using proprietary DPC++, there are a few additional steps. Proprietary DPC++
expects the Intel OpenCL runtime so we have to make PoCL appear as one::

    unset OCL_ICD_FILENAMES
    export POCL_DRIVER_VERSION_OVERRIDE=2023.16.7.0.21_160000
    export POCL_CPU_VENDOR_ID_OVERRIDE=32902

Now that everything is set up, verify that PoCL is detected::

    sycl-ls
    [opencl:cpu][opencl:0] Portable Computing Language, cpu-znver1-AMD Ryzen Threadripper 2990WX 32-Core Processor OpenCL 3.0 PoCL HSTR: cpu-x86_64-pc-linux-gnu-znver1

Below is a simple SYCL program to test the setup. It selects the device automatically, so this will drop the possible GPUs out of the list::

    export ONEAPI_DEVICE_SELECTOR=opencl:cpu

.. code-block:: c++

    // hello_nd_range.cpp
    #include <sycl/sycl.hpp>
    #include <iostream>

    #define SUB_GROUP_SIZE 2

    using namespace sycl;

    int main() {
        constexpr int global_size = 8;
        constexpr int local_size = 4;

        queue q;
        {
            q.submit([&](handler &h) {

                std::cout << "One dimensional nd_range with global_size: " << global_size << ", local_size: " << local_size << ", sg_size: " << SUB_GROUP_SIZE << "\n";

                range<1> global(global_size);
                range<1> local(local_size);
                nd_range<1> range(global, local);

                h.parallel_for(range, [=](nd_item<1> idx) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {

                    int workgroup_id_x = idx.get_group(0);
                    int global_id_x = idx.get_global_id(0);
                    int local_id_x = idx.get_local_id(0);
                    int sg_local_id = idx.get_sub_group().get_local_id();
                    int sg_id = idx.get_sub_group().get_group_id();
                    sycl::ext::oneapi::experimental::printf("hello from: (global_id %d) (local_id: %d) (wg_id: %d) (sg_id: %d) (sg_local id: %d)\n",global_id_x, local_id_x,workgroup_id_x, sg_id, sg_local_id);
                });
            }).wait();
        }
        return 0;
    }

Compile and run (use ``icpx`` for propietary version, and ``clang++`` for open-source version)::

    clang++ hello_nd_range.cpp -fsycl -o hello
    ./hello

    One dimensional nd_range with global_size: 8, local_size: 4, sg_size: 2
    hello from: (global_id 0) (local_id: 0) (wg_id: 0) (sg_id: 0) (sg_local id: 0)
    hello from: (global_id 1) (local_id: 1) (wg_id: 0) (sg_id: 0) (sg_local id: 1)
    hello from: (global_id 2) (local_id: 2) (wg_id: 0) (sg_id: 1) (sg_local id: 0)
    hello from: (global_id 3) (local_id: 3) (wg_id: 0) (sg_id: 1) (sg_local id: 1)
    hello from: (global_id 4) (local_id: 0) (wg_id: 1) (sg_id: 0) (sg_local id: 0)
    hello from: (global_id 5) (local_id: 1) (wg_id: 1) (sg_id: 0) (sg_local id: 1)
    hello from: (global_id 6) (local_id: 2) (wg_id: 1) (sg_id: 1) (sg_local id: 0)
    hello from: (global_id 7) (local_id: 3) (wg_id: 1) (sg_id: 1) (sg_local id: 1)
