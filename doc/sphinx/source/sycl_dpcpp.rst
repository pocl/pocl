.. _sycl_dpcpp:

Using PoCL as the OpenCL backend for DPC++
------------------------------------------

SYCL is a programming model that enables single-source C++ development for
heterogeneous computing. Compared to OpenCL, SYCL operates at a higher level
of abstraction, and implementations can use varying backends
for device offloading (e.g., OpenCL, Level Zero, and CUDA). It is worth noting
that a SYCL implementation is not required to support OpenCL as a backend.

DPC++ is Intel's implementation of SYCL that supports OpenCL. When the OpenCL backend
is utilized, the DPC++ runtime translates SYCL API calls into corresponding OpenCL
API calls and forwards them to the OpenCL runtime.

The toolchain flow, when PoCL is used as the OpenCL backend for DPC++, is as follows:

- The DPC++ Clang++ frontend compiles the SYCL kernel into LLVM IR.
- ``llvm-spirv`` is used to translate LLVM IR to SPIR-V.
- SPIR-V is ingested by PoCL, where it is translated back into LLVM IR.
- PoCL applies additional transformations to the LLVM IR.
- If using a CPU driver, PoCL leverages ``llc`` (LLVM backend) to lower the kernel to machine code.

It should be pointed out that there are two versions of DPC++:

- the **Intel(R) oneAPI DPC++/C++ Compiler**
- the **oneAPI DPC++/C++ Compiler**.

The former is proprietary and thus distributed in binary form, whereas the latter is
open-source.

This page covers the following steps:

- How to obtain, install, and set up DPC++ (the proprietary or the open-source version)
- How to build PoCL to support DPC++.
- Verification with an example program.


Intel(R) oneAPI DPC++/C++ Compiler installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DPC++ is available in various bundles. Installing the oneAPI Base Toolkit is the simplest way to install DPC++ and its dependencies.

Choose a suitable installer from:

https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

Run the installer. The page above provides corresponding instructions for the selected installer.
Pay attention to the default installation path and choose a suitable one if necessary.

The oneAPI Base Toolkit includes various components, some of which are not needed to run SYCL applications with PoCL.

For a minimal setup, pick:

- Intel oneAPI DPC++ Library
- Intel oneAPI DPC++/C++ compiler
- Intel Distribution for GDB (Required by the compiler)
- Intel oneAPI Threading Building Blocks (Required by the compiler)
- Intel oneAPI Math Kernel Library (Useful, but not required here)

After installation, run the initialization script to set the environment variables::

    source <path-to-oneapi-installation>/setvars.sh

.. important::
   ``setvars.sh`` must be run in every new shell session unless added to ``.bashrc`` (or an equivalent).


Now, DPC++ should be set up. This can be verified by checking the available SYCL backends (In this example, Intel OpenCL was detected).:

.. code-block:: none

    sycl-ls
    [opencl:cpu][opencl:0] Intel(R) OpenCL, AMD Ryzen Threadripper 2990WX 32-Core Processor OpenCL 3.0 (Build 0) [2024.18.12.0.05_160000]
    [opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [23.43.027642]


The initialization script also adds the compiler to the ``PATH``:

.. code-block:: none

    icpx --version
    Intel(R) oneAPI DPC++/C++ Compiler 2025.0.4 (2025.0.4.20241205)


oneAPI DPC++/C++ Compiler installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The sources for the open-source DPC++ compiler can be obtained from the `DPC++ repository <https://github.com/intel/llvm>`__.

Official detailed instructions can be found `here <https://intel.github.io/llvm-docs/GetStartedGuide.html#build-dpc-toolchain>`__.
The build process is managed using two Python scripts: ``configure.py`` and ``compile.py``, which handle most of the heavy lifting.
The ``configure.py`` is essentially a wrapper for **CMake**, so checking its contents can provide further details.

For a basic setup, run::

    git clone git@github.com:intel/llvm
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
~~~~~~~~~~~~~~~~~~~~~~~
PoCL doesn't normally require ``llvm-spirv``, but in this case, it is a strict
dependency because PoCL needs to convert the SPIR-V produced by DPC++ back to LLVM IR.

You must check out and build a version of ``llvm-spirv`` that corresponds to the LLVM
version PoCL uses as its kernel compiler. For example, if the PoCL kernel compiler uses
**LLVM 18**, then ``llvm-spirv`` should be checked out from the ``llvm_release_180`` branch.

.. note::
   DPC++ ships with its own ``llvm-spirv``, which is typically based on the latest release.
   However, this version is intended for internal usage by DPC++ and cannot be used by PoCL.


.. warning::
   Although the versions of ``llvm-spirv`` used by DPC++ and PoCL do not have to be an exact
   match, it is recommended to use versions that are reasonably close to each other.


Example PoCL build:

.. code-block:: none

    git clone git@github.com:pocl/pocl.git
    cd pocl
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=<path-to-installation-directory> -DLLVM_SPIRV=<path-to-llvm-spirv> -DWITH_LLVM_CONFIG=<path-to-llvm-config>
    ninja install

To make PoCL visible to the ICD loader, either register the PoCL ICD (https://github.com/KhronosGroup/OpenCL-ICD-Loader#registering-icds)
or set the ``OCL_ICD_FILENAMES`` or ``OCL_ICD_VENDORS`` environment variables. ``OCL_ICD_VENDORS`` only works on Linux/Android,
whereas ``OCL_ICD_FILENAMES`` works on all platforms (see https://github.com/KhronosGroup/OpenCL-ICD-Loader#table-of-debug-environment-variables
for more information).

On Linux:

.. code-block:: none

    export OCL_ICD_VENDORS=<path-to-pocl-installation>/etc/OpenCL/vendors


Compiling with DPC++ using PoCL as the backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If using proprietary DPC++, there is one additional step.
By default PoCL is blocked by the DPC++ runtime. To enable PoCL, we need to set the ``SYCL_DEVICE_ALLOWLIST``
environment variable. This variable is a comma-separated list of parameters that the DPC++ runtime uses to select allowed devices.
It can be used quite flexibly. For example, to select only CPU devices:

.. code-block:: none

    export SYCL_DEVICE_ALLOWLIST="DeviceType:cpu"

To allow all available devices, use:

.. code-block:: none

    export SYCL_DEVICE_ALLOWLIST=""


To select only PoCL, you can use the PoCL vendor ID:

.. code-block:: none

    export SYCL_DEVICE_ALLOWLIST="DeviceVendorId:0x10006"


For more information about how to use the DPC++ environment variables, see:

https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md

Now that everything is set up, verify that PoCL is detected:

.. code-block:: none

    sycl-ls
    [opencl:cpu][opencl:0] Portable Computing Language, cpu-znver1-AMD Ryzen Threadripper 2990WX 32-Core Processor OpenCL 3.0 PoCL HSTR: cpu-x86_64-pc-linux-gnu-znver1

Below is a simple SYCL program to test the setup. It selects the device automatically, so this will drop the possible GPUs out of the list:

.. code-block:: none

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

Compile and run (use ``icpx`` for proprietary version, and ``clang++`` for open-source version)::

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
