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

HSA support implementation status as of 2016-05-17
--------------------------------------------------

What’s Implemented
~~~~~~~~~~~~~~~~~~~

* global/local/private memory
* barriers
* most of the OpenCL 1.2 kernel builtins
* OpenCL 2.0 shared virtual memory (SVM)
* OpenCL 2.0 atomics

What's Missing
~~~~~~~~~~~~~~~

* printf() is not implemented, this should wait until we have a proper in-tree printf() in pocl with a stdout ring buffer
* several builtins are not implemented yet (logb, remainder, nextafter); some are suboptimal or may give incorrect results with under/overflows (most of the builtins are taken from vecmathlib library, rewritten to fit HSAIL).
* image support is not implemented
* support for GPU devices other than Kaveri; currently only Kaveri and phsa-based CPU Agents have been tested
* support for 32bit HSA devices

About the Shared Virtual Memory Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenCL 2.0 SVM is a feature that lets you share virtual memory between CPU and GPUs.
Note that while SVM works in pocl, one must carefully align all structs explicitly (both struct
members and struct itself). This is because the alignment of the structs with the host's
compiler might differ from the one in the device.

For example, you can see the issue in Intel's SVM examples:

.. code-block:: c

    typedef struct _Element
    {
        global float* internal; //points to the "value" of another Element from the same array
        global float* external; //points to the entry in a separate array of floating-point values
        float value;
    } Element;

This *may* work with Intel's OpenCL SDK in case only using CPU devices, but crashes when offlodaing to HSA
via pocl's HSA driver. The reason is that when using HSA, pocl compiles this header with two different
compilers: usually gcc/clang for host C code and, llvm-HSAIL (Clang) for the device side,
and they do *not* use the same alignment rules.

The C standard specify almost nothing with regards to struct alignment in memory, so one must take care
to explicitly specify alignment when using structs in shared memory.

A proper way to declare the struct would be to utilize the widely supported 'aligned' attribute.

.. code-block:: c

    typedef struct _Element
    {
        global float* internal __attribute__ ((aligned (8))); //points to the "value" of another Element from the same array
        global float* external __attribute__ ((aligned (8))); //points to the entry in a separate array of floating-point values
        float value __attribute__ ((aligned (8)));
    } Element __attribute__ ((aligned (32)));


phsa
~~~~~

`Portable HSA (phsa) <https://github.com/HSAFoundation/phsa>`_ provides similar portable HSA implementation
for CPUs/DSPs and other processors as pocl aims to do for OpenCL. Using phsa one can implement HSA Agent support
for any processor which has a gcc backend with ease.

pocl supports phsa as a backend for its HSA driver, thus any processor utilizing phsa for HSA Agent support
can get OpenCL support via pocl. We used phsa for testing the HSA driver works with other devices and
runtimes than AMD's.

Known Issues
---------------

OpenCL 2.0 Atomics and HSA Memory Scope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is a "memory scope" parameter present in HSA, which applies to atomic memory instructions or
memory fences. Its purpose is to limit the scope of these instructions. However, pocl translates
to HSAIL via LLVM bitcode, and the "atomicrmw" LLVM instruction only takes a memory order parameter, not scope.
For this reason the memory scope in HSAIL is always the widest "system" scope.

Multiple HSA Agent Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

While multiple OpenCL device support is not a problem for pocl, the HSA 1.0 specification lacks a "loader/proxy"
feature that OpenCL has in ICD. Thus, support for devices is limited to what the linked HSA runtime supports.

Currently, if one wants to control multiple HSA Agents as multiple pocl OpenCL devices, one needs to implement
a HSA runtime that lists all the Agents to pocl. There is no capability to load multiple HSA runtimes in pocl
as we consider it out of scope and a job for a proxy HSA runtime similar to ICD.

Performance
-------------

We conducted preliminary benchmarking with a set of test cases to serve as a basis for future optimization
efforts.

Evaluation Setup
~~~~~~~~~~~~~~~~~~

Hardware: AMD A10-7800, 8GB 1600Mhz of dual-channel memory, TDP set to 65W

* Configuration 1: Windows 10 x86-64, AMD Crimson drivers
* Configuration 2: Ubuntu 15.04 x86-64, kernel 4.0.0 & runtime 1.0.3 from https://github.com/HSAFoundation

Test applications from AMD SDK 3.0 samples/opencl/bin/x86_64. The tests were run with -i (iterations)
parameter ranging from 10 to 200 (longer tests were ran with fewer iterations).

The performance currently lags behind the AMD's proprietary OpenCL on Windows by a factor of 1x to 5x

===================================================  ==============  ======================  =============== ============= =============================
AMD SDK example with arguments                       AMD runtime(s)  other(GB/s,opts/s etc)  POCL runtime(s) other         POCL/AMD (>1.0 = POCL slower)
===================================================  ==============  ======================  =============== ============= =============================
BitonicSort -q -t -x 1048576                         0.0978          10713500                0.2116          4954540       2.162
BinomialOption -q -t -x 10000                        0.0164          25855.1                 0.0233          37030.3       1.416
BlackScholes -s -q -t -x 16777216                    0.0098          1708340000              0.0790          212347000     8.045
DCT -q -t -x 4000 -y 4000                            0.0493          -                       0.0582          -             1.181
FastWalshTransform -q -t -x 134217728                1.5895          -                       2.4367          -             1.533
FloydWarshall -q -t -x 512                           0.0671          -                       0.1802          -             2.682
MatrixTranspose -t -x 8192  -q                       0.0317          16920500000             0.1675          3204580000    5.280
MatrixMultiplication -q -t -x 1024 -y 1024 -z 2048   0.0175          245.07                  0.0776          55.29         4.432
QuasiRandomSequence -q -t -y 10200 -x 10000          0.0009          2754120000              0.0100          1188730000    10.603
Reduction -q -t -x 100000000                         0.1108          -                       0.1165          -             1.051
SimpleConvolution -q -t -x 204800                    0.1056          0.565378                0.1154          1.68136       2.973
===================================================  ==============  ======================  =============== ============= =============================

We briefly analyzed the bottlenecks and the first clear issue is that we have recently introduced out-of-order queues
in pocl, and the driver layer changed significantly with this regard, and it has not yet been fully optimized for HSA.
There is ongoing work in this area. The slow kernel launches may be the reason why extremely short kernels like QuasiRandomSequence
are >5x slower.

The other major issue is that the LLVM 3.7 based HSAIL compiler is sometimes producing clearly suboptimal code. If we take
MatrixMultiplication as an example, the GPU code generated by the proprietary AMD OpenCL driver on windows uses 76 VGPRs, 26 SGPRs and
has no spills. The HSAIL code from pocl contains about 70 spills! While the HSA PRM (programmer's reference manual) states "the
finalizer might be able to deploy extra hardware registers and remove the spills", it's likely not successful in this case, assuming
AMD's HSAIL finalizer is putting only minimal effort to optimize the code to provide fast finalization times.

This hopefully will change when LLVM-HSAIL is updated to later LLVM versions and its main bottlenecks are optimized, or in case
new AMD SDK versions do optimization in the finalization of the suboptimal HSAIL input.

Credits
----------

The current implementation was mainly done by our `Customized Parallel Computing <https://tuni.fi/cpc>`_ group of
Tampere University, Finland with early prototype code contributions from the Programming Language Lab
at National Tsing-Hua University, Hsinchu, Taiwan.

CPC group thanks HSA Foundation and ARTEMIS JU (under grant agreement no 621439, ALMARVI) for funding
this initial pocl HSA driver work. This driver added GPU device support to pocl for the first time, and, on the
other hand, produced an easier path for HSA-supported devices to implement the OpenCL API by utilizing the pocl
code base as a starting point.

In the future we hope to see more effort put in optimizing the results to reach the performance of the
proprietary SDKs on HSA devices.
