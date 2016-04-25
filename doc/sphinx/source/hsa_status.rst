.. _hsa-status:

HSA implementation status (as of April 2016)
============================================

What’s implemented
------------------

* global/local/private memory
* atomics, barriers
* most of the OpenCL kernel library builtins
* SVM (shared virtual memory)
* OpenCL 2.0 atomics

What's missing
--------------

* printf() is not implemented
* several builtins are not implemented yet (logb, remainder, nextafter); some are suboptimal or may give incorrect results with under/overflows (most of the builtins are taken from vecmathlib library, rewritten to fit HSAIL).
* image support is not implemented
* the support of new devices
* support of 32bit HSA devices

Shared Virtual Memory
---------------------
OpenCL 2.0 SVM is a feature that lets you fully share memory between CPU and GPU including sharing pointers. Note that while SVM works in Pocl, one must carefully align all structs explicitly (both struct members and struct itself). For example, you can see this in Intel SVM examples:

.. code-block:: c

    typedef struct _Element
    {
        global float* internal; //points to the "value" of another Element from the same array
        global float* external; //points to the entry in a separate array of floating-point values
        float value;
    } Element;

This *may* work with Intel's OpenCL SDK, but crashes with Pocl's HSA driver immediately. The reason is, Pocl compiles this header with two different compilers (gcc for host code, llvm-HSAIL compiler for GPU part)
and they do NOT use the same alignment rules. The C standards specify almost nothing with regards to struct alignment in memory, so one must take care to explicitly specify alignment when using structs in shared memory. So the proper way to declare the struct would be:

.. code-block:: c

    typedef struct _Element
    {
        global float* internal __attribute__ ((aligned (8))); //points to the "value" of another Element from the same array
        global float* external __attribute__ ((aligned (8))); //points to the entry in a separate array of floating-point values
        float value __attribute__ ((aligned (8)));
    } Element __attribute__ ((aligned (32)));


PHSA
----
This project (Portable HSA, https://github.com/HSAFoundation/phsa) aims to support generic CPUs as HSA devices.
Aims to support all CPUs which have a gcc backend. Pocl will support this runtime as backend for its HSA driver.
Currently (early 2016) this is still a work in progress.

Caveats/Bugs/missing features
=============================

OpenCL 2.0 atomics
------------------
There is a "memory scope" parameter present in HSA, which applies to atomic memory instructions or memory fences. It's purpose is to limit the scope of these instructions. However, Pocl translates to HSAIL via LLVM bitcode, and the "atomicrmw" LLVM instruction only takes a memory order parameter, not scope.
For this reason the memory scope in HSAIL is always "system".

Multiple HSA device(agent) support
-----------------------------------
While multiple device support should not be a problem for Pocl, the HSA specification lacks the "loader" feature that e.g. OpenCL has in ICD. Thus support for devices is limited to what the linked HSA runtime supports.

Performance
===========

Setup
-----
Hardware: AMD A10-7800, 8GB 1600Mhz of dual-channel memory, TDP set to 65W

* Configuration 1: Windows 10 x86-64, AMD Crimson drivers
* Configuration 2: Ubuntu 15.04 x86-64, kernel 4.0.0 & runtime 1.0.3 from https://github.com/HSAFoundation

Tests: from AMD SDK 3.0 samples/opencl/bin/x86_64
Tests were run with -i (iterations) parameter ranging from 10 to 200 (slower tests were ran with fewer iterations).

Performance generally lags behind proprietary AMD's OpenCL on Windows by a factor of 1x to 5x

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

The first issue is, we have recently introduced out-of-order queues in Pocl, thus the driver model changed significantly, and has not yet been fully optimized.
There is ongoing work in this area. This issue (Pocl overhead) may be the reason why extremely short kernels like QuasiRandomSequence are >5x slower.

The other issue is that the HSAIL compiler is likely producing suboptimal code. If we take MatrixMultiplication as an example, the GPU code produced by the proprietary AMD OpenCL driver on windows uses 76 VGPRs, 26 SGPRs and has no spills. The HSAIL from pocl contains about 70 spills. While the HSA PRM (programmer's reference manual) states "the finalizer might be able to deploy extra hardware registers and remove the spills", it's likely not successful in this case.
This may change when AMD releases a better HSAIL compiler/finalizer.
