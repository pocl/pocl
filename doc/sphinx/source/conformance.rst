.. _pocl-conformance:

=======================
Pocl OpenCL conformance
=======================

Conformance related CMake options
---------------------------------

- ``-DENABLE_CONFORMANCE=ON/OFF``
  This is mostly related to the kernel library (the runtime is always conformant
  on x86). Defaults to ON.
  Non-conformant kernel library might be somewhat faster, at the expense of
  precision and/or range. Note that conformance was tested **only** on certain
  hardware and software (Linux, x86-64, CPU with AVX & FMA instructions).

How to run the conformance test suite on your hardware
------------------------------------------------------

First you need to enable the suite in the pocl's external test suite set.
This is done by adding switch ``-DENABLE_TESTSUITES=conformance``
to the cmake command line. After this ``make prepare_examples`` fetches and
prepares the conformance suite for testing.

To run a shortened version of the conformance suite, run: ``ctest -L conformance_suite_mini``
This might take a few hours on slow hardware. There is also a ``conformance_suite_micro``
label, which takes about 20-30 minutes on slow hardware.

To run the full conformance testsuite, run: ``ctest -L conformance_suite_full``
Note that this can take a week to finish on slow hardware, and about a day
on fast hardware (6C/12T Intel or equivalent).

Known issues with the conformance testsuite
-------------------------------------------

- the "not" operator test (``math_brute_force/bruteforce not``) may fail to
  compile with LLVM 4.0 with certain vector sizes on some hardware.
  This does not seem to affect the rest of the testsuite in any way, and
  appears to be fixed with LLVM 5.0

- a few tests from ``basic/test_basic`` may fail / segfault because they
  request a huge amount of memory for buffers.

- a few tests from ``conversions/test_conversions`` may report failures when
  compiled with -O2. This is likely a bug in the test; the same test from branch
  cl20_trunk of CTS passes.

- math_brute_force tests may occasionally fail with an empty build log,
  see pocl issue #614.

- a few tests may run much faster if you limit the reported Global memory size
  with POCL_MEMORY_LIMIT env var. In particular, "kernel_image_methods" test
  with "max_images" argument.

- two tests in ``api/test_api`` fail with LLVM 5.0 because of
  LLVM commit 1c1154229a41b688f9:

    ``[OpenCL] Do not generate "kernel_arg_type_qual" metadata for non-pointer args``

  This is a bug in CTS, which tests for non-pointer type qualifiers, not in pocl.
  See:

  https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf page 169:

  ``CL_KERNEL_ARG_TYPE_VOLATILE`` is returned if the **argument is a pointer**
  and the referenced type is declared with the volatile qualifier.
  Similarly, ``CL_KERNEL_ARG_TYPE_RESTRICT`` or ``CL_KERNEL_ARG_TYPE_CONST`` is
  returned if the **argument is a pointer** and the referenced type is declared with
  the restrict or const qualifier

.. _sigfpe-handler:

Known issues in pocl / things to be aware of
--------------------------------------------

- Integer division by zero. OpenCL 1.2 specification requires that division by
  zero on integers results in undefined values, instead of raising exceptions.
  This requires pocl to install a handler of SIGFPE. Unfortunately signal
  handlers are per-process not per-thread, and pocl drivers do not run in a
  separate process, which means that integer division by zero will not raise
  SIGFPE for the entire pocl library and also the user's program. The handler
  may be disabled by setting the env variable POCL_SIGFPE_HANDLER to 0.
  Note that this is currently only relevant for x86(-64) + Linux, on all other
  systems this issue is not handled in any way (thus Pocl is likely
  non-conformant there).

- Several options to clBuildProgram() are accepted but currently have no effect.
  This is related mostly to optimization options like `-cl-fast-relaxed-math`.
  The `-cl-denorms-are-zero` and `-cl-fp32-correctly-rounded-divide-sqrt`
  options are honored.

- Many of ``native_`` and ``half_`` variants of kernel library functions are mapped
  to the "full" variants.

- the optional OpenGL / D3D extensions are not supported. There is experimental
  support for SPIR

- clUnloadCompiler() only actually unload LLVM after all programs & kernels
  have been released.

- clSetUserEventStatus() called with negative status. The Spec leaves the behaviour
  in this case as "implementation defined", and this part of pocl is
  only very lightly tested by the conformance tests. clSetUserEventStatus()
  called with CL_COMPLETE works as expected, and is heavily used by
  the conversions conformance test.

Conformance tests results (kernel library precision) on tested hardware
-----------------------------------------------------------------------

Note that it's impossible to test double precision on the entire range,
therefore the results may vary.

x86-64 CPU with AVX2+FMA, LLVM 4.0, tested on Nov 1, 2017
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

====================   =========================   ===========================================================
        NAME                Worst ULP                 WHERE
====================   =========================   ===========================================================
             add            0.00                      {0x0p+0, 0x0p+0}
             addD           0.00                      {0x0p+0, 0x0p+0}
      assignment            0.00                      0x0p+0
      assignmentD           0.00                      0x0p+0
            cbrt            0.50                      -0x1.5629d2p+116
            cbrtD           0.59                      0x1.0000000000136p+1022
            ceil            0.00                      0x0p+0
            ceilD           0.00                      0x0p+0
        copysign            0.00                      {0x0p+0, 0x0p+0}
        copysignD           0.00                      {0x0p+0, 0x0p+0}
             cos            2.37                      0x1.1338ccp+20
             cosD           2.27                      -0x1.d10000000074p+380
            cosh            2.41                      -0x1.602166p+2
            coshD           1.43                      -0x1.98000000003efp+5
           cospi            1.94                      0x1.d73b56p-2
           cospiD           2.46                      -0x1.adffffffffa91p-2
          divide            0.00                      {0x0p+0, 0x0p+0}
          divideD           0.00                      {0x0p+0, 0x0p+0}
             exp            0.95                      -0x1.762532p+2
             expD           0.94                      0x1.2f0000000023dp+7
           exp10            0.79                      -0x1.309022p+5
           exp10D           0.64                      -0x1.34ffffffffcc9p+8
            exp2            0.79                      -0x1.fa3d0ep+6
            exp2D           0.75                      -0x1.ff00000000417p+9
           expm1            1.00                      -0x1.7a0002p-25
           expm1D           0.99                      -0x1.26p+5
            fabs            0.00                      0x0p+0
            fabsD           0.00                      0x0p+0
            fdim            0.00                      {0x0p+0, 0x0p+0}
            fdimD           0.00                      {0x0p+0, 0x0p+0}
           floor            0.00                      0x0p+0
           floorD           0.00                      0x0p+0
             fma            0.00                      {0x0p+0, 0x0p+0, 0x0p+0}
             fmaD           0.00                      {0x0p+0, 0x0p+0, 0x0p+0}
            fmax            0.00                      {0x0p+0, 0x0p+0}
            fmaxD           0.00                      {0x0p+0, 0x0p+0}
            fmin            0.00                      {0x0p+0, 0x0p+0}
            fminD           0.00                      {0x0p+0, 0x0p+0}
            fmod            0.00                      {0x0p+0, 0x0p+0}
            fmodD           0.00                      {0x0p+0, 0x0p+0}
           fract            { 0.00, 0.00}             {0x0p+0, 0x0p+0}
           fractD           { 0.00, 0.00}             {0x0p+0, 0x0p+0}
           frexp            { 0.00, 0}                 0x0p+0
           frexpD           { 0.00, 0}                 0x0p+0
           hypot            1.93                      {0x1.17c998p-127, -0x1.5fedb8p-127}
           hypotD           1.73                      {0x1.5d2ebeed7663cp-1022, 0x1.67457048a2318p-1022}
           ldexp            0.00                      {0x0p+0, 0}
           ldexpD           0.00                      {0x0p+0, 0}
           log10            0.50                      0x1.7fee2ep-1
           log10D           0.50                      0x1.9100000000639p+1022
             log            0.63                      0x1.7fcb3ep-1
             logD           0.75                      0x1.7d00000000381p+0
           log1p            1.00                      -0x1.fa0002p-126
           log1pD           1.00                      -0x1.e000000000001p-1022
            log2            0.59                      0x1.1107a2p+0
            log2D           0.72                      0x1.120000000063dp+0
            logb            0.00                      0x0p+0
            logbD           0.00                      0x0p+0
             mad            0.00                      {0x0p+0, 0x0p+0, 0x0p+0} no ULP check
             madD           0.00                      {0x0p+0, 0x0p+0, 0x0p+0} no ULP check
          maxmag            0.00                      {0x0p+0, 0x0p+0}
          maxmagD           0.00                      {0x0p+0, 0x0p+0}
          minmag            0.00                      {0x0p+0, 0x0p+0}
          minmagD           0.00                      {0x0p+0, 0x0p+0}
            modf        { 0.00, 0.00}                 {0x0p+0, 0x0p+0}
            modfD       { 0.00, 0.00}                 {0x0p+0, 0x0p+0}
        multiply            0.00                      {0x0p+0, 0x0p+0}
        multiplyD           0.00                      {0x0p+0, 0x0p+0}
             nan            0.00                      0x0p+0
             nanD           0.00                      0x0p+0
       nextafter            0.00                      {0x0p+0, 0x0p+0}
       nextafterD           0.00                      {0x0p+0, 0x0p+0}
             pow            0.82                      {0x1.91237cp-1, 0x1.4da146p+8}
             powD           0.80                      {0x1.2bfb4b18164c9p+65, -0x1.b78438ae9c3bdp-8}
            pown            0.65                      {-0x1.9p+6, -2}
            pownD           0.62                      {-0x1.7ffffffffffffp+1, 3}
            powr            0.82                      {0x1.91237cp-1, 0x1.4da146p+8}
            powrD           0.80                      {0x1.2bfb4b18164c9p+65, -0x1.b78438ae9c3bdp-8}
       remainder            0.00                      {0x0p+0, 0x0p+0}
       remainderD           0.00                      {0x0p+0, 0x0p+0}
          remquo        { 0.00, 0}                    0x0p+0
          remquoD       { 0.00, 0}                    0x0p+0
            rint            0.00                      0x0p+0
            rintD           0.00                      0x0p+0
           rootn            0.69                      {-0x1.e2fe6ep-74, -141}
           rootnD           0.68                      {-0x1.8000000000001p+1, 3}
           round            0.00                      0x0p+0
           roundD           0.00                      0x0p+0
           rsqrt            1.49                      0x1.019566p+124
           rsqrtD           1.49                      0x1.01ffffffffa39p+1016
             sin            2.48                      -0x1.09f07ap+21
             sinD           1.87                      -0x1.f2fffffffffbap+32
          sincos        { 2.48, 2.37}                 {0x1.09f07ap+21, 0x1.1338ccp+20}
          sincosD       { 1.87, 2.27}                 {0x1.f2fffffffffbap+32, 0x1.d10000000074p+380}
            sinh            2.32                      0x1.e76078p+2
            sinhD           1.53                      -0x1.3100000000278p+4
           sinpi            2.13                      -0x1.45f3ep-9
           sinpiD           2.50                      -0x1.46000000000dap-7
            sqrt            0.00                      0x0p+0
            sqrtD           0.00                      0x0p+0
        subtract            0.00                      {0x0p+0, 0x0p+0}
        subtractD           0.00                      {0x0p+0, 0x0p+0}
             tan            4.35                      -0x1.b4eba2p+22
             tanD           4.00                      -0x1.2f000000003edp+333
            tanh            1.18                      -0x1.ca742ap-1
            tanhD           1.19                      0x1.f400000000395p-1
           tanpi            4.21                      -0x1.f99d16p-3
           tanpiD           4.09                      0x1.f6000000001d3p-3
           trunc            0.00                      0x0p+0
           truncD           0.00                      0x0p+0
====================   =========================   ===========================================================
