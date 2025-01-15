**************************
Release Notes for PoCL 6.1
**************************

===========================
Driver-specific features
===========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CPU drivers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* New experimental support for Defined Built-in Kernels (DBK) has
  been added to the CPU drivers. These DBKs allow for a
  standardized set of built-in kernels with well-defined
  semantics that can be configured during creation of the OpenCL
  program. Currently the following DBKs are implemented: GEMM,
  matrix multiplication, JPEG en-/de-code, and ONNX runtime
  inference. The Extension documentation draft can be found on
  `github <https://github.com/KhronosGroup/OpenCL-Docs/pull/1007>`_.
  Please note that these DBKs are still a proof-of-concept and
  are subject to change without notice.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Work-group vectorization improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Improve work-group vectorization of kernels which use the global
  id as the iteration variable. Now the loop vectorizer should
  generate wide loads/stores more often than falling back to
  scatter/gather.
* The WG vectorization with the 'loopvec' method is enabled again,
  there was an issue when upgrading to the latest LLVM version.
  Now there's a regression test in place for avoiding accidents
  like this in the futre.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Miscellaneous
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* New env variable POCL_MAX_COMPUTE_UNITS. This is similar to
  existing POCL_CPU_MAX_CU_COUNT, but is not driver-specific,
  hence can be used to limit the CU count of all drivers that
  support limiting CU count. If both are specified, the driver
  specific variable takes precedence.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remote driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The remote driver now includes support for dynamic device addition
  and network discovery. This new feature allows discovery of remote
  servers located in LAN or WAN environments and enables runtime
  addition of discovered devices to the remote client's platform.
  Network discovery is performed thorugh mDNS, unicast-DNS-SD, and
  DHT-based mechanisms, using Avahi and OpenDHT libraries.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Level Zero driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Various improvements were made:

* Added support for Intel NPU (or "AI Boost" in the CPU specification)
  as a custom device. Source compilation is not supported yet but GEMM
  and matrix multiplications can be offloaded to the NPU device using
  DBKs. Note that the feature is in very experimental stage and the
  provided DBKs maybe subject to changes.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AlmaIF driver (FPGA interfacing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

===================================
Support for Julia
===================================

It is now possible to use PoCL with Julia through the OpenCL.jl package.
The integration is still considered experimental, and the OpenCL.jl
interface package itself is under active (re)development, but it is
already possible to run many kernels using PoCL as the backend.
For example:

.. code-block:: julia

    using OpenCL, pocl_jll, Test

    const source = """
       __kernel void vadd(__global const float *a,
                          __global const float *b,
                          __global float *c) {
          int i = get_global_id(0);
          c[i] = a[i] + b[i];
        }"""

    dims = (2,)
    a = round.(rand(Float32, dims) * 100)
    b = round.(rand(Float32, dims) * 100)
    c = similar(a)

    d_a = CLArray(a)
    d_b = CLArray(b)
    d_c = CLArray(c)

    prog = cl.Program(; source) |> cl.build!
    kern = cl.Kernel(prog, "vadd")

    len = prod(dims)
    clcall(kern, Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}},
           d_a, d_b, d_c; global_size=(len,))
    c = Array(d_c)
    @test a+b ≈ c

OpenCL.jl also provides a high-level Julia to SPIR-V compiler,
making it possible to significantly simplify the above example:

.. code-block:: julia

    # import packages, allocate data, etc

    function vadd(a, b, c)
        i = get_global_id()
        @inbounds c[i] = a[i] + b[i]
        return
    end

    @opencl global_size=len vadd(d_a, d_b, d_c)

The aim of this work is to provide a CPU fallback for executing
Julia's GPU kernels and applications by leveraging the CPU drivers
in PoCL. For more information, refer to
`the blog post on OpenCL.jl 0.10 <https://juliagpu.org/post/2025-01-13-opencl_0.10/>`_.

===================================
Notable fixes
===================================

There were a lot of fixed done over the release cycles. Some of the
most notable/user facing ones are listed below:

===================================
Deprecation/feature removal notices
===================================

 * The old "work-item replication" work-group function generation
   method was removed to clean up the kernel compiler. It did not
   anymore have any use cases that could not be covered by fully
   unrolling "loops".

