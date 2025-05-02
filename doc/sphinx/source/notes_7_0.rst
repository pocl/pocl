**************************
Release Notes for PoCL 7.0
**************************

===========================
Release highlights
===========================

* PoCL received the official OpenCL 3.0 conformance stamp with the
  `CPU <https://www.khronos.org/conformance/adopters/conformant-products/opencl#submission_450>`_ (x86-64) and
  `Level Zero <https://www.khronos.org/conformance/adopters/conformant-products/opencl#submission_453>`_
  drivers. Conformance testing via OpenCL-CTS passed with both OpenCL C
  and SPIR-V compilation modes, for both of the drivers.

* Proper support for Windows platform added, tested with CPU and Level Zero drivers.
  The CPU driver supports both MinGW and MSVC toolchains, Level Zero only MSVC for
  now.

* Support for LLVM versions 19 and 20.

* Support building with Khronos ICD, if the ICD supports OpenCL 3.0.

* Support for dynamic device addition and network discovery in
  PoCL-Remote.

* Initial support for Julia input through the OpenCL.jl package.

=============================
Notable user facing changes
=============================

* CMake 3.15 is now required to build PoCL.

* New env variable POCL_MAX_COMPUTE_UNITS. This is similar to
  existing POCL_CPU_MAX_CU_COUNT, but is not driver-specific,
  hence can be used to limit the CU count of all drivers that
  support limiting CU count. If both are specified, the driver
  specific variable takes precedence.

* The callbacks for context destruction, memory destruction and event
  callbacks were moved to a separate thread (that runs asynchronously).
  This might have unexpected effect on the behavior of program using
  libpocl, since callbacks might be executed with some delay.

* Level Zero devices, if found, are shown in the platform by default
  in addition to the CPU drivers.

* Works with the upstream llama.cpp's OpenCL backend without
  modifications on PoCL-LZ.

* Various fixes and improvements to SYCL input via the Intel DPC++.

* Various improvements to support running OpenVINO out-of-the-box.

===========================
Driver-specific features
===========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CPU drivers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* When built with -DENABLE_CONFORMANCE=ON, PoCL's CPU driver
  is now able to pass the full OpenCL-CTS conformance testsuite.

* Proper and now CI-tested Windows platform support, using both
  MinGW and MSVC toolchains.

* Support for generating NSIS packages on Windows.

* The cl_khr_command_buffer implementation has been updated to v0.9.6.
  Note that this version is not backwards compatible with previous versions.

* Implemented version 1.0.2 of the cl_ext_buffer_device_address extension.

* Implemented version 0.9.3 of the cl_khr_command_buffer_mutable_dispatch.

* Implemented cl_intel_subgroups_{char/short/long} extensions.

* Implemented missing intel_sub_groups_shuffle_down/up for scalar types.

* Implemented cl_intel_subgroup shuffle/xor/up/down with vector arguments
  (this is required by SYCL-CTS).

* Implemented read_imageh() function.

* SPIR-V support has been improved to support additional subgroup-related
  functions.

* SPIR-V up to version 1.5 is now supported with LLVM 20, and up to 1.4
  with LLVM 19.

* The max_mem_alloc_size of the CPU device is now equal to global_mem_size
  (previously was limited to 1/4 of global_mem_size).

* Handling of LLVM's 'Unreachable' instructions has been improved. The
  new handling depends on the workitem handler; with "loopvec" the
  Unreachable is removed; with CBS the unreachable is converted to set
  an error flag, which the CPU driver then uses to set the NDRange
  event's state to failed.

* With "distro" builds, the kernel-lib variants now include
  the "generic" CPU variant as a fallback.

* OpenCL C printf() now correctly supports all OpenCL vector types.

* Added option to disable the support of subnormal floats.

* Added option to avoid division-by-zero exceptions using an LLVM pass
  instead of SIGFPE handler (CMake option ENABLE_SIGFPE_HANDLER).
  This has the advantage of being a platform-independent and
  scheduler-independent solution that works also with the
  oneTBB and OpenMP runtimes.

* Support for using LLVMSPIRVLib for SPIR-V input translation,
  as an alternative to calling llvm-spirv binary.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
CPU kernel compiler improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Improved the work-group vectorization of kernels which use the global
  id as the iteration variable. Now the loop vectorizer should
  generate wide loads/stores more often than falling back to
  scatter/gather when global ids are used.

* The implicit WG vectorization with the 'loopvec' method is reenabled,
  after accidentally disabling it when upgrading to the latest LLVM
  version.

* New env variables POCL_VECTORIZER_FORCE_VECTOR_WIDTH and
  POCL_VECTORIZER_PREFER_VECTOR_WIDTH enable more control over the vector width
  chosen by LLVM when using 'loopvec' WG vectorization.

* Inlining of builtin functions has been improved to inline more
  aggressively. This should result in much better performance
  of kernels using many builtins.

* PoCL can now use libraries of vectorized math functions (SLEEF
  on ARM, libmvec or Intel SVML on x86) to implement OpenCL's math
  builtins with vector arguments. This is currently enabled only when
  building with ENABLE_CONFORMANCE=OFF and LLVM >= 18.0

* Rematerialization of work-item private variables across parallel
  regions. Certain values are recomputed instead of storing the
  value in workgroup context arrays. This usually reduces the
  required stack size significantly and improves loop
  vectorization opportunities.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remote driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Support for dynamic device addition and network discovery.
  This new feature allows discovery of remote servers located
  in LAN or WAN environments and enables runtime addition of
  discovered devices to the remote client's platform. Network
  discovery is performed thorugh mDNS, unicast-DNS-SD, and
  DHT-based mechanisms, using Avahi and OpenDHT libraries.

* Support for passing more (vendor-specific) device infos
  from the pocld server to libpocl's remote driver.

* Command buffers that refer to only a single remote device are
  now handled entirely on the server side as described in
  `this publication <https://doi.org/10.1145/3731125.3731129>`
  for massive reductions in communication overhead.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Level Zero driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* When built with -DENABLE_CONFORMANCE=ON, PoCL's Level Zero driver
  is now able to pass the full OpenCL-CTS conformance testsuite.

* Added support for many OpenCL extensions and language features, such as
  __opencl_c_ext_fpXX_XYZ_atomic_load_store, cl_intel_subgroups_XYZ,
  cl_khr_subgroups_XYZ, cl_intel_device_attribute_query,
  cl_intel_split_work_group_barrier, cl_khr_integer_dot_product,
  cl_khr_device_uuid, cl_khr_create_command_queue, cl_khr_pci_bus_info,
  and SPV_INTEL_inline_assembly.

* Support for cl_khr_command_buffer.

* Support for bundling all required files in the built libpocl library.
  This currently only works with the Level Zero driver, CPU driver cannot
  be yet made self-contained.

* Initial support for LLVM's SPIRV backend.

* Windows platform support.

* Support for using LLVMSPIRVLib for SPIR-V input translation,
  instead of llvm-spirv binary.

* Support for LLVM 19 and 20.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CUDA driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Support for LLVM 19 and 20.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Proxy driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Support for SPIR-V input (clCreateProgramWithIL).

* Support for using 'tree-sitter' library to parse &
  extract kernel argument metadata directly from source

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

The initial goal of this work is to provide a CPU fallback for executing
Julia's GPU kernels and applications by leveraging the CPU drivers
in PoCL. For more information, refer to
`the blog post on OpenCL.jl 0.10 <https://juliagpu.org/post/2025-01-13-opencl_0.10/>`_.

===================================
Experimental and work-in-progress
===================================

* New experimental support for Defined Built-in Kernels (DBK) has
  been added to the CPU drivers. These DBKs allow for a
  standardized set of built-in kernels with well-defined
  semantics that can be configured during creation of the OpenCL
  program. Currently the following prototype DBKs are implemented:
  GEMM, matrix multiplication, JPEG en-/de-code, and ONNX runtime
  inference. The Extension documentation draft can be found on
  `github <https://github.com/KhronosGroup/OpenCL-Docs/pull/1007>`_.
  Please note that these DBKs are still under experimentation and
  are subject to change without notice.

* Added support for Intel NPU (or "AI Boost" in the CPU specification)
  as a custom device. Source compilation is not supported yet but GEMM
  and matrix multiplications can be offloaded to the NPU device using
  DBKs. Note that the feature is in very experimental stage and the
  supported DBKs subject to changes.

===================================
Deprecation/feature removal notices
===================================

* The old "work-item replication" work-group function generation
  method was removed to clean up the kernel compiler. It did not
  anymore have any use cases that could not be covered by fully
  unrolling "loops".

* Removed support for building tests & examples with OpenCL < 3.0;
  the tests & examples are always built with PoCL's own CL 3.0 headers,
  and building ICD-enabled PoCL requires ICD that supports OpenCL 3.0.

