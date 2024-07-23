Usage
===========

The basic usage of pocl should be as easy as any other OpenCL implementation.

While it is possible to link against pocl directly, the recommended way is to
use the ICD interface.

.. _linking-with-icd:

Linking your program with pocl through an icd loader
----------------------------------------------------

You can link your OpenCL program against an ICD loader. If your ICD loader is
correctly configured to load pocl, your program will be able to use pocl.
See the section below for more information about ICD and  ICD loaders.

Example of compiling an OpenCL host program using the free ocl-icd loader::

   gcc example1.c -o example `pkg-config --libs --cflags OpenCL`

Example of compiling an OpenCL host program using the AMD ICD loader (no
pkg-config support)::

   gcc example1.c -o example -lOpenCL

Installable client driver (ICD)
-------------------------------

pocl is built with the ICD extensions of OpenCL by default. This allows you
to have several OpenCL implementations concurrently on your computer, and
select the one to use at runtime by selecting the corresponding cl_platform.
ICD support can be disabled by adding the flag::

  -DENABLE_ICD=OFF

to the CMake invocation.

The ocl-icd ICD loader allows to use the OCL_ICD_VENDORS environment variable
to specify a (non-standard) replacement for the /etc/OpenCL/vendors directory.

An ICD loader is an OpenCL library acting as a "proxy" to one of the various OpenCL
implementations installed in the system. pocl does not provide an ICD loader itself,
but NVidia, AMD, Intel, Khronos, and the free ocl-icd project each provides one.

* `ocl-icd <https://github.com/OCL-dev/ocl-icd>`_
* `Khronos <http://www.khronos.org/opencl/>`_

Linking your program directly with pocl
---------------------------------------

Passing the appropriate linker flags is enough to use pocl in your
program. However, please bear in mind that:

The pkg-config tool is used to locate the libraries and headers in
the installation directory.

Example of compiling an OpenCL host program against pocl using
the pkg-config::

   gcc example1.c -o example `pkg-config --libs --cflags pocl`

In this link mode, your program will always require the pocl OpenCL library. It
wont be able to run with another OpenCL implementation without recompilation.

Using pocl on MacOSX
--------------------

On MacOSX, you can either link your program directly with pocl or link through the
ICD loader by KhronosGroup.

Even if you use an ICD loader, the Apple OpenCL implementation will still be invisible,
unless you use a wrapper library to expose the Apple OpenCL implementation as an ICD.

When ENABLE_ICD is turned off and an application links directly with PoCL, the only
platform that is visible to the application will be PoCL.

.. _pocl-env-variables:

Tuning pocl behavior with ENV variables
---------------------------------------

The behavior of pocl can be controlled with multiple environment variables
listed below. The variables are helpful both when using and when developing
pocl.

.. highlight:: bash

- **POCL_AFFINITY**

  Linux-only, specific to 'cpu' driver. If set to 1, each thread of
  the driver sets its affinity to its index. This may be useful
  with very long running kernels, or when using subdevices.
  Defaults to 0 (most people don't need this).

- **POCL_BINARY_SPECIALIZE_WG**

  By default the PoCL program binaries store generic kernel binaries which
  can be executed across any grid dimensions. This configuration variable
  can be used to also include specialized work-group functions in the binaries, by
  defining a comma separated list of strings that describe the specialized
  versions. The strings adhere to the directory names in the PoCL cache
  from which the binaries are captured.

  Example::

    POCL_BINARY_SPECIALIZE_WG='2-1-1,0-0-0-goffs0,13-1-1-smallgrid,128-2-1-goffs0-smallgrid' poclcc [...]

  This makes poclcc generate a binary which contains the generic work-group
  function binary, a work-group function that is specialized for local size
  of 2x1x1, another with generic local size but specialized for the global
  offset at origo, one with local size of 13x1x1, but which is specialized
  for a "small grid" (size defined by the device driver), and finally one
  that is specialized for local size 128x2x1, an origo global offset and
  a small grid.

- **POCL_BITCODE_FINALIZER**

  Defines a custom command that can manipulate the final kernel work-group
  function bitcode produced after all LLVM optimizations and before entering code
  generation. This can be useful, for example, to add instrumentation to the LLVM
  bitcode before proceeding to the backend.

  Example::

    POCL_BITCODE_FINALIZER='verificarlo %(bc) --emit-llvm -o %(bc)' examples/example1/example1

  This results in running the above command with '%(bc)' strings replaced with
  the path of the final bitcode's temporary file. Note that the modified
  bitcode should be written over the same file for it to get picked to the
  code generation.

  Please note that setting the env doesn't force regeneration of the kernel
  binaries if they are found in the kernel compiler cache. You can either
  use POCL_KERNEL_CACHE=0 to disable the kernel cache, or wipe the kernel
  cache directory manually to force kernel binary rebuild.

- **POCL_BUILDING**

 If  set, the pocl helper scripts, kernel library and headers are
 searched first from the pocl build directory. Only has effect if
 ENABLE_POCL_BUILDING was enabled at build (by default it is).

- **POCL_CACHE_DIR**

 If this is set to an existing directory, pocl uses it as the cache
 directory for all compilation results. This allows reusing compilation
 results between pocl invocations. If this env is not set, then the
 default cache directory will be used, which is ``$XDG_CACHE_HOME/pocl/kcache``
 (if set) or ``$HOME/.cache/pocl/kcache/`` on Unix-like systems.

- **POCL_CPU_LOCAL_MEM_SIZE**

 Set the local memory size of the CPU devices (cpu, cpu-minimal, cpu-tbb) to the
 given amount in bytes instead of the default one.

- **POCL_CPU_MAX_CU_COUNT**

 The maximum number of threads created for work group execution in the
 'cpu' device driver. The default is to determine this from the number of
 hardware threads available in the CPU.

- **POCL_CPU_VENDOR_ID_OVERRIDE**

 Overrides the vendor id reported by PoCL for the CPU drivers.
 For example, setting the vendor id to be 32902 (0x8086) and setting the driver
 version using **POCL_DRIVER_VER_OVERRIDE** to "2023.16.7.0.21_160000" (or such) can
 be used to convince binary-distributed DPC++ compilers to compile and run SYCL
 programs on the PoCL-CPU driver.

- **POCL_DEBUG**

 Enables debug messages to stderr. This will be mostly messages from error
 condition checks in OpenCL API calls and Event/API timing information.
 Useful to e.g. distinguish between various reasons a call could return
 CL_INVALID_VALUE. If clock_gettime is available, messages
 will include a timestamp.

 The old way (setting POCL_DEBUG to 1) has been updated to support categories.
 Using this limits the amount of debug messages produced. Current options are:
 'error', 'warning', 'general', 'memory', 'llvm', 'events', 'cache', 'locking',
 'refcounts', 'timing', 'hsa', 'tce', 'cuda', 'vulkan', 'proxy' and 'all'.
 Note: setting POCL_DEBUG to 1 still works and equals error+warning+general.

- **POCL_DEBUG_LLVM_PASSES**

 When set to 1, enables debug output from LLVM passes during optimization.

- **POCL_DEVICES** and **POCL_x_PARAMETERS**

 POCL_DEVICES is a space separated list of the device instances to be enabled.
 This environment variable is used for the following devices:

 *         **cpu-minimal** A minimalistic example device driver for executing
                           kernels on the host CPU. No multithreading.

 *         **cpu**      Execution of OpenCL kernels on the host CPU using
                        (by default) all available CPU threads via pthread library.

 *         **cpu-tbb**  Uses the Intel Threading Building Blocks (or oneTBB) library
                        for task scheduling on the host CPU.

 *         **cuda**     An experimental driver that uses libcuda to execute on NVIDIA GPUs.

 *         **hsa**      Uses HSA Runtime API to control HSA-compliant
                        kernel agents that support HSAIL finalization
			(deprecated).

 *         **vulkan**   An experimental driver that uses Vulkan and SPIR-V for executing on
	                Vulkan supported devices.

 *         **ttasim**   Device that simulates a TTA device using the
                        TCE's ttasim library. Enabled only if TCE libraries
                        installed.

 *         **level0**   An experimental driver that uses libze to execute on Intel GPUs.

 If POCL_DEVICES is not set, one cpu device will be used.
 To specify parameters for drivers, the POCL_<drivername><instance>_PARAMETERS
 environment variable can be specified (where drivername is in uppercase).
 Example::

  export POCL_DEVICES="cpu ttasim ttasim"
  export POCL_TTASIM0_PARAMETERS="/path/to/my/machine0.adf"
  export POCL_TTASIM1_PARAMETERS="/path/to/my/machine1.adf"

 Creates three devices, one 'cpu' device with multithreading and two
 TTA device simulated with the ttasim. The ttasim devices gets a path to
 the architecture description file of the tta to simulate as a parameter.
 POCL_TTASIM0_PARAMETERS will be passed to the first ttasim driver instantiated
 and POCL_TTASIM1_PARAMETERS to the second one.

- **POCL_DRIVER_VERSION_OVERRIDE**

  Can be used to override the driver version reported by PoCL.
  See **POCL_CPU_VENDOR_ID_OVERRIDE** for an example use case.

- **POCL_EXTRA_BUILD_FLAGS**

 Adds the contents of the environment variable to all clBuildProgram() calls.
 E.g. ``POCL_EXTRA_BUILD_FLAGS="-g -cl-opt-disable"`` can be useful for force
 adding debug data all the built kernels to help debugging kernel issues
 with tools such as gdb or valgrind.

- **POCL_IGNORE_CL_STD**

 Ignores any ``--cl-std`` options passed to clBuildProgram(). This is useful
 to force-run programs that set the version to 2.x although they do not need
 all of its features which the targeted 3.x driver might not implement.

- **POCL_KERNEL_CACHE**

 If this is set to 0 at runtime, kernel compilation files will be deleted at
 clReleaseProgram(). Note that it's currently not possible for pocl to avoid
 interacting with LLVM via on-disk files, so pocl requires some disk space at
 least temporarily (at runtime).

- **POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES**

 If this is set to 1, the kernel compiler cache/temporary directory that
 contains all the intermediate compiler files are left as it is. This
 will be handy for debugging

- **POCL_LEVEL0_JIT**

 Sets up Just-In-Time compilation in the Level0 driver.
 (see :ref:`pocl-level0-driver` for details)
 Accepted values: {0,1,auto}

 *   0 = always disable JIT
 *   1 = always use JIT,
 *   auto (default) = guess based on program's kernel count & SPIR-V size.

- **POCL_LEVEL0_LINK_OPT**

 If non-empty string, runs llvm-opt with this option after the linking step,
 before converting to SPIRV and handing over to L0 driver. Default: empty.

- **POCL_LLVM_VERIFY**

  if enabled, some drivers (CUDA, CPU, Level0) use an extra step of
  verification of LLVM modules at certain stages (program.bc always,
  kernel bitcode (parallel.bc) only with some drivers).
  Defaults to 0 if CMAKE_BUILD_TYPE=Debug and 1 otherwise.

- **POCL_MAX_WORK_GROUP_SIZE**

 Forces the maximum WG size returned by the device or kernel work group queries
 to be at most this number. For certain devices, this is can only be lower than
 their hardware limits.

- **POCL_MEMORY_LIMIT**

 Integer option, unit: gigabytes. Limits the total global memory size
 reported by pocl for the CPU devices (this will also affect
 local/constant/max-alloc-size numbers, since these are derived from
 global mem size).

- **POCL_OFFLINE_COMPILE**

 Bool. When enabled(==1), some drivers will create virtual devices which are only
 good for creating pocl binaries. Requires those drivers to be compiled with support
 for compilation for those devices.


- **POCL_SIGFPE_HANDLER**

 Defaults to 1. If set to 0, pocl will not install the SIGFPE handler.
 See :ref:`known-issues`

- **POCL_SIGUSR2_HANDLER**

 When set to 1 (default 0), pocl installs a SIGUSR2 handler that will print
 some debugging information. Currently it prints the count of live cl_* objects
 by type (buffers, events, etc).

- **POCL_STARTUP_DELAY**

  Default 0. If set to an integer N > 0, libpocl will make a pause of N seconds
  once, when it's loading. Useful e.g. to set up a LTTNG tracing session.

- **POCL_TBB_DEV_PER_NUMA_NODE** can be set to either 0 or 1 (default). If set,
  PoCL TBB driver creates a separate OpenCL device per each NUMA node.

- **POCL_TBB_GRAIN_SIZE** can be set specify a grain size for all
  dimensions. More information can be found in TBB documentation.

- **POCL_TBB_PARTITIONER** can be set to one of ``affinity``,``auto``,
  ``simple``,``static`` to select a partitioner. If no
  partitioner is selected, the TBB library will select the auto partitioner by
  default. More information can be found in TBB documentation.

- **POCL_TRACING**, **POCL_TRACING_OPT** and **POCL_TRACING_FILTER**

 If POCL_TRACING is set to some tracer name, then all events
 will be traced automatically. Depending on the backend, traces
 may be output in different formats and collected in a different way.
 POCL_TRACING_FILTER is a comma separated list of string to
 indicate which event status should be filtered. For instance to trace
 complete and running events POCL_TRACING_FILTER should be set
 to "complete,running". Default behavior is to trace all events.

 * **cq** -- Dumps a simple per-kernel execution time statistics at the
          program exit time which is collected from command queue
          start and finish time stamps. Useful for quick and easy profiling
          purposes with accurate kernel execution time stamps produced
          in a per device way. Currently only tracks kernel timings, and
          POCL_TRACING_FILTER has no effect.
 * **text** -- Basic text logger for each events state
              Use POCL_TRACING_OPT=<file> to set the
              output file. If not specified, it defaults to
              pocl_trace_event.log
 * **lttng** -- LTTNG tracepoint support. Requires pocl to be built with ``-DENABLE_LTTNG=YES``.
              When activated, a lttng session must be started.
              The following tracepoints are available:

              * pocl_trace:ndrange_kernel -> Kernel execution
              * pocl_trace:read_buffer    -> Read buffer
              * pocl_trace:write_buffer   -> Write buffer
              * pocl_trace:copy_buffer    -> Copy buffer
              * pocl_trace:map            -> Map image/buffer
              * pocl_trace:command        -> other commands

              For more information, please see lttng documentation:
              http://lttng.org/docs/#doc-tracing-your-own-user-application

- **POCL_VECTORIZER_REMARKS**

 When set to 1, prints out remarks produced by the loop vectorizer of LLVM
 during kernel compilation.

- **POCL_VULKAN_VALIDATE**

 When set to 1, and the Vulkan implementation has the validation layers,
 enables the validation layers in the driver. You will also need POCL_DEBUG=vulkan
 or POCL_DEBUG=all to see the output printed.

- **POCL_WORK_GROUP_METHOD**

 The kernel compiler method to produce the work group functions from
 multiple work items. Legal values:

 * **auto**   -- Choose the best available method depending on the
              kernel and the work group size. Use
              POCL_FULL_REPLICATION_THRESHOLD=N to set the
              maximum local size for a work group to be
              replicated fully with 'repl'. Otherwise,
              'loops' is used.

 * **loops**  -- Create for-loops that execute the work items
              (under stabilization). The drawback is the
              need to save the thread contexts in arrays.

              The loops will be unrolled a certain number of
              times of which maximum can be controlled with
              POCL_WILOOPS_MAX_UNROLL_COUNT=N environment
              variable (default is to not perform unrolling).

 * **loopvec** -- Create work-item for-loops (see 'loops') and execute
               the LLVM LoopVectorizer. The loops are not unrolled
               but the unrolling decision is left to the generic
               LLVM passes (the default).

 * **repl**   -- Replicate and chain all work items. This results
              in more easily scalarizable private variables, thus
              might avoid storing work-item context to memory.
              However, the code bloat is increased with larger
              WG sizes.
    
 * **cbs**    -- Use continuation-based synchronization to execute work-items
              on non-SPMD devices.
              CBS is expected to work for kernels that 'loops' does not support.
              For most other kernels it is expected to perform slightly worse.
              Also enables the LLVM LoopVectorizer.

              An in-depth explanation of the implementation of CBS and how it
              compares to the other approaches can be found in
              [this thesis](https://joameyer.de/hipsycl/Thesis_JoachimMeyer.pdf).

- **POCL_WORK_GROUP_SPECIALIZATION**

  PoCL specializes work-groups at kernel command launch time by default
  to optimize the execution performance with the cost of cached variations
  of the kernels with the different specialization values.

  The kernel command parameters PoCL currently specializes with include
  the local size, global offset zero or non-zero and maximum grid size.
  The specialization can be disabled by setting this environment variable to 0.
