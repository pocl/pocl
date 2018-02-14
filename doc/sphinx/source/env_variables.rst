Tuning pocl behavior with ENV variables
---------------------------------------

The behavior of pocl can be controlled with multiple environment variables
listed below. The variables are helpful both when using and when developing
pocl.

- **POCL_AFFINITY**
 Linux-only, specific to pthread driver. If set to 1, each thread of
 the pthread CPU driver sets its affinity to its index. This may be
 useful with very long running kernels, or when using subdevices
 (lets any idle cores enter deeper sleep). Defaults to 0 (most
 people don't need this).

- **POCL_BUILDING**

 If  set, the pocl helper scripts, kernel library and headers are 
 searched first from the pocl build directory.

- **POCL_CACHE_DIR**

 If this is set to an existing directory, pocl uses it as the cache
 directory for all compilation results. This allows reusing compilation
 results between pocl invocations. If this env is not set, then the
 default cache directory will be used, which is ``$XDG_CACHE_HOME/pocl/kcache``
 (if set) or ``$HOME/.cache/pocl/kcache/`` on Unix-like systems.

- **POCL_DEBUG**

 Enables debug messages to stderr. This will be mostly messages from error
 condition checks in OpenCL API calls and Event/API timing information.
 Useful to e.g. distinguish between various reasons a call could return
 CL_INVALID_VALUE. If clock_gettime is available, messages
 will include a timestamp.

 The old way (setting POCL_DEBUG to 1) has been updated to support categories.
 Using this limits the amount of debug messages produced. Current options are:
 error,warning,general,memory,llvm,events,cache,locking,refcounts,timing,hsa,tce,all.
 Note: setting POCL_DEBUG to 1 still works and equals error+warning+general.

- **POCL_DEBUG_LLVM_PASSES**

 When set to 1, enables debug output from LLVM passes during optimization.

- **POCL_DEVICES** and **POCL_x_PARAMETERS**

 POCL_DEVICES is a space separated list of the device instances to be enabled.
 This environment variable is used for the following devices:

 *         **basic**    A minimalistic example device driver for executing
                        kernels on the host CPU. No multithreading.

 *         **pthread**  Native kernel execution on the host CPU with
                        threaded execution of work groups using pthreads.

 *         **ttasim**   Device that simulates a TTA device using the
                        TCE's ttasim library. Enabled only if TCE libraries
                        installed.

 *         **hsa***     Uses HSA Runtime API to control HSA-compliant
                        kernel agents that support HSAIL finalization.

 If POCL_DEVICES is not set, one pthread device will be used.
 To specify parameters for drivers, the POCL_<drivername><instance>_PARAMETERS
 environment variable can be specified (where drivername is in uppercase).
 Example::

  export POCL_DEVICES="pthread ttasim ttasim"
  export POCL_TTASIM0_PARAMETERS="/path/to/my/machine0.adf"
  export POCL_TTASIM1_PARAMETERS="/path/to/my/machine1.adf"

 Creates three devices, one CPU device with pthread multithreading and two
 TTA device simulated with the ttasim. The ttasim devices gets a path to
 the architecture description file of the tta to simulate as a parameter.
 POCL_TTASIM0_PARAMETERS will be passed to the first ttasim driver instantiated
 and POCL_TTASIM1_PARAMETERS to the second one.

- **POCL_IMPLICIT_FINISH**

 Add an implicit call to clFinish afer every clEnqueue* call. Useful mostly for
 pocl internal development, and is enabled only if pocl is configured with
 ``--enable-debug``.

- **POCL_KERNEL_CACHE**

 If this is set to 0 at runtime, kernel compilation files will be deleted at
 clReleaseProgram(). Note that it's currently not possible for pocl to avoid
 interacting with LLVM via on-disk files, so pocl requires some disk space at
 least temporarily (at runtime).

- **POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES**

 If this is set to 1, the kernel compiler cache/temporary directory that
 contains all the intermediate compiler files are left as it is. This
 will be handy for debugging

- **POCL_MAX_PTHREAD_COUNT**

 The maximum number of threads created for work group execution in the
 pthread device driver. The default is to determine this from the number of
 hardware threads available in the CPU.

- **POCL_MAX_WORK_GROUP_SIZE**

 Forces the maximum WG size returned by the device or kernel work group queries
 to be at most this number. For certain devices, this is can only be lower than
 their hardware limits.

- **POCL_MEMORY_LIMIT**

 Integer option, unit: gigabytes. Limits the total global memory size
 reported by pocl for the pthread/basic devices (this will also affect
 local/constant/max-alloc-size numbers, since these are derived from
 global mem size).

- **POCL_OFFLINE_COMPILE**

 Bool. When enabled(==1), some drivers will create virtual devices which are only
 good for creating pocl binaries. Requires those drivers to be compiled with support
 for compilation for those devices.

- **POCL_VECTORIZER_REMARKS**

 When set to 1, prints out remarks produced by the loop vectorizer of LLVM
 during kernel compilation.

- **POCL_WORK_GROUP_METHOD**

 The kernel compiler method to produce the work group functions from
 multiple work items. Legal values:

    auto   -- Choose the best available method depending on the
              kernel and the work group size. Use
              POCL_FULL_REPLICATION_THRESHOLD=N to set the
              maximum local size for a work group to be
              replicated fully with 'repl'. Otherwise,
              'loops' is used.

    loops  -- Create for-loops that execute the work items
              (under stabilization). The drawback is the
              need to save the thread contexts in arrays.

              The loops will be unrolled a certain number of
              times of which maximum can be controlled with
              POCL_WILOOPS_MAX_UNROLL_COUNT=N environment
              variable (default is to not perform unrolling).

    loopvec -- Create work-item for-loops (see 'loops') and execute
               the LLVM LoopVectorizer. The loops are not unrolled
               but the unrolling decision is left to the generic
               LLVM passes (the default).

    repl   -- Replicate and chain all work items. This results
              in more easily scalarizable private variables, thus
              might avoid storing work-item context to memory.
              However, the code bloat is increased with larger
              WG sizes.

- **POCL_SIGFPE_HANDLER**

 Defaults to 1. If set to 0, pocl will not install the SIGFPE handler.
 See :ref:`sigfpe-handler`

- **POCL_TRACE_EVENT**, **POCL_TRACE_EVENT_OPT** and **POCL_TRACE_EVENT_FILTER**

 If POCL_TRACE_EVENT is set to some tracer name, then all events
 will be traced automatically. Depending on the backend, traces
 may be output in different formats.
 POCL_TRACE_EVENT_FILTER is a comma separated list of string to 
 indicate which event status should be filtered. For instance to trace
 complete and running events POCL_TRACE_EVENT_FILTER should be set
 to "complete,running". Default behavior is to trace all events.

    text   -- Basic text logger for each events state
              Use POCL_TRACE_EVENT_OPT=<file> to set the 
              output file. If not specified, it defaults to
              pocl_trace_event.log
    lttng  -- LTTNG tracepoint support. When activated, a lttng session
              must be started. The following tracepoints are available:
               - pocl_trace:ndrange_kernel -> Kernel execution
               - pocl_trace:read_buffer    -> Read buffer
               - pocl_trace:write_buffer   -> Write buffer
               - pocl_trace:copy_buffer    -> Copy buffer
               - pocl_trace:map            -> Map image/buffer
               - pocl_trace:command        -> other commands

              For more information, please see lttng documentation:
              http://lttng.org/docs/#doc-tracing-your-own-user-application

