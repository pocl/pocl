The behavior of pocl can be controlled with multiple environment variables listed
below.

* POCL_BUILDING

 If set, the pocl helper scripts, kernel library and headers are 
 searched first from the pocl build directory.

* POCL_CACHE_DIR

 If this is set to an existing directory, pocl uses it as the cache
 directory for all compilation results. This allows reusing compilation
 results between pocl invocations. If this env is not set, then the
 default cache directory will be used

* POCL_DEBUG

 Enables debug messages to stderr. This will be mostly messages from error
 condition checks in OpenCL API calls. Useful to e.g. distinguish between various
 reasons a call can return CL_INVALID_VALUE. If clock_gettime is available,
 messages will include a timestamp.

* POCL_DEVICES and POCL_x_PARAMETERS

 POCL_DEVICES is a space separated list of the device instances to be enabled.
 This environment variable is used for the following devices:

 *         basic        A minimalistic example device driver for executing
                        kernels on the host CPU. No multithreading.

 *         pthread      Native kernel execution on the host CPU with
                        threaded execution of work groups using pthreads.

 *         ttasim       Device that simulates a TTA device using the
                        TCE's ttasim library. Enabled only if TCE libraries
                        installed.

 If POCL_DEVICES is not set, one pthread device will be used.
 To specify parameters for drivers, the POCL_<drivername><instance>_PARAMETERS
 environment variable can be specified (where drivername is in uppercase).
 Example:

  export POCL_DEVICES="pthread ttasim ttasim"
  export POCL_TTASIM0_PARAMETERS="/path/to/my/machine0.adf"
  export POCL_TTASIM1_PARAMETERS="/path/to/my/machine1.adf"

 Creates three devices, one CPU device with pthread multithreading and two
 TTA device simulated with the ttasim. The ttasim devices gets a path to
 the architecture description file of the tta to simulate as a parameter.
 POCL_TTASIM0_PARAMETERS will be passed to the first ttasim driver instantiated
 and POCL_TTASIM1_PARAMETERS to the second one.

* POCL_IMPLICIT_FINISH

 Add an implicit call to clFinish afer every clEnqueue* call. Useful mostly for
 pocl internal development, and is enabled only if pocl is configured with
 '--enable-debug'.

* POCL_KERNEL_CACHE

 If this is set to 0 at runtime, kernel-cache will be forcefully disabled even if
 its enabled in configure step

* POCL_KERNEL_CACHE_IGNORE_INCLUDES

 By default, the kernel compiler cache does not cache kernels that 
 have #include clauses. Setting this to 1 changes this so that the
 includes are ignored and not scanned for changes. Use this to
 improve the kernel compiler hit ratio in case you know that the 
 included files are not modified across runs.

* POCL_KERNEL_COMPILER_OPT_SWITCH

 Override the default "-O3" that is passed to the LLVM opt as a final
 optimization switch.

* POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES

 If this is set to 1, the kernel compiler cache/temporary directory that
 contains all the intermediate compiler files are left as it is. This
 will be handy for debugging

* POCL_MAX_PTHREAD_COUNT

 The maximum number of threads created for work group execution in the
 pthread device driver. The default is to determine this from the number of
 hardware threads available in the CPU.

* POCL_MAX_WORK_GROUP_SIZE

 Forces the maximum WG size returned by the device or kernel work group queries
 to be at most this number.

* POCL_VECTORIZER_REMARKS

 When set to 1, prints out remarks produced by the loop vectorizer of LLVM
 during kernel compilation.

* POCL_VERBOSE

 If set to 1, output the LLVM commands as they are executed to compile
 and run kernels.

* POCL_WORK_GROUP_METHOD

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
