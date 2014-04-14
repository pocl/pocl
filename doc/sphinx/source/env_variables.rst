The behavior of pocl can be controlled with multiple environment variables listed
below.

* POCL_BUILDING

 If set, the pocl helper scripts, kernel library and headers are 
 searched first from the pocl build directory.

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

* POCL_KERNEL_COMPILER_OPT_SWITCH

 Override the default "-O3" that is passed to the LLVM opt as a final
 optimization switch.

* POCL_LEAVE_TEMP_DIRS

 If this is set to 1, the kernel compiler temporary directory that contains
 all the intermediate compiler files is left to /tmp. Otherwise, it is
 be cleaned in clReleaseProgram.

* POCL_MAX_PTHREAD_COUNT

 The maximum number of threads created for work group execution in the
 pthread device driver. The default is to determine this from the number of
 hardware threads available in the CPU.

* POCL_MAX_WORK_GROUP_SIZE

 Forces the maximum WG size returned by the device or kernel work group queries
 to be at most this number.

* POCL_TEMP_DIR

 If this is set to an existing directory, pocl uses it as the temporary
 directory for all compilation results. This allows reusing compilation
 results between pocl invocations. If this env is non-NULL, the temp
 directory is not deleted after the Program is freed. Note: the same
 temp dir will be used for all OpenCL programs thus programs
 containing kernels with the same name might use the wrong kernels
 when using this env.

* POCL_USE_PCH

 Use precompiled headers for the OpenCL C built-ins when compiling kernels.
 This is an experimental feature which is known to break on some platforms.

* POCL_VECTORIZE_WORK_GROUPS

 If set to 1, enables the (experimental) work group vectorizer that builds
 vector instructions from multiple work items. Disabled by default for now as it
 worsens the performance almost all of the cases in benchmark.py.

* POCL_VECTORIZE_VECTOR_WIDTH

 If set to number, indicates the width of vector to use during vectorization. Default
 is 8 lanes.

* POCL_VECTORIZE_MEM_ONLY

 If set to 1, indicates that only the memory access operations should be
 vectorized.

* POCL_VECTORIZE_NO_FP

 If set to 1, indicates the vectorization of floating point operations is
 forbidden.

* POCL_VERBOSE

If set to 1, output the LLVM commands as they are executed to compile
and run kernels.

* POCL_WORK_GROUP_METHOD

 The kernel compiler method to produce the work group functions from
 multiple work items. Legal values:

    auto   -- Choose the best available method depending on the
              kernel and the work group size (default). Use
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
               LLVM passes.

    repl   -- Replicate and chain all work items. This results
              in more easily scalarizable private variables.
              However, the code bloat is increased with larger
              local sizes.
