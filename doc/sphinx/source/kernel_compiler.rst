Kernel compiler
---------------

The compilation of kernels in pocl is performed roughly as follows.
In release 0.9 the scripts (referred to below) were replaced by direct
LLVM API calls. The structure remains, e.g. calling script ``pocl-build`` was
replaced with calling function ``call_pocl_build()``. See ``lib/CL/pocl_llvm_api.cc``

#. Produce an LLVM bitcode of the single kernel function.

   The kernel compiler of pocl relies on the OpenCL C frontend of the Clang 
   for parsing the kernel descriptions to LLVM bytecode. The output from
   Clang is a description of the kernel function for a single work-item.

   Done with the help of ``pocl-build`` script that invokes the Clang. See
   ``clBuildProgram.c``.

#. Link in the built-in functions.

   The OpenCL C builtin functions are precompiled to LLVM *bitcode* libraries
   residing under ``lib/kernel/$TARGET``. These are linked to the kernel using
   the ``llvm-link`` tool when the helper script ``pocl-workgroup`` (see the next item).

#. Produce the work-group function.

   The single work-item kernel function is converted to a "work-group" function that
   executes the kernel for all work-items in the local space. This is done
   for targets that cannot execute the single work-item descriptions directly for
   multiple work-item work-groups. This includes the common CPU targets that are not 
   optimized to the "Single Program Multiple Data" (SPMD) workloads. In contrast, 
   GPU architectures (SIMT or SIMD style datapaths) often can input a single kernel 
   description and take care of the parallel execution of multiple kernel instances 
   using their scheduling hardware.

   This part is performed when a kernel execution command is executed (see 
   ``clEnqueueNDRangeKernel.c``).  Only at this point the work-group dimensions are 
   known, after which it is possible to produce functions of the single kernel functions 
   that execute the whole work-group.

#. Code generation for the target.

   The work-group function (which is still in LLVM IR) of the kernel along with the launcher 
   functions are finally converted to the machine code of the target device. This is done in
   the device layer's implementation of the kernel run command. For example, see ``llvm_codegen()``
   in ``lib/CL/devices/common.c``. This function generates a dynamically loaded object of the
   work-group function for actually launching the kernel. The function is called from the CPU
   device layer implementations (``pocl_basic_run()`` of ``lib/CL/devices/basic/basic.c``).
   

Work group function generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The work-group function simply produces the execution of the whole local
space, i.e., it executes the kernel code for all work-items in a work-group. 
Currently one work-group function is produced for each different local
sizes enqueued with ``clEnqueueNDRangeKernel``.

Producing the work-group functions sounds trivial at first, but due to the work-group 
barriers, it becomes slightly complex to perform statically (at compile time). That is, 
one cannot simply create a loop around the whole kernel function, but execute
each region between the barriers for each work-item before proceedings to the
next region, etc.

The work-group functions are produced from a single kernel description using 
a set of LLVM passes in pocl which are described in the following. Most of
the passes can be omitted for targets which input single kernel descriptions
to the instruction set. The source code for the passes is located 
inside *lib/llvmopencl*.

The most complex part of the work-group generation is the part that analyzes
the "barrier regions" and produces static execution of multiple work-items
for them.

The part that analyzes the barrier regions (chains of basic block between
barriers) is done in ``Kernel::getParallelRegions``. It analyzes the kernel
and returns a set of ``ParallelRegion`` objects (set of basic blocks constituting
a single-entry single-exit control flow graphs) inside the kernel to form
the basis for the multi-WI execution. These regions should be executed for
all work-items before proceeding to the next region. However, it is important
to note that the work-items can execute the regions in any order due to the
parallel semantics of OpenCL C.

After the ``ParallelRegions`` have been formed, the static multiple 
work-item execution can be produced in multiple ways.

Currently, two styles of output are supported for the work-group functions:
"fully replicated" (``WorkitemReplication``) and "work-item loops" (``WorkitemLoops``). 
The former is suitable for smaller local sizes and static multi-issue machines; it simply 
duplicates the code for the different work-items to produce the work-groups. 
The latter produces loops around the parallel regions that loop across the
local space. The loops are annotated as parallel using the LLVM parallel loop
annotation. This helps in producing vectorized versions of the work-group
functions using the plain LLVM inner loop vectorizer.

Because in ``WorkitemLoops`` there are only a subset of work-items "alive"
at the same time (the current parallel region iteration), one has to store
variables produced by the work-item in case they are used in other parallel
regions (work-item loops). These variables are stored in "context arrays" and
restore code is injected before the later uses of the variables. 

The context data treatment is not needed for the ``WorkitemReplication`` method because in 
that case, all the work-items are "live" at the same time, and the work-item variables 
are replicated as scalars for each work-item which are visible across the whole 
work-group function without needing to restore them separately.

Work-group autovectorization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The work-group functions can be vectorized "horizontally" (across multiple
work-items in the work-group function) by using the
``WorkitemLoops`` to produce parallel loops for the parallel regions. These
loops are then attempted to vectorized using the standard LLVM's inner-loop 
vectorizer. 

In order to improve vectorization opportunities, some of the "outer loops" (loops inside the 
kernels written by the OpenCL C programmer) are converted to parallel inner loops 
using a pocl pass called ``ImplicitLoopBarriers``. It adds an implicit barrier to the 
beginning of the loop body to force its treatment similarly to a loop with work-group 
barriers (which are executed "horizontally"; each work-item executes the same iteration
of the kernel loop before proceeding to the next one). This allows parallelizing work-items 
across the work-group per kernel for-loop iteration, potentially leading to easier 
horizontal vectorization. The idea is similar to loop switching where the parallel work-item 
loop is switched with the kernel for-loop.

An example should clarify this. A kernel where the work-item loop is created around 
the kernel's loop (here *parallel_WI_loop* marks the place where the work-item loop
is created).

.. code-block:: c

 __kernel
 void DCT(__global float * output,
          __global float * input,
          __global float * dct8x8,
          __local  float * inter,
          const    uint    width,
          const    uint    blockWidth,
          const    uint    inverse)
 {
     /* ... */
 /* parallel_WI_loop { */
     for(uint k=0; k < blockWidth; k++)
     {
         uint index1 = (inverse)? i*blockWidth + k : k * blockWidth + i;
         uint index2 = getIdx(groupIdx, groupIdy, j, k, blockWidth, width);
 
         acc += dct8x8[index1] * input[index2];
     }
     inter[j*blockWidth + i] = acc;
 /* } */
     barrier(CLK_LOCAL_MEM_FENCE);
     /* ... */
 }

The kernel-loop cannot be easily vectorized as the ``blockWidth`` is a kernel parameter,
i.e., the vectorizer does not know how many times the loop iterates. Also, for vectorizing
intra kernel-loops the compiler has to perform the regular sequential C alias analysis to 
figure out whether and how the loop iterations are dependent on each other. 

In contrast, when we are able to place the parallel work-item loop *inside* the
kernel-loop, we create a potentially more easily vectorizable loop that executes
operations from multiple work-items in parallel:

.. code-block:: c

 /* ... */
 for(uint k=0; k < blockWidth; k++)
 {
 /* parallel_WI_loop { */
   uint index1 = (inverse)? i*blockWidth + k : k * blockWidth + i;
   uint index2 = getIdx(groupIdx, groupIdy, j, k, blockWidth, width);
   
   acc += dct8x8[index1] * input[index2];
   /* } */
   /* implicit barrier added here */
 }
 inter[j*blockWidth + i] = acc;
 barrier(CLK_LOCAL_MEM_FENCE);

 /* ... */

The difficulty with this pass is that, of course, we need to make sure it is legal to 
add the barrier. The OpenCL barrier semantics require either all or none of the WIs to
reach the barrier at each iteration. This is satisfied at least when

* The loop exit condition does not depend on the WI, and
* all or none of the WIs always enter the loop.

In order to prove these cases, a pass called ``VariableUniformityAnalysis`` is used to
separate variables that are *uniform* (same for all work-items) and *variable* (vary
between work-items). It falls back to *variable* in case it cannot prove the
uniformity.

.. _wg-functions:

Creating the work-group function launchers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The kernel compiler creates functions for launching the work-group functions that
are built into the same module as the kernel. These functions can be used as
access points from the host code or from separate control/scheduler code at the device
side.

``Workgroup`` pass creates a launcher which calls the work-group function using the arguments
passed from the host side. It also setups a "context struct" which contains the data needed 
by functions that query the work-group ids etc. This context struct is added as a new argument 
to the original kernel argument list.

``Workgroup`` generates two versions for launching the kernel which are used to
depending which style of parameter passing is desired: 

* ``KERNELNAME_workgroup()`` 

 for the case where the host and device shares 
 a single memory (the basic CPU host+device setup). Scalars are passes directly in the
 argument array and everything resides in the default address space 0. 

* ``KERNELNAME_workgroup_fast()`` 

 can be used when there is a separate argument space located in a separate global 
 address space (from the device point of view). This assumes that buffer arguments (pointers) are
 passed directly as pointer values and scalar values are also passed
 as pointers to objects in an "argument space" in the global memory (that is
 accessible from the host). Explicit global address space identifier is used to access
 the argument data.


*NOTE: There's a plan to remove the first workgroup function and unify the way the
workgroups are called from the host code. Thus, the former version might go away.*

Assisting transformations
^^^^^^^^^^^^^^^^^^^^^^^^^

Several transformations are done to the LLVM bytecode to assist in the work-group
generation effort. Most of them are required by the actual parallel region formation.
Some of them are listed in the following:

* ``Flatten`` 

 Fully inlines everything inside the kernel so there are no function
 calls in the resulting kernel function. It does it by adding the LLVM attribute ``AlwaysInLine``
 to all child functions of the kernel after which the LLVM pass ``-always-inline``
 is used to actually perform the inlining. This pass is not strictly required unless
 the child functions of the kernel contain barrier calls.

* ``WorkitemHandlerChooser`` 

 Does the choice of how to produce the work-group
 functions for the kernel at hand (the loops or the full replication).

* ``PHIsToAllocas`` 

 Required by the ``WorkitemLoops`` but not by the ``WorkitemReplication`` work-group
 function generation method. 
 It converts all PHIs to allocas in order to make it possible to inject context restore code 
 in the beginning of join points. This is due to the limitation that PHI nodes must
 be at the beginning of the basic blocks and in some cases we need to restore
 variables (load from a context array in memory) used by the PHI nodes because 
 they originate from a different parallel region. It is similar to ``-reg2mem``
 of LLVM except that it touches only PHI nodes.

* ``AllocasToEntry`` 

 Can be used by targets that do not support dynamic stack objects to
 move all stack allocations to the function entry block. 

* ``GenerateHeader``

 This pass is used to produce a metadata file of the kernel. The file contains
 information of the argument types that are used by the host side. The data is
 passed to the host side via a plugin module that contains a struct with the info.
 The name, GenerateHeader, comes from this. It generates a C header file with the
 info which is compiled to the plugin module. It is clear that this way of 
 retrieving the metadata is very cumbersome and slow, and the functionality is 
 being refactored to use ``libClang`` directly from the host code to retrieve
 the information.

* ``AutomaticLocals``

 This pass is converts the automatic local buffers
 to kernel arguments. This is to enforce the similar treatment of the both
 types of local buffers, the ones passed as arguments and the ones instantiated
 in the kernel.

* ``TargetAddressSpaces``

 Internally pocl uses fixed address space ids to denote the different OpenCL
 address spaces.  That is, Clang generates LLVM IR that uses these address space
 ids, even with targets that have a single flat address space in reality.
 This is to differentiate the different type of pointers for treating locals
 correctly, and also for assisting alias analysis (different address spaces are
 disjoint, thus accesses to them won't alias each other).

 TargetAddressSpaces is a pass that converts these fake ids to the ones expected
 by the target.  This pass can be short cutted in case the backend for the target
 can flatten the ids safely.  However, recently (as of LLVM 3.7) there has been new
 problems with some optimizations (at least LoopVectorizer) that get confused with
 the fake ids.  Therefore it is recommended the ids are flattened even if not
 strictly needed for a target at hand to avoid these issues and make some LLVM
 optimizations more efficient.

 A more robust version of AS handling might be to rely on metadata when differentiating
 the pointers and already in Clang use the target's address spaces in the IR. This
 would ensure the LLVM IR passes would not get confused by the fake ids.


.. _opencl-optimizations:

Other OpenCL-specific optimizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``WorkitemAliasAnalyzer``

 Adds OpenCL-specific information to the alias analyzer. Currently exploits the
 fact that accesses from two work-items cannot alias within the same "parallel
 region" and that the OpenCL C address spaces are disjoint (accesses to different
 address spaces do not alias).


