Memory management
-----------------

This section explains how pocl supports multiple address spaces and
host-side memory management of device memory.

Multiple logical address spaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, Clang (at least version 3.3 and older) converts the OpenCL C address space 
qualifiers to *target specific* address space identifiers. That is, e.g., for the common CPU 
targets with single uniform address space, all of the OpenCL address spaces are mapped to the
address space identifier 0 (the default C address space). For multiple address space
LLVM backends such as AMD GPUs there are different ids produced for the OpenCL C address spaces,
but they differ from those of the TCE backend, etc. Thus, after the Clang processing of
the kernel source, the information of the original OpenCL C address spaces is lost or is 
target specific, preventing or complicating the special treatment of the pointers pointing 
to (logically) different address spaces (e.g. OpenCL disjoint address space alias analysis,
see :ref:`opencl-optimizations`).

pocl's kernel compiler needs to know the original logical address spaces in the kernel during
some of its processing steps. In order to unify these parts of the kernel compiler, pocl 
uses the "fake address space map" mechanism of Clang to force pocl-known *separate* ids to be 
produced for each of the OpenCL C logical address spaces in the frontend. 

Before the code generation, the forced OpenCL C logical address space ids should be mapped to 
the backend understood ones. This can be done in the kernel compiler pass ``TargetAddressSpaces``. 
It goes through all the memory references in the bitcode and maps their address space ids to the 
target specific ones. In case it is known that the targeted backend either understands the logical
address space ids (or simply maps everything to 0 there aswell), this processing is
skipped (and left for the backend). 

Managing the device memories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a buffer is allocated on the device, the device layer implementation is responsible for
making sure the device has enough free space on the memory the given address space is mapped to
and for returning a handle for later referring to that memory. 

When all the memories are mapped to a single address space shared with the host memory (the case 
with CPU host+device setups), one could simply use ``malloc()`` for this. However, for the 
heterogeneous device setup where the device has separate memories, one cannot
use the host's malloc function for managing the memory spaces. For this, pocl implements a simple
memory allocator called ``bufalloc``. With bufalloc it is possible to manage chunks of memory 
allocated from a region of addresses. The allocator is optimized for speed and to minimize
fragmentation assuming largish chunks of memory (the input/output buffers) are allocated and 
freed at once.

Bufalloc can be used for host-side management of continuous ranges of memories on the
device side. Bufalloc is used for managing the memory also in the ``pthread/basic`` 
CPU device implementations for testing and optimization purposes. For an example of 
its use for managing memory in the heterogeneous separated memory setup, one should take 
a look at the TCE device layer code (``lib/CL/devices/tce/tce_common.cc``). For TCE devices 
it is assumed there are actual separated physical address spaces for both the *local* and *global* 
address spaces. The device layer implementation manages allocations from both of these spaces 
using two instances of bufalloc memory regions.

When passing buffer pointers to the kernel/work-group launchers, the memory addresses are
passed as integer values. The values passed from the host are casted to the actual
address-space qualified LLVM IR pointers for calling the kernels with correct types
by the work-group function (see :ref:`wg-functions`).

Custom memory management for pthread device
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enabled by CMake option USE_POCL_MEMMANAGER. This is only useful for certain
uncommon setups, where pocl is expected to allocate a huge number of queue or
event objects. For most available OpenCL programs / tests / benchmarks, there
is no measurable difference in speed.

Advantages:
* allocation of queues/events/command objects can be a lot faster

Disadvantages:
* memory allocated for those objects is never free()d;
  it's only returned to allocation pool
* debugging tools will not detect use-after-free bugs on said objects
