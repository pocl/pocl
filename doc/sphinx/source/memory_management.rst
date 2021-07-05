Memory management in PoCL
--------------------------

This explains how PoCL implements buffers in device memory, and how it deals
with various other aspects such as SVM and sub-buffers.

On-device memory, SVM, subbuffers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a buffer is created by clCreateBuffer, by default nothing is allocated
in device memory. The actual allocation of on-device memory happens when
the buffer is first used, by enqueueing a command. If the device memory
doesn't have enough free space, the enqueue fails.

In OpenCL, buffers are "per context" not "per device". Buffer can only
have one valid content, even though that content might be present on
multiple devices. However, if two commands write to the same buffer locations
(e.g. on two devices), without properly synchronizing, that is undefined
behaviour. There can be multiple readers, but only one writer. PoCL
keeps track of buffer content validity across devices by versioning.
The buffer (cl_mem) has a "latest content version" number;
plus each buffer instance on every device has its version number.

The version is updated in libpocl, at enqueue time, in
pocl_create_migration_commands() called by pocl_create_command().

The drivers do not need to care about the version (except setting
it to 0 when they first allocate on-device memory for a buffer).
The migration code in pocl_create_migration_commands() updates
and uses the versions to decide what needs to be migrated,
then it creates & enqueues migration events (with dependencies),
and the drivers only see those events. IOW decisions are made at
enqueue time (and should be fast), and execution then proceeds
(in parallel if event dependencies allow) at any time later.

Buffer content is automatically migrated between devices by PoCL as needed
(when the on-device buffer's content version is older than latest version).
PoCL will try to find the best possible way to migrate buffer content, but
at worst it will fallback to double copying, source-device to host-memory
followed by host-memory to destination device. Buffer migration commands
are inserted before the enqued command, linked by event dependencies, and
put on special hidden queues (this is because they can execute in parallel
with other commands, as long as their event dependencies allow it).

SVM memory avoids the code paths that deal with cl_mem and migrations.
From PoCL's memory management POV, SVM can be divided in two types:
coarse-grained and fine-grained. The former only shares the Virtual AS,
and needs to be mapped/unmapped for host access, while the latter shares
also physical memory. PoCL currently only supports fine-grained SVM; the
memory architecture allows also coarse-grained, but some special cases
like ``clCreateBuffer(CL_USE_HOST_PTR, svm-pointer)`` are currently
broken with coarse-grained and need to be fixed. The reason is that
the the svm pointer given to clCreateBuffer() will be used to set
the cl_mem->mem_host_ptr, and the code assumes that mem_host_ptr is
a memory which physically exists on the host (in case it's needed for
temporary stuff like buffer migration). This breaks with coarse-grained
SVM because it is just a virtual AS pointer, not backed by physical memory,
and needs to be svm-mapped before usage. The fix (probably) would be to
detect svm pointers handed to clCreateBuffer, and if it's coarse-grained,
figure out what is the actual on-device memory pointer, and use it to fill
`mem->devices_ptrs[svm-device-index]` with proper information,
not as mem_host_ptr.

Subbuffers are currently implemented in a way that inside the clEnqueue
API calls they are translated into a (parent buffer, offset) pair, so in
the internal driver API, the drivers only ever see buffers. This was done
for multiple reasons - the Specification is a bit unclear on subbuffers
(when should they be synchronized with the parent buffer), and the
feature is not important enough ATM to further complicate the driver
code.


Bufalloc
^^^^^^^^^^

Device drivers in PoCL need to manage the global memory of devices, by
allocating and freeing chunks of it for the OpenCL buffers. In some cases
(like CPU driver), this is simple because the memory management can be
delegated to an existing solution (like malloc). In other cases, the
driver only has access to a region of continuous memory, and it needs
to implement its own solution for memory management.

For this, pocl implements a simple memory allocator called ``bufalloc``.
With bufalloc it is possible to manage chunks of memory allocated from a region of addresses.
The allocator is optimized for speed and to minimize fragmentation assuming largish chunks of
memory (the input/output buffers) are allocated and freed at once.

Bufalloc can be used for host-side management of continuous ranges of memories on the
device side. Bufalloc can optionally be used to manage memory also in the ``pthread/basic``
CPU device implementations for testing and optimization purposes.

For an example of its use for managing memory in the heterogeneous separated memory setup,
one should take a look at the TCE device layer code (``lib/CL/devices/tce/tce_common.cc``).
For TCE devices it is assumed there are actual separated physical address spaces for both
the *local* and *global* address spaces. The device layer implementation manages allocations
from both of these spaces using two instances of bufalloc memory regions.

When passing buffer pointers to the kernel/work-group launchers, the memory addresses are
passed as integer values. The values passed from the host are casted to the actual
address-space qualified LLVM IR pointers for calling the kernels with correct types
by the work-group function (see :ref:`wg-functions`).

Bufalloc for CPU device is enabled by CMake option USE_POCL_MEMMANAGER. This is only
useful for certain uncommon setups, where pocl is expected to allocate a huge number
of queue or event objects. For most available OpenCL programs / tests / benchmarks,
there is no measurable difference in speed.

Advantages:
* allocation of queues/events/command objects can be a lot faster

Disadvantages:
* memory allocated for those objects is never free()d; it's only returned to allocation pool
* debugging tools will not detect use-after-free bugs on said objects
