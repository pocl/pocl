===================================
OpenCL Extensions Supported by PoCL
===================================

PoCL supports a number of OpenCL extensions. The exact
list of extensions depends on the driver backend in use
as well as the exact options PoCL was built with.
Applications should always query available extensions
before attempting to use their functionality.

Full extension specifications can be found on:

https://www.khronos.org/registry/OpenCL/

Some highlights from the list of supported extensions:

cl_pocl_content_size
~~~~~~~~~~~~~~~~~~~~~~~

This extension provides a way to to indicate
a buffer which will hold the meaningful
bytes of another buffer, after kernel execution.

This allows the implementation to reduce the amount
of data copied when moving buffers between devices
e.g. when the data is compressed and its exact
length is not known ahead of time.


cl_khr_command_buffer
~~~~~~~~~~~~~~~~~~~~~~~

This extension provides a way to record a sequence
of OpenCL commands that can be executed as a single
invocation. Command parameters are validated and
commands are prepared at command buffer recording
time, reducing the overhead of dispatching the sequence
and allowing drivers to optimize the scheduling of
commands within a buffer.

cl_intel_exec_by_local_thread
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supported by the CPU (pthread) driver on x86-64 and AArch64, except on
Windows and arm64e, and not with the OpenMP scheduler. Every command enqueued on a command queue created with the
`CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL` property is executed by the thread
that enqueues it, before the enqueue call returns; its event is then
complete. This removes the cost of handing small kernels to a worker thread
and waiting for them. The enqueuing thread waits for the dependencies of the
command, if there are any, which includes the previous command of an in-order
queue that other threads enqueue to as well. A native kernel that enqueues on
its own in-order queue therefore waits for itself forever. Blocking and
non-blocking calls behave the same.

Kernels run on a stack PoCL allocates, as large as the stack of the worker
threads, with default rounding and masked floating-point exceptions. The
application thread's floating-point environment is saved before and restored
after (on x86-64, except for the x87 exception flags). All work-groups run on the enqueuing thread, also on
subdevices. An NDRange enqueue fails with `CL_OUT_OF_HOST_MEMORY`, without
having done anything, when the memory for running it on the calling thread
can't be allocated. Command buffers can't be used with such queues.

cl_ext_defined_builtin_kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The purpose of this extension is to provide a standardized
set of built-in kernels with well-defined semantics.
See :ref:`defined-built-in-kernels` for more details on
DBKs and PoCL's implementation of them.
