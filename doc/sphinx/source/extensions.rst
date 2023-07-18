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
