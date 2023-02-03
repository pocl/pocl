=======================================================
How to write a new user-space device driver for PoCL
=======================================================

The first required steps are adding the CMake infrastructure, and
making the driver known to device initialization. Required CMake
changes are: a boolean option to enable the driver (optionally
autodetected at CMake time), adding the bool to config.h.in.cmake,
finding all required libraries, linking & compilation options,
and correctly linking the driver depending on ``ENABLE_LOADABLE_DRIVERS``
(these can be copied from existing drivers). To make the driver known
to device initialization in devices.c, add necessary #includes, and
the new driver name to ``pocl_device_types`` and ``pocl_devices_init_ops``.

Setting device properties
-----------------------------------

In the driver's init callback, the driver should discover and set up each device's properties.
These are stored as members of ``struct _cl_device_id``, e.g. ``max_work_item_dimensions`` or
``max_compute_units``. Some of them are internal to PoCL, e.g. ``device_side_printf`` is a bool
that changes the processing of the 'printf' function in PoCL's LLVM passes. There are some helpers
for setting up properties; for setting up versioned (OpenCL 3.0) properties, see ``lib/CL/devices/common.c``;
the same file also has ``pocl_init_default_device_infos`` which sets up some reasonable defaults.
By default the entire struct is memset to zero, so all properties are zero / false.


PoCL's internal driver API
-----------------------------------

Next step is to implement PoCL's internal driver API operations (callbacks) for the device.
They're described in lib/CL/pocl_cl.h, in ``struct pocl_device_ops``. The declarations for all
operations can be obtained by including ``prototypes.inc`` and using the ``GEN_PROTOTYPES`` macro.
The declarations will be named ``pocl_<DRIVER_NAME>_<OP_NAME>``, e.g. ``pocl_level0_map_mem``.
Many of these callbacks are optional; there's no need to implement all to get a useful driver.
Any callback that is implemented, needs to be assigned in the ``ops`` struct of the
``struct _cl_device_id`` in the ``pocl_<DRIVER>_init_device_ops`` function. For an example, see
``pocl_basic_init_device_ops`` in ``lib/CL/devices/basic/basic.c``. If the callback is optional
and will not be implemented, it can be left as NULL in ``ops``.

The callbacks can be divided into several groups:

* probe & initialization (required)
* memory mapping, allocation & freeing (required)
* event handling (required)
* compilation callbacks (required)
* command execution callbacks (optional)
* SVM callbacks (optional)
* Image callbacks (optional)
* Object create / free callbacks (optional)
* D2D migration callbacks (optional)

1. probe & initialization - ``probe, init, reinit, uninit``

   you'll need to implement at least ``probe`` and ``init``; ``probe`` is only called once, when
   libpocl is loaded, and its job is to find out how many devices are available.
   ``init`` is then called for each device in succession. Implementing ``reinit`` & ``uninit``
   is optional - these release any resources after the last context is released.
   This is useful mainly for debugging memory leaks; since there is no ``clReleasePlatform``
   in OpenCL, calling uninit should release everything - anything left over is a leak.

2. memory allocation - ``alloc_mem_obj, free``

   these allocate & free device-side memory. See documentation on ``struct _cl_mem->device_ptrs``
   and ``pocl_mem_identifier``. If the device supports images, these must also allocate/free images.
   Buffers (cl_mem objects) in OpenCL are per-context; the allocation of the backing storage
   (device memory) in PoCL is done lazily. IOW a buffer is allocated only on devices which actually run
   commands using those buffers. This happens at clEnqueueXYZ time (``clEnqueue -> pocl_create_command()
    -> can_run_command() -> device->ops->alloc_mem_obj()``).

3. memory mapping - ``get_mapping_ptr, free_mapping_ptr``

   these are required to support clEnqueueMapBuffer; they don't execute the actual mapping,
   only return the host pointer where the mapping will exist.

4. event handling - ``submit, join, flush, notify, broadcast, wait_event, update_event``

   these are required to properly react to and propagate event status changes. With a few exceptions
   (basic, cuda, almaif), the implementations for these are almost identical. Implementing custom
   handling is possible, but must be done extremely carefully, as it can easily lead to issues
   with locking and data races.

5. compilation callbacks - ``build_source, build_binary, link_program`` etc

   the implementation of these depends on what the device intends to support.

   a. if the device will only support offline compilation, this requires only a minimal
      implementation of ``build_binary``; PoCL will automatically unpack pocl-binaries
      into the cache directory and setup kernel metadata.

   b. if the device wants to support online (OpenCL C) compilation using LLVM infrastructure,
      it can use the generic ``pocl_driver_XYZ`` callback implementations from ``common_driver.c``.
      These will invoke Clang/LLVM compilation, optimization & linking steps.
      However, the device needs to setup its metadata like LLVM triple, whether it's SPMD,
      and many others. It will also need to provide its builtin library. The existing
      implementations for per-device builtin libraries are in lib/kernel; there are many
      builtins that are device-agnostic, however some are device-specific.

   c. if the device wants to support online (source) compilation but is NOT using LLVM,
      it will have to provide its own implementation of all compilation callbacks and
      kernel metadata extraction.

   ``build_source`` - required; but only if driver wants to support compilation from source.
   Note that builds from SPIR-V input are (currently) handled by the ``build_binary`` callback.

   ``build_binary`` - required; all drivers (except custom devices) are assumed to support
   clCreateProgramWithBinary() with at least pocl-binaries as input. If the files stored
   in pocl-binaries have everything required, then the implementation of this callback can be
   trivial; otherwise it needs to do the transformation of binaries to final device binaries.

   ``link_program`` - is required to support the ``clCompile/clLinkProgram`` combo; it's not
   *strictly* required for a useful driver, since most OpenCL programs don't use these.

   ``supports_binary`` - optional; is called to check whether the driver supports a specific binary.
   Note that pocl-binaries are always supported, so this is only called for other binaries.

   ``setup_metadata`` - required; after build, it's called to set up the
   program & kernel metadata like number of arguments, argument types and so on.

   ``post_build_program`` - optional; it's called as part of clBuild/Link/Compile
   *after* everything else has been set up (including metadata).

   ``build_poclbinary`` & ``compile_kernel`` - optional; clGetProgramInfo() called
   with CL_PROGRAM_BINARIES will call these, if they're not NULL; the purpose is
   to do any extra steps necessary to have the program cache directory in a "useful"
   state, when the cache directory can be serialized into a pocl-binary.

6. command execution callbacks - ``read, copy, write, map_mem`` etc

   These are optional because command execution can be implemented in multiple ways.

   a. there is a helper function in PoCL for executing commands in the driver, called ``pocl_exec_command``.
      This helper does some preparations, then calls the driver's callback for the command (e.g.
      ``device->ops->unmap`` for the EnqueUnmap type command), and then cleanups after the command.
      The advantage is that this is the simplest way to implement a command; the disadvantage is
      that ``pocl_exec_command`` is synchronous and it does not do any optimization
      by grouping commands.

   b. the other way to implement command execution is to not use ``pocl_exec_command`` and
      device->ops command callbacks, in which case you can leave those NULL and implement
      commands your way. The PoCL library is driven by events, and how the driver executes
      the commands of events, is not important to the runtime, as long as events are correctly
      moved through their stages (submitted->queued->running->complete) and all of the
      "bookkeeping" (e.g. event callbacks) is handled properly.

   A simple driver implementation using ``pocl_exec_command()`` could look like this:
     * implement the command execution callbacks (device->ops->read etc)
     * create a background thread in ``pocl_DRIVER_init`` and a simple FIFO queue;
     * when a new event arrives with a command to execute, e.g. through ``pocl_DRIVER_notify``
       or ``pocl_DRIVER_submit``, check if the event is ready to execute, if it is,
       push into FIFO queue;
     * in the background thread, create a loop that waits for commands to arrive in the FIFO queue,
       then for each command, call ``pocl_exec_command`` - this will take care of calling
       the correct device->ops command callback, and various bookkeeping

7. SVM callbacks - ``svm_free, svm_alloc, svm_map, svm_unmap`` etc

   only required if the device supports SVM. To support SVM, driver will also need to set up
   some properties in ``struct _cl_device_id``, at least ``svm_allocation_priority``, ``svm_caps``,
   and ``atomic_memory_capabilities`` + ``atomic_fence_capabilities``.

8. image support - ``copy_image_rect, write_image_rect, map_image`` etc

   only required if device supports images. To support images, driver will also need to set up
   some properties in ``struct _cl_device_id``, at least ``image_support``, ``num_image_formats``
   and ``image_formats`` but many others - search for ``image`` in the struct,
   also look at other driver's ``ops->init``.

9. create / free callbacks - ``free_event_data, create_kernel, init_queue, create_sampler`` etc

   all of these are optional. Only necessary if the driver needs some to set up / tear down
   some device-specific (hardware or "backend" API) resources for a cl_object. E.g. the Level0
   driver uses ``free_kernel`` to release the API's ``ze_kernel_handle_t`` handle. The "free"
   callbacks are called only after the refcount on the object reached zero, so it is safe
   to destroy the resource. The "create" callbacks are called after the OpenCL part of
   the cl_object has been set up.

10. device2device migration callbacks - ``can_migrate_d2d, migrate_d2d``

   optional; they're used to implement direct migration of buffers between
   two devices. Direct means avoiding copying the buffer content to
   host memory and then from host memory to the 2nd device.
