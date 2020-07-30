Debugging OpenCL applications with PoCL
========================================


There are several ways to debug applications with PoCL,
differing in debugging coverage and impact on speed.

This document assumes debugging OpenCL kernel code by using
the CPU pthread driver, since that is currently
the most mature driver in PoCL.

"Offline" debugging
--------------------

Is done by setting ``POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES`` env var to 1.

If set to 1, the intermediate output files from the kernel
compilation process are left in PoCL's cache. Be default
they are deleted and only the final output is left in the cache.

This is useful for manually inspecting compilation stages,
but it's also useful for GDB and Valgrind debugging (see below).

Simple debugging with PoCL's debug messages
--------------------------------------------

Upsides:
  * doesn't require recompiling the application or PoCL

Downsides:
  * very limited scope

Setup:

The only thing required is to set "POCL_DEBUG" environment variable
to some value. The most useful values are:

 * ``POCL_DEBUG=err,warn`` - this will limit the output to errors and
   warnings. This might help spot some OpenCL API calls which return
   an error value. Also helps if a call can return CL_INVALID_VALUE for
   multiple reasons, since PoCL prints the reason.

 * ``POCL_DEBUG=refcount`` - this will limit the output to refcount increases
   and decreases. Might help spot some memory leaks

Example::

     gcc example.c -o example -lOpenCL
     export POCL_DEBUG=refcount
     ./example

Output::

    [2020-07-20 12:37:18.472185807]POCL: in fn POclReleaseContext at line 48:
      | REFCOUNTS |  Release Context
    [2020-07-20 12:37:18.472196073]POCL: in fn POclReleaseContext at line 56:
      | REFCOUNTS |  Free Context 0x5566430d84a0
    [2020-07-20 12:37:18.472207597]POCL: in fn POclReleaseCommandQueue at line 41:
      | REFCOUNTS |  Release Command Queue 0x5566430d85f0  0
    [2020-07-20 12:37:18.472228759]POCL: in fn POclReleaseCommandQueue at line 55:
      | REFCOUNTS |  Free Command Queue 0x5566430d85f0

"Release X" is printed when the refcount is lowered by 1;
"Free X" is printed when the refcount becomes 0 and the object is actually freed.


Debugging with Thread/Address sanitizers
-----------------------------------------------

Currently PoCL recognizes four sanitizers:
Address, Leak, Undefined behaviour and Thread.

Corresponding PoCL CMake options to enable them are:
``ENABLE_ASAN, ENABLE_LSAN, ENABLE_UBSAN, ENABLE_TSAN.``

See also "handling LLVM" below.

Upsides:
  * much faster than Valgrind
  * minimal false positives
  * can check undefined behaviour (most other tools can't)

Downsides:
  * requires rebuilding both the application and PoCL;
  * the application and PoCL's runtime code are compiled with sanitizer,
    but at the moment, the kernels are not compiled with sanitizer

Setup:
  * for e.g. Address sanitizer, build PoCL with these flags::

       -DENABLE_ASAN=1 -DENABLE_ICD=0 -DCMAKE_BUILD_TYPE=Debug

  * this will result in ``lib/CL/libOpenCL.so``; rebuild your application
    with the correct ``-fsanitize=X`` flag and link it to ``lib/CL/libOpenCL.so``

Example:

  building an "example.c" with Address sanitizer::

        gcc -O0 -ggdb -fsanitize=address -fno-omit-frame-pointer -pthread -o example.o -c example.c
        gcc -fsanitize=address -o example example.o -lasan -Wl,-rpath,<pocl-build-dir>/lib/CL <pocl-build-dir>/lib/CL/libOpenCL.so

Output:

  if there's an OpenCL object remaining, ASan will print a backtrace with an OpenCL call name in it::

      Indirect leak of 8 byte(s) in 1 object(s) allocated from:
        #0 0x7fa8f7b0a198 in calloc (/usr/lib/x86_64-linux-gnu/libasan.so.5+0xee198)
        #1 0x7fa8f7607bc0 in pocl_unique_device_list /tmp/lib/CL/pocl_util.c:866
        #2 0x7fa8f75d37ca in POclCreateContext /tmp/lib/CL/clCreateContext.c:172
        #3 0x55d50f21e428 in poclu_get_any_device2 /tmp/lib/poclu/misc.c:84
        #4 0x55d50f21c165 in main /tmp/examples/example1/example1.c:59
        #5 0x7fa8f707bb96 in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x21b96)

  if there's any memory leak in the user's program, AddrSanitizer will print something like::

      Direct leak of 64 byte(s) in 1 object(s) allocated from:
        #0 0x7f738e999f90 in __interceptor_malloc (/usr/lib/x86_64-linux-gnu/libasan.so.5+0xedf90)
        #1 0x562f6f33e493 in main /tmp/examples/example1/example1.c:74
        #2 0x7f738df0bb96 in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x21b96)



Debugging with Valgrind
-----------------------------------------------

Upsides:
  * the entire application including kernels can be debugged
  * does not strictly require recompilation (though for usable
    backtraces, requires debuginfo)

Downsides:
  * can be very slow, especially with computationally intensive kernels
  * may report some false positives

Setup:
  * build PoCL with ``-DCMAKE_BUILD_TYPE=Debug``
  * ``export POCL_EXTRA_BUILD_FLAGS="-g -cl-opt-disable"``,
    or add these flags to the ``clBuildProgram`` call.
    (this will cause all kernels to compile with debuginfo)
  * run your application with valgrind

See also "handling LLVM" below.

Example 1:

Uninitializing both LLVM (calling clUnloadPlatformCompiler) and drivers::

      POCL_ENABLE_UNINIT=1 valgrind ./examples/example1/example1

Output 1::

    ==18291== LEAK SUMMARY:
    ==18291==    definitely lost: 40 bytes in 1 blocks
    ==18291==    indirectly lost: 0 bytes in 0 blocks
    ==18291==      possibly lost: 0 bytes in 0 blocks
    ==18291==    still reachable: 545,683 bytes in 2,705 blocks
    ==18291==         suppressed: 0 bytes in 0 blocks
    ==18291== Rerun with --leak-check=full to see details of leaked memory

Example 2:

Uninitializing LLVM (calling clUnloadPlatformCompiler) but not drivers::

     valgrind ./examples/example1/example1

Output 2::

    ==18301== LEAK SUMMARY:
    ==18301==    definitely lost: 0 bytes in 0 blocks
    ==18301==    indirectly lost: 0 bytes in 0 blocks
    ==18301==      possibly lost: 2,816 bytes in 8 blocks
    ==18301==    still reachable: 403,199,350 bytes in 2,720 blocks
    ==18301==         suppressed: 0 bytes in 0 blocks
    ==18301== Rerun with --leak-check=full to see details of leaked memory

Example 3:

Both LLVM and drivers left (not calling clUnloadPlatformCompiler)::

     valgrind ./examples/example1/example1

Output 3::

    ==18726== LEAK SUMMARY:
    ==18726==    definitely lost: 536 bytes in 2 blocks
    ==18726==    indirectly lost: 1,299,332 bytes in 3,433 blocks
    ==18726==      possibly lost: 53,773,316 bytes in 524,329 blocks
    ==18726==    still reachable: 411,350,622 bytes in 73,488 blocks
    ==18726==         suppressed: 0 bytes in 0 blocks


Debugging with GDB
-----------------------------------------------

Upsides:
  * very fast
  * the entire application including kernels can be debugged
  * does not strictly require recompilation (but it is highly recommended if PoCL wasn't compiled with debuginfo)
  * stepping inside kernels posible (but a little tricky)

Downsides:
  * limited scope (not the best tool for memory leaks & race conditions)

Setup:
  * build PoCL with ``-DCMAKE_BUILD_TYPE=Debug``
  * ``export POCL_EXTRA_BUILD_FLAGS="-g -cl-opt-disable"``,
    or add these flags to the ``clBuildProgram`` call.
    (this will cause all kernels to compile with debuginfo)
  * ``export POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES=1``
    (this will leave the source files in PoCL's cache)
  * (optional) ``export POCL_MAX_PTHREAD_COUNT=1``
    (this limits the pthread driver to a single worker thread)
  * run your application with gdb

Example 1:

Let's say we have an `example` host program with a `dot_product` kernel with this source::

    __kernel void dot_product (__global const float4 *a,
                               __global const float4 *b,
                               __global float4 *c)
    {
      size_t gid = get_global_id(0);

      gid += 18298392UL;
      c[gid] = a[gid] * b[gid] + (float4)(1.0f, 6.0f, 9.0f, 4.0f);
    }

Run it in gdb::

    POCL_DEBUG=all gdb ./example

Output 1:

The program crashes since it tries to access memory beyond buffer boundaries::

    [2020-06-30 08:28:14.888355355]POCL: in fn pocl_check_kernel_disk_cache at line 963:
      |   GENERAL |  Built a WG function: /tmp/POCL_CACHE/BJ/JMEICBEBICMMDJCKNIADBFKHIMHDBIIKHCHED/dot_product/2-1-1-goffs0-smallgrid/dot_product.so

    Thread 8 "example" received signal SIGSEGV, Segmentation fault.
    [Switching to Thread 0x7fffddffe700 (LWP 10585)]
    0x00007fffec532458 in dot_product (a=0x5555557bb580, b=0x5555557e6500, c=0x5555557ba480) at /tmp/POCL_CACHE/tempfile-1c-aa-cd-3e-5e.cl:10
    10    c[gid] = a[gid] * b[gid] + (float4)(1.0f, 6.0f, 9.0f, 4.0f);
    (gdb) list
    5            __global const float4 *b, __global float4 *c)
    6   {
    7     size_t gid = get_global_id(0);
    8
    9     gid += 18298392UL;
    10    c[gid] = a[gid] * b[gid] + (float4)(1.0f, 6.0f, 9.0f, 4.0f);
    11  }
    (gdb) print gid
    $1 = 18298392
    (gdb) bt
    #0  0x00007fffec532458 in dot_product (a=0x5555557bb580, b=0x5555557e6500, c=0x5555557ba480) at /tmp/POCL_CACHE/tempfile-1c-aa-cd-3e-5e.cl:10
    #1  0x00007fffec5324c3 in _pocl_kernel_dot_product_workgroup ()
       from /tmp/POCL_CACHE/BJ/JMEICBEBICMMDJCKNIADBFKHIMHDBIIKHCHED/dot_product/2-1-1-goffs0-smallgrid/dot_product.so
    #2  0x00007ffff72924ed in work_group_scheduler (k=0x7fffb91935c0, thread_data=0x5555557ae600)
        at /tmp/pocl_source/lib/CL/devices/pthread/pthread_scheduler.c:307
    #3  0x00007ffff7292b72 in pthread_scheduler_get_work (td=0x5555557ae600) at /tmp/pocl_source/lib/CL/devices/pthread/pthread_scheduler.c:454
    #4  0x00007ffff7292fd2 in pocl_pthread_driver_thread (p=0x5555557ae600) at /tmp/pocl_source/lib/CL/devices/pthread/pthread_scheduler.c:530
    #5  0x00007fffee90e6db in start_thread (arg=0x7fffddffe700) at pthread_create.c:463
    #6  0x00007ffff78faa3f in clone () at ../sysdeps/unix/sysv/linux/x86_64/clone.S:95

Example 2:

Lets say we want to step the kernel from previous example. Launch gdb::

    POCL_MAX_PTHREAD_COUNT=1 gdb ./example

We can't set a breakpoint inside a kernel before running the program,
because PoCL writes the source it receives at runtime
(by ``clCreateProgramWithSource``) to a temporary file.

First we need to find some place in the pthread driver sources
where the kernel is fully compiled but not yet executed.
A good place is where the pthread driver launches the
workgroup function. In the pthread driver, this is a call to
``(struct kernel_run_command)->workgroup``, looking like this::

	  k->workgroup ((uint8_t*)arguments, (uint8_t*)&pc,
		gids[0], gids[1], gids[2]);

Lets say it's at ``lib/CL/devices/pthread/pthread_scheduler.c:307``, set the breakpoint::

    (gdb) break /tmp/pocl_source/lib/CL/devices/pthread/pthread_scheduler.c:307
	No source file named /tmp/pocl_source/lib/CL/devices/pthread/pthread_scheduler.c.
	Make breakpoint pending on future shared library load? (y or [n]) y
	Breakpoint 1 (/tmp/pocl_source/lib/CL/devices/pthread/pthread_scheduler.c:307) pending.

Run the program::

	(gdb) r
	Starting program: /tmp/pocl_build/example
	[Thread debugging using libthread_db enabled]
	Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
	[New Thread 0x7fffedf36700 (LWP 11851)]
	[Switching to Thread 0x7fffedf36700 (LWP 11851)]

	Thread 2 "example" hit Breakpoint 1, work_group_scheduler (k=0x7fffe9001640, thread_data=0x5555557ad500)
		at /tmp/pocl_source/lib/CL/devices/pthread/pthread_scheduler.c:307
	307	          k->workgroup ((uint8_t*)arguments, (uint8_t*)&pc,


Now it's about to launch the ``dot_product`` kernel for a single workgroup, so let's find the kernel source file::

	(gdb) info functions dot_product
	All functions matching regular expression "dot_product":

	File /tmp/POCL_CACHE/tempfile-1c-aa-cd-3e-5e.cl:
	void dot_product(const float4 *, const float4 *, float4 *);

	Non-debugging symbols:
	0x00007fffed534300  _pocl_kernel_dot_product@plt
	0x00007fffed5344a0  _pocl_kernel_dot_product_workgroup
	0x00007fffed5344d0  _pocl_kernel_dot_product_workgroup_fast

The kernel is in ``/tmp/POCL_CACHE/tempfile-1c-aa-cd-3e-5e.cl``.

Let's set a breakpoint on line 9 where gid is modified::

	(gdb) break /tmp/POCL_CACHE/tempfile-1c-aa-cd-3e-5e.cl:9
	Breakpoint 2 at 0x7fffed534395: file /tmp/POCL_CACHE/tempfile-1c-aa-cd-3e-5e.cl, line 9.

Continue the program::

	(gdb) c
	Continuing.
	Thread 2 "example" hit Breakpoint 2, dot_product (a=0x55555575b300, b=0x5555557be480, c=0x5555557df080) at /tmp/POCL_CACHE/tempfile-1c-aa-cd-3e-5e.cl:9
	9	  gid += 18298392UL;

We can now step through the kernel::

	(gdb) print gid
	$1 = 0
	(gdb) s
	10	  c[gid] = a[gid] * b[gid] + (float4)(1.0f, 6.0f, 9.0f, 4.0f);
	(gdb) print gid
	$2 = 18298392


Handling LLVM and driver-allocated memory
-----------------------------------------------

Both valgrind and sanitizers might report a huge amount of memory leaks
coming from PoCL; this is caused mainly by two factors,
LLVM and driver-held static data.

The OpenCL API unfortunately doesn't provide any API entry to uninitialize
the entire implementation (e.g. all driver data). It does provide API
entries to unload compiler: ``clUnloadPlatformCompiler()`` and ``clUnloadCompiler()``.

User can use these to ask PoCL to unload all LLVM data; note that with
PoCL, this only works if all cl_programs and cl_kernels have been released.

Usage is simple: call ``clUnloadPlatformCompiler()`` once, after
all other opencl objects have been released, right before program exit.

If the user sets ``POCL_ENABLE_UNINIT`` env var to 1, PoCL will also try to
unload driver data. This feature might not work reliably so it's
not official yet.

Example: running a program compiled with AddrSanitizer, which calls
``clUnloadPlatformCompiler()``, with ``POCL_DEBUG=all POCL_ENABLE_UNINIT=1``
env variables will result in (if the program has no memleaks)::


    [2020-06-20 15:25:01.722343448]POCL: in fn POclReleaseContext at line 50:
      | REFCOUNTS |  Free Context 0x60f000000310

    [2020-06-20 15:25:01.722369150]POCL: in fn void pocl_llvm_release() at line 370:
      |      LLVM |  releasing LLVM

    [2020-06-20 15:25:01.823218919]POCL: in fn pocl_check_uninit_devices at line 107:
      | REFCOUNTS |  Zero contexts left, calling pocl_uninit_devices

    [2020-06-20 15:25:01.823266761]POCL: in fn pocl_uninit_devices at line 334:
      |   GENERAL |  UNINIT all devices

Running the same program with empty PoCL cache and removed
``clUnloadPlatformCompiler()`` call (therefore with LLVM context
alive at program exit), ASan will print a lot of memory leaks::

    Indirect leak of 8 byte(s) in 1 object(s) allocated from:
        #0 0x7f99eef43ba0 in operator new(unsigned long) (/usr/lib/x86_64-linux-gnu/libasan.so.5+0xefba0)
        #1 0x7f99eead5aea in WorkItemAliasAnalysis::runOnFunction(llvm::Function&) /tmp/lib/llvmopencl/WorkItemAliasAnalysis.cc:130
        #2 0x7f99e6f76ed5 in llvm::FPPassManager::runOnFunction(llvm::Function&) (/usr/lib/llvm-10/lib/libLLVM-10.so.1+0xb11ed5)

    SUMMARY: AddressSanitizer: 1047772 byte(s) leaked in 3046 allocation(s).
