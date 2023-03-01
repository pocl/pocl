Debugging OpenCL applications with PoCL
========================================


There are several ways to debug applications with PoCL,
differing in debugging coverage and impact on speed.

This document chapter describes means for debugging OpenCL kernel code by using
the CPU drivers of PoCL.

Basic printf debugging
----------------------

The CPU drivers of PoCL flush the OpenCL 1.2 printf() API output immediately
at the end of the printf call. This is in contrast to some other drivers which
flush the output only at the end of the kernel command's execution, making
debugging crashing (segfaulting) kernels difficult since they never finish
the command, thus any debug printouts won't get printed out.

Kernel compiler debugging
-------------------------

Inspecting the kernel compiler intermediate results can be done by
setting ``POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES`` env var to 1.
This causes the intermediate output files from the kernel
compilation process to be left in PoCL's disk cache for inspection.
By default these files are deleted, and only the final executable output is left
in the cache.

This is useful for manually inspecting the LLVM IR of the compilation stages,
but it's also useful for GDB and Valgrind debugging as described
later.

Simple debugging with PoCL's debug log
--------------------------------------------

Upsides:
  * doesn't require recompiling the application or PoCL

Downsides:
  * very limited scope

Setup:

Just set the "POCL_DEBUG" environment variable to some value.
The most useful values are:

 * ``POCL_DEBUG=err,warn`` - this will limit the output to errors and
   warnings. These messages might help spot some OpenCL API calls which return
   an error value. Also it helps if a call can return CL_INVALID_VALUE for
   multiple reasons, since PoCL prints a more specific reason in that case.

 * ``POCL_DEBUG=refcount`` - this will limit the output to refcount increases
   and decreases. Might help spot CL object leaks

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

"Release X" is printed when the refcount is lowered by 1.
"Free X" is printed when the refcount becomes 0 and the object is actually freed.

Debugging with GDB
-----------------------------------------------

Upsides:
  * the entire OpenCL application, including the launched kernels can be debugged
  * does not require PoCL recompilation (but it is recommended, if PoCL wasn't compiled with debuginfo)
  * single stepping kernels

Downsides:
  * limited scope (not the best tool for tracking memory leaks & race conditions)

Setup:
  * Optional: build PoCL with ``-DCMAKE_BUILD_TYPE=Debug``
  * ``export POCL_EXTRA_BUILD_FLAGS="-g -cl-opt-disable"``,
    or add these flags to the ``clBuildProgram`` call.
    This will cause all kernels to compile with debuginfo.
  * ``export POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES=1``
    This will leave the source files in PoCL's cache.
  * Optional: ``export POCL_MAX_PTHREAD_COUNT=1``
    This limits the pthread driver to a single worker thread.
  * Run your application with gdb, as usual.

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

Note: printing variables (e.g. gid) could instead result in this:

    (gdb) print gid
    $1 = {{{18298392, 9223372036854775822, 0, 0}}}

This happens when PoCL uses the "loops" workgroup method. The high-level overview of "loops"
is that PoCL it creates a 3D for-loop (for each dimension of workgroup-size) around the kernel
code, and the LLVM optimizer then tries to vectorize that loop. For this to work, PoCL must
create a copy of variables in private address space, one copy for each workitem in the
workgroup; that's why the variable printed is an array.

Example 2:

Lets say we want to step the "dot_product" kernel from the previous example. Launch gdb::

    POCL_MAX_PTHREAD_COUNT=1 gdb ./example

Make a breakpoint on the kernel name::

	(gdb) break dot_product
	Function "dot_product" not defined.
	Make breakpoint pending on future shared library load? (y or [n]) y
	Breakpoint 1 (dot_product) pending.

Run the program::

	(gdb) r
	Starting program: /tmp/example
	[Thread debugging using libthread_db enabled]
	Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
	[New Thread 0x7fffedf36700 (LWP 18595)]
	[Switching to Thread 0x7fffedf36700 (LWP 18595)]

	Thread 2 "example" hit Breakpoint 1, dot_product (a=0x5555557bc080, b=0x5555557e5380, c=0x5555557baf00) at /tmp/POCL_CACHE/tempfile-db-70-03-45-d6.cl:7
	7	  size_t gid = get_global_id(0);

We can now step through the kernel::

	(gdb) print gid
	$1 = 140737103657472
	(gdb) next
	9	  gid += 18298392UL;
	(gdb) print gid
	$2 = 0
	(gdb) next
	10	  c[gid] = a[gid] * b[gid] + (float4)(1.0f, 6.0f, 9.0f, 4.0f);
	(gdb) print gid
	$3 = 18298392


Debugging with Valgrind
-----------------------------------------------

Upsides:
  * The entire application including kernels can be debugged.
  * Does not strictly require recompilation (though for usable
    backtraces, requires debuginfo).

Downsides:
  * Can be very slow, especially with computationally intensive kernels.
  * May report some leaks which are not ones (see below).

Setup:
  * Optional: build PoCL with ``-DENABLE_VALGRIND=ON -DCMAKE_BUILD_TYPE=Debug``
  * ``export POCL_EXTRA_BUILD_FLAGS="-g -cl-opt-disable"``,
    or add these flags to the ``clBuildProgram`` call.
    This will cause all kernels to compile with debuginfo.
  * Run your application with valgrind as normally.

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

Debugging with Thread/Address sanitizers
-----------------------------------------------

Currently PoCL recognizes four sanitizers:
Address, Leak, Undefined behaviour and Thread.

Corresponding PoCL CMake options to enable them are:
``ENABLE_ASAN, ENABLE_LSAN, ENABLE_UBSAN, ENABLE_TSAN.``

Upsides:
  * Much faster than Valgrind.
  * Less false detections.
  * Can check undefined behaviour (most other tools can't).

Downsides:
  * Requires rebuilding both the application and PoCL.
  * The application and PoCL's runtime code are compiled with sanitizer,
    but at the moment, the kernels cannot be compiled with the sanitizer.

Setup:
  * For example, to use the Address Sanitizer (ASan), build PoCL with these flags::

       -DENABLE_ASAN=1 -DENABLE_ICD=0 -DCMAKE_BUILD_TYPE=Debug

  * This will result in ``lib/CL/libOpenCL.so``. Rebuild your application
    with the correct ``-fsanitize=X`` flag and link it to ``lib/CL/libOpenCL.so``.

Example:

  Building an "example.c" with the ASan::

        gcc -O0 -ggdb -fsanitize=address -fno-omit-frame-pointer -pthread -o example.o -c example.c
        gcc -fsanitize=address -o example example.o -lasan -Wl,-rpath,<pocl-build-dir>/lib/CL <pocl-build-dir>/lib/CL/libOpenCL.so

Output:

  If there's an OpenCL object remaining, ASan will print a backtrace with an OpenCL call name in it::

      Indirect leak of 8 byte(s) in 1 object(s) allocated from:
        #0 0x7fa8f7b0a198 in calloc (/usr/lib/x86_64-linux-gnu/libasan.so.5+0xee198)
        #1 0x7fa8f7607bc0 in pocl_unique_device_list /tmp/lib/CL/pocl_util.c:866
        #2 0x7fa8f75d37ca in POclCreateContext /tmp/lib/CL/clCreateContext.c:172
        #3 0x55d50f21e428 in poclu_get_any_device2 /tmp/lib/poclu/misc.c:84
        #4 0x55d50f21c165 in main /tmp/examples/example1/example1.c:59
        #5 0x7fa8f707bb96 in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x21b96)

  If there's any memory leak in the user's program, ASan will print something like::

      Direct leak of 64 byte(s) in 1 object(s) allocated from:
        #0 0x7f738e999f90 in __interceptor_malloc (/usr/lib/x86_64-linux-gnu/libasan.so.5+0xedf90)
        #1 0x562f6f33e493 in main /tmp/examples/example1/example1.c:74
        #2 0x7f738df0bb96 in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x21b96)

Handling LLVM and driver-allocated memory
-----------------------------------------------

Both valgrind and sanitizers might report a huge amount of memory leaks
coming from PoCL; this is caused mainly by two factors;
LLVM and driver-held static data.

The problem is that the OpenCL API unfortunately doesn't provide any API entry to uninitialize
the entire implementation (e.g. all driver data). It does provide API
entries to unload compiler though: ``clUnloadPlatformCompiler()`` and ``clUnloadCompiler()``.

User can use these to ask PoCL to unload all LLVM data, but it should be noted
that with PoCL, the LLVM data is freed only if all cl_programs and cl_kernels
have been released before calling it.

Usage is simple: call ``clUnloadPlatformCompiler()`` once after
all other OpenCL objects have been released, right before the
program exit.

If the user sets ``POCL_ENABLE_UNINIT`` env var to 1, PoCL will also try to
unload driver data. This feature might not work reliably so it's currently
considered experimental.

Example: Running a program compiled with AddrSanitizer, which calls
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
``clUnloadPlatformCompiler()`` call (therefore, with LLVM context
alive at program exit), ASan will print a lot of memory leaks::

    Indirect leak of 8 byte(s) in 1 object(s) allocated from:
        #0 0x7f99eef43ba0 in operator new(unsigned long) (/usr/lib/x86_64-linux-gnu/libasan.so.5+0xefba0)
        #1 0x7f99eead5aea in WorkItemAliasAnalysis::runOnFunction(llvm::Function&) /tmp/lib/llvmopencl/WorkItemAliasAnalysis.cc:130
        #2 0x7f99e6f76ed5 in llvm::FPPassManager::runOnFunction(llvm::Function&) (/usr/lib/llvm-10/lib/libLLVM-10.so.1+0xb11ed5)

    SUMMARY: AddressSanitizer: 1047772 byte(s) leaked in 3046 allocation(s).
