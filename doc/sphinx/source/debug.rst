Debugging OpenCL applications with Pocl
========================================


There are several ways to debug applications with Pocl,
differing in debugging coverage and impact on speed.


Simple debugging with Pocl's debug messages
--------------------------------------------

Upsides:
  * doesn't require recompiling the application or Pocl

Downsides:
  * very limited scope

Setup:

The only thing required is to set "POCL_DEBUG" environment variable
to some value. The most useful values are:

 * ``POCL_DEBUG=err,warn`` - this will limit the output to errors and
   warnings. This might help spot some OpenCL API calls which return
   an error value. Also helps if a call can return CL_INVALID_VALUE for
   multiple reasons, since Pocl prints the reason.

 * ``POCL_DEBUG=refcount`` - this will limit the output to refcount increases
   and decreases. Might help spot some memory leaks

Example::

     gcc example1.c -o example -lOpenCL
     export POCL_DEBUG=refcount
     ./example1

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

Currently pocl recognizes four sanitizers:
Address, Leak, Undefined behaviour and Thread.

Corresponding pocl CMake flags to enable them are:
``ENABLE_ASAN, ENABLE_LSAN, ENABLE_UBSAN, ENABLE_TSAN.``

See also "handling LLVM" below.

Downsides:
  * requires rebuilding both the application and pocl;
  * the application and pocl's runtime code are compiled with sanitizer,
    but ATM the kernels are not compiled with sanitizer

Upsides:
  * much faster than Valgrind

Setup:
  * for e.g. Address sanitizer, build pocl with these flags::

       -DENABLE_ASAN=1 -DENABLE_ICD=0 -DCMAKE_BUILD_TYPE=Debug

  * this will result in ``lib/CL/libOpenCL.so``; rebuild your application
    with the correct ``-fsanitize=X`` flag and link it to ``lib/CL/libOpenCL.so``

Example:

  building an "example.c" with Address sanitizer::

        gcc -O0 -ggdb -fsanitize=address -fno-omit-frame-pointer -pthread -o example.o -c example.c
        gcc -fsanitize=address -o example example.o -lasan -Wl,-rpath,<pocl-build-dir>/lib/CL <pocl-build-dir>/lib/CL/libOpenCL.so

Output:

  if there's an OpenCL object remaining, ASan will print something with the OpenCL call name in it::

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

Downsides:
  * extremely slow esp with computationally intensive kernels
  * ocassionally false positive reports

Upsides:
  * the entire application including kernels can be debugged
  * does not strictly require recompilation (though for usable
    backtraces, requires debuginfo)

Setup:
  * build Pocl with -DCMAKE_BUILD_TYPE=Debug
  * ``export POCL_EXTRA_BUILD_FLAGS="-g -cl-opt-disable"`` (this will cause
    all kernels to compile with debuginfo)
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

The setup is similar to Valgrind; build pocl with ``-DCMAKE_BUILD_TYPE=Debug``
and export ``POCL_EXTRA_BUILD_FLAGS="-g -cl-opt-disable"`` before running gdb.

Handling LLVM and driver-allocated memory
-----------------------------------------------

Both valgrind and sanitizers might report a huge amount of memory leaks
coming from Pocl; this is caused mainly by two factors,
LLVM and driver-held static data.

The OpenCL API unfortunately doesn't provide any API entry to uninitialize
the entire implementation (e.g. all driver data). It does provide API
entries to unload compiler: ``clUnloadPlatformCompiler()`` and ``clUnloadCompiler()``.

User can use these to ask Pocl to unload all LLVM data; note that with
Pocl, this only works if all cl_programs and cl_kernels have been released.

Usage is simple: call ``clUnloadPlatformCompiler()`` once, after
all other opencl objects have been released, right before program exit.

If the user sets ``POCL_ENABLE_UNINIT`` env var to 1, Pocl will also try to
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

Running the same program with empty pocl cache and removed
``clUnloadPlatformCompiler()`` call (therefore with LLVM context
alive at program exit), ASan will print a lot of memory leaks::

	Indirect leak of 8 byte(s) in 1 object(s) allocated from:
		#0 0x7f99eef43ba0 in operator new(unsigned long) (/usr/lib/x86_64-linux-gnu/libasan.so.5+0xefba0)
		#1 0x7f99eead5aea in WorkItemAliasAnalysis::runOnFunction(llvm::Function&) /tmp/lib/llvmopencl/WorkItemAliasAnalysis.cc:130
		#2 0x7f99e6f76ed5 in llvm::FPPassManager::runOnFunction(llvm::Function&) (/usr/lib/llvm-10/lib/libLLVM-10.so.1+0xb11ed5)

	SUMMARY: AddressSanitizer: 1047772 byte(s) leaked in 3046 allocation(s).
