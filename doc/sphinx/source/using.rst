Basic usage
===========

The basic usage of pocl should be as easy as any other OpenCL implementation.

While it is possible to link against pocl directly, the recommended way is to 
use the ICD interface.

Android applications can use pocl using jni. App has to dlopen
“/data/data/org.pocl.libs/files/lib/libpocl.so” and dlsym OpenCL function
symbols from it.

.. _linking-with-icd:

Linking your program with pocl through an icd loader
----------------------------------------------------

You can link your OpenCL program against an ICD loader. If your ICD loader is
correctly configured to load pocl, your program will be able to use pocl.
See the section below for more information about ICD and  ICD loaders.

Example of compiling an OpenCL host program using the free ocl-icd loader::

   gcc example1.c -o example `pkg-config --libs --cflags OpenCL`

Example of compiling an OpenCL host program using the AMD ICD loader (no
pkg-config support)::

   gcc example1.c -o example -lOpenCL

Installable client driver (ICD)
-------------------------------

pocl is built with the ICD extensions of OpenCL by default. This allows you 
to have several OpenCL implementations concurrently on your computer, and 
select the one to use at runtime by selecting the corresponding cl_platform. 
ICD support can be disabled by adding the flag::

  --disable-icd

to the ./configure script.

In case you also give the --prefix=$INSTALL option to ./configure, you need to 
copy the icd file to where your ICD loader finds it, e.g.::

  cp $INSTALL/etc/OpenCL/vendors/pocl.icd /etc/OpenCL/vendors/pocl.icd

The ocl-icd ICD loader allows to use the OCL_ICD_VENDORS environment variable
to specify a (non-standard) replacement for the /etc/OpenCL/vendors directory.

An ICD loader is an OpenCL library acting as a "proxy" to one of the various OpenCL
implementations installed in the system. pocl does not provide an ICD loader itself, 
but NVidia, AMD, Intel, Khronos, and the free ocl-icd project each provides one.

* `ocl-icd <https://forge.imag.fr/projects/ocl-icd/>`_
* `Khronos <http://www.khronos.org/opencl/>`_

Linking your program directly with pocl
---------------------------------------

Passing the appropriate linker flags is enough to use pocl in your
program. However, please bear in mind that:

#. The current distribution only supports one device, "native",
   which runs the kernels in the host system.
#. Current implementation of both host and kernel runtime libraries
   is not complete. If your program uses any of the unimplemented
   API calls, it will not work. Please implement the mssing APIs
   when you need them and submit us a patch :)

The pkg-config tool is used to locate the libraries and headers in
the installation directory. 

Example of compiling an OpenCL host program against pocl using
the pkg-config::

   gcc example1.c -o example `pkg-config --libs --cflags pocl`

In this link mode, your program will always require the pocl OpenCL library. It
wont be able to run with another OpenCL implementation without recompilation.

Pocl needs to be configured with the --enable-direct-linkage option (enabled
by default)

Using pocl on Android
---------------------

Since pocl is installed in a non-standard path, dynamic linking is not possible.
App has to dlopen “/data/data/org.pocl.libs/files/lib/libpocl.so” and dlsym
OpenCL function symbols from it.

Refer examples/pocl-android-sample/ for hello-world android app that uses pocl.
This app uses a third-party stub OpenCL library that does dlopen/dlsym on its behalf

Vecmathlib
----------

Vecmathlib (aka VML)
`<https://bitbucket.org/eschnett/vecmathlib/wiki/Home>`_ provides
optimized implementations for math builtins such as sqrt, sin, cos,
etc. These are highly recommended as they can be inlined to the call
site and lead to better optimized kernels. A copy of Vecmathlib is
distributed with pocl for convenience in the directory
`lib/kernel/vecmathlib`.

To use VML, you need to have a functional clang++ installed.
Currently, VML is enabled only for x86_64.

Wiki
----

`Wiki <https://github.com/pocl/pocl/wiki>`_
contains some further information and serves as a "scratch pad".


