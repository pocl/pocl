How to use PoCL as SYCL's OpenCL runtime backend on ARM
=======================================================

SYCL is a C++-based programming model that enables you to create one program that can 
execute in a wide variety of devices. For example, you can have a program in
SYCL that can execute on an AMD GPU and an NVIDIA GPU using the same C++ code.
To enable this, SYCL implementations need different backends that can run on a
specific device (e.g. CUDA or OpenCL). This is where PoCL comes to play, providing
a portable OpenCL implementation to use as SYCL's OpenCL runtime.

The objective of this tutorial is to have Intel's oneAPI DPC++ as a SYCL
implementation able to produce programs that can run on ARM using PoCL as the
OpenCL backend.

The tutorial has 2 main parts. Compile Intel's LLVM (oneAPI DPC++) on ARM
and compile PoCL on ARM. We will install DPC++ and then we will install PoCL
independently from DPC++ using a vanilla LLVM (not Intel's version).

Software versions
-----------------

Note that these are the versions I used, you should consider using the
most recent versions.

Listing tags from git repository (obtained with
``git describe --tags``).:

-  DPC++ - Intel LLVM: sycl-nightly/20230413_160000-2-g097d21c
-  PoCL: v4.0
-  SPIRV-Tools: sdk-1.3.243.0-33-gdd03c1f
-  OpenCL-Headers: v2023.02.06-5-g8c4f011
-  OpenCL-ICD-Loader: v2023.02.06-2-gece9144
-  Vanilla LLVM: version 16.0.0 obtained from `this
   link <https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.0/llvm-16.0.0.src.tar.xz>`__.

Installation of DPC++ (Intel’s LLVM)
------------------------------------

Prerequisites
^^^^^^^^^^^^^

1. An installation of
   `spirv-tools <https://github.com/KhronosGroup/SPIRV-Tools/>`__. Build
   with CMake.

Installation
^^^^^^^^^^^^

Official installation instructions can be found at https://intel.github.io/llvm-docs/GetStartedGuide.html#build-dpc-toolchain.

1. Clone the `repository of DPC++ <https://github.com/intel/llvm/>`__ on the branch ``sycl``:

::

   export DPCPP_HOME=~/sycl_workspace
   mkdir $DPCPP_HOME
   cd $DPCPP_HOME

   git clone https://github.com/intel/llvm -b sycl

2. Make SPIRV-Tools available. Replace the paths accordingly:

::

   export PKG_CONFIG_PATH=<YOUR SPIRV-TOOLS PREFIX>/lib64/pkgconfig:$PKG_CONFIG_PATH

3. Modify the configure.py script. This script generates the CMake
   command that configures the compilation. The suggested changes to the final cmake command
   to be generated are:

* Add ``-DLLVM_ENABLE_RUNTIMES=openmp`` if you intend to use OpenMP.
* Add ``-DBUILD_SHARED_LIBS=ON``
* Add ``-DLLVM_ENABLE_EH=ON``
* Add ``-DLLVM_ENABLE_RTTI=ON``
* Add ``-DLLVM_PARALLEL_LINK_JOBS=64``
* Add ``-DOPENMP_ENABLE_LIBOMPTARGET=OFF``
* Add ``-DLLVM_STATIC_LINK_CXX_STDLIB=ON``
* Comment the line with ``"-DBUILD_SHARED_LIBS={}".format(llvm_build_shared_libs),"``

4. Execute the configure script. Replace the build dir accordingly:

::

   python $DPCPP_HOME/llvm/buildbot/configure.py --cmake-gen "Ninja" -o <YOUR BUILD DIR> --host-target "AArch64" -t Release

5. Execute the compile script:

::

   python $DPCPP_HOME/llvm/buildbot/compile.py -o <YOUR BUILD DIR> -j 128

6. I recommend that you execute the following command to try and get a
   full installation of Intel LLVM. I am not entirely sure if it is
   needed, but I strongly recommend it.

::

   cd <YOUR BUILD DIR> && ninja install

7. When using this LLVM you should export some environment variables. I
   suggest you create a script ``env-sycl.sh script`` that exports these variables
   for you. Replace the paths accordingly:

::

   #!/bin/bash

   BASE_PATH=$DPCPP_HOME/<YOUR BUILD DIR>/install
   export PATH=${BASE_PATH}/bin:$PATH

   export CPLUS_INCLUDE_PATH=${BASE_PATH}/include/:$CPLUS_INCLUDE_PATH
   export C_INCLUDE_PATH=${BASE_PATH}/include/:$C_INCLUDE_PATH

   export CPLUS_INCLUDE_PATH=${BASE_PATH}/include/sycl:$CPLUS_INCLUDE_PATH
   export C_INCLUDE_PATH=${BASE_PATH}/include/sycl:$C_INCLUDE_PATH

   export LD_LIBRARY_PATH=${BASE_PATH}/lib:$LD_LIBRARY_PATH
   export LIBRARY_PATH=${BASE_PATH}/lib:$LIBRARY_PATH

   export LD_LIBRARY_PATH=${BASE_PATH}/lib64:$LD_LIBRARY_PATH
   export LIBRARY_PATH=${BASE_PATH}/lib64:$LIBRARY_PATH

   export PKG_CONFIG_PATH=${BASE_PATH}/lib64/pkgconfig/:${BASE_PATH}/lib/pkgconfig/:${BASE_PATH}/share/pkgconfig/:$PKG_CONFIG_PATH

   export CC=clang CXX=clang++

Installation of vanilla LLVM
----------------------------

We will install LLVM 16 (vanilla version, not Intel's). This is the LLVM that PoCL will use,
and is needed to compile PoCL. Notice that we want a static LLVM.

1. Download and uncompress `LLVM tar
   file <https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.0/llvm-16.0.0.src.tar.xz>`__.
2. Execute cmake inside a directory ``build``. If you want to learn more
   about how to configure LLVM installation see `this
   link <https://llvm.org/docs/CMake.html>`__. Replace the paths needed accordingly.

::

   cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_OPTIMIZED_TABLEGEN=ON \
   -DLLVM_TARGETS_TO_BUILD=AArch64 -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" -DLLVM_BUILD_TOOLS=ON \
   -DLLVM_ENABLE_RUNTIMES="openmp" -DBUILD_SHARED_LIBS=OFF -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_PARALLEL_LINK_JOBS=48 -DCMAKE_INSTALL_PREFIX=<YOUR VANILLA LLVM PREFIX> -DLLVM_ENABLE_DOXYGEN=OFF \
   -DLLVM_ENABLE_SPHINX=OFF -DLLVM_ENABLE_LLD=OFF -DLLVM_ENABLE_BINDINGS=OFF -DLLVM_ENABLE_LIBXML2=OFF \
   -DOPENMP_ENABLE_LIBOMPTARGET=OFF -DLLVM_STATIC_LINK_CXX_STDLIB=ON ../llvm

3. ninja install

Installation of PoCL
--------------------

.. _prerequisites-1:

Prerequisites
^^^^^^^^^^^^^

1. An installation of LLVM. This we did in the last section.

2. You will need to install
   `OpenCL-ICD-Loader <https://github.com/KhronosGroup/OpenCL-ICD-Loader>`__
   and
   `OpenCL-Headers <https://github.com/KhronosGroup/OpenCL-Headers>`__.
   Installation is simple, but you should have a specific git checkout
   for both repositories depending on your Intel LLVM version. The git
   checkouts can be found at the `Intel LLVM repository, at file
   ./opencl/CMakeLists.txt, at lines 23 and
   24 <https://github.com/intel/llvm/blob/c5d04bcc0b7b7adf93a9f4c57faf6fada06575cf/opencl/CMakeLists.txt#L23>`__.

PoCL installation
^^^^^^^^^^^^^^^^^

1. After you have both the ICD-Loader and the Opencl Headers installed
   you will need to set up the corresponding variables appropiately:

::

   VVV_ICD_LOADER=<YOUR ICD-LOADER PREFIX>
   VVV_OCL_HEADERS=<YOUR OPENCL HEADERS PREFIX>
   export CPLUS_INCLUDE_PATH=${VVV_OCL_HEADERS}/include:$CPLUS_INCLUDE_PATH/
   export C_INCLUDE_PATH=${VVV_OCL_HEADERS}/include:$CPLUS_INCLUDE_PATH/
   export PKG_CONFIG_PATH=${VVV_ICD_LOADER}/lib64/pkgconfig/:${VVV_OCL_HEADERS}/share/pkgconfig:$PKG_CONFIG_PATH
   export LIBRARY_PATH=${VVV_ICD_LOADER}/lib64/:$LIBRARY_PATH
   export LD_LIBRARY_PATH=${VVV_ICD_LOADER}/lib64:$LD_LIBRARY_PATH

2. Clone the `PoCL repository <https://github.com/pocl/pocl>`__, create
   a build directory and from inside execute CMake. Remember to replace the paths accordingly:

::

   CC="clang" CXX="clang++" cmake -G Ninja -DCMAKE_INSTALL_PREFIX=<YOUR POCL INSTALLATION PREFIX> \
   -DENABLE_ICD=ON -DCMAKE_PREFIX_PATH="<YOUR VANILLA LLVM PREFIX>;$VVV_ICD_LOADER;$VVV_OCL_HEADERS" \
   -DENABLE_SPIRV=ON -DLLVM_SPIRV=<FULL PATH TO THE llvm-spirv BINARY FROM INTEL LLVM>  \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSTATIC_LLVM=ON ..

3. ninja install
4. OPTIONAL: Run the test suite: ``ctest -j 128 -L internal``.
5. When using PoCL you should export some environment variables. I
   suggest you create a env-pocl.sh script that exports the variables
   for you:

::

   #!/bin/bash

   BASE_PATH=<YOUR POCL INSTALLATION PREFIX>

   # BIN
   export PATH=${BASE_PATH}/bin:$PATH

   # HEADERS
   export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${BASE_PATH}/include/
   export C_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${BASE_PATH}/include/

   # LIBS
   export LD_LIBRARY_PATH=${BASE_PATH}/lib64:$LD_LIBRARY_PATH
   export LIBRARY_PATH=${BASE_PATH}/lib64:$LIBRARY_PATH

   export PKG_CONFIG_PATH=${BASE_PATH}/lib64/pkgconfig/::$PKG_CONFIG_PATH

   export OCL_ICD_VENDORS=${BASE_PATH}/etc/OpenCL/vendors/

   # Variables for debugging programs
   export VVV_pocl_help="SYCL_PI_TRACE=2 POCL_DEBUG=all OCL_ICD_ENABLE_TRACE=1"


Using SYCL with PoCL
--------------------

I uploaded a simple example to test if SYCL is working with PoCL. It
just tests that you can compile and execute simple SYCL programs using
PoCL as the OpenCL implementation that SYCL uses.

1. ``source env-pocl.sh``
2. ``source env-sycl.sh``
3. The first test you should do is validate that the SYCL runtime can
   find and query simple information from the PoCL runtime. This is how
   it looks for me:

::

   [host@user]$ sycl-ls
   [opencl:cpu:0] Portable Computing Language, pthread-0xd01 OpenCL 4.0 PoCL HSTR: pthread-aarch64-unknown-linux-gnu-tsv110 [4.0-pre next-0-gbbb3d72]

4. After that, you can try with this test:

.. code-block:: c++

    // t.cpp
    #include <CL/sycl.hpp>
    #include <iostream>

    #define N 10

    int main() {
      sycl::queue q;


      sycl::event ex;
      int* d_buf = sycl::malloc_device<int>(N, q   );
      int* h_buf = sycl::malloc_host<int>(N, q );



      for(int i = 0; i < N; i ++){
            h_buf[i] = i*i;
      }
      q.memcpy(d_buf, h_buf, N*sizeof(int)).wait();

      q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> it){
        const int i = it[0];
        d_buf[i] += i;
      }).wait();
      q.memcpy(h_buf, d_buf, N*sizeof(int)).wait();

      int correct = 1;
      for(int i = 0; i < N; i ++){
        if(h_buf[i] != i*i + i){
            std::cerr << "ERROR: h_buf[" << i << "]=" << h_buf[i] << " and shuold be " << i*i + i << std::endl;
            correct =0;
        }
      }
      if(correct){
        std::cout << "Results are correct!!\n";
      }

      //# Print the device name
      std::cout << "Device 1: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

      return 0;
    }


5. ``clang++ -fsycl t.cpp && ./a.out``

Alternative way to test SYCL with PoCL
--------------------------------------

In addition to previous example, it's now possible to build PoCL with support for external SYCL testsuites,
though this has so far been tested only with x86-64.

In the PoCL installation step, add the following options to the CMake command ::

    "-DENABLE_TESTSUITES=dpcpp-book-samples;oneapi-samples;simple-sycl-samples;intel-compute-samples"
     -DSYCL_CXX_COMPILER=<DPCPP_BASE_PATH>/bin/clang++ -DSYCL_LIBDIR=<DPCPP_BASE_PATH>/lib

where DPCPP_BASE_PATH is the BASE_PATH from the env-sycl.sh. The quotes around -DENABLE_TESTSUITES
are required, since it contains semicolon. After building PoCL with ``ninja install``,
you must build the external testsuites with ``ninja prepare_examples``. After the successful build,
there should be a new ctest label for each testsuite. Hence you can run the tests with::

    ctest -L "dpcpp-book-samples|oneapi-samples|simple-sycl-samples|intel-compute-samples"

...check that you're using PoCL (with sycl-ls) before running ctest.

Known issues
------------

1. ``queue.memset()`` is not supported using PoCL right now (see 
`issue #1223 <https://github.com/pocl/pocl/issues/1223>`__). 
You should use ``queue.fill()`` instead.

2. Querying an event's execution status with 
``event.get_info<sycl::info::event::command_execution_status>()``
might return an invalid value when the OpenCL event is in ``CL_QUEUED`` state. 
This is a known issue of OpenCL's backend at Intel's DPC++ compiler. 
See `issue #9099 on Intel's LLVM repository <https://github.com/intel/llvm/issues/9099>`__.

3. When trying to compile with ``-O0`` you will get a runtime error 
saying that some OpenCL kernel has an ``undefined symbol: _group_id_x``.

Troubleshooting
---------------

If you ever have runtime errors, these errors can be located at three
parts: they might be in the ICD-Loader, in PoCL or in SYCL.

-  To query debug information from the ICD-Loader:
   ``OCL_ICD_ENABLE_TRACE=1 ./a.out``
-  To query debug information from the POCL runtime:
   ``POCL_DEBUG=all ./a.out``
-  To query debug information from the SYCL runtime:
   ``SYCL_PI_TRACE=2 ./a.out``

These variables can be combined if needed.
