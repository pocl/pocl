.. _defined-built-in-kernels:

============================
Defined Built-in Kernels
============================

Defined Built-in Kernels (DBK) allow for a standardized set of built-in kernels with well-defined semantics that can be
configured during creation of the OpenCL program. The extension draft goes more into details and can be found on
`github <https://github.com/KhronosGroup/OpenCL-Docs/pull/1007>`_. Some of these DBKs require PoCL to be built in a
specific way, therefore please check below for building and usage instructions.

exp_jpeg_encode And exp_jpeg_decode
______________________________________

An experimental set of DBKs dedicated to compressing raw RGB images to JPEG and back.

Building
^^^^^^^^

Currently, only the CPU devices support these DBKs.
The PoCL CPU devices support these DBKs by making use of the `libjpeg-turbo <https://libjpeg-turbo.org>`_ library,
specifically API version 3.0 and above. API version 3.0 might not be available on your distro of choice, in that case
refer to the `build instructions <https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/BUILDING.md>`_
on how to build it. After running `make install`, the library will be installed in `/opt` . CMake might have some
trouble finding this library, therefore add the following line to your CMake arguments when building PoCL:

    -Dlibjpeg-turbo_DIR=/opt/libjpeg-turbo/lib64/cmake/libjpeg-turbo/

Usage
^^^^^

The JPEG DBKs work on buffers of RGB images, similar to other image manipulation frameworks. The characteristics of the
buffer are as follows: point (0,0) is the top left of the image and each pixel consists of 3 bytes in RGB order without
padding. So to get a pixel value from a certain row, add an offset of row_index * width * 3 to the index. The fourcc
code that comes closest to this is the BI_RGB 24bpp format. Check the `include/CL/cl_exp_defined_builtin_kernels.h` for
info on DBK attributes and `tests/runtime/test_dbk_jpeg.c` for an example program making use of the JPEG DBKs.


exp_onnx_inference
__________________

This experimental DBK provides the ability to ingest an ONNX file and perform
inference with the contained network.

Building with ONNX kernel support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to support this builtin kernel through the CPU devices, PoCL needs to be built with ONNX Runtime
support.


Distro-provided ONNX Runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are on a Linux distribution that packages ONNX Runtime, install the
development package for it and build PoCL normally. The ONNX Runtime should be
detected automatically if it is a sufficiently recent version that ships with
CMake configuration files.

Building with a custom ONNX Runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some popular Linux distributions don't provide prebuilt packages of ONNX Runtime
so you may have to
`build your own <https://onnxruntime.ai/docs/build/inferencing.html>` and tell
CMake where to find it. Steps for building a suitable copy of ONNX Runtime (run
outside of the PoCL source tree) are as follows:

    git clone --recursive https://github.com/microsoft/onnxruntime/
    cd onnxruntime
    ./build.sh --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=${HOME}/onnxruntime-bin
    # wait for a looong time
    make -C build/Linux/Release install # install the built files to the CMAKE_INSTALL_PREFIX specified above

Feel free to switch the ``CMAKE_INSTALL_PREFIX`` in the above build command to
whatever you prefer, just make sure to pass it to cmake when configuring PoCL.

.. note::
    ONNXRuntime generates CMake target definitions when installing, but these
    are specific to the build type specified with the --config flag. If you are
    building PoCL in a different configuration than Release, you may have to
    change the --config flag in the ONNXRuntime to match the configuration of
    your PoCL build.

Now proceed to building PoCL as usual, but tell cmake where to find the
configuration files of ONNX Runtime:

    mkdir build
    cd build
    cmake .. -Donnxruntime_DIR=${HOME}/onnxruntime-bin/lib/cmake/onnxruntime
    make -j$(nproc)
