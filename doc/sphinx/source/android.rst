
Android support
---------------------

It is possible to build and use PoCL on Android. However, the setup requires a number of options to be set.
To see an example project, have a look at the `PoCL-R Reference Android Java Client <https://github.com/cpc/PoCL-R-Reference-Android-Java-Client>`_ .
This Reference app uses both the :ref:`proxy<proxy-label>` and :ref:`remote<remote-label>` device in its example apps. It also builds a custom version of `JOCL <http://jocl.org/>`_ so
that PoCL can be used in Java instead of calling C code using the Java Native Interface (jni). These guidelines assume
that Android studio is used as an IDE, but it should be possible to do something similar with a different IDE. It is also
assumed that a recent enough version of the NDK and CMake (the one found in the SDK tools of Android Studio) have been
installed via Android Studio. Versions that have been used before include: NDK 25.1.8937393 and 26.0.10792818 and CMake
3.22.1.

CMake Arguments
~~~~~~~~~~~~~~~~~~

A number of features in PoCL such as CPU devices and the icd loader are not available on Android. Below is a list of
recommended CMake options::

    -DENABLE_LLVM=0 -DHOST_DEVICE_BUILD_HASH=00000000 -DENABLE_ICD=0 -DENABLE_LOADABLE_DRIVERS=0 -DENABLE_HOST_CPU_DEVICES=0 -DENABLE_HWLOC=0 -DENABLE_POCLCC=0 -DENABLE_TESTS=0 -DENABLE_EXAMPLES=0 -DBUILD_SHARED_LIBS=0 -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_NDK=${ANDROID_NDK} -DANDROID_PLATFORM=${ANDROID_PLATFORM_LEVEL} -DANDROID_ABI=${ANDROID_ABI} -DANDROID_NATIVE_API_LEVEL=${ANDROID_PLATFORM_LEVEL}

It is recommended to Build PoCL as an external project in the CMakeLists.txt that belongs to the native code of the
Android project. This will set the ``ANDROID_NDK, ANDROID_PLATFORM_LEVEL`` and ``ANDROID_ABI`` to what you are building the
project for. By default, Android Studio will build native code for multiple architectures (ARM 32/64 and x86), so the
``ANDROID_ABI`` will change for each architecture. Adding pocl as a library dependency to your native code will ensure that
it is packed into the APK. It is recommended to set ``-DBUILD_SHARED_LIBS=0`` so that PoCL gets built as a static library
(libpocl.a) as this is easier to use.

Building Remote Client
~~~~~~~~~~~~~~~~~~~~~~~~

If you want to make use of PoCL-R, you can add ``-DENABLE_REMOTE_CLIENT=YES`` to the cmake options
and make sure that network access is allowed in the `AndroidManifest.xml`.


Building Proxy Device
~~~~~~~~~~~~~~~~~~~~~~~~

The proxy device allow you make use of any system provided OpenCL implementation as well as any devices provided by PoCL
at the same time. Combined with the remote device, this allows you for example to easily switch between executing kernels
locally or remotely or create a pipeline where work is done on both devices at the same time. To make use of the Proxy
device on Android, You first need to make sure that the phone comes with an OpenCL library and that is whitelisted by
the vendor. Starting with API level 24, vendors need whitelist libraries that are allowed to be dlopened. To check that
OpenCL is whitelisted do this:

    1. adb into the phone
    2. run::

        cat /vendor/etc/public.libraries.txt

    3. check that `libOpenCL.so` is there

For newer Android versions (Android 12 and up), you also need to add::

        <uses-native-library
            android:name="libOpenCL.so"
            android:required="false" />

to the ``<applications>`` element of the `AndroidManifest.xml`

Once you know that your phone comes with an OpenCL library, it's possible to use the proxy device. To build the proxy device add the
following CMake options to the ones mentioned before: ``-DENABLE_PROXY_DEVICE=YES -DVISIBILITY_HIDDEN=NO``. This will build
the proxy device and pocl as a static library. If you want to use JOCL, you need to also add ``-DPROXY_USE_LIBOPENCL_STUB=YES``
and set ``-DBUILD_SHARED_LIBS=YES``. This will build a dynamic library of pocl.

*NOTE:* The proxy driver suffers from the same issues the remote driver has with :ref:`Mali GPUs<remote-issues-label>`.
See that section for a workaround.


Setting PoCL Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to set PoCL environment variables is to create a native function that calls stdlib.h's setenv function.

Using JOCL
~~~~~~~~~~~~~~

It is possible to use JOCL on Android. However, by default JOCL does not get built for Android. It also doesn't look for libpocl.
See the android reference client readme on how to build JOCL for android and a submodule to our JOCL repo that looks for
`libpocl.so` on Android.

