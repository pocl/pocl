# This is an example Toolchain file to cross-compile for ARM/MIPS/other
# boards from x86_64. Copy & modify. Skip 4-8 if using LLVM less build
#
# Steps:
# 1) Install g++ and gcc cross-compilers
#    (apt install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf)
# 2) On your board, install libltdl, ocl-icd and libhwloc + their development headers
# 3) copy the entire root filesystem of the board somewhere on your host,
#    then set CMAKE_FIND_ROOT_PATH below to this path
# 4) Build clang and llvm for the build machine and install them. ($BUILD_PREFIX)
# 5) Build clang and llvm for the host machine and install them. ($HOST_PREFIX)
# 6) copy llvm-config from build to host. (cp $BUILD_PREFIX/bin/llvm-config $HOST_PREFIX/bin/llvm-config)
# 7) Install pkg-config for build
# 8) Install hwloc, ocl-icd for host and set `PKG_CONFIG_PATH` env variable to the paths
#    eg: export PKG_CONFIG_PATH=/path/to/hwloc/prefix/lib/pkgconfig:/path/to/opencl/prefix/lib/pkgconfig
# 9) run cmake like this:
#          cmake -DHOST_DEVICE_BUILD_HASH=<SOME_HASH> -DOCS_AVAILABLE=<0 if LLVM-less, 1 if with LLVM>
#           -DCMAKE_TOOLCHAIN_FILE=<path-to-this-file>
#           -DCMAKE_PREFIX_PATH=$HOST_PREFIX
#           -DLLC_TRIPLE=<your-triple (e.g.arm-gnueabihf-linux-gnu)
#           -DLLVM_HOST_TARGET=<your-triple (e.g.arm-gnueabihf-linux-gnu)
#           -DLLC_HOST_CPU=<your-cpu (e.g. armv7a)>
#           -DLLVM_BINDIR=$BUILD_PREFIX/bin
#           <path-to-pocl-source>

SET(CMAKE_SYSTEM_NAME Linux)

# specify the cross compiler
SET(CMAKE_C_COMPILER   /usr/bin/arm-linux-gnueabihf-gcc)
SET(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabihf-g++)

# should work, but does not yet. Instead set FIND_ROOT below
# set(CMAKE_SYSROOT /home/a/zynq/ZYNQ_ROOT)
# where is the target environment
SET(CMAKE_FIND_ROOT_PATH  /path/to/target_ROOT)
# where to find libraries in target environment
SET(CMAKE_LIBRARY_PATH /path/to/target_ROOT/usr/lib/arm-linux-gnueabihf)


# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
