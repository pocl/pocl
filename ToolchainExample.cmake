# This is an example Toolchain file to cross-compile for ARM/MIPS/other
# boards from x86_64. Copy & modify. Skip 4-8 if using LLVM less build
#
# x86_64 = "build"
# ARM/MIPS/other board = "host" or "board"
#
# Steps:
# (note: hwloc is now optional)
# 1) on build system, install g++ and gcc cross-compilers
#    (apt install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf)
# 2) On the board, install ocl-icd and libhwloc (optional) + their development headers
# 3) copy the entire root filesystem of the board somewhere to the build system,
#    then set CMAKE_FIND_ROOT_PATH below to this path
# 4) Build clang and llvm for the build system and install them. ($BUILD_PREFIX)
# 5) Build clang and llvm for the host and install them. ($HOST_PREFIX)
# 6) copy llvm-config from build to host. (cp $BUILD_PREFIX/bin/llvm-config $HOST_PREFIX/bin/llvm-config)
# 7) Install pkg-config for build
# 8) Install hwloc, ocl-icd for host and set `PKG_CONFIG_PATH` env variable to the paths
#    eg: export PKG_CONFIG_PATH=/path/to/hwloc/prefix/lib/pkgconfig:/path/to/opencl/prefix/lib/pkgconfig
# 9) run cmake like this:
#          cmake -DHOST_DEVICE_BUILD_HASH=<SOME_HASH> (see below)
#           -DENABLE_LLVM=<0 if LLVM-less, 1 if with LLVM>
#           -DCMAKE_TOOLCHAIN_FILE=<path-to-this-file>
#           -DCMAKE_PREFIX_PATH=$HOST_PREFIX
#           -DLLC_TRIPLE=<your-triple (e.g.arm-gnueabihf-linux-gnu)
#           -DLLVM_HOST_TARGET=<your-triple (e.g.arm-gnueabihf-linux-gnu)
#           -DLLC_HOST_CPU=<your-cpu (e.g. armv7a)>
#           -DLLVM_BINDIR=$BUILD_PREFIX/bin
#           <path-to-pocl-source>
#
# ... where SOME_HASH is a string that can be set to anything;
# when loading OpenCL program binaries, PoCL uses it to check
# that the PoCL which built the binary is compatible with the
# PoCL that's loading the binary.

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
