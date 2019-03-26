# This is an example Toolchain file to cross-compile for ARM/MIPS/other
# boards from x86_64. Copy & modify
#
# Steps:
# 1) Install g++ and gcc cross-compilers
#    (apt install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf)
# 2) On your board, install libltdl, ocl-icd and libhwloc + their development headers
# 3) copy the entire root filesystem of the board somewhere on your host,
     then set CMAKE_FIND_ROOT_PATH below to this path
# 4) run cmake like this:
#          cmake -DHOST_DEVICE_BUILD_HASH=<SOME_HASH> -DENABLE_LLVM=0
#           -DCMAKE_TOOLCHAIN_FILE=<path-to-this-file>
#           -DLLC_TRIPLE=<your-triple (e.g.arm-gnueabihf-linux-gnu)
#           -DLLC_HOST_CPU=<your-cpu (e.g. armv7a)>
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
