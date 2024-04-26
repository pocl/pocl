# Toolchain file to cross-compile for ARM 64bit systems from x86_64

SET(CMAKE_SYSTEM_NAME Linux)

# specify the cross compiler
SET(CMAKE_C_COMPILER   /usr/bin/aarch64-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)

# CPU specific flags, be sure to change accordingly
SET(CMAKE_CXX_FLAGS "-mcpu=cortex-a72")
SET(CMAKE_C_FLAGS "-mcpu=cortex-a72")

set(CMAKE_SYSROOT /mnt/ROOTFS)
# where is the target environment
SET(CMAKE_FIND_ROOT_PATH /mnt/ROOTFS)
# where to find libraries in target environment
SET(CMAKE_LIBRARY_PATH /mnt/ROOTFS/usr/lib/aarch64-linux-gnu)

# search for programs in the build host directories
# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

