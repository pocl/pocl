# Toolchain file to cross-compile for ARM/MIPS/other boards from x86_64

SET(CMAKE_SYSTEM_NAME Linux)

# specify the cross compiler
SET(CMAKE_C_COMPILER   /usr/bin/riscv64-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER /usr/bin/riscv64-linux-gnu-g++)

# CPU specific flags, be sure to change accordingly
SET(CMAKE_CXX_FLAGS "-mabi=lp64d -march=rv64imafdczbb0p93_zba0p93 -mcpu=sifive-u74 -mtune=sifive-7-series")
SET(CMAKE_C_FLAGS "-mabi=lp64d -march=rv64imafdczbb0p93_zba0p93 -mcpu=sifive-u74 -mtune=sifive-7-series")

set(CMAKE_SYSROOT /mnt/ROOTFS)
# where is the target environment
SET(CMAKE_FIND_ROOT_PATH /mnt/ROOTFS)
# where to find libraries in target environment
SET(CMAKE_LIBRARY_PATH /mnt/ROOTFS/usr/lib/riscv64-linux-gnu)

# search for programs in the build host directories
# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

