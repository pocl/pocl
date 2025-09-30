# Toolchain file to cross-compile for RISC-V boards from x86_64
# Copy & modify.

# See the user manual's RISC-V section in the install instructions
# to get started.

SET(CMAKE_SYSTEM_NAME Linux)

# specify the cross compiler
SET(CMAKE_C_COMPILER   /usr/bin/riscv64-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER /usr/bin/riscv64-linux-gnu-g++)

# be sure to change accordingly
# CPU specific flags for Starfive JH7110
SET(CMAKE_CXX_FLAGS "-mabi=lp64d -march=rv64imafdczbb0p93_zba0p93 -mcpu=sifive-u74 -mtune=sifive-7-series")
SET(CMAKE_C_FLAGS "-mabi=lp64d -march=rv64imafdczbb0p93_zba0p93 -mcpu=sifive-u74 -mtune=sifive-7-series")

# CPU specific flags for Spacemit-X60
SET(CMAKE_C_FLAGS "-mabi=lp64d -march=rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zba_zbb_zbc_zbs_zkt_zvfh_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt -mcpu=spacemit-x60")
SET(CMAKE_CXX_FLAGS "-mabi=lp64d -march=rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zba_zbb_zbc_zbs_zkt_zvfh_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt -mcpu=spacemit-x60")

set(CMAKE_SYSROOT /mnt/ROOTFS)
# where is the target environment
SET(CMAKE_FIND_ROOT_PATH /mnt/ROOTFS)
# where to find libraries in target environment
SET(CMAKE_LIBRARY_PATH /mnt/ROOTFS/usr/lib/riscv64-linux-gnu)

# ensure pkg-config searches in target directories
set(ENV{PKG_CONFIG_DIR} "")
set(ENV{PKG_CONFIG_LIBDIR} "${CMAKE_SYSROOT}/usr/lib/pkgconfig:${CMAKE_SYSROOT}/usr/share/pkgconfig:${CMAKE_SYSROOT}/usr/lib/riscv64-linux-gnu/pkgconfig")
set(ENV{PKG_CONFIG_SYSROOT_DIR} ${CMAKE_SYSROOT})

# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

SET(CMAKE_CROSSCOMPILING TRUE)
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR riscv64)

