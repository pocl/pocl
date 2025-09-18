# This is an example Toolchain file to cross-compile for RISC-V
# boards. Copy & modify. Skip 4-8 if using LLVM less build
#
# See the user manual's RISC-V section in the install instructions
# to get started.

SET(CMAKE_CROSSCOMPILING TRUE)
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR riscv64)

# specify the cross compiler
SET(CMAKE_C_COMPILER   /usr/bin/riscv64-linux-gnu-gcc-14)
SET(CMAKE_CXX_COMPILER /usr/bin/riscv64-linux-gnu-g++-14)

SET(CMAKE_C_FLAGS "-march=rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zba_zbb_zbc_zbs_zkt_zvfh_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt -mabi=lp64d")
SET(CMAKE_CXX_FLAGS "-march=rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zba_zbb_zbc_zbs_zkt_zvfh_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt -mabi=lp64d")

# should work, but does not yet. Instead set FIND_ROOT below
set(CMAKE_SYSROOT $ENV{BOARD_ROOT})
# where is the target environment
SET(CMAKE_FIND_ROOT_PATH $CMAKE_SYS_ROOT)
# where to find libraries in target environment
SET(CMAKE_LIBRARY_PATH $CMAKE_SYS_ROOT)

set(ENV{PKG_CONFIG_DIR} "")
set(ENV{PKG_CONFIG_LIBDIR} "${CMAKE_SYSROOT}/usr/lib/pkgconfig:${CMAKE_SYSROOT}/usr/share/pkgconfig:${CMAKE_SYSROOT}/usr/lib/riscv64-linux-gnu/pkgconfig")
set(ENV{PKG_CONFIG_SYSROOT_DIR} ${CMAKE_SYSROOT})

# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

