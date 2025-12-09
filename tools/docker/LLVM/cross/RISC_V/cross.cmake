# Toolchain file to cross-compile for RISC-V boards from x86_64
# Copy & modify.

# See the user manual's RISC-V section in the install instructions
# to get started.

SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_CROSSCOMPILING TRUE)
SET(CMAKE_SYSTEM_PROCESSOR riscv64)

# specify the cross compiler
SET(CMAKE_C_COMPILER   /usr/bin/riscv64-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER /usr/bin/riscv64-linux-gnu-g++)

# be sure to change accordingly
# CPU specific flags for Starfive JH7110
SET(CMAKE_CXX_FLAGS_INIT "-mabi=lp64d -march=rv64imafdczbb0p93_zba0p93 -mcpu=sifive-u74 -mtune=sifive-7-series")
SET(CMAKE_C_FLAGS_INIT "-mabi=lp64d -march=rv64imafdczbb0p93_zba0p93 -mcpu=sifive-u74 -mtune=sifive-7-series")

# CPU specific flags for Spacemit-X60 & GCC 13
#SET(CMAKE_C_FLAGS_INIT "-mabi=lp64d -march=rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zba_zbb_zbc_zbs_zkt_zvfh_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt -mcpu=spacemit-x60")
#SET(CMAKE_CXX_FLAGS_INIT "-mabi=lp64d -march=rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zba_zbb_zbc_zbs_zkt_zvfh_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt -mcpu=spacemit-x60")

# CPU specific flags for Spacemit-X60 & GCC 14
#SET(CMAKE_C_FLAGS_INIT "-mabi=lp64d -march=rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zfhmin_zca_zcd_zba_zbb_zbc_zbs_zkt_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt")
#SET(CMAKE_CXX_FLAGS_INIT "-mabi=lp64d -march=rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zfhmin_zca_zcd_zba_zbb_zbc_zbs_zkt_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt")

set(CMAKE_SYSROOT /mnt/ROOTFS)
# where to find libraries in target environment
set(CMAKE_LIBRARY_PATH "${CMAKE_SYSROOT}/usr/lib/riscv64-linux-gnu")

# ensure pkg-config searches in target directories
set(ENV{PKG_CONFIG_DIR} "")
set(ENV{PKG_CONFIG_LIBDIR} "${CMAKE_SYSROOT}/usr/lib/pkgconfig:${CMAKE_SYSROOT}/usr/share/pkgconfig:${CMAKE_SYSROOT}/usr/lib/riscv64-linux-gnu/pkgconfig")
set(ENV{PKG_CONFIG_SYSROOT_DIR} ${CMAKE_SYSROOT})
