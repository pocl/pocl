
#cmakedefine BUILD_HSA
#cmakedefine BUILD_CUDA
#cmakedefine BUILD_BASIC
#cmakedefine BUILD_PTHREAD
#cmakedefine BUILD_ACCEL

#define BUILDDIR "@BUILDDIR@"

/* "Build with ICD" */
#cmakedefine BUILD_ICD

#define CMAKE_BUILD_TYPE "@CMAKE_BUILD_TYPE@"

#cmakedefine ENABLE_ASAN
#cmakedefine ENABLE_LSAN
#cmakedefine ENABLE_TSAN
#cmakedefine ENABLE_UBSAN

#cmakedefine ENABLE_CONFORMANCE

#cmakedefine ENABLE_HWLOC

#cmakedefine ENABLE_HOST_CPU_DEVICES

#cmakedefine ENABLE_POCL_BUILDING

#cmakedefine ENABLE_POCL_FLOAT_CONVERSION

#cmakedefine ENABLE_RELOCATION

#cmakedefine ENABLE_SLEEF

#cmakedefine ENABLE_SPIR

#cmakedefine ENABLE_SPIRV

#cmakedefine HAVE_FORK

#cmakedefine HAVE_VFORK

#cmakedefine HAVE_CLOCK_GETTIME

#cmakedefine HAVE_FDATASYNC

#cmakedefine HAVE_FSYNC

#cmakedefine HAVE_MKOSTEMPS

#cmakedefine HAVE_MKSTEMPS

#cmakedefine HAVE_MKDTEMP

#cmakedefine HAVE_FUTIMENS

#cmakedefine HAVE_LTTNG_UST

#cmakedefine HAVE_LTDL

#cmakedefine HAVE_OCL_ICD

#cmakedefine HAVE_POSIX_MEMALIGN

#cmakedefine HAVE_UTIME

#cmakedefine OCS_AVAILABLE

/* this is used all over the runtime code */
#define HOST_CPU_CACHELINE_SIZE @HOST_CPU_CACHELINE_SIZE@



#ifdef ENABLE_HOST_CPU_DEVICES

#define HOST_AS_FLAGS  "@HOST_AS_FLAGS@"

#define HOST_CLANG_FLAGS  "@HOST_CLANG_FLAGS@"

#define HOST_DEVICE_EXTENSIONS "@HOST_DEVICE_EXTENSIONS@"

#cmakedefine HOST_CPU_FORCED

#define HOST_LD_FLAGS  "@HOST_LD_FLAGS@"

#define HOST_LLC_FLAGS  "@HOST_LLC_FLAGS@"

#cmakedefine HOST_FLOAT_SOFT_ABI

#define HOST_DEVICE_BUILD_HASH "@HOST_DEVICE_BUILD_HASH@"

#endif



#ifdef BUILD_HSA

#cmakedefine HAVE_HSA_EXT_AMD_H

#define AMD_HSA @AMD_HSA@

#define HSA_DEVICE_EXTENSIONS "@HSA_DEVICE_EXTENSIONS@"

#define HSAIL_ASM "@HSAIL_ASM@"

#define HSAIL_ENABLED @HSAIL_ENABLED@

#endif





#define CMAKE_BUILD_TYPE "@CMAKE_BUILD_TYPE@"

#define LINK_COMMAND "@LINK_COMMAND@"

#cmakedefine LINK_WITH_CLANG






#ifdef OCS_AVAILABLE

#define KERNELLIB_HOST_CPU_VARIANTS "@KERNELLIB_HOST_CPU_VARIANTS@"

#cmakedefine KERNELLIB_HOST_DISTRO_VARIANTS

#define CLANG "@CLANG@"

#define CLANG_RESOURCE_DIR "@CLANG_RESOURCE_DIR@"

#define CLANGXX "@CLANGXX@"

#define LLVM_LLC "@LLVM_LLC@"

#define LLVM_SPIRV "@LLVM_SPIRV@"

/* "Using LLVM 3.6" */
#cmakedefine LLVM_3_6

/* "Using LLVM 3.7" */
#cmakedefine LLVM_3_7

/* "Using LLVM 3.8" */
#cmakedefine LLVM_3_8

/* "Using LLVM 3.9" */
#cmakedefine LLVM_3_9

/* "Using LLVM 4.0" */
#cmakedefine LLVM_4_0

/* "Using LLVM 5.0" */
#cmakedefine LLVM_5_0

/* "Using LLVM 6.0" */
#cmakedefine LLVM_6_0

/* "Using LLVM 7.0" */
#cmakedefine LLVM_7_0

/* "Using LLVM 8.0" */
#cmakedefine LLVM_8_0

#cmakedefine LLVM_BUILD_MODE_DEBUG

#ifndef LLVM_VERSION
#define LLVM_VERSION "@LLVM_VERSION_FULL@"
#endif


#endif



/* Defined to greatest expected alignment for extended types, in bytes. */
#define MAX_EXTENDED_ALIGNMENT @MAX_EXTENDED_ALIGNMENT@

/* used in lib/CL/devices/basic */
#define OCL_KERNEL_TARGET  "@OCL_KERNEL_TARGET@"
#define OCL_KERNEL_TARGET_CPU  "@OCL_KERNEL_TARGET_CPU@"

#define PACKAGE_VERSION "@PACKAGE_VERSION@"

#define POCL_KCACHE_SALT "@POCL_KCACHE_SALT@"

#define POCL_KERNEL_CACHE_DEFAULT @POCL_KERNEL_CACHE_DEFAULT@

#define HOST_DEVICE_ADDRESS_BITS @HOST_DEVICE_ADDRESS_BITS@

#cmakedefine POCL_DEBUG_MESSAGES

#define POCL_INSTALL_PRIVATE_HEADER_DIR "@POCL_INSTALL_PRIVATE_HEADER_DIR@"

#define POCL_INSTALL_PRIVATE_DATADIR "@POCL_INSTALL_PRIVATE_DATADIR@"

#define POCL_INSTALL_PRIVATE_DATADIR_REL "@POCL_INSTALL_PRIVATE_DATADIR_REL@"

#cmakedefine POCL_USE_FAKE_ADDR_SPACE_IDS

#cmakedefine POCL_ASSERTS_BUILD

/* these are *host* values */

/* used in tce_common.c & pocl_llvm_api.cc  */
#define SRCDIR  "@SRCDIR@"

#cmakedefine TCEMC_AVAILABLE

#cmakedefine TCE_AVAILABLE

#define TCE_DEVICE_EXTENSIONS "@TCE_DEVICE_EXTENSIONS@"

/* Defined on big endian systems */
#define WORDS_BIGENDIAN @WORDS_BIGENDIAN@

/* Disable cl_khr_fp16 because fp16 is not supported */
#cmakedefine _CL_DISABLE_HALF

/* Disable cl_khr_fp64 because fp64 is not supported */
#cmakedefine _CL_DISABLE_DOUBLE

#define POCL_CL_VERSION "1.2"

#define HSA_DEVICE_CL_VERSION_MAJOR 1
#define HSA_DEVICE_CL_VERSION_MINOR 2

#define CUDA_DEVICE_CL_VERSION_MAJOR 1
#define CUDA_DEVICE_CL_VERSION_MINOR 2

#define HOST_DEVICE_CL_VERSION_MAJOR 1
#define HOST_DEVICE_CL_VERSION_MINOR 2

#define TCE_DEVICE_CL_VERSION_MAJOR 1
#define TCE_DEVICE_CL_VERSION_MINOR 2


#cmakedefine USE_POCL_MEMMANAGER
