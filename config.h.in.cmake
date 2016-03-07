
/* The normal alignment of `double16', in bytes. */
#define ALIGNOF_DOUBLE16 @ALIGNOF_DOUBLE16@

/* The normal alignment of `float16', in bytes. */
#define ALIGNOF_FLOAT16 @ALIGNOF_FLOAT16@

#cmakedefine BUILD_HSA

#define POCL_BUILT_WITH_CMAKE

#define BUILDDIR "@BUILDDIR@"

/* "Build with ICD" */
#cmakedefine BUILD_ICD

#ifndef LLVM_VERSION
#define LLVM_VERSION "@LLVM_VERSION_FULL@"
#endif

#define CLANG "@CLANG@"

/* clang++ executable */
#define CLANGXX "@CLANGXX@"

#define HSAIL_ASM "@HSAIL_ASM@"

/* clang++ compiler flags */
/* TODO in sources */
#define KERNEL_CLANGXX_FLAGS "@KERNEL_CLANGXX_FLAGS@"

/* "Using a SPIR generator Clang from Khronos." */
#cmakedefine CLANG_SPIR


/* TODO in sources */
#define KERNEL_CL_FLAGS  "@KERNEL_CL_FLAGS@"


#cmakedefine DIRECT_LINKAGE


#define FORCED_CLFLAGS  "@FORCED_CLFLAGS@"

#cmakedefine HAVE_FORK

#cmakedefine HAVE_VFORK

#cmakedefine HAVE_CLOCK_GETTIME

#cmakedefine HAVE_LTTNG_UST

#cmakedefine HAVE_OCL_ICD

/* Defined if posix_memalign is available. */
#cmakedefine HAVE_POSIX_MEMALIGN

#cmakedefine HAVE_HSA_EXT_AMD_H


#define HOST  "@HOST@"

#define HOST_AS_FLAGS  "@HOST_AS_FLAGS@"

#define HOST_CLANG_FLAGS  "@HOST_CLANG_FLAGS@"

#define HOST_DEVICE_EXTENSIONS "@HOST_DEVICE_EXTENSIONS@"

#define HOST_CPU  "@HOST_CPU@"

#define HOST_LD_FLAGS  "@HOST_LD_FLAGS@"

#define HOST_LLC_FLAGS  "@HOST_LLC_FLAGS@"

#cmakedefine HOST_FLOAT_SOFT_ABI

#define HSA_DEVICE_EXTENSIONS "@HSA_DEVICE_EXTENSIONS@"


#define KERNELLIB_HOST_CPU_VARIANTS "@KERNELLIB_HOST_CPU_VARIANTS@"

#cmakedefine KERNELLIB_HOST_DISTRO_VARIANTS

#define LLVM_LLC "@LLVM_LLC@"


/* "Using LLVM 3.7" */
#cmakedefine LLVM_3_7

/* "Using LLVM 3.8" */
#cmakedefine LLVM_3_8

#cmakedefine LLVM_3_9



/* Defined to greatest expected alignment for extended types, in bytes. */
#define MAX_EXTENDED_ALIGNMENT @MAX_EXTENDED_ALIGNMENT@



/* used in lib/CL/devices/basic */
#define OCL_KERNEL_TARGET  "@OCL_KERNEL_TARGET@"
#define OCL_KERNEL_TARGET_CPU  "@OCL_KERNEL_TARGET_CPU@"


#define PACKAGE_VERSION "@PACKAGE_VERSION@"


#define POCL_KERNEL_CACHE_DEFAULT @POCL_KERNEL_CACHE_DEFAULT@

#define POCL_DEVICE_ADDRESS_BITS @POCL_DEVICE_ADDRESS_BITS@

#cmakedefine POCL_DEBUG_MESSAGES

#define POCL_INSTALL_PRIVATE_HEADER_DIR "@POCL_INSTALL_PRIVATE_HEADER_DIR@"

#define POCL_INSTALL_PRIVATE_DATADIR "@POCL_INSTALL_PRIVATE_DATADIR@"

/* these are *host* values */

/* The size of `__fp16', as computed by sizeof. */
#define SIZEOF___FP16  @SIZEOF___FP16@



/* used in tce_common.c & pocl_llvm_api.cc  */
#define SRCDIR  "@SRCDIR@"





#cmakedefine TCEMC_AVAILABLE

#cmakedefine TCE_AVAILABLE

#define TCE_DEVICE_EXTENSIONS "@TCE_DEVICE_EXTENSIONS@"

/* "Use vecmathlib if available for the target." */
#cmakedefine USE_VECMATHLIB


/* Defined on big endian systems */
#define WORDS_BIGENDIAN @WORDS_BIGENDIAN@

/* Disable cl_khr_int64 when a clang bug is present */
#cmakedefine _CL_DISABLE_LONG

/* Disable cl_khr_fp16 because fp16 is not supported */
#cmakedefine _CL_DISABLE_HALF

#define POCL_CL_VERSION "2.0"

#define HSA_DEVICE_CL_VERSION_MAJOR 2
#define HSA_DEVICE_CL_VERSION_MINOR 0

#define HOST_DEVICE_CL_VERSION_MAJOR 2
#define HOST_DEVICE_CL_VERSION_MINOR 0

#define TCE_DEVICE_CL_VERSION_MAJOR 1
#define TCE_DEVICE_CL_VERSION_MINOR 2
