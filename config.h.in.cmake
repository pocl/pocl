
#cmakedefine BUILD_HSA
#cmakedefine BUILD_CUDA
#cmakedefine BUILD_BASIC
#cmakedefine BUILD_TBB
#cmakedefine BUILD_PTHREAD
#cmakedefine BUILD_ACCEL
#cmakedefine BUILD_VULKAN

#cmakedefine BUILD_PROXY

#define BUILDDIR "@BUILDDIR@"

/* "Build with ICD" */
#cmakedefine BUILD_ICD

#define CMAKE_BUILD_TYPE "@CMAKE_BUILD_TYPE@"

#cmakedefine HAVE_CLSPV
#define CLSPV "@CLSPV@"
#define CLSPV_REFLECTION "@CLSPV_REFLECTION@"

#cmakedefine ENABLE_ASAN
#cmakedefine ENABLE_LSAN
#cmakedefine ENABLE_TSAN
#cmakedefine ENABLE_UBSAN

#cmakedefine ENABLE_EXTRA_VALIDITY_CHECKS

#cmakedefine ENABLE_CONFORMANCE

#cmakedefine ENABLE_HWLOC

#cmakedefine ENABLE_HOST_CPU_DEVICES

#cmakedefine ENABLE_POCL_BUILDING

#cmakedefine ENABLE_POCL_FLOAT_CONVERSION

#cmakedefine ENABLE_RELOCATION

#cmakedefine ENABLE_EGL_INTEROP
#cmakedefine ENABLE_OPENGL_INTEROP

#ifdef ENABLE_OPENGL_INTEROP
#cmakedefine ENABLE_CL_GET_GL_CONTEXT
#endif

#cmakedefine ENABLE_SLEEF

#cmakedefine ENABLE_SPIR

#cmakedefine ENABLE_SPIRV

#cmakedefine HAVE_DLFCN_H

#cmakedefine HAVE_FORK

#cmakedefine HAVE_VFORK

#cmakedefine HAVE_CLOCK_GETTIME

#cmakedefine HAVE_FDATASYNC

#cmakedefine HAVE_FSYNC

#cmakedefine HAVE_GETRLIMIT

#cmakedefine HAVE_MKOSTEMPS

#cmakedefine HAVE_MKSTEMPS

#cmakedefine HAVE_MKDTEMP

#cmakedefine HAVE_FUTIMENS

#cmakedefine HAVE_LTTNG_UST

#cmakedefine HAVE_OCL_ICD

#cmakedefine HAVE_POSIX_MEMALIGN

#cmakedefine HAVE_SLEEP

#cmakedefine HAVE_UTIME

#cmakedefine HAVE_VALGRIND

#cmakedefine ENABLE_LLVM

#cmakedefine ENABLE_LOADABLE_DRIVERS

/* this is used all over the runtime code */
#define HOST_CPU_CACHELINE_SIZE @HOST_CPU_CACHELINE_SIZE@

#if defined(BUILD_CUDA)

#define CUDA_DEVICE_EXTENSIONS "@CUDA_DEVICE_EXTENSIONS@"

#endif

#if defined(BUILD_BASIC) || defined(BUILD_PTHREAD)

#define HOST_AS_FLAGS  "@HOST_AS_FLAGS@"

#define HOST_CLANG_FLAGS  "@HOST_CLANG_FLAGS@"

#define HOST_DEVICE_EXTENSIONS "@HOST_DEVICE_EXTENSIONS@"

#cmakedefine HOST_CPU_FORCED

#define HOST_LD_FLAGS  "@HOST_LD_FLAGS@"

#define HOST_LLC_FLAGS  "@HOST_LLC_FLAGS@"

#cmakedefine HOST_FLOAT_SOFT_ABI

#endif

#define HOST_DEVICE_BUILD_HASH "@HOST_DEVICE_BUILD_HASH@"

#define DEFAULT_DEVICE_EXTENSIONS "@DEFAULT_DEVICE_EXTENSIONS@"

#ifdef BUILD_HSA

#cmakedefine HAVE_HSA_EXT_AMD_H

#define AMD_HSA @AMD_HSA@

#define HSA_DEVICE_EXTENSIONS "@HSA_DEVICE_EXTENSIONS@"

#define HSAIL_ASM "@HSAIL_ASM@"

#define HSAIL_ENABLED @HSAIL_ENABLED@

#endif


#define CMAKE_BUILD_TYPE "@CMAKE_BUILD_TYPE@"

#define LINK_COMMAND "@LINK_COMMAND@"






#ifdef ENABLE_LLVM

#define KERNELLIB_HOST_CPU_VARIANTS "@KERNELLIB_HOST_CPU_VARIANTS@"

#cmakedefine KERNELLIB_HOST_DISTRO_VARIANTS

#define CLANG "@CLANG@"

#define CLANG_RESOURCE_DIR "@CLANG_RESOURCE_DIR@"

#define CLANGXX "@CLANGXX@"

#define LLVM_LLC "@LLVM_LLC@"

#define LLVM_SPIRV "@LLVM_SPIRV@"

/* "Using LLVM 6.0" */
#cmakedefine LLVM_6_0

/* "Using LLVM 7.0" */
#cmakedefine LLVM_7_0

/* "Using LLVM 8.0" */
#cmakedefine LLVM_8_0

#cmakedefine LLVM_9_0

#cmakedefine LLVM_10_0

#cmakedefine LLVM_11_0

#cmakedefine LLVM_MAJOR @LLVM_VERSION_MAJOR@

#cmakedefine LLVM_BUILD_MODE_DEBUG

#ifndef LLVM_VERSION
#define LLVM_VERSION "@LLVM_VERSION_FULL@"
#endif


#endif



/* Defined to greatest expected alignment for extended types, in bytes. */
#define MAX_EXTENDED_ALIGNMENT @MAX_EXTENDED_ALIGNMENT@

#define PRINTF_BUFFER_SIZE @PRINTF_BUFFER_SIZE@


/* used in lib/CL/devices/basic */
#define OCL_KERNEL_TARGET  "@OCL_KERNEL_TARGET@"
#define OCL_KERNEL_TARGET_CPU  "@OCL_KERNEL_TARGET_CPU@"

#define POCL_VERSION_BASE "@POCL_VERSION_BASE@"
#define POCL_VERSION_FULL "@POCL_VERSION_FULL@"

#define POCL_KERNEL_CACHE_DEFAULT @POCL_KERNEL_CACHE_DEFAULT@

#define HOST_DEVICE_ADDRESS_BITS @HOST_DEVICE_ADDRESS_BITS@

#cmakedefine POCL_DEBUG_MESSAGES

#define POCL_INSTALL_PRIVATE_HEADER_DIR "@POCL_INSTALL_PRIVATE_HEADER_DIR@"

#define POCL_INSTALL_PRIVATE_DATADIR "@POCL_INSTALL_PRIVATE_DATADIR@"

#define POCL_INSTALL_PRIVATE_DATADIR_REL "@POCL_INSTALL_PRIVATE_DATADIR_REL@"

#define POCL_INSTALL_PRIVATE_LIBDIR "@POCL_INSTALL_PRIVATE_LIBDIR@"

#define POCL_INSTALL_PRIVATE_LIBDIR_REL "@POCL_INSTALL_PRIVATE_LIBDIR_REL@"

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

#define POCL_CL_VERSION "2.0"

#define HSA_DEVICE_CL_VERSION_MAJOR 1
#define HSA_DEVICE_CL_VERSION_MINOR 2

#define CUDA_DEVICE_CL_VERSION_MAJOR 1
#define CUDA_DEVICE_CL_VERSION_MINOR 2

#define HOST_DEVICE_CL_VERSION_MAJOR @HOST_DEVICE_CL_VERSION_MAJOR@
#define HOST_DEVICE_CL_VERSION_MINOR @HOST_DEVICE_CL_VERSION_MINOR@

#define TCE_DEVICE_CL_VERSION_MAJOR 1
#define TCE_DEVICE_CL_VERSION_MINOR 2


#cmakedefine USE_POCL_MEMMANAGER
