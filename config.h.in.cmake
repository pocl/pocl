
/* The normal alignment of `double16', in bytes. */
#cmakedefine ALIGNOF_DOUBLE16 @ALIGNOF_DOUBLE16@

/* The normal alignment of `float16', in bytes. */
#cmakedefine ALIGNOF_FLOAT16 @ALIGNOF_FLOAT16@


#cmakedefine BUILD_SPU

#cmakedefine BUILDDIR "@BUILDDIR@"

/* "Build with ICD" */
#cmakedefine BUILD_ICD



#cmakedefine CLANG "@CLANG@"

/* clang++ executable */
#cmakedefine CLANGXX "@CLANGXX@"

/* clang++ compiler flags */
/* TODO in sources */
#cmakedefine KERNEL_CLANGXX_FLAGS "@KERNEL_CLANGXX_FLAGS@"

/* "Using a SPIR generator Clang from Khronos." */
#cmakedefine CLANG_SPIR


/* TODO in sources */
#cmakedefine KERNEL_CL_FLAGS  "@KERNEL_CL_FLAGS@"


/* "Use a custom buffer allocator" */
#cmakedefine CUSTOM_BUFFER_ALLOCATOR



#cmakedefine DIRECT_LINKAGE



#cmakedefine FORCED_CLFLAGS  "@FORCED_CLFLAGS@"



#cmakedefine HAVE_OCL_ICD





/* Defined if posix_memalign is available. */
#cmakedefine HAVE_POSIX_MEMALIGN


#define HOST  "@HOST@"

#define HOST_AS_FLAGS  "@HOST_AS_FLAGS@"

#define HOST_CLANG_FLAGS  "@HOST_CLANG_FLAGS@"

#define HOST_CPU  "@HOST_CPU@"

#define HOST_LD_FLAGS  "@HOST_LD_FLAGS@"

#define HOST_LLC_FLAGS  "@HOST_LLC_FLAGS@"

#cmakedefine HOST_FLOAT_SOFT_ABI

#cmakedefine LLC "@LLC@"


/* "Using LLVM 3.2" */
#cmakedefine LLVM_3_2

/* "Using LLVM 3.3" */
#cmakedefine LLVM_3_3

/* "Using LLVM 3.4" */
#cmakedefine LLVM_3_4

/* "Using upcoming LLVM 3.5" */
#cmakedefine LLVM_3_5



/* Defined to greatest expected alignment for extended types, in bytes. */
#cmakedefine MAX_EXTENDED_ALIGNMENT @MAX_EXTENDED_ALIGNMENT@



/* used in lib/CL/devices/basic */
#cmakedefine OCL_KERNEL_TARGET  "@OCL_KERNEL_TARGET@"
#cmakedefine OCL_KERNEL_TARGET_CPU  "@OCL_KERNEL_TARGET_CPU@"





#cmakedefine POCL_DEVICE_ADDRESS_BITS @POCL_DEVICE_ADDRESS_BITS@

#cmakedefine PACKAGE_VERSION "@PACKAGE_VERSION@"

#cmakedefine POCL_INSTALL_PRIVATE_HEADER_DIR "@POCL_INSTALL_PRIVATE_HEADER_DIR@"

#cmakedefine POCL_INSTALL_PRIVATE_DATADIR "@POCL_INSTALL_PRIVATE_DATADIR@"

/* these are *host* values */

/* The size of `__fp16', as computed by sizeof. */
#cmakedefine SIZEOF___FP16  @SIZEOF___FP16@



/* used in tce_common.c & pocl_llvm_api.cc  */
#cmakedefine SRCDIR  "@SRCDIR@"





#cmakedefine TCEMC_AVAILABLE

#cmakedefine TCE_AVAILABLE


/* "Use vecmathlib if available for the target." */
#cmakedefine USE_VECMATHLIB


/* Defined on big endian systems */
#cmakedefine WORDS_BIGENDIAN

/* Disable cl_khr_int64 when a clang bug is present */
#cmakedefine _CL_DISABLE_LONG

/* Disable cl_khr_fp16 because fp16 is not supported */
#cmakedefine _CL_DISABLE_HALF
