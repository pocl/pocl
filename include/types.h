// Basic definitions

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

// #if __SIZEOF_POINTER__ == 8
// typedef ulong size_t;
// typedef long ptrdiff_t;
// typedef long intptr_t;
// typedef ulong uintptr_t;
// #elif __SIZEOF_POINTER__ == 4
// typedef uint size_t;
// typedef int ptrdiff_t;
// typedef int intptr_t;
// typedef uint uintptr_t;
// #else
// #  error "Pointer size not known"
// #endif

typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __INTPTR_TYPE__ intptr_t;
typedef unsigned __INTPTR_TYPE__ uintptr_t;



// Note: These definitions below may differ between languages, since
// OpenCL and C/C++ may use different sizes for "long". Be careful
// when including this file from C or C++ kernel library source files.

// LLVM always supports fp16 (aka half)
#define cl_khr_fp16

// Is long supported?
#if __SIZEOF_LONG__ == 8
#  define cles_khr_int64
#else
#  undef cles_khr_int64
#endif

// Is double supported?
#if defined cles_khr_int64 && __SIZEOF_DOUBLE__ == 8
#  define cl_khr_fp64
#else
#  undef cl_khr_fp64
#endif
