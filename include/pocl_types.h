// Scalar type definitions

#if defined cl_khr_fp64 && !defined cl_khr_int64
#  error "cl_khr_fp64 requires cl_khr_int64"
#endif

#ifdef __CBUILD__

#ifndef _TCE__
/* Define the fixed width OpenCL data types using the target-specified ones
   from stdint.h. */
#include <stdint.h>

typedef uint8_t uchar;
typedef uint16_t ushort;
typedef uint32_t uint;

#ifdef __TCE__
/* TCE's 64b support and proper stdint.h is a WiP. Workarounds here. */
typedef unsigned ulong;
typedef unsigned int size_t;
/* #define double float  */

#else
typedef uint64_t ulong;
typedef __SIZE_TYPE__ size_t;
#endif

#else

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned uint;
typedef unsigned ulong;
typedef unsigned int size_t;

#endif

#endif



/* Disable undefined datatypes */

/* The definitions below intentionally lead to errors if these types
   are used when they are not available in the language. This prevents
   accidentally using them if the compiler does not disable these
   types, but only e.g. defines them with an incorrect size.*/

#ifndef __CBUILD__

#ifndef cl_khr_int64
typedef struct error_undefined_type_long error_undefined_type_long;
#  define long error_undefined_type_long
typedef struct error_undefined_type_ulong error_undefined_type_ulong;
#  define ulong error_undefined_type_ulong
#endif

#endif

#ifdef __CBUILD__
#ifndef cl_khr_fp16
typedef short half;
#endif
#endif

#ifndef __CBUILD__
#ifndef cl_khr_fp64
typedef struct error_undefined_type_double error_undefined_type_double;
#  define double error_undefined_type_double
#endif
#endif

#ifndef __CBUILD__

/* Define unsigned datatypes */

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
#ifdef cl_khr_int64
typedef unsigned long ulong;
#endif

/* Define pointer helper types */

typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef ptrdiff_t intptr_t;
typedef size_t uintptr_t;

#endif
