// Scalar type definitions

#if defined cl_khr_fp64 && !defined cl_khr_int64
#  error "cl_khr_fp64 requires cl_khr_int64"
#endif

#ifdef __CBUILD__

/* Define the required OpenCL C fixed size types in a C compatible way.
   The C-based kernel library implementations need these. Avoid using
   stdint.h types as it might not be available for Clang during
   kernel compilation. */
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long long ulong;

typedef __SIZE_TYPE__ size_t;

#endif


/* Disable undefined datatypes */

/* The definitions below intentionally lead to errors if these types
   are used when they are not available in the language. This prevents
   accidentally using them if the compiler does not disable these
   types, but only e.g. defines them with an incorrect size.*/

#ifndef cl_khr_int64
typedef struct error_undefined_type_long error_undefined_type_long;
#  define long error_undefined_type_long
typedef struct error_undefined_type_ulong error_undefined_type_ulong;
#  define ulong error_undefined_type_ulong
#endif

#ifdef __CBUILD__
#ifndef cl_khr_fp16
typedef short half;
#endif
#endif


#ifndef cl_khr_fp64
typedef struct error_undefined_type_double error_undefined_type_double;
#  define double error_undefined_type_double
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
