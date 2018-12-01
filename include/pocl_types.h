/* pocl_types.h - The basic OpenCL C device side scalar data types.

   Copyright (c) 2018 Pekka Jääskeläinen / Tampere University of Technology

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#if defined cl_khr_fp64 && !defined cl_khr_int64
#  error "cl_khr_fp64 requires cl_khr_int64"
#endif

#ifdef __CBUILD__

#ifndef __TCE__
/* Define the fixed width OpenCL data types using the target-specified ones
   from stdint.h. */
#include <stdint.h>

typedef uint8_t uchar;
typedef uint16_t ushort;
typedef uint32_t uint;
typedef uint64_t ulong;
typedef __SIZE_TYPE__ size_t;

#else

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned uint;
typedef unsigned ulong;
typedef unsigned int size_t;

#endif

#ifndef cl_khr_fp16
typedef short half;
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

#ifndef cl_khr_fp64
typedef struct error_undefined_type_double error_undefined_type_double;
#  define double error_undefined_type_double
#endif

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
