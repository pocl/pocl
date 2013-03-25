/* OpenCL built-in library: nan()

   Copyright (c) 2011 Erik Schnetter <eschnetter@perimeterinstitute.ca>
                      Perimeter Institute for Theoretical Physics
   
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

#include "templates.h"

#define FLT_MANT_MASK ((1U  << (uint) FLT_MANT_DIG) - 1U )
#define DBL_MANT_MASK ((1UL << (ulong)DBL_MANT_DIG) - 1UL)

/* Fall-back implementation which ignores the nancode */
// DEFINE_EXPR_V_U(nan, (vtype)(0.0/0.0))

/* LLVM's __builtin_nan expects a char* argument, so we need to roll
   our own. Note this assume IEEE bit layout. */
// #define __builtin_nanf(nancode)                                         \
//   ((~FLT_MANT_MASK & as_uint(NANF)) | (FLT_MANT_MASK & nancode))
// #define __builtin_nan(nancode)                                          \
//   ((~DBL_MANT_MASK & as_ulong(NAN)) | (DBL_MANT_MASK & nancode))

// DEFINE_BUILTIN_V_U(nan)

/* This is faster than the above because it is vectorised */
#ifdef cl_khr_fp64
DEFINE_EXPR_V_U(nan,
                ({
                  utype nanbits =
                    sizeof(stype)==4 /* float  */ ? ((utype)(~FLT_MANT_MASK & as_uint ((float) NAN)) | ((utype)FLT_MANT_MASK & a)) :
                    sizeof(stype)==8 /* double */ ? ((utype)(~DBL_MANT_MASK & as_ulong((double)NAN)) | ((utype)DBL_MANT_MASK & a)) :
                    (utype)0;
                  *(vtype*)&nanbits;
                }))
#else
DEFINE_EXPR_V_U(nan,
                ({
                  utype nanbits =
                    sizeof(stype)==4 /* float  */ ? ((utype)(~FLT_MANT_MASK & as_uint ((float) NAN)) | ((utype)FLT_MANT_MASK & a)) :
                    (utype)0;
                  *(vtype*)&nanbits;
                }))
#endif
