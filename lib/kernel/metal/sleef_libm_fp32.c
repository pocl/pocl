/* OpenCL built-in library: float libm functions for the Metal (AIR) target

   Copyright (c) 2026 PoCL developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

/* The generic kernel library implements these with Clang builtins, which
   become calls to the C library functions. There is no C library on Metal,
   so provide them from SLEEF's portable scalar single-precision code.

   That code incidentally uses double (unsuffixed literals, which the Metal
   library compiles with -cl-single-precision-constant, and some helpers with
   double parameters), which Apple GPUs lack: make it float. */

/* (pocl_types.h poisons double when cl_khr_fp64 is not defined) */
#undef double
#define PURE_C
#define DORENAME
#include "misc.h"
#define double float
#include "sleefsp.c"

float cbrtf (float x) { return xcbrtf_u1 (x); }
float erff (float x) { return xerff_u1 (x); }
float erfcf (float x) { return xerfcf_u15 (x); }
float lgammaf (float x) { return xlgammaf_u1 (x); }
float tgammaf (float x) { return xtgammaf_u1 (x); }
float expm1f (float x) { return xexpm1f (x); }
float log1pf (float x) { return xlog1pf (x); }
float hypotf (float x, float y) { return xhypotf_u05 (x, y); }
float nextafterf (float x, float y) { return xnextafterf (x, y); }
int ilogbf (float x) { return xilogbf (x); }
