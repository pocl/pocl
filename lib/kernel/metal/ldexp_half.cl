/* OpenCL built-in library: ldexp() for half, for the Metal (AIR) target

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

/* The (halfN, int) overloads of ldexp; core-math/promotedf16.cl has the
   others. Scaling in float is exact for every result that is representable
   in half, so the conversion back rounds once. */

#ifdef cl_khr_fp16

#define LDEXP_HALF_SCALAR_EXP(N)                                              \
  half##N _CL_OVERLOADABLE ldexp (half##N a, int b)                           \
  {                                                                           \
    return convert_half##N (ldexp (convert_float##N (a), b));                 \
  }

LDEXP_HALF_SCALAR_EXP (2)
LDEXP_HALF_SCALAR_EXP (3)
LDEXP_HALF_SCALAR_EXP (4)
LDEXP_HALF_SCALAR_EXP (8)
LDEXP_HALF_SCALAR_EXP (16)

#endif
