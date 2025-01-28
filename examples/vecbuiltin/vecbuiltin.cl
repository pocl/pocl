/* vecbuiltin - simple invocation of a builtins on global buffer of values

   Copyright (c) 2025 Michal Babej / Intel Finland Oy

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

kernel void
vecbuiltin (__global const float *a,
            __global const float *b,
            __global float *c)
{
  size_t gid = get_global_id (0);
  c[gid] = sin (a[gid]) + cos (b[gid]) + log2 (125.0f + a[gid])
           + pow (a[gid], 1.5f) + exp (a[gid]) + exp2 (b[gid]) + fabs (a[gid])
           + fma (a[gid], 4.0f, b[gid]) + fmax (a[gid], b[gid])
           + fmin (a[gid], b[gid]) + log10 (125.0f + a[gid])
           + log (125.0f + a[gid]) + rint (a[gid]) + round (b[gid])
           + sqrt (b[gid]) + ceil (b[gid]) + tan (a[gid]) + pown (a[gid], 4)
           + floor (a[gid]) + trunc (b[gid]);

  /* these exist as builtins but for some reason LLVM won't vectorize them: */
#if 0
  ldexp (a[gid], 4)
  + sinh (b[gid]) + cosh (b[gid]) + tanh (b[gid])
  + asin (b[gid]) + acos (b[gid]) + atan (b[gid])
  + exp10 (a[gid])
   + frexp (b[gid])
#endif

  /* these are vectorized, but they're expressions
   * converted to other operations: */
#if 0
  + isfinite (a[gid]) + isinf (b[gid]) + isnan (a[gid]) + isnormal (b[gid]);
#endif
}
