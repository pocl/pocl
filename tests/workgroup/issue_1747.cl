/* Tests a conditional barrier miscompilation case reported in #1747.
   Doesn't reproduce from OpenCL C, it seems.

   Copyright (c) 2025 Pekka Jääskeläinen / Intel Finland Oy

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

__kernel void
test_kernel (global int *out)
{
  /* The problem doesn't occur with x-dim. */
  size_t idx = get_global_id(0);
  out[idx] = 0;
  if (idx < 2)
    {
      /* If we launch this with multiple work-groups with size 2 each,
         the first WG should branch in and synch with the barrier, but
         the other one not -- should be fine! */
      barrier (CLK_GLOBAL_MEM_FENCE);
      out[idx] = 1;
    }
}
