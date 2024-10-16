/* do_while_with_barrier.cl -
   A reproducer for a WI-loop formation issue reduced from
   conformance/src/conformance/test_conformance/basic/test_local.cpp

   Copyright (c) 2024 Pekka Jääskeläinen / Intel Finland Oy

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

#define printf(__X, __Y) // printf (__X, __Y)

__kernel void
test_kernel (__global int *output)
{
  int tid = get_local_id (0);
  int lsize = get_local_size(0);

  output[tid] = 0;
  if (lsize == 1)
    if (tid == 0) {
      output[0] = 1;
      return;
    }

  do {
    barrier(CLK_LOCAL_MEM_FENCE);
    lsize /= 2;
  } while (lsize);

  /* Note, the tid == 0 check must be the same as in the early
     return. There's likely some unexpected preopt messing the CFG up,
     converting it to what the WI-loop formation cannot handle.*/
  if (tid == 0) {
    output[0] = 1;
    /* It gets stuck here. */
    printf ("lid %lu\n", get_local_id(0));
  }
}
