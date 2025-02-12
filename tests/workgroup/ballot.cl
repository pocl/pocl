/* ballot.cl - Reproduce the simple Ballot test case from chipStar.

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

/* There was a bug with subgroup size initialization during the 6.1 development
   cycle that affected the SG id computations. */

/* This currently relies on the SG size = local size X for the size.
   TODO: Use the required subgroup size. */
kernel void
test_kernel (global int *output)
{
  volatile int foo = 0xBADF00D1;
  int vote = (foo >> get_sub_group_local_id ()) & 1;
  uint4 bar = sub_group_ballot (vote);
#if 0
  printf("lid %d sgid %d sglid %d sgsize %d vote %d gllid %d\n",
         get_local_id(0), get_sub_group_id(),
         get_sub_group_local_id(), get_sub_group_size(), vote,
         get_local_linear_id());
#endif
  if (get_local_id (0) == 0)
    {
      output[0] = bar.x;
      output[1] = bar.y;
      output[2] = bar.z;
      output[3] = bar.w;
    }
  else
    output[get_local_id (0)] = 0;
}
