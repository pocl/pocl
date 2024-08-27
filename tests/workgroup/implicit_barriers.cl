/* implicit_barriers.cl -
   Tests the different places where the kernel compiler should inject implicit
   barriers to produce nicer formed parallel regions for the loop vectorizer.

   Copyright (c) 2013 Pekka Jääskeläinen / TUT
                 2024 Pekka Jääskeläinen / Intel Finland Oy

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
test_kernel (void)
{
  unsigned group_id = get_group_id (0);
  unsigned local_id = get_local_id (0);
  local int buf[512];
  buf[local_id] = get_local_size(0) - local_id;
  barrier(CLK_LOCAL_MEM_FENCE);

  printf ("LOCAL_ID=%d before if\n", local_id);
  /* This case tries to mimic mri-q / Parboil where the compiler inserts
     a shortcut check because it's not clear if the for loop iterates even
     once. The uniform if tries to model that check. */
  if (group_id == 0) /* uniform check */
  {
    /* Uniform branch. After implicit-loop-barriers, this if will have
       conditional barriers inside (to isolate the loop). Then
       implicit-cond-barriers should inject a barrier just after the branch
       to minimize the "peeling effects". */

    /* [implicit barrier expected here to minimize peeling due to cond.barrier]
     */

    printf ("LOCAL_ID=%d inside if\n", local_id);

    volatile int a = 0; /* volatile to ensure it won't be unrolled */
    int index = 0;      /* to produce additional phi node */

    /* [implicit barrier expected here to enforce a nicely formed inner loop]
     */
    /* this loop should have PHIs left intact as it's not a b-loop */
    do
      {
        printf ("LOCAL_ID=%d inside for, iteration %d, value %d\n", local_id,
                a, buf[index]);
        ++index;
        ++a;
      }
    while (a < 2);
    /* [implicit barrier expected here to isolate the non-barrier loop] */
  }
  printf ("LOCAL_ID=%d after if\n", local_id);
}
