/* for_bug - reproduces a tail replication bug exposed by URNG of AMD SDK when
   adding a barrier to the loop

   Copyright (c) 2013 Pekka Jääskeläinen / TUT
   
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



__kernel void
test_kernel (void)
{
  int gid_x = get_global_id (0);
  int k = 0;
  int i;
  volatile int foo[15000];

/* This bug reproduces only if the last 'if' in the loop
   writes to a memory, thus cannot be converted to a select. 

   This produces a crash with 'repl' and an infinite loop with 'wiloops'. 

   It is caused by a loop structure where there are two paths to the
   latch block which decrements the iteration variable. The first path
   skips the last if, the second executes it. This confuses the
   barrier tail replication.
*/

  for (i = 16; i > 0; i--) {
      barrier(CLK_LOCAL_MEM_FENCE);
      printf ("gid_x %u after barrier at iteration %d\n", gid_x, i);
      k += gid_x;
      if(i < 15)
          foo[i] = k*160 * gid_x;
  }
  /* If it did not crash and the program does not go to an inifinite
     loop, assume OK. */
  if (gid_x == 0)
      printf("OK\n");
}
