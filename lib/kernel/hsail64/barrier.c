/* OpenCL built-in library: HSAIL barrier

   Copyright (c) 2015 Pekka Jääskeläinen of
                      Tampere University of Technology

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

/* TODO: the noduplicate attribute for the builtin declarations
   doesn't seem to propagate to LLVM IR. Might need to define this
   in .ll due to this. */
void __builtin_hsail_memfence (int, int) __attribute__((noduplicate));
void __builtin_barrier () __attribute__((noduplicate));

void _Z7barrierj(int flags)
{
/* HSA specs say that the barrier is only for cflow and there's a need for mem
   fences to implement the OpenCL barrier() semantics.

    2 = sequentially consistent acquire, 3 = release

    3 = work-group (local) scope, 4 = agent (global) scope , 5 = system (~SVM) scope)
*/

  /* Release fence */
  if (flags & CLK_GLOBAL_MEM_FENCE)
    __builtin_hsail_memfence (3, 4);
  else
    __builtin_hsail_memfence (3, 3);

/* Looking at test/CodeGen/HSAIL/llvm.hsail.barrier.ll the magic
   number 34 seems to denote ALL WI synching. */
  __builtin_hsail_barrier (34);

  /* Acquire fence */
  if (flags & CLK_GLOBAL_MEM_FENCE)
    __builtin_hsail_memfence (2, 4);
  else
    __builtin_hsail_memfence (2, 3);
}
