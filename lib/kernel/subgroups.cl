/* OpenCL built-in library: subgroup basic functionality

   Copyright (c) 2022-2023 Pekka Jääskeläinen / Intel Finland Oy

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

/* See subgroups.c for further documentation. */

void _CL_OVERLOADABLE
sub_group_barrier (cl_mem_fence_flags flags)
{
  /* This should work as long as there are no diverging
     subgroups -- right? It models all subgroups of the WG
     stepping in lockstep. */
  work_group_barrier (flags);
}

void _CL_OVERLOADABLE
sub_group_barrier (cl_mem_fence_flags flags, memory_scope scope)
    __attribute__ ((noduplicate))
{
  work_group_barrier (flags);
}

void _CL_OVERLOADABLE
sub_group_barrier (memory_scope scope)
    __attribute__ ((noduplicate))
{
  work_group_barrier (CLK_GLOBAL_MEM_FENCE);
}



int _CL_OVERLOADABLE
sub_group_any (int predicate)
{
  return sub_group_reduce_max ((unsigned)predicate);
}

int _CL_OVERLOADABLE
sub_group_all (int predicate)
{
  return sub_group_reduce_min ((unsigned)predicate);
}
#ifdef cl_intel_subgroups
uint _CL_OVERLOADABLE
intel_sub_group_shuffle_down(uint current, uint next, uint delta) {
  int idx = get_sub_group_local_id() + delta;
  uint cur_idx = (idx >= get_max_sub_group_size()) ? 0 : idx;
  uint other_cur = sub_group_shuffle(current, cur_idx);
  int next_idx = (idx > get_max_sub_group_size()) ? idx - get_sub_group_size() : 0;
  uint other_next = sub_group_shuffle(next, next_idx);
  return idx >= get_sub_group_size() ? other_cur : other_next;
}
#endif
