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

uint2 _CL_OVERLOADABLE
intel_sub_group_block_read2 (const global uint *p)
{
  return (uint2)(p[get_sub_group_local_id ()],
                 p[get_sub_group_local_id () + get_max_sub_group_size ()]);
}

uint8 _CL_OVERLOADABLE
intel_sub_group_block_read8 (const global uint *p)
{
  uint sglid = get_sub_group_local_id ();
  uint sgsize = get_max_sub_group_size ();
  return (uint8)(p[sglid], p[sglid + sgsize], p[sglid + 2 * sgsize],
                 p[sglid + 3 * sgsize], p[sglid + 4 * sgsize],
                 p[sglid + 5 * sgsize], p[sglid + 6 * sgsize],
                 p[sglid + 7 * sgsize]);
}

#ifdef cl_intel_subgroups
/* https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups_short.html
 */
ushort8 _CL_OVERLOADABLE
intel_sub_group_block_read_us8 (const global ushort *p)
{
  uint sglid = get_sub_group_local_id ();
  uint sgsize = get_max_sub_group_size ();
  return (ushort8)(p[sglid], p[sglid + sgsize], p[sglid + 2 * sgsize],
                   p[sglid + 3 * sgsize], p[sglid + 4 * sgsize],
                   p[sglid + 5 * sgsize], p[sglid + 6 * sgsize],
                   p[sglid + 7 * sgsize]);
}

uint _CL_OVERLOADABLE
intel_sub_group_shuffle_down (uint current, uint next, uint delta)
{
  int idx = get_sub_group_local_id () + delta;
  uint cur_idx = (idx >= get_max_sub_group_size ()) ? 0 : idx;
  uint other_cur = sub_group_shuffle (current, cur_idx);
  int next_idx
      = (idx > get_max_sub_group_size ()) ? idx - get_sub_group_size () : 0;
  uint other_next = sub_group_shuffle (next, next_idx);
  return idx >= get_sub_group_size () ? other_cur : other_next;
}
#endif
