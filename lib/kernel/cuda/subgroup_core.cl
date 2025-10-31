/* OpenCL built-in library: subgroup core functions

   Copyright (c) 2025 Jan Solanti / Tampere University

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

/* Magic variable that is linked in from pocl-ptx-gen.c */
extern uint _pocl_sub_group_size;

size_t _CL_OVERLOADABLE get_local_id (unsigned int dimindx);
size_t _CL_OVERLOADABLE get_local_linear_id (void);
size_t _CL_OVERLOADABLE get_local_size (unsigned int dimindx);

void _CL_OVERLOADABLE
sub_group_barrier (cl_mem_fence_flags flags, memory_scope scope)
  __attribute__ ((noduplicate))
{
  sub_group_barrier (flags);
}

uint _CL_OVERLOADABLE
get_sub_group_size (void)
{
  return _pocl_sub_group_size;
}

uint _CL_OVERLOADABLE
get_max_sub_group_size (void)
{
  return get_sub_group_size ();
}

uint _CL_OVERLOADABLE
get_num_sub_groups (void)
{
  uint tmp = get_local_size (0) * get_local_size (1) * get_local_size (2);
  uint rem = tmp % get_max_sub_group_size () == 0 ? 0 : 1;
  return (tmp / get_max_sub_group_size ()) + rem;
}

uint _CL_OVERLOADABLE
get_enqueued_num_sub_groups (void)
{
  return get_num_sub_groups ();
}

uint _CL_OVERLOADABLE
get_sub_group_id (void)
{
  return (uint)get_local_linear_id () / get_max_sub_group_size ();
}

uint _CL_OVERLOADABLE
get_sub_group_local_id (void)
{
  return (uint)get_local_linear_id () % get_max_sub_group_size ();
}

static size_t _CL_OVERLOADABLE
get_first_llid (void)
{
  return get_sub_group_id () * get_max_sub_group_size ();
}

#define SUB_GROUP_BROADCAST_T(TYPE)                                           \
  TYPE _CL_OVERLOADABLE sub_group_shuffle (TYPE, uint);                       \
  TYPE _CL_OVERLOADABLE sub_group_broadcast (TYPE val, uint id)               \
  {                                                                           \
    return sub_group_shuffle (val, id);                                       \
  }

SUB_GROUP_BROADCAST_T (int)
SUB_GROUP_BROADCAST_T (uint)
SUB_GROUP_BROADCAST_T (long)
SUB_GROUP_BROADCAST_T (ulong)
__IF_FP16 (SUB_GROUP_BROADCAST_T (half))
SUB_GROUP_BROADCAST_T (float)
__IF_FP64 (SUB_GROUP_BROADCAST_T (double))
