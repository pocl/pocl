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

/* The default implementation of subgroups is the simplest possible one of
   always having one subgroup executing the innermost dimension.

   Next, the plan is to allow the default to be changed explicitly by
   means of the intel_reqd_sub_group_size annotation as described in
   https://registry.khronos.org/OpenCL/extensions/intel/
   cl_intel_required_subgroup_size.html

   This forms a minimal viable feature set sufficient to emulate different
   warp sizes for CUDA/HIP execution. Performance via efficient vectorization
   is not a priority for now.
 */

#include <math.h>

/**
 * \brief Internal pseudo function which allocates space from the work-group
 * thread's stack (basically local memory) for each work-item.
 *
 * It's expanded in WorkitemLoops.cc to an alloca().
 *
 * @param element_size The size of an element to allocate (for all WIs in the
 * WG).
 * @param align The alignment of the start of chunk.
 * @return pointer to the allocated stack space (freed at unwind).
 */
void *__pocl_work_group_alloca (size_t element_size, size_t align);

/**
 * \brief Internal pseudo function which allocates space from the work-group
 * thread's stack (basically local memory).
 *
 * It's expanded in WorkitemLoops.cc to an alloca().
 *
 * @param bytes The size of data to allocate in bytes.
 * @param align The alignment of the start of chunk.
 * @return pointer to the allocated stack space (freed at unwind).
 */
void *__pocl_local_mem_alloca (size_t bytes, size_t align);

size_t _CL_OVERLOADABLE get_local_size (unsigned int dimindx);

size_t _CL_OVERLOADABLE get_local_id (unsigned int dimindx);

uint _CL_OVERLOADABLE
get_sub_group_size (void)
{
  /* By default 1 SG per WG_x. */
  return get_local_size (0);
}

uint _CL_OVERLOADABLE
get_max_sub_group_size (void)
{
  return get_sub_group_size ();
}

uint _CL_OVERLOADABLE
get_num_sub_groups (void)
{
  return (uint)get_local_size (1) * get_local_size (2);
}

uint _CL_OVERLOADABLE
get_enqueued_num_sub_groups (void)
{
  return 1;
}

uint _CL_OVERLOADABLE
get_sub_group_id (void)
{
  return get_local_id (2) * get_local_size (1) + get_local_id (1);
}

uint _CL_OVERLOADABLE
get_sub_group_local_id (void)
{
  return (uint)get_local_id (0);
}

void _CL_OVERLOADABLE sub_group_barrier (cl_mem_fence_flags flags);

#define SUB_GROUP_SHUFFLE_T(TYPE)                                             \
  __attribute__ ((always_inline)) TYPE _CL_OVERLOADABLE sub_group_shuffle (   \
      TYPE val, uint index)                                                   \
  {                                                                           \
    volatile TYPE *temp_storage                                               \
        = __pocl_work_group_alloca (sizeof (TYPE), sizeof (TYPE));            \
    temp_storage[get_sub_group_local_id ()] = val;                            \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    return temp_storage[index % get_sub_group_size ()];                       \
  }

SUB_GROUP_SHUFFLE_T (char)
SUB_GROUP_SHUFFLE_T (uchar)
SUB_GROUP_SHUFFLE_T (short)
SUB_GROUP_SHUFFLE_T (ushort)
SUB_GROUP_SHUFFLE_T (int)
SUB_GROUP_SHUFFLE_T (uint)
SUB_GROUP_SHUFFLE_T (long)
SUB_GROUP_SHUFFLE_T (ulong)
SUB_GROUP_SHUFFLE_T (float)
SUB_GROUP_SHUFFLE_T (double)

#define SUB_GROUP_SHUFFLE_XOR_T(TYPE)                                         \
  __attribute__ ((always_inline)) TYPE _CL_OVERLOADABLE                       \
  sub_group_shuffle_xor (TYPE val, uint mask)                                 \
  {                                                                           \
    volatile TYPE *temp_storage                                               \
        = __pocl_work_group_alloca (sizeof (TYPE), sizeof (TYPE));            \
    temp_storage[get_sub_group_local_id ()] = val;                            \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    return temp_storage[(get_sub_group_local_id () ^ mask)                    \
                        % get_sub_group_size ()];                             \
  }

SUB_GROUP_SHUFFLE_XOR_T (char)
SUB_GROUP_SHUFFLE_XOR_T (uchar)
SUB_GROUP_SHUFFLE_XOR_T (short)
SUB_GROUP_SHUFFLE_XOR_T (ushort)
SUB_GROUP_SHUFFLE_XOR_T (int)
SUB_GROUP_SHUFFLE_XOR_T (uint)
SUB_GROUP_SHUFFLE_XOR_T (long)
SUB_GROUP_SHUFFLE_XOR_T (ulong)
SUB_GROUP_SHUFFLE_XOR_T (float)
SUB_GROUP_SHUFFLE_XOR_T (double)

#define SUB_GROUP_BROADCAST_T(TYPE)                                           \
  __attribute__ ((always_inline)) TYPE _CL_OVERLOADABLE sub_group_broadcast ( \
      TYPE val, uint id)                                                      \
  {                                                                           \
    return sub_group_shuffle (val, id);                                       \
  }

SUB_GROUP_BROADCAST_T (int)
SUB_GROUP_BROADCAST_T (uint)
SUB_GROUP_BROADCAST_T (long)
SUB_GROUP_BROADCAST_T (ulong)
SUB_GROUP_BROADCAST_T (float)
SUB_GROUP_BROADCAST_T (double)

#define SUB_GROUP_REDUCE_OT(OPNAME, OPERATION, TYPE)                          \
  __attribute__ ((always_inline))                                             \
  TYPE _CL_OVERLOADABLE sub_group_reduce_##OPNAME (TYPE val)                  \
  {                                                                           \
    volatile TYPE *temp_storage                                               \
        = __pocl_work_group_alloca (sizeof (TYPE), sizeof (TYPE));            \
    temp_storage[get_sub_group_local_id ()] = val;                            \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    if (get_sub_group_local_id () == 0)                                       \
      {                                                                       \
        for (uint i = 1; i < get_sub_group_size (); ++i)                      \
          {                                                                   \
            TYPE a = temp_storage[0], b = temp_storage[i];                    \
            temp_storage[0] = OPERATION;                                      \
          }                                                                   \
      }                                                                       \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    return temp_storage[0];                                                   \
  }

#define SUB_GROUP_REDUCE_T(OPNAME, OPERATION)                                 \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, int)                                \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, uint)                               \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, long)                               \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, ulong)                              \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, float)                              \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, double)

SUB_GROUP_REDUCE_T (add, a + b)
SUB_GROUP_REDUCE_T (min, a > b ? b : a)
SUB_GROUP_REDUCE_T (max, a > b ? a : b)

#define SUB_GROUP_SCAN_INCLUSIVE_OT(OPNAME, OPERATION, TYPE)                  \
  __attribute__ ((always_inline))                                             \
  TYPE _CL_OVERLOADABLE sub_group_scan_inclusive_##OPNAME (TYPE val)          \
  {                                                                           \
    volatile TYPE *data                                                       \
        = __pocl_work_group_alloca (sizeof (TYPE), sizeof (TYPE));            \
    data[get_sub_group_local_id ()] = val;                                    \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    if (get_sub_group_local_id () == 0)                                       \
      {                                                                       \
        for (uint i = 1; i < get_sub_group_size (); ++i)                      \
          {                                                                   \
            TYPE a = data[i - 1], b = data[i];                                \
            data[i] = OPERATION;                                              \
          }                                                                   \
      }                                                                       \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    return data[get_sub_group_local_id ()];                                   \
  }

#define SUB_GROUP_SCAN_INCLUSIVE_T(OPNAME, OPERATION)                         \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, int)                        \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, uint)                       \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, long)                       \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, ulong)                      \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, float)                      \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, double)

SUB_GROUP_SCAN_INCLUSIVE_T (add, a + b)
SUB_GROUP_SCAN_INCLUSIVE_T (min, a > b ? b : a)
SUB_GROUP_SCAN_INCLUSIVE_T (max, a > b ? a : b)

#define SUB_GROUP_SCAN_EXCLUSIVE_OT(OPNAME, OPERATION, TYPE, ID)              \
  __attribute__ ((always_inline))                                             \
  TYPE _CL_OVERLOADABLE sub_group_scan_exclusive_##OPNAME (TYPE val)          \
  {                                                                           \
    volatile TYPE *data                                                       \
        = __pocl_work_group_alloca (sizeof (TYPE), sizeof (TYPE));            \
    data[get_sub_group_local_id () + 1] = val;                                \
    data[0] = ID;                                                             \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    if (get_sub_group_local_id () == 0)                                       \
      {                                                                       \
        for (uint i = 1; i < get_sub_group_size (); ++i)                      \
          {                                                                   \
            TYPE a = data[i - 1], b = data[i];                                \
            data[i] = OPERATION;                                              \
          }                                                                   \
      }                                                                       \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    return data[get_sub_group_local_id ()];                                   \
  }

SUB_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, int, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, uint, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, long, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, ulong, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, float, 0.0f)
SUB_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, double, 0.0)

SUB_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, int, INT_MAX)
SUB_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, uint, UINT_MAX)
SUB_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, long, LONG_MAX)
SUB_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, ulong, ULONG_MAX)
SUB_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, float, +INFINITY)
SUB_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, double, +INFINITY)

SUB_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, int, INT_MIN)
SUB_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, uint, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, long, LONG_MIN)
SUB_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, ulong, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, float, -INFINITY)
SUB_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, double, -INFINITY)

__attribute__ ((always_inline)) uint4 _CL_OVERLOADABLE
sub_group_ballot (int predicate)
{
  uint *flags = __pocl_local_mem_alloca (sizeof (uint) * 4, sizeof (uint) * 4);
  char *res = __pocl_work_group_alloca (sizeof (char), 4);
  if (get_sub_group_local_id () < 128)
    res[get_sub_group_local_id ()] = !!predicate;
  sub_group_barrier (CLK_LOCAL_MEM_FENCE);
  if (get_sub_group_local_id () == 0)
    {
      flags[0] = flags[1] = flags[2] = flags[3] = ~0;
      for (uint i = 0; i < get_sub_group_size () && i < 128; ++i)
        {
          flags[i / 32] |= res[i] << (i % 32);
        }
    }
  sub_group_barrier (CLK_LOCAL_MEM_FENCE);
  return *(uint4 *)flags;
}
