/* OpenCL built-in library: work-group collective functions

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

#include "work_group_alloca.h"

#define INFINITY (__builtin_inf())

size_t _CL_OVERLOADABLE get_local_id (unsigned int dimindx);
size_t _CL_OVERLOADABLE get_local_linear_id (void);
size_t _CL_OVERLOADABLE get_local_size (unsigned int dimindx);
void _CL_OVERLOADABLE
    POCL_BUILTIN_PREFIX (work_group_barrier) (cl_mem_fence_flags flags);

#define work_group_barrier POCL_BUILTIN_PREFIX (work_group_barrier)

/* Align the stack temporary data by this multiple to facilitate easier
   vectorization. */
#define ALIGN_ELEMENT_MULTIPLE 32

static size_t
get_total_local_size ()
{
  return get_local_size (0) * get_local_size (1) * get_local_size (2);
}

#define WORK_GROUP_SHUFFLE_PT(PREFIX, TYPE)                                   \
  __attribute__ ((always_inline)) static TYPE _CL_OVERLOADABLE                \
      PREFIX##work_group_shuffle (TYPE val, size_t id)                        \
  {                                                                           \
    volatile TYPE *temp_storage = __pocl_work_group_alloca (                  \
        sizeof (TYPE), ALIGN_ELEMENT_MULTIPLE * sizeof (TYPE), 0);            \
    temp_storage[get_local_linear_id ()] = val;                               \
    work_group_barrier (CLK_LOCAL_MEM_FENCE);                                 \
    return temp_storage[id % get_total_local_size ()];                        \
  }

/* Define both the non-prefixed (khr) and Intel-prefixed shuffles. */
#define WORK_GROUP_SHUFFLE_T(TYPE) WORK_GROUP_SHUFFLE_PT (, TYPE)

WORK_GROUP_SHUFFLE_T (char)
WORK_GROUP_SHUFFLE_T (uchar)
WORK_GROUP_SHUFFLE_T (short)
WORK_GROUP_SHUFFLE_T (ushort)
WORK_GROUP_SHUFFLE_T (int)
WORK_GROUP_SHUFFLE_T (uint)
WORK_GROUP_SHUFFLE_T (long)
WORK_GROUP_SHUFFLE_T (ulong)
WORK_GROUP_SHUFFLE_T (float)
WORK_GROUP_SHUFFLE_T (double)

#define WORK_GROUP_BROADCAST_T(TYPE)                                          \
  __attribute__ ((always_inline)) TYPE _CL_OVERLOADABLE                       \
  work_group_broadcast (TYPE val, size_t x)                                   \
  {                                                                           \
    return work_group_shuffle (val, x);                                       \
  }                                                                           \
  __attribute__ ((always_inline)) TYPE _CL_OVERLOADABLE                       \
  work_group_broadcast (TYPE val, size_t x, size_t y)                         \
  {                                                                           \
    return work_group_shuffle (val, y * get_local_size (0) + x);              \
  }                                                                           \
  __attribute__ ((always_inline)) TYPE _CL_OVERLOADABLE                       \
  work_group_broadcast (TYPE val, size_t x, size_t y, size_t z)               \
  {                                                                           \
    return work_group_shuffle (val,                                           \
                               z * get_local_size (1) * get_local_size (0)    \
                                   + y * get_local_size (0) + x);             \
  }

WORK_GROUP_BROADCAST_T (int)
WORK_GROUP_BROADCAST_T (uint)
WORK_GROUP_BROADCAST_T (long)
WORK_GROUP_BROADCAST_T (ulong)
WORK_GROUP_BROADCAST_T (float)
WORK_GROUP_BROADCAST_T (double)

#define WORK_GROUP_REDUCE_OT(OPNAME, OPERATION, TYPE)                         \
  __attribute__ ((always_inline))                                             \
  TYPE _CL_OVERLOADABLE work_group_reduce_##OPNAME (TYPE val)                 \
  {                                                                           \
    volatile TYPE *temp_storage = __pocl_work_group_alloca (                  \
        sizeof (TYPE), ALIGN_ELEMENT_MULTIPLE * sizeof (TYPE), 0);            \
    temp_storage[get_local_linear_id ()] = val;                               \
    work_group_barrier (CLK_LOCAL_MEM_FENCE);                                 \
    if (get_local_linear_id () == 0)                                          \
      {                                                                       \
        for (uint i = 1; i < get_total_local_size (); ++i)                    \
          {                                                                   \
            TYPE a = temp_storage[0], b = temp_storage[i];                    \
            temp_storage[0] = OPERATION;                                      \
          }                                                                   \
      }                                                                       \
    work_group_barrier (CLK_LOCAL_MEM_FENCE);                                 \
    return temp_storage[0];                                                   \
  }

#define WORK_GROUP_REDUCE_T(OPNAME, OPERATION)                                \
  WORK_GROUP_REDUCE_OT (OPNAME, OPERATION, int)                               \
  WORK_GROUP_REDUCE_OT (OPNAME, OPERATION, uint)                              \
  WORK_GROUP_REDUCE_OT (OPNAME, OPERATION, long)                              \
  WORK_GROUP_REDUCE_OT (OPNAME, OPERATION, ulong)                             \
  WORK_GROUP_REDUCE_OT (OPNAME, OPERATION, float)                             \
  WORK_GROUP_REDUCE_OT (OPNAME, OPERATION, double)

WORK_GROUP_REDUCE_T (add, a + b)
WORK_GROUP_REDUCE_T (min, a > b ? b : a)
WORK_GROUP_REDUCE_T (max, a > b ? a : b)

#define WORK_GROUP_SCAN_INCLUSIVE_OT(OPNAME, OPERATION, TYPE)                 \
  __attribute__ ((always_inline))                                             \
  TYPE _CL_OVERLOADABLE work_group_scan_inclusive_##OPNAME (TYPE val)         \
  {                                                                           \
    volatile TYPE *data = __pocl_work_group_alloca (                          \
        sizeof (TYPE), ALIGN_ELEMENT_MULTIPLE * sizeof (TYPE), 0);            \
    data[get_local_linear_id ()] = val;                                       \
    work_group_barrier (CLK_LOCAL_MEM_FENCE);                                 \
    if (get_local_linear_id () == 0)                                          \
      {                                                                       \
        for (uint i = 1; i < get_total_local_size (); ++i)                    \
          {                                                                   \
            TYPE a = data[i - 1], b = data[i];                                \
            data[i] = OPERATION;                                              \
          }                                                                   \
      }                                                                       \
    work_group_barrier (CLK_LOCAL_MEM_FENCE);                                 \
    return data[get_local_linear_id ()];                                      \
  }

#define WORK_GROUP_SCAN_INCLUSIVE_T(OPNAME, OPERATION)                        \
  WORK_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, int)                       \
  WORK_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, uint)                      \
  WORK_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, long)                      \
  WORK_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, ulong)                     \
  WORK_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, float)                     \
  WORK_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, double)

WORK_GROUP_SCAN_INCLUSIVE_T (add, a + b)
WORK_GROUP_SCAN_INCLUSIVE_T (min, a > b ? b : a)
WORK_GROUP_SCAN_INCLUSIVE_T (max, a > b ? a : b)

#define WORK_GROUP_SCAN_EXCLUSIVE_OT(OPNAME, OPERATION, TYPE, ID)             \
  __attribute__ ((always_inline))                                             \
  TYPE _CL_OVERLOADABLE work_group_scan_exclusive_##OPNAME (TYPE val)         \
  {                                                                           \
    volatile TYPE *data = __pocl_work_group_alloca (                          \
        sizeof (TYPE), ALIGN_ELEMENT_MULTIPLE * sizeof (TYPE),                \
        sizeof (TYPE));                                                       \
    data[get_local_linear_id () + 1] = val;                                   \
    data[0] = ID;                                                             \
    work_group_barrier (CLK_LOCAL_MEM_FENCE);                                 \
    if (get_local_linear_id () == 0)                                          \
      {                                                                       \
        for (uint i = 1; i < get_total_local_size (); ++i)                    \
          {                                                                   \
            TYPE a = data[i - 1], b = data[i];                                \
            data[i] = OPERATION;                                              \
          }                                                                   \
      }                                                                       \
    work_group_barrier (CLK_LOCAL_MEM_FENCE);                                 \
    return data[get_local_linear_id ()];                                      \
  }

WORK_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, int, 0)
WORK_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, uint, 0)
WORK_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, long, 0)
WORK_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, ulong, 0)
WORK_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, float, 0.0f)
WORK_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, double, 0.0)

WORK_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, int, INT_MAX)
WORK_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, uint, UINT_MAX)
WORK_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, long, LONG_MAX)
WORK_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, ulong, ULONG_MAX)
WORK_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, float, +INFINITY)
WORK_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, double, +INFINITY)

WORK_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, int, INT_MIN)
WORK_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, uint, 0)
WORK_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, long, LONG_MIN)
WORK_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, ulong, 0)
WORK_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, float, -INFINITY)
WORK_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, double, -INFINITY)

__attribute__ ((always_inline)) int _CL_OVERLOADABLE
work_group_any (int predicate)
{
  /* The results for all of the WIs. */
  int *flags = __pocl_work_group_alloca (
      sizeof (int), ALIGN_ELEMENT_MULTIPLE * sizeof (int), 0);
  /* The final result. */
  flags[get_local_linear_id ()] = !!predicate;
  int *result = __pocl_work_group_alloca (sizeof (int), sizeof (int), 0);
  work_group_barrier (CLK_LOCAL_MEM_FENCE);
  if (get_local_linear_id () == 0)
    {
      *result = 0;
      for (uint i = 0; i < get_total_local_size (); ++i)
        *result |= flags[i];
    }
  work_group_barrier (CLK_LOCAL_MEM_FENCE);
  return *result;
}

__attribute__ ((always_inline)) int _CL_OVERLOADABLE
work_group_all (int predicate)
{
  return !work_group_any (!!predicate);
}
