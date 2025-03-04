/* OpenCL built-in library: subgroup basic functionality

   Copyright (c) 2022-2024 Pekka Jääskeläinen / Intel Finland Oy

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

/* The default implementation of subgroups for CPU drivers. It uses work-group
   sized local buffers for exchanging the data. The subgroup size is by default
   the local X dimension side, unless restricted with the
   intel_reqd_sub_group_size metadata.
 */

#include "templates.h"
#include "work_group_alloca.h"

/* Magic variable that is expanded in Workgroup.cc */
extern uint _pocl_sub_group_size;

size_t _CL_OVERLOADABLE get_local_id (unsigned int dimindx);
size_t _CL_OVERLOADABLE get_local_linear_id (void);
size_t _CL_OVERLOADABLE get_local_size (unsigned int dimindx);

void _CL_OVERLOADABLE _CL_CONVERGENT work_group_barrier(cl_mem_fence_flags);
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

uint _CL_OVERLOADABLE _CL_CONVERGENT sub_group_reduce_max(uint );
int _CL_OVERLOADABLE
sub_group_any (int predicate)
{
  return sub_group_reduce_max ((unsigned)predicate);
}

uint _CL_OVERLOADABLE _CL_CONVERGENT sub_group_reduce_min(uint );
int _CL_OVERLOADABLE
sub_group_all (int predicate)
{
  return sub_group_reduce_min ((unsigned)predicate);
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
  return (uint)get_local_size (0) * get_local_size (1) * get_local_size (2)
         / get_max_sub_group_size ();
}

uint _CL_OVERLOADABLE
get_enqueued_num_sub_groups (void)
{
  return 1;
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

uint4 _CL_OVERLOADABLE
sub_group_ballot (int predicate)
{
  /* The results for the ballot for all of the WG's SGs. */
  uint4 *flags = __pocl_work_group_alloca (sizeof (uint4), sizeof (uint4), 0);
  /* Temporary storage for the predicate flags of all WIs in the WG. */
  char *res = __pocl_work_group_alloca (sizeof (char), 4, 0);
  res[get_local_linear_id ()] = !!predicate;
  sub_group_barrier (CLK_LOCAL_MEM_FENCE);
  if (get_sub_group_local_id () == 0)
    {
      flags[get_sub_group_id ()] = 0;
      uint *f = (uint *)(flags + get_sub_group_id ());
      for (uint i = 0; i < get_sub_group_size () && i < 128; ++i)
        {
          f[i / 32] |= res[get_first_llid () + i] << (i % 32);
        }
    }
  sub_group_barrier (CLK_LOCAL_MEM_FENCE);
  return flags[get_sub_group_id ()];
}

#define SUB_GROUP_SHUFFLE_PT(PREFIX, TYPE)                                    \
  TYPE _CL_OVERLOADABLE PREFIX##sub_group_shuffle (TYPE val, uint index)      \
  {                                                                           \
    volatile TYPE *temp_storage                                               \
      = __pocl_work_group_alloca (sizeof (TYPE), sizeof (TYPE), 0);           \
    temp_storage[get_local_linear_id ()] = val;                               \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    return temp_storage[get_first_llid () + index % get_sub_group_size ()];   \
  }

/* Define both the non-prefixed (khr) and Intel-prefixed shuffles. */
#define SUB_GROUP_SHUFFLE_T(TYPE)                                             \
  SUB_GROUP_SHUFFLE_PT (, TYPE)                                               \
  SUB_GROUP_SHUFFLE_PT (intel_, TYPE)

SUB_GROUP_SHUFFLE_T (char)
SUB_GROUP_SHUFFLE_T (uchar)
SUB_GROUP_SHUFFLE_T (short)
SUB_GROUP_SHUFFLE_T (ushort)
SUB_GROUP_SHUFFLE_T (int)
SUB_GROUP_SHUFFLE_T (uint)
SUB_GROUP_SHUFFLE_T (long)
SUB_GROUP_SHUFFLE_T (ulong)
__IF_FP16 (SUB_GROUP_SHUFFLE_T (half))
SUB_GROUP_SHUFFLE_T (float)
__IF_FP64 (SUB_GROUP_SHUFFLE_T (double))

#ifdef cl_intel_subgroups
#define SUB_GROUP_SHUFFLE_VEC(TYPE)                                           \
  SUB_GROUP_SHUFFLE_PT (intel_, TYPE##2)                                      \
  SUB_GROUP_SHUFFLE_PT (intel_, TYPE##3)                                      \
  SUB_GROUP_SHUFFLE_PT (intel_, TYPE##4)                                      \
  SUB_GROUP_SHUFFLE_PT (intel_, TYPE##8)                                      \
  SUB_GROUP_SHUFFLE_PT (intel_, TYPE##16)

SUB_GROUP_SHUFFLE_VEC (float)
SUB_GROUP_SHUFFLE_VEC (int)
SUB_GROUP_SHUFFLE_VEC (uint)
#endif

#define SUB_GROUP_SHUFFLE_XOR_PT(PREFIX, TYPE)                                \
  TYPE _CL_OVERLOADABLE PREFIX##sub_group_shuffle_xor (TYPE val, uint mask)   \
  {                                                                           \
    volatile TYPE *temp_storage                                               \
      = __pocl_work_group_alloca (sizeof (TYPE), sizeof (TYPE), 0);           \
    temp_storage[get_local_linear_id ()] = val;                               \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    return temp_storage[get_first_llid ()                                     \
                        + (get_sub_group_local_id () ^ mask)                  \
                            % get_sub_group_size ()];                         \
  }

/* Define both the non-prefixed (khr) and Intel-prefixed shuffles. */
#define SUB_GROUP_SHUFFLE_XOR_T(TYPE)                                         \
  SUB_GROUP_SHUFFLE_XOR_PT (, TYPE)                                           \
  SUB_GROUP_SHUFFLE_XOR_PT (intel_, TYPE)

SUB_GROUP_SHUFFLE_XOR_T (char)
SUB_GROUP_SHUFFLE_XOR_T (uchar)
SUB_GROUP_SHUFFLE_XOR_T (short)
SUB_GROUP_SHUFFLE_XOR_T (ushort)
SUB_GROUP_SHUFFLE_XOR_T (int)
SUB_GROUP_SHUFFLE_XOR_T (uint)
SUB_GROUP_SHUFFLE_XOR_T (long)
SUB_GROUP_SHUFFLE_XOR_T (ulong)
__IF_FP16 (SUB_GROUP_SHUFFLE_XOR_T (half))
SUB_GROUP_SHUFFLE_XOR_T (float)
__IF_FP64 (SUB_GROUP_SHUFFLE_XOR_T (double))

#ifdef cl_intel_subgroups
#define SUB_GROUP_SHUFFLE_XOR_VEC(TYPE)                                       \
  SUB_GROUP_SHUFFLE_XOR_PT (intel_, TYPE##2)                                  \
  SUB_GROUP_SHUFFLE_XOR_PT (intel_, TYPE##3)                                  \
  SUB_GROUP_SHUFFLE_XOR_PT (intel_, TYPE##4)                                  \
  SUB_GROUP_SHUFFLE_XOR_PT (intel_, TYPE##8)                                  \
  SUB_GROUP_SHUFFLE_XOR_PT (intel_, TYPE##16)

SUB_GROUP_SHUFFLE_XOR_VEC (float)
SUB_GROUP_SHUFFLE_XOR_VEC (int)
SUB_GROUP_SHUFFLE_XOR_VEC (uint)
#endif

#define SUB_GROUP_BROADCAST_T(TYPE)                                           \
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

#define SUB_GROUP_REDUCE_OT(OPNAME, OPERATION, TYPE)                          \
  TYPE _CL_OVERLOADABLE sub_group_reduce##OPNAME (TYPE val)                   \
  {                                                                           \
    volatile TYPE *temp_storage                                               \
      = __pocl_work_group_alloca (sizeof (TYPE), sizeof (TYPE), 0);           \
    temp_storage[get_local_linear_id ()] = val;                               \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    if (get_sub_group_local_id () == 0)                                       \
      {                                                                       \
        for (uint i = 1; i < get_sub_group_size (); ++i)                      \
          {                                                                   \
            TYPE a = temp_storage[get_first_llid ()],                         \
                 b = temp_storage[get_first_llid () + i];                     \
            temp_storage[get_first_llid ()] = OPERATION;                      \
          }                                                                   \
      }                                                                       \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    return temp_storage[get_first_llid ()];                                   \
  }

#define SUB_GROUP_REDUCE_T(OPNAME, OPERATION)                                 \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, int)                                \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, uint)                               \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, long)                               \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, ulong)                              \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, float)                              \
  __IF_FP16 (SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, half))                   \
  __IF_FP64 (SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, double))

SUB_GROUP_REDUCE_T (_add, (a + b))
SUB_GROUP_REDUCE_T (_min, (a > b ? b : a))
SUB_GROUP_REDUCE_T (_max, (a > b ? a : b))

#define SUB_GROUP_SCAN_INCLUSIVE_OT(OPNAME, OPERATION, TYPE)                  \
  TYPE _CL_OVERLOADABLE sub_group_scan_inclusive##OPNAME (TYPE val)           \
  {                                                                           \
    volatile TYPE *data                                                       \
      = __pocl_work_group_alloca (sizeof (TYPE), sizeof (TYPE), 0);           \
    data[get_local_linear_id ()] = val;                                       \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    if (get_sub_group_local_id () == 0)                                       \
      {                                                                       \
        for (uint i = 1; i < get_sub_group_size (); ++i)                      \
          {                                                                   \
            TYPE a = data[get_first_llid () + i - 1],                         \
                 b = data[get_first_llid () + i];                             \
            data[get_first_llid () + i] = OPERATION;                          \
          }                                                                   \
      }                                                                       \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    return data[get_local_linear_id ()];                                      \
  }

#define SUB_GROUP_SCAN_INCLUSIVE_T(OPNAME, OPERATION)                         \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, int)                        \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, uint)                       \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, long)                       \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, ulong)                      \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, float)                      \
  __IF_FP16 (SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, half))           \
  __IF_FP64 (SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, double))

SUB_GROUP_SCAN_INCLUSIVE_T (_add, (a + b))
SUB_GROUP_SCAN_INCLUSIVE_T (_min, (a > b ? b : a))
SUB_GROUP_SCAN_INCLUSIVE_T (_max, (a > b ? a : b))

#define SUB_GROUP_SCAN_EXCLUSIVE_OT(OPNAME, OPERATION, TYPE, ID)              \
  TYPE _CL_OVERLOADABLE sub_group_scan_exclusive##OPNAME (TYPE val)           \
  {                                                                           \
    volatile TYPE *data = __pocl_work_group_alloca (                          \
      sizeof (TYPE), sizeof (TYPE), sizeof (TYPE));                           \
    data[get_local_linear_id () + 1] = val;                                   \
    data[get_first_llid ()] = ID;                                             \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    if (get_sub_group_local_id () == 0)                                       \
      {                                                                       \
        for (uint i = 1; i < get_sub_group_size (); ++i)                      \
          {                                                                   \
            TYPE a = data[get_first_llid () + i - 1],                         \
                 b = data[get_first_llid () + i];                             \
            data[get_first_llid () + i] = OPERATION;                          \
          }                                                                   \
      }                                                                       \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    return data[get_local_linear_id ()];                                      \
  }

SUB_GROUP_SCAN_EXCLUSIVE_OT (_add, a + b, int, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_add, a + b, uint, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_add, a + b, long, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_add, a + b, ulong, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_add, a + b, float, 0.0f)
__IF_FP16 (SUB_GROUP_SCAN_EXCLUSIVE_OT (_add, a + b, half, 0))
__IF_FP64 (SUB_GROUP_SCAN_EXCLUSIVE_OT (_add, a + b, double, 0))

SUB_GROUP_SCAN_EXCLUSIVE_OT (_min, a > b ? b : a, int, INT_MAX)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_min, a > b ? b : a, uint, UINT_MAX)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_min, a > b ? b : a, long, LONG_MAX)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_min, a > b ? b : a, ulong, ULONG_MAX)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_min, a > b ? b : a, float, +INFINITY)
__IF_FP16 (
  SUB_GROUP_SCAN_EXCLUSIVE_OT (_min, a > b ? b : a, half, (half)(+INFINITY)))
__IF_FP64 (SUB_GROUP_SCAN_EXCLUSIVE_OT (
  _min, a > b ? b : a, double, (double)(+INFINITY)))

SUB_GROUP_SCAN_EXCLUSIVE_OT (_max, a > b ? a : b, int, INT_MIN)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_max, a > b ? a : b, uint, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_max, a > b ? a : b, long, LONG_MIN)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_max, a > b ? a : b, ulong, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (_max, a > b ? a : b, float, -INFINITY)
__IF_FP16 (
  SUB_GROUP_SCAN_EXCLUSIVE_OT (_max, a > b ? a : b, half, (half)(-INFINITY)))
__IF_FP64 (SUB_GROUP_SCAN_EXCLUSIVE_OT (
  _max, a > b ? a : b, double, (double)(-INFINITY)))

#ifdef cl_intel_subgroups

#define INTEL_SG_SHUFFLE_DOWN_T(TYPE)                                         \
  TYPE _CL_OVERLOADABLE intel_sub_group_shuffle_down (TYPE current,           \
                                                      TYPE next, uint delta)  \
  {                                                                           \
    uint idx = get_sub_group_local_id () + delta;                             \
    uint cur_idx = (idx >= get_max_sub_group_size ()) ? 0 : idx;              \
    TYPE cur_val = intel_sub_group_shuffle (current, cur_idx);                \
    uint next_idx                                                             \
      = (idx > get_max_sub_group_size ()) ? idx - get_sub_group_size () : 0;  \
    TYPE next_val = intel_sub_group_shuffle (next, next_idx);                 \
    return idx >= get_sub_group_size () ? next_val : cur_val;                 \
  }

INTEL_SG_SHUFFLE_DOWN_T (char)
INTEL_SG_SHUFFLE_DOWN_T (uchar)
INTEL_SG_SHUFFLE_DOWN_T (short)
INTEL_SG_SHUFFLE_DOWN_T (ushort)
INTEL_SG_SHUFFLE_DOWN_T(uint)
INTEL_SG_SHUFFLE_DOWN_T(int)
INTEL_SG_SHUFFLE_DOWN_T(float)
__IF_INT64 (INTEL_SG_SHUFFLE_DOWN_T(long)
            INTEL_SG_SHUFFLE_DOWN_T(ulong))
__IF_FP16 (INTEL_SG_SHUFFLE_DOWN_T(half))
__IF_FP64 (INTEL_SG_SHUFFLE_DOWN_T(double))

#define INTEL_SG_SHUFFLE_DOWN_VEC(TYPE)                                       \
  INTEL_SG_SHUFFLE_DOWN_T (TYPE##2)                                           \
  INTEL_SG_SHUFFLE_DOWN_T (TYPE##3)                                           \
  INTEL_SG_SHUFFLE_DOWN_T (TYPE##4)                                           \
  INTEL_SG_SHUFFLE_DOWN_T (TYPE##8)                                           \
  INTEL_SG_SHUFFLE_DOWN_T (TYPE##16)

INTEL_SG_SHUFFLE_DOWN_VEC (float)
INTEL_SG_SHUFFLE_DOWN_VEC (int)
INTEL_SG_SHUFFLE_DOWN_VEC (uint)

#define INTEL_SG_SHUFFLE_UP_T(TYPE)                                           \
  TYPE _CL_OVERLOADABLE intel_sub_group_shuffle_up (TYPE previous,            \
                                                    TYPE current, uint delta) \
  {                                                                           \
    uint idx = get_sub_group_local_id () - delta;                             \
    uint cur_idx = (idx < 0) ? 0 : idx;                                       \
    TYPE cur_val = intel_sub_group_shuffle (current, cur_idx);                \
    uint prev_idx = (idx < 0) ? idx + get_max_sub_group_size () : 0;          \
    TYPE prev_val = intel_sub_group_shuffle (previous, prev_idx);             \
    return (idx < 0) ? prev_val : cur_val;                                    \
  }

INTEL_SG_SHUFFLE_UP_T (char)
INTEL_SG_SHUFFLE_UP_T (uchar)
INTEL_SG_SHUFFLE_UP_T (short)
INTEL_SG_SHUFFLE_UP_T (ushort)
INTEL_SG_SHUFFLE_UP_T(uint)
INTEL_SG_SHUFFLE_UP_T(int)
INTEL_SG_SHUFFLE_UP_T(float)
__IF_INT64 (INTEL_SG_SHUFFLE_UP_T(long)
            INTEL_SG_SHUFFLE_UP_T(ulong))
__IF_FP16 (INTEL_SG_SHUFFLE_UP_T(half))
__IF_FP64 (INTEL_SG_SHUFFLE_UP_T(double))

#define INTEL_SG_SHUFFLE_UP_VEC(TYPE)                                         \
  INTEL_SG_SHUFFLE_UP_T (TYPE##2)                                             \
  INTEL_SG_SHUFFLE_UP_T (TYPE##3)                                             \
  INTEL_SG_SHUFFLE_UP_T (TYPE##4)                                             \
  INTEL_SG_SHUFFLE_UP_T (TYPE##8)                                             \
  INTEL_SG_SHUFFLE_UP_T (TYPE##16)

INTEL_SG_SHUFFLE_UP_VEC (float)
INTEL_SG_SHUFFLE_UP_VEC (int)
INTEL_SG_SHUFFLE_UP_VEC (uint)

#define INTEL_SG_BLOCK_READ_WRITE_T(TYPE, SUFFIX)                             \
  TYPE _CL_OVERLOADABLE intel_sub_group_block_read##SUFFIX (                  \
    const global TYPE *p)                                                     \
  {                                                                           \
    return p[get_sub_group_local_id ()];                                      \
  }                                                                           \
                                                                              \
  TYPE##2 _CL_OVERLOADABLE intel_sub_group_block_read##SUFFIX##2(             \
    const global TYPE *p)                                                     \
  {                                                                           \
    return (TYPE##2) (                                                        \
      p[get_sub_group_local_id ()],                                           \
      p[get_sub_group_local_id () + get_max_sub_group_size ()]);              \
  }                                                                           \
                                                                              \
  TYPE##4 _CL_OVERLOADABLE intel_sub_group_block_read##SUFFIX##4(             \
    const global TYPE *p)                                                     \
  {                                                                           \
    uint sglid = get_sub_group_local_id ();                                   \
    uint sgsize = get_max_sub_group_size ();                                  \
    return (TYPE##4) (p[sglid], p[sglid + sgsize], p[sglid + 2 * sgsize],     \
                      p[sglid + 3 * sgsize]);                                 \
  }                                                                           \
                                                                              \
  TYPE##8 _CL_OVERLOADABLE intel_sub_group_block_read##SUFFIX##8(             \
    const global TYPE *p)                                                     \
  {                                                                           \
    uint sglid = get_sub_group_local_id ();                                   \
    uint sgsize = get_max_sub_group_size ();                                  \
    return (TYPE##8) (p[sglid], p[sglid + sgsize], p[sglid + 2 * sgsize],     \
                      p[sglid + 3 * sgsize], p[sglid + 4 * sgsize],           \
                      p[sglid + 5 * sgsize], p[sglid + 6 * sgsize],           \
                      p[sglid + 7 * sgsize]);                                 \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE intel_sub_group_block_write##SUFFIX (global TYPE *p,  \
                                                             TYPE data)       \
  {                                                                           \
    uint sglid = get_sub_group_local_id ();                                   \
    p[sglid] = data;                                                          \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE intel_sub_group_block_write##SUFFIX##2(               \
    global TYPE * p, TYPE##2 data)                                            \
  {                                                                           \
    uint sglid = get_sub_group_local_id ();                                   \
    uint sgsize = get_max_sub_group_size ();                                  \
    p[sglid] = data.x;                                                        \
    p[sglid + sgsize] = data.y;                                               \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE intel_sub_group_block_write##SUFFIX##4(               \
    global TYPE * p, TYPE##4 data)                                            \
  {                                                                           \
    uint sglid = get_sub_group_local_id ();                                   \
    uint sgsize = get_max_sub_group_size ();                                  \
    p[sglid] = data.s0;                                                       \
    p[sglid + sgsize] = data.s1;                                              \
    p[sglid + 2 * sgsize] = data.s2;                                          \
    p[sglid + 3 * sgsize] = data.s3;                                          \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE intel_sub_group_block_write##SUFFIX##8(               \
    global TYPE * p, TYPE##8 data)                                            \
  {                                                                           \
    uint sglid = get_sub_group_local_id ();                                   \
    uint sgsize = get_max_sub_group_size ();                                  \
    p[sglid] = data.s0;                                                       \
    p[sglid + sgsize] = data.s1;                                              \
    p[sglid + 2 * sgsize] = data.s2;                                          \
    p[sglid + 3 * sgsize] = data.s3;                                          \
    p[sglid + 4 * sgsize] = data.s4;                                          \
    p[sglid + 5 * sgsize] = data.s5;                                          \
    p[sglid + 6 * sgsize] = data.s6;                                          \
    p[sglid + 7 * sgsize] = data.s7;                                          \
  }

#define INTEL_SG_BLOCK_READ_WRITE_T_16(TYPE, SUFFIX)                          \
  TYPE##16 _CL_OVERLOADABLE intel_sub_group_block_read##SUFFIX##16(           \
    const global TYPE *p)                                                     \
  {                                                                           \
    uint sglid = get_sub_group_local_id ();                                   \
    uint sgsize = get_max_sub_group_size ();                                  \
    return (TYPE##16) (                                                       \
      p[sglid], p[sglid + sgsize], p[sglid + 2 * sgsize],                     \
      p[sglid + 3 * sgsize], p[sglid + 4 * sgsize], p[sglid + 5 * sgsize],    \
      p[sglid + 6 * sgsize], p[sglid + 7 * sgsize], p[sglid + 8 * sgsize],    \
      p[sglid + 9 * sgsize], p[sglid + 10 * sgsize], p[sglid + 11 * sgsize],  \
      p[sglid + 12 * sgsize], p[sglid + 13 * sgsize], p[sglid + 14 * sgsize], \
      p[sglid + 15 * sgsize]);                                                \
  }                                                                           \
  void _CL_OVERLOADABLE intel_sub_group_block_write##SUFFIX##16(              \
    global TYPE * p, TYPE##16 data)                                           \
  {                                                                           \
    uint sglid = get_sub_group_local_id ();                                   \
    uint sgsize = get_max_sub_group_size ();                                  \
    p[sglid] = data.s0;                                                       \
    p[sglid + sgsize] = data.s1;                                              \
    p[sglid + 2 * sgsize] = data.s2;                                          \
    p[sglid + 3 * sgsize] = data.s3;                                          \
    p[sglid + 4 * sgsize] = data.s4;                                          \
    p[sglid + 5 * sgsize] = data.s5;                                          \
    p[sglid + 6 * sgsize] = data.s6;                                          \
    p[sglid + 7 * sgsize] = data.s7;                                          \
    p[sglid + 8 * sgsize] = data.s8;                                          \
    p[sglid + 9 * sgsize] = data.s9;                                          \
    p[sglid + 10 * sgsize] = data.sA;                                         \
    p[sglid + 11 * sgsize] = data.sB;                                         \
    p[sglid + 12 * sgsize] = data.sC;                                         \
    p[sglid + 13 * sgsize] = data.sD;                                         \
    p[sglid + 14 * sgsize] = data.sE;                                         \
    p[sglid + 15 * sgsize] = data.sF;                                         \
  }

INTEL_SG_BLOCK_READ_WRITE_T (uint, )

#ifdef cl_intel_subgroups_short
/* https://registry.khronos.org/OpenCL/extensions/intel/
 * cl_intel_subgroups_short.html
 */

INTEL_SG_BLOCK_READ_WRITE_T (ushort, _us)
#endif

#ifdef cl_intel_subgroups_char
/* https://registry.khronos.org/OpenCL/extensions/intel/
 * cl_intel_subgroups_char.html
 */

INTEL_SG_BLOCK_READ_WRITE_T (uchar, _uc)
INTEL_SG_BLOCK_READ_WRITE_T_16 (uchar, _uc)
#endif

#if defined(cl_intel_subgroups_short) || defined(cl_intel_subgroups_char)
INTEL_SG_BLOCK_READ_WRITE_T (uint, _ui)
#endif

#endif
