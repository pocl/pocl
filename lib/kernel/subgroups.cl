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

#define INTEL_SG_SHUFFLE_DOWN_T(TYPE)                                         \
  TYPE _CL_OVERLOADABLE intel_sub_group_shuffle_down (TYPE current,           \
  TYPE next, uint delta)                                                      \
  {                                                                           \
    uint idx = get_sub_group_local_id () + delta;                             \
    uint cur_idx = (idx >= get_max_sub_group_size ()) ? 0 : idx;              \
    TYPE cur_val = sub_group_shuffle (current, cur_idx);                      \
    uint next_idx                                                             \
         = (idx > get_max_sub_group_size()) ? idx - get_sub_group_size () : 0;\
    TYPE next_val = sub_group_shuffle (next, next_idx);                       \
    return idx >= get_sub_group_size () ? next_val : cur_val;                 \
  }

INTEL_SG_SHUFFLE_DOWN_T(uint)
INTEL_SG_SHUFFLE_DOWN_T(int)
INTEL_SG_SHUFFLE_DOWN_T(float)
__IF_INT64 (INTEL_SG_SHUFFLE_DOWN_T(long)
            INTEL_SG_SHUFFLE_DOWN_T(ulong))
__IF_FP16 (INTEL_SG_SHUFFLE_DOWN_T(half))
__IF_FP64 (INTEL_SG_SHUFFLE_DOWN_T(double))

#define INTEL_SG_SHUFFLE_UP_T(TYPE)                                           \
  TYPE _CL_OVERLOADABLE intel_sub_group_shuffle_up (TYPE previous,            \
  TYPE current, uint delta)                                                   \
  {                                                                           \
    uint idx = get_sub_group_local_id () - delta;                             \
    uint cur_idx = (idx < 0) ? 0 : idx;                                       \
    TYPE cur_val = sub_group_shuffle(current, cur_idx);                       \
    uint prev_idx = (idx < 0) ? idx + get_max_sub_group_size () : 0;          \
    TYPE prev_val = sub_group_shuffle (previous, prev_idx);                   \
    return (idx < 0) ? prev_val : cur_val;                                    \
  }

INTEL_SG_SHUFFLE_UP_T(uint)
INTEL_SG_SHUFFLE_UP_T(int)
INTEL_SG_SHUFFLE_UP_T(float)
__IF_INT64 (INTEL_SG_SHUFFLE_UP_T(long)
            INTEL_SG_SHUFFLE_UP_T(ulong))
__IF_FP16 (INTEL_SG_SHUFFLE_UP_T(half))
__IF_FP64 (INTEL_SG_SHUFFLE_UP_T(double))

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
#endif

#if defined(cl_intel_subgroups_short) || defined(cl_intel_subgroups_char)
INTEL_SG_BLOCK_READ_WRITE_T (uint, _ui)
#endif

#endif
