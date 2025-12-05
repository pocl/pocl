/* OpenCL built-in library: subgroup functions for >=SM70 hardware

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

#define FULL_MASK 0xFFFFFFFF

int _CL_OVERLOADABLE
sub_group_all (int predicate)
{
  return __nvvm_vote_all_sync (FULL_MASK, !!predicate);
}

int _CL_OVERLOADABLE
sub_group_any (int predicate)
{
  return __nvvm_vote_any_sync (FULL_MASK, !!predicate);
}

#define SUBGROUP_SHUFFLE_I32_PT(PREFIX, TYPE)                                 \
  TYPE _CL_OVERLOADABLE PREFIX##sub_group_shuffle (TYPE value, uint idx)      \
  {                                                                           \
    return __nvvm_shfl_sync_idx_i32 (FULL_MASK, value, idx,                   \
                                     get_max_sub_group_size () - 1);          \
  }

#define SUBGROUP_SHUFFLE_I32(TYPE)                                            \
  SUBGROUP_SHUFFLE_I32_PT (, TYPE)                                            \
  SUBGROUP_SHUFFLE_I32_PT (intel_, TYPE)

SUBGROUP_SHUFFLE_I32 (char);
SUBGROUP_SHUFFLE_I32 (uchar);
SUBGROUP_SHUFFLE_I32 (short);
SUBGROUP_SHUFFLE_I32 (ushort);
SUBGROUP_SHUFFLE_I32 (int);
SUBGROUP_SHUFFLE_I32 (uint);

float _CL_OVERLOADABLE
sub_group_shuffle (float value, uint idx)
{
  return __nvvm_shfl_sync_idx_f32 (FULL_MASK, value, idx,
                                   get_max_sub_group_size () - 1);
}

float _CL_OVERLOADABLE
intel_sub_group_shuffle (float value, uint idx)
{
  return __nvvm_shfl_sync_idx_f32 (FULL_MASK, value, idx,
                                   get_max_sub_group_size () - 1);
}

half _CL_OVERLOADABLE
sub_group_shuffle (half value, uint idx)
{
  return (half)sub_group_shuffle ((float)value, idx);
}

half _CL_OVERLOADABLE
intel_sub_group_shuffle (half value, uint idx)
{
  return (half)sub_group_shuffle ((float)value, idx);
}

#define SUBGROUP_SHUFFLE_2xI32_PT(PREFIX, TYPE)                               \
  TYPE _CL_OVERLOADABLE PREFIX##sub_group_shuffle (TYPE value, uint idx)      \
  {                                                                           \
    uint low, high;                                                           \
    __asm__ volatile ("mov.b64 {%0,%1}, %2;"                                  \
                      : "=r"(low), "=r"(high)                                 \
                      : "d"(value));                                          \
    low = __nvvm_shfl_sync_idx_i32 (FULL_MASK, low, idx,                      \
                                    get_max_sub_group_size () - 1);           \
    high = __nvvm_shfl_sync_idx_i32 (FULL_MASK, high, idx,                    \
                                     get_max_sub_group_size () - 1);          \
    __asm__ volatile ("mov.b64 %0, {%1,%2};"                                  \
                      : "=d"(value)                                           \
                      : "r"(low), "r"(high));                                 \
    return value;                                                             \
  }

#define SUBGROUP_SHUFFLE_2xI32(TYPE)                                          \
  SUBGROUP_SHUFFLE_2xI32_PT (, TYPE)                                          \
  SUBGROUP_SHUFFLE_2xI32_PT (intel_, TYPE)

SUBGROUP_SHUFFLE_2xI32 (long);
SUBGROUP_SHUFFLE_2xI32 (ulong);
SUBGROUP_SHUFFLE_2xI32 (double);

#define SUBGROUP_SHUFFLE_XOR_I32_PT(PREFIX, TYPE)                             \
  TYPE _CL_OVERLOADABLE PREFIX##sub_group_shuffle_xor (TYPE value, uint mask) \
  {                                                                           \
    return __nvvm_shfl_sync_bfly_i32 (FULL_MASK, value, mask,                 \
                                      get_max_sub_group_size () - 1);         \
  }

#define SUBGROUP_SHUFFLE_XOR_I32(TYPE)                                        \
  SUBGROUP_SHUFFLE_XOR_I32_PT (, TYPE)                                        \
  SUBGROUP_SHUFFLE_XOR_I32_PT (intel_, TYPE)

SUBGROUP_SHUFFLE_XOR_I32 (char);
SUBGROUP_SHUFFLE_XOR_I32 (uchar);
SUBGROUP_SHUFFLE_XOR_I32 (short);
SUBGROUP_SHUFFLE_XOR_I32 (ushort);
SUBGROUP_SHUFFLE_XOR_I32 (int);
SUBGROUP_SHUFFLE_XOR_I32 (uint);

float _CL_OVERLOADABLE
sub_group_shuffle_xor (float value, uint mask)
{
  return __nvvm_shfl_sync_bfly_f32 (FULL_MASK, value, mask,
                                    get_max_sub_group_size () - 1);
}

float _CL_OVERLOADABLE
intel_sub_group_shuffle_xor (float value, uint mask)
{
  return __nvvm_shfl_sync_bfly_f32 (FULL_MASK, value, mask,
                                    get_max_sub_group_size () - 1);
}

half _CL_OVERLOADABLE
sub_group_shuffle_xor (half value, uint mask)
{
  return (half)__nvvm_shfl_sync_bfly_f32 (FULL_MASK, value, mask,
                                          get_max_sub_group_size () - 1);
}

half _CL_OVERLOADABLE
intel_sub_group_shuffle_xor (half value, uint mask)
{
  return (half)__nvvm_shfl_sync_bfly_f32 (FULL_MASK, value, mask,
                                          get_max_sub_group_size () - 1);
}

#define SUBGROUP_SHUFFLE_XOR_2xI32_PT(PREFIX, TYPE)                           \
  TYPE _CL_OVERLOADABLE PREFIX##sub_group_shuffle_xor (TYPE value, uint mask) \
  {                                                                           \
    uint low, high;                                                           \
    __asm__ volatile ("mov.b64 {%0,%1}, %2;"                                  \
                      : "=r"(low), "=r"(high)                                 \
                      : "d"(value));                                          \
    low = __nvvm_shfl_sync_bfly_i32 (FULL_MASK, low, mask,                    \
                                     get_max_sub_group_size () - 1);          \
    high = __nvvm_shfl_sync_bfly_i32 (FULL_MASK, high, mask,                  \
                                      get_max_sub_group_size () - 1);         \
    __asm__ volatile ("mov.b64 %0, {%1,%2};"                                  \
                      : "=d"(value)                                           \
                      : "r"(low), "r"(high));                                 \
    return value;                                                             \
  }

#define SUBGROUP_SHUFFLE_XOR_2xI32(TYPE)                                      \
  SUBGROUP_SHUFFLE_XOR_2xI32_PT (, TYPE)                                      \
  SUBGROUP_SHUFFLE_XOR_2xI32_PT (intel_, TYPE)

SUBGROUP_SHUFFLE_XOR_2xI32 (long);
SUBGROUP_SHUFFLE_XOR_2xI32 (ulong);
SUBGROUP_SHUFFLE_XOR_2xI32 (double);

// XXX: Simplify by using __nvvm_redux_sync* if/when there's a SM80+ kernellib
#define SUB_GROUP_REDUCE_OT(OPNAME, OPERATION, TYPE)                          \
  TYPE _CL_OVERLOADABLE sub_group_reduce##OPNAME (TYPE val)                   \
  {                                                                           \
    uint lane = get_sub_group_local_id ();                                    \
    for (uint srcmask = get_max_sub_group_size () / 2; srcmask >= 1;          \
         srcmask /= 2)                                                        \
      {                                                                       \
        uint src_lane = lane ^ srcmask;                                       \
        TYPE a = val;                                                         \
        TYPE b = sub_group_shuffle_xor (a, srcmask);                          \
        /* Ignore values from inactive lanes */                               \
        if ((1 << src_lane) & __nvvm_vote_any_sync (FULL_MASK, 1))            \
          {                                                                   \
            val = OPERATION;                                                  \
          }                                                                   \
      }                                                                       \
    return val;                                                               \
  }

#define SUB_GROUP_REDUCE_T(OPNAME, OPERATION)                                 \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, int)                                \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, uint)                               \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, long)                               \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, ulong)                              \
  __IF_FP16 (SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, half))                   \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, float)                              \
  __IF_FP64 (SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, double))

SUB_GROUP_REDUCE_T (_add, (a + b))
SUB_GROUP_REDUCE_T (_min, (a > b ? b : a))
SUB_GROUP_REDUCE_T (_max, (a > b ? a : b))