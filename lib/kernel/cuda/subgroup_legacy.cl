/* OpenCL built-in library: subgroup functions for <SM70 hardware

   Copyright (c) 2024 Henry Linjamäki / Intel Finland Oy

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

#define SUBGROUP_SHUFFLE_I32_PT(PREFIX, TYPE)                                 \
  TYPE _CL_OVERLOADABLE PREFIX##sub_group_shuffle (TYPE value, uint idx)      \
  {                                                                           \
    uint active = __nvvm_vote_ballot (1);                                     \
    return __nvvm_shfl_idx_i32 (value, idx, 31);                              \
  }

#define SUBGROUP_SHUFFLE_I32(TYPE)                                             \
  SUBGROUP_SHUFFLE_I32_PT(, TYPE)                                              \
  SUBGROUP_SHUFFLE_I32_PT(intel_, TYPE)

SUBGROUP_SHUFFLE_I32(char);
SUBGROUP_SHUFFLE_I32(uchar);
SUBGROUP_SHUFFLE_I32(short);
SUBGROUP_SHUFFLE_I32(ushort);
SUBGROUP_SHUFFLE_I32(int);
SUBGROUP_SHUFFLE_I32(uint);

float _CL_OVERLOADABLE sub_group_shuffle(float value, uint idx) {
  return __nvvm_shfl_idx_f32(value, idx, 31);
}

float _CL_OVERLOADABLE intel_sub_group_shuffle(float value, uint idx) {
  return __nvvm_shfl_idx_f32(value, idx, 31);
}

#define SUBGROUP_SHUFFLE_XOR_I32_PT(PREFIX, TYPE)                              \
  TYPE _CL_OVERLOADABLE PREFIX##sub_group_shuffle_xor(TYPE value, uint mask) { \
    return __nvvm_shfl_bfly_i32(value, mask, 31);                              \
  }

#define SUBGROUP_SHUFFLE_XOR_I32(TYPE)                                         \
  SUBGROUP_SHUFFLE_XOR_I32_PT(, TYPE)                                          \
  SUBGROUP_SHUFFLE_XOR_I32_PT(intel_, TYPE)

SUBGROUP_SHUFFLE_XOR_I32(char);
SUBGROUP_SHUFFLE_XOR_I32(uchar);
SUBGROUP_SHUFFLE_XOR_I32(short);
SUBGROUP_SHUFFLE_XOR_I32(ushort);
SUBGROUP_SHUFFLE_XOR_I32(int);
SUBGROUP_SHUFFLE_XOR_I32(uint);

float _CL_OVERLOADABLE sub_group_shuffle_xor(float value, uint mask) {
  return __nvvm_shfl_bfly_f32(value, mask, 31);
}

float _CL_OVERLOADABLE intel_sub_group_shuffle_xor(float value, uint mask) {
  return __nvvm_shfl_bfly_f32(value, mask, 31);
}

int _CL_OVERLOADABLE
sub_group_all (int predicate)
{
  uint active = __nvvm_vote_ballot (1);
  __nvvm_bar_warp_sync (active);
  return __nvvm_vote_all (!!predicate);
}

int _CL_OVERLOADABLE
sub_group_any (int predicate)
{
  uint active = __nvvm_vote_ballot (1);
  __nvvm_bar_warp_sync (active);
  return __nvvm_vote_any (!!predicate);
}

#define SUB_GROUP_REDUCE_OT(OPNAME, OPERATION, TYPE, NVTYPE)                  \
  TYPE _CL_OVERLOADABLE sub_group_reduce##OPNAME (TYPE val)                   \
  {                                                                           \
    TYPE tmp = val;                                                           \
    for (int offset = 16; offset > 0; offset /= 2)                            \
      {                                                                       \
        TYPE a = tmp;                                                         \
        uint active = __nvvm_vote_ballot (1);                                 \
        __nvvm_bar_warp_sync (active);                                        \
        TYPE b = __nvvm_shfl_down_##NVTYPE (tmp, offset, 31);                 \
        tmp = OPERATION;                                                      \
      }                                                                       \
    return sub_group_broadcast (tmp, 0);                                      \
  }

#define SUB_GROUP_REDUCE_T(OPNAME, OPERATION)                                 \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, int, i32)                           \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, uint, i32)                          \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, long, i32)                          \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, ulong, i32)                         \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, float, f32)                         \
  __IF_FP16 (SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, half, f32))              \
  __IF_FP64 (SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, double, f32))

SUB_GROUP_REDUCE_T (_add, (a + b))
SUB_GROUP_REDUCE_T (_min, (a > b ? b : a))
SUB_GROUP_REDUCE_T (_max, (a > b ? a : b))
