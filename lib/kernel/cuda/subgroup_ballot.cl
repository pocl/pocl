/* OpenCL built-in library: subgroup functions

   Copyright (c) 2024 Henry Linjamäki / Intel Finland Oy

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

uint4 _CL_OVERLOADABLE sub_group_ballot(int predicate) {
  uint4 result = (uint4)(0u);
  // NOTE:
  // * LLVM/NVPTX backend requires at least SM 3.0 and PTX 6.0 to select this
  //   intrinsic.
  // * vote.ballot is not supported in PTX 6.4+ for SM 7.0+ architectures.
  result.x = __nvvm_vote_ballot(!!predicate);
  return result;
}

#define SUB_GROUP_REDUCE_OT(OPNAME, OPERATION, TYPE)                          \
  TYPE _CL_OVERLOADABLE sub_group_shuffle_xor (TYPE, uint);                   \
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
        if ((1 << src_lane) & sub_group_ballot (1).x)                         \
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
