/* OpenCL built-in library: shuffle() / shuffle2()

   Copyright (c) 2018 Michal Babej / Tampere University of Technology

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

   Original:
   * Written by Kalle Raiskila, 2014
   * No rights reserved.
*/

#define _CL_IMPLEMENT_SHUFFLE(ELTYPE, MTYPE, N, M)                            \
  ELTYPE##N __attribute__ ((overloadable))                                    \
      shuffle (ELTYPE##M in, MTYPE##N mask)                                   \
  {                                                                           \
    const MTYPE##N valid_bits = (MTYPE##N) (M - 1);                           \
    mask = mask & valid_bits;                                                 \
    union                                                                     \
    {                                                                         \
      ELTYPE##M v;                                                            \
      ELTYPE s[M];                                                            \
    } in_u;                                                                   \
    in_u.v = in;                                                              \
    union                                                                     \
    {                                                                         \
      ELTYPE##N v;                                                            \
      ELTYPE s[N];                                                            \
    } out_u;                                                                  \
    for (size_t i = 0; i < N; ++i)                                            \
      {                                                                       \
        out_u.s[i] = in_u.s[mask[i]];                                         \
      }                                                                       \
    return out_u.v;                                                           \
  }                                                                           \
                                                                              \
  ELTYPE##N __attribute__ ((overloadable))                                    \
      shuffle2 (ELTYPE##M in1, ELTYPE##M in2, MTYPE##N mask)                  \
  {                                                                           \
    const MTYPE##N valid_bits = (MTYPE##N) (2 * M - 1);                       \
    mask = mask & valid_bits;                                                 \
    union                                                                     \
    {                                                                         \
      ELTYPE##M v;                                                            \
      ELTYPE s[M];                                                            \
    } in1_u, in2_u;                                                           \
    in1_u.v = in1;                                                            \
    in2_u.v = in2;                                                            \
    union                                                                     \
    {                                                                         \
      ELTYPE##N v;                                                            \
      ELTYPE s[N];                                                            \
    } out_u;                                                                  \
    for (size_t i = 0; i < N; ++i)                                            \
      {                                                                       \
        MTYPE m = mask[i];                                                    \
        out_u.s[i] = ((m < M) ? (in1_u.s[m]) : (in2_u.s[m - M]));             \
      }                                                                       \
    return out_u.v;                                                           \
  }

#define _CL_IMPLEMENT_SHUFFLE_M(ELTYPE, MTYPE, M)                             \
  _CL_IMPLEMENT_SHUFFLE (ELTYPE, MTYPE, M, 2)                                 \
  _CL_IMPLEMENT_SHUFFLE (ELTYPE, MTYPE, M, 4)                                 \
  _CL_IMPLEMENT_SHUFFLE (ELTYPE, MTYPE, M, 8)                                 \
  _CL_IMPLEMENT_SHUFFLE (ELTYPE, MTYPE, M, 16)

#define _CL_IMPLEMENT_SHUFFLE_MN(ELTYPE, MTYPE)                               \
  _CL_IMPLEMENT_SHUFFLE_M (ELTYPE, MTYPE, 2)                                  \
  _CL_IMPLEMENT_SHUFFLE_M (ELTYPE, MTYPE, 4)                                  \
  _CL_IMPLEMENT_SHUFFLE_M (ELTYPE, MTYPE, 8)                                  \
  _CL_IMPLEMENT_SHUFFLE_M (ELTYPE, MTYPE, 16)

_CL_IMPLEMENT_SHUFFLE_MN (char, uchar)
_CL_IMPLEMENT_SHUFFLE_MN (uchar, uchar)
_CL_IMPLEMENT_SHUFFLE_MN (short, ushort)
_CL_IMPLEMENT_SHUFFLE_MN (ushort, ushort)
__IF_FP16 (_CL_IMPLEMENT_SHUFFLE_MN (half, ushort))
_CL_IMPLEMENT_SHUFFLE_MN (int, uint)
_CL_IMPLEMENT_SHUFFLE_MN (uint, uint)
_CL_IMPLEMENT_SHUFFLE_MN (float, uint)
__IF_INT64 (_CL_IMPLEMENT_SHUFFLE_MN (long, ulong)
                _CL_IMPLEMENT_SHUFFLE_MN (ulong, ulong))
__IF_FP64 (_CL_IMPLEMENT_SHUFFLE_MN (double, ulong))
