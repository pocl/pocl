/* OpenCL built-in library: async_work_group_strided_copy()

   Copyright (c) 2018 pocl developers

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

// TODO: Use get_local_linear_id() when available
#define IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS_SINGLE(GENTYPE)                    \
  __attribute__ ((overloadable)) event_t async_work_group_strided_copy (      \
    __local GENTYPE *dst, const __global GENTYPE *src, size_t num_gentypes,   \
    size_t src_stride, event_t event)                                         \
  {                                                                           \
    size_t lid =                                                              \
      get_local_id(0) +                                                       \
      (get_local_id(1) +                                                      \
       (get_local_id(2) * get_local_size(1))) * get_local_size(0);            \
    size_t lsz = get_local_size(0) * get_local_size(1) * get_local_size(2);   \
    for (size_t i = lid; i < num_gentypes; i+=lsz)                            \
      dst[i] = src[i * src_stride];                                           \
    return event;                                                             \
  }                                                                           \
                                                                              \
  __attribute__ ((overloadable)) event_t async_work_group_strided_copy (      \
      __global GENTYPE *dst, const __local GENTYPE *src, size_t num_gentypes, \
      size_t dst_stride, event_t event)                                       \
  {                                                                           \
    size_t lid =                                                              \
      get_local_id(0) +                                                       \
      (get_local_id(1) +                                                      \
       (get_local_id(2) * get_local_size(1))) * get_local_size(0);            \
    size_t lsz = get_local_size(0) * get_local_size(1) * get_local_size(2);   \
    for (size_t i = lid; i < num_gentypes; i+=lsz)                            \
      dst[i * dst_stride] = src[i];                                           \
    return event;                                                             \
  }

#define IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS(GENTYPE)                           \
  IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS_SINGLE (GENTYPE)                         \
  IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS_SINGLE (GENTYPE##2)                      \
  IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS_SINGLE (GENTYPE##3)                      \
  IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS_SINGLE (GENTYPE##4)                      \
  IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS_SINGLE (GENTYPE##8)                      \
  IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS_SINGLE (GENTYPE##16)

IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS (char);
IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS (uchar);
IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS (short);
IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS (ushort);
IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS (int);
IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS (uint);
__IF_INT64 (IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS (long));
__IF_INT64 (IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS (ulong));

__IF_FP16 (IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS (half));
IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS (float);
__IF_FP64 (IMPLEMENT_ASYNC_STRIDED_COPY_FUNCS (double));
