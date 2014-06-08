/* OpenCL built-in C library for pocl: shuffle()
 * Written by Kalle Raiskila, 2014
 * No rights reserved.
 */

#define _CL_IMPLEMENT_SHUFFLE(ELTYPE, MTYPE, N, M)               \
  ELTYPE##N __attribute__ ((overloadable))                       \
  shuffle(ELTYPE##M in, MTYPE##N mask)                           \
  {                                                              \
    MTYPE i;                                                     \
    ELTYPE##N out;                                               \
    for(i=0; i<N; i++)                                           \
      out[i] = in[mask[i]];                                      \
    return out;                                                  \
  }                                                              \
                                                                 \
  ELTYPE##N __attribute__ ((overloadable))                       \
  shuffle2(ELTYPE##M in1, ELTYPE##M in2, MTYPE##N mask)          \
  {                                                              \
    MTYPE i;                                                     \
    ELTYPE##N out;                                               \
    for(i=0; i<N; i++) {                                         \
      MTYPE m = mask[i];                                         \
      if(m<M)                                                    \
        out[i] = in1[m];                                         \
      else                                                       \
        out[i] = in2[m-M];                                       \
    }                                                            \
    return out;                                                  \
  }

#define _CL_IMPLEMENT_SHUFFLE_M(ELTYPE, MTYPE, M)                       \
  _CL_IMPLEMENT_SHUFFLE(ELTYPE, MTYPE, M, 2)                            \
  _CL_IMPLEMENT_SHUFFLE(ELTYPE, MTYPE, M, 3)                            \
  _CL_IMPLEMENT_SHUFFLE(ELTYPE, MTYPE, M, 4)                            \
  _CL_IMPLEMENT_SHUFFLE(ELTYPE, MTYPE, M, 8)                            \
  _CL_IMPLEMENT_SHUFFLE(ELTYPE, MTYPE, M, 16)

#define _CL_IMPLEMENT_SHUFFLE_MN(ELTYPE, MTYPE)                         \
  _CL_IMPLEMENT_SHUFFLE_M(ELTYPE, MTYPE, 2)                             \
  _CL_IMPLEMENT_SHUFFLE_M(ELTYPE, MTYPE, 3)                             \
  _CL_IMPLEMENT_SHUFFLE_M(ELTYPE, MTYPE, 4)                             \
  _CL_IMPLEMENT_SHUFFLE_M(ELTYPE, MTYPE, 8)                             \
  _CL_IMPLEMENT_SHUFFLE_M(ELTYPE, MTYPE, 16)

_CL_IMPLEMENT_SHUFFLE_MN(char , uchar )
_CL_IMPLEMENT_SHUFFLE_MN(uchar , uchar )
_CL_IMPLEMENT_SHUFFLE_MN(short , ushort )
_CL_IMPLEMENT_SHUFFLE_MN(ushort , ushort )
_CL_IMPLEMENT_SHUFFLE_MN(int , uint )
_CL_IMPLEMENT_SHUFFLE_MN(uint , uint )
_CL_IMPLEMENT_SHUFFLE_MN(float , uint )
__IF_FP16(
_CL_IMPLEMENT_SHUFFLE_MN(half , ushort ))
__IF_FP64(
_CL_IMPLEMENT_SHUFFLE_MN(long , ulong )
_CL_IMPLEMENT_SHUFFLE_MN(ulong , ulong )
_CL_IMPLEMENT_SHUFFLE_MN(double , ulong ))
