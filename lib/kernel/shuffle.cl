/* OpenCL built-in C library for pocl: shuffle()
 * Written by Kalle Raiskila, 2014
 * No rights reserved.
 */

#define _CL_IMPLEMENT_SHUFFLE(ELTYPE, MTYPE, N, M)              \
  ELTYPE##N __attribute__ ((overloadable))                      \
  shuffle(ELTYPE##M in, MTYPE##N mask)                          \
  {                                                             \
    MTYPE msize = M==3 ? 4 : M;                                 \
    ELTYPE##N out;                                              \
    for (int i=0; i<N; ++i) {                                   \
      MTYPE m = mask[i] & (msize-1);                            \
      out[i] = in[m];                                           \
    }                                                           \
    return out;                                                 \
  }                                                             \
                                                                \
  ELTYPE##N __attribute__ ((overloadable))                      \
  shuffle2(ELTYPE##M in1, ELTYPE##M in2, MTYPE##N mask)         \
  {                                                             \
    MTYPE msize = M==3 ? 4 : M;                                 \
    ELTYPE##N out;                                              \
    for (int i=0; i<N; ++i) {                                   \
      MTYPE m = mask[i] & (2*msize-1);                          \
      if (m < msize)                                            \
        out[i] = in1[m];                                        \
      else                                                      \
        out[i] = in2[m-msize];                                  \
    }                                                           \
    return out;                                                 \
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

_CL_IMPLEMENT_SHUFFLE_MN(char  , uchar )
_CL_IMPLEMENT_SHUFFLE_MN(uchar , uchar )
_CL_IMPLEMENT_SHUFFLE_MN(short , ushort)
_CL_IMPLEMENT_SHUFFLE_MN(ushort, ushort)
__IF_FP16(
_CL_IMPLEMENT_SHUFFLE_MN(half  , ushort))
_CL_IMPLEMENT_SHUFFLE_MN(int   , uint  )
_CL_IMPLEMENT_SHUFFLE_MN(uint  , uint  )
_CL_IMPLEMENT_SHUFFLE_MN(float , uint  )
__IF_INT64(
_CL_IMPLEMENT_SHUFFLE_MN(long  , ulong )
_CL_IMPLEMENT_SHUFFLE_MN(ulong , ulong ))
__IF_FP64(
_CL_IMPLEMENT_SHUFFLE_MN(double, ulong ))



#if 0

// Implement half shuffles via reinterpreting as short
#define _CL_IMPLEMENT_SHUFFLE_HALF(N, M)                                \
  half##N __attribute__((overloadable))                                 \
  shuffle(half##M x, ushort##N mask)                                    \
  {                                                                     \
    return as_half##N(shuffle(as_short##M(x), mask));                   \
  }                                                                     \
  half##N __attribute__((overloadable))                                 \
  shuffle2(half##M x, half##M y, ushort##N mask)                        \
  {                                                                     \
    return as_half##N(shuffle2(as_short##M(x), as_short##M(y), mask));  \
  }

// LLVM 3.4 cannot handle the 3-element half case
#define _CL_IMPLEMENT_SHUFFLE_HALF_M(M)         \
  _CL_IMPLEMENT_SHUFFLE_HALF(M, 2)              \
  /*_CL_IMPLEMENT_SHUFFLE_HALF(M, 3)*/          \
  _CL_IMPLEMENT_SHUFFLE_HALF(M, 4)              \
  _CL_IMPLEMENT_SHUFFLE_HALF(M, 8)              \
  _CL_IMPLEMENT_SHUFFLE_HALF(M, 16)

#define _CL_IMPLEMENT_SHUFFLE_HALF_MN           \
  _CL_IMPLEMENT_SHUFFLE_HALF_M(2)               \
  /*_CL_IMPLEMENT_SHUFFLE_HALF_M(3)*/           \
  _CL_IMPLEMENT_SHUFFLE_HALF_M(4)               \
  _CL_IMPLEMENT_SHUFFLE_HALF_M(8)               \
  _CL_IMPLEMENT_SHUFFLE_HALF_M(16)

__IF_FP16(
_CL_IMPLEMENT_SHUFFLE_HALF_MN)

#endif
