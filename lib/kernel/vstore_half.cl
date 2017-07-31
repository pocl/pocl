/* OpenCL built-in library: vstore_half()

   Copyright (c) 2011 Universidad Rey Juan Carlos
   Copyright (c) 2017 Michal Babej / Tampere University of Technology

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

/* The following function is rewritten to OpenCL from a
 * FP16 header-only library using The MIT License (MIT)
 *
 * https://github.com/Maratyszcza/FP16
 */

ushort
_cl_float2half (float f)
{
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
  float base = (fabs(f) * scale_to_inf) * scale_to_zero;
  const uint w = as_uint(f);
  const uint shl1_w = w + w;
  const uint sign = w & 0x80000000;
  uint bias = shl1_w & 0xFF000000;
  if (bias < (uint)(0x71000000)) {
    bias = (uint) (0x71000000);
  }
  base = as_float((bias >> 1) + (uint)(0x07800000)) + base;
  const uint bits = as_uint(base);
  const uint exp_bits = (bits >> 13) & (uint)(0x00007C00);
  const uint mantissa_bits = bits & (uint)(0x00000FFF);
  const uint nonsign = exp_bits + mantissa_bits;
  return ((sign >> 16) | (shl1_w > (uint)(0xFF000000) ? (ushort)0x7E00 : convert_ushort(nonsign)));
}




#define _cl_float2half_rte _cl_float2half
#define _cl_float2half_rtn _cl_float2half
#define _cl_float2half_rtz _cl_float2half
#define _cl_float2half_rtp _cl_float2half

#ifdef __F16C__

ushort4 _cl_float2half4 (const float4 data);
ushort8 _cl_float2half8 (const float8 data);
ushort4 _cl_float2half4_rte (const float4 data);
ushort8 _cl_float2half8_rte (const float8 data);
ushort4 _cl_float2half4_rtn (const float4 data);
ushort8 _cl_float2half8_rtn (const float8 data);
ushort4 _cl_float2half4_rtp (const float4 data);
ushort8 _cl_float2half8_rtp (const float8 data);
ushort4 _cl_float2half4_rtz (const float4 data);
ushort8 _cl_float2half8_rtz (const float8 data);

#define IMPLEMENT_VSTORE_HALF(MOD, SUFFIX)                                    \
                                                                              \
  void _CL_OVERLOADABLE vstore_half##SUFFIX (float data, size_t offset,       \
                                             MOD half *p)                     \
  {                                                                           \
    ((MOD ushort *)p)[offset] = _cl_float2half (data);                        \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half2##SUFFIX (float2 data, size_t offset,     \
                                              MOD half *p)                    \
  {                                                                           \
    vstore_half##SUFFIX (data.lo, offset * 2, p);                             \
    vstore_half##SUFFIX (data.hi, offset * 2 + 1, p);                         \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half3##SUFFIX (float3 data, size_t offset,     \
                                              MOD half *p)                    \
  {                                                                           \
    vstore_half##SUFFIX (data.x, offset * 3, p);                              \
    vstore_half##SUFFIX (data.y, offset * 3 + 1, p);                          \
    vstore_half##SUFFIX (data.z, offset * 3 + 2, p);                          \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half4##SUFFIX (float4 data, size_t offset,     \
                                              MOD half *p)                    \
  {                                                                           \
    ((MOD ushort4 *)p)[offset] = _cl_float2half4##SUFFIX (data);              \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half8##SUFFIX (float8 data, size_t offset,     \
                                              MOD half *p)                    \
  {                                                                           \
    ((MOD ushort8 *)p)[offset] = _cl_float2half8##SUFFIX (data);              \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half16##SUFFIX (float16 data, size_t offset,   \
                                               MOD half *p)                   \
  {                                                                           \
    ((MOD ushort8 *)p)[offset * 2] = _cl_float2half8##SUFFIX (data.lo);       \
    ((MOD ushort8 *)p)[offset * 2 + 1] = _cl_float2half8##SUFFIX (data.hi);   \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half##SUFFIX (float data, size_t offset,      \
                                              MOD half *p)                    \
  {                                                                           \
    ((MOD ushort *)p)[offset] = _cl_float2half (data);                        \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half2##SUFFIX (float2 data, size_t offset,    \
                                               MOD half *p)                   \
  {                                                                           \
    vstorea_half##SUFFIX (data.lo, offset * 2, p);                            \
    vstorea_half##SUFFIX (data.hi, offset * 2 + 1, p);                        \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half3##SUFFIX (float3 data, size_t offset,    \
                                               MOD half *p)                   \
  {                                                                           \
    vstorea_half2##SUFFIX (data.lo, offset * 2, p);                           \
    vstorea_half2##SUFFIX ((float2) (data.z, 0.0f), offset * 2 + 1, p);       \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half4##SUFFIX (float4 data, size_t offset,    \
                                               MOD half *p)                   \
  {                                                                           \
    ((MOD ushort4 *)p)[offset] = _cl_float2half4##SUFFIX (data);              \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half8##SUFFIX (float8 data, size_t offset,    \
                                               MOD half *p)                   \
  {                                                                           \
    ((MOD ushort8 *)p)[offset] = _cl_float2half8##SUFFIX (data);              \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half16##SUFFIX (float16 data, size_t offset,  \
                                                MOD half *p)                  \
  {                                                                           \
    ((MOD ushort8 *)p)[offset * 2] = _cl_float2half8##SUFFIX (data.lo);       \
    ((MOD ushort8 *)p)[offset * 2 + 1] = _cl_float2half8##SUFFIX (data.hi);   \
  }

// __F16C__
#else

#define IMPLEMENT_VSTORE_HALF(MOD, SUFFIX)                                    \
                                                                              \
  void _CL_OVERLOADABLE vstore_half##SUFFIX (float data, size_t offset,       \
                                             MOD half *p)                     \
  {                                                                           \
    if (isnan (data))                                                         \
      ((MOD ushort *)p)[offset] = (ushort)0xffff;                             \
    else                                                                      \
      ((MOD ushort *)p)[offset] = _cl_float2half##SUFFIX (data);              \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half2##SUFFIX (float2 data, size_t offset,     \
                                              MOD half *p)                    \
  {                                                                           \
    vstore_half##SUFFIX (data.lo, offset * 2, p);                             \
    vstore_half##SUFFIX (data.hi, offset * 2 + 1, p);                         \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half3##SUFFIX (float3 data, size_t offset,     \
                                              MOD half *p)                    \
  {                                                                           \
    vstore_half##SUFFIX (data.x, offset * 3, p);                              \
    vstore_half##SUFFIX (data.y, offset * 3 + 1, p);                          \
    vstore_half##SUFFIX (data.z, offset * 3 + 2, p);                          \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half4##SUFFIX (float4 data, size_t offset,     \
                                              MOD half *p)                    \
  {                                                                           \
    vstore_half2##SUFFIX (data.lo, offset * 2, p);                            \
    vstore_half2##SUFFIX (data.hi, offset * 2 + 1, p);                        \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half8##SUFFIX (float8 data, size_t offset,     \
                                              MOD half *p)                    \
  {                                                                           \
    vstore_half4##SUFFIX (data.lo, offset * 2, p);                            \
    vstore_half4##SUFFIX (data.hi, offset * 2 + 1, p);                        \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half16##SUFFIX (float16 data, size_t offset,   \
                                               MOD half *p)                   \
  {                                                                           \
    vstore_half8##SUFFIX (data.lo, offset * 2, p);                            \
    vstore_half8##SUFFIX (data.hi, offset * 2 + 1, p);                        \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half##SUFFIX (float data, size_t offset,      \
                                              MOD half *p)                    \
  {                                                                           \
    if (isnan (data))                                                         \
      ((MOD ushort *)p)[offset] = (ushort)0xffff;                             \
    else                                                                      \
      ((MOD ushort *)p)[offset] = _cl_float2half##SUFFIX (data);              \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half2##SUFFIX (float2 data, size_t offset,    \
                                               MOD half *p)                   \
  {                                                                           \
    vstorea_half##SUFFIX (data.lo, offset * 2, p);                            \
    vstorea_half##SUFFIX (data.hi, offset * 2 + 1, p);                        \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half3##SUFFIX (float3 data, size_t offset,    \
                                               MOD half *p)                   \
  {                                                                           \
    vstorea_half2##SUFFIX (data.lo, offset * 2, p);                           \
    vstorea_half2##SUFFIX ((float2) (data.z, 0.0f), offset * 2 + 1, p);       \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half4##SUFFIX (float4 data, size_t offset,    \
                                               MOD half *p)                   \
  {                                                                           \
    vstorea_half2##SUFFIX (data.lo, offset * 2, p);                           \
    vstorea_half2##SUFFIX (data.hi, offset * 2 + 1, p);                       \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half8##SUFFIX (float8 data, size_t offset,    \
                                               MOD half *p)                   \
  {                                                                           \
    vstorea_half4##SUFFIX (data.lo, offset * 2, p);                           \
    vstorea_half4##SUFFIX (data.hi, offset * 2 + 1, p);                       \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half16##SUFFIX (float16 data, size_t offset,  \
                                                MOD half *p)                  \
  {                                                                           \
    vstorea_half8##SUFFIX (data.lo, offset * 2, p);                           \
    vstorea_half8##SUFFIX (data.hi, offset * 2 + 1, p);                       \
  }

#endif

IMPLEMENT_VSTORE_HALF (__global, )
IMPLEMENT_VSTORE_HALF (__global, _rte)
IMPLEMENT_VSTORE_HALF (__global, _rtz)
IMPLEMENT_VSTORE_HALF (__global, _rtp)
IMPLEMENT_VSTORE_HALF (__global, _rtn)
IMPLEMENT_VSTORE_HALF (__local, )
IMPLEMENT_VSTORE_HALF (__local, _rte)
IMPLEMENT_VSTORE_HALF (__local, _rtz)
IMPLEMENT_VSTORE_HALF (__local, _rtp)
IMPLEMENT_VSTORE_HALF (__local, _rtn)
IMPLEMENT_VSTORE_HALF (__private, )
IMPLEMENT_VSTORE_HALF (__private, _rte)
IMPLEMENT_VSTORE_HALF (__private, _rtz)
IMPLEMENT_VSTORE_HALF (__private, _rtp)
IMPLEMENT_VSTORE_HALF (__private, _rtn)

#ifdef cl_khr_fp64

#ifdef __F16C__

#define IMPLEMENT_VSTORE_HALF_DBL(MOD, SUFFIX)                                \
                                                                              \
  void _CL_OVERLOADABLE vstore_half##SUFFIX (double data, size_t offset,      \
                                             MOD half *p)                     \
  {                                                                           \
    ((MOD ushort *)p)[offset]                                                 \
        = _cl_float2half##SUFFIX (convert_float##SUFFIX (data));              \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half2##SUFFIX (double2 data, size_t offset,    \
                                              MOD half *p)                    \
  {                                                                           \
    vstore_half##SUFFIX (data.lo, offset * 2, p);                             \
    vstore_half##SUFFIX (data.hi, offset * 2 + 1, p);                         \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half3##SUFFIX (double3 data, size_t offset,    \
                                              MOD half *p)                    \
  {                                                                           \
    vstore_half##SUFFIX (data.x, offset * 3, p);                              \
    vstore_half##SUFFIX (data.y, offset * 3 + 1, p);                          \
    vstore_half##SUFFIX (data.z, offset * 3 + 2, p);                          \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half4##SUFFIX (double4 data, size_t offset,    \
                                              MOD half *p)                    \
  {                                                                           \
    ((MOD ushort4 *)p)[offset]                                                \
        = _cl_float2half4##SUFFIX (convert_float4##SUFFIX (data));            \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half8##SUFFIX (double8 data, size_t offset,    \
                                              MOD half *p)                    \
  {                                                                           \
    ((MOD ushort8 *)p)[offset]                                                \
        = _cl_float2half8##SUFFIX (convert_float8##SUFFIX (data));            \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half16##SUFFIX (double16 data, size_t offset,  \
                                               MOD half *p)                   \
  {                                                                           \
    ((MOD ushort8 *)p)[offset * 2]                                            \
        = _cl_float2half8##SUFFIX (convert_float8##SUFFIX (data.lo));         \
    ((MOD ushort8 *)p)[offset * 2 + 1]                                        \
        = _cl_float2half8##SUFFIX (convert_float8##SUFFIX (data.hi));         \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half##SUFFIX (double data, size_t offset,     \
                                              MOD half *p)                    \
  {                                                                           \
    ((MOD ushort *)p)[offset]                                                 \
        = _cl_float2half (convert_float##SUFFIX (data));                      \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half2##SUFFIX (double2 data, size_t offset,   \
                                               MOD half *p)                   \
  {                                                                           \
    vstorea_half##SUFFIX (data.lo, offset * 2, p);                            \
    vstorea_half##SUFFIX (data.hi, offset * 2 + 1, p);                        \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half3##SUFFIX (double3 data, size_t offset,   \
                                               MOD half *p)                   \
  {                                                                           \
    vstorea_half2##SUFFIX (data.lo, offset * 2, p);                           \
    vstorea_half2##SUFFIX ((float2) (data.z, 0.0f), offset * 2 + 1, p);       \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half4##SUFFIX (double4 data, size_t offset,   \
                                               MOD half *p)                   \
  {                                                                           \
    ((MOD ushort4 *)p)[offset]                                                \
        = _cl_float2half4##SUFFIX (convert_float4##SUFFIX (data));            \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half8##SUFFIX (double8 data, size_t offset,   \
                                               MOD half *p)                   \
  {                                                                           \
    ((MOD ushort8 *)p)[offset]                                                \
        = _cl_float2half8##SUFFIX (convert_float8##SUFFIX (data));            \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half16##SUFFIX (double16 data, size_t offset, \
                                                MOD half *p)                  \
  {                                                                           \
    ((MOD ushort8 *)p)[offset * 2]                                            \
        = _cl_float2half8##SUFFIX (convert_float8##SUFFIX (data.lo));         \
    ((MOD ushort8 *)p)[offset * 2 + 1]                                        \
        = _cl_float2half8##SUFFIX (convert_float8##SUFFIX (data.hi));         \
  }

// __F16C__
#else

#define IMPLEMENT_VSTORE_HALF_DBL(MOD, SUFFIX)                                \
                                                                              \
  void _CL_OVERLOADABLE vstore_half##SUFFIX (double data, size_t offset,      \
                                             MOD half *p)                     \
  {                                                                           \
    if (isnan (data))                                                         \
      ((MOD ushort *)p)[offset] = (ushort)0xffff;                             \
    else                                                                      \
      ((MOD ushort *)p)[offset]                                               \
          = _cl_float2half (convert_float##SUFFIX (data));                    \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half2##SUFFIX (double2 data, size_t offset,    \
                                              MOD half *p)                    \
  {                                                                           \
    vstore_half##SUFFIX (data.lo, offset * 2, p);                             \
    vstore_half##SUFFIX (data.hi, offset * 2 + 1, p);                         \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half3##SUFFIX (double3 data, size_t offset,    \
                                              MOD half *p)                    \
  {                                                                           \
    vstore_half##SUFFIX (data.x, offset * 3, p);                              \
    vstore_half##SUFFIX (data.y, offset * 3 + 1, p);                          \
    vstore_half##SUFFIX (data.z, offset * 3 + 2, p);                          \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half4##SUFFIX (double4 data, size_t offset,    \
                                              MOD half *p)                    \
  {                                                                           \
    vstore_half2##SUFFIX (data.lo, offset * 2, p);                            \
    vstore_half2##SUFFIX (data.hi, offset * 2 + 1, p);                        \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half8##SUFFIX (double8 data, size_t offset,    \
                                              MOD half *p)                    \
  {                                                                           \
    vstore_half4##SUFFIX (data.lo, offset * 2, p);                            \
    vstore_half4##SUFFIX (data.hi, offset * 2 + 1, p);                        \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore_half16##SUFFIX (double16 data, size_t offset,  \
                                               MOD half *p)                   \
  {                                                                           \
    vstore_half8##SUFFIX (data.lo, offset * 2, p);                            \
    vstore_half8##SUFFIX (data.hi, offset * 2 + 1, p);                        \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half##SUFFIX (double data, size_t offset,     \
                                              MOD half *p)                    \
  {                                                                           \
    if (isnan (data))                                                         \
      ((MOD ushort *)p)[offset] = (ushort)0xffff;                             \
    else                                                                      \
      ((MOD ushort *)p)[offset]                                               \
          = _cl_float2half (convert_float##SUFFIX (data));                    \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half2##SUFFIX (double2 data, size_t offset,   \
                                               MOD half *p)                   \
  {                                                                           \
    vstorea_half##SUFFIX (data.lo, offset * 2, p);                            \
    vstorea_half##SUFFIX (data.hi, offset * 2 + 1, p);                        \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half3##SUFFIX (double3 data, size_t offset,   \
                                               MOD half *p)                   \
  {                                                                           \
    vstorea_half2##SUFFIX (data.lo, offset * 2, p);                           \
    vstorea_half2##SUFFIX ((double2) (data.z, 0.0), offset * 2 + 1, p);       \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half4##SUFFIX (double4 data, size_t offset,   \
                                               MOD half *p)                   \
  {                                                                           \
    vstorea_half2##SUFFIX (data.lo, offset * 2, p);                           \
    vstorea_half2##SUFFIX (data.hi, offset * 2 + 1, p);                       \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half8##SUFFIX (double8 data, size_t offset,   \
                                               MOD half *p)                   \
  {                                                                           \
    vstorea_half4##SUFFIX (data.lo, offset * 2, p);                           \
    vstorea_half4##SUFFIX (data.hi, offset * 2 + 1, p);                       \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstorea_half16##SUFFIX (double16 data, size_t offset, \
                                                MOD half *p)                  \
  {                                                                           \
    vstorea_half8##SUFFIX (data.lo, offset * 2, p);                           \
    vstorea_half8##SUFFIX (data.hi, offset * 2 + 1, p);                       \
  }

#endif

IMPLEMENT_VSTORE_HALF_DBL (__global, )
IMPLEMENT_VSTORE_HALF_DBL (__global, _rte)
IMPLEMENT_VSTORE_HALF_DBL (__global, _rtz)
IMPLEMENT_VSTORE_HALF_DBL (__global, _rtp)
IMPLEMENT_VSTORE_HALF_DBL (__global, _rtn)
IMPLEMENT_VSTORE_HALF_DBL (__local, )
IMPLEMENT_VSTORE_HALF_DBL (__local, _rte)
IMPLEMENT_VSTORE_HALF_DBL (__local, _rtz)
IMPLEMENT_VSTORE_HALF_DBL (__local, _rtp)
IMPLEMENT_VSTORE_HALF_DBL (__local, _rtn)
IMPLEMENT_VSTORE_HALF_DBL (__private, )
IMPLEMENT_VSTORE_HALF_DBL (__private, _rte)
IMPLEMENT_VSTORE_HALF_DBL (__private, _rtz)
IMPLEMENT_VSTORE_HALF_DBL (__private, _rtp)
IMPLEMENT_VSTORE_HALF_DBL (__private, _rtn)

#endif
