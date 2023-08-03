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

/* The following code is taken & adapted from half library @
 * http://half.sourceforge.net
 *
 * half - IEEE 754-based half-precision floating point library.
 *
 * Copyright (c) 2012-2017 Christian Rau <rauy@users.sourceforge.net>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation
 * files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies of the
 * Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "templates.h"

#define ROUND_TOWARD_INFINITY 1
#define ROUND_TOWARD_NEG_INFINITY 2
#define ROUND_TOWARD_ZERO 3
#define ROUND_TO_NEAREST 4

static ushort
_cl_float2half_round (float f, int round_mode)
{
  uint bits = as_uint (f);
  ushort hbits = (bits >> 16) & 0x8000;
  bits &= 0x7FFFFFFF;
  int exp = bits >> 23;
  if (exp == 255)
    {
      ushort temp = (((bits & 0x7FFFFF) != 0) ? 0x03FF : 0x0);
      return (hbits | 0x7C00 | temp);
    }
  if (exp > 142)
    {
      if (round_mode == ROUND_TOWARD_INFINITY)
        return hbits | 0x7C00 - (hbits >> 15);
      if (round_mode == ROUND_TOWARD_NEG_INFINITY)
        return hbits | 0x7BFF + (hbits >> 15);
      return hbits | 0x7BFF + (round_mode != ROUND_TOWARD_ZERO);
    }
  int g, s;
  if (exp > 112)
    {
      g = (bits >> 12) & 1;
      s = (bits & 0xFFF) != 0;
      hbits |= ((exp - 112) << 10) | ((bits >> 13) & 0x3FF);
    }
  else if (exp > 101)
    {
      int i = 125 - exp;
      bits = (bits & 0x7FFFFF) | 0x800000;
      g = (bits >> i) & 1;
      s = (bits & ((1L << i) - 1)) != 0;
      hbits |= bits >> (i + 1);
    }
  else
    {
      g = 0;
      s = bits != 0;
    }
  if (round_mode == ROUND_TO_NEAREST)
    hbits += g & (s | hbits);
  else if (round_mode == ROUND_TOWARD_INFINITY)
    hbits += ~(hbits >> 15) & (s | g);
  else if (round_mode == ROUND_TOWARD_NEG_INFINITY)
    hbits += (hbits >> 15) & (g | s);
  return hbits;
}

static ushort
_cl_float2half (float d)
{
  return _cl_float2half_round (d, ROUND_TO_NEAREST);
}

static ushort
_cl_float2half_rte (float d)
{
  return _cl_float2half_round (d, ROUND_TO_NEAREST);
}

static ushort
_cl_float2half_rtz (float d)
{
  return _cl_float2half_round (d, ROUND_TOWARD_ZERO);
}

static ushort
_cl_float2half_rtn (float d)
{
  return _cl_float2half_round (d, ROUND_TOWARD_NEG_INFINITY);
}

static ushort
_cl_float2half_rtp (float d)
{
  return _cl_float2half_round (d, ROUND_TOWARD_INFINITY);
}

#ifdef cl_khr_fp64

static ushort
_cl_double2half_round (double value, int round_mode)
{
  ulong bits = as_ulong (value);
  uint hi = (bits >> 32);
  uint lo = (bits & 0xFFFFFFFF);
  ushort hbits = (hi >> 16) & 0x8000;
  hi &= 0x7FFFFFFF;
  int exp = hi >> 20;
  if (exp == 2047)
    {
      ushort temp = ((bits & 0xFFFFFFFFFFFFF) != 0 ? 0x03FF : 0x0);
      return (hbits | 0x7C00 | temp);
    }
  if (exp > 1038)
    {
      if (round_mode == ROUND_TOWARD_INFINITY)
        return (hbits | 0x7C00 - (hbits >> 15));
      if (round_mode == ROUND_TOWARD_NEG_INFINITY)
        return (hbits | 0x7BFF + (hbits >> 15));
      return (hbits | 0x7BFF + (round_mode != ROUND_TOWARD_ZERO));
    }
  int g;
  int s = (lo != 0);
  if (exp > 1008)
    {
      g = (hi >> 9) & 1;
      s |= (hi & 0x1FF) != 0;
      hbits |= ((exp - 1008) << 10) | ((hi >> 10) & 0x3FF);
    }
  else if (exp > 997)
    {
      int i = 1018 - exp;
      hi = (hi & 0xFFFFF) | 0x100000;
      g = (hi >> i) & 1;
      s |= (hi & ((1L << i) - 1)) != 0;
      hbits |= hi >> (i + 1);
    }
  else
    {
      g = 0;
      s |= hi != 0;
    }
  if (round_mode == ROUND_TO_NEAREST)
    hbits += g & (s | hbits);
  else if (round_mode == ROUND_TOWARD_INFINITY)
    hbits += ~(hbits >> 15) & (s | g);
  else if (round_mode == ROUND_TOWARD_NEG_INFINITY)
    hbits += (hbits >> 15) & (g | s);
  return hbits;
}

static ushort
_cl_double2half (double d)
{
  return _cl_double2half_round (d, ROUND_TO_NEAREST);
}

static ushort
_cl_double2half_rte (double d)
{
  return _cl_double2half_round (d, ROUND_TO_NEAREST);
}

static ushort
_cl_double2half_rtz (double d)
{
  return _cl_double2half_round (d, ROUND_TOWARD_ZERO);
}

static ushort
_cl_double2half_rtn (double d)
{
  return _cl_double2half_round (d, ROUND_TOWARD_NEG_INFINITY);
}

static ushort
_cl_double2half_rtp (double d)
{
  return _cl_double2half_round (d, ROUND_TOWARD_INFINITY);
}

#endif

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
    ((MOD ushort *)p)[offset] = _cl_float2half##SUFFIX (data);                \
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
    ((MOD ushort *)p)[offset] = _cl_float2half##SUFFIX (data);                \
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
    ((MOD ushort *)p)[offset] = _cl_float2half##SUFFIX (data);                \
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
    ((MOD ushort *)p)[offset] = _cl_float2half##SUFFIX (data);                \
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

IF_GEN_AS(IMPLEMENT_VSTORE_HALF (__generic, ))
IF_GEN_AS(IMPLEMENT_VSTORE_HALF (__generic, _rte))
IF_GEN_AS(IMPLEMENT_VSTORE_HALF (__generic, _rtz))
IF_GEN_AS(IMPLEMENT_VSTORE_HALF (__generic, _rtp))
IF_GEN_AS(IMPLEMENT_VSTORE_HALF (__generic, _rtn))


#ifdef cl_khr_fp64

///#ifdef __F16C__
#if 0

#define IMPLEMENT_VSTORE_HALF_DBL(MOD, SUFFIX)                                \
                                                                              \
  void _CL_OVERLOADABLE vstore_half##SUFFIX (double data, size_t offset,      \
                                             MOD half *p)                     \
  {                                                                           \
    ((MOD ushort *)p)[offset] = _cl_double2half##SUFFIX (data);               \
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
    ((MOD ushort *)p)[offset] = _cl_double2half##SUFFIX (data);               \
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
    ((MOD ushort *)p)[offset] = _cl_double2half##SUFFIX (data);               \
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
    ((MOD ushort *)p)[offset] = _cl_double2half##SUFFIX (data);               \
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

#endif // __F16C__

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

IF_GEN_AS(IMPLEMENT_VSTORE_HALF_DBL (__generic, ))
IF_GEN_AS(IMPLEMENT_VSTORE_HALF_DBL (__generic, _rte))
IF_GEN_AS(IMPLEMENT_VSTORE_HALF_DBL (__generic, _rtz))
IF_GEN_AS(IMPLEMENT_VSTORE_HALF_DBL (__generic, _rtp))
IF_GEN_AS(IMPLEMENT_VSTORE_HALF_DBL (__generic, _rtn))

#endif // cl_khr_fp64
