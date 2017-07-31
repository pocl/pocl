/* OpenCL built-in library: vload_store_half_f16c()

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

/* Accelerated float-half conversions on x86-64 using
 * these builtins from Clang:
 *    BUILTIN(__builtin_ia32_vcvtps2ph, "V8sV4fIi", "")
 *    BUILTIN(__builtin_ia32_vcvtps2ph256, "V8sV8fIi", "")
 *    BUILTIN(__builtin_ia32_vcvtph2ps, "V4fV8s", "")
 *    BUILTIN(__builtin_ia32_vcvtph2ps256, "V8fV8s", "")
 *    _mm_cvtps_ph(a, int);
 *    _mm256_cvtps_ph(a, int);
 *    _mm_cvtph_ps(a);
 *    _mm256_cvtph_ps(a);
 */

/* TODO
 * If case of a denormal operand,
 * the correct normal result is returned.
 * MXCSR.DAZ is ignored and is treated as if it 0.
 * No denormal exception is reported on MXCSR.
 */

/* Clang defines the __F16C__ macro for x86 cpus which support F16C extension */

#ifdef __F16C__




#include <x86intrin.h>

/** FLOAT -> HALF vec4 ************************************************/

typedef union
{
  __m128 i;
  float4 low, hi;
} f2h4_i;

typedef union
{
  __m128i o;
  ushort4 low, hi;
} f2h4_o;

ushort4
_cl_float2half4_rte (const float4 data)
{
  f2h4_i ui;
  f2h4_o uo;
  ui.low = data;
  uo.o = _mm_cvtps_ph (ui.i, 0);
  return uo.low;
}

ushort4
_cl_float2half4_rtn (const float4 data)
{
  f2h4_i ui;
  f2h4_o uo;
  ui.low = data;
  uo.o = _mm_cvtps_ph (ui.i, 1);
  return uo.low;
}

ushort4
_cl_float2half4_rtp (const float4 data)
{
  f2h4_i ui;
  f2h4_o uo;
  ui.low = data;
  uo.o = _mm_cvtps_ph (ui.i, 2);
  return uo.low;
}

ushort4
_cl_float2half4_rtz (const float4 data)
{
  f2h4_i ui;
  f2h4_o uo;
  ui.low = data;
  uo.o = _mm_cvtps_ph (ui.i, 3);
  return uo.low;
}

ushort4
_cl_float2half4 (const float4 data)
{
  return _cl_float2half4_rte (data);
}

/** FLOAT -> HALF vec8 ************************************************/

typedef union
{
  __m256 i;
  float8 f;
} f2h8_i;

typedef union
{
  __m128i o;
  ushort8 f;
} f2h8_o;

ushort8
_cl_float2half8_rte (const float8 data)
{
  f2h8_i ui;
  f2h8_o uo;
  ui.f = data;
  uo.o = _mm256_cvtps_ph (ui.i, 0);
  return uo.f;
}

ushort8
_cl_float2half8_rtn (const float8 data)
{
  f2h8_i ui;
  f2h8_o uo;
  ui.f = data;
  uo.o = _mm256_cvtps_ph (ui.i, 1);
  return uo.f;
}

ushort8
_cl_float2half8_rtp (const float8 data)
{
  f2h8_i ui;
  f2h8_o uo;
  ui.f = data;
  uo.o = _mm256_cvtps_ph (ui.i, 2);
  return uo.f;
}

ushort8
_cl_float2half8_rtz (const float8 data)
{
  f2h8_i ui;
  f2h8_o uo;
  ui.f = data;
  uo.o = _mm256_cvtps_ph (ui.i, 3);
  return uo.f;
}

ushort8
_cl_float2half8 (const float8 data)
{
  return _cl_float2half8_rte (data);
}

/** HALF -> FLOAT vec4 ************************************************/

typedef union
{
  __m128i i;
  ushort4 low, hi;
} h2f4_i;

typedef union
{
  __m128 o;
  float4 f;
} h2f4_o;

float4
_cl_half2float4 (const ushort4 data)
{
  h2f4_i ui;
  h2f4_o uo;
  ui.low = data;
  uo.o = _mm_cvtph_ps (ui.i);
  return uo.f;
}

/** HALF -> FLOAT vec8 ************************************************/

typedef union
{
  __m128i i;
  ushort8 u;
} h2f8_i;

typedef union
{
  __m256 o;
  float8 f;
} h2f8_o;

float8
_cl_half2float8 (const ushort8 data)
{
  h2f8_i ui;
  h2f8_o uo;
  ui.u = data;
  uo.o = _mm256_cvtph_ps (ui.i);
  return uo.f;
}

#endif
