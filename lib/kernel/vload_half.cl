/* OpenCL built-in library: vload_half()

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

float
_cl_half2float (ushort h)
{
  const uint w = convert_uint(h) << 16;
  const uint sign = w & (uint)(0x80000000);
  const uint two_w = w + w;
  const uint exp_offset = (uint)(0xE0) << 23;
  const float exp_scale = 0x1.0p-112f;
  const float normalized_value = as_float((two_w >> 4) + exp_offset) * exp_scale;
  const uint magic_mask = (uint)(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value = as_float((two_w >> 17) | magic_mask) - magic_bias;
  const uint denormalized_cutoff = (uint)(1) << 27;
  const uint result = sign |
             (two_w < denormalized_cutoff ? as_uint(denormalized_value) : as_uint(normalized_value));
  return as_float(result);
}

#ifdef __F16C__

float4 _cl_half2float4 (const ushort4 data);
float8 _cl_half2float8 (const ushort8 data);

#define IMPLEMENT_VLOAD_HALF(MOD)                                             \
                                                                              \
  float _CL_OVERLOADABLE vload_half (size_t offset, const MOD half *p)        \
  {                                                                           \
    return _cl_half2float (((const MOD ushort *)p)[offset]);                  \
  }                                                                           \
                                                                              \
  float2 _CL_OVERLOADABLE vload_half2 (size_t offset, const MOD half *p)      \
  {                                                                           \
    return (float2) (vload_half (offset * 2, p),                              \
                     vload_half (offset * 2 + 1, p));                         \
  }                                                                           \
                                                                              \
  float3 _CL_OVERLOADABLE vload_half3 (size_t offset, const MOD half *p)      \
  {                                                                           \
    return (float3) (vload_half (offset * 3, p),                              \
                     vload_half (offset * 3 + 1, p),                          \
                     vload_half (offset * 3 + 2, p));                         \
  }                                                                           \
                                                                              \
  float4 _CL_OVERLOADABLE vload_half4 (size_t offset, const MOD half *p)      \
  {                                                                           \
    return _cl_half2float4 (((const MOD ushort4 *)p)[offset]);                \
  }                                                                           \
                                                                              \
  float8 _CL_OVERLOADABLE vload_half8 (size_t offset, const MOD half *p)      \
  {                                                                           \
    return _cl_half2float8 (((const MOD ushort8 *)p)[offset]);                \
  }                                                                           \
                                                                              \
  float16 _CL_OVERLOADABLE vload_half16 (size_t offset, const MOD half *p)    \
  {                                                                           \
    float8 hi = _cl_half2float8 (((const MOD ushort8 *)p)[offset * 2]);       \
    float8 lo = _cl_half2float8 (((const MOD ushort8 *)p)[offset * 2 + 1]);   \
    return (float16) (hi, lo);                                                \
  }                                                                           \
                                                                              \
  float _CL_OVERLOADABLE vloada_half (size_t offset, const MOD half *p)       \
  {                                                                           \
    return _cl_half2float (((const MOD ushort *)p)[offset]);                  \
  }                                                                           \
                                                                              \
  float2 _CL_OVERLOADABLE vloada_half2 (size_t offset, const MOD half *p)     \
  {                                                                           \
    return (float2) (vloada_half (offset * 2, p),                             \
                     vloada_half (offset * 2, p + 1));                        \
  }                                                                           \
                                                                              \
  float3 _CL_OVERLOADABLE vloada_half3 (size_t offset, const MOD half *p)     \
  {                                                                           \
    float4 tmp = vloada_half4 (offset, p);                                    \
    return (float3) (tmp.xyz);                                                \
  }                                                                           \
                                                                              \
  float4 _CL_OVERLOADABLE vloada_half4 (size_t offset, const MOD half *p)     \
  {                                                                           \
    return _cl_half2float4 (((const MOD ushort4 *)p)[offset]);                \
  }                                                                           \
                                                                              \
  float8 _CL_OVERLOADABLE vloada_half8 (size_t offset, const MOD half *p)     \
  {                                                                           \
    return _cl_half2float8 (((const MOD ushort8 *)p)[offset]);                \
  }                                                                           \
                                                                              \
  float16 _CL_OVERLOADABLE vloada_half16 (size_t offset, const MOD half *p)   \
  {                                                                           \
    float8 hi = _cl_half2float8 (((const MOD ushort8 *)p)[offset * 2]);       \
    float8 lo = _cl_half2float8 (((const MOD ushort8 *)p)[offset * 2 + 1]);   \
    return (float16) (hi, lo);                                                \
  }                                                                           \
                                                                              \
// __F16C__
#else

#define IMPLEMENT_VLOAD_HALF(MOD)                                             \
                                                                              \
  float _CL_OVERLOADABLE vload_half (size_t offset, const MOD half *p)        \
  {                                                                           \
    ushort h = ((const MOD ushort *)p)[offset];                               \
    return _cl_half2float (h);                                                \
  }                                                                           \
                                                                              \
  float2 _CL_OVERLOADABLE vload_half2 (size_t offset, const MOD half *p)      \
  {                                                                           \
    return (float2) (vload_half (offset * 2, p),                              \
                     vload_half (offset * 2 + 1, p));                         \
  }                                                                           \
                                                                              \
  float3 _CL_OVERLOADABLE vload_half3 (size_t offset, const MOD half *p)      \
  {                                                                           \
    return (float3) (vload_half (offset * 3, p),                              \
                     vload_half (offset * 3 + 1, p),                          \
                     vload_half (offset * 3 + 2, p));                         \
  }                                                                           \
                                                                              \
  float4 _CL_OVERLOADABLE vload_half4 (size_t offset, const MOD half *p)      \
  {                                                                           \
    return (float4) (vload_half2 (offset * 2, p),                             \
                     vload_half2 (offset * 2 + 1, p));                        \
  }                                                                           \
                                                                              \
  float8 _CL_OVERLOADABLE vload_half8 (size_t offset, const MOD half *p)      \
  {                                                                           \
    return (float8) (vload_half4 (offset * 2, p),                             \
                     vload_half4 (offset * 2 + 1, p));                        \
  }                                                                           \
                                                                              \
  float16 _CL_OVERLOADABLE vload_half16 (size_t offset, const MOD half *p)    \
  {                                                                           \
    return (float16) (vload_half8 (offset * 2, p),                            \
                      vload_half8 (offset * 2 + 1, p));                       \
  }                                                                           \
                                                                              \
  float _CL_OVERLOADABLE vloada_half (size_t offset, const MOD half *p)       \
  {                                                                           \
    return _cl_half2float (((const MOD ushort *)p)[offset]);                  \
  }                                                                           \
                                                                              \
  float2 _CL_OVERLOADABLE vloada_half2 (size_t offset, const MOD half *p)     \
  {                                                                           \
    return (float2) (vloada_half (offset * 2, p),                             \
                     vloada_half (offset * 2 + 1, p));                        \
  }                                                                           \
                                                                              \
  float3 _CL_OVERLOADABLE vloada_half3 (size_t offset, const MOD half *p)     \
  {                                                                           \
    float4 tmp = vloada_half4 (offset, p);                                    \
    return (float3) (tmp.xyz);                                                \
  }                                                                           \
                                                                              \
  float4 _CL_OVERLOADABLE vloada_half4 (size_t offset, const MOD half *p)     \
  {                                                                           \
    return (float4) (vloada_half2 (offset * 2, p),                            \
                     vloada_half2 (offset * 2 + 1, p));                       \
  }                                                                           \
                                                                              \
  float8 _CL_OVERLOADABLE vloada_half8 (size_t offset, const MOD half *p)     \
  {                                                                           \
    return (float8) (vloada_half4 (offset * 2, p),                            \
                     vloada_half4 (offset * 2 + 1, p));                       \
  }                                                                           \
                                                                              \
  float16 _CL_OVERLOADABLE vloada_half16 (size_t offset, const MOD half *p)   \
  {                                                                           \
    return (float16) (vloada_half8 (offset * 2, p),                           \
                      vloada_half8 (offset * 2 + 1, p));                      \
  }

#endif




IMPLEMENT_VLOAD_HALF(__global)
IMPLEMENT_VLOAD_HALF(__local)
IMPLEMENT_VLOAD_HALF(__constant)
IMPLEMENT_VLOAD_HALF(__private)
#ifdef __opencl_c_generic_address_space
IMPLEMENT_VLOAD_HALF(__generic)
#endif
