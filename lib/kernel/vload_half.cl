/* OpenCL built-in library: vload_half()

   Copyright (c) 2011 Erik Schnetter <eschnetter@perimeterinstitute.ca>
                      Perimeter Institute for Theoretical Physics
   
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



#ifdef cl_khr_fp16

/*
  half:   1 sign bit,  5 exponent bits, 10 mantissa bits
  float:  1 sign bit,  8 exponent bits, 23 mantissa bits
  double: 1 sign bit, 10 exponent bits, 53 mantissa bits
*/



#define IMPLEMENT_VLOAD_HALF(MOD)                                       \
                                                                        \
  float _CL_OVERLOADABLE                                                \
  vload_half(size_t offset, const MOD half *p)                          \
  {                                                                     \
    /* This conversion always succeeds */                               \
    short hval = ((const MOD short*)p)[offset];                         \
    short hsign = (hval & (short)0x8000) >> (short)15;                  \
    short hexp = (hval & (short)0x7c00) >> (short)10;                   \
    short hmant = hval & (short)0x03ff;                                 \
    bool isdenorm = hexp == (short)0;                                   \
    bool isinfnan = hexp == (short)31;                                  \
    hexp -= (short)15;                                                  \
    int fsign = (int)hsign << 31;                                       \
    int fexp = (__builtin_expect(isdenorm, false) ? 0 :                 \
                __builtin_expect(isinfnan, false) ? 255 : (int)hexp + 127); \
    fexp <<= 23;                                                        \
    int fmant = (int)hmant << 13;                                       \
    int fval = fsign | fexp | fmant;                                    \
    return as_float(fval);                                              \
  }                                                                     \
                                                        \
  float2 _CL_OVERLOADABLE                               \
  vload_half2(size_t offset, const MOD half *p)         \
  {                                                     \
    return (float2)(vload_half(0, &p[offset*2]),        \
                    vload_half(0, &p[offset*2+1]));     \
  }                                                     \
                                                        \
  float3 _CL_OVERLOADABLE                               \
  vload_half3(size_t offset, const MOD half *p)         \
  {                                                     \
    return (float3)(vload_half2(0, &p[offset*3]),       \
                    vload_half(0, &p[offset*3+2]));     \
  }                                                     \
                                                        \
  float4 _CL_OVERLOADABLE                               \
  vload_half4(size_t offset, const MOD half *p)         \
  {                                                     \
    return (float4)(vload_half2(0, &p[offset*4]),       \
                    vload_half2(0, &p[offset*4+2]));    \
  }                                                     \
                                                        \
  float8 _CL_OVERLOADABLE                               \
  vload_half8(size_t offset, const MOD half *p)         \
  {                                                     \
    return (float8)(vload_half4(0, &p[offset*8]),       \
                    vload_half4(0, &p[offset*8+4]));    \
  }                                                     \
                                                        \
  float16 _CL_OVERLOADABLE                              \
  vload_half16(size_t offset, const MOD half *p)        \
  {                                                     \
    return (float16)(vload_half8(0, &p[offset*16]),     \
                     vload_half8(0, &p[offset*16+8]));  \
  }                                                     \
                                                        \
  float2 _CL_OVERLOADABLE                               \
  vloada_half2(size_t offset, const MOD half *p)        \
  {                                                     \
    return (float2)(vload_half(0, &p[offset*2]),        \
                    vload_half(0, &p[offset*2+1]));     \
  }                                                     \
                                                        \
  float3 _CL_OVERLOADABLE                               \
  vloada_half3(size_t offset, const MOD half *p)        \
  {                                                     \
    return (float3)(vloada_half2(0, &p[offset*4]),      \
                    vload_half(0, &p[offset*4+2]));     \
  }                                                     \
                                                        \
  float4 _CL_OVERLOADABLE                               \
  vloada_half4(size_t offset, const MOD half *p)        \
  {                                                     \
    return (float4)(vloada_half2(0, &p[offset*4]),      \
                    vloada_half2(0, &p[offset*4+2]));   \
  }                                                     \
                                                        \
  float8 _CL_OVERLOADABLE                               \
  vloada_half8(size_t offset, const MOD half *p)        \
  {                                                     \
    return (float8)(vloada_half4(0, &p[offset*8]),      \
                    vloada_half4(0, &p[offset*8+4]));   \
  }                                                     \
                                                        \
  float16 _CL_OVERLOADABLE                              \
  vloada_half16(size_t offset, const MOD half *p)       \
  {                                                     \
    return (float16)(vloada_half8(0, &p[offset*16]),    \
                     vloada_half8(0, &p[offset*16+8])); \
  }



IMPLEMENT_VLOAD_HALF(__global)
IMPLEMENT_VLOAD_HALF(__local)
IMPLEMENT_VLOAD_HALF(__constant)
/* IMPLEMENT_VLOAD_HALF(__private) */

#endif
