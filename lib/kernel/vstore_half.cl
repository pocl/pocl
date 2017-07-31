/* OpenCL built-in library: vstore_half()

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
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



/*
  half:        1 sign bit,  5 exponent bits,  10 mantissa bits, exponent offset 15
  float:       1 sign bit,  8 exponent bits,  23 mantissa bits, exponent offset 127
  double:      1 sign bit, 10 exponent bits,  53 mantissa bits, exponent offset 1023
  long double: 1 sign bit, 15 exponent bits, 112 mantissa bits, exponent offset 16383
*/

// Clang supports "half" only on ARM
// TODO: Create autoconf test for this
#ifdef __ARM_ARCH

ushort _cl_float2half(float data)
{
  half hdata = data;
  return *(const ushort*)&hdata;
}

#else

#define HALF_MAXPLUS 0x1.ffdp15f /* "one more" than HALF_MAX */
#undef HALF_MIN
#define HALF_MIN     0x1.0p-14f
#define HALF_ZERO    ((short)0x0000) /* zero */
#define HALF_INF     ((short)0x4000) /* infinity */
#define HALF_SIGN    ((short)0x8000) /* sign bit */

ushort _cl_float2half(float data)
{
  /* IDEA: modify data (e.g. add "1/2") to round correctly */
  uint fval = as_uint(data);
  uint fsign = (fval & 0x80000000U) >> 31U;
  uint fexp = (fval & 0x7f800000U) >> 23U;
  uint fmant = fval & 0x007fffffU;
  bool isdenorm = fexp == 0U;
  bool isinfnan = fexp == 255U;
  fexp -= 127U;
  ushort hsign = (ushort)fsign << (ushort)15;
  ushort hexp = (__builtin_expect(isdenorm, false) ? (ushort)0 :
                 __builtin_expect(isinfnan, false) ? (ushort)31 :
                 (ushort)fexp + (ushort)15);
  /* TODO: this always truncates */
  ushort hmant = (ushort)(fmant >> 13);
  ushort hval;
  if (__builtin_expect(fabs(data) >= HALF_MAXPLUS, false)) {
    hval = signbit(data)==0 ? HALF_INF : HALF_INF | HALF_SIGN;
  } else if (__builtin_expect(fabs(data) < HALF_MIN, false)) {
    hval = signbit(data)==0 ? HALF_ZERO : HALF_ZERO | HALF_SIGN;
  } else {
    hval = hsign | hexp | hmant;
  }
  return hval;
}

#endif



#define IMPLEMENT_VSTORE_HALF(MOD, SUFFIX)                              \
                                                                        \
  void _CL_OVERLOADABLE                                                 \
  vstore_half##SUFFIX(float data, size_t offset, MOD half *p)           \
  {                                                                     \
    ((MOD ushort*)p)[offset] = _cl_float2half(data);                    \
  }                                                                     \
                                                                        \
  void _CL_OVERLOADABLE                                                 \
  vstore_half2##SUFFIX(float2 data, size_t offset, MOD half *p)         \
  {                                                                     \
    vstore_half##SUFFIX(data.lo, 0, &p[offset*2]);                      \
    vstore_half##SUFFIX(data.hi, 0, &p[offset*2+1]);                    \
  }                                                                     \
                                                                        \
  void _CL_OVERLOADABLE                                                 \
  vstore_half3##SUFFIX(float3 data, size_t offset, MOD half *p)         \
  {                                                                     \
    vstore_half2##SUFFIX(data.lo, 0, &p[offset*3]);                     \
    vstore_half##SUFFIX(data.s2, 0, &p[offset*3+2]);                    \
  }                                                                     \
                                                                        \
  void _CL_OVERLOADABLE                                                 \
  vstore_half4##SUFFIX(float4 data, size_t offset, MOD half *p)         \
  {                                                                     \
    vstore_half2##SUFFIX(data.lo, 0, &p[offset*4]);                     \
    vstore_half2##SUFFIX(data.hi, 0, &p[offset*4+2]);                   \
  }                                                                     \
                                                                        \
  void _CL_OVERLOADABLE                                                 \
  vstore_half8##SUFFIX(float8 data, size_t offset, MOD half *p)         \
  {                                                                     \
    vstore_half4##SUFFIX(data.lo, 0, &p[offset*8]);                     \
    vstore_half4##SUFFIX(data.hi, 0, &p[offset*8+4]);                   \
  }                                                                     \
                                                                        \
  void _CL_OVERLOADABLE                                                 \
  vstore_half16##SUFFIX(float16 data, size_t offset, MOD half *p)       \
  {                                                                     \
    vstore_half8##SUFFIX(data.lo, 0, &p[offset*16]);                    \
    vstore_half8##SUFFIX(data.hi, 0, &p[offset*16+8]);                  \
  }                                                                     \
                                                                        \
  void _CL_OVERLOADABLE                                                 \
  vstorea_half2##SUFFIX(float2 data, size_t offset, MOD half *p)        \
  {                                                                     \
    vstore_half##SUFFIX(data.lo, 0, &p[offset*2]);                      \
    vstore_half##SUFFIX(data.hi, 0, &p[offset*2+1]);                    \
  }                                                                     \
                                                                        \
  void _CL_OVERLOADABLE                                                 \
  vstorea_half3##SUFFIX(float3 data, size_t offset, MOD half *p)        \
  {                                                                     \
    vstorea_half2##SUFFIX(data.lo, 0, &p[offset*3]);                    \
    vstore_half##SUFFIX(data.s2, 0, &p[offset*3+2]);                    \
  }                                                                     \
                                                                        \
  void _CL_OVERLOADABLE                                                 \
  vstorea_half4##SUFFIX(float4 data, size_t offset, MOD half *p)        \
  {                                                                     \
    vstorea_half2##SUFFIX(data.lo, 0, &p[offset*4]);                    \
    vstorea_half2##SUFFIX(data.hi, 0, &p[offset*4+2]);                  \
  }                                                                     \
                                                                        \
  void _CL_OVERLOADABLE                                                 \
  vstorea_half8##SUFFIX(float8 data, size_t offset, MOD half *p)        \
  {                                                                     \
    vstorea_half4##SUFFIX(data.lo, 0, &p[offset*8]);                    \
    vstorea_half4##SUFFIX(data.hi, 0, &p[offset*8+4]);                  \
  }                                                                     \
                                                                        \
  void _CL_OVERLOADABLE                                                 \
  vstorea_half16##SUFFIX(float16 data, size_t offset, MOD half *p)      \
  {                                                                     \
    vstorea_half8##SUFFIX(data.lo, 0, &p[offset*16]);                   \
    vstorea_half8##SUFFIX(data.hi, 0, &p[offset*16+8]);                 \
  }



IMPLEMENT_VSTORE_HALF(__global  ,     )
IMPLEMENT_VSTORE_HALF(__global  , _rte)
IMPLEMENT_VSTORE_HALF(__global  , _rtz)
IMPLEMENT_VSTORE_HALF(__global  , _rtp)
IMPLEMENT_VSTORE_HALF(__global  , _rtn)
IMPLEMENT_VSTORE_HALF(__local   ,     )
IMPLEMENT_VSTORE_HALF(__local   , _rte)
IMPLEMENT_VSTORE_HALF(__local   , _rtz)
IMPLEMENT_VSTORE_HALF(__local   , _rtp)
IMPLEMENT_VSTORE_HALF(__local   , _rtn)
IMPLEMENT_VSTORE_HALF(__private ,     )
IMPLEMENT_VSTORE_HALF(__private , _rte)
IMPLEMENT_VSTORE_HALF(__private , _rtz)
IMPLEMENT_VSTORE_HALF(__private , _rtp)
IMPLEMENT_VSTORE_HALF(__private , _rtn)
