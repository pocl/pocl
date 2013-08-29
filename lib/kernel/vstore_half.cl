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



#define IMPLEMENT_VSTORE_HALF(MOD, SUFFIX)                              \
                                                                        \
  void _CL_OVERLOADABLE                                                 \
  vstore_half##SUFFIX(float data, size_t offset, MOD half *p)           \
  {                                                                     \
    p[offset] = data;                                                   \
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
/* IMPLEMENT_VSTORE_HALF(__private ,     ) */
/* IMPLEMENT_VSTORE_HALF(__private , _rte) */
/* IMPLEMENT_VSTORE_HALF(__private , _rtz) */
/* IMPLEMENT_VSTORE_HALF(__private , _rtp) */
/* IMPLEMENT_VSTORE_HALF(__private , _rtn) */
