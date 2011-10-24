/* OpenCL built-in library: minmag()

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

#undef minmag



#define CL_DEFINE_FUNC2(NAME, EXPR)                                     \
                                                                        \
  float __attribute__ ((overloadable)) cl_##NAME(float a, float b)      \
  {                                                                     \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  float2 __attribute__ ((overloadable)) cl_##NAME(float2 a, float2 b)   \
  {                                                                     \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  float3 __attribute__ ((overloadable)) cl_##NAME(float3 a, float3 b)   \
  {                                                                     \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  float4 __attribute__ ((overloadable)) cl_##NAME(float4 a, float4 b)   \
  {                                                                     \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  float8 __attribute__ ((overloadable)) cl_##NAME(float8 a, float8 b)   \
  {                                                                     \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  float16 __attribute__ ((overloadable)) cl_##NAME(float16 a, float16 b) \
  {                                                                     \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double __attribute__ ((overloadable)) cl_##NAME(double a, double b)   \
  {                                                                     \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double2 __attribute__ ((overloadable)) cl_##NAME(double2 a, double2 b) \
  {                                                                     \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double3 __attribute__ ((overloadable)) cl_##NAME(double3 a, double3 b) \
  {                                                                     \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double4 __attribute__ ((overloadable)) cl_##NAME(double4 a, double4 b) \
  {                                                                     \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double8 __attribute__ ((overloadable)) cl_##NAME(double8 a, double8 b) \
  {                                                                     \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double16 __attribute__ ((overloadable)) cl_##NAME(double16 a, double16 b) \
  {                                                                     \
    return EXPR;                                                        \
  }



CL_DEFINE_FUNC2(minmag, fmin(fabs(a), fabs(b)))
