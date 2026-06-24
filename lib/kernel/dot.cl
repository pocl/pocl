/* OpenCL built-in library: dot()

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

float _CL_OVERLOADABLE dot(float a, float b)
{
  return a * b;
}

float _CL_OVERLOADABLE dot(float2 a, float2 b)
{
  return a.lo * b.lo + a.hi * b.hi;
}

float _CL_OVERLOADABLE dot(float3 a, float3 b)
{
  return dot(a.s01, b.s01) + a.s2 * b.s2;
}

float _CL_OVERLOADABLE dot(float4 a, float4 b)
{
  return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}

float _CL_OVERLOADABLE dot(float8 a, float8 b)
{
  return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}

float _CL_OVERLOADABLE dot(float16 a, float16 b)
{
  return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}

#ifdef cl_khr_fp64
double _CL_OVERLOADABLE dot(double a, double b)
{
  return a * b;
}

double _CL_OVERLOADABLE dot(double2 a, double2 b)
{
  return a.lo * b.lo + a.hi * b.hi;
}

double _CL_OVERLOADABLE dot(double3 a, double3 b)
{
  return dot(a.s01, b.s01) + a.s2 * b.s2;
}

double _CL_OVERLOADABLE dot(double4 a, double4 b)
{
  return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}

double _CL_OVERLOADABLE dot(double8 a, double8 b)
{
  return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}

double _CL_OVERLOADABLE dot(double16 a, double16 b)
{
  return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}
#endif

#ifdef cl_khr_fp16
half _CL_OVERLOADABLE dot(half a, half b)
{
  return a * b;
}

half _CL_OVERLOADABLE dot(half2 a, half2 b)
{
  return a.lo * b.lo + a.hi * b.hi;
}

half _CL_OVERLOADABLE dot(half3 a, half3 b)
{
  return dot(a.s01, b.s01) + a.s2 * b.s2;
}

half _CL_OVERLOADABLE dot(half4 a, half4 b)
{
  return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}

half _CL_OVERLOADABLE dot(half8 a, half8 b)
{
  return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}

half _CL_OVERLOADABLE dot(half16 a, half16 b)
{
  return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}
#endif

