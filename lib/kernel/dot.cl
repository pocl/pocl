/* OpenCL built-in library: dot()

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

float __attribute__ ((__overloadable__)) dot(float a, float b)
{
  return a * b;
}

float __attribute__ ((__overloadable__)) dot(float2 a, float2 b)
{
  return a.lo * b.lo + a.hi * b.hi;
}

float __attribute__ ((__overloadable__)) dot(float3 a, float3 b)
{
  return dot(a.s01, b.s01) + a.s2 * b.s2;
}

float __attribute__ ((__overloadable__)) dot(float4 a, float4 b)
{
  return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}

float __attribute__ ((__overloadable__)) dot(float8 a, float8 b)
{
  return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}

float __attribute__ ((__overloadable__)) dot(float16 a, float16 b)
{
  return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}

#ifdef cl_khr_fp64
double __attribute__ ((__overloadable__)) dot(double a, double b)
{
  return a * b;
}

double __attribute__ ((__overloadable__)) dot(double2 a, double2 b)
{
  return a.lo * b.lo + a.hi * b.hi;
}

double __attribute__ ((__overloadable__)) dot(double3 a, double3 b)
{
  return dot(a.s01, b.s01) + a.s2 * b.s2;
}

double __attribute__ ((__overloadable__)) dot(double4 a, double4 b)
{
  return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}

double __attribute__ ((__overloadable__)) dot(double8 a, double8 b)
{
  return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}

double __attribute__ ((__overloadable__)) dot(double16 a, double16 b)
{
  return dot(a.lo, b.lo) + dot(a.hi, b.hi);
}
#endif
