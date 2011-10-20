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

float __attribute__ ((overloadable))
dot(float a, float b)
{
  return a * b;
}

float __attribute__ ((overloadable))
dot(float2 a, float2 b)
{
  return a.s0 * b.s0 + a.s1 * b.s1;
}

float __attribute__ ((overloadable))
dot(float3 a, float3 b)
{
  return dot(a.s01, b.s01) + a.s2 * b.s2;
}

float __attribute__ ((overloadable))
dot(float4 a, float4 b)
{
  return dot(a.s01, b.s01) + dot(a.s23, b.s23);
}

float __attribute__ ((overloadable))
dot(float8 a, float8 b)
{
  return dot(a.s0123, b.s0123) + dot(a.s4567, b.s4567);
}

float __attribute__ ((overloadable))
dot(float16 a, float16 b)
{
  return dot(a.s01234567, b.s01234567) + dot(a.s89abcdef, b.s89abcdef);
}

double __attribute__ ((overloadable))
dot(double a, double b)
{
  return a * b;
}

double __attribute__ ((overloadable))
dot(double2 a, double2 b)
{
  return a.s0 * b.s0 + a.s1 * b.s1;
}

double __attribute__ ((overloadable))
dot(double3 a, double3 b)
{
  return dot(a.s01, b.s01) + a.s2 * b.s2;
}

double __attribute__ ((overloadable))
dot(double4 a, double4 b)
{
  return dot(a.s01, b.s01) + dot(a.s23, b.s23);
}

double __attribute__ ((overloadable))
dot(double8 a, double8 b)
{
  return dot(a.s0123, b.s0123) + dot(a.s4567, b.s4567);
}

double __attribute__ ((overloadable))
dot(double16 a, double16 b)
{
  return dot(a.s01234567, b.s01234567) + dot(a.s89abcdef, b.s89abcdef);
}
