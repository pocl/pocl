/* OpenCL built-in library: sin()

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

#undef sin

float sinf(float a);
double sin(double a);



float __attribute__ ((overloadable))
cl_sin(float a)
{
  return sinf(a);
}

float2 __attribute__ ((overloadable))
cl_sin(float2 a)
{
  return (float2)(cl_sin(a.s0), cl_sin(a.s1));
}

float3 __attribute__ ((overloadable))
cl_sin(float3 a)
{
  return (float3)(cl_sin(a.s01), cl_sin(a.s2));
}

float4 __attribute__ ((overloadable))
cl_sin(float4 a)
{
  return (float4)(cl_sin(a.s01), cl_sin(a.s23));
}

float8 __attribute__ ((overloadable))
cl_sin(float8 a)
{
  return (float8)(cl_sin(a.s0123), cl_sin(a.s4567));
}

float16 __attribute__ ((overloadable))
cl_sin(float16 a)
{
  return (float16)(cl_sin(a.s01234567), cl_sin(a.s89abcdef));
}

double __attribute__ ((overloadable))
cl_sin(double a)
{
  return sin(a);
}

double2 __attribute__ ((overloadable))
cl_sin(double2 a)
{
  return (double2)(cl_sin(a.s0), cl_sin(a.s1));
}

double3 __attribute__ ((overloadable))
cl_sin(double3 a)
{
  return (double3)(cl_sin(a.s01), cl_sin(a.s2));
}

double4 __attribute__ ((overloadable))
cl_sin(double4 a)
{
  return (double4)(cl_sin(a.s01), cl_sin(a.s23));
}

double8 __attribute__ ((overloadable))
cl_sin(double8 a)
{
  return (double8)(cl_sin(a.s0123), cl_sin(a.s4567));
}

double16 __attribute__ ((overloadable))
cl_sin(double16 a)
{
  return (double16)(cl_sin(a.s01234567), cl_sin(a.s89abcdef));
}
