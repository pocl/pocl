/* OpenCL built-in library: forward function calls with 3 arguments to libc

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

#define CONCAT(a,b) a##b

float CONCAT(TRIG,f)(float a, float b, float c);
double TRIG(double a, double b, double c);



float __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(float a, float b, float c)
{
  return CONCAT(TRIG,f)(a, b, c);
}

float2 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(float2 a, float2 b, float2 c)
{
  return (float2)(CONCAT(cl_,TRIG)(a.s0, b.s0, c.s0),
                  CONCAT(cl_,TRIG)(a.s1, b.s1, c.s1));
}

float3 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(float3 a, float3 b, float3 c)
{
  return (float3)(CONCAT(cl_,TRIG)(a.s01, b.s01, c.s01),
                  CONCAT(cl_,TRIG)(a.s2, b.s2, c.s2));
}

float4 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(float4 a, float4 b, float4 c)
{
  return (float4)(CONCAT(cl_,TRIG)(a.s01, b.s01, c.s01),
                  CONCAT(cl_,TRIG)(a.s23, b.s23, c.s23));
}

float8 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(float8 a, float8 b, float8 c)
{
  return (float8)(CONCAT(cl_,TRIG)(a.s0123, b.s0123, c.s0123),
                  CONCAT(cl_,TRIG)(a.s4567, b.s4567, c.s4567));
}

float16 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(float16 a, float16 b, float16 c)
{
  return (float16)(CONCAT(cl_,TRIG)(a.s01234567, b.s01234567, c.s01234567),
                   CONCAT(cl_,TRIG)(a.s89abcdef, b.s89abcdef, c.s89abcdef));
}



double __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(double a, double b, double c)
{
  return CONCAT(TRIG,f)(a, b, c);
}

double2 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(double2 a, double2 b, double2 c)
{
  return (double2)(CONCAT(cl_,TRIG)(a.s0, b.s0, c.s0),
                   CONCAT(cl_,TRIG)(a.s1, b.s1, c.s1));
}

double3 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(double3 a, double3 b, double3 c)
{
  return (double3)(CONCAT(cl_,TRIG)(a.s01, b.s01, c.s01),
                   CONCAT(cl_,TRIG)(a.s2, b.s2, c.s2));
}

double4 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(double4 a, double4 b, double4 c)
{
  return (double4)(CONCAT(cl_,TRIG)(a.s01, b.s01, c.s01),
                   CONCAT(cl_,TRIG)(a.s23, b.s23, c.s23));
}

double8 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(double8 a, double8 b, double8 c)
{
  return (double8)(CONCAT(cl_,TRIG)(a.s0123, b.s0123, c.s0123),
                   CONCAT(cl_,TRIG)(a.s4567, b.s4567, c.s4567));
}

double16 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(double16 a, double16 b, double16 c)
{
  return (double16)(CONCAT(cl_,TRIG)(a.s01234567, b.s01234567, c.s01234567),
                    CONCAT(cl_,TRIG)(a.s89abcdef, b.s89abcdef, c.s89abcdef));
}
