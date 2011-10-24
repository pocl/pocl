/* OpenCL built-in library: forward function calls to libc

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

float CONCAT(TRIG,f)(float a);
double TRIG(double a);



float __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(float a)
{
  return CONCAT(TRIG,f)(a);
}

float2 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(float2 a)
{
  return (float2)(CONCAT(cl_,TRIG)(a.s0),
                  CONCAT(cl_,TRIG)(a.s1));
}

float3 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(float3 a)
{
  return (float3)(CONCAT(cl_,TRIG)(a.s01),
                  CONCAT(cl_,TRIG)(a.s2));
}

float4 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(float4 a)
{
  return (float4)(CONCAT(cl_,TRIG)(a.s01),
                  CONCAT(cl_,TRIG)(a.s23));
}

float8 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(float8 a)
{
  return (float8)(CONCAT(cl_,TRIG)(a.s0123),
                  CONCAT(cl_,TRIG)(a.s4567));
}

float16 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(float16 a)
{
  return (float16)(CONCAT(cl_,TRIG)(a.s01234567),
                   CONCAT(cl_,TRIG)(a.s89abcdef));
}



double __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(double a)
{
  return TRIG(a);
}

double2 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(double2 a)
{
  return (double2)(CONCAT(cl_,TRIG)(a.s0),
                   CONCAT(cl_,TRIG)(a.s1));
}

double3 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(double3 a)
{
  return (double3)(CONCAT(cl_,TRIG)(a.s01),
                   CONCAT(cl_,TRIG)(a.s2));
}

double4 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(double4 a)
{
  return (double4)(CONCAT(cl_,TRIG)(a.s01),
                   CONCAT(cl_,TRIG)(a.s23));
}

double8 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(double8 a)
{
  return (double8)(CONCAT(cl_,TRIG)(a.s0123),
                   CONCAT(cl_,TRIG)(a.s4567));
}

double16 __attribute__ ((overloadable))
CONCAT(cl_,TRIG)(double16 a)
{
  return (double16)(CONCAT(cl_,TRIG)(a.s01234567),
                    CONCAT(cl_,TRIG)(a.s89abcdef));
}
