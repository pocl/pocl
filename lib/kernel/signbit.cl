/* OpenCL built-in library: signbit()

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
   FITNESS FOR A PARTICULAR PURPOSE AND NONORDEREDRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/



/* Correct, but probably slower */
/* DEFINE_BUILTIN_L_V(signbit) */



#define IMPLEMENT_SIGNBIT_BUILTIN_FLOAT  __builtin_signbitf(a)
#define IMPLEMENT_SIGNBIT_BUILTIN_DOUBLE __builtin_signbit(a)
#define IMPLEMENT_SIGNBIT_DIRECT                \
  ({                                            \
    int bits = CHAR_BIT * sizeof(stype);        \
    *(jtype*)&a >> (jtype)(bits-1);             \
  })

#define IMPLEMENT_DIRECT(NAME, VTYPE, STYPE, JTYPE, EXPR)       \
  JTYPE _CL_OVERLOADABLE NAME(VTYPE a)                          \
  {                                                             \
    typedef VTYPE vtype;                                        \
    typedef STYPE stype;                                        \
    typedef JTYPE jtype;                                        \
    return EXPR;                                                \
  }



IMPLEMENT_DIRECT(signbit, float  , float, int  , IMPLEMENT_SIGNBIT_BUILTIN_FLOAT)
IMPLEMENT_DIRECT(signbit, float2 , float, int2 , IMPLEMENT_SIGNBIT_DIRECT)
IMPLEMENT_DIRECT(signbit, float3 , float, int3 , IMPLEMENT_SIGNBIT_DIRECT)
IMPLEMENT_DIRECT(signbit, float4 , float, int4 , IMPLEMENT_SIGNBIT_DIRECT)
IMPLEMENT_DIRECT(signbit, float8 , float, int8 , IMPLEMENT_SIGNBIT_DIRECT)
IMPLEMENT_DIRECT(signbit, float16, float, int16, IMPLEMENT_SIGNBIT_DIRECT)

#ifdef cl_khr_fp64
IMPLEMENT_DIRECT(signbit, double  , double, int   , IMPLEMENT_SIGNBIT_BUILTIN_DOUBLE)
IMPLEMENT_DIRECT(signbit, double2 , double, long2 , IMPLEMENT_SIGNBIT_DIRECT)
IMPLEMENT_DIRECT(signbit, double3 , double, long3 , IMPLEMENT_SIGNBIT_DIRECT)
IMPLEMENT_DIRECT(signbit, double4 , double, long4 , IMPLEMENT_SIGNBIT_DIRECT)
IMPLEMENT_DIRECT(signbit, double8 , double, long8 , IMPLEMENT_SIGNBIT_DIRECT)
IMPLEMENT_DIRECT(signbit, double16, double, long16, IMPLEMENT_SIGNBIT_DIRECT)
#endif
