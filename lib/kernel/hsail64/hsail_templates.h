/* OpenCL built-in library: builtin implementation templates for HSAIL

   Copyright (c) 2011-2013 Erik Schnetter
   Copyright (c) 2015 Michal Babej

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


#define IMPLEMENT_BUILTIN_V_V(NAME, VTYPE, LO, HI)      \
  VTYPE __attribute__ ((overloadable))                  \
  NAME(VTYPE a)                                         \
  {                                                     \
    return (VTYPE)(NAME(a.LO), NAME(a.HI));             \
  }


#define DEFINE_BUILTIN_V_V_32(NAME, BUILTIN)            \
  __attribute__((overloadable))  float NAME(float f) __asm("llvm.hsail." #BUILTIN ".f32");   \
  IMPLEMENT_BUILTIN_V_V(NAME, float2  , lo, hi)         \
  IMPLEMENT_BUILTIN_V_V(NAME, float4  , lo, hi)         \
  IMPLEMENT_BUILTIN_V_V(NAME, float3  , lo, s2)         \
  IMPLEMENT_BUILTIN_V_V(NAME, float8  , lo, hi)         \
  IMPLEMENT_BUILTIN_V_V(NAME, float16 , lo, hi)         \


#define DEFINE_BUILTIN_V_V(NAME, BUILTIN)               \
  DEFINE_BUILTIN_V_V_32(NAME, BUILTIN)                  \
  __IF_FP64(                                            \
  __attribute__((overloadable))  double NAME(double f) __asm("llvm.hsail." #BUILTIN ".f64");   \
  IMPLEMENT_BUILTIN_V_V(NAME, double2 , lo, hi)         \
  IMPLEMENT_BUILTIN_V_V(NAME, double3 , lo, s2)         \
  IMPLEMENT_BUILTIN_V_V(NAME, double4 , lo, hi)         \
  IMPLEMENT_BUILTIN_V_V(NAME, double8 , lo, hi)         \
  IMPLEMENT_BUILTIN_V_V(NAME, double16, lo, hi))
