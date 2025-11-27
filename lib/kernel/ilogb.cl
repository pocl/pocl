/* OpenCL built-in library: ilogb()

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

#define IMPLEMENT_EXPR_J_V(NAME, EXPR, VTYPE, STYPE, RTYPE, JTYPE)     \
RTYPE __attribute__ ((overloadable))                                  \
convert_rtype(JTYPE X) { return convert_##RTYPE(X); };                                     \
RTYPE __attribute__ ((overloadable))                                  \
    NAME(VTYPE a)                                                         \
{                                                                     \
        typedef VTYPE vtype;                                                \
        typedef STYPE stype;                                                \
        typedef JTYPE jtype;                                                \
        typedef RTYPE rtype;                                              \
        union { vtype v; jtype j; } aa;                                   \
        aa.v = fabs(a);                                                    \
        return EXPR;                                                        \
}
#define DEFINE_EXPR_J_V(NAME, EXPR)                                     \
__IF_FP16(                                                            \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, half    , half  , int,   ushort)      \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, half2   , half  , int2 , ushort2)      \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, half3   , half  , int3 , ushort3)      \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, half4   , half  , int4 , ushort4)      \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, half8   , half  , int8 , ushort8)      \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, half16  , half  , int16, ushort16))     \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, float   , float , int  , uint  )      \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, float2  , float , int2 , uint2  )      \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, float3  , float , int3 , uint3  )      \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, float4  , float , int4 , uint4  )      \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, float8  , float , int8 , uint8  )      \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, float16 , float , int16, uint16  )      \
__IF_FP64(                                                            \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, double  , double, int   , ulong)      \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, double2 , double, int2  , ulong2)      \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, double3 , double, int3  , ulong3)      \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, double4 , double, int4  , ulong4)      \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, double8 , double, int8  , ulong8)      \
    IMPLEMENT_EXPR_J_V(NAME, EXPR, double16, double, int16 , ulong16))


DEFINE_EXPR_J_V(ilogb,
({
  (sizeof(stype) == 2) ? convert_rtype((aa.j >> 14) + (jtype)(15)) :
  (sizeof(stype) == 4) ? convert_rtype((aa.j >> 9) + (jtype)(127)) :
  (sizeof(stype) == 8) ? convert_rtype((aa.j >> 12) + (jtype)(1023)) :
  0;
}))

/*
DEFINE_EXPR_V_VV(maxmag,
                 ({
                     fabs(a) > fabs(b) ? a :
                         fabs(b) > fabs(a) ? b :
                         fmax(a, b);
                 }))
*/
