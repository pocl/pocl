/* OpenCL built-in library: length()

   Copyright (c) 2011 Erik Schnetter <eschnetter@perimeterinstitute.ca>
                      Perimeter Institute for Theoretical Physics
   Copyright (c) 2014 Advanced Micro Devices, Inc.
   Copyright (c) 2017 Michal Babej / Tampere University of Technology
   Copyright (c) 2026 Google LLC

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

// Scalars
_CL_OVERLOADABLE float length(float p) {
  return fabs(p);
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CL_OVERLOADABLE double length(double p) {
  return fabs(p);
}
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CL_OVERLOADABLE half length(half p) {
  return fabs(p);
}
#endif


// Macro for robust vector length (with scaling)
#define IMPLEMENT_LENGTH_V(VTYPE, STYPE, MIN_VAL, SCALE_UP, SCALE_DOWN, INF_SCALE_UP, INF_SCALE_DOWN) \
_CL_OVERLOADABLE STYPE length(VTYPE p) { \
  STYPE l2 = dot(p, p); \
  if (l2 < MIN_VAL) { \
    p *= SCALE_UP; \
    return sqrt(dot(p, p)) * SCALE_DOWN; \
  } else if (l2 == INFINITY) { \
    p *= INF_SCALE_DOWN; \
    return sqrt(dot(p, p)) * INF_SCALE_UP; \
  } \
  return sqrt(l2); \
}

// Float vectors
IMPLEMENT_LENGTH_V(float2, float, FLT_MIN, 0x1.0p+86F, 0x1.0p-86F, 0x1.0p+65F, 0x1.0p-65F)
IMPLEMENT_LENGTH_V(float3, float, FLT_MIN, 0x1.0p+86F, 0x1.0p-86F, 0x1.0p+65F, 0x1.0p-65F)
IMPLEMENT_LENGTH_V(float4, float, FLT_MIN, 0x1.0p+86F, 0x1.0p-86F, 0x1.0p+65F, 0x1.0p-65F)
IMPLEMENT_LENGTH_V(float8, float, FLT_MIN, 0x1.0p+86F, 0x1.0p-86F, 0x1.0p+65F, 0x1.0p-65F)
IMPLEMENT_LENGTH_V(float16, float, FLT_MIN, 0x1.0p+86F, 0x1.0p-86F, 0x1.0p+65F, 0x1.0p-65F)

// Double vectors
#ifdef cl_khr_fp64
IMPLEMENT_LENGTH_V(double2, double, DBL_MIN, 0x1.0p+563, 0x1.0p-563, 0x1.0p+513, 0x1.0p-513)
IMPLEMENT_LENGTH_V(double3, double, DBL_MIN, 0x1.0p+563, 0x1.0p-563, 0x1.0p+513, 0x1.0p-513)
IMPLEMENT_LENGTH_V(double4, double, DBL_MIN, 0x1.0p+563, 0x1.0p-563, 0x1.0p+513, 0x1.0p-513)
IMPLEMENT_LENGTH_V(double8, double, DBL_MIN, 0x1.0p+563, 0x1.0p-563, 0x1.0p+513, 0x1.0p-513)
IMPLEMENT_LENGTH_V(double16, double, DBL_MIN, 0x1.0p+563, 0x1.0p-563, 0x1.0p+513, 0x1.0p-513)
#endif

// Macro for half vectors (casting to float)
#define IMPLEMENT_LENGTH_HALF_V(VTYPE, FLOAT_VTYPE) \
_CL_OVERLOADABLE half length(VTYPE p) { \
  return (half)length(convert_##FLOAT_VTYPE(p)); \
}

// Half vectors
#ifdef cl_khr_fp16
IMPLEMENT_LENGTH_HALF_V(half2, float2)
IMPLEMENT_LENGTH_HALF_V(half3, float3)
IMPLEMENT_LENGTH_HALF_V(half4, float4)
IMPLEMENT_LENGTH_HALF_V(half8, float8)
IMPLEMENT_LENGTH_HALF_V(half16, float16)
#endif

#undef IMPLEMENT_LENGTH_V
#undef IMPLEMENT_LENGTH_HALF_V
