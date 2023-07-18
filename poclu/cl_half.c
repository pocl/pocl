/**
 * \brief cl_half - helper functions for the cl_half datatype

   Copyright (c) 2013 Pekka Jääskeläinen / Tampere University of Technology
                      Timo Viitanen / Tampere University of Technology
   Copyright (c) 2015 Matias Koskela / Tampere University of Technology

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

   \file
*/

// for exp2
#define _ISOC99_SOURCE

#include <math.h>
#include <stdint.h>

#include "pocl_opencl.h"

/**
 * \brief union to store data when converting float to half.
 */
typedef union
{
  int32_t i;
  float f;
} FloatConvUnion;

/**
 * \todo not exposed via poclu.h should it be added or removed?
 */
cl_half
poclu_float_to_cl_half_fast(float value)
{
  FloatConvUnion u;
  u.f = value;
  cl_half half = (u.i & 0x007FFFFF) >> 13;
  half |=(u.i & 0x07800000) >> 13;
  half |=(u.i & 0x40000000) >> 16;
  half |=(u.i & 0x80000000) >> 16;
  return half;
}

/**
 * \todo not exposed via poclu.h should it be added or removed?
 */
cl_half
poclu_float_to_cl_half(float value)
{
  FloatConvUnion u;
  u.f = value;
  cl_half half = (u.i >> 16) & 0x8000; // sign
  cl_half fraction = (u.i >> 12) & 0x007ff; // fraction with extra bit for rounding
  cl_half exponent = (u.i >> 23)  & 0xff; // exponent

  if(exponent < 0x0067) // Return signed zero if zero or value is too small for denormal half
    return half;

  if(exponent > 0x008e){// value was NaN or Inf
    half |= 0x7c00u; // Make into inf
    half |= exponent == 255 && (u.i & 0x007fffffu); // If value was NaN make this into NaN
    return half;
  }

  if(exponent < 0x0071){// Denormal
    fraction |= 0x0800u;

    // rounding
    half |= (fraction >> (0x0072 - exponent)) + ((fraction >> (0x0071 - exponent)) & 1);
    return half;
  }

  half |= ((exponent - 0x0070) << 10) | (fraction >> 1);
  half += fraction & 1;// rounding
  return half;
}

/**
 * \todo not exposed via poclu.h should it be added or removed?
 */
cl_half
poclu_float_to_cl_half_ceil(float value)
{
  FloatConvUnion u;
  u.f = value;
  cl_half half = (u.i >> 16) & 0x8000; // sign
  int negative = half;
  cl_half fraction = (u.i >> 13) & 0x003ff;
  int32_t fractionLeftOver =  u.i & 0x00001fff;
  cl_half exponent = (u.i >> 23)  & 0xff; // exponent

  if(exponent < 0x0067) // Return signed zero if zero or too small denormal for half
    return half;

  if(exponent > 0x008e){// value was NaN or Inf
    half |= 0x7c00u; // Make into inf
    half |= exponent == 255 && (u.i & 0x007fffffu); // If value was NaN make this into NaN
    return half;
  }

  if(exponent < 0x0071){// Denormal
    fraction |= 0x0400u;
    fraction >>= 0x0071 - exponent;
    if(!negative && fractionLeftOver)
      fraction += 1; // Round up

    half |=fraction;
    return half;
  }

  half |= ((exponent - 0x0070) << 10) | fraction;
  if(!negative && fractionLeftOver)
    half += 1; // Round up

  return half;
}

/**
 * \todo not exposed via poclu.h should it be added or removed?
 */
cl_half
poclu_float_to_cl_half_floor(float value)
{
  FloatConvUnion u;
  u.f = value;
  cl_half half = (u.i >> 16) & 0x8000; // sign
  int negative = half;
  cl_half fraction = (u.i >> 13) & 0x003ff;
  int32_t fractionLeftOver =  u.i & 0x00001fff;
  cl_half exponent = (u.i >> 23)  & 0xff; // exponent

  if(exponent < 0x0067) // Return signed zero if zero or too small denormal for half
    return half;

  if(exponent > 0x008e){// value was NaN or Inf
    half |= 0x7c00u; // Make into inf
    half |= exponent == 255 && (u.i & 0x007fffffu); // If value was NaN make this into NaN
    return half;
  }

  if(exponent < 0x0071){// Denormal
    fraction |= 0x0400u;
    fraction >>= 0x0071 - exponent;
    if(negative && fractionLeftOver)
      fraction += 1; // Round up
    half |=fraction;
    return half;
  }

  half |= ((exponent - 0x0070) << 10) | fraction;
  if(negative && fractionLeftOver)
    half += 1; // Round up

  return half;
}

#ifndef INFINITY
#define INFINITY 1.0/0.0
#endif

#ifndef NAN
#define NAN 0.0/0.0
#endif

float
poclu_cl_half_to_float(cl_half value)
{
  if (value == 0xFC00) {
    return -INFINITY;
  }
  if (value == 0x7C00) {
    return INFINITY;
  }

  int sgn = ((value & 0x8000) >> 15);
  int exp = (value & 0x7C00) >> 10;
  int mant = value & 0x03FF;

  if (exp == 0x1F && mant != 0) {
    return NAN;
  }

  float v = (exp == 0) ? mant : mant | 0x0400; // 1.x if not denormal
  v /= 0x400;
  float mul = exp2((float)exp - 15);
  v *= mul;
  if (sgn) {
    v *= -1;
  }
  return v;
}
