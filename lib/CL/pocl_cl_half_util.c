/* cl_half_util.c - collection of helpful functions for manipulating cl_half
   datatypes.

   Copyright (c) 2017 Michal Babej / Tampere University of Technology
                 2024 Robin Bijl / Tampere University

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "pocl_cl_half_util.h"

static int const shift = 13;
static int const shiftSign = 16;

static int32_t const infN = 0x7F800000;  /* flt32 infinity */
static int32_t const maxN = 0x477FE000;  /* max flt16 normal as a flt32 */
static int32_t const minN = 0x38800000;  /* min flt16 normal as a flt32 */
static int32_t const signN = 0x80000000; /* flt32 sign bit */

/* static int32_t const infC = infN >> shift;
 * static int32_t const infC = 0x3FC00;
 * static int32_t const nanN = (infC + 1) << shift; // minimum flt16 nan as a
 * flt32
 */
static int32_t const nanN = 0x7f802000;
/* static int32_t const maxC = maxN >> shift; */
static int32_t const maxC = 0x23bff;
/* static int32_t const minC = minN >> shift;
 * static int32_t const minC = 0x1c400;
 * static int32_t const signC = signN >> shiftSign; // flt16 sign bit
 */
static int32_t const signC = 0x40000; /* flt16 sign bit */

static int32_t const mulN = 0x52000000; /* (1 << 23) / minN */
static int32_t const mulC = 0x33800000; /* minN / (1 << (23 - shift)) */

static int32_t const subC = 0x003FF; /* max flt32 subnormal down shifted */
static int32_t const norC = 0x00400; /* min flt32 normal down shifted */

/* static int32_t const maxD = infC - maxC - 1; */
static int32_t const maxD = 0x1c000;
/* static int32_t const minD = minC - subC - 1; */
static int32_t const minD = 0x1c000;

typedef union
{
  float f;
  int32_t si;
  uint32_t ui;
} H2F_Bits;

float
pocl_half_to_float (uint16_t value)
{
  H2F_Bits v;
  v.ui = value;
  int32_t sign = v.si & signC;
  v.si ^= sign;
  sign <<= shiftSign;
  v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
  v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
  H2F_Bits s;
  s.si = mulC;
  s.f *= v.si;
  int32_t mask = -(norC > v.si);
  v.si <<= shift;
  v.si ^= (s.si ^ v.si) & mask;
  v.si |= sign;
  return v.f;
}

uint16_t
pocl_float_to_half (float value)
{
  H2F_Bits v, s;
  v.f = value;
  uint32_t sign = v.si & signN;
  v.si ^= sign;
  sign >>= shiftSign;
  s.si = mulN;
  s.si = s.f * v.f;
  v.si ^= (s.si ^ v.si) & -(minN > v.si);
  v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
  v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
  v.ui >>= shift;
  v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
  v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
  return v.ui | sign;
}
