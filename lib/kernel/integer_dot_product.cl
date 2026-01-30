/* OpenCL built-in library: cl_khr_integer_dot_product

   Copyright (c) 2026 Michal Babej / Tampere University

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

#include "templates.h"

#define IMPLEMENT_DOT_PRODUCT(RTYPE, TYPE1, TYPE2, CONV1, CONV2)              \
  RTYPE _CL_OVERLOADABLE dot (TYPE1 arg1, TYPE2 arg2)                         \
  {                                                                           \
    RTYPE##4 res = convert_##RTYPE##4(CONV1 (arg1) * CONV2 (arg2));           \
    return res.x + res.y + res.z + res.w;                                     \
  }

IMPLEMENT_DOT_PRODUCT (uint, uchar4, uchar4, convert_uint4, convert_uint4)
IMPLEMENT_DOT_PRODUCT (int, uchar4, char4, convert_int4, convert_int4)
IMPLEMENT_DOT_PRODUCT (int, char4, uchar4, convert_int4, convert_int4)
IMPLEMENT_DOT_PRODUCT (int, char4, char4, convert_int4, convert_int4)

#define IMPLEMENT_DOT_PRODUCT_SAT(RTYPE, TYPE1, TYPE2, CONV1, CONV2)          \
  RTYPE _CL_OVERLOADABLE dot_acc_sat (TYPE1 arg1, TYPE2 arg2, RTYPE acc)      \
  {                                                                           \
    RTYPE##4 res = convert_##RTYPE##4(CONV1 (arg1) * CONV2 (arg2));           \
    RTYPE res_scalar = res.x + res.y + res.z + res.w;                         \
    return add_sat (res_scalar, acc);                                         \
  }

IMPLEMENT_DOT_PRODUCT_SAT (uint, uchar4, uchar4, convert_uint4, convert_uint4)
IMPLEMENT_DOT_PRODUCT_SAT (int, uchar4, char4, convert_int4, convert_int4)
IMPLEMENT_DOT_PRODUCT_SAT (int, char4, uchar4, convert_int4, convert_int4)
IMPLEMENT_DOT_PRODUCT_SAT (int, char4, char4, convert_int4, convert_int4)

uint _CL_OVERLOADABLE
dot_4x8packed_uu_uint (uint a, uint b)
{
  return dot (as_uchar4 (a), as_uchar4 (b));
}
int _CL_OVERLOADABLE
dot_4x8packed_ss_int (uint a, uint b)
{
  return dot (as_char4 (a), as_char4 (b));
}
int _CL_OVERLOADABLE
dot_4x8packed_us_int (uint a, uint b)
{
  return dot (as_uchar4 (a), as_char4 (b));
}
int _CL_OVERLOADABLE
dot_4x8packed_su_int (uint a, uint b)
{
  return dot (as_char4 (a), as_uchar4 (b));
}

uint _CL_OVERLOADABLE
dot_acc_sat_4x8packed_uu_uint (uint a, uint b, uint acc)
{
  return dot_acc_sat (as_uchar4 (a), as_uchar4 (b), acc);
}
int _CL_OVERLOADABLE
dot_acc_sat_4x8packed_ss_int (uint a, uint b, int acc)
{
  return dot_acc_sat (as_char4 (a), as_char4 (b), acc);
}
int _CL_OVERLOADABLE
dot_acc_sat_4x8packed_us_int (uint a, uint b, int acc)
{
  return dot_acc_sat (as_uchar4 (a), as_char4 (b), acc);
}
int _CL_OVERLOADABLE
dot_acc_sat_4x8packed_su_int (uint a, uint b, int acc)
{
  return dot_acc_sat (as_char4 (a), as_uchar4 (b), acc);
}
