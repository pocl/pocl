/* OpenCL built-in library: tanpi_fp64.cl

   Copyright (c) 2017-2026 Michal Babej / Tampere University

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

_CL_OVERLOADABLE vtype tanpi(vtype x) {
  itype ix = as_itype (x);
  itype xsgn = ix & (itype)0x8000000000000000LL;
  itype xnsgn = xsgn ^ (itype)0x8000000000000000LL;
  ix ^= xsgn;
  vtype absx = fabs (x);
  itype iax = convert_itype (absx);
  vtype r = absx - convert_vtype (iax);
  itype xodd
    = xsgn
      ^ as_itype ((iax & (itype)0x1) != (itype)0 ? (itype)0x8000000000000000LL
                                                 : (itype)0);

  // Initialize with return for +-Inf and NaN
  itype ir = (itype)QNANBITPATT_DP64;

  // 2^53 <= |x| < Inf, the result is always even integer
  ir = ix < (itype)PINFBITPATT_DP64 ? xsgn : ir;

  // 2^52 <= |x| < 2^53, the result is always integer
  ir = ix < (itype)0x4340000000000000LL ? xodd : ir;

  // 0x1.0p-14 <= |x| < 2^53, result depends on which 0.25 interval

  // r < 1.0
  vtype a = (vtype)1.0 - r;
  itype e = (itype)0;
  itype s = xnsgn;

  // r <= 0.75
  itype c = r <= (vtype)0.75;
  vtype t = r - (vtype)0.5;
  a = c ? t : a;
  e = c ? (itype)1 : e;
  s = c ? xsgn : s;

  // r < 0.5
  c = r < (vtype)0.5;
  t = (vtype)0.5 - r;
  a = c ? t : a;
  s = c ? xnsgn : s;

  // r <= 0.25
  c = r <= (vtype)0.25;
  a = c ? r : a;
  e = c ? (itype)0 : e;
  s = c ? xsgn : s;

  vtype api = a * M_PI;
  vtype lo, hi;
  __pocl_tan_piby4 (api, (vtype)0.0, &lo, &hi);
  itype jr = s ^ as_itype (e != (itype)0 ? hi : lo);

  itype si = xodd | (itype)0x7ff0000000000000LL;
  jr = (r == (vtype)0.5) ? si : jr;

  ir = (ix < (itype)0x4330000000000000LL) ? jr : ir;

  return as_vtype (ir);
}
