/* OpenCL built-in library: expfrexp()

   Copyright (c) 2017 Michal Babej / Tampere University of Technology

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


_CL_OVERLOADABLE _CL_ALWAYSINLINE itype
_cl_expfrexp(vtype x)
{
  itype ret = (itype)0;

  // denorms
  itype cond = (fabs(x) < (vtype)FLT_MIN);
  x = cond ? (x * 0x1p30f) : x;
  ret = cond ? (ret - (itype)30) : ret;

//  ret += (as_itype((as_utype(x) >> 23) & (utype)0xFF) - (itype)0x7E);

  ret += (as_itype( (as_utype(x) << 1) >> 24 ) - (itype)0x7E);

  ret = (x == (vtype)0.0f) ? (itype)0 : ret;
  ret = (isnan(x) | isinf(x)) ? (itype)0 : ret;
  return ret;
}
