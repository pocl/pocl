/* OpenCL built-in library: frexp()

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

#define FAST_TRUNC(x) convert_vtype(convert_itype(x))

_CL_OVERLOADABLE vtype round(vtype d)
{
  vtype x = d + (vtype)0.5f;
  vtype fr = x - (vtype)(1UL << 31) * FAST_TRUNC(x * (1.0 / (1UL << 31)));
  fr = fr - FAST_TRUNC(fr);
  x = (fr == (vtype)0 && x <= (vtype)0) ? (x - (vtype)(1f)) : x;
  fr = (fr < (vtype)0) ? fr+(vtype)(1.0) : fr;
  x = (d == 0.49999999999999994449) ? (vtype)(0.0) : x;  // nextafterf(0.5, 0)
  return (isinf(d) || fabs(d) >= (vtype)(1UL << 52)) ? d : copysign(x - fr, d);
}
