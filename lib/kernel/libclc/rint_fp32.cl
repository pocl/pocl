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

_CL_OVERLOADABLE vtype rint(vtype d)
{
  vtype x = d + (vtype)0.5f;
  itype isodd = ((itype)1 & convert_itype(x)) ? (itype)-1 : (itype)0;
  vtype fr = x - FAST_TRUNC(x);

  fr = ((fr < (vtype)0) | ((fr == (vtype)0) & isodd)) ? fr+(vtype)(1.0f) : fr;

  x = (d == (vtype)0.50000005960464477539f) ? (vtype)(0) : x;  // nextafterf(0.5, 1)

  return (isinf(d) || fabs(d) >= (vtype)(1UL << 23)) ? d : copysign(x - fr, d);
}
