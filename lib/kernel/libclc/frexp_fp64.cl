/* OpenCL built-in library: frexp()

   Copyright (c) 2017 Michal Babej / Tampere University of Technology
   Copyright (c) 2024 Michal Babej / Intel Finland Oy
   Copyright (c) 2024 LLVM Project / Apache License version 2.0

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

_CL_OVERLOADABLE vtype frexp(vtype x, inttype ADDRSPACE *exp)
{
  itype i = as_itype (x);
  itype ai = i & (itype)(0x7fffffffffffffffL);
  itype d = (ai > (itype)0) & (ai < (itype)0x0010000000000000L);
  // scale subnormal by 2^54 without multiplying
  vtype s = as_vtype (ai | (itype)(0x0370000000000000L)) - (vtype)(0x1.0p-968);
  ai = select (ai, as_itype (s), d);
  itype e = (ai >> 52) - (itype)1022 - select ((itype)0, (itype)54, d);
  itype t = (ai == (itype)0) | (e == (itype)1025);
  i = (i & (itype)(0x8000000000000000L)) | (itype)(0x3fe0000000000000L)
      | (ai & (itype)(0x000fffffffffffffL));
  *exp = convert_inttype (select (e, (itype)0L, t));
  return select (as_vtype (i), x, t);
}
