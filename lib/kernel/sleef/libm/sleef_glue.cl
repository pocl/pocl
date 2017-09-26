/* OpenCL built-in library: a few hand-written SLEEF indirect function calls

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

#include "sleef_cl.h"

#ifdef SLEEF_VEC_128_AVAILABLE

_CL_ALWAYSINLINE double2 Sleef_ldexpd2_long (double2 x, long2 k);
_CL_ALWAYSINLINE long2 Sleef_ilogbd2_long (double2 x);

_CL_ALWAYSINLINE double2
Sleef_ldexpd2 (double2 x, int2 k)
{
  int4 tmp = (int4) (k, k);
  return Sleef_ldexpd2_long (x, as_long2 (tmp));
}

_CL_ALWAYSINLINE int2
Sleef_ilogbd2 (double2 x)
{
  int4 r = as_int4 (Sleef_ilogbd2_long (x));
  return r.xy;
}

_CL_ALWAYSINLINE long2 Sleef_expfrexpd2_long (double2 x);

_CL_ALWAYSINLINE int2
Sleef_expfrexpd2 (double2 x)
{
  return convert_int2 (Sleef_expfrexpd2_long (x));
}

_CL_ALWAYSINLINE long4 Sleef_expfrexpd4_long (double4 x);

_CL_ALWAYSINLINE int4
Sleef_expfrexpd4 (double4 x)
{
  return convert_int4 (Sleef_expfrexpd4_long (x));
}

_CL_ALWAYSINLINE long8 Sleef_expfrexpd8_long (double8 x);

_CL_ALWAYSINLINE int8
Sleef_expfrexpd8 (double8 x)
{
  return convert_int8 (Sleef_expfrexpd8_long (x));
}

#endif
