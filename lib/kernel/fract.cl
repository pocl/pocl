/* OpenCL built-in library: fract()

   Copyright (c) 2011 Erik Schnetter <eschnetter@perimeterinstitute.ca>
                      Perimeter Institute for Theoretical Physics
   
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

#include "templates.h"


#ifdef cl_khr_fp64
DEFINE_EXPR_V_VPV (fract, ({
                     vtype fl = select ((vtype)floor (a), (vtype)NAN,
                                        (itype)isnan (a));
                     fl = select ((vtype)fl, (vtype)a, (itype)isinf (a));
                     *b = fl;
                     vtype ret = fmin (a - floor (a),
                                       (vtype) (sizeof (stype) == 4
                                                    ? 0x1.fffffep-1f
                                                    : 0x1.fffffffffffffp-1));
                     ret = select ((vtype)ret, (vtype)0.0, (itype)isinf (a));
                     select ((vtype)ret, (vtype) (NAN), (itype)isnan (a));
                   }))
#else
DEFINE_EXPR_V_VPV (fract, ({
                     vtype fl = select ((vtype)floor (a), (vtype)NAN,
                                        (itype)isnan (a));
                     fl = select ((vtype)fl, (vtype)0.0f, (itype)isinf (a));
                     *b = fl;
                     vtype ret = fmin (a - floor (a), (vtype)0x1.fffffep-1f);
                     select ((vtype)ret, (vtype)NAN, (itype)isnan (a));
                   }))
#endif
