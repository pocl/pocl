/* OpenCL built-in library: step()

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

#define USING_ASTYPE_HELPERS
#include "templates.h"

// This segfaults Clang 3.0, so we work around
// DEFINE_EXPR_V_VV(step, b < a ? (vtype)0.0 : (vtype)1.0)

DEFINE_EXPR_V_VV(step,
                 ({
                   jtype zero = 0;
                   jtype one  = 1;
                   jtype result = b < a ? zero : one;
                   _cl_step_as_vtype(result);
                 }))

// DEFINE_EXPR_V_VV(step, (vtype)0.5 + copysign((vtype)0.5, b - a))

DEFINE_EXPR_V_SV(step, step((vtype)a, b))
