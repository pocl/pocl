/* OpenCL built-in library: sqrt()

   Copyright (c) 2015 Michal Babej / Tampere University of Technology

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

#include "hsail_templates.h"

/* TODO:

   Clang does not recognize __builtin_sqrt* in CGBuiltin.cpp (around line 1285)
   to be a call to sqrt and cannot selectively convert it to the llvm.sqrt.*
   intrinsics with looser math flags. Therefore we have to call it as a libcall
   so it regonized it. For the double we still call the __builtin_sqrt() to
   disambiguate from the sqrt() to avoid infinite recursion. Probably the
   correct fix is to patch CGBuiltin.cpp to recognize also the call via
   __builtin_sqrt*. */

/* TODO HSAIL:
   related to previous TODO, neither __builtin_sqrt() nor any
  "high-level" intrinsics (llvm.sqrt, llvm.hsail.sqrt) works.

  I have implemented a workaround using bitcode in sqrt_default.ll, this however
  uses intrinsics with fixed behaviour (rounding mode etc.) so it's not a long
  term solution.
*/

#undef sqrt

float _CL_OVERLOADABLE _sqrt_default_f32(float a);
IMPLEMENT_VECWITHSCALARS(_sqrt_default_f32, V_V, float, float)
double _CL_OVERLOADABLE _sqrt_default_f64(double a);
IMPLEMENT_VECWITHSCALARS(_sqrt_default_f64, V_V, double, double)

IMPLEMENT_EXPR_VECS_AND_SCALAR(_cl_sqrt, V_V, EXPR_, float, int,_sqrt_default_f32(a) )
IMPLEMENT_EXPR_VECS_AND_SCALAR(_cl_sqrt, V_V, EXPR_, double, long, _sqrt_default_f64(a) )
