/* OpenCL built-in library: nan()

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

/* Fall-back implementation which ignores the nancode */
// DEFINE_EXPR_V_U(nan, (vtype)NAN)

DEFINE_EXPR_V_U(nan,
                ({
                  utype nancode = a;
                  // number of bits in the mantissa
                  sutype mant_dig =
                    TYPED_CONST(sutype,
                                HALF_MANT_DIG, FLT_MANT_DIG, DBL_MANT_DIG) - 1;
                  // mask for the mantissa
                  sutype nan_mant_mask = (1 << mant_dig) - 1;
                  // mask out bits that can't be stored in the mantissa
                  // this also ensures the sign bit is zero,
                  // i.e. that the nan is quiet
                  nancode &= nan_mant_mask;
                  // ensure the nancode is not zero
                  nancode = nancode ? nancode : (sutype)1;
                  // create the exponent
                  sutype nan_exp =
                    TYPED_CONST(sutype,
                                HALF_MAX_EXP - HALF_MIN_EXP,
                                FLT_MAX_EXP - FLT_MIN_EXP,
                                DBL_MAX_EXP - DBL_MIN_EXP) + 2;
                  nan_exp <<= mant_dig;
                  // combine exponent and nancode
                  utype val = nan_exp | nancode;
                  _cl_nan_as_vtype(val);
                }))
