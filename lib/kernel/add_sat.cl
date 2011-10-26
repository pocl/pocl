/* OpenCL built-in library: add_sat()

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
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

// This could do with some testing
// This could probably also be optimised (i.e. the ?: operators eliminated)
DEFINE_EXPR_G_GG(add_sat,
                 (sgtype)-1 < (sgtype)0 ?
                 /* signed */
                 ({
                   gtype min = (sgtype)(sizeof(sgtype)==1 ? CHAR_MIN :
                                        sizeof(sgtype)==2 ? SHRT_MIN :
                                        sizeof(sgtype)==4 ? INT_MIN :
                                        sizeof(sgtype)==8 ? LONG_MIN :
                                        0);
                   gtype max = (sgtype)(sizeof(sgtype)==1 ? CHAR_MAX :
                                        sizeof(sgtype)==2 ? SHRT_MAX :
                                        sizeof(sgtype)==4 ? INT_MAX :
                                        sizeof(sgtype)==8 ? LONG_MAX :
                                        0);
                   (a^b) < (gtype)0 ?
                     /* different signs: all is fine */
                     a+b :
                     /* same sign: test for overflow */
                     a >= (gtype)0 ?
                     /* positive: compare to max */
                     (a > max-b ? max : a+b) :
                     /* negative: compare to min */
                     (a < min-b ? min : a+b);
                 }) :
                 /* unsigned */
                 ({
                   gtype max = (sgtype)-1;
                   a > max-b ? max : a+b;
                 }))
