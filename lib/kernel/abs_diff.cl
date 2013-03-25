/* OpenCL built-in library: abs_diff()

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

// DEFINE_EXPR_UG_GG(abs_diff, abs(a-b))

// This could probably also be optimised
DEFINE_EXPR_UG_GG(abs_diff,
                  (sgtype)-1 < (sgtype)0 ?
                  /* signed */
                  ({
                    (a^b) >= (gtype)0 ?
                      /* same sign: no overflow/underflow */
                      abs(a-b) :
                      /* different signs */
                      abs(a) + abs(b);
                  }) :
                  /* unsigned */
                  ({
                    /* This abs prevents a type error; it is not
                       exectued for signed types, and is a no-op for
                       unsigned types */
                    abs(a > b ? a-b : b-a);
                  }))
