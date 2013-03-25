/* OpenCL built-in library: sub_sat()

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

// Available SSE2 builtins:
//    char     __builtin_ia32_psubsb128
//    short    __builtin_ia32_psubsw128
//    uchar    __builtin_ia32_psubusb128
//    ushort   __builtin_ia32_psubusw128
// Other types don't seem to be supported.

DEFINE_EXPR_G_GG(sub_sat,
                 (sgtype)-1 < (sgtype)0 ?
                 /* signed */
                 ({
                   int bits = CHAR_BIT * sizeof(sgtype);
                   gtype min = (sgtype)1 << (sgtype)(bits-1);
                   gtype max = min - (sgtype)1;
                   (a^b) >= (gtype)0 ?
                     /* same sign: no overflow/underflow */
                     a-b :
                     a >= (gtype)0 ?
                     /* a positive, b negative: can overflow */
                     (a > max+b ? max : a-b) :
                     /* a negative, b positive: can underflow */
                     (a < min+b ? min : a-b);
                 }) :
                 /* unsigned */
                 ({
                   gtype min = (sgtype)0;
                   a < min+b ? min : a-b;
                 }))
