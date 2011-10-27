/* OpenCL built-in library: mad_sat()

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

// Decompose a and b into their upper and lower halves.
//    C is 2^(N/2), where N is the number of bits in our datatype.
//    al, bl, ah, bh are signed (if the datatype is signed).
// This yields:
//    a = ah C + al
//    b = bh C + bl
//    c = ch C + cl
// The mad operation can then be written as:
//    a b + c = (ah C + al) (bh C + bl) + (ch C + cl)
//            = ah bh C C + (ah bl + al bh + ch) C + al bl + cl
// Note that none of the multiplications can overflow.

#define SHI(x) (((x) - SLO(x)) >> (sgtype)(bits/2))
#define SLO(x) (((x) << (sgtype)(bits/2)) >> (sgtype)(bits/2))
#define SCOMBINE(hi,lo) (((hi) << (sgtype)(bits/2)) + (lo))

#define UHI(x) ((x) >> (sgtype)(bits/2))
#define ULO(x) ((x) & (sgtype)(bits/2 - 1))
#define UCOMBINE(hi,lo) (((hi) << (sgtype)(bits/2)) | (lo))

DEFINE_EXPR_G_GGG(mad_sat,
                  (sgtype)-1 < (sgtype)0 ?
                  /* signed */
                  ({
                    int bits = CHAR_BIT * sizeof(sgtype);
                    gtype min = (sgtype)1 << (sgtype)(bits-1);
                    gtype max = min + (sgtype)1;
                    gtype ah = SHI(a);
                    gtype al = SLO(a);
                    gtype bh = SHI(b);
                    gtype bl = SLO(b);
                    gtype ch = SHI(c);
                    gtype cl = SLO(c);
                    gtype ahbh = ah * bh;
                    gtype ahbl = ah * bl;
                    gtype albh = al * bh;
                    gtype albl = al * bl;
                    gtype abcll = SLO(albl + cl);
                    gtype abclh = SLO(ahbl + albh + ch + SHI(albl + cl));
                    gtype abch = ahbh + SHI(ahbl + albh + ch + SHI(albl + cl));
                    abch == (gtype)0 ?
                      /* no overflow */
                      SCOMBINE(abclh, abcll) :
                      /* overflow */
                      abch>=(gtype)0 ? max : min;
                  }) :
                  /* unsigned */
                  ({
                    int bits = CHAR_BIT * sizeof(sgtype);
                    gtype max = (sgtype)-1;
                    gtype ah = UHI(a);
                    gtype al = ULO(a);
                    gtype bh = UHI(b);
                    gtype bl = ULO(b);
                    gtype ch = UHI(c);
                    gtype cl = ULO(c);
                    gtype ahbh = ah * bh;
                    gtype ahbl = ah * bl;
                    gtype albh = al * bh;
                    gtype albl = al * bl;
                    gtype abcll = ULO(albl + cl);
                    gtype abclh = ULO(ahbl + albh + ch + UHI(albl + cl));
                    gtype abch = ahbh + UHI(ahbl + albh + ch + UHI(albl + cl));
                    abch == (gtype)0 ?
                      /* no overflow */
                      UCOMBINE(abclh, abcll) :
                      /* overflow */
                      max;
                  }))
