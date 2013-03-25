/* OpenCL built-in library: mul_hi()

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



// Decompose a and b into their upper and lower halves.
//    C is 2^(N/2), where N is the number of bits in our datatype.
//    al, bl, ah, bh are signed (if the datatype is signed).
//    signed:   0 <= xl < C;   -C/2 <= xh < C/2
//    unsigned: 0 <= xl, xh < C
// This yields:
//    a = ah C + al
//    b = bh C + bl
// The multiplication can then be written as:
//    a b = (ah C + al) (bh C + bl)
//        = ah bh C C + (ah bl + al bh) C + al bl
// Note that none of the multiplications can overflow.

#define SLO(x)                                                          \
  ((gtype)(x) & (gtype)(((sgtype)1 << (sgtype)(bits/2)) - (sgtype)1))
#define SHI(x) ((gtype)(x) >> (sgtype)(bits/2))
#define SHI1(x) ((gtype)(x) >> (sgtype)(bits/2-1))
#define SUHI(x) ((ugtype)(x) >> (sugtype)(bits/2))
#define SCOMBINE(hi,lo)                                         \
  ((gtype)((gtype)((hi) << (sgtype)(bits/2)) + (gtype)(lo)))

#define ULO(x)                                                          \
  ((gtype)(x) & (gtype)(((sgtype)1 << (sgtype)(bits/2)) - (sgtype)1))
#define UHI(x) ((gtype)(x) >> (sgtype)(bits/2))
#define UHI1(x) ((gtype)(x) >> (sgtype)(bits/2-1))
#define UCOMBINE(hi,lo)                                         \
  ((gtype)((gtype)((hi) << (sgtype)(bits/2)) + (gtype)(lo)))

#define U2S(x) ({ union { gtype s; ugtype u; } conv; conv.u=(x); conv.s; })
#define S2U(x) ({ union { gtype s; ugtype u; } conv; conv.s=(x); conv.u; })



DEFINE_EXPR_G_GG(mul_hi,
                 (sgtype)-1 < (sgtype)0 ?
                 /* signed */
                 ({
                   int bits = CHAR_BIT * sizeof(sgtype);
                   gtype ah = SHI(a);
                   gtype al = SLO(a);
                   gtype bh = SHI(b);
                   gtype bl = SLO(b);
                   gtype ahbh = ah * bh;
                   gtype ahbl = ah * bl;
                   gtype albh = al * bh;
                   ugtype albl = S2U(al) * S2U(bl);
                   // gtype abll = SLO(albl);
                   // gtype ablh = SLO(abhl + albh + SHI(albl));
                   // gtype abhl = SLO(ahbh + SHI(abhl + albh + SHI(albl)));
                   // gtype abhh = SHI(ahbh + SHI(abhl + albh + SHI(albl)));
                   gtype abh =
                     ahbh + SHI1(hadd(ahbl, (gtype)(albh + U2S(SUHI(albl)))));
                   abh;
                 }) :
                 /* unsigned */
                 ({
                   int bits = CHAR_BIT * sizeof(sgtype);
                   gtype ah = UHI(a);
                   gtype al = ULO(a);
                   gtype bh = UHI(b);
                   gtype bl = ULO(b);
                   gtype ahbh = ah * bh;
                   gtype ahbl = ah * bl;
                   gtype albh = al * bh;
                   gtype albl = al * bl;
                   // gtype abll = ULO(albl);
                   // gtype ablh = ULO(abhl + albh + UHI(albl));
                   // gtype abhl = ULO(ahbh + UHI(abhl + albh + UHI(albl)));
                   // gtype abhh = UHI(ahbh + UHI(abhl + albh + UHI(albl)));
                   gtype abh =
                     ahbh + UHI1(hadd(ahbl, (gtype)(albh + UHI(albl))));
                   abh;
                 }))
