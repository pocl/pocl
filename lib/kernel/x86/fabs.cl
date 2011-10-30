/* OpenCL built-in library: fabs()

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

#include "../templates.h"

DEFINE_EXPR_V_V(fabs,
                ({
                  int bits = CHAR_BIT * sizeof(stype);
                  jtype sign_mask = (jtype)1 << (jtype)(bits - 1);
                  jtype result = ~sign_mask & *(jtype*)&a;
                  *(vtype*)&result;
                }))

// TODO: Use these explicitly, until llvm generates efficient code
// 
// float fabs_ff(float a, float b)
// {
//   __asm__ ("andss %[b], %[a]" : [a] "=x" (a) : "[a]" (a), [b] "x" (b));
//   return a;
// }
// 
// float4 fabs_f4f4(float4 a, float4 b)
// {
//   __asm__ ("andps %[b], %[a]" : [a] "=x" (a) : "[a]" (a), [b] "x" (b));
//   return a;
// }
// 
// double fabs_dd(double a, double b)
// {
//   __asm__ ("andsd %[b], %[a]" : [a] "=x" (a) : "[a]" (a), [b] "x" (b));
//   return a;
// }
// 
// double2 fabs_d2d2(double2 a, double2 b)
// {
//   __asm__ ("andpd %[b], %[a]" : [a] "=x" (a) : "[a]" (a), [b] "x" (b));
//   return a;
// }
