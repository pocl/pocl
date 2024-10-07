/* OpenCL built-in library: isinf()

   Copyright (c) 2024 Henry Linjamäki / Intel Finland Oy

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

int __attribute__((overloadable)) isinf(half a) {
  return __builtin_isinf(a);
}

IMPLEMENT_BUILTIN_L_V(isinf, short2, half2, half, lo, hi)
IMPLEMENT_BUILTIN_L_V(isinf, short3, half3, half, lo, s2)
IMPLEMENT_BUILTIN_L_V(isinf, short4, half4, half, lo, hi)
IMPLEMENT_BUILTIN_L_V(isinf, short8, half8, half, lo, hi)
IMPLEMENT_BUILTIN_L_V(isinf, short16, half16, half, lo, hi)
