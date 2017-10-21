/* OpenCL built-in library: singlevec.h

   Copyright (c) 2017 Michal Babej / Tampere University of Technology

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

/* These macros help dealing with differences between true-false representation
 * between clang extended vectors (-1, 0) and scalar expressions (1, 0). */

#undef SV_ANY
#undef SV_NOT
#undef SV_AND
#undef SV_OR
#undef SV_ODD32
#undef SV_ODD64

#ifdef SINGLEVEC

#define SV_ANY(BOOL) (BOOL)
#define SV_NOT(x) (!(x))
#define SV_OR ||
#define SV_AND &&
#define SV_ODD32(x) (x & 1)
#define SV_ODD64(x) (x & 1)

#else

#define SV_ANY(BOOL) (any(BOOL))
#define SV_NOT(x) (~(x))
#define SV_OR |
#define SV_AND &
#define SV_ODD32(x) (x << 31)
#define SV_ODD64(x) (x << 63)

#endif
