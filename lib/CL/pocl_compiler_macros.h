/* pocl_compiler_macros.h - collection of macros to tell the compiler that a
   warning for a piece of code is deliberate.

   Copyright (c) 2024 Robin Bijl / Tampere University

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#ifndef POCL_POCL_COMPILER_MACROS_H
#define POCL_POCL_COMPILER_MACROS_H

#if (defined(__GNUC__) && (__GNUC__ > 6)) || (defined(__clang__) && __clang_major__ >= 12)
#define POCL_FALLTHROUGH __attribute__((fallthrough))
#else
#define POCL_FALLTHROUGH
#endif

#if defined(__GNUC__)
#define POCL_UNUSED __attribute__((unused))
#else
#define POCL_UNUSED
#endif

#endif //POCL_POCL_COMPILER_MACROS_H
