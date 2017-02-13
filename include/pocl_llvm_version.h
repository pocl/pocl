/* pocl/pocl_llvm_version.h - Macros for checking the current Clang/LLVM version.

   Copyright (c) 2017 Pekka Jääskeläinen / Tampere University of Technology

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

#ifndef _POCL_LLVM_VERSION_H
#define _POCL_LLVM_VERSION_H

#if (__clang_major__ == 3)
# if (__clang_minor__ == 7)
# undef LLVM_3_7
# define LLVM_3_7
#elif (__clang_minor__ == 8)
# undef LLVM_3_8
# define LLVM_3_8
#elif (__clang_minor__ == 9)
# undef LLVM_3_9
# define LLVM_3_9
#endif

#elif (__clang_major__ == 4)

# undef LLVM_4_0
# define LLVM_4_0

#else

#error Unsupported Clang/LLVM version.

#endif

/* Additional LLVM version macros to simplify ifdefs */
#if (defined LLVM_3_6)
# define LLVM_OLDER_THAN_3_7 1
# define LLVM_OLDER_THAN_3_8 1
# define LLVM_OLDER_THAN_3_9 1
# define LLVM_OLDER_THAN_4_0 1
#endif

#if (defined LLVM_3_7)
# define LLVM_OLDER_THAN_3_8 1
# define LLVM_OLDER_THAN_3_9 1
# define LLVM_OLDER_THAN_4_0 1
#endif

#if (defined LLVM_3_8)
# define LLVM_OLDER_THAN_3_9 1
# define LLVM_OLDER_THAN_4_0 1
#endif

#if (defined LLVM_3_9)
# define LLVM_OLDER_THAN_4_0 1
#endif

#endif
