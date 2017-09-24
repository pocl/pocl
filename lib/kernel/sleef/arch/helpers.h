/* OpenCL built-in library: SLEEF helpers.h

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

/************************/
#if defined(PURE_C)

  #ifdef DORENAME
    #include "rename.h"
  #endif

#elif defined(VEC128)

  #ifdef DORENAME
    #include "rename_vec128.h"
  #endif

  #ifdef __ARM_NEON
    #define CONFIG 1
    #ifdef __aarch64__
      #define ENABLE_ADVSIMD
      #include "helperadvsimd.h"
    #else
      #define ENABLE_NEON32
      #include "helperneon32.h"
    #endif

  #elif defined(__AVX2__)
    #define CONFIG 1
    #define ENABLE_AVX2
    #include "helperavx2_128.h"

  #elif defined(__SSE4_1__)
    #define CONFIG 4
    #define ENABLE_SSE4
    #include "helpersse2.h"

  #elif defined(__SSE3__)
    #define CONFIG 3
    #define ENABLE_SSE2
    #include "helpersse2.h"

  #elif defined(__SSE2__)
    #define CONFIG 2
    #define ENABLE_SSE2
    #include "helpersse2.h"

  #else
    #error 128bit vectors unavailable
  #endif

#elif defined(VEC256)

  #ifdef DORENAME
    #include "rename_vec256.h"
  #endif

  #if defined(__AVX2__)
    #define CONFIG 1
    #define ENABLE_AVX2
    #include "helperavx2.h"

  #elif defined(__FMA4__)
    #define CONFIG 4
    #define ENABLE_FMA4
    #define ENABLE_AVX
    #include "helperavx.h"

  #elif defined(__AVX__)
    #define CONFIG 1
    #define ENABLE_AVX
    #include "helperavx.h"

  #else
    #error 256bit vectors unavailable
  #endif

#elif defined(VEC512)

  #ifdef DORENAME
    #include "rename_vec512.h"
  #endif

  #ifdef __AVX512F__
    #define CONFIG 1
    #define ENABLE_AVX512F
    #include "helperavx512f.h"
  #else
    #error 512bit vectors unavailable
  #endif

#else
#error Please specify valid vector size with -DVECxxx
#endif

/* TODO this one is completely untested. */

#ifdef ENABLE_VECEXT
#define CONFIG 1
#include "helpervecext.h"
#ifdef DORENAME
#include "renamevecext.h"
#endif
#endif
