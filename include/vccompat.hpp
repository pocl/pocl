/* pocl/include/vccompat.h - Compatibility header to provide some functions 
   which are not found from VC++. 

   All functions should be static inline so that they can be included in many places
   without having problem of symbol collision.

   Copyright (c) 2014 Mikael Lepistö <elhigu@gmail.com>
   
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

#if !defined(VCCOMPAT_HPP) && defined(_WIN32)
#define VCCOMPAT_HPP

/* Suppress min/max macros in windows.h.  They wreck havoc on C++ code.  */
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define __restrict__ __restrict

#include <intrin.h>
#define __builtin_popcount __popcnt

// ERROR is used as label for goto in some OCL API functions
#undef ERROR

#ifdef _MSC_VER
static inline char* strtok_r(char *str, const char *delim, char **saveptr) {
  return strtok_s(str, delim, saveptr);
}
#endif

#define _USE_MATH_DEFINES

#define srand48(x) srand(x)
#define drand48() (((double)rand()) / RAND_MAX)

#define random rand
#define srandom(x) srand(x)

#include <sys/utime.h>
#define utime _utime

#ifdef _MSC_VER
// Sleep takes milliseconds.
#define sleep(x) Sleep(x * 1000)
#define usleep(x) Sleep((x / 1000) ? x / 1000 : 1)
#else
#include <unistd.h>
#endif

static inline int setenv(const char *name, const char *value, int overwrite) {
  return _putenv_s(name, value);
}

#ifdef _MSC_VER

#define RTLD_NOW 1
#define RTLD_LOCAL 1

#endif

/**
 * Filesystem stuff
 */
#include <io.h>
#define R_OK    4       /* Test for read permission.  */
#define W_OK    2       /* Test for write permission.  */
#define F_OK    0       /* Test for existence.  */

#include <stdlib.h>
#include <direct.h>
#include <process.h>

#define mkdir(a,b) _mkdir(a)

/**
 * TODO: test these implementations...
 */

/* Commented out: unused, and actually incorrect/unsafe.
static inline void gen_random(char *s, const int len) {
  static const char alphanum[] =
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz";

  for (int i = 0; i < len; ++i) {
    s[i] = alphanum[rand() % (sizeof(alphanum)-1)];
  }
  s[len] = 0;
}

static inline void mkdtemp(char *temp) {
  int rnd_start = strlen(temp) - 6;
  gen_random(&temp[rnd_start], 6);
  mkdir(temp);
}
*/

/**
 * Memory allocation functions
 */
#include <malloc.h>

static int posix_memalign(void **p, size_t align, size_t size) { 
   void *buf = _aligned_malloc(size, align);
   if (buf == NULL) return errno;
   *p = buf;
   return 0;
}

#ifdef _MSC_VER
#define alloca _alloca
#endif

#endif
