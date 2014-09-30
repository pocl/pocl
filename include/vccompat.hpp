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

#ifndef VCCOMPAT_HPP
#define VCCOMPAT_HPP

#include <Windows.h>
#define __restrict__ __restrict

// ERROR is used as label for goto in some OCL API functions
#undef ERROR

// if this causes linking problems, use inline function below...
#define snprintf _snprintf

/*
static inline int snprintf(char *str, size_t size, const char *format, ...) {
   va_list args;
   va_start(args, format);
   _snprintf(str, size, format, args);
   va_end(args);
}
*/

static inline char* strtok_r(char *str, const char *delim, char **saveptr) {
   return strtok_s(str, delim, saveptr);
}

/**
 * ltdl compatibility functions
 */
typedef HMODULE lt_dlhandle;

static inline lt_dlhandle lt_dlopen(const char* filename) {
  return (lt_dlhandle)LoadLibrary(filename);
}

static inline int lt_dlerror(void) {
   return GetLastError();
}

static inline void *lt_dlsym(lt_dlhandle handle, const char *symbol) {
  return GetProcAddress(handle, symbol);
}

static inline void lt_dlinit(void) {
   // separate init not needed in windows
}

/**
 * Filesystem stuff
 */
#include <io.h>
#define R_OK    4       /* Test for read permission.  */
#define W_OK    2       /* Test for write permission.  */
#define F_OK    0       /* Test for existence.  */

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

#define alloca _alloca

#endif
