/* common.h - common code that can be reused between device driver 
              implementations

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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

#ifndef POCL_COMMON_H
#define POCL_COMMON_H

#include "pocl_cl.h"

/* Determine preferred vector sizes */
#if defined(__AVX__)
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR   16
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT   8
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT     4
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG    2
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT   4
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE  2
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_CHAR      16
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_SHORT      8
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_INT        4
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_LONG       2
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_FLOAT      8
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_DOUBLE     4
#elif defined(__SSE2__)
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR   16
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT   8
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT     4
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG    2
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT   4
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE  2
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_CHAR      16
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_SHORT      8
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_INT        4
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_LONG       2
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_FLOAT      4
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_DOUBLE     2
#else
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR    1
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT   1
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT     1
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG    1
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT   1
#  define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE  1
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_CHAR       1
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_SHORT      1
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_INT        1
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_LONG       1
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_FLOAT      1
#  define POCL_DEVICES_NATIVE_VECTOR_WIDTH_DOUBLE     1
#endif
/* Half is internally represented as short */
#define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT
#define POCL_DEVICES_NATIVE_VECTOR_WIDTH_HALF POCL_DEVICES_NATIVE_VECTOR_WIDTH_SHORT

const char* llvm_codegen (const char* tmpdir);

#endif
