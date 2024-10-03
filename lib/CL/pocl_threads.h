/* OpenCL runtime library: utility functions for thread operations

   Copyright (c) 2024 Michal Babej / Intel Finland Oy

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

/** \file pocl_threads.h
 *
 * This file selects the thread implementation based on the value of
 * ENABLE_LLVM_PLATFORM_SUPPORT.
 *
 * PoCL core should use only the abstractions from pocl_threads_{c/cpp}.h for
 * threading and synchronization.
 */

#ifndef POCL_THREADS_H
#define POCL_THREADS_H

#include "config.h"

#ifdef ENABLE_LLVM_PLATFORM_SUPPORT

#include "pocl_threads_cpp.hh"

#else

#include "pocl_threads_c.h"

#endif

#endif // POCL_THREADS_H
