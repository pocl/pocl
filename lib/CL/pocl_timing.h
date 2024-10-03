/* OpenCL runtime library: time measurement utility functions

   Copyright (c) 2012-2019 pocl developers
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

#ifndef POCL_TIMING_H
#define POCL_TIMING_H

#include <stdint.h>

#ifdef HAVE_CLOCK_GETTIME
#include <time.h>
#endif

#include "config.h"

#include "pocl_export.h"

#ifdef __cplusplus
extern "C" {
#endif

POCL_EXPORT
uint64_t pocl_gettimemono_ns();

POCL_EXPORT
uint64_t pocl_gettimer_resolution ();

/* only used in pocl_debug.h */
int pocl_gettimereal(int *year, int *mon, int *day, int *hour, int *min, int *sec, int* nanosec);

#ifdef __cplusplus
}
#endif

#endif
