/* OpenCL runtime library: integrated command queue profile collecting
   functionality

   Copyright (c) 2019 Pekka Jääskeläinen

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

/* The basic idea of the 'cq' profiler is to move most of the execution profile
   impact (additional code executed, cache footprints changed etc. due to
   profiling enabled) to the initialization (CQ creation time) with minimal impact
   at runtime. This is accomplished by enabling the basic profiling queue feature
   which is expected to be a minimally intrusive per-device specific way to
   collect time stamps. The events with the profiling time stamps are just
   accumulated until there is a "non-intrusive" spot to collect/analyze the data.
   The default implementation just accumulates all the events and analyzes them
   atexit() with a simple per kernel execution time printout.

   One thing to keep in mind is that the command queue time stamps are per-device
   timer stamps, and there is no requirement (AFAIK) to have a synchronized
   global clock counter across the devices in the system. This means that _differences_
   between time stamps in the same device does make sense, but in a multi-device
   platform, the CQ timestamps are not necessarily a robust mechanism to draw
   a global picture of the overall execution orderings etc. OpenCL 2.1 introduced
   clGetHostTimer() and clGetDeviceAndHostTimer() which should help.
*/

#ifndef POCL_CQ_PROFILING_H
#define POCL_CQ_PROFILING_H

#include "CL/cl.h"

/* This is set to 1 in case the cq profiling was enabled via POCL_TRACING=cq. */
extern int pocl_cq_profiling_enabled;

void pocl_cq_profiling_init ();
void pocl_cq_profiling_register_event (cl_event event);

#endif
