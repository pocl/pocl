/* OpenCL built-in library: work-item queries for the Metal (AIR) target

   Copyright (c) 2026 PoCL developers

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

/* Metal passes the work-item coordinates as kernel arguments instead of
   providing intrinsics to query them. The kernel library reads these
   placeholder variables, and the Metal driver's AIR lowering pass replaces
   them in each kernel with the matching (uint3) kernel input argument, or
   with the launch parameters passed in a buffer. (They are variables rather
   than functions because PoCL's linker rejects undefined functions.) */

#ifndef POCL_AIR_WORKITEM_H
#define POCL_AIR_WORKITEM_H

extern const uint3 _pocl_air_thread_position_in_grid;
extern const uint3 _pocl_air_thread_position_in_threadgroup;
extern const uint3 _pocl_air_threadgroup_position_in_grid;
extern const uint3 _pocl_air_threads_per_threadgroup;
extern const uint3 _pocl_air_threadgroups_per_grid;
extern const uint3 _pocl_air_threads_per_grid;
/* { work_dim, global_offset[0], global_offset[1], global_offset[2] } */
extern const uint _pocl_air_launch_params[4];

#endif
