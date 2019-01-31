/* OpenCL built-in library: Facilities for spawning work-group functions in
   the device side: pocl_spawn_wg ()

   Copyright (c) 2018 Pekka Jääskeläinen

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

#include "pocl_context.h"
#include "pocl_device.h"
#include "pocl_workgroup_func.h"

/**
   Launches a work-group. Does not necessarily block, thus can spawn a separate
   thread for executing the WG.

   _pocl_finish_work_groups () should be called to wait until the in-flight
   WGs have finished.

   @param wg_func_ptr Pointer to the pocl-generated WG function for the kernel.
   @param args A flat argument buffer passed to the kernel.
   @param group_x, group_y, group_z The group id of this launch.
   @param pc The execution context struct. */
void
_pocl_spawn_wg (void *wg_func_ptr, uchar *args, uchar *ctx,
		size_t group_x, size_t group_y, size_t group_z)
{
  ((pocl_workgroup_func)wg_func_ptr)(args, ctx, group_x, group_y, group_z);
}

/* Blocks until all currently launched WGs have finished.

   The default implementation does nothing as the work-groups are executed
   synchronously.
*/
void
_pocl_finish_all_wgs (uchar *pc)
{
}
