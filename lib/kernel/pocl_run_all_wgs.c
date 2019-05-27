/* OpenCL built-in library: Facilities for spawning work-group functions in
   the device side: _pocl_run_all_wgs()

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

#include "pocl_device.h"
#include "pocl_context.h"
#include "pocl_types.h"

void
_pocl_finish_all_wgs (uchar *);

void
_pocl_spawn_wg (void *restrict wg_func_ptr, uchar *restrict args,
		uchar *restrict ctx,
		size_t group_x, size_t group_y, size_t group_z);

/* Launches all the work-groups in the grid using the given work group function.
   Blocks until all WGs have been executed to the end.

   @param wg_func_ptr Pointer to the pocl-generated WG function for the kernel.
   @param args The flat argument buffer.
   @param pc The context struct for getting the dimensions etc.  */
void
_pocl_run_all_wgs (void *restrict wg_func_ptr, uchar *restrict args,
                   uchar *restrict pcptr, void *d)
{
  struct pocl_context *pc = (struct pocl_context*)pcptr;
  for (size_t gz = 0; gz < pc->num_groups[2]; ++gz)
    for (size_t gy = 0; gy < pc->num_groups[1]; ++gy)
      for (size_t gx = 0; gx < pc->num_groups[0]; ++gx)
	_pocl_spawn_wg (wg_func_ptr, args, pcptr, gx, gy, gz);

  _pocl_finish_all_wgs (pcptr);
}
