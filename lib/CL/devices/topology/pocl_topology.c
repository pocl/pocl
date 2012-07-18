/* pocl_topology.c - retrieving the topology of OpenCL devices using the hwloc

   Copyright (c) 2012 Cyril Roelandt
   
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

#include <hwloc.h>
#include <pocl_cl.h>

/* We may want to protect these with a mutex, but it's probably not needed for
 * now. */
static int init = 0;
static hwloc_topology_t pocl_topology;

static void
pocl_topology_init(void)
{
  int ret;

  ret = pocl_hwloc_topology_init(&pocl_topology);
  if (ret == -1)
    POCL_ABORT("Cannot initialize the topology.\n");
  ret = pocl_hwloc_topology_load(pocl_topology);
  if (ret == -1)
    POCL_ABORT("Cannot load the topology.\n");

  init = 1;
  /* When should we call hwloc_topology_destroy() ? */
}

void
pocl_topology_set_global_mem_size(cl_device_id device)
{
  if (!init)
    pocl_topology_init();

  device->global_mem_size = 
    pocl_hwloc_get_root_obj(pocl_topology)->memory.total_memory;
}

void
pocl_topology_set_max_mem_alloc_size(cl_device_id device)
{
#define MIN_MAX_MEM_ALLOC_SIZE (128*1024*1024)
  if (device->global_mem_size/4 > MIN_MAX_MEM_ALLOC_SIZE)
    device->max_mem_alloc_size = device->global_mem_size/4;
  else
    device->max_mem_alloc_size = MIN_MAX_MEM_ALLOC_SIZE;
}
