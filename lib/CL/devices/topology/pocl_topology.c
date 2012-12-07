/* pocl_topology.c - retrieving the topology of OpenCL devices

   Copyright (c) 2012 Cyril Roelandt and Pekka Jääskeläinen
   
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

#include <pocl_cl.h>
#include <hwloc.h>

#include "pocl_topology.h"

void
pocl_topology_detect_device_info(cl_device_id device)
{
  hwloc_topology_t pocl_topology;

  int ret = hwloc_topology_init(&pocl_topology);
  if (ret == -1)
    POCL_ABORT("Cannot initialize the topology.\n");
  ret = hwloc_topology_load(pocl_topology);
  if (ret == -1)
    POCL_ABORT("Cannot load the topology.\n");

  device->global_mem_size = hwloc_get_root_obj(pocl_topology)->memory.total_memory;

  if (device->global_mem_size/4 > MIN_MAX_MEM_ALLOC_SIZE)
    device->max_mem_alloc_size = device->global_mem_size/4;
  else
    device->max_mem_alloc_size = MIN_MAX_MEM_ALLOC_SIZE;

  device->local_mem_size = device->max_constant_buffer_size = device->max_mem_alloc_size;
}
