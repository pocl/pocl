/* pocl_topology.c - retrieving the topology of OpenCL devices

   Copyright (c) 2012,2015 Cyril Roelandt and Pekka Jääskeläinen
   
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
#include <stdlib.h>

#if HWLOC_API_VERSION >= 0x00020000
#define HWLOC_API_2
#else
#undef HWLOC_API_2
#endif

#include "pocl_topology.h"

/*
 * Sets up:
 *  max_compute_units
 *  global_mem_size
 *  global_mem_cache_type
 *  global_mem_cacheline_size
 *  global_mem_cache_size
 */

int
pocl_topology_detect_device_info(cl_device_id device)
{
  hwloc_topology_t pocl_topology;
  int ret = 0;

#ifdef HWLOC_API_2
  if (hwloc_get_api_version () < 0x20000)
    POCL_MSG_ERR ("pocl was compiled against libhwloc 2.x but is"
                  "actually running against libhwloc 1.x \n");
#else
  if (hwloc_get_api_version () >= 0x20000)
    POCL_MSG_ERR ("pocl was compiled against libhwloc 1.x but is"
                  "actually running against libhwloc 2.x \n");
#endif

  /*
   * hwloc's OpenCL backend causes problems at the initialization stage
   * because it reloads libpocl.so via the ICD loader.
   *
   * See: https://github.com/pocl/pocl/issues/261
   *
   * The only trick to stop hwloc from initializing the OpenCL plugin
   * I could find is to point the plugin search path to a place where there
   * are no plugins to be found.
   */
  setenv ("HWLOC_PLUGINS_PATH", "/dev/null", 1);

  ret = hwloc_topology_init (&pocl_topology);
  if (ret == -1)
  {
    POCL_MSG_ERR ("Cannot initialize the topology.\n");
    return ret;
  }

#ifdef HWLOC_API_2
  hwloc_topology_set_io_types_filter(pocl_topology, HWLOC_TYPE_FILTER_KEEP_NONE);
  hwloc_topology_set_type_filter (pocl_topology, HWLOC_OBJ_SYSTEM, HWLOC_TYPE_FILTER_KEEP_NONE);
  hwloc_topology_set_type_filter (pocl_topology, HWLOC_OBJ_GROUP, HWLOC_TYPE_FILTER_KEEP_NONE);
  hwloc_topology_set_type_filter (pocl_topology, HWLOC_OBJ_BRIDGE, HWLOC_TYPE_FILTER_KEEP_NONE);
  hwloc_topology_set_type_filter (pocl_topology, HWLOC_OBJ_MISC, HWLOC_TYPE_FILTER_KEEP_NONE);
  hwloc_topology_set_type_filter (pocl_topology, HWLOC_OBJ_PCI_DEVICE, HWLOC_TYPE_FILTER_KEEP_NONE);
  hwloc_topology_set_type_filter (pocl_topology, HWLOC_OBJ_OS_DEVICE, HWLOC_TYPE_FILTER_KEEP_NONE);
#else
  hwloc_topology_ignore_type (pocl_topology, HWLOC_TOPOLOGY_FLAG_WHOLE_IO);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_SYSTEM);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_GROUP);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_BRIDGE);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_MISC);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_PCI_DEVICE);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_OS_DEVICE);
#endif

  ret = hwloc_topology_load (pocl_topology);
  if (ret == -1)
  {
    POCL_MSG_ERR ("Cannot load the topology.\n");
    goto exit_destroy;
  }

#ifdef HWLOC_API_2
  device->global_mem_size =
      hwloc_get_root_obj(pocl_topology)->total_memory;
#else
  device->global_mem_size =
      hwloc_get_root_obj(pocl_topology)->memory.total_memory;
#endif

  // Try to get the number of CPU cores from topology
  int depth = hwloc_get_type_depth(pocl_topology, HWLOC_OBJ_PU);
  if(depth != HWLOC_TYPE_DEPTH_UNKNOWN)
    device->max_compute_units = hwloc_get_nbobjs_by_depth(pocl_topology, depth);

  /* Find information about global memory cache by looking at the first
   * cache covering the first PU */
  do {
      size_t cache_size = 0, cacheline_size = 0;

      hwloc_obj_t core
          = hwloc_get_next_obj_by_type (pocl_topology, HWLOC_OBJ_CORE, NULL);
      if (core)
        {
          hwloc_obj_t cache
              = hwloc_get_shared_cache_covering_obj (pocl_topology, core);
          if ((cache) && (cache->attr))
            {
              cacheline_size = cache->attr->cache.linesize;
              cache_size = cache->attr->cache.size;
            }
          else
            core = NULL; /* fallback to L1 cache size */
        }

      hwloc_obj_t pu
          = hwloc_get_next_obj_by_type (pocl_topology, HWLOC_OBJ_PU, NULL);
      if (!core && pu)
        {
          hwloc_obj_t cache
              = hwloc_get_shared_cache_covering_obj (pocl_topology, pu);
          if ((cache) && (cache->attr))
            {
              cacheline_size = cache->attr->cache.linesize;
              cache_size = cache->attr->cache.size;
            }
        }

      if (!cache_size || !cacheline_size)
        break;

      device->global_mem_cache_type
          = 0x2; // CL_READ_WRITE_CACHE, without including all of CL/cl.h
      device->global_mem_cacheline_size = cacheline_size;
      device->global_mem_cache_size = cache_size;
  } while (0);

  // Destroy topology object and return
exit_destroy:
  hwloc_topology_destroy (pocl_topology);
  return ret;

}


