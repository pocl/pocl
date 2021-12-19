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

#include <stdlib.h>
#include <assert.h>

#include "config.h"

#include <pocl_cl.h>
#include <pocl_file_util.h>

#include "pocl_topology.h"

#ifdef ENABLE_HWLOC

#include <hwloc.h>
#if HWLOC_API_VERSION >= 0x00020000
#define HWLOC_API_2
#else
#undef HWLOC_API_2
#endif

#endif

/*
 * Sets up:
 *  max_compute_units
 *  global_mem_size
 *  global_mem_cache_type
 *  global_mem_cacheline_size
 *  global_mem_cache_size
 *  local_mem_size
 *  max_constant_buffer_size
 */

#ifdef ENABLE_HWLOC

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
#ifndef _WIN32
  setenv ("HWLOC_PLUGINS_PATH", "/dev/null", 1);
#endif

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
  size_t shared_cache_size = 0, nonshared_cache_size = 0, cacheline_size = 0;
  hwloc_obj_t cache = NULL;

  hwloc_obj_t core
      = hwloc_get_next_obj_by_type (pocl_topology, HWLOC_OBJ_CORE, NULL);
  if (core)
    {
      cache = hwloc_get_shared_cache_covering_obj (pocl_topology, core);
      if ((cache) && (cache->attr))
        {
          cacheline_size = cache->attr->cache.linesize;
          shared_cache_size = cache->attr->cache.size;
        }
      else
        core = NULL; /* fallback to L1 cache size */
    }

  hwloc_obj_t pu
      = hwloc_get_next_obj_by_type (pocl_topology, HWLOC_OBJ_PU, NULL);
  if (!core && pu)
    {
      cache = hwloc_get_shared_cache_covering_obj (pocl_topology, pu);
      if ((cache) && (cache->attr))
        {
          cacheline_size = cache->attr->cache.linesize;
          shared_cache_size = cache->attr->cache.size;
        }
    }

  if (cache)
    {
      /* cache should now contain the first shared cache.
       * get the first cache with depth one larger, which
       * should be the last non-shared cache. */
      unsigned shared_depth = cache->depth;
      unsigned nonshared_depth = shared_depth + 1;
      unsigned nonshared_cachenum
          = hwloc_get_nbobjs_by_depth (pocl_topology, nonshared_depth);
      if (nonshared_cachenum > 0)
        {
          cache = hwloc_get_obj_by_depth (pocl_topology, nonshared_depth, 0);
#ifdef HWLOC_API_2
          if (hwloc_obj_type_is_cache(cache->type) || hwloc_obj_type_is_dcache(cache->type))
#else
          if (cache->type == HWLOC_OBJ_CACHE)
#endif
            {
              nonshared_cache_size = cache->attr->cache.size;
            }
        }
    }

  if (shared_cache_size > 0 && cacheline_size > 0)
    {
      device->global_mem_cache_type
          = 0x2; // CL_READ_WRITE_CACHE, without including all of CL/cl.h
      device->global_mem_cacheline_size = cacheline_size;
      device->global_mem_cache_size = shared_cache_size;
    }
  if (nonshared_cache_size > 0)
    {
      device->local_mem_size = nonshared_cache_size;
      device->max_constant_buffer_size = nonshared_cache_size;
    }
  // Destroy topology object and return
exit_destroy:
  hwloc_topology_destroy (pocl_topology);
  return ret;

}

// #ifdef HWLOC
#elif defined(__linux__) || defined(__ANDROID__)

#define L3_CACHE_SIZE "/sys/devices/system/cpu/cpu0/cache/index3/size"
#define L2_CACHE_SIZE "/sys/devices/system/cpu/cpu0/cache/index2/size"
#define CPUS "/sys/devices/system/cpu/possible"
#define MEMINFO "/proc/meminfo"

int
pocl_topology_detect_device_info (cl_device_id device)
{
  device->global_mem_cacheline_size = HOST_CPU_CACHELINE_SIZE;
  device->global_mem_cache_type
      = 0x2; // CL_READ_WRITE_CACHE, without including all of CL/cl.h

  /* global mem cache size */

  char *content;
  uint64_t filesize;

  if (pocl_read_file (L3_CACHE_SIZE, &content, &filesize) == 0)
    {
      long val = atol (content);
      device->global_mem_cache_size = val * 1024;
      POCL_MEM_FREE (content);
    }
  else
    {
      if (pocl_read_file (L2_CACHE_SIZE, &content, &filesize) == 0)
        {
          long val = atol (content);
          device->global_mem_cache_size = val * 1024;
          POCL_MEM_FREE (content);
        }
      else
        {
          POCL_MSG_WARN (
              "Could not figure out CPU cache size, using bogus value\n");
          device->global_mem_cache_size = 1 << 20;
        }
    }

  /* global_mem_size */
  if (pocl_read_file (MEMINFO, &content, &filesize) == 0)
    {
      char *tmp = content;
      unsigned long memsize_kb;
      size_t i;

      while (*tmp && (*tmp != '\n'))
        ++tmp;
      *tmp = 0;
      tmp = content;
      while (*tmp && (*tmp != 0x20))
        ++tmp;
      while (*tmp && (*tmp == 0x20))
        ++tmp;
      int items = sscanf (tmp, "%lu kB", &memsize_kb);

      assert (items == 1);

      device->global_mem_size = memsize_kb * 1024;
      POCL_MEM_FREE (content);
    }
  else
    {
      POCL_MSG_WARN ("Cannot get memory size\n");
      device->global_mem_size = 256 << 20;
    }

  /* max_compute_units */
  if (pocl_read_file (CPUS, &content, &filesize) == 0)
    {
      assert (content);
      assert (filesize > 0);
      unsigned long start, end;
      int items = sscanf (content, "%lu-%lu", &start, &end);
      assert (items == 2);
      device->max_compute_units = (unsigned)end + 1;
      POCL_MEM_FREE (content);
    }
  else
    {
      POCL_MSG_WARN ("Cannot get logical CPU number\n");
      device->max_compute_units = 1;
    }

  return 0;
}

#else

#error Dont know how to get HWLOC-provided values on this system!

#endif
