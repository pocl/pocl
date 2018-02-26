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

#include "pocl_topology.h"

#if defined(__x86_64__) || defined(__i386__)

enum VendorSignatures
{
  SIG_INTEL = 0x756e6547 /* Genu */,
  SIG_AMD = 0x68747541 /* Auth */
};

/// getX86CpuIDAndInfo - Execute the specified cpuid and return the 4 values in
/// the specified arguments.  If we can't run cpuid on the host, return true.
static int
getX86CpuIDAndInfo (unsigned value, unsigned *rEAX, unsigned *rEBX,
                    unsigned *rECX, unsigned *rEDX)
{
#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__)
  // gcc doesn't know cpuid would clobber ebx/rbx. Preserve it manually.
  __asm__("movq\t%%rbx, %%rsi\n\t"
          "cpuid\n\t"
          "xchgq\t%%rbx, %%rsi\n\t"
          : "=a"(*rEAX), "=S"(*rEBX), "=c"(*rECX), "=d"(*rEDX)
          : "a"(value));
  return 0;
#elif defined(__i386__)
  __asm__("movl\t%%ebx, %%esi\n\t"
          "cpuid\n\t"
          "xchgl\t%%ebx, %%esi\n\t"
          : "=a"(*rEAX), "=S"(*rEBX), "=c"(*rECX), "=d"(*rEDX)
          : "a"(value));
  return 0;
#else
  return 1;
#endif
#elif defined(_MSC_VER)
  // The MSVC intrinsic is portable across x86 and x64.
  int registers[4];
  __cpuid (registers, value);
  *rEAX = registers[0];
  *rEBX = registers[1];
  *rECX = registers[2];
  *rEDX = registers[3];
  return 0;
#else
  return 1;
#endif
}

#endif

int
pocl_topology_detect_device_info(cl_device_id device)
{
  hwloc_topology_t pocl_topology;
  int ret = 0;

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

  hwloc_topology_ignore_type (pocl_topology, HWLOC_TOPOLOGY_FLAG_WHOLE_IO);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_SYSTEM);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_GROUP);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_BRIDGE);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_MISC);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_PCI_DEVICE);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_OS_DEVICE);

  ret = hwloc_topology_load (pocl_topology);
  if (ret == -1)
  {
    POCL_MSG_ERR ("Cannot load the topology.\n");
    goto exit_destroy;
  }

  device->global_mem_size =
      hwloc_get_root_obj(pocl_topology)->memory.total_memory;

  // Try to get the number of CPU cores from topology
  int depth = hwloc_get_type_depth(pocl_topology, HWLOC_OBJ_PU);
  if(depth != HWLOC_TYPE_DEPTH_UNKNOWN)
    device->max_compute_units = hwloc_get_nbobjs_by_depth(pocl_topology, depth);

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#if defined(__x86_64__) || defined(__i386__)
  unsigned Vendor, EAX, ECX, EDX;
  if (getX86CpuIDAndInfo (0, &EAX, &Vendor, &ECX, &EDX))
    device->vendor_id = 0x0086;
  else
    {
      if (Vendor == SIG_INTEL)
        device->vendor_id = 0x8086;
      else if (Vendor == SIG_AMD)
        device->vendor_id = 0x1022;
      else
        /* unknown x86 */
        device->vendor_id = 0x0086;
    }
#else
  device->vendor_id = 0x0000;
#endif
#else
  device->vendor_id = 0x0000;
#endif

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


