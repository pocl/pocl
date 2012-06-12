#include <pocl_cl.h>
#include <hwloc.h>

/* We may want to protect these with a mutex, but it's probably not needed for
 * now. */
static int init = 0;
static hwloc_topology_t pocl_topology;

static void
pocl_topology_init(void)
{
  int ret;

  ret = hwloc_topology_init(&pocl_topology);
  if (ret == -1)
    POCL_ABORT("Cannot initialize the topology.\n");
  ret = hwloc_topology_load(pocl_topology);
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

  device->global_mem_size = hwloc_get_root_obj(pocl_topology)->memory.total_memory;
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
