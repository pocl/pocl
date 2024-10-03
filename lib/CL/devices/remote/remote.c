/* remote.c - a pocl device driver which controls remote devices

   Copyright (c) 2018 Michal Babej / Tampere University of Technology
   Copyright (c) 2019-2023 Jan Solanti / Tampere University
   Copyright (c) 2023-2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "remote.h"
#include "common.h"
#include "config.h"
#include "devices.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "common_driver.h"
#include "pocl_cache.h"
#include "pocl_cl.h"
#include "pocl_file_util.h"
#include "pocl_mem_management.h"
#include "pocl_timing.h"
#include "pocl_util.h"
#include "utlist.h"
#include <CL/cl.h>

#include "communication.h"
#include "messages.h"

/*
  TODO / problematic:
  kernel arg info - arg types (currently working, but still a hack)
*/

typedef struct remote_svm_management_data_s
{
  /* A context to store device-wide data (currently only the pinned memory
     buffer for the SVM allocations). TODO: The same buffer should be allocated
     from all devices in the context. */
  cl_context device_context;

  /* TODO: This should be allocated from clSVMAlloc, it should be the shadow
     buffer. Who frees the shadow buffer and when? clSVMFree(). */
  cl_mem pinned_device_allocation;

  /* The starting address and size of the SVM region in the host side. */
  size_t host_svm_region_start_addr;
  size_t host_svm_region_size;

  /* Bufalloc memory book keeping data for handing out SVM allocations. */
  memory_region_t allocations;
  /* TODO: We need to record the SVM region "gaps" from each device to the
     region to not allow allocating them in the host side. */
} remote_svm_management_data_t;

/* One of the devices will take care of the SVM management, which is the
   one that initializes this global. */
remote_svm_management_data_t *svm_data = NULL;

static int
is_svm_ptr (void *ptr)
{
  if (svm_data == NULL)
    return 0;
  return (size_t)ptr >= svm_data->host_svm_region_start_addr
         && (size_t)ptr < (svm_data->host_svm_region_start_addr
                           + svm_data->host_svm_region_size);
}

/**
 * See pocl_remote_svm_alloc()'s comment for info how the allocation works
 * when SVM is enabled.
 */
cl_int
pocl_remote_alloc_mem_obj (cl_device_id device, cl_mem mem, void *host_ptr)
{
  remote_device_data_t *d = (remote_device_data_t *)device->data;
  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];
  assert (p->mem_ptr == NULL);

  POCL_LOCK (d->mem_lock);

  /* remote driver doesn't preallocate */
  if ((mem->flags & CL_MEM_ALLOC_HOST_PTR) && (mem->mem_host_ptr == NULL))
    goto ERROR;

  int r;
  if (host_ptr == NULL
      && (device->svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER))
    {
      /* When SVM is enabled, allocate all buffers from the SVM region. */
      mem->mem_host_ptr = pocl_bufalloc (&svm_data->allocations, mem->size);
      if (mem->mem_host_ptr == NULL)
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;

      /* Fix the remote buffer's host pointer to point to the remote's SVM
         pool, so the remote buffer contains the remote's host pointer. */
      mem->mem_host_ptr += d->svm_region_offset;
      r = pocl_network_create_buffer (d, mem, &p->device_addr);

      /* ...and back so the host side buffer points to the host side ptr. */
      mem->mem_host_ptr -= d->svm_region_offset;
    }
  else if (mem->is_image)
    {
      r = pocl_network_create_image (d, mem);
    }
  else if ((mem->flags & CL_MEM_USE_HOST_PTR) && host_ptr != NULL)
    {
      /* We use a preallocated host pointer. It can be either an SVM pointer
         or non-SVM host pointer. */
      /* Send the allocation to the server for updating its internal cl_mem.
         The SVM pointer should have been requested previously
         via svm_alloc. */

      /* Fix the remote buffer's host pointer to point to the remote's SVM
         pool, so the remote buffer contains the remote's host pointer. */
      int is_svm = is_svm_ptr (host_ptr);
      if (is_svm)
        {
          POCL_MSG_PRINT_MEMORY (
              "cl_mem with SVM ptr %p, offset adjusting with %zu to %p.\n",
              host_ptr, d->svm_region_offset, host_ptr + d->svm_region_offset);
          mem->mem_host_ptr = host_ptr + d->svm_region_offset;
        }
      r = pocl_network_create_buffer (d, mem, &p->device_addr);

      if (is_svm)
        /* Set it back, since the client might inspect the host pointer, which
           should now point again to the host region. */
        mem->mem_host_ptr = host_ptr;
    }
  else
    {
      r = pocl_network_create_buffer (d, mem, &p->device_addr);
    }

  if (r != 0)
    goto ERROR;

  /* The device-specific "address" is an id reference to the remote buffer,
     which then contains the actual physical address of the controlled
     device. */
  p->mem_ptr = (void *)mem->id;
  p->version = 0;
  POCL_UNLOCK (d->mem_lock);

  POCL_MSG_PRINT_MEMORY ("REMOTE DEVICE ALLOC PTR %p SIZE %zu\n", p->mem_ptr,
                         mem->size);
  return CL_SUCCESS;

ERROR:
  POCL_MSG_PRINT_MEMORY (
      "REMOTE DEVICE ALLOC HOST PTR %p SIZE %zu FAILED error %d\n", host_ptr,
      mem->size, r);
  POCL_UNLOCK (d->mem_lock);
  return CL_MEM_OBJECT_ALLOCATION_FAILURE;
}

cl_int
pocl_remote_alloc_subbuffer (cl_device_id device, cl_mem sub_buf)
{
  remote_device_data_t *d = (remote_device_data_t *)device->data;
  pocl_mem_identifier *p = &sub_buf->device_ptrs[device->global_mem_id];

  int r = pocl_network_create_buffer (d, sub_buf, &p->device_addr);

  if (r != 0)
    return CL_OUT_OF_RESOURCES;

  /* The device-specific "address" is an id reference to the remote buffer,
     which then contains the actual physical address of the controlled
     device. */
  p->mem_ptr = (void *)sub_buf->id;
  p->version = 0;

  POCL_MSG_PRINT_MEMORY (
    "Remote device allocated a sub-buffer %p size %zu orig %zu\n", p->mem_ptr,
    sub_buf->size, sub_buf->origin);

  return CL_SUCCESS;
}

void
pocl_remote_svm_free (cl_device_id device, void *svm_ptr)
{
  remote_device_data_t *d = (remote_device_data_t *)device->data;

  POCL_MSG_PRINT_MEMORY ("Remote free SVM PTR (client %p remote %p)\n",
                         svm_ptr, svm_ptr + d->svm_region_offset);

  /* This is a device-side svm pointer that identifies the
     object on device side. */
  uint64_t mem_id = (uint64_t)svm_ptr + d->svm_region_offset;

  POCL_LOCK (d->mem_lock);
  int r = pocl_network_free_buffer (d, mem_id, 1);
  assert (r == 0);
  POCL_UNLOCK (d->mem_lock);
}

void
pocl_remote_free (cl_device_id device, cl_mem mem)
{
  remote_device_data_t *d = (remote_device_data_t *)device->data;
  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];
  assert (p->mem_ptr != NULL);

  uint32_t mem_id = (uintptr_t)p->mem_ptr;

  POCL_LOCK (d->mem_lock);
  int r;
  if (mem->is_image)
    {
      r = pocl_network_free_image (d, mem->id);
    }
  else
    {
      r = pocl_network_free_buffer (d, mem->id, 0);
    }
  assert (r == 0);

  POCL_MSG_PRINT_MEMORY ("REMOTE DEVICE FREE PTR %p SIZE %zu\n", p->mem_ptr,
                         mem->size);

  if (mem->mem_host_ptr != NULL && !(mem->flags & CL_MEM_USE_HOST_PTR)
      && (device->svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER))
    {
      /* When SVM is enabled, we allocate all buffers from the SVM region
         in the driver. Thus we should also free the SVM space. */
      if (pocl_free_buffer (&svm_data->allocations, (memory_address_t) mem->mem_host_ptr) == NULL)
        {
          POCL_MSG_ERR ("Failed freeing internal SVM allocation %p.\n",
                        mem->mem_host_ptr);
        }
      mem->mem_host_ptr = NULL;
    }

  p->mem_ptr = NULL;
  p->version = 0;
  POCL_UNLOCK (d->mem_lock);
}

/** SVM allocation is done in two parts: First we return a host-side SVM
   address from this function and then expect clSVMAlloc() to create the
   cl_mem shadow buffer which is then actually allocated from the remote as
   well.

   When SVM is enabled, the remote driver will allocate all allocations from a
   preallocated SVM memory pool. This includes both SVM allocations and cl_mem
   buffer allocations. The remote allocations must be subbuffers of a larger
   cl_mem representing the entire SVM region because OpenCL spec allows buffers
   to point only to starts of SVM allocations.

   The shadow cl_mems all have internally a host SVM pointer, which is
   translated at the server side to the device side SVM regions thanks to the
   device-side subbuffers holding device-side SVM region fixed addressess,
   which is adjusted in pocl_remote_alloc_mem_obj(). See
   setup_svm_memory_pool() for more info.
*/
void *
pocl_remote_svm_alloc (cl_device_id dev, cl_svm_mem_flags flags, size_t size)

{
  return pocl_bufalloc (&svm_data->allocations, size);
}

void *
pocl_remote_usm_alloc (cl_device_id dev, unsigned alloc_type,
                       cl_mem_alloc_flags_intel flags, size_t size,
                       cl_int *err_code)
{
  *err_code = CL_SUCCESS;
  /* Implement all non-system USM types as CG allocations. */
  if (alloc_type == CL_MEM_TYPE_HOST_INTEL
      || alloc_type == CL_MEM_TYPE_DEVICE_INTEL
      || alloc_type == CL_MEM_TYPE_SHARED_INTEL)
    return pocl_remote_svm_alloc (dev, CL_MEM_READ_WRITE, size);
  else
    return NULL;
}

void
pocl_remote_usm_free (cl_device_id dev, void *usm_ptr)
{
  pocl_remote_svm_free (dev, usm_ptr);
}

static const char remote_device_name[] = "remote";
const char *remote_device_name_ptr = remote_device_name;

void
pocl_remote_init_device_ops (struct pocl_device_ops *ops)
{
  memset (ops, 0, sizeof (struct pocl_device_ops));
  ops->device_name = remote_device_name;

  ops->probe = pocl_remote_probe;
  ops->init = pocl_remote_init;
  ops->post_init = pocl_remote_setup_peer_mesh;
  // ops->uninit = pocl_remote_uninit;
  // ops->reinit = pocl_remote_reinit;

  ops->alloc_mem_obj = pocl_remote_alloc_mem_obj;
  ops->alloc_subbuffer = pocl_remote_alloc_subbuffer;
  ops->svm_alloc = pocl_remote_svm_alloc;
  ops->svm_free = pocl_remote_svm_free;

  ops->usm_alloc = pocl_remote_usm_alloc;
  ops->usm_free = pocl_remote_usm_free;

  ops->free = pocl_remote_free;
  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;

  ops->get_device_info_ext = pocl_remote_get_device_info_ext;
  ops->set_kernel_exec_info_ext = pocl_remote_set_kernel_exec_info_ext;

  ops->can_migrate_d2d = pocl_remote_can_migrate_d2d;

  ops->create_kernel = pocl_remote_create_kernel;
  ops->free_kernel = pocl_remote_free_kernel;
  ops->init_queue = pocl_remote_init_queue;
  ops->free_queue = pocl_remote_free_queue;

  ops->build_source = pocl_remote_build_source;
  ops->link_program = pocl_remote_link_program;
  ops->build_binary = pocl_remote_build_binary;
  ops->build_builtin = pocl_remote_build_builtin;
  ops->free_program = pocl_remote_free_program;
  ops->setup_metadata = pocl_remote_setup_metadata;
  ops->supports_binary = pocl_remote_supports_binary;

  ops->join = pocl_remote_join;
  ops->submit = pocl_remote_submit;
  ops->broadcast = pocl_broadcast;
  ops->notify = pocl_remote_notify;
  ops->flush = pocl_remote_flush;
  ops->wait_event = pocl_remote_wait_event;
  ops->update_event = pocl_remote_update_event;
  ops->free_event_data = pocl_remote_free_event_data;
  ops->notify_cmdq_finished = pocl_remote_notify_cmdq_finished;
  ops->notify_event_finished = pocl_remote_notify_event_finished;
  ops->build_hash = pocl_remote_build_hash;

  ops->create_sampler = pocl_remote_create_sampler;
  ops->free_sampler = pocl_remote_free_sampler;
}

char *
pocl_remote_build_hash (cl_device_id device)
{
  REMOTE_DEV_DATA;
  return strdup (data->build_hash);
}

unsigned
pocl_remote_probe (struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count (ops->device_name);

  if (env_count > MAX_REMOTE_DEVICES)
    POCL_ABORT (
        "This pocl supports at most %u remote devices. This can be fixed by "
        "recompiling with -DMAX_REMOTE_DEVICES=n CMake option.\n",
        MAX_REMOTE_DEVICES);

  if (env_count <= 0)
    return 0;

  return (unsigned)env_count;
}

void
pocl_remote_update_event (cl_device_id device, cl_event event)
{
  uint64_t ts = pocl_gettimemono_ns ();
  switch (event->status)
    {
    // these three can be overwritten by finish_running_cmd
    case CL_QUEUED:
      event->time_queue = ts;
      break;
    case CL_SUBMITTED:
      event->time_submit = ts;
      break;
    case CL_RUNNING:
      event->time_start = ts;
      break;
    // this is set AFTER finish_running_cmd, so don't overwrite it
    case CL_COMPLETE:
      if (event->time_end == 0)
        event->time_end = ts;
      break;
    default:
      break;
    }
}

/**
 * \brief Allocates a memory region from the host process to map SVM
 * allocations to and initializes SVM memory management.
 *
 * The first remote device with SVM capabilities reported from the server
 * side tries to mmap() a similar size region at the same starting address on
 * the host. Since this is done at best-effort without guaranteed success, we
 * often end up with an offset which needs to be added to all the memory
 * accesses of kernels to adjust to the difference between the starting
 * addressess of the host and device regions where the SVM allocations are
 * mapped. The server-side can fix a non-zero host-device SVM offset by
 * manipulating all of the kernels' address computations.
 *
 * \param [inout] device the device to update the pool info to.
 * \returns zero on a successful mapping.
 *
 */
static int
setup_svm_memory_pool (cl_device_id device)
{
  remote_device_data_t *ddata = (remote_device_data_t *)device->data;
  if (ddata->device_svm_region_start_addr == 0
      || ddata->device_svm_region_size == 0)
    {
      POCL_MSG_PRINT_REMOTE ("Device side SVM region missing.\n");
      return -1;
    }

  if (svm_data != NULL)
    {
      /* Let the first remote device take care of the SVM allocation. */
      device->svm_allocation_priority = 0;

      ddata->svm_region_offset = ddata->device_svm_region_start_addr
                                 - svm_data->host_svm_region_start_addr;
      POCL_MSG_PRINT_REMOTE ("Host SVM region already allocated. "
                             "SVM pool offset for this device: %zd.\n",
                             ddata->svm_region_offset);

      /* Shrink the host SVM region to the smallest remote SVM region size. */
      if (svm_data->host_svm_region_size > ddata->device_svm_region_size)
        {
          POCL_MSG_PRINT_REMOTE (
              "Remote SVM region smaller than the host region."
              "Shrinking to %zu MB.\n",
              ddata->device_svm_region_size / (1024 * 1024));
          svm_data->allocations.last_chunk->size
              = ddata->device_svm_region_size;
          svm_data->host_svm_region_size = ddata->device_svm_region_size;
          /* TODO: mremap() to free the unallocatable host VM space */
        }

      /* TODO: Add the remote SVM region gaps to the memory manager so they
         won't get allocated. */
      return 0;
    }

  /* This is the first SVM-capable remote device that should handle the
     SVMAllocs. */
  device->svm_allocation_priority = 10;
  /* Initialize the driver-scope SVM management data. */
  svm_data = (remote_svm_management_data_t *)malloc (
      sizeof (remote_svm_management_data_t));

  svm_data->host_svm_region_start_addr = 0;
  svm_data->host_svm_region_size = 0;

  void *requested_address = (void *)ddata->device_svm_region_start_addr;

  POCL_MSG_PRINT_MEMORY (
      "Attempting to map a host SVM region of size %zu MB at '%p'.\n",
      ddata->device_svm_region_size / (1024 * 1024), requested_address);

  void *addr
      = mmap (requested_address, ddata->device_svm_region_size,
              PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, -1, 0);
  if (addr == MAP_FAILED)
    {
      /* TODO: Try mapping a smaller region than the device's in the host side.
       */
      perror ("mmap error");
      /* TODO: free the pinned buffer. */
      POCL_MSG_PRINT_REMOTE (
          "Unable to mmap() a local memory pool for remote CG SVM.\n");
      free (svm_data);

      /* Let the rest of the devices try. */
      svm_data = NULL;
      device->svm_allocation_priority = 0;

      return -2;
    }

  svm_data->host_svm_region_start_addr = (size_t)addr;
  svm_data->host_svm_region_size = ddata->device_svm_region_size;
  ddata->svm_region_offset = ddata->device_svm_region_start_addr
                             - svm_data->host_svm_region_start_addr;

  pocl_init_mem_region (&svm_data->allocations, (memory_address_t)addr,
                        ddata->device_svm_region_size);

  /* We must ensure the allocation is aligned to the largest OpenCL datatype
     since the remote will use a subbuffer to track the region and some targets
     (most notable PoCL-CPU) will require the alignment for the offset.
     Similarly, PoCL-D returns only SVM regions chunks that are max aligned. */
  svm_data->allocations.alignment = sizeof (cl_long16);

  POCL_MSG_PRINT_MEMORY (
      "Host SVM region allocated at '%p'. SVM pool offset: %zd.\n", addr,
      ddata->svm_region_offset);

  return 0;
}

static void *pocl_remote_driver_pthread (void *cldev);

cl_int
pocl_remote_init (unsigned j, cl_device_id device, const char *parameters)
{
  if ((parameters == NULL) || (strlen (parameters) == 0))
    {
      POCL_MSG_ERR ("No parameters given for pocl remote device #%u. Required "
                    "parameters are in the form:"
                    "POCL_REMOTEX_PARAMETERS=hostname[:port]/INDEX - port is "
                    "optional, defaults to ",
                    j);
      return CL_DEVICE_NOT_FOUND;
    }

  if (j >= MAX_REMOTE_DEVICES)
    POCL_ABORT (
        "This pocl supports at most %u remote devices. This can be fixed by "
        "recompiling with -DMAX_REMOTE_DEVICES=n CMake option.\n",
        MAX_REMOTE_DEVICES);

  remote_device_data_t *d;
  d = (remote_device_data_t *)calloc (1, sizeof (remote_device_data_t));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;
  device->data = d;

  device->has_own_timer = CL_TRUE;

  // TODO: add list of all remotes to create_or_find_server from here
  if (pocl_network_init_device (device, d, j, parameters))
    return CL_DEVICE_NOT_FOUND;

  const char *magic = "pocl";
  device->vendor_id
      = (unsigned)(magic[0] | magic[1] << 8 | magic[2] << 16 | magic[3] << 24);

  device->vendor_id += j;

  POCL_INIT_LOCK (d->mem_lock);

  if (pocl_network_fetch_devinfo (device, 0, 0, NULL,  NULL))
    return CL_DEVICE_NOT_FOUND;

  assert (device->short_name);
  char *res = calloc (1000, sizeof (char));
  snprintf (res, 1000, "pocl-remote: %s", device->short_name);
  d->build_hash = res;

  POCL_INIT_COND (d->wakeup_cond);
  POCL_INIT_LOCK (d->wq_lock);
  d->work_queue = NULL;

  POCL_CREATE_THREAD (d->driver_thread_id, pocl_remote_driver_pthread, device);

  /* Setup SVM. */

  /* CG SVM is implemented via pinned device memory pools and SPIR-V
     manipulation (TODO: document more thoroughly after landing with a working
     solution). */
  if (setup_svm_memory_pool (device) == 0)
    {
      device->svm_caps = CL_DEVICE_SVM_COARSE_GRAIN_BUFFER;
      device->device_usm_capabs = device->host_usm_capabs
        = device->single_shared_usm_capabs
        = CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL;

      /* The CG SVM support can be used for "pinned buffers" as well and
         USM. */
      const char *bonus_extensions = "cl_ext_buffer_device_address "
                                     " cl_intel_unified_shared_memory";
      unsigned exts_str_size
          = strlen (device->extensions) + 1 + strlen (bonus_extensions);
      char *exts_w_pinned = calloc (exts_str_size + 1, 1);
      strncpy (exts_w_pinned, device->extensions, strlen (device->extensions));
      exts_w_pinned[strlen (device->extensions)] = ' ';
      strncpy (exts_w_pinned + strlen (device->extensions) + 1,
               bonus_extensions, strlen (bonus_extensions));
      /* The const char * to void * cast is fine here since this value is
         set in pocl_network_setup_devinfo with a strdup. */
      free ((void *)device->extensions);
      device->extensions = exts_w_pinned;
    }

  return CL_SUCCESS;
}

cl_int
pocl_remote_setup_peer_mesh (struct pocl_device_ops *ops)
{
  return pocl_network_setup_peer_mesh ();
}

cl_int
pocl_remote_uninit (unsigned j, cl_device_id device)
{
  // TODO thread signal
  pocl_network_free_device (device);
  return CL_SUCCESS;
}

/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/

// TODO program->binaries, program->binary_sizes set up, but caching on them is
// wrong

static void
create_build_hash (cl_program program, cl_device_id device, unsigned device_i)
{
  assert (program->build_hash[device_i][2] == 0);

  // TODO program->binaries, program->binary_sizes set up, but need caching
  SHA1_CTX hash_ctx;
  pocl_SHA1_Init (&hash_ctx);
  pocl_SHA1_Update (&hash_ctx, (uint8_t *)program->binaries[device_i],
                    program->binary_sizes[device_i]);

  char *dev_hash = device->ops->build_hash (device);
  pocl_SHA1_Update (&hash_ctx, (const uint8_t *)dev_hash, strlen (dev_hash));
  free (dev_hash);

  uint8_t digest[SHA1_DIGEST_SIZE];
  pocl_SHA1_Final (&hash_ctx, digest);

  unsigned char *hashstr = program->build_hash[device_i];
  size_t i;
  for (i = 0; i < SHA1_DIGEST_SIZE; i++)
    {
      *hashstr++ = (digest[i] & 0x0F) + 65;
      *hashstr++ = ((digest[i] & 0xF0) >> 4) + 65;
    }
  *hashstr = 0;

  program->build_hash[device_i][2] = '/';

  return;
}

/**
 * Move build logs into the program object according to the device list
 * specified for the program.
 */
static void
setup_build_logs (cl_program program, unsigned num_relevant_devices,
                  unsigned *build_indexes, char **build_logs)
{
  unsigned i, j;
  for (i = 0; i < num_relevant_devices; ++i)
    {
      unsigned build_i = build_indexes[i];
      assert (build_i < program->num_devices);

      program->build_log[build_i] = build_logs[i];
      build_logs[i] = NULL;
    }
}

/**
 * Setup mapping from global device indices to indices in the list of devices
 * the program has been requested to be built for.
 */
static unsigned
setup_relevant_devices (cl_program program, cl_device_id device,
                        unsigned *build_indexes, char **build_logs,
                        uint32_t *relevant_devices,
                        uint32_t *relevant_platforms,
                        char **binaries, size_t *binary_sizes,
                        size_t *total_binary_request_size)
{
  remote_server_data_t *server
      = ((remote_device_data_t *)device->data)->server;
  unsigned num_relevant_devices = 0;
  char program_bc_path[POCL_MAX_PATHNAME_LENGTH];
  unsigned i, j;

  for (i = 0; i < program->num_devices; ++i)
    {
      assert (program->devices[i]);
      unsigned real_i = i;
      if (strncmp (program->devices[i]->ops->device_name, remote_device_name,
                   7)
          != 0)
        continue;

      remote_device_data_t *ddata = program->devices[i]->data;
      if (ddata->server == server)
        {
          build_indexes[num_relevant_devices] = i;
          build_logs[num_relevant_devices] = NULL;
          if (program->source)
            {
              binaries[num_relevant_devices] = NULL;
              binary_sizes[num_relevant_devices] = 0;
            }
          else if (program->program_il_size > 0)
            {
              /* SPIR-V. */
              binaries[num_relevant_devices] = program->program_il;
              *total_binary_request_size += sizeof (uint32_t);
              binary_sizes[num_relevant_devices] = program->program_il_size;
              *total_binary_request_size += program->program_il_size;
            }
          else if (program->binary_sizes[real_i] > 0)
            {
              /* Target-specific binaries. */
              pocl_cache_program_bc_path (program_bc_path, program, real_i);

              assert (pocl_exists (program_bc_path));
              assert (program->binaries[real_i]);

              binaries[num_relevant_devices] = program->binaries[real_i];
              *total_binary_request_size += sizeof (uint32_t);
              binary_sizes[num_relevant_devices]
                  = program->binary_sizes[real_i];
              *total_binary_request_size += program->binary_sizes[real_i];
            }
          else
            {
              /* Linking pre-compiled programs. */
            }
          relevant_platforms[num_relevant_devices]
              = ddata->remote_platform_index;
          relevant_devices[num_relevant_devices] = ddata->remote_device_index;
          ++num_relevant_devices;
        }
    }

  return num_relevant_devices;
}

// SYNCHRONOUS

#define WRITE_BYTES(var)                                                      \
  memcpy (buf, &var, sizeof (var));                                           \
  buf += sizeof (var);                                                        \
  assert ((size_t)(buf - buffer) <= buffer_size);

#define WRITE_STRING(str, len)                                                \
  memcpy (buf, str, len);                                                     \
  buf += len;                                                                 \
  assert ((size_t)(buf - buffer) <= buffer_size);

int
pocl_remote_free_program (cl_device_id device, cl_program program,
                          unsigned program_device_i)
{
  remote_device_data_t *d = device->data;

  // this can happen if the build fails
  if (program->data == NULL)
    return CL_SUCCESS;

  if (program->data[program_device_i] == NULL)
    return CL_SUCCESS;

  program_data_t *pd = program->data[program_device_i];
  --pd->refcount;
  if (pd->refcount == 0)
    {
      if (pd->kernel_meta_bytes)
        POCL_MEM_FREE (pd->kernel_meta_bytes);
      POCL_MEM_FREE (program->data[program_device_i]);
    }
  else
    program->data[program_device_i] = NULL;

  int err = pocl_network_free_program (d, (uint32_t)program->id);

  return err;
}

int
pocl_remote_build_source (cl_program program, cl_uint device_i,
                          cl_uint num_input_headers,
                          const cl_program *input_headers,
                          const char **header_include_names,
                          int link_program)
{
  cl_device_id device = program->devices[device_i];
  remote_device_data_t *d = device->data;
  assert (strncmp (device->ops->device_name, remote_device_name, 7) == 0);

  assert (program->source);
  size_t source_len = strlen (program->source);
  POCL_RETURN_ERROR_ON ((source_len == 0), CL_BUILD_PROGRAM_FAILURE,
                        "remote driver does not build empty programs ATM\n");

  uint32_t prog_id = program->id;
  assert (prog_id);

  if (program->data[device_i] != NULL)
    {
      POCL_MSG_PRINT_REMOTE ("Program %i already built for device %u \n",
                             prog_id, device_i);
      return CL_SUCCESS;
    }
  else
    POCL_MSG_PRINT_REMOTE ("Building Program %i for device %u \n", prog_id,
                           device_i);

  unsigned i, j;
  unsigned num_relevant_devices = 0;
  unsigned num_devices = program->num_devices;

  uint32_t relevant_devices[num_devices];
  uint32_t relevant_platforms[num_devices];

  unsigned build_indexes[num_devices];
  char *build_logs[num_devices];

  char *binaries[num_devices];
  size_t binary_sizes[num_devices];

  size_t total_binary_request_size = sizeof (uint32_t);
  int err;

  num_relevant_devices = setup_relevant_devices (
      program, device,
      build_indexes, build_logs,
      relevant_devices, relevant_platforms,
      binaries, binary_sizes, &total_binary_request_size);

  assert (num_relevant_devices > 0);

  char *kernel_meta_bytes = NULL;
  size_t kernel_meta_size = 0;

  err = pocl_network_build_or_link_program (
      d, program->source, strlen (program->source), CL_FALSE, CL_FALSE,
      CL_FALSE, prog_id, program->compiler_options, &kernel_meta_bytes,
      &kernel_meta_size, relevant_devices, relevant_platforms,
      num_relevant_devices, build_logs, binaries, binary_sizes,
      d->svm_region_offset, !link_program, 0);

  setup_build_logs (program, num_relevant_devices, build_indexes, build_logs);

  if (err)
    return err;

  program_data_t *pd = malloc (sizeof (program_data_t));
  pd->kernel_meta_bytes = kernel_meta_bytes;
  pd->kernel_meta_size = kernel_meta_size;
  pd->refcount = num_relevant_devices;

  // for source builds, get the binaries
  {
    char program_bc_path[POCL_MAX_PATHNAME_LENGTH];
    char temp_path[POCL_MAX_PATHNAME_LENGTH];

    for (i = 0; i < num_relevant_devices; ++i)
      {
        unsigned real_i = build_indexes[i];
        assert (real_i < program->num_devices);
        POCL_MSG_PRINT_REMOTE ("DEV i %u real_i %u\n", i, real_i);

        program->data[real_i] = pd;
        program->binary_sizes[real_i] = binary_sizes[i];
        binary_sizes[i] = 0;
        program->binaries[real_i] = binaries[i];
        binaries[i] = NULL;

        assert (program->binary_sizes[real_i] > 0);
        POCL_MSG_PRINT_REMOTE ("BINARY SIZE [%u]: %zu \n", real_i,
                               program->binary_sizes[real_i]);

        create_build_hash (program, device, real_i);

        pocl_cache_create_program_cachedir (program, real_i, NULL, 0,
                                            program_bc_path);

        if (pocl_exists (program_bc_path) == 0)
          {
            err = pocl_cache_write_generic_objfile (
                temp_path, (char *)program->binaries[real_i],
                program->binary_sizes[real_i]);
            assert (err == 0);
            pocl_rename (temp_path, program_bc_path);
          }
      }
  }

  return CL_SUCCESS;
}

int
pocl_remote_build_binary (cl_program program, cl_uint device_i,
                          int link_program, int spirv_build)
{
  cl_device_id device = program->devices[device_i];
  remote_device_data_t *d = device->data;
  assert (strncmp (device->ops->device_name, remote_device_name, 7) == 0);

  if (program->data[device_i] != NULL)
    {
      POCL_MSG_PRINT_LLVM ("Program already built for device %u \n", device_i);
      return CL_SUCCESS;
    }
  uint32_t prog_id = program->id;
  assert (prog_id);

  unsigned i, j;
  unsigned num_relevant_devices = 0;
  unsigned num_devices = program->num_devices;

  uint32_t relevant_devices[num_devices];
  uint32_t relevant_platforms[num_devices];

  unsigned build_indexes[num_devices];
  char *build_logs[num_devices];

  char *binaries[num_devices];
  size_t binary_sizes[num_devices];

  size_t total_binary_request_size = sizeof (uint32_t);
  int err;

  num_relevant_devices = setup_relevant_devices (
      program, device,
      build_indexes, build_logs,
      relevant_devices, relevant_platforms,
      binaries, binary_sizes, &total_binary_request_size);

  assert (num_relevant_devices > 0);

  char *kernel_meta_bytes = NULL;
  size_t kernel_meta_size = 0;
  assert ((!spirv_build && program->pocl_binaries[device_i] != NULL)
          || (spirv_build && program->program_il != NULL));
  {
    char *buffer = malloc (total_binary_request_size);
    assert (buffer);
    char *buf = buffer;
    size_t buffer_size = total_binary_request_size;

    WRITE_BYTES (num_relevant_devices);
    for (i = 0; i < num_relevant_devices; ++i)
      {
        uint32_t s = binary_sizes[i];
        WRITE_BYTES (s);
        WRITE_STRING (binaries[i], s);
      }

    assert ((size_t)(buf - buffer) == total_binary_request_size);

    err = pocl_network_build_or_link_program (
        d, buffer, total_binary_request_size, CL_TRUE, CL_FALSE, spirv_build,
        prog_id, program->compiler_options, &kernel_meta_bytes,
        &kernel_meta_size, relevant_devices, relevant_platforms,
        num_relevant_devices, build_logs, NULL, NULL, d->svm_region_offset,
        !link_program, 0);
    free (buffer);
  }

  ///////////////////////////////////////////////////////////////////////////

  setup_build_logs (program, num_relevant_devices, build_indexes, build_logs);

  if (err)
    return err;

  program_data_t *pd = malloc (sizeof (program_data_t));
  pd->kernel_meta_bytes = kernel_meta_bytes;
  pd->kernel_meta_size = kernel_meta_size;
  pd->refcount = num_relevant_devices;

  for (i = 0; i < num_relevant_devices; ++i)
    {
      unsigned real_i = build_indexes[i];
      assert (real_i < program->num_devices);
      POCL_MSG_PRINT_REMOTE ("DEV i %u real_i %u\n", i, real_i);

      program->data[real_i] = pd;
      assert ((!spirv_build && program->binary_sizes[real_i] > 0)
              || spirv_build && program->program_il_size > 0);
      assert ((!spirv_build && program->binaries[real_i] != NULL)
              || (spirv_build && program->program_il != NULL));

      if (spirv_build)
        {
          /* Dump the SPIR-V given as an input. */
          char spirv_path[POCL_MAX_PATHNAME_LENGTH];
          pocl_cache_tempname (spirv_path, ".spv", NULL);

          assert (program->program_il != NULL);
          assert (program->program_il_size > 0);

          err = pocl_write_file (spirv_path, program->program_il,
                                 program->program_il_size, 0);
          POCL_RETURN_ERROR_ON (
              (err != 0), CL_BUILD_PROGRAM_FAILURE,
              "failed to write the SPIR-V file into cache\n");
          pocl_cache_create_program_cachedir (
              program, real_i, program->program_il, program->program_il_size,
              spirv_path);
        }
    }

  return CL_SUCCESS;
}

int
pocl_remote_build_builtin (cl_program program, cl_uint device_i)
{
  cl_device_id device = program->devices[device_i];
  remote_device_data_t *d = device->data;
  assert (strncmp (device->ops->device_name, remote_device_name, 7) == 0);

  assert (program->num_builtin_kernels > 0);

  uint32_t prog_id = program->id;
  assert (prog_id);

  if (program->data[device_i] != NULL)
    {
      POCL_MSG_PRINT_REMOTE ("Program %i already built for device %u \n",
                             prog_id, device_i);
      return CL_SUCCESS;
    }
  else
    POCL_MSG_PRINT_REMOTE (
        "Building Program %i with builtins for device %u \n", prog_id,
        device_i);

  char *kernel_meta_bytes = NULL;
  size_t kernel_meta_size = 0;

  char *build_log = NULL;

  int err = pocl_network_build_or_link_program (
      d, program->concated_builtin_names,
      strlen (program->concated_builtin_names), CL_FALSE, CL_TRUE, CL_FALSE,
      prog_id, program->compiler_options, &kernel_meta_bytes,
      &kernel_meta_size,
      &d->remote_device_index,   // relevant_devices,
      &d->remote_platform_index, // relevant_platforms,,
      1, &build_log, NULL, 0, 0, 0, 0);

  if (err)
    return err;

  program_data_t *pd = malloc (sizeof (program_data_t));
  pd->kernel_meta_bytes = kernel_meta_bytes;
  pd->kernel_meta_size = kernel_meta_size;
  pd->refcount = 1;

  program->data[device_i] = pd;

  return CL_SUCCESS;
}

int pocl_remote_link_program (cl_program program, cl_uint device_i,
                              cl_uint num_input_programs,
                              const cl_program *input_programs, int create_library)
{
  /* Refer to the programs via their program ids, assuming they are found on
     the server side after compiling with clCompileProgram(). */
  cl_device_id device = program->devices[device_i];
  remote_device_data_t *d = device->data;
  assert (strncmp (device->ops->device_name, remote_device_name, 7) == 0);

  if (program->data[device_i] != NULL)
    {
      POCL_MSG_PRINT_LLVM ("Program already linked for device %u \n",
                           device_i);
      return CL_SUCCESS;
    }
  uint32_t target_prog_id = program->id;
  assert (target_prog_id);

  unsigned i, j;
  unsigned num_relevant_devices = 0;
  unsigned num_devices = program->num_devices;

  uint32_t relevant_devices[num_devices];
  uint32_t relevant_platforms[num_devices];

  unsigned build_indexes[num_devices];
  char *build_logs[num_devices];

  char *binaries[num_devices];
  size_t binary_sizes[num_devices];

  size_t total_binary_request_size = sizeof (uint32_t);
  int err;

  char program_bc_path[POCL_MAX_PATHNAME_LENGTH];

  /* We are creating a new program out of the previously compiled programs,
     there is no build dir for the linked program yet. */
  create_build_hash (program, device, device_i);
  pocl_cache_create_program_cachedir (program, device_i, NULL, 0,
                                      program_bc_path);

  num_relevant_devices = setup_relevant_devices (
      program, device, build_indexes, build_logs, relevant_devices,
      relevant_platforms, binaries, binary_sizes, &total_binary_request_size);

  assert (num_relevant_devices > 0);

  char *kernel_meta_bytes = NULL;
  size_t kernel_meta_size = 0;

  uint32_t input_prog_ids[num_input_programs + 1];
  input_prog_ids[0] = num_input_programs;
  for (i = 0; i < num_input_programs; ++i)
    input_prog_ids[i + 1] = input_programs[i]->id;

  total_binary_request_size
      = sizeof (uint32_t) + sizeof (uint32_t) * num_input_programs;

  err = pocl_network_build_or_link_program (
      d, (const void *)&input_prog_ids[0], total_binary_request_size, 0, 0, 0,
      target_prog_id, program->compiler_options, &kernel_meta_bytes,
      &kernel_meta_size, relevant_devices, relevant_platforms,
      num_relevant_devices, build_logs, binaries, binary_sizes, 0, 0, 1);

  ///////////////////////////////////////////////////////////////////////////

  setup_build_logs (program, num_relevant_devices, build_indexes, build_logs);

  if (err)
    return err;

  program_data_t *pd = malloc (sizeof (program_data_t));
  pd->kernel_meta_bytes = kernel_meta_bytes;
  pd->kernel_meta_size = kernel_meta_size;
  pd->refcount = num_relevant_devices;

  /* Dump the binaries to the cache, similarly like this was a source build,
     only the "sources" were the precompiled programs. */
  {
    char program_bc_path[POCL_MAX_PATHNAME_LENGTH];
    char temp_path[POCL_MAX_PATHNAME_LENGTH];

    for (i = 0; i < num_relevant_devices; ++i)
      {
        unsigned real_i = build_indexes[i];
        assert (real_i < program->num_devices);
        POCL_MSG_PRINT_REMOTE ("DEV i %u real_i %u\n", i, real_i);

        program->data[real_i] = pd;
        program->binary_sizes[real_i] = binary_sizes[i];
        binary_sizes[i] = 0;
        program->binaries[real_i] = binaries[i];
        binaries[i] = NULL;

        assert (program->binary_sizes[real_i] > 0);
        POCL_MSG_PRINT_REMOTE ("BINARY SIZE [%u]: %zu \n", real_i,
                               program->binary_sizes[real_i]);

        if (pocl_exists (program_bc_path) == 0)
          {
            err = pocl_cache_write_generic_objfile (
                temp_path, (char *)program->binaries[real_i],
                program->binary_sizes[real_i]);
            assert (err == 0);
            pocl_rename (temp_path, program_bc_path);
          }
      }
  }

  return CL_SUCCESS;
}

int
pocl_remote_supports_binary (cl_device_id device, size_t length,
                             const char *binary)
{
  if (pocl_bitcode_is_spirv_execmodel_kernel (binary, length)
      && device->supported_spir_v_versions != NULL
      && strncmp (device->supported_spir_v_versions, "SPIR-V", 6) == 0)
    return 1;
  /* We should delegate to the remote here to be strict. */
  return 0;
}

int
pocl_remote_setup_metadata (cl_device_id device, cl_program program,
                            unsigned program_device_i)
{
  if (program->data == NULL)
    return 0;

  if (program->data[program_device_i] == NULL)
    return 0;

  program_data_t *pd = program->data[program_device_i];

  if (pd->kernel_meta_bytes)
    {
      size_t num_kernels = 0;
      pocl_kernel_metadata_t *kernel_meta = NULL;
      int err = pocl_network_setup_metadata (pd->kernel_meta_bytes,
                                             pd->kernel_meta_size, program,
                                             &num_kernels, &kernel_meta);
      assert (err == CL_SUCCESS);
      program->num_kernels = num_kernels;
      program->kernel_meta = kernel_meta;
      return 1;
    }
  else
    return 0;
}

/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/

int
pocl_remote_create_kernel (cl_device_id device, cl_program program,
                           cl_kernel kernel, unsigned device_i)
{
  assert (program->data[device_i] != NULL);
  uint32_t prog_id = (uint32_t)program->id;
  assert (prog_id);

  assert (kernel->data != NULL);
  assert (kernel->data[device_i] == NULL);
  kernel_data_t *kd = (kernel_data_t *)calloc (1, sizeof (kernel_data_t));
  kernel->data[device_i] = kd;
  uint32_t kern_id = (uint32_t)kernel->id;
  assert (kern_id);

  unsigned i;
  for (i = 0; i < kernel->meta->num_args; ++i)
    {
      POCL_MSG_PRINT_REMOTE ("CREATE KERNEL | ARG %u TYPE: %u  \n", i,
                             kernel->meta->arg_info[i].type);
    }

  kd->arg_array = calloc ((kernel->meta->num_args), sizeof (uint64_t));
  kd->ptr_is_svm = calloc ((kernel->meta->num_args), sizeof (unsigned char));

  return pocl_network_create_kernel (device->data, kernel->name, prog_id,
                                     kern_id, kd);
}

int
pocl_remote_free_kernel (cl_device_id device, cl_program program,
                         cl_kernel kernel, unsigned device_i)
{
  assert (kernel->data != NULL);

  // may happen if creating kernel fails
  if (kernel->data[device_i] == NULL)
    return CL_SUCCESS;

  kernel_data_t *kd = (kernel_data_t *)kernel->data[device_i];
  uint32_t kern_id = (uint32_t)kernel->id;
  uint32_t prog_id = (uint32_t)program->id;
  assert (kern_id);

  int err = pocl_network_free_kernel (device->data, kd, kern_id, prog_id);

  POCL_MEM_FREE (kd->arg_array);
  POCL_MEM_FREE (kd->ptr_is_svm);
  POCL_MEM_FREE (kd->pod_arg_storage);
  POCL_MEM_FREE (kd);
  kernel->data[device_i] = NULL;

  return err;
}

int
pocl_remote_init_queue (cl_device_id device, cl_command_queue queue)
{
  assert (queue->data == NULL);

  remote_device_data_t *d = device->data;

  remote_queue_data_t *dd = pocl_aligned_malloc (HOST_CPU_CACHELINE_SIZE,
                                                 sizeof (remote_queue_data_t));
  assert (dd);
  queue->data = dd;

  POCL_INIT_COND (dd->cq_cond);

  dd->printf_buffer = calloc (1, device->printf_buffer_size);
  uint32_t queue_id = (uint32_t)queue->id;
  assert (queue_id);

  return pocl_network_create_queue (d, queue_id);
}

int
pocl_remote_free_queue (cl_device_id device, cl_command_queue queue)
{
  remote_device_data_t *d = device->data;
  remote_queue_data_t *qd = queue->data;

  if (queue->data == NULL)
    return CL_SUCCESS;

  uint32_t queue_id = (uint32_t)queue->id;
  assert (queue_id);

  int err = pocl_network_free_queue (d, queue_id);
  if (err != CL_SUCCESS)
    goto ERROR;

  POCL_DESTROY_COND (qd->cq_cond);

  POCL_MEM_FREE (qd->printf_buffer);

ERROR:
  POCL_MEM_FREE (queue->data);
  return err;
}

int
pocl_remote_create_sampler (cl_device_id device, cl_sampler samp,
                            unsigned device_i)
{
  uint32_t samp_id = (uint32_t)samp->id;
  assert (samp_id);
  assert (samp->device_data[device_i] == NULL);
  int err = pocl_network_create_sampler (device->data, samp->normalized_coords,
                                         samp->addressing_mode,
                                         samp->filter_mode, samp_id);
  samp->device_data[device_i] = (void *)((uintptr_t)samp_id);
  return err;
}

int
pocl_remote_free_sampler (cl_device_id device, cl_sampler samp,
                          unsigned device_i)
{
  uint32_t samp_id = (uint32_t)samp->id;
  assert (samp_id);
  assert (samp->device_data[device_i] != NULL);
  int err = pocl_network_free_sampler (device->data, samp_id);
  samp->device_data[device_i] = NULL;
  return err;
}

static void
remote_push_command (_cl_command_node *node)
{
  cl_device_id device = node->device;
  remote_device_data_t *d = (remote_device_data_t *)device->data;

  POCL_LOCK (d->wq_lock);
  DL_APPEND (d->work_queue, node);
  POCL_SIGNAL_COND (d->wakeup_cond);
  POCL_UNLOCK (d->wq_lock);
}

void
pocl_remote_submit (_cl_command_node *node, cl_command_queue cq)
{
  cl_event e = node->sync.event.event;
  assert (e->data == NULL);

  pocl_remote_event_data_t *e_d = NULL;
  e_d = calloc (1, sizeof (pocl_remote_event_data_t));
  assert (e_d);
  POCL_INIT_COND (e_d->event_cond);
  e->data = (void *)e_d;

  node->ready = 1;
  if (pocl_command_is_ready (node->sync.event.event))
    {
      pocl_update_event_submitted (node->sync.event.event);
      remote_push_command (node);
    }
  POCL_UNLOCK_OBJ (node->sync.event.event);
  return;
}

void
pocl_remote_notify_cmdq_finished (cl_command_queue cq)
{
  /* must be called with CQ already locked.
   * this must be a broadcast since there could be multiple
   * user threads waiting on the same command queue */
  remote_queue_data_t *dd = cq->data;
  POCL_BROADCAST_COND (dd->cq_cond);
}

void
pocl_remote_join (cl_device_id device, cl_command_queue cq)
{
  POCL_LOCK_OBJ (cq);
  remote_queue_data_t *dd = cq->data;

  while (1)
    {
      if (cq->command_count == 0)
        {
          POCL_UNLOCK_OBJ (cq);
          return;
        }
      else
        {
          POCL_MSG_PRINT_EVENTS (
            "remote: waiting for commands(s), last event id %zu\n",
            cq->last_event.event->id);
          POCL_WAIT_COND (dd->cq_cond, cq->pocl_lock);
        }
    }
}

void
pocl_remote_flush (cl_device_id device, cl_command_queue cq)
{
  // TODO later...
}

void
pocl_remote_notify (cl_device_id device, cl_event event, cl_event finished)
{
  _cl_command_node *node = event->command;

  POCL_MSG_PRINT_EVENTS ("remote: notify finished event %lu to event %lu \n",
                         finished->id, event->id);

  if (finished->status < CL_COMPLETE)
    {
      pocl_update_event_failed (event);
      return;
    }

  if (!node->ready)
    {
      POCL_MSG_PRINT_EVENTS (
          "remote: command related to the notified event %lu not ready\n",
          event->id);
      return;
    }

  if (pocl_command_is_ready (node->sync.event.event))
    {
      assert (event->status == CL_QUEUED);
      pocl_update_event_submitted (event);
      remote_push_command (node);
    }
  else
    {
      POCL_MSG_PRINT_EVENTS (
          "remote: sync event %lu is not ready for the notified event %lu\n",
          node->sync.event.event->id, event->id);
    }

  return;
}

void
pocl_remote_wait_event (cl_device_id device, cl_event event)
{
  POCL_MSG_PRINT_EVENTS ("remote: device->wait_event on event %lu\n",
                         event->id);
  pocl_remote_event_data_t *e_d = (pocl_remote_event_data_t *)event->data;

  POCL_LOCK_OBJ (event);
  while (event->status > CL_COMPLETE)
    {
      POCL_WAIT_COND (e_d->event_cond, event->pocl_lock);
    }
  POCL_UNLOCK_OBJ (event);

  POCL_MSG_PRINT_EVENTS (
      "remote: wait on event %lu finished with status: %i\n", event->id,
      event->status);
  assert (event->status <= CL_COMPLETE);
}

void
pocl_remote_free_event_data (cl_event event)
{
  assert (event->data != NULL);
  pocl_remote_event_data_t *e_d = (pocl_remote_event_data_t *)event->data;
  POCL_DESTROY_COND (e_d->event_cond);
  POCL_MEM_FREE (event->data);
}

void
pocl_remote_notify_event_finished (cl_event event)
{
  pocl_remote_event_data_t *e_d = (pocl_remote_event_data_t *)event->data;
  POCL_BROADCAST_COND (e_d->event_cond);
}

static void
remote_finish_command (void *arg, _cl_command_node *node,
                       size_t extra_rep_bytes)
{
  assert (node);
  cl_event event = node->sync.event.event;
  _cl_command_t *cmd = &node->command;
  remote_device_data_t *d = arg;
  cl_mem m = NULL;

  switch (node->type)
    {
    case CL_COMMAND_READ_BUFFER:
      if (extra_rep_bytes < node->command.read.size)
        {
          m = node->command.read.src;
          POCL_LOCK_OBJ (m);
          m->content_size = extra_rep_bytes;
          if (node->command.read.content_size)
            *node->command.read.content_size = extra_rep_bytes;
          POCL_UNLOCK_OBJ (m);
        }
      break;

    case CL_COMMAND_MAP_BUFFER:
      if (extra_rep_bytes < node->command.map.mapping->size)
        {
          m = node->command.map.buffer;
          POCL_LOCK_OBJ (m);
          m->content_size = extra_rep_bytes;
          POCL_UNLOCK_OBJ (m);
        }
      break;

    case CL_COMMAND_FILL_BUFFER:
      break;
    }

  POCL_LOCK (d->wq_lock);
  DL_APPEND (d->finished_list, node);
  POCL_SIGNAL_COND (d->wakeup_cond);
  POCL_UNLOCK (d->wq_lock);
}

int
pocl_remote_async_read (void *data, _cl_command_node *node,
                        void *__restrict__ host_ptr,
                        pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                        size_t offset, size_t size)
{
  uint32_t queue_id = (uint32_t)node->sync.event.event->queue->id;
  uintptr_t mem_id = (uintptr_t)src_mem_id->mem_ptr;
  uintptr_t size_id = 0;

  uint32_t content_size_id = 0;
  if (node->command.read.src_content_size_mem_id)
    content_size_id
        = (uintptr_t)node->command.read.src_content_size_mem_id->mem_ptr;

  return pocl_network_read (queue_id, data, mem_id, 0, content_size_id,
                            host_ptr, offset, size, remote_finish_command,
                            data, node);
}

int
pocl_remote_can_migrate_d2d (cl_device_id dest, cl_device_id source)
{
  // either is not remote, can't migrate D2D
  if ((strncmp (dest->ops->device_name, remote_device_name, 7) != 0)
      || (strncmp (source->ops->device_name, remote_device_name, 7) != 0))
    return 0;

  // both are remote
  remote_device_data_t *ddest = (remote_device_data_t *)dest->data;
  remote_device_data_t *dsrc = (remote_device_data_t *)source->data;

  // peer migration between remote servers
  if (ddest->server != dsrc->server)
    {
      return 10;
    }

  // migration within 1 server but 2 different platforms
  if (ddest->remote_platform_index != dsrc->remote_platform_index)
    {
      return 20;
    }

  // migration within 1 server & 1 platforms
  if (ddest->remote_device_index != dsrc->remote_device_index)
    {
      return 30;
    }

  // same device
  return 40;
}

int
pocl_remote_async_migrate_d2d (void *dest_data, void *source_data,
                               _cl_command_node *node, cl_mem mem,
                               pocl_mem_identifier *p)
{
  uintptr_t mem_id = (uintptr_t)p->mem_ptr;
  uint32_t queue_id = (uint32_t)node->sync.event.event->queue->id;

  uint32_t size_id = 0;
  if (mem->size_buffer != NULL)
    size_id
        = (uintptr_t)node->command.migrate.src_content_size_mem_id->mem_ptr;

  int r;
  r = pocl_network_migrate_d2d (
      queue_id, mem_id, size_id, mem->is_image, mem->image_height,
      mem->image_width, mem->image_depth, mem->size, dest_data, source_data,
      remote_finish_command, dest_data, node);

  assert (r == 0);
  return r;
}

int
pocl_remote_async_write (void *data, _cl_command_node *node,
                         const void *__restrict__ host_ptr,
                         pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                         size_t offset, size_t size)
{
  uintptr_t mem_id = (uintptr_t)dst_mem_id->mem_ptr;

  uint32_t queue_id = (uint32_t)node->sync.event.event->queue->id;

  return pocl_network_write (queue_id, data, mem_id, 0, host_ptr, offset, size,
                             remote_finish_command, data, node);
}

int
pocl_remote_async_copy (void *data, _cl_command_node *node,
                        pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                        pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                        size_t dst_offset, size_t src_offset, size_t size)
{
  uintptr_t src_id = (uintptr_t)src_mem_id->mem_ptr;
  uintptr_t dst_id = (uintptr_t)dst_mem_id->mem_ptr;

  if ((src_id == dst_id) && (src_offset == dst_offset))
    {
      return 1;
    }

  uint32_t queue_id = (uint32_t)node->sync.event.event->queue->id;

  uint32_t content_size_id = 0;
  if (node->command.copy.src_content_size_mem_id)
    content_size_id
        = (uintptr_t)node->command.copy.src_content_size_mem_id->mem_ptr;

  int r = pocl_network_copy (queue_id, data, src_id, dst_id, content_size_id,
                             src_offset, dst_offset, size,
                             remote_finish_command, data, node);
  assert (r == 0);
  return 0;
}

int
pocl_remote_async_copy_rect (void *data, _cl_command_node *node,
                             pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                             pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                             const size_t *__restrict__ const dst_origin,
                             const size_t *__restrict__ const src_origin,
                             const size_t *__restrict__ const region,
                             size_t const dst_row_pitch,
                             size_t const dst_slice_pitch,
                             size_t const src_row_pitch,
                             size_t const src_slice_pitch)
{
  uintptr_t src_id = (uintptr_t)src_mem_id->mem_ptr;
  uintptr_t dst_id = (uintptr_t)dst_mem_id->mem_ptr;
  uint32_t queue_id = (uint32_t)node->sync.event.event->queue->id;

  size_t src_offset = src_origin[0] + src_row_pitch * src_origin[1]
                      + src_slice_pitch * src_origin[2];
  size_t dst_offset = dst_origin[0] + dst_row_pitch * dst_origin[1]
                      + dst_slice_pitch * dst_origin[2];

  if ((src_id == dst_id) && (src_offset == dst_offset))
    {
      return 1;
    }

  POCL_MSG_PRINT_REMOTE ("ASYNC COPY: \nregion %zu %zu %zu\n"
                         "  src_origin %zu %zu %zu\n"
                         "  dst_origin %zu %zu %zu\n"
                         "  dst_row_pitch %zu, dst_slice_pitch %zu\n"
                         "  src_row_pitch %zu, src_slice_pitch %zu\n",

                         region[0], region[1], region[2], src_origin[0],
                         src_origin[1], src_origin[2], dst_origin[0],
                         dst_origin[1], dst_origin[2], dst_row_pitch,
                         dst_slice_pitch, src_row_pitch, src_slice_pitch);

  int r = pocl_network_copy_rect (
      queue_id, data, src_id, dst_id, dst_origin, src_origin, region,
      dst_row_pitch, dst_slice_pitch, src_row_pitch, src_slice_pitch,
      remote_finish_command, data, node);
  assert (r == 0);
  return 0;
}

void
pocl_remote_async_write_rect (void *data, _cl_command_node *node,
                              const void *__restrict__ const host_ptr,
                              pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                              const size_t *__restrict__ const buffer_origin,
                              const size_t *__restrict__ const host_origin,
                              const size_t *__restrict__ const region,
                              size_t const buffer_row_pitch,
                              size_t const buffer_slice_pitch,
                              size_t const host_row_pitch,
                              size_t const host_slice_pitch)
{
  uintptr_t dst_id = (uintptr_t)dst_mem_id->mem_ptr;

  uint32_t queue_id = (uint32_t)node->sync.event.event->queue->id;

  size_t byte_offset_start = host_origin[0] + host_row_pitch * host_origin[1]
                             + host_slice_pitch * host_origin[2];

  size_t total_size = region[0] + host_row_pitch * (region[1] - 1)
                      + host_slice_pitch * (region[2] - 1);

  char const *__restrict__ const adjusted_host_ptr
      = (char const *)host_ptr + byte_offset_start;

  POCL_MSG_PRINT_REMOTE (
      "ASYNC WRITE: \nregion %zu %zu %zu\n"
      "  buffer_origin %zu %zu %zu\n"
      "  host_origin %zu %zu %zu\n"
      "  offset %zu total_size %zu\n"
      "  buffer_row_pitch %zu, buffer_slice_pitch %zu\n"
      "  host_row_pitch %zu, host_slice_pitch %zu\n",
      region[0], region[1], region[2], buffer_origin[0], buffer_origin[1],
      buffer_origin[2], host_origin[0], host_origin[1], host_origin[2],
      (size_t)(adjusted_host_ptr - (const char *)host_ptr), total_size,
      buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch);

  /*************************************************************************/

  int r = pocl_network_write_rect (
      queue_id, data, dst_id, buffer_origin, region, buffer_row_pitch,
      buffer_slice_pitch, adjusted_host_ptr, total_size, remote_finish_command,
      data, node);
  assert (r == 0);
}

void
pocl_remote_async_read_rect (void *data, _cl_command_node *node,
                             void *__restrict__ const host_ptr,
                             pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                             const size_t *__restrict__ const buffer_origin,
                             const size_t *__restrict__ const host_origin,
                             const size_t *__restrict__ const region,
                             size_t const buffer_row_pitch,
                             size_t const buffer_slice_pitch,
                             size_t const host_row_pitch,
                             size_t const host_slice_pitch)
{
  uintptr_t src_id = (uintptr_t)src_mem_id->mem_ptr;

  uint32_t queue_id = (uint32_t)node->sync.event.event->queue->id;

  size_t byte_offset_start = host_origin[0] + host_row_pitch * host_origin[1]
                             + host_slice_pitch * host_origin[2];

  size_t total_size = region[0] + host_row_pitch * (region[1] - 1)
                      + host_slice_pitch * (region[2] - 1);

  char *__restrict__ adjusted_host_ptr = (char *)host_ptr + byte_offset_start;

  POCL_MSG_PRINT_REMOTE (
      "ASYNC READ: \nregion %zu %zu %zu\n"
      "  buffer_origin %zu %zu %zu\n"
      "  host_origin %zu %zu %zu\n"
      "  offset %zu total_size %zu\n"
      "  buffer_row_pitch %zu, buffer_slice_pitch %zu\n"
      "  host_row_pitch %zu, host_slice_pitch %zu\n",
      region[0], region[1], region[2], buffer_origin[0], buffer_origin[1],
      buffer_origin[2], host_origin[0], host_origin[1], host_origin[2],
      (size_t)(adjusted_host_ptr - (char *)host_ptr), total_size,
      buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch);

  int r = pocl_network_read_rect (queue_id, data, src_id, buffer_origin,
                                  region, buffer_row_pitch, buffer_slice_pitch,
                                  adjusted_host_ptr, total_size,
                                  remote_finish_command, data, node);
  assert (r == 0);
}

void
pocl_remote_async_memfill (void *data, _cl_command_node *node,
                           pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                           size_t size, size_t offset,
                           const void *__restrict__ pattern,
                           size_t pattern_size)
{
  uintptr_t dst_id = (uintptr_t)dst_mem_id->mem_ptr;

  uint32_t queue_id = (uint32_t)node->sync.event.event->queue->id;

  int r = pocl_network_fill_buffer (queue_id, data, dst_id, size, offset,
                                    pattern, pattern_size,
                                    remote_finish_command, data, node);
  assert (r == 0);
}

int
pocl_remote_async_map_svm_buffer (remote_device_data_t *data,
                                  _cl_command_node *node)
{
  void *svm_ptr = node->command.svm_map.svm_ptr;
  size_t buf_size = node->command.svm_map.size;
  uint32_t queue_id = (uint32_t)node->sync.event.event->queue->id;

  POCL_MSG_PRINT_MEMORY ("REMOTE: MAP SVM buf read "
                         "svm_ptr %p of size %zu\n",
                         svm_ptr, buf_size);

  int r = pocl_network_read (queue_id, data, 0, 1, 0,
                             node->command.svm_map.svm_ptr, 0, buf_size,
                             remote_finish_command, data, node);
  assert (r == 0);
  return 0;
}

int
pocl_remote_async_map_mem (void *data, _cl_command_node *node,
                           pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                           mem_mapping_t *map)
{
  uintptr_t mem_id = (uintptr_t)src_mem_id->mem_ptr;

  void *host_ptr = map->host_ptr;
  size_t offset = map->offset;
  size_t size = map->size;

  uint32_t queue_id = (uint32_t)node->sync.event.event->queue->id;

  uintptr_t size_id = 0;
  if (src_buf->size_buffer != NULL)
    size_id = (uintptr_t)src_buf->size_buffer
                  ->device_ptrs[node->device->global_mem_id]
                  .mem_ptr;

  POCL_MSG_PRINT_MEMORY ("REMOTE: MAP memcpy() "
                         "src_id %lu + offset %zu"
                         "to dst_host_ptr %p\n",
                         mem_id, offset, host_ptr);

  int r = pocl_network_read (queue_id, data, mem_id, 0, size_id, host_ptr,
                             offset, size, remote_finish_command, data, node);
  assert (r == 0);
  return 0;
}

/**
 * Unmaps an SVM buffer.
 *
 * Synchronizes the client-side content to the remote SVM allocation.
 */
int
pocl_remote_async_unmap_svm_buffer (remote_device_data_t *data,
                                    _cl_command_node *node)
{
  uint32_t queue_id = (uint32_t)node->sync.event.event->queue->id;

  void *svm_ptr = node->command.svm_unmap.svm_ptr;
  size_t buf_size = node->command.svm_unmap.size;
  POCL_MSG_PRINT_MEMORY ("REMOTE: UNMAP SVM buf write "
                         "svm_ptr %p of size %zu\n",
                         svm_ptr, buf_size);
  int r = pocl_network_write (queue_id, data, 0, 1, svm_ptr, 0, buf_size,
                              remote_finish_command, data, node);
  assert (r == 0);
  return 0;
}

int
pocl_remote_async_unmap_mem (void *data, _cl_command_node *node,
                             pocl_mem_identifier *dst_mem_id,
                             mem_mapping_t *map)
{
  /* it could be CL_MAP_READ | CL_MAP_WRITE(..invalidate) which has to be
   * handled like a write */
  if (map->map_flags == CL_MAP_READ)
    {
      // we can skip talking to remote, but we still need to call the callback.
      return 1;
    }

  uintptr_t mem_id = dst_mem_id == NULL ? 0 : (uintptr_t)dst_mem_id->mem_ptr;

  void *host_ptr = map->host_ptr;
  assert (host_ptr != NULL);
  size_t offset = map->offset;
  size_t size = map->size;

  uint32_t queue_id = (uint32_t)node->sync.event.event->queue->id;

  POCL_MSG_PRINT_MEMORY ("REMOTE: UNMAP memcpy() "
                         "host_ptr %p to mem_id %lu + offset %zu size %zu\n",
                         host_ptr, mem_id, offset, size);
  int r = pocl_network_write (queue_id, data, mem_id, 0, host_ptr, offset,
                              size, remote_finish_command, data, node);
  assert (r == 0);
  return 0;
}

void
pocl_remote_async_run (void *data, _cl_command_node *cmd)
{
  uint32_t queue_id = (uint32_t)cmd->sync.event.event->queue->id;

  struct pocl_argument *al = NULL;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  unsigned dev_i = cmd->program_device_i;
  int requires_kernarg_update = 0;

  pocl_kernel_metadata_t *kernel_md = kernel->meta;
  remote_device_data_t *ddata = (remote_device_data_t *)data;

  kernel_data_t *kd = (kernel_data_t *)(kernel->data[dev_i]);
  assert (kd != NULL);

  /* TODO this is unecessarily rerun if pod_total_size == 0. */
  if (kd->pod_arg_storage == NULL)
    {
      assert (kd->pod_total_size == 0);
      for (i = 0; i < kernel_md->num_args; ++i)
        {
          al = &(cmd->command.run.arguments[i]);
          if (ARG_IS_LOCAL (kernel_md->arg_info[i]))
            continue;
          if (kernel_md->arg_info[i].type == POCL_ARG_TYPE_NONE)
            {
              kd->pod_total_size += al->size;
            }
        }
      if (kd->pod_total_size > 0)
        kd->pod_arg_storage = calloc (1, kd->pod_total_size);
    }

  char *pod_arg_pointer = kd->pod_arg_storage;
  uint64_t *arg_array = kd->arg_array;
  unsigned char *ptr_is_svm = kd->ptr_is_svm;

  /* Process the kernel arguments.  */
  for (i = 0; i < kernel_md->num_args; ++i)
    {
      ptr_is_svm[i] = 0;
      al = &(cmd->command.run.arguments[i]);
      assert (al->is_set > 0);
      if (ARG_IS_LOCAL (kernel_md->arg_info[i]))
        {
          requires_kernarg_update = 1;
          arg_array[i] = al->size;
        }
      else if (al->is_raw_ptr)
        {
          arg_array[i] = (uint64_t) * (void **)al->value;
          POCL_MSG_PRINT_MEMORY (
            "Adding SVM pool offset %zu to an SVM ptr arg %u (%p to %p)\n",
            ddata->svm_region_offset, i, (void *)arg_array[i],
            (char *)arg_array[i] + ddata->svm_region_offset);
          arg_array[i] = arg_array[i] + ddata->svm_region_offset;
          requires_kernarg_update = 1;
          ptr_is_svm[i] = 1;
        }
      else if ((kernel_md->arg_info[i].type == POCL_ARG_TYPE_POINTER)
               || (kernel_md->arg_info[i].type == POCL_ARG_TYPE_IMAGE))
        {
          /* cl_mem and cl_image refer to opaque identifiers */
          uint32_t mem_id = 0;
          if (al->value)
            {
              cl_mem mem = (*(cl_mem *)(al->value));
              if (mem)
                mem_id
                    = (uintptr_t)(mem->device_ptrs[cmd->device->global_mem_id]
                                      .mem_ptr);
            }
          else
            {
              POCL_MSG_WARN ("NULL PTR ARG DETECTED: %s / ARG %i: %s \n",
                             kernel->name, i, kernel_md->arg_info[i].name);
            }

          if (arg_array[i] != mem_id)
            {
              requires_kernarg_update = 1;
              arg_array[i] = mem_id;
            }
        }
      else if (kernel_md->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          cl_sampler s = *(cl_sampler *)(al->value);
          uint32_t remote_id
              = (uintptr_t)(s->device_data[cmd->device->dev_id]);
          if (arg_array[i] != remote_id)
            {
              requires_kernarg_update = 1;
              arg_array[i] = remote_id;
            }
        }
      else
        {
          assert (kernel_md->arg_info[i].type == POCL_ARG_TYPE_NONE);
          arg_array[i] = al->size;
          if (memcmp (pod_arg_pointer, al->value, al->size) != 0)
            {
              requires_kernarg_update = 1;
              memcpy (pod_arg_pointer, al->value, al->size);
              assert (pod_arg_pointer
                      <= (kd->pod_arg_storage + kd->pod_total_size));
            }
          pod_arg_pointer += al->size;
        }
    }

  assert (pod_arg_pointer <= (kd->pod_arg_storage + kd->pod_total_size));

  vec3_t local
      = { cmd->command.run.pc.local_size[0], cmd->command.run.pc.local_size[1],
          cmd->command.run.pc.local_size[2] };

  ulong *ptr = cmd->command.run.pc.num_groups;
  vec3_t global = { ptr[0] * local.x, ptr[1] * local.y, ptr[2] * local.z };

  vec3_t offset = { cmd->command.run.pc.global_offset[0],
                    cmd->command.run.pc.global_offset[1],
                    cmd->command.run.pc.global_offset[2] };

  int r = pocl_network_run_kernel (queue_id, data, kernel, kd,
                                   requires_kernarg_update,
                                   cmd->command.run.pc.work_dim, local, global,
                                   offset, remote_finish_command, data, cmd);
  assert (r == 0);
}

cl_int
pocl_remote_async_copy_image_rect (
    void *data, _cl_command_node *node, cl_mem src_image, cl_mem dst_image,
    pocl_mem_identifier *src_mem_id, pocl_mem_identifier *dst_mem_id,
    const size_t *src_origin, const size_t *dst_origin, const size_t *region)
{
  uint32_t src_image_id = (uintptr_t)src_mem_id->mem_ptr;
  uint32_t dst_image_id = (uintptr_t)dst_mem_id->mem_ptr;

  POCL_MSG_PRINT_REMOTE ("REMOTE COPY IMAGE RECT \n"
                         "dst_image %p remote id %u \n"
                         "src_image %p remote id %u \n"
                         "dst_origin [0,1,2] %zu %zu %zu \n"
                         "src_origin [0,1,2] %zu %zu %zu \n"
                         "region [0,1,2] %zu %zu %zu \n",
                         dst_image, dst_image_id, src_image, src_image_id,
                         dst_origin[0], dst_origin[1], dst_origin[2],
                         src_origin[0], src_origin[1], src_origin[2],
                         region[0], region[1], region[2]);

  uint32_t queue_id = (uint32_t)node->sync.event.event->queue->id;

  int r = pocl_network_copy_image_rect (
      queue_id, data, src_image_id, dst_image_id, src_origin, dst_origin,
      region, remote_finish_command, data, node);
  return r;
}

cl_int
pocl_remote_async_write_image_rect (void *data, _cl_command_node *node,
                                    cl_mem dst_image,
                                    pocl_mem_identifier *dst_mem_id,
                                    const void *__restrict__ src_host_ptr,
                                    pocl_mem_identifier *src_mem_id,
                                    const size_t *origin, const size_t *region,
                                    size_t src_row_pitch,
                                    size_t src_slice_pitch, size_t src_offset)
{
  uint32_t dst_id = (uintptr_t)dst_mem_id->mem_ptr;

  POCL_MSG_PRINT_REMOTE ("REMOTE WRITE IMAGE RECT \n"
                         "dst_image %p remote id %u \n"
                         "src_hostptr %p \n"
                         "origin [0,1,2] %zu %zu %zu \n"
                         "region [0,1,2] %zu %zu %zu \n"
                         "row %zu slice %zu offset %zu \n",
                         dst_image, dst_id, src_host_ptr, origin[0], origin[1],
                         origin[2], region[0], region[1], region[2],
                         src_row_pitch, src_slice_pitch, src_offset);

  /* copies a region from host OR device buffer to device image.
   * clEnqueueCopyBufferToImage: src_mem_id = buffer,
   *     src_host_ptr = NULL, src_row_pitch = src_slice_pitch = 0
   * clEnqueueWriteImage: src_mem_id = NULL,
   *     src_host_ptr = host pointer, src_offset = 0 */

  uint32_t queue_id = (uint32_t)node->sync.event.event->queue->id;

  if (src_host_ptr == NULL)
    {
      uint32_t src_id = (uintptr_t)src_mem_id->mem_ptr;
      return pocl_network_copy_buf2img (queue_id, data, src_id, src_offset,
                                        dst_id, origin, region,
                                        remote_finish_command, data, node);
    }
  else
    {
      size_t px = dst_image->image_elem_size * dst_image->image_channels;
      if (src_row_pitch == 0)
        src_row_pitch = px * region[0];
      if (src_slice_pitch == 0)
        src_slice_pitch = src_row_pitch * region[1];

      assert (src_host_ptr);

      size_t size = region[0] * px + src_row_pitch * (region[1] - 1)
                    + src_slice_pitch * (region[2] - 1);
      int r = pocl_network_write_image_rect (
          queue_id, data, dst_id, origin, region, src_host_ptr,
          size, // adjusted_host_ptr,
          remote_finish_command, data, node);

      return r;
    }
}

cl_int
pocl_remote_async_read_image_rect (void *data, _cl_command_node *node,
                                   cl_mem src_image,
                                   pocl_mem_identifier *src_mem_id,
                                   void *__restrict__ dst_host_ptr,
                                   pocl_mem_identifier *dst_mem_id,
                                   const size_t *origin, const size_t *region,
                                   size_t dst_row_pitch,
                                   size_t dst_slice_pitch, size_t dst_offset)
{
  uint32_t src_id = (uintptr_t)src_mem_id->mem_ptr;

  POCL_MSG_PRINT_REMOTE ("REMOTE READ IMAGE RECT \n"
                         "src_image %p remote id %u \n"
                         "dst_hostptr %p \n"
                         "origin [0,1,2] %zu %zu %zu \n"
                         "region [0,1,2] %zu %zu %zu \n"
                         "row %zu slice %zu offset %zu \n",
                         src_image, src_id, dst_host_ptr, origin[0], origin[1],
                         origin[2], region[0], region[1], region[2],
                         dst_row_pitch, dst_slice_pitch, dst_offset);

  /* copies a region from device image to host or device buffer
   * clEnqueueCopyImageToBuffer: dst_mem_id = buffer,
   *     dst_host_ptr = NULL, dst_row_pitch = dst_slice_pitch = 0
   * clEnqueueReadImage: dst_mem_id = NULL,
   *     dst_host_ptr = host pointer, dst_offset = 0
   */

  uint32_t queue_id = (uint32_t)node->sync.event.event->queue->id;

  if (dst_host_ptr == NULL)
    {
      uint32_t dst_id = (uintptr_t)dst_mem_id->mem_ptr;
      return pocl_network_copy_img2buf (queue_id, data, dst_id, dst_offset,
                                        src_id, origin, region,
                                        remote_finish_command, data, node);
    }
  else
    {

      size_t px = src_image->image_elem_size * src_image->image_channels;
      if (dst_row_pitch == 0)
        dst_row_pitch = px * region[0];
      if (dst_slice_pitch == 0)
        dst_slice_pitch = dst_row_pitch * region[1];

      assert (dst_host_ptr);

      size_t size = region[0] * px + dst_row_pitch * (region[1] - 1)
                    + dst_slice_pitch * (region[2] - 1);

      int r = pocl_network_read_image_rect (queue_id, data, src_id, origin,
                                            region, dst_host_ptr,
                                            size, // adjusted_host_ptr,
                                            remote_finish_command, data, node);

      return r;
    }
}

cl_int
pocl_remote_async_map_image (void *data, _cl_command_node *node,
                             pocl_mem_identifier *mem_id, cl_mem src_image,
                             mem_mapping_t *map)
{
  if (map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION)
    {
      // we can skip talking to remote, but we still need to call the callback.
      return 1;
    }

  int r = pocl_remote_async_read_image_rect (
      data, node, src_image, mem_id, map->host_ptr, NULL, map->origin,
      map->region, map->row_pitch, map->slice_pitch, 0);

  return r;
}

cl_int
pocl_remote_async_unmap_image (void *data, _cl_command_node *node,
                               pocl_mem_identifier *mem_id, cl_mem dst_image,
                               mem_mapping_t *map)
{
  /* it could be CL_MAP_READ | CL_MAP_WRITE(..invalidate) which has to be
   * handled like a write */
  if (map->map_flags == CL_MAP_READ)
    {
      // we can skip talking to remote, but we still need to call the callback.
      return 1;
    }

  int r = pocl_remote_async_write_image_rect (
      data, node, dst_image, mem_id, map->host_ptr, NULL, map->origin,
      map->region, map->row_pitch, map->slice_pitch, 0);
  return r;
}

cl_int
pocl_remote_async_fill_image (void *data, _cl_command_node *node,
                              pocl_mem_identifier *image_data,
                              const size_t *origin, const size_t *region,
                              cl_uint4 *fill_pixel)
{
  uint32_t image_id = (uintptr_t)image_data->mem_ptr;

  POCL_MSG_PRINT_REMOTE ("REMOTE FILL IMAGE \n"
                         "image ID %u data %p \n"
                         "origin [0,1,2] %zu %zu %zu \n"
                         "region [0,1,2] %zu %zu %zu \n",
                         image_id, image_data, origin[0], origin[1], origin[2],
                         region[0], region[1], region[2]);

  uint32_t queue_id = (uint32_t)node->sync.event.event->queue->id;

  int r = pocl_network_fill_image (queue_id, data, image_id, origin, region,
                                   fill_pixel, remote_finish_command, data,
                                   node);

  return r;
}

static void
remote_start_command (remote_device_data_t *d, _cl_command_node *node)
{
  _cl_command_t *cmd = &node->command;
  cl_event event = node->sync.event.event;
  cl_command_queue cq = node->sync.event.event->queue;

  if (*(cq->device->available) == CL_FALSE)
    {
      pocl_update_event_device_lost (event);
      goto EARLY_FINISH;
    }
  pocl_update_event_running (event);

  switch (node->type)
    {
    case CL_COMMAND_MIGRATE_MEM_OBJECTS:
      switch (cmd->migrate.type)
        {
        case ENQUEUE_MIGRATE_TYPE_D2H:
          {
            int r;
            cl_mem m = node->migr_infos->buffer;
            if (m->is_image)

              {
                size_t region[3]
                    = { m->image_width, m->image_height, m->image_depth };
                if (region[2] == 0)
                  region[2] = 1;
                if (region[1] == 0)
                  region[1] = 1;
                size_t origin[3] = { 0, 0, 0 };
                r = pocl_remote_async_read_image_rect (
                  d, node, m, &m->device_ptrs[node->device->global_mem_id],
                  m->mem_host_ptr, NULL, origin, region, 0, 0, 0);
              }
            else
              {
                r = pocl_remote_async_read (
                  d, node, m->mem_host_ptr,
                  &m->device_ptrs[node->device->global_mem_id], m, 0, m->size);
              }
            assert (r == 0);
            break;
          }
        case ENQUEUE_MIGRATE_TYPE_H2D:
          {
            int r;
            cl_mem m = node->migr_infos->buffer;
            if (m->is_image)
              {
                size_t region[3]
                    = { m->image_width, m->image_height, m->image_depth };
                if (region[2] == 0)
                  region[2] = 1;
                if (region[1] == 0)
                  region[1] = 1;
                size_t origin[3] = { 0, 0, 0 };
                /* TODO: truncate region if it exceeds content size? */
                r = pocl_remote_async_write_image_rect (
                  d, node, m, &m->device_ptrs[node->device->global_mem_id],
                  m->mem_host_ptr, NULL, origin, region, 0, 0, 0);
              }
            else
              {
                r = pocl_remote_async_write (
                  d, node, m->mem_host_ptr,
                  &m->device_ptrs[node->device->global_mem_id], m, 0,
                  cmd->migrate.migration_size);
              }
            assert (r == 0);
            break;
          }
        case ENQUEUE_MIGRATE_TYPE_D2D:
          {
            cl_device_id dev = cmd->migrate.src_device;
            assert (dev);
            pocl_remote_async_migrate_d2d (
              d, dev->data, node, node->migr_infos->buffer,
              &node->migr_infos->buffer->device_ptrs[dev->global_mem_id]);
            break;
          }
        case ENQUEUE_MIGRATE_TYPE_NOP:
          {
            goto EARLY_FINISH;
          }
        }
      return;

    case CL_COMMAND_READ_BUFFER:
      pocl_remote_async_read (
        d, node, cmd->read.dst_host_ptr,
        &cmd->read.src->device_ptrs[node->device->global_mem_id],
        cmd->read.src, cmd->read.offset, cmd->read.size);
      return;

    case CL_COMMAND_WRITE_BUFFER:
      pocl_remote_async_write (
        d, node, cmd->write.src_host_ptr,
        &cmd->write.dst->device_ptrs[node->device->global_mem_id],
        cmd->write.dst, cmd->write.offset, cmd->write.size);
      return;

    case CL_COMMAND_COPY_BUFFER:
      if (pocl_remote_async_copy (
            d, node, &cmd->copy.dst->device_ptrs[node->device->global_mem_id],
            cmd->copy.dst,
            &cmd->copy.src->device_ptrs[node->device->global_mem_id],
            cmd->copy.src, cmd->copy.dst_offset, cmd->copy.src_offset,
            cmd->copy.size))
        goto EARLY_FINISH;
      return;

    case CL_COMMAND_READ_BUFFER_RECT:
      pocl_remote_async_read_rect (
        d, node, cmd->read_rect.dst_host_ptr,
        &cmd->read_rect.src->device_ptrs[node->device->global_mem_id],
        cmd->read_rect.src, cmd->read_rect.buffer_origin,
        cmd->read_rect.host_origin, cmd->read_rect.region,
        cmd->read_rect.buffer_row_pitch, cmd->read_rect.buffer_slice_pitch,
        cmd->read_rect.host_row_pitch, cmd->read_rect.host_slice_pitch);
      return;

    case CL_COMMAND_WRITE_BUFFER_RECT:
      pocl_remote_async_write_rect (
        d, node, cmd->write_rect.src_host_ptr,
        &cmd->write_rect.dst->device_ptrs[node->device->global_mem_id],
        cmd->write_rect.dst, cmd->write_rect.buffer_origin,
        cmd->write_rect.host_origin, cmd->write_rect.region,
        cmd->write_rect.buffer_row_pitch, cmd->write_rect.buffer_slice_pitch,
        cmd->write_rect.host_row_pitch, cmd->write_rect.host_slice_pitch);
      return;

    case CL_COMMAND_COPY_BUFFER_RECT:
      if (pocl_remote_async_copy_rect (
            d, node,
            &cmd->copy_rect.dst->device_ptrs[node->device->global_mem_id],
            cmd->copy_rect.dst,
            &cmd->copy_rect.src->device_ptrs[node->device->global_mem_id],
            cmd->copy_rect.src, cmd->copy_rect.dst_origin,
            cmd->copy_rect.src_origin, cmd->copy_rect.region,
            cmd->copy_rect.dst_row_pitch, cmd->copy_rect.dst_slice_pitch,
            cmd->copy_rect.src_row_pitch, cmd->copy_rect.src_slice_pitch))
        goto EARLY_FINISH;
      return;

    case CL_COMMAND_FILL_BUFFER:
      pocl_remote_async_memfill (
        d, node, &cmd->memfill.dst->device_ptrs[node->device->global_mem_id],
        cmd->memfill.dst, cmd->memfill.size, cmd->memfill.offset,
        cmd->memfill.pattern, cmd->memfill.pattern_size);
      return;

    case CL_COMMAND_MAP_BUFFER:
      if (pocl_remote_async_map_mem (
            d, node,
            &cmd->map.buffer->device_ptrs[node->device->global_mem_id],
            cmd->map.buffer, cmd->map.mapping))
        goto EARLY_FINISH;
      return;

    case CL_COMMAND_UNMAP_MEM_OBJECT:

      if (cmd->unmap.buffer->is_image == CL_FALSE
          || IS_IMAGE1D_BUFFER (cmd->unmap.buffer))
        {
          if (pocl_remote_async_unmap_mem (
                d, node,
                &cmd->unmap.buffer->device_ptrs[node->device->global_mem_id],
                cmd->unmap.mapping)
              > 0)
            goto EARLY_FINISH;
        }
      else
        {
          if (pocl_remote_async_unmap_image (
                d, node,
                &cmd->unmap.buffer->device_ptrs[node->device->global_mem_id],
                cmd->unmap.buffer, cmd->unmap.mapping)
              > 0)
            goto EARLY_FINISH;
        }
      return;

    case CL_COMMAND_NDRANGE_KERNEL:
      {
        pocl_remote_async_run (d, node);
        return;
      }

    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
      pocl_remote_async_read_image_rect (
        d, node, cmd->read_image.src,
        &cmd->read_image.src->device_ptrs[node->device->global_mem_id], NULL,
        &cmd->read_image.dst->device_ptrs[node->device->global_mem_id],
        cmd->read_image.origin, cmd->read_image.region,
        cmd->read_image.dst_row_pitch, cmd->read_image.dst_slice_pitch,
        cmd->read_image.dst_offset);
      return;

    case CL_COMMAND_READ_IMAGE:
      pocl_remote_async_read_image_rect (
        d, node, cmd->read_image.src,
        &cmd->read_image.src->device_ptrs[node->device->global_mem_id],
        cmd->read_image.dst_host_ptr, NULL, cmd->read_image.origin,
        cmd->read_image.region, cmd->read_image.dst_row_pitch,
        cmd->read_image.dst_slice_pitch, cmd->read_image.dst_offset);
      return;

    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      pocl_remote_async_write_image_rect (
        d, node, cmd->write_image.dst,
        &cmd->write_image.dst->device_ptrs[node->device->global_mem_id], NULL,
        &cmd->write_image.src->device_ptrs[node->device->global_mem_id],
        cmd->write_image.origin, cmd->write_image.region,
        cmd->write_image.src_row_pitch, cmd->write_image.src_slice_pitch,
        cmd->write_image.src_offset);
      return;

    case CL_COMMAND_WRITE_IMAGE:
      pocl_remote_async_write_image_rect (
        d, node, cmd->write_image.dst,
        &cmd->write_image.dst->device_ptrs[node->device->global_mem_id],
        cmd->write_image.src_host_ptr, NULL, cmd->write_image.origin,
        cmd->write_image.region, cmd->write_image.src_row_pitch,
        cmd->write_image.src_slice_pitch, cmd->write_image.src_offset);
      return;

    case CL_COMMAND_COPY_IMAGE:
      pocl_remote_async_copy_image_rect (
        d, node, cmd->copy_image.src, cmd->copy_image.dst,
        &cmd->copy_image.src->device_ptrs[node->device->global_mem_id],
        &cmd->copy_image.dst->device_ptrs[node->device->global_mem_id],
        cmd->copy_image.src_origin, cmd->copy_image.dst_origin,
        cmd->copy_image.region);
      return;

    case CL_COMMAND_FILL_IMAGE:
      pocl_remote_async_fill_image (
        d, node,
        &cmd->fill_image.dst->device_ptrs[node->device->global_mem_id],
        cmd->fill_image.origin, cmd->fill_image.region,
        &cmd->fill_image.orig_pixel);
      return;

    case CL_COMMAND_MAP_IMAGE:
      if (pocl_remote_async_map_image (
            d, node,
            &cmd->map.buffer->device_ptrs[node->device->global_mem_id],
            cmd->map.buffer, cmd->map.mapping))
        goto EARLY_FINISH;
      return;

    case CL_COMMAND_SVM_UNMAP:
      if (pocl_remote_async_unmap_svm_buffer (d, node) > 0)
        goto EARLY_FINISH;
      return;

    case CL_COMMAND_SVM_MAP:
      if (pocl_remote_async_map_svm_buffer (d, node) > 0)
        goto EARLY_FINISH;
      return;

    case CL_COMMAND_MARKER:
    case CL_COMMAND_BARRIER:
    case CL_COMMAND_COMMAND_BUFFER_KHR:
      goto EARLY_FINISH;

    default:
      POCL_ABORT_UNIMPLEMENTED ("Unimplemented remote command.\n");
    }

EARLY_FINISH:
  POCL_LOCK (d->wq_lock);
  DL_APPEND (d->finished_list, node);
  POCL_UNLOCK (d->wq_lock);
}

static void *
pocl_remote_driver_pthread (void *cldev)
{
  cl_device_id device = (cl_device_id)cldev;
  remote_device_data_t *d = (remote_device_data_t *)device->data;
  _cl_command_node *cmd = NULL;
  _cl_command_node *finished = NULL;

  /* Sleep so we have time to run the task graph dumper in main(). */
  /* sleep (2); */
  POCL_LOCK (d->wq_lock);

  while (1)
    {
      if (d->driver_thread_exit_requested)
        {
          POCL_UNLOCK (d->wq_lock);
          return NULL;
        }

      cmd = d->work_queue;
      if (cmd)
        {
          DL_DELETE (d->work_queue, cmd);
          POCL_UNLOCK (d->wq_lock);

          assert (cmd->sync.event.event->status == CL_SUBMITTED);

          remote_start_command (d, cmd);

          POCL_LOCK (d->wq_lock);
        }

      finished = d->finished_list;
      if (finished)
        {
          DL_DELETE (d->finished_list, finished);
          POCL_UNLOCK (d->wq_lock);

          cl_event event = finished->sync.event.event;

          const char *cstr = pocl_command_to_str (finished->type);
          char msg[128] = "Event ";
          strcat (msg, cstr);

          POCL_UPDATE_EVENT_COMPLETE_MSG (event, msg);

          POCL_LOCK (d->wq_lock);
        }

      if ((d->work_queue == NULL) && (d->finished_list == NULL)
          && (d->driver_thread_exit_requested == 0))
        {
          POCL_WAIT_COND (d->wakeup_cond, d->wq_lock);
          // since cond_wait returns with locked mutex, might as well retry
        }
    }
}

cl_int
pocl_remote_set_kernel_exec_info_ext (cl_device_id dev,
                                      unsigned program_device_i,
                                      cl_kernel kernel, cl_uint param_name,
                                      size_t param_value_size,
                                      const void *param_value)
{

  switch (param_name)
    {
    case CL_KERNEL_EXEC_INFO_SVM_PTRS:
    case CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL:
      {
        for (int i = 0; i < param_value_size / sizeof (void *); ++i)
          {
            struct _pocl_ptr_list_node *n
                = malloc (sizeof (struct _pocl_ptr_list_node));
            n->ptr = ((void **)param_value)[i];
            DL_APPEND (kernel->indirect_raw_ptrs, n);
            POCL_MSG_PRINT_MEMORY ("Set a indirect SVM/USM ptr %p\n", n->ptr);
          }
        return CL_SUCCESS;
      }
    case CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL:
    case CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL:
    case CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL:
      {
        kernel->can_access_all_raw_buffers_indirectly = 1;
        return CL_SUCCESS;
      }
    default:
      return CL_INVALID_VALUE;
    }
}

cl_int
pocl_remote_get_device_info_ext (cl_device_id device,
                                 cl_device_info param_name,
                                 size_t param_value_size,
                                 void *param_value,
                                 size_t *param_value_size_ret)
{

  switch (param_name)
    {
    case CL_DEVICE_REMOTE_TRAFFIC_STATS_POCL:
      {
        size_t traffic_data_size = 6 * sizeof (int64_t);
        POCL_RETURN_GETINFO_INNER (
          traffic_data_size,
          pocl_remote_get_traffic_stats (param_value, device));
      }
    }

  return pocl_network_fetch_devinfo (device, param_name, param_value_size,
                                     param_value, param_value_size_ret);
}
