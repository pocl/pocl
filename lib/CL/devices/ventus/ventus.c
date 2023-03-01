/* ventus.c - a pocl device driver for ventus gpgpu 

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos and
                 2011-2021 Pekka Jääskeläinen

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

#include "ventus.h"
#include "common.h"
#include "config.h"
#include "config2.h"
#include "cpuinfo.h"
#include "devices.h"
#include "pocl_local_size.h"
#include "pocl_util.h"
#include "topology/pocl_topology.h"
#include "utlist.h"

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <utlist.h>

#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_mem_management.h"
#include "pocl_timing.h"
#include "pocl_workgroup_func.h"

#include "common_driver.h"

#ifdef ENABLE_LLVM
#include "pocl_llvm.h"
#endif

  // from driver/include/ventus.h
#if !defined(ENABLE_LLVM)
#include <ventus.h>
#endif

  /* ENABLE_LLVM means to compile the kernel using pocl compiler,
 but for ventus(ventus has its own LLVM) it should be OFF. */

struct vt_device_data_t {
#if !defined(ENABLE_LLVM)
  vt_device_h vt_device;
  size_t vt_print_buf_d;
  vt_buffer_h vt_print_buf_h;
  uint32_t printf_buffer;
  uint32_t printf_buffer_position;
#endif

struct vt_buffer_data_t {
#if !defined(ENABLE_LLVM)
  vt_device_h vt_device;
  vt_buffer_h staging_buf;
#endif
  size_t dev_mem_addr;
};


  /* Maximum kernels occupancy */
  #define MAX_KERNELS 16

  // default WG size in each dimension & total WG size.
  #define DEFAULT_WG_SIZE 4096

  // location is local memory where to store kernel parameters
  #define KERNEL_ARG_BASE_ADDR 0x7fff0000

  // allocate 1MB OpenCL print buffer
  #define PRINT_BUFFER_SIZE (1024 * 1024)


  /* List of commands ready to be executed */
  _cl_command_node *ready_list;
  /* List of commands not yet ready to be executed */
  _cl_command_node *command_list;
  /* Lock for command list related operations */
  pocl_lock_t cq_lock;

  /* Currently loaded kernel. */
  cl_kernel current_kernel;

};


struct kernel_context_t {
  uint32_t num_groups[3];
  uint32_t global_offset[3];
  uint32_t local_size[3];
  uint32_t printf_buffer;
  uint32_t printf_buffer_position;
  uint32_t printf_buffer_capacity;
  uint32_t work_dim;
};

static size_t ALIGNED_CTX_SIZE = 4 * ((sizeof(struct kernel_context_t) + 3) / 4);

// FIXME: Do not use hardcoded library search path!
static const char *ventus_final_ld_flags[] = {
  "-nodefaultlibs",
  CLANG_RESOURCE_DIR"/../../crt0.o",
  "-L"CLANG_RESOURCE_DIR"/../../",
  "-lworkitem",
  NULL
};

void
pocl_ventus_init_device_ops(struct pocl_device_ops *ops)
{
  ops->device_name = "ventus";
  ops->probe = pocl_ventus_probe;

  ops->uninit = pocl_ventus_uninit;
  ops->reinit = NULL;
  ops->init = pocl_ventus_init;

  ops->alloc_mem_obj = pocl_ventus_alloc_mem_obj;
  ops->free = pocl_ventus_free;

  ops->read = pocl_ventus_read;
  ops->read_rect = NULL;
  ops->write = pocl_ventus_write;
  ops->write_rect = NULL;

  ops->run = pocl_ventus_run; 
  ops->run_native = NULL;
  /***********************No need to modify for now*************************/

  ops->copy = NULL;
  ops->copy_with_size = NULL;
  ops->copy_rect = NULL;

  ops->memfill = NULL;
  ops->map_mem = NULL;
  ops->unmap_mem = NULL;
  ops->get_mapping_ptr = NULL;
  ops->free_mapping_ptr = NULL;

  /* for ventus,pocl does not need to compile the kernel,so they are set to NULL */
  ops->build_source = NULL;
  ops->link_program = NULL;
  ops->build_binary = NULL;
  ops->free_program = NULL;
  ops->setup_metadata = NULL;
  ops->supports_binary = NULL;
  ops->build_poclbinary = NULL;
  ops->compile_kernel = NULL;

  ops->join = pocl_ventus_join;
  ops->submit = pocl_ventus_submit;
  ops->broadcast = pocl_broadcast;
  ops->notify = pocl_ventus_notify;
  ops->flush = pocl_ventus_flush;

  ops->build_hash = pocl_ventus_build_hash; 
  ops->compute_local_size = NULL;

  ops->get_device_info_ext = NULL;

  // Currently ventus does not support svm
  ops->svm_free = NULL;
  ops->svm_alloc = NULL;
  /* no need to implement these two as they're noop
   * and pocl_exec_command takes care of it */
  ops->svm_map = NULL;
  ops->svm_unmap = NULL;
  ops->svm_copy = NULL;
  ops->svm_fill = NULL;

  ops->create_kernel = NULL;
  ops->free_kernel = NULL;
  ops->create_sampler = NULL;
  ops->free_sampler = NULL;

  ops->can_migrate_d2d = NULL;
  ops->migrate_d2d = NULL;

  ops->copy_image_rect = NULL;
  ops->write_image_rect = NULL;
  ops->read_image_rect = NULL;
  ops->map_image = NULL;
  ops->unmap_image = NULL;
  ops->fill_image = NULL;
}

char *
pocl_ventus_build_hash (cl_device_id device)
{
  char* res = (char *)calloc(1000, sizeof(char));
  snprintf(res, 1000, "THU-%s", device->llvm_cpu);
  return res;
}

unsigned int
pocl_ventus_probe(struct pocl_device_ops *ops)
{
  if (0 == strcmp(ops->device_name, "ventus"))
    return 1;
  return 0;
}

cl_int
pocl_ventus_init (unsigned j, cl_device_id dev, const char* parameters)
{
  struct vt_device_data_t *d;
  cl_int ret = CL_SUCCESS;
  int err;

  d = (struct vt_device_data_t *) calloc (1, sizeof (struct vt_device_data_t));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;


#if !defined(ENABLE_LLVM)
  vt_device_h vt_device;

  err = vt_dev_open(&vt_device);
  if (err != 0) {
    free(d);
    return CL_DEVICE_NOT_FOUND;
  }
  
  device->device_side_printf = 1;
  device->printf_buffer_size = PRINT_BUFFER_SIZE;

  // add storage for position pointer
  uint32_t print_buf_dev_size = PRINT_BUFFER_SIZE + sizeof(uint32_t);

  size_t vt_print_buf_d;
  err = vt_mem_alloc(vt_device, print_buf_dev_size, &vt_print_buf_d);
  if (err != 0) {
    vt_dev_close(vt_device);
    free(d);
    return CL_INVALID_DEVICE;
  }  
  
  vt_buffer_h vt_print_buf_h;
  err = vt_buf_alloc(vt_device, print_buf_dev_size, &vt_print_buf_h);
  if (err != 0) {
    vt_dev_close(vt_device);
    free(d);
    return CL_OUT_OF_HOST_MEMORY;
  }  
    
  // clear print position to zero
  uint8_t* staging_ptr = (uint8_t*)vt_host_ptr(vt_print_buf_h);
  memset(staging_ptr + PRINT_BUFFER_SIZE, 0, sizeof(uint32_t));
  err = vt_copy_to_dev(vt_print_buf_h, vt_print_buf_d + PRINT_BUFFER_SIZE, sizeof(uint32_t), PRINT_BUFFER_SIZE);
  if (err != 0) {
    vt_buf_free(vt_print_buf_h);
    vt_dev_close(vt_device);
    free(d);
    return CL_OUT_OF_HOST_MEMORY;
  }
    
  d->vt_device      = vt_device;
  d->vt_print_buf_d = vt_print_buf_d;
  d->vt_print_buf_h = vt_print_buf_h; 
  d->printf_buffer  = vt_print_buf_d;
  d->printf_buffer_position = vt_print_buf_d + PRINT_BUFFER_SIZE;
#endif 


  d->current_kernel = NULL;

  dev->data = d;

  SETUP_DEVICE_CL_VERSION(2, 0);
  dev->type = CL_DEVICE_TYPE_GPU;
  dev->long_name = "Ventus GPGPU device";
  dev->vendor = "THU";
  dev->vendor_id = 0x1234; // TODO: Update vendor id!
  dev->version = "2.0";
  dev->available = CL_TRUE;
  dev->compiler_available = CL_TRUE;
  dev->linker_available = CL_TRUE;
  dev->extensions = "";
  dev->profile = "FULL_PROFILE";
  dev->endian_little = CL_TRUE;

  dev->max_mem_alloc_size = 100 * 1024 * 1024;
  dev->mem_base_addr_align = 4;

  dev->max_constant_buffer_size = 32768;     // TODO: Update this to conformant to OCL 2.0
  dev->local_mem_size = 131072;     // TODO: Update this to conformant to OCL 2.0
  dev->global_mem_size = 1024 * 1024 * 1024; // 1G ram
  dev->global_mem_cache_type = CL_READ_WRITE_CACHE;
  dev->global_mem_cacheline_size = 64; // FIXME: Is this accurate?
  dev->global_mem_cache_size = 32768;  // FIXME: Is this accurate?
  dev->image_max_buffer_size = dev->max_mem_alloc_size / 16;

  dev->image2d_max_width = 1024; // TODO: Update
  dev->image3d_max_width = 1024; // TODO: Update

  dev->max_work_item_dimensions = 3;
  dev->max_work_group_size = 1024;
  dev->max_work_item_sizes[0] = 1024;
  dev->max_work_item_sizes[1] = 1024;
  dev->max_work_item_sizes[2] = 1024;
  dev->max_parameter_size = 64;
  dev->max_compute_units = 1;
  dev->max_clock_frequency = 600; // TODO: This is frequency in MHz
  dev->address_bits = 32;

  // Supports device side printf
  dev->device_side_printf = 1;
  dev->printf_buffer_size = PRINTF_BUFFER_SIZE;

  // Doesn't support partition
  dev->max_sub_devices = 1;
  dev->num_partition_properties = 1;
  dev->num_partition_types = 0;
  dev->partition_type = NULL;

  // Doesn't support SVM
  dev->svm_allocation_priority = 0;

  dev->final_linkage_flags = ventus_final_ld_flags;

  // TODO: Do we have builtin kernels for Ventus?

#ifdef ENABLE_LLVM
  dev->llvm_target_triplet = OCL_KERNEL_TARGET;
  dev->llvm_cpu = OCL_KERNEL_TARGET_CPU;
#else
  dev->llvm_target_triplet = "";
  dev->llvm_cpu = "";
#endif


#if (!defined(ENABLE_CONFORMANCE))
  /* full memory consistency model for atomic memory and fence operations
  https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#opencl-3.0-backwards-compatibility */
  dev->atomic_memory_capabilities = CL_DEVICE_ATOMIC_ORDER_RELAXED
                                       | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
                                       | CL_DEVICE_ATOMIC_ORDER_SEQ_CST
                                       | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP
                                       | CL_DEVICE_ATOMIC_SCOPE_DEVICE
                                       | CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES;
  dev->atomic_fence_capabilities = CL_DEVICE_ATOMIC_ORDER_RELAXED
                                       | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
                                       | CL_DEVICE_ATOMIC_ORDER_SEQ_CST
                                       | CL_DEVICE_ATOMIC_SCOPE_WORK_ITEM
                                       | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP
                                       | CL_DEVICE_ATOMIC_SCOPE_DEVICE;
#endif

  POCL_INIT_LOCK (d->cq_lock);

  assert (dev->printf_buffer_size > 0);
  d->printf_buffer = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                          dev->printf_buffer_size);
  assert (d->printf_buffer != NULL);

  return ret;
}

void
pocl_ventus_run (void *data, _cl_command_node *cmd)
{
  struct vt_device_data_t *d;
  size_t x, y, z;
  unsigned i;
  unsigned dev_i = cmd->device_i;
  cl_kernel kernel = cmd->command.run.kernel;
  cl_program program = kernel->program;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &cmd->command.run.pc;
  int err;
  
  assert(data != NULL);
  d = (struct vt_device_data_t *)data;

  // calculate kernel arguments buffer size
  size_t abuf_size = 0;  
  size_t abuf_args_size = 4 * (meta->num_args + meta->num_locals);
  size_t abuf_ext_size = 0;
  {
    // pocl_context data
    abuf_size += ALIGNED_CTX_SIZE; 
    // argument data    
    abuf_size += abuf_args_size;
    for (i = 0; i < meta->num_args; ++i) {  
      auto al = &(cmd->command.run.arguments[i]);  
      if (ARG_IS_LOCAL(meta->arg_info[i])
       && !cmd->device->device_alloca_locals) {
        abuf_size += 4;
        abuf_size += al->size;
        abuf_ext_size += al->size;
      } else
      if ((meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
       || (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
       || (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)) {
        abuf_size += 4;
      } else {
        abuf_size += al->size;
      }
    }
  }
  assert(abuf_size <= 0xffff);

  // allocate kernel arguments buffer
  vt_buffer_h staging_buf;
  err = vt_buf_alloc(d->vt_device, abuf_size, &staging_buf);
  assert(0 == err);

  // update kernel arguments buffer
  {
    auto abuf_ptr = (uint8_t*)vt_host_ptr(staging_buf);
    assert(abuf_ptr);

    // write context data
    {
      kernel_context_t ctx;
      for (int i = 0; i < 3; ++i) {
        ctx.num_groups[i] = pc->num_groups[i];
        ctx.global_offset[i] = pc->global_offset[i];
        ctx.local_size[i] = pc->local_size[i];        
      }
      ctx.work_dim = pc->work_dim;      
      ctx.printf_buffer = d->printf_buffer;
      ctx.printf_buffer_position = d->printf_buffer_position;
      ctx.printf_buffer_capacity = PRINT_BUFFER_SIZE;

      memset(abuf_ptr, 0, ALIGNED_CTX_SIZE);
      memcpy(abuf_ptr, &ctx, sizeof(kernel_context_t));
      print_data("*** ctx=", abuf_ptr, ALIGNED_CTX_SIZE);
    }

    // write arguments    
    uint32_t args_base_addr = KERNEL_ARG_BASE_ADDR;
    uint32_t args_addr = args_base_addr + ALIGNED_CTX_SIZE + abuf_args_size;
    uint32_t args_ext_addr = (args_base_addr + abuf_size) - abuf_ext_size;
    for (i = 0; i < meta->num_args; ++i) {
      uint32_t addr = ALIGNED_CTX_SIZE + i * 4;
      auto al = &(cmd->command.run.arguments[i]);
      if (ARG_IS_LOCAL(meta->arg_info[i])) {
        if (cmd->device->device_alloca_locals) {
          memcpy(abuf_ptr + addr, &al->size, 4);
          print_data("*** locals=", abuf_ptr + addr, 4);
        } else {
          memcpy(abuf_ptr + addr, &args_addr, 4);          
          memcpy(abuf_ptr + (args_addr - args_base_addr), &args_ext_addr, 4);
          args_addr += 4;
          args_ext_addr += al->size;
          std::abort();
        }
      } else
      if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
        memcpy(abuf_ptr + addr, &args_addr, 4);        
        if (al->value == NULL) {
          memset(abuf_ptr + (args_addr - args_base_addr), 0, 4);
          print_data("*** null=", abuf_ptr + (args_addr - args_base_addr), 4); 
        } else {
          cl_mem m = (*(cl_mem *)(al->value));
          auto buf_data = (vt_buffer_data_t*)m->device_ptrs[cmd->device->dev_id].mem_ptr;
          auto dev_mem_addr = buf_data->dev_mem_addr + al->offset;
          memcpy(abuf_ptr + (args_addr - args_base_addr), &dev_mem_addr, 4);
          print_data("*** ptr=", abuf_ptr + (args_addr - args_base_addr), 4);
        }
        args_addr += 4;
      } else 
      if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) {
        std::abort();
      } else 
      if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
        std::abort();
      } else {
        memcpy(abuf_ptr + addr, &args_addr, 4);
        memcpy(abuf_ptr + (args_addr - args_base_addr), al->value, al->size);
        print_data("*** arg-addr=", abuf_ptr + addr, 4);
        print_data("*** arg-value=", abuf_ptr + (args_addr - args_base_addr), al->size);
        args_addr += al->size;
      }
    }

    // upload kernel arguments buffer
    err = vt_copy_to_dev(staging_buf, args_base_addr, abuf_size, 0);
    assert(0 == err);

    // release staging buffer
    err = vt_buf_free(staging_buf);
    assert(0 == err);
    
    // upload kernel to device
    if (NULL == d->current_kernel 
     || d->current_kernel != kernel) {    
       d->current_kernel = kernel;
      char program_bin_path[POCL_FILENAME_LENGTH];
      pocl_cache_final_binary_path (program_bin_path, program, dev_i, kernel, NULL, 0);
      err = vt_upload_kernel_file(d->vt_device, program_bin_path);      
      assert(0 == err);
    }
  }
    
  // quick off kernel execution
  err = vt_start(d->vt_device);
  assert(0 == err);

  // wait for the execution to complete
  err = vt_ready_wait(d->vt_device, -1);
  assert(0 == err);

  // flush print buffer 
  {
    auto print_ptr = (uint8_t*)vt_host_ptr(d->vt_print_buf_h);
    err = vt_copy_from_dev(d->vt_print_buf_h, d->vt_print_buf_d + PRINT_BUFFER_SIZE, sizeof(uint32_t), PRINT_BUFFER_SIZE);
    assert(0 == err);
    uint32_t print_size = *(uint32_t*)(print_ptr + PRINT_BUFFER_SIZE);
    if (print_size != 0) {
      err = vt_copy_from_dev(d->vt_print_buf_h, d->vt_print_buf_d, print_size, 0);
      assert(0 == err);      
      
      write (STDOUT_FILENO, print_ptr, print_size);
      
      memset(print_ptr + PRINT_BUFFER_SIZE, 0, sizeof(uint32_t));
      err = vt_copy_to_dev(d->vt_print_buf_h, d->vt_print_buf_d, sizeof(uint32_t), PRINT_BUFFER_SIZE);
      assert(0 == err);
    }
  }

  pocl_release_dlhandle_cache(cmd);
}


cl_int
pocl_ventus_uninit (unsigned j, cl_device_id device)
{
  struct vt_device_data_t *d = (struct vt_device_data_t*)device->data;
  if (NULL == d)
  return CL_SUCCESS;  

  #if !defined(OCS_AVAILABLE)
    vt_buf_free(d->vt_print_buf_h);
    vt_dev_close(d->vt_device);
  #endif

  POCL_DESTROY_LOCK(d->cq_lock);
  POCL_MEM_FREE(d);
  device->data = NULL;
  
  return CL_SUCCESS;
}


static void ventus_command_scheduler (struct vt_device_data_t *d)
{
  _cl_command_node *node;

  /* execute commands from ready list */
  while ((node = d->ready_list))
    {
      assert (pocl_command_is_ready(node->sync.event.event));
      assert (node->sync.event.event->status == CL_SUBMITTED);
      CDL_DELETE (d->ready_list, node);
      POCL_UNLOCK (d->cq_lock);
      pocl_exec_command (node);
      POCL_LOCK (d->cq_lock);
    }

  return;
}

void
pocl_ventus_submit (_cl_command_node *node, cl_command_queue cq)
{
  struct vt_device_data_t *d = (struct vt_device_data_t *)node->device->data;

  if (node != NULL && node->type == CL_COMMAND_NDRANGE_KERNEL)
    pocl_check_kernel_dlhandle_cache (node, 1, 1);

  node->ready = 1;
  POCL_LOCK (d->cq_lock);
  pocl_command_push(node, &d->ready_list, &d->command_list);

  POCL_UNLOCK_OBJ (node->sync.event.event);
  ventus_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);

  return;
}

void pocl_ventus_flush (cl_device_id device, cl_command_queue cq)
{
  struct vt_device_data_t *d = (struct vt_device_data_t *)device->data;

  POCL_LOCK (d->cq_lock);
  ventus_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);
}

void
pocl_ventus_join (cl_device_id device, cl_command_queue cq)
{
  struct vt_device_data_t *d = (struct vt_device_data_t *)device->data;

  POCL_LOCK (d->cq_lock);
  ventus_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);

  return;
}

void
pocl_ventus_notify (cl_device_id device, cl_event event, cl_event finished)
{
  struct vt_device_data_t *d = (struct vt_device_data_t *)device->data;
  _cl_command_node * volatile node = event->command;

  if (finished->status < CL_COMPLETE)
    {
      pocl_update_event_failed (event);
      return;
    }

  if (!node->ready)
    return;

  if (pocl_command_is_ready (event))
    {
      if (event->status == CL_QUEUED)
        {
          pocl_update_event_submitted (event);
          POCL_LOCK (d->cq_lock);
          CDL_DELETE (d->command_list, node);
          CDL_PREPEND (d->ready_list, node);
          ventus_command_scheduler (d);
          POCL_UNLOCK (d->cq_lock);
        }
      return;
    }
}

void
pocl_ventus_compile_kernel (_cl_command_node *cmd, cl_kernel kernel,
                           cl_device_id device, int specialize)
{
  if (cmd != NULL && cmd->type == CL_COMMAND_NDRANGE_KERNEL)
    pocl_check_kernel_dlhandle_cache (cmd, 0, specialize);
}

void pocl_ventus_free(cl_device_id device, cl_mem memobj) {
  cl_mem_flags flags = memobj->flags;
  auto buf_data = (vt_buffer_data_t*)memobj->device_ptrs[device->dev_id].mem_ptr;

  /* The host program can provide the runtime with a pointer 
  to a block of continuous memory to hold the memory object 
  when the object is created (CL_MEM_USE_HOST_PTR). 
  Alternatively, the physical memory can be managed 
  by the OpenCL runtime and not be directly accessible 
  to the host program.*/
  if (flags & CL_MEM_USE_HOST_PTR 
   || memobj->shared_mem_allocation_owner != device) {
    std::abort(); //TODO
  } else {
    vt_buf_free(buf_data->staging_buf);
    vt_mem_free(buf_data->vt_device, buf_data->dev_mem_addr);
  }
  if (memobj->flags | CL_MEM_ALLOC_HOST_PTR)
    memobj->mem_host_ptr = NULL;
}


static void *
pocl_ventus_malloc(cl_device_id device, cl_mem_flags flags, size_t size,
                   void *host_ptr) {
  auto d = (vt_device_data_t *)device->data;
  void *b = NULL;
  pocl_global_mem_t *mem = device->global_memory;
  int err;

  if (flags & CL_MEM_USE_HOST_PTR) {
    std::abort(); //TODO
  }

  vt_buffer_h staging_buf;
  err = vt_buf_alloc(d->vt_device, size, &staging_buf);
  if (err != 0)
    return nullptr;

  size_t dev_mem_addr;
  err = vt_mem_alloc(d->vt_device, size, &dev_mem_addr);
  if (err != 0) {
    vt_buf_free(staging_buf);
    return nullptr;
  }

  if (flags & CL_MEM_COPY_HOST_PTR) {
    auto buf_ptr = vt_host_ptr(staging_buf);
    memcpy((void*)buf_ptr, host_ptr, size);
    err = vt_copy_to_dev(staging_buf, dev_mem_addr, size, 0);
    if (err != 0) {
      vt_buf_free(staging_buf);
      return nullptr;
    }
  }

  auto buf_data = new vt_buffer_data_t();
  buf_data->vt_device    = d->vt_device;
  buf_data->staging_buf  = staging_buf;
  buf_data->dev_mem_addr = dev_mem_addr;

  return buf_data;
}


cl_int
pocl_ventus_alloc_mem_obj(cl_device_id device, cl_mem mem_obj, void *host_ptr) {
  void *b = NULL;
  cl_mem_flags flags = mem_obj->flags;
  unsigned i;

  /* Check if some driver has already allocated memory for this mem_obj
     in our global address space, and use that. */
  for (i = 0; i < mem_obj->context->num_devices; ++i) {
    if (!mem_obj->device_ptrs[i].available)
      continue;
    if (mem_obj->device_ptrs[i].global_mem_id == device->global_mem_id && mem_obj->device_ptrs[i].mem_ptr != NULL) {
      mem_obj->device_ptrs[device->dev_id].mem_ptr = mem_obj->device_ptrs[i].mem_ptr;
    
      POCL_MSG_PRINT_INFO("VENTUS: alloc_mem_obj, use already allocated memory\n");
      std::abort(); // TODO
      return CL_SUCCESS;
    }
  }

  /* Memory for this global memory is not yet allocated -> we'll allocate it. */
  b = pocl_ventus_malloc(device, flags, mem_obj->size, host_ptr);
  if (b == NULL)
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;

  /* Take ownership if not USE_HOST_PTR. */
  if (~flags & CL_MEM_USE_HOST_PTR)
    mem_obj->shared_mem_allocation_owner = device;

  mem_obj->device_ptrs[device->dev_id].mem_ptr = b;

  if (flags & CL_MEM_ALLOC_HOST_PTR) {
    std::abort(); // TODO
  }

  return CL_SUCCESS;
}

void pocl_ventus_read(void *data,
                      void *__restrict__ host_ptr,
                      pocl_mem_identifier *src_mem_id,
                      cl_mem src_buf,
                      size_t offset, 
                      size_t size) {
  int vt_err;
  struct vt_device_data_t *d = (struct vt_device_data_t *)data;                      
  auto buf_data = (vt_buffer_data_t*)src_mem_id->mem_ptr;
  vt_err = vt_copy_from_dev(buf_data->staging_buf, buf_data->dev_mem_addr, offset + size, 0);
  assert(0 == vt_err);
  auto buf_ptr = vt_host_ptr(buf_data->staging_buf);
  assert(buf_ptr);
  memcpy(host_ptr, (char *)buf_ptr + offset, size);
}

void pocl_ventus_write(void *data,
                       const void *__restrict__ host_ptr,
                       pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf,
                       size_t offset, 
                       size_t size) {
  auto buf_data = (vt_buffer_data_t*)dst_mem_id->mem_ptr;
  auto buf_ptr = vt_host_ptr(buf_data->staging_buf);
  memcpy((char *)buf_ptr + offset, host_ptr, size);
  auto vt_err = vt_copy_to_dev(buf_data->staging_buf, buf_data->dev_mem_addr, offset + size, 0);
  assert(0 == vt_err);
}
