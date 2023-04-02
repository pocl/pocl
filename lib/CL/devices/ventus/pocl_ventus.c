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
#include <ctype.h>
#include <limits.h>
#include <stdio.h>
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
//#if !defined(ENABLE_LLVM)
#include "ventus.h"
#include "pocl_ventus.h"
//#endif

  /* ENABLE_LLVM means to compile the kernel using pocl compiler,
 but for ventus(ventus has its own LLVM) it should be OFF. */

struct vt_device_data_t {
//#if !defined(ENABLE_LLVM)
  vt_device_h vt_device;
//#endif


  #define MAX_KERNELS 16

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
  
  /* printf buffer */
  void *printf_buffer;
};

#define KNL_ENTRY 0
#define KNL_ARG_BASE 4
#define KNL_WORK_DIM 8
#define KNL_GL_SIZE_X 12
#define KNL_GL_SIZE_Y 16
#define KNL_GL_SIZE_Z 20
#define KNL_LC_SIZE_X 24
#define KNL_LC_SIZE_Y 28
#define KNL_LC_SIZE_Z 32
#define KNL_GL_OFFSET_X 36
#define KNL_GL_OFFSET_Y 40
#define KNL_GL_OFFSET_Z 44
#define KNL_PRINT_ADDR 48
#define KNL_PRINT_SIZE 52
#define KNL_MAX_METADATA_SIZE 64


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
  ops->compile_kernel = NULL;  //or use int (*build_builtin) (cl_program program, cl_uint device_i);

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
  snprintf(res, 1000, "THU-ventus-GPGPU");
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

  vt_device_h vt_device;

  err = vt_dev_open(&vt_device);
  if (err != 0) {
    free(d);
    return CL_DEVICE_NOT_FOUND;
  }
  
  /*
  // add storage for position pointer
  uint32_t print_buf_dev_size = PRINT_BUFFER_SIZE + sizeof(uint32_t);
  size_t vt_print_buf_d;
  err = vt_buf_alloc(vt_device, print_buf_dev_size, &vt_print_buf_d);
  if (err != 0) {
    vt_dev_close(vt_device);
    free(d);
    return CL_INVALID_DEVICE;
  }  */
    
  d->vt_device   = vt_device;

  d->current_kernel = NULL;

  dev->data = d;

  SETUP_DEVICE_CL_VERSION(2, 0);
  dev->type = CL_DEVICE_TYPE_GPU;
  dev->long_name = "Ventus GPGPU device";
  dev->vendor = "THU";
  dev->vendor_id = 0; // TODO: Update vendor id!
  dev->version = "2.0";
  dev->available = CL_TRUE;
  dev->compiler_available = CL_TRUE;
  dev->linker_available = CL_TRUE;
  dev->extensions = ""; // no extention support now
  dev->profile = "FULL_PROFILE";
  dev->endian_little = CL_TRUE;

  dev->max_mem_alloc_size = 100 * 1024 * 1024; //100M
  dev->mem_base_addr_align = 4;

  dev->max_constant_buffer_size = 32768;     // TODO: Update this to conformant to OCL 2.0 // no cpmstant buffer now
  dev->local_mem_size = 64 * 1024;     // TODO: Update this to conformant to OCL 2.0 // 64kB per SM
  dev->global_mem_size = 1024 * 1024 * 1024; // 1G ram
  dev->global_mem_cache_type = CL_READ_WRITE_CACHE;
  dev->global_mem_cacheline_size = 128; // 128 Bytes for 32 thread
  dev->global_mem_cache_size = 64 * 128; 
  dev->image_max_buffer_size = dev->max_mem_alloc_size / 16;

  dev->image2d_max_width = 1024; // TODO: Update
  dev->image3d_max_width = 1024; // TODO: Update

  dev->max_work_item_dimensions = 3;
  dev->max_work_group_size = 1024;
  dev->max_work_item_sizes[0] = 1024;
  dev->max_work_item_sizes[1] = 1024;
  dev->max_work_item_sizes[2] = 1024;
  dev->max_parameter_size = 64;
  dev->max_compute_units = 16 * 32; // 16 SM comprise 16 int32 and 16 fp32 cores
  dev->max_clock_frequency = 100; // TODO: This is frequency in MHz
  dev->address_bits = 32;

  // Supports device side printf
  dev->device_side_printf = 0;
  dev->printf_buffer_size = 0;

  dev->device_alloca_locals = 1; //

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


  return ret;
}

void
pocl_ventus_run (void *data, _cl_command_node *cmd)
{
  struct vt_device_data_t *d;
  size_t x, y, z;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  cl_program program = kernel->program;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &cmd->command.run.pc;
  int err;

    uint64_t num_thread=8;
    uint64_t num_warp=(pc->local_size[0]*pc->local_size[1]*pc->local_size[2] + num_thread-1)/ num_thread;
    uint64_t num_workgroups[3];
    num_workgroups[0]=pc->num_groups[0];num_workgroups[1]=pc->num_groups[1];num_workgroups[2]=pc->num_groups[2];
    uint64_t num_workgroup=num_workgroups[0]*num_workgroups[1]*num_workgroups[2];
    uint64_t num_processor=num_warp*num_workgroup;
    uint64_t ldssize=0x1000;
    uint64_t pdssize=0x1000;
    uint64_t pdsbase=0x8a000000;
    uint64_t start_pc=0x80000000;
    uint64_t knlbase=0x90000000;
    uint64_t sgpr_usage=32;
    uint64_t vgpr_usage=32;
    

/*
step1 upload kernel_rom & allocate its mem (load cache file?)
step2 allocate kernel argument & arg buffer
      notice kernel arg buffer is offered by command.run.
step3 prepare kernel metadata (pc(start_pc=8000) & kernel entrance(0x8000005c) & arg pointer )
step4 prepare driver metadata
step5 make a writefile for chisel      
*/
  assert(cmd->device->data != NULL);
  d = (struct vt_device_data_t *)cmd->device->data;

  void **arguments = (void **)malloc (sizeof (void *)
                                      * (meta->num_args + meta->num_locals));


//TODO 1: support local buffer as argument. Notice in current structure, allocated localmembuffer will be mapped to ddr space.
//TODO 2: print buffer support in pocl
  /* Process the kernel arguments. Convert the opaque buffer
     pointers to real device pointers, allocate dynamic local
     memory buffers, etc. */
  for (i = 0; i < meta->num_args; ++i)
    {
      auto al = &(cmd->command.run.arguments[i]);
      if (ARG_IS_LOCAL(meta->arg_info[i]))   
        {
          if (cmd->device->device_alloca_locals)
            {
              /* Local buffers are allocated in the device side work-group
                 launcher. Let's pass only the sizes of the local args in
                 the arg buffer. */
              printf("not support local buffer arg yet.\n");
              //arguments[i] = (void *)al->size;
            }
          else
            {
              arguments[i] = malloc (sizeof (void *));
              //*(void **)(arguments[i]) =pocl_aligned_malloc(MAX_EXTENDED_ALIGNMENT, al->size);
            }
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
        {
          /* It's legal to pass a NULL pointer to clSetKernelArguments. In
             that case we must pass the same NULL forward to the kernel.
             Otherwise, the user must have created a buffer with per device
             pointers stored in the cl_mem. */
          arguments[i] = malloc (sizeof (void *));
          if (al->value == NULL)
            {
              *(void **)arguments[i] = NULL;
            }
          else
            {
              void *ptr = NULL;uint64_t dev_mem_addr;
              if (al->is_svm)
                {
                  ptr = *(void **)al->value;
                }
              else
                {
                  cl_mem m = (*(cl_mem *)(al->value));
                  m->flags=m->flags & CL_MEM_COPY_HOST_PTR
                  err=pocl_ventus_alloc_mem_obj(cmd->device, m, m->mem_host_ptr)
                  assert(0 == CL_SUCCESS);
                  ptr = m->device_ptrs[cmd->device->global_mem_id].mem_ptr;
                }
              *(void **)arguments[i] = (uint64_t *)ptr;
            }
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
          dev_image_t di;
          pocl_fill_dev_image_t (&di, al, cmd->device);

          void *devptr = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                              sizeof (dev_image_t));
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = devptr;
          memcpy (devptr, &di, sizeof (dev_image_t));
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          dev_sampler_t ds;
          pocl_fill_dev_sampler_t (&ds, al);
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = (void *)ds;
        }
      else
        {
          arguments[i] = al->value;
        }
    }

  if (cmd->device->device_alloca_locals)
    {
      printf("notice that ventus hasn't support local buffer as argument yet.\n");
      /* Local buffers are allocated in the device side work-group
         launcher. Let's pass only the sizes of the local args in
         the arg buffer. */
      for (i = 0; i < meta->num_locals; ++i) 
        {
          size_t s = meta->local_sizes[i]; //TODO: create local_buf at ddr, and map argument to this addr.
          size_t j = meta->num_args + i;
          *(size_t *)(arguments[j]) = s;
        }
    }
  else
    {
      for (i = 0; i < meta->num_locals; ++i)
        {
          size_t s = meta->local_sizes[i];
          size_t j = meta->num_args + i;
          arguments[j] = malloc (sizeof (void *));
          void *pp = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, s);
          *(void **)(arguments[j]) = pp;
        }
    }

  /*pc->printf_buffer = d->printf_buffer;
  assert (pc->printf_buffer != NULL);
  pc->printf_buffer_capacity = cmd->device->printf_buffer_size;
  assert (pc->printf_buffer_capacity > 0);
  uint32_t position = 0;
  pc->printf_buffer_position = &position;
  pc->global_var_buffer = program->gvar_storage[dev_i];*/

// create argument buffer now.
uint64_t abuf_size = 0;  
    for (i = 0; i < meta->num_args; ++i) {  
      pocl_argument* al = &(cmd->command.run.arguments[i]);  
      if (ARG_IS_LOCAL(meta->arg_info[i])&& cmd->device->device_alloca_locals) {
        abuf_size += 4;
        //abuf_size += al->size;
      } else
      if ((meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
       || (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
       || (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)) {
        abuf_size += 4;
      } else {
        abuf_size += al->size;
      }
    }
  
  assert(abuf_size <= 0xffff);
  char* abuf_args_data = malloc(abuf_size);
  uint64_t abuf_args_p = 0;
  for(i = 0; i < meta->num_args; ++i) {  
      pocl_argument* al = &(cmd->command.run.arguments[i]);  
      if (ARG_IS_LOCAL(meta->arg_info[i])&& cmd->device->device_alloca_locals) {
        uint32_t local_vaddr=0;
        memcpy(abuf_args_data+abuf_args_p,&local_vaddr,4);
        abuf_args_p+=4;
      } else
      if ((meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
       || (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
       || (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)) {
        memcpy(abuf_args_data+abuf_args_p,((cl_mem)(al->value))->device_ptrs->mem_ptr,4);
        abuf_args_p+=4;
      } else {
        memcpy(abuf_args_data+abuf_args_p,al->value,al->size);
        abuf_args_p+=al->size;
      }
    }
  uint64_t arg_dev_mem_addr;
  err = vt_buf_alloc(d->vt_device, abuf_size, &arg_dev_mem_addr,0,0,0);
  if (err != 0) {
    return CL_DEVICE_NOT_AVAILABLE;
  }
  err = vt_copy_to_dev(d->vt_device,arg_dev_mem_addr,abuf_args_data, abuf_size, 0,0);
  if (err != 0) {
    return CL_DEVICE_NOT_AVAILABLE;
  }

//pass in vmem file  
  char filename[]="vecadd.riscv";
  vt_upload_kernel_file(d->vt_device,filename,0);
  //after checking pocl_cache_binary, use the following to pass in.
   /*if (NULL == d->current_kernel || d->current_kernel != kernel) {    
       d->current_kernel = kernel;
      char program_bin_path[POCL_FILENAME_LENGTH];
      pocl_cache_final_binary_path (program_bin_path, program, dev_i, kernel, NULL, 0);
      err = vt_upload_kernel_file(d->vt_device, program_bin_path,0);      
      assert(0 == err);
    }*/
  uint32_t kernel_entry=0x8000005c;
  ldssize=0x1000; //pass from elf file
  pdssize=0x1000; //pass from elf file
  start_pc=0x80000000; // start.S baseaddr, now lock to 0x80000000
  sgpr_usage=32;
  vgpr_usage=32;
  uint64_t pc_src_size=0x10000000;
  uint64_t pc_dev_mem_addr;
  err = vt_buf_alloc(d->vt_device, pc_src_size, &pc_dev_mem_addr,0,0,0);
  if (err != 0) {
    return CL_DEVICE_NOT_AVAILABLE;
  }

  
  
//prepare privatemem
  uint64_t pds_src_size=pdssize*num_thread*num_warp*num_workgroup;
  uint64_t pds_dev_mem_addr;
  err = vt_buf_alloc(d->vt_device, pds_src_size, &pds_dev_mem_addr,0,0,0);
  if (err != 0) {
    return CL_DEVICE_NOT_AVAILABLE;
  }

  

//prepare kernel_metadata
  char *kernel_metadata=memset(KNL_MAX_METADATA_SIZE);
  memset(kernel_metadata,0,KNL_MAX_METADATA_SIZE);
  memcpy(kernel_metadata+KNL_ENTRY,&kernel_entry,4);
  uint32_t arg_dev_mem_addr_32=(uint32_t)arg_dev_mem_addr;
  memcpy(kernel_metadata+KNL_ARG_BASE,&arg_dev_mem_addr_32,4);
  memcpy(kernel_metadata+KNL_WORK_DIM,&(pc->work_dim),4);
  uint32_t local_size_32[3];local_size_32[0]=(uint32_t)pc->local_size[0];local_size_32[1]=(uint32_t)pc->local_size[1];local_size_32[2]=(uint32_t)pc->local_size[2];
  uint32_t global_offset_32[3];global_offset_32[0]=(uint32_t)pc->global_offset[0];global_offset_32[1]=(uint32_t)pc->global_offset[1];global_offset_32[2]=(uint32_t)pc->global_offset[2];
  uint32_t global_size_32[3];global_size_32[0]=(uint32_t)pc->num_groups[0];global_size_32[1]=(uint32_t)pc->num_groups[1];global_size_32[2]=(uint32_t)pc->num_groups[2];
  memcpy(kernel_metadata+KNL_GL_SIZE_X,global_size_32[0],4);
  memcpy(kernel_metadata+KNL_GL_SIZE_Y,global_size_32[1],4);
  memcpy(kernel_metadata+KNL_GL_SIZE_Z,global_size_32[2],4);
  memcpy(kernel_metadata+KNL_LC_SIZE_X,local_size_32[0],4);
  memcpy(kernel_metadata+KNL_LC_SIZE_Y,local_size_32[1],4);
  memcpy(kernel_metadata+KNL_LC_SIZE_Z,local_size_32[2],4);
  memcpy(kernel_metadata+KNL_GL_OFFSET_X,global_offset_32[0],4);
  memcpy(kernel_metadata+KNL_GL_OFFSET_Y,global_offset_32[1],4);
  memcpy(kernel_metadata+KNL_GL_OFFSET_Z,global_offset_32[2],4);
//memcpy(kernel_metadata+KNL_PRINT_ADDR,global_offset_32[0],4);
  uint64_t knl_dev_mem_addr;
  err = vt_buf_alloc(d->vt_device, KNL_MAX_METADATA_SIZE, &knl_dev_mem_addr,0,0,0);
  if (err != 0) {
    return CL_DEVICE_NOT_AVAILABLE;
  }
  err = vt_copy_to_dev(d->vt_device,knl_dev_mem_addr,kernel_metadata, KNL_MAX_METADATA_SIZE, 0,0);
  if (err != 0) {
    return CL_DEVICE_NOT_AVAILABLE;
  }

  uint64_t pdsbase=pds_dev_mem_addr;
  uint64_t knlbase=knl_dev_mem_addr;
  struct meta_data driver_meta;
    driver_meta.kernel_id=0;
    driver_meta.kernel_size=num_workgroups;
    driver_meta.wf_size=num_thread;
    driver_meta.wg_size=num_warp;
    driver_meta.metaDataBaseAddr=knlbase;
    driver_meta.ldsSize=ldssize;
    driver_meta.pdsSize=pdssize;
    driver_meta.sgprUsage=sgpr_usage;
    driver_meta.vgprUsage=vgpr_usage;
    driver_meta.pdsBaseAddr=pdsbase;

// prepare a write function

#ifdef WRITE_CHISEL_TEST

#endif



//pass metadata to "run" 
  // quick off kernel execution
  err = vt_start(d->vt_device, &driver_meta,0);
  assert(0 == err);

  // wait for the execution to complete
  err = vt_ready_wait(d->vt_device, -1);
  assert(0 == err);

  // move print buffer back or wait to read?     



for (i = 0; i < meta->num_args; ++i)
    {
      if (ARG_IS_LOCAL (meta->arg_info[i]))
        {
          if (!cmd->device->device_alloca_locals)
            {
              POCL_MEM_FREE(*(void **)(arguments[i]));
              POCL_MEM_FREE(arguments[i]);
            }
          else
            {
              /* Device side local space allocation has deallocation via stack
                 unwind. */
            }
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE
               || meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          if (meta->arg_info[i].type != POCL_ARG_TYPE_SAMPLER)
            POCL_MEM_FREE (*(void **)(arguments[i]));
          POCL_MEM_FREE(arguments[i]);
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
        {
          POCL_MEM_FREE(arguments[i]);
        }
    }

  if (!cmd->device->device_alloca_locals)
    for (i = 0; i < meta->num_locals; ++i)
      {
        POCL_MEM_FREE (*(void **)(arguments[meta->num_args + i]));
        POCL_MEM_FREE (arguments[meta->num_args + i]);
      }
  free(arguments);
  free(abuf_args_data);
  free(kernel_metadata);

  

  pocl_release_dlhandle_cache(cmd);
}


cl_int
pocl_ventus_uninit (unsigned j, cl_device_id device)
{
  struct vt_device_data_t *d = (struct vt_device_data_t*)device->data;
  if (NULL == d)
  return CL_SUCCESS;  

  vt_dev_close(d->vt_device);
  
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
  printf("in pocl ventus compile kernel func\n");
  if (cmd != NULL && cmd->type == CL_COMMAND_NDRANGE_KERNEL)
    pocl_check_kernel_dlhandle_cache (cmd, 0, specialize);
}

void pocl_ventus_free(cl_device_id device, cl_mem memobj) {
  cl_mem_flags flags = memobj->flags;
  auto d = (vt_device_data_t *)device->data;
  uint64_t dev_mem_addr = *(memobj->device_ptrs[device->dev_id].mem_ptr);

  /* The host program can provide the runtime with a pointer 
  to a block of continuous memory to hold the memory object 
  when the object is created (CL_MEM_USE_HOST_PTR). 
  Alternatively, the physical memory can be managed 
  by the OpenCL runtime and not be directly accessible 
  to the host program.*/
  if (flags & CL_MEM_USE_HOST_PTR 
   || memobj->shared_mem_allocation_owner != device) {
    abort(); //TODO
  } else {
    vt_buf_free(d->vt_device,mem_obj->size,dev_mem_addr,0,0);
    free(memobj->mem_host_ptr);
    memobj->mem_host_ptr = nullptr;
  }
  if (memobj->flags | CL_MEM_ALLOC_HOST_PTR)
    memobj->mem_host_ptr = NULL;
}


cl_int
pocl_ventus_alloc_mem_obj(cl_device_id device, cl_mem mem_obj, void *host_ptr) {
  
  cl_mem_flags flags = mem_obj->flags;
  unsigned i;
  printf("allocating mem in pocl\n");

  /* Check if some driver has already allocated memory for this mem_obj
     in our global address space, and use that. */
  for (i = 0; i < mem_obj->context->num_devices; ++i) {
    if (!mem_obj->device_ptrs[i].available)
      continue;
    if (mem_obj->device_ptrs[i].global_mem_id == device->global_mem_id && mem_obj->device_ptrs[i].mem_ptr != NULL) {
      mem_obj->device_ptrs[device->dev_id].mem_ptr = mem_obj->device_ptrs[i].mem_ptr;
    
      POCL_MSG_PRINT_INFO("VENTUS: alloc_mem_obj, use already allocated memory\n");
      abort(); // TODO
      return CL_SUCCESS;
    }
  }

  /* Memory for this global memory is not yet allocated -> we'll allocate it. */
  auto d = (vt_device_data_t *)device->data;
  pocl_global_mem_t *mem = device->global_memory;
  int err;
  if (flags & CL_MEM_USE_HOST_PTR) {
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }
  uint64_t dev_mem_addr;
  err = vt_buf_alloc(d->vt_device, mem_obj->size, &dev_mem_addr,0,0,0);
  if (err != 0) {
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  if (flags & CL_MEM_COPY_HOST_PTR) {
    err = vt_copy_to_dev(d->vt_device,dev_mem_addr,host_ptr, mem_obj->size, 0,0);
    if (err != 0) {
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }
  }

  /* Take ownership if not USE_HOST_PTR. */
  if (~flags & CL_MEM_USE_HOST_PTR)
    mem_obj->shared_mem_allocation_owner = device;

  mem_obj->device_ptrs[device->dev_id].mem_ptr = memset(sizeof(uint64_t));
  *(mem_obj->device_ptrs[device->dev_id].mem_ptr)=dev_mem_addr;

  if (flags & CL_MEM_ALLOC_HOST_PTR) {
    abort(); // TODO
  }

  return CL_SUCCESS;
}

void pocl_ventus_read(void *data,
                      void *__restrict__ host_ptr,
                      pocl_mem_identifier *src_mem_id,
                      cl_mem src_buf,
                      size_t offset, 
                      size_t size) {
  struct vt_device_data_t *d = (struct vt_device_data_t *)data;                      
  auto err = vt_copy_from_dev(d->vt_device,*(src_mem_id->mem_ptr)+offset,host_ptr,size,0,0);
  assert(0 == err);
}

void pocl_ventus_write(void *data,
                       const void *__restrict__ host_ptr,
                       pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf,
                       size_t offset, 
                       size_t size) {
  struct vt_device_data_t *d = (struct vt_device_data_t *)data;
  auto err = vt_copy_to_dev(d->vt_device,*(dst_mem_id->mem_ptr)+offset,host_ptr,size,0,0);
  assert(0 == err);
}
