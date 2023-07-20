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
#include <string>
#include <unistd.h>
#include <utlist.h>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <map>

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
  "-Wl," CLANG_RESOURCE_DIR"/../../crt0.o",
  "-Wl," CLANG_RESOURCE_DIR"/../../riscv32clc.o",
  "-Wl,--gc-sections",
  "-L" CLANG_RESOURCE_DIR"/../../",
  "-lworkitem",
  NULL
};

static const char *ventus_other_compile_flags[] = {
	"-I" CLANG_RESOURCE_DIR"/../../../../libclc/generic/include",
	"-O1",
	"-Wl,-T," CLANG_RESOURCE_DIR"/../../../../utils/ldscripts/ventus/elf32lriscv.ld",
	NULL
};

static const char *ventus_objdump_flags[] = {
  "-d",
  "--mattr=+v",
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
  ops->build_source = pocl_ventus_build_source;
  ops->post_build_program = pocl_ventus_post_build_program;
  ops->link_program = NULL;
  ops->build_binary = NULL;
  ops->free_program = NULL;
  ops->setup_metadata = pocl_ventus_setup_metadata;
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
  snprintf(res, 1000, "THU-ventus-GPGPU\n");
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
  dev->available = CL_TRUE;
  dev->compiler_available = CL_TRUE;
  dev->linker_available = CL_TRUE;

  char extensions[1024];
  extensions[0] = 0;
  strcat (extensions, "cl_khr_fp64"
                      " __opencl_c_generic_address_space");
  dev->extensions = strdup (extensions); // no extention support now

  dev->profile = "FULL_PROFILE";
  dev->endian_little = CL_TRUE;

  dev->max_mem_alloc_size = 500 * 1024 * 1024; //100M
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

  // float rounding mode
  dev->single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO
                          | CL_FP_ROUND_TO_INF | CL_FP_FMA | CL_FP_INF_NAN
                          | CL_FP_DENORM;

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
#define PRINT_CHISEL_TESTCODE
#ifdef PRINT_CHISEL_TESTCODE
void fp_write_file(FILE *fp,void *p,uint64_t size){
  for (size_t i = 0; i < (size+sizeof(uint32_t)-1) / sizeof(uint32_t); ++i) 
    fprintf(fp,"%08x\n",*((uint32_t*)p+i));
} 
#endif

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

    uint64_t num_thread=4;
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

#ifdef PRINT_CHISEL_TESTCODE
    uint64_t c_num_buffer=0;
    uint64_t c_max_num_buffer=1024;
    uint64_t c_buffer_base[c_max_num_buffer];
    uint64_t c_buffer_size[c_max_num_buffer];
    uint64_t c_buffer_allocsize[c_max_num_buffer];
    std::string metadata_name_s = std::string(meta->name)+".metadata";
    const char *c_metadata_name = metadata_name_s.c_str();
    std::string data_name_s = std::string(meta->name)+".data";
    const char *c_data_name = data_name_s.c_str();
    FILE *fp_metadata=fopen(c_metadata_name,"w");
    FILE *fp_data=fopen(c_data_name,"w");

    //assume that chisel_test won't use cases with 32 or more input buffer.
#endif

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
      pocl_argument* al = &(cmd->command.run.arguments[i]);
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
              void *ptr = NULL;
              if (al->is_svm)
                {
                  ptr = *(void **)al->value;
                }
              else
                {
                  cl_mem m = (*(cl_mem *)(al->value));
                  err=pocl_ventus_alloc_mem_obj(cmd->device, m, m->mem_host_ptr);
                  assert(0 == CL_SUCCESS);
                  ptr = malloc(sizeof(uint64_t));
                  memcpy(ptr,m->device_ptrs[cmd->device->global_mem_id].mem_ptr,sizeof(uint64_t));

                  #ifdef PRINT_CHISEL_TESTCODE

                      c_buffer_base[c_num_buffer] = *((uint64_t *) ptr);
                      c_buffer_size[c_num_buffer] = m->size;
                      c_buffer_allocsize[c_num_buffer] = m->size;
                      c_num_buffer = c_num_buffer + 1;
                      assert(c_num_buffer <= c_max_num_buffer);
                      if(m->mem_host_ptr)
                          fp_write_file(fp_data, (m->mem_host_ptr), m->size);
                      else {
                          void* zero_data = new uint64_t [m->size];
                          memset(zero_data,0,m->size);
                          fp_write_file(fp_data, zero_data, m->size);
                          delete static_cast<uint64_t*>(zero_data);
                      }
                  #endif
                }
                ((void **)arguments)[i] = ptr;
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
  char* abuf_args_data = (char*)malloc(abuf_size);
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
       // memcpy(abuf_args_data+abuf_args_p,((cl_mem)(al->value))->device_ptrs->mem_ptr,4);
          memcpy(abuf_args_data+abuf_args_p,arguments[i],4);
        abuf_args_p+=4;
      } else {
        memcpy(abuf_args_data+abuf_args_p,al->value,al->size);
        abuf_args_p+=al->size;
      }
    }
  uint64_t arg_dev_mem_addr;
  err = vt_buf_alloc(d->vt_device, abuf_size, &arg_dev_mem_addr,0,0,0);
  if (err != 0) {
    abort();
  }

  #ifdef PRINT_CHISEL_TESTCODE
    c_buffer_base[c_num_buffer]=arg_dev_mem_addr;
    c_buffer_size[c_num_buffer]=abuf_size;
    c_buffer_allocsize[c_num_buffer]=abuf_size;
    c_num_buffer=c_num_buffer+1;
    assert(c_num_buffer<=c_max_num_buffer);
    fp_write_file(fp_data,abuf_args_data,abuf_size);
  #endif

  err = vt_copy_to_dev(d->vt_device,arg_dev_mem_addr,abuf_args_data, abuf_size, 0,0);
  if (err != 0) {
    abort();
  }

  //after checking pocl_cache_binary, use the following to pass in.
   /*if (NULL == d->current_kernel || d->current_kernel != kernel) {    
       d->current_kernel = kernel;
      char program_bin_path[POCL_FILENAME_LENGTH];
      pocl_cache_final_binary_path (program_bin_path, program, dev_i, kernel, NULL, 0);
      err = vt_upload_kernel_file(d->vt_device, program_bin_path,0);      
      assert(0 == err);
    }*/

   /**
    *这个是kernel函数的入口地址，可以在终端执行nm -s object.riscv看到kernel函数的入口
    *clCreateKernel.c line 79
    */
	uint32_t kernel_entry;
#ifdef __linux__
    std::string kernel_name(meta->name);
   	std::string kernel_entry_cmd = std::string(R"(nm -s object.riscv | grep )") +kernel_name+ std::string(R"( | grep -o '^[^ ]*')");
	FILE *fp0 = popen(kernel_entry_cmd.c_str(), "r");
	if(fp0 == NULL) {
		POCL_MSG_ERR("running compile kernel failed");
		return;
	}
	char temp2[1024];
	while (fgets(temp2, 1024, fp0) != NULL)
	{
		kernel_entry = static_cast<uint32_t>(std::strtoul(temp2, nullptr, 16));
	}
	int status2=pclose(fp0);
	if (status2 == -1) {
		perror("pclose() failed");
		exit(EXIT_FAILURE);
	} else {
		POCL_MSG_PRINT_LLVM("Kernel entry of \"%s\" is : \"0x%x\"\n", kernel->name, kernel_entry);
	}
#elif
	POCL_MSG_ERR("This operate system is not supported now by ventus, please use linux! \n");
	exit(1);
#endif

  ldssize=0x1000; //pass from elf file
  pdssize=0x1000; //pass from elf file
  start_pc=0x80000000; // start.S baseaddr, now lock to 0x80000000
  sgpr_usage=32;
  vgpr_usage=32;
  uint64_t pc_src_size=0x10000000;
  uint64_t pc_dev_mem_addr = 0x80000000;
//  ///TODO 在这个地址放程序段
//  err = vt_buf_alloc(d->vt_device, pc_src_size, &pc_dev_mem_addr,0,0,0);
//  if (err != 0) {
//    abort();
//  }
  /// parsing object file to obtain vmem file using assembler
#ifdef __linux__
	std::string assembler_path = CLANG;
    assembler_path = assembler_path.substr(0,assembler_path.length()-6);
	assembler_path += "/../../assemble.sh";
  	system((std::string("chmod +x ") + assembler_path).c_str());
  	assembler_path += " object";
  	system(assembler_path.c_str());
	POCL_MSG_PRINT_LLVM("Vmem file has been written to object.vmem\n");
#elif
	POCL_MSG_ERR("This operate system is not supported now by ventus, please use linux! \n");
	exit(1);
#endif
	//pass in vmem file
	char filename[]="object.riscv";
	///将text段搬到ddr(not related to spike),并且起始地址必须是0x80000000(spike专用)，verilator需要先解析出vmem,然后上传程序段
	vt_upload_kernel_file(d->vt_device,filename,0);
  #ifdef PRINT_CHISEL_TESTCODE
    //this file includes all kernels of executable file, kernel actually to be executed is determined by metadata.
	std::ifstream vmem_file("object.vmem");
	vmem_file.seekg(0, vmem_file.end);
	auto size = vmem_file.tellg();
	std::string content;
	content.resize(size);
	vmem_file.seekg(0, vmem_file.beg);
	vmem_file.read(&content[0], size);
	content.erase(std::remove(content.begin(), content.end(), '\n'), content.end());
	int vmem_line_count = content.length() / 8;
	uint32_t* vmem_content = new uint32_t[vmem_line_count];
	for (int i = 0; i < vmem_line_count; i++) {
		std::string substring = (content).substr(i * 8, 8); // 每次提取8个字符
		unsigned int value = std::stoul(substring, nullptr, 16); // 转换为无符号整数
		memcpy(vmem_content + i, &value, sizeof(uint32_t)); // 复制到数组中
	}
	fp_write_file(fp_data,vmem_content, vmem_line_count*sizeof(uint32_t));
	fp_write_file(fp_metadata, &(pc_dev_mem_addr), sizeof(uint64_t));
	delete []vmem_content;
	content.clear();

	c_buffer_base[c_num_buffer]=pc_dev_mem_addr;
	c_buffer_size[c_num_buffer]=vmem_line_count*sizeof(uint32_t);
	c_buffer_allocsize[c_num_buffer]=pc_src_size;
	c_num_buffer=c_num_buffer+1;
	assert(c_num_buffer<=c_max_num_buffer);
  #endif

  
  
//prepare privatemem
  uint64_t pds_src_size=pdssize*num_thread*num_warp*num_workgroup;
  uint64_t pds_dev_mem_addr;
  err = vt_buf_alloc(d->vt_device, pds_src_size, &pds_dev_mem_addr,0,0,0);
  if (err != 0) {
    abort();
  }
  #ifdef PRINT_CHISEL_TESTCODE
    c_buffer_base[c_num_buffer]=pds_dev_mem_addr;
    c_buffer_size[c_num_buffer]=0;
    c_buffer_allocsize[c_num_buffer]=pds_src_size;
    c_num_buffer=c_num_buffer+1;
    assert(c_num_buffer<=c_max_num_buffer);
  #endif



//prepare kernel_metadata
  char *kernel_metadata= (char*)malloc(sizeof(char)*KNL_MAX_METADATA_SIZE);
  memset(kernel_metadata,0,KNL_MAX_METADATA_SIZE);
  memcpy(kernel_metadata+KNL_ENTRY,&kernel_entry,4);
  uint32_t arg_dev_mem_addr_32=(uint32_t)arg_dev_mem_addr;
  memcpy(kernel_metadata+KNL_ARG_BASE,&arg_dev_mem_addr_32,4);
  memcpy(kernel_metadata+KNL_WORK_DIM,&(pc->work_dim),4);
  uint32_t local_size_32[3];local_size_32[0]=(uint32_t)pc->local_size[0];local_size_32[1]=(uint32_t)pc->local_size[1];local_size_32[2]=(uint32_t)pc->local_size[2];
  uint32_t global_offset_32[3];global_offset_32[0]=(uint32_t)pc->global_offset[0];global_offset_32[1]=(uint32_t)pc->global_offset[1];global_offset_32[2]=(uint32_t)pc->global_offset[2];
  uint32_t global_size_32[3];global_size_32[0]=(uint32_t)pc->num_groups[0]*local_size_32[0];global_size_32[1]=(uint32_t)pc->num_groups[1]*local_size_32[1];global_size_32[2]=(uint32_t)pc->num_groups[2]*local_size_32[2];
  memcpy(kernel_metadata+KNL_GL_SIZE_X,&global_size_32[0],4);
  memcpy(kernel_metadata+KNL_GL_SIZE_Y,&global_size_32[1],4);
  memcpy(kernel_metadata+KNL_GL_SIZE_Z,&global_size_32[2],4);
  memcpy(kernel_metadata+KNL_LC_SIZE_X,&local_size_32[0],4);
  memcpy(kernel_metadata+KNL_LC_SIZE_Y,&local_size_32[1],4);
  memcpy(kernel_metadata+KNL_LC_SIZE_Z,&local_size_32[2],4);
  memcpy(kernel_metadata+KNL_GL_OFFSET_X,&global_offset_32[0],4);
  memcpy(kernel_metadata+KNL_GL_OFFSET_Y,&global_offset_32[1],4);
  memcpy(kernel_metadata+KNL_GL_OFFSET_Z,&global_offset_32[2],4);
//memcpy(kernel_metadata+KNL_PRINT_ADDR,global_offset_32[0],4);
  uint64_t knl_dev_mem_addr;
  err = vt_buf_alloc(d->vt_device, KNL_MAX_METADATA_SIZE, &knl_dev_mem_addr,0,0,0);
  if (err != 0) {
    abort();
  }
  err = vt_copy_to_dev(d->vt_device,knl_dev_mem_addr,kernel_metadata, KNL_MAX_METADATA_SIZE, 0,0);
  if (err != 0) {
    abort();
  }
  POCL_MSG_PRINT_INFO("kernel metadata has been written to 0x%x\n", knl_dev_mem_addr);
  #ifdef PRINT_CHISEL_TESTCODE
    c_buffer_base[c_num_buffer]=knl_dev_mem_addr;
    c_buffer_size[c_num_buffer]=KNL_MAX_METADATA_SIZE;
    c_buffer_allocsize[c_num_buffer]=KNL_MAX_METADATA_SIZE;
    c_num_buffer=c_num_buffer+1;
    assert(c_num_buffer<=c_max_num_buffer);
    fp_write_file(fp_data,kernel_metadata,KNL_MAX_METADATA_SIZE);
  #endif


  pdsbase=pds_dev_mem_addr;
  knlbase=knl_dev_mem_addr;
  struct meta_data driver_meta;
    driver_meta.kernel_id=0;
    driver_meta.kernel_size[0]=num_workgroups[0];
    driver_meta.kernel_size[1]=num_workgroups[1];
    driver_meta.kernel_size[2]=num_workgroups[2];
    driver_meta.wf_size=num_thread;
    driver_meta.wg_size=num_warp;
    driver_meta.metaDataBaseAddr=knlbase;
    driver_meta.ldsSize=ldssize;
    driver_meta.pdsSize=pdssize;
    driver_meta.sgprUsage=sgpr_usage;
    driver_meta.vgprUsage=vgpr_usage;
    driver_meta.pdsBaseAddr=pdsbase;

// prepare a write function

  #ifdef PRINT_CHISEL_TESTCODE
    fp_write_file(fp_metadata,&(driver_meta.kernel_id),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.kernel_size[0]),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.kernel_size[1]),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.kernel_size[2]),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.wf_size),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.wg_size),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.metaDataBaseAddr),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.ldsSize),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.pdsSize),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.sgprUsage),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.vgprUsage),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(driver_meta.pdsBaseAddr),sizeof(uint64_t));
    fp_write_file(fp_metadata,&(c_num_buffer),sizeof(uint64_t));
    for(int i=0;i<c_num_buffer;i++)  fp_write_file(fp_metadata,&c_buffer_base[i],sizeof(uint64_t));
    for(int i=0;i<c_num_buffer;i++)  fp_write_file(fp_metadata,&c_buffer_size[i],sizeof(uint64_t));
    for(int i=0;i<c_num_buffer;i++)  fp_write_file(fp_metadata,&c_buffer_allocsize[i],sizeof(uint64_t));
    fclose(fp_metadata);
    fclose(fp_data);
  #endif



//pass metadata to "run" 
  // quick off kernel execution
  err = vt_start(d->vt_device, &driver_meta,0);
  assert(0 == err);

  // wait for the execution to complete
  err = vt_ready_wait(d->vt_device, 1000);
  assert(0 == err);

  // move print buffer back or wait to read?     

    // rename log file from spike and add index for log
    const char* sp_logname = "object.riscv.log";
    static std::map<std::string, int> knl_name_list;
    auto it = knl_name_list.find(meta->name);
    if(it != knl_name_list.end())
        it->second++;
    else
        knl_name_list[meta->name] = 0;
    char newName[256]; // 假设文件名不超过 255 个字符
    FILE* logfp = fopen(sp_logname, "r");
    if(logfp) {
        fclose(logfp);
        strcpy(newName, meta->name);
        sprintf(newName, "%s_%d.log",meta->name,knl_name_list[meta->name]);
        //strcat(newName, ".log");
        if(rename(sp_logname, newName) == 0) {
            printf("Log file %s renamed successfully to %s.\n", sp_logname, newName);
        } else {
            printf("Unable to rename the log file %s.\n", sp_logname);
        }
    }

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

  
  
  //pocl_release_dlhandle_cache(cmd);
 
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


void ventus_command_scheduler (struct vt_device_data_t *d)
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

  //if (node != NULL && node->type == CL_COMMAND_NDRANGE_KERNEL)
  //  pocl_check_kernel_dlhandle_cache (node, 1, 1);

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
  vt_device_data_t* d = (vt_device_data_t *)device->data;
  uint64_t dev_mem_addr = *((uint64_t*)(memobj->device_ptrs[device->dev_id].mem_ptr));

  /* The host program can provide the runtime with a pointer 
  to a block of continuous memory to hold the memory object 
  when the object is created (CL_MEM_USE_HOST_PTR). 
  Alternatively, the physical memory can be managed 
  by the OpenCL runtime and not be directly accessible 
  to the host program.*/
  if (flags & CL_MEM_USE_HOST_PTR) {
    abort(); //TODO
  } else {
    vt_buf_free(d->vt_device,memobj->size,&dev_mem_addr,0,0);
    free(memobj->mem_host_ptr);
    memobj->mem_host_ptr = NULL;
  }
  if (memobj->flags | CL_MEM_ALLOC_HOST_PTR)
    memobj->mem_host_ptr = NULL;
}


cl_int
pocl_ventus_alloc_mem_obj(cl_device_id device, cl_mem mem_obj, void *host_ptr) {
  
  cl_mem_flags flags = mem_obj->flags;
  unsigned i;
  vt_device_data_t* d = (vt_device_data_t *)device->data;
  pocl_global_mem_t *mem = device->global_memory;
  int err;
  uint64_t dev_mem_addr;
  // if this memory object has not been allocated device memory space,
  // then allocating a device memory and binding the memory pointer to cl_mem object
  if(!mem_obj->device_ptrs[device->dev_id].mem_ptr) {
      err = vt_buf_alloc(d->vt_device, mem_obj->size, &dev_mem_addr,0,0,0);
      if (err != 0) {
          return CL_MEM_OBJECT_ALLOCATION_FAILURE;
      }
      free(mem_obj->device_ptrs[device->dev_id].mem_ptr);
      mem_obj->device_ptrs[device->dev_id].mem_ptr= malloc(sizeof(uint64_t));
      memcpy((mem_obj->device_ptrs[device->dev_id].mem_ptr),&dev_mem_addr,sizeof(uint64_t));
  }

  // if the memory object has been allocated device memory pointer and
  // if the flags indicates that copy data from host ptr, then do the following operations.
  if ((flags & CL_MEM_COPY_HOST_PTR) && mem_obj->device_ptrs[device->dev_id].mem_ptr) {
    err = vt_copy_to_dev(d->vt_device,*(uint64_t*)(mem_obj->device_ptrs[device->dev_id].mem_ptr),mem_obj->mem_host_ptr, mem_obj->size, 0,0);
    if (err != 0) {
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }
  }

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
  int err = vt_copy_from_dev(d->vt_device,*((uint64_t*)(src_mem_id->mem_ptr))+offset,host_ptr,size,0,0);
  assert(0 == err);
}

void pocl_ventus_write(void *data,
                       const void *__restrict__ host_ptr,
                       pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf,
                       size_t offset, 
                       size_t size) {
  struct vt_device_data_t *d = (struct vt_device_data_t *)data;
  int err = vt_copy_to_dev(d->vt_device,*((uint64_t*)(dst_mem_id->mem_ptr))+offset,host_ptr,size,0,0);
  assert(0 == err);
}



int pocl_ventus_setup_metadata  (cl_device_id device, cl_program program,
                                 unsigned program_device_i)
{
    return pocl_driver_setup_metadata  (device, program,
     program_device_i);
}

int pocl_ventus_build_program(cl_program program,
                            unsigned device_i,
                            cl_uint num_input_headers,
                            const cl_program *input_headers,
                            const char **header_include_names,
                            int linking_program)

{
  return 0;//unused file now.
}

int pocl_ventus_build_source (cl_program program, cl_uint device_i,
                              cl_uint num_input_headers,
                              const cl_program *input_headers,
                              const char **header_include_names,
                              int link_builtin_lib) {
    return pocl_driver_build_source(program, device_i, num_input_headers,
                                       input_headers, header_include_names, 0);
}

int pocl_ventus_post_build_program (cl_program program, cl_uint device_i) {
  const char* clang_path(CLANG);
	if (!pocl_exists(clang_path)) {
		POCL_MSG_ERR("$CLANG: '%s' doesn't exist\n", clang_path);
		return -1;
	}
  std::stringstream ss_cmd;
	std::stringstream ss_out;

  char program_bc_path[POCL_FILENAME_LENGTH];
    

  //pocl_cache_create_program_cachedir(program, device_i, program->source,
  //                                     strlen(program->source),
  //                                     program_bc_path);  
  //TODO: move .cl and .riscv file into program_bc_path, and let spike read file from this path.
  std::ofstream outfile("object.cl");
  outfile << program->source;
  outfile.close();

  cl_device_id device = program->devices[device_i];

    ss_cmd << clang_path <<" -cl-std=CL2.0 " << "-target " << device->llvm_target_triplet << " -mcpu=" << device->llvm_cpu  << " object.cl " << " -o " << "object.riscv ";
	for(int i = 0; ventus_final_ld_flags[i] != NULL; i++) {
		ss_cmd << ventus_final_ld_flags[i] << " ";
	}
	for(int i = 0; ventus_other_compile_flags[i] != NULL; i++) {
		ss_cmd << ventus_other_compile_flags[i] << " ";
	}
  ss_cmd << "-Wl,--init=" << program->kernel_meta->name << " ";
#ifdef POCL_DEBUG_FLAG_GENERAL
	ss_cmd << " -w ";
#endif
	ss_cmd << program->compiler_options << std::endl;
	POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());

	FILE *fp = popen(ss_cmd.str().c_str(), "r");
	if(fp == NULL) {
		POCL_MSG_ERR("running compile kernel failed");
		return -1;
	}
	char temp[1024];

	while (fgets(temp, 1024, fp) != NULL)
	{
		ss_out << temp;
	}
	int status=pclose(fp);
    if (status == -1) {
        perror("pclose() failed");
        exit(EXIT_FAILURE);
    } else {
        POCL_MSG_PRINT_LLVM("after calling clang, the output is : \"%s\"\n", ss_out.str().c_str());
    }


    /*const char* env_var = std::getenv("POCL_PRINT_CHISEL_TESTCODE");
    if (env_var != nullptr) {
        if (std::string(env_var) == "y") {
            // 使用宏定义1
            #ifndef PRINT_CHISEL_TESTCODE
            #define PRINT_CHISEL_TESTCODE
            #endif
            printf("generate chisel testcode\n");
        } 
    } */


  pocl_ventus_release_IR(program);
return 0;

}
