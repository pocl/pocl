/* basic.c - a minimalistic pocl device driver layer implementation

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos and
                 2011-2014 Pekka Jääskeläinen / Tampere University of Technology
   
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

#include "basic.h"
#include "cpuinfo.h"
#include "topology/pocl_topology.h"
#include "common.h"
#include "utlist.h"
#include "devices.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>

#ifndef _MSC_VER
#  include <sys/time.h>
#  include <sys/resource.h>
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include "pocl_cache.h"

#define max(a,b) (((a) > (b)) ? (a) : (b))

#define COMMAND_LENGTH 2048
#define WORKGROUP_STRING_LENGTH 1024

struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  /* Loaded kernel dynamic library handle. */
  lt_dlhandle current_dlhandle;
};

static const cl_image_format supported_image_formats[] = {
    { CL_R, CL_SNORM_INT8 },
    { CL_R, CL_SNORM_INT16 },
    { CL_R, CL_UNORM_INT8 },
    { CL_R, CL_UNORM_INT16 },
    { CL_R, CL_UNORM_SHORT_565 }, 
    { CL_R, CL_UNORM_SHORT_555 },
    { CL_R, CL_UNORM_INT_101010 }, 
    { CL_R, CL_SIGNED_INT8 },
    { CL_R, CL_SIGNED_INT16 }, 
    { CL_R, CL_SIGNED_INT32 },
    { CL_R, CL_UNSIGNED_INT8 }, 
    { CL_R, CL_UNSIGNED_INT16 },
    { CL_R, CL_UNSIGNED_INT32 }, 
    { CL_R, CL_HALF_FLOAT },
    { CL_R, CL_FLOAT },
    { CL_Rx, CL_SNORM_INT8 },
    { CL_Rx, CL_SNORM_INT16 },
    { CL_Rx, CL_UNORM_INT8 },
    { CL_Rx, CL_UNORM_INT16 },
    { CL_Rx, CL_UNORM_SHORT_565 }, 
    { CL_Rx, CL_UNORM_SHORT_555 },
    { CL_Rx, CL_UNORM_INT_101010 }, 
    { CL_Rx, CL_SIGNED_INT8 },
    { CL_Rx, CL_SIGNED_INT16 }, 
    { CL_Rx, CL_SIGNED_INT32 },
    { CL_Rx, CL_UNSIGNED_INT8 }, 
    { CL_Rx, CL_UNSIGNED_INT16 },
    { CL_Rx, CL_UNSIGNED_INT32 }, 
    { CL_Rx, CL_HALF_FLOAT },
    { CL_Rx, CL_FLOAT },
    { CL_A, CL_SNORM_INT8 },
    { CL_A, CL_SNORM_INT16 },
    { CL_A, CL_UNORM_INT8 },
    { CL_A, CL_UNORM_INT16 },
    { CL_A, CL_UNORM_SHORT_565 }, 
    { CL_A, CL_UNORM_SHORT_555 },
    { CL_A, CL_UNORM_INT_101010 }, 
    { CL_A, CL_SIGNED_INT8 },
    { CL_A, CL_SIGNED_INT16 }, 
    { CL_A, CL_SIGNED_INT32 },
    { CL_A, CL_UNSIGNED_INT8 }, 
    { CL_A, CL_UNSIGNED_INT16 },
    { CL_A, CL_UNSIGNED_INT32 }, 
    { CL_A, CL_HALF_FLOAT },
    { CL_A, CL_FLOAT },
    { CL_RG, CL_SNORM_INT8 },
    { CL_RG, CL_SNORM_INT16 },
    { CL_RG, CL_UNORM_INT8 },
    { CL_RG, CL_UNORM_INT16 },
    { CL_RG, CL_UNORM_SHORT_565 }, 
    { CL_RG, CL_UNORM_SHORT_555 },
    { CL_RG, CL_UNORM_INT_101010 }, 
    { CL_RG, CL_SIGNED_INT8 },
    { CL_RG, CL_SIGNED_INT16 }, 
    { CL_RG, CL_SIGNED_INT32 },
    { CL_RG, CL_UNSIGNED_INT8 }, 
    { CL_RG, CL_UNSIGNED_INT16 },
    { CL_RG, CL_UNSIGNED_INT32 }, 
    { CL_RG, CL_HALF_FLOAT },
    { CL_RG, CL_FLOAT },
    { CL_RGx, CL_SNORM_INT8 },
    { CL_RGx, CL_SNORM_INT16 },
    { CL_RGx, CL_UNORM_INT8 },
    { CL_RGx, CL_UNORM_INT16 },
    { CL_RGx, CL_UNORM_SHORT_565 }, 
    { CL_RGx, CL_UNORM_SHORT_555 },
    { CL_RGx, CL_UNORM_INT_101010 }, 
    { CL_RGx, CL_SIGNED_INT8 },
    { CL_RGx, CL_SIGNED_INT16 }, 
    { CL_RGx, CL_SIGNED_INT32 },
    { CL_RGx, CL_UNSIGNED_INT8 }, 
    { CL_RGx, CL_UNSIGNED_INT16 },
    { CL_RGx, CL_UNSIGNED_INT32 }, 
    { CL_RGx, CL_HALF_FLOAT },
    { CL_RGx, CL_FLOAT },
    { CL_RA, CL_SNORM_INT8 },
    { CL_RA, CL_SNORM_INT16 },
    { CL_RA, CL_UNORM_INT8 },
    { CL_RA, CL_UNORM_INT16 },
    { CL_RA, CL_UNORM_SHORT_565 }, 
    { CL_RA, CL_UNORM_SHORT_555 },
    { CL_RA, CL_UNORM_INT_101010 }, 
    { CL_RA, CL_SIGNED_INT8 },
    { CL_RA, CL_SIGNED_INT16 }, 
    { CL_RA, CL_SIGNED_INT32 },
    { CL_RA, CL_UNSIGNED_INT8 }, 
    { CL_RA, CL_UNSIGNED_INT16 },
    { CL_RA, CL_UNSIGNED_INT32 }, 
    { CL_RA, CL_HALF_FLOAT },
    { CL_RA, CL_FLOAT },
    { CL_RGBA, CL_SNORM_INT8 },
    { CL_RGBA, CL_SNORM_INT16 },
    { CL_RGBA, CL_UNORM_INT8 },
    { CL_RGBA, CL_UNORM_INT16 },
    { CL_RGBA, CL_UNORM_SHORT_565 }, 
    { CL_RGBA, CL_UNORM_SHORT_555 },
    { CL_RGBA, CL_UNORM_INT_101010 }, 
    { CL_RGBA, CL_SIGNED_INT8 },
    { CL_RGBA, CL_SIGNED_INT16 }, 
    { CL_RGBA, CL_SIGNED_INT32 },
    { CL_RGBA, CL_UNSIGNED_INT8 }, 
    { CL_RGBA, CL_UNSIGNED_INT16 },
    { CL_RGBA, CL_UNSIGNED_INT32 }, 
    { CL_RGBA, CL_HALF_FLOAT },
    { CL_RGBA, CL_FLOAT },
    { CL_INTENSITY, CL_UNORM_INT8 }, 
    { CL_INTENSITY, CL_UNORM_INT16 }, 
    { CL_INTENSITY, CL_SNORM_INT8 }, 
    { CL_INTENSITY, CL_SNORM_INT16 }, 
    { CL_INTENSITY, CL_HALF_FLOAT }, 
    { CL_INTENSITY, CL_FLOAT },
    { CL_LUMINANCE, CL_UNORM_INT8 }, 
    { CL_LUMINANCE, CL_UNORM_INT16 }, 
    { CL_LUMINANCE, CL_SNORM_INT8 }, 
    { CL_LUMINANCE, CL_SNORM_INT16 }, 
    { CL_LUMINANCE, CL_HALF_FLOAT }, 
    { CL_LUMINANCE, CL_FLOAT },
    { CL_RGB, CL_UNORM_SHORT_565 }, 
    { CL_RGB, CL_UNORM_SHORT_555 },
    { CL_RGB, CL_UNORM_INT_101010 }, 
    { CL_RGBx, CL_UNORM_SHORT_565 }, 
    { CL_RGBx, CL_UNORM_SHORT_555 },
    { CL_RGBx, CL_UNORM_INT_101010 }, 
    { CL_ARGB, CL_SNORM_INT8 },
    { CL_ARGB, CL_UNORM_INT8 },
    { CL_ARGB, CL_SIGNED_INT8 },
    { CL_ARGB, CL_UNSIGNED_INT8 }, 
    { CL_BGRA, CL_SNORM_INT8 },
    { CL_BGRA, CL_UNORM_INT8 },
    { CL_BGRA, CL_SIGNED_INT8 },
    { CL_BGRA, CL_UNSIGNED_INT8 }
 };


void
pocl_basic_init_device_ops(struct pocl_device_ops *ops)
{
  ops->device_name = "basic";

  ops->init_device_infos = pocl_basic_init_device_infos;
  ops->probe = pocl_basic_probe;
  ops->uninit = pocl_basic_uninit;
  ops->init = pocl_basic_init;
  ops->alloc_mem_obj = pocl_basic_alloc_mem_obj;
  ops->free = pocl_basic_free;
  ops->read = pocl_basic_read;
  ops->read_rect = pocl_basic_read_rect;
  ops->write = pocl_basic_write;
  ops->write_rect = pocl_basic_write_rect;
  ops->copy = pocl_basic_copy;
  ops->copy_rect = pocl_basic_copy_rect;
  ops->fill_rect = pocl_basic_fill_rect;
  ops->map_mem = pocl_basic_map_mem;
  ops->compile_submitted_kernels = pocl_basic_compile_submitted_kernels;
  ops->run = pocl_basic_run;
  ops->run_native = pocl_basic_run_native;
  ops->get_timer_value = pocl_basic_get_timer_value;
  ops->get_supported_image_formats = pocl_basic_get_supported_image_formats;
}

void
pocl_basic_init_device_infos(struct _cl_device_id* dev)
{
  dev->type = CL_DEVICE_TYPE_CPU;
  dev->vendor_id = 0;
  dev->max_compute_units = 0;
  dev->max_work_item_dimensions = 3;

  /*
    The hard restriction will be the context data which is
    stored in stack that can be as small as 8K in Linux.
    Thus, there should be enough work-items alive to fill up
    the SIMD lanes times the vector units, but not more than
    that to avoid stack overflow and cache trashing.
  */
  dev->max_work_item_sizes[0] = dev->max_work_item_sizes[1] =
	  dev->max_work_item_sizes[2] = dev->max_work_group_size = 1024*4;

  dev->preferred_wg_size_multiple = 8;
  dev->preferred_vector_width_char = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR;
  dev->preferred_vector_width_short = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT;
  dev->preferred_vector_width_int = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT;
  dev->preferred_vector_width_long = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG;
  dev->preferred_vector_width_float = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT;
  dev->preferred_vector_width_double = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE;
  dev->preferred_vector_width_half = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF;
  /* TODO: figure out what the difference between preferred and native widths are */
  dev->native_vector_width_char = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR;
  dev->native_vector_width_short = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT;
  dev->native_vector_width_int = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT;
  dev->native_vector_width_long = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG;
  dev->native_vector_width_float = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT;
  dev->native_vector_width_double = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE;
  dev->native_vector_width_half = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF;
  dev->max_clock_frequency = 0;
  dev->address_bits = POCL_DEVICE_ADDRESS_BITS;

  dev->image_support = CL_TRUE;
  /* Use the minimum values until we get a more sensible
     upper limit from somewhere. */
  dev->max_read_image_args = dev->max_write_image_args = 128;
  dev->image2d_max_width = dev->image2d_max_height = 8192;
  dev->image3d_max_width = dev->image3d_max_height = dev->image3d_max_depth = 2048;
  dev->image_max_buffer_size = 65536;
  dev->image_max_array_size = 2048;
  dev->max_samplers = 16;
  dev->max_constant_args = 8;

  dev->max_mem_alloc_size = 0;

  dev->max_parameter_size = 1024;
  dev->min_data_type_align_size = MAX_EXTENDED_ALIGNMENT; // this is in bytes
  dev->mem_base_addr_align = MAX_EXTENDED_ALIGNMENT*8; // this is in bits
  dev->half_fp_config = 0;
  dev->single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN;
  dev->double_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN;
  dev->global_mem_cache_type = CL_NONE;
  dev->global_mem_cacheline_size = 0;
  dev->global_mem_cache_size = 0;
  dev->global_mem_size = 0;
  dev->max_constant_buffer_size = 0;
  dev->local_mem_type = CL_GLOBAL;
  dev->local_mem_size = 0;
  dev->error_correction_support = CL_FALSE;
  dev->host_unified_memory = CL_TRUE;
  dev->profiling_timer_resolution = 0;
  dev->endian_little = !(WORDS_BIGENDIAN);
  dev->available = CL_TRUE;
  dev->compiler_available = CL_TRUE;
  dev->spmd = CL_FALSE;
  dev->execution_capabilities = CL_EXEC_KERNEL | CL_EXEC_NATIVE_KERNEL;
  dev->queue_properties = CL_QUEUE_PROFILING_ENABLE;
  dev->platform = 0;

  dev->parent_device = NULL;
  // basic does not support partitioning
  dev->max_sub_devices = 1;
  dev->num_partition_properties = 1;
  dev->partition_properties = calloc(dev->num_partition_properties,
    sizeof(cl_device_partition_property));
  dev->num_partition_types = 0;
  dev->partition_type = NULL;

  /* printf buffer size is meaningless for pocl, so just set it to
   * the minimum value required by the spec
   */
  dev->printf_buffer_size = 1024*1024 ;
  dev->vendor = "pocl";
  dev->profile = "FULL_PROFILE";
  /* Note: The specification describes identifiers being delimited by
     only a single space character. Some programs that check the device's
     extension  string assume this rule. Future extension additions should
     ensure that there is no more than a single space between
     identifiers. */

#ifndef _CL_DISABLE_LONG
#define DOUBLE_EXT "cl_khr_fp64 "
#else
#define DOUBLE_EXT 
#endif

#ifndef _CL_DISABLE_HALF
#define HALF_EXT "cl_khr_fp16 "
#else
#define HALF_EXT
#endif

  dev->extensions = DOUBLE_EXT HALF_EXT "cl_khr_byte_addressable_store";

  dev->llvm_target_triplet = OCL_KERNEL_TARGET;
  dev->llvm_cpu = OCL_KERNEL_TARGET_CPU;
  dev->has_64bit_long = 1;
  dev->autolocals_to_args = 1;
}

unsigned int
pocl_basic_probe(struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);

  /* No env specified, so pthread will be used instead of basic */
  if(env_count < 0)
    return 0;

  return env_count;
}

#define MIN_MAX_MEM_ALLOC_SIZE (128*1024*1024)

/* set maximum allocation sizes for buffers and images */
void
pocl_basic_set_buffer_image_limits(cl_device_id device)
{
  /* Maximum allocation size: we don't have hardware limits, so we
   * can potentially allocate the whole memory for a single buffer, unless
   * of course there are limits set at the operating system level. Of course
   * we still have to respect the OpenCL-commanded minimum */
  size_t alloc_limit = SIZE_MAX;

#ifndef _MSC_VER
  // TODO getrlimit equivalent under Windows
  struct rlimit limits;
  int ret = getrlimit(RLIMIT_DATA, &limits);
  if (ret == 0)
    alloc_limit = limits.rlim_cur;
#endif
  if (alloc_limit > device->global_mem_size)
    alloc_limit = device->global_mem_size;
  else if (alloc_limit < MIN_MAX_MEM_ALLOC_SIZE)
    alloc_limit = MIN_MAX_MEM_ALLOC_SIZE;
  // TODO in theory now if alloc_limit was > rlim_cur and < rlim_max
  // we should try and setrlimit to alloc_limit, or allocations might fail

  device->local_mem_size = device->max_constant_buffer_size =
    device->max_mem_alloc_size = alloc_limit;

  /* We don't have hardware limitations on the buffer-backed image sizes,
   * so we set the maximum size in terms of the maximum amount of pixels
   * that fix in max_mem_alloc_size. A single pixel can take up to 4 32-bit channels,
   * i.e. 16 bytes.
   */
  size_t max_pixels = device->max_mem_alloc_size/16;
  if (max_pixels > device->image_max_buffer_size)
    device->image_max_buffer_size = max_pixels;

  /* Similarly, we can take the 2D image size limit to be the largest power of 2
   * whose square fits in image_max_buffer_size; since the 2D image size limit
   * starts at a power of 2, it's a simple matter of doubling.
   * This is actually completely arbitrary, another equally valid option
   * would be to have each maximum dimension match the image_max_buffer_size.
   */
  max_pixels = device->image2d_max_width;
  // keep doubing until we go over
  while (max_pixels <= device->image_max_buffer_size/max_pixels)
    max_pixels *= 2;
  // halve before assignment
  max_pixels /= 2;
  if (max_pixels > device->image2d_max_width)
    device->image2d_max_width = device->image2d_max_height = max_pixels;

  /* Same thing for 3D images, of course with cubes. Again, totally arbitrary. */
  max_pixels = device->image3d_max_width;
  // keep doubing until we go over
  while (max_pixels*max_pixels <= device->image_max_buffer_size/max_pixels)
    max_pixels *= 2;
  // halve before assignment
  max_pixels /= 2;
  if (max_pixels > device->image3d_max_width)
  device->image3d_max_width = device->image3d_max_height =
    device->image3d_max_depth = max_pixels;

}

void
pocl_basic_init (cl_device_id device, const char* parameters)
{
  struct data *d;
  static int global_mem_id;
  static int first_basic_init = 1;
  
  if (first_basic_init)
    {
      first_basic_init = 0;
      global_mem_id = device->dev_id;
    }
  device->global_mem_id = global_mem_id;

  d = (struct data *) calloc (1, sizeof (struct data));
  
  d->current_kernel = NULL;
  d->current_dlhandle = 0;
  device->data = d;
  /* hwloc probes OpenCL device info at its initialization in case
     the OpenCL extension is enabled. This causes to printout 
     an unimplemented property error because hwloc is used to
     initialize global_mem_size which it is not yet. Just put 
     a nonzero there for now. */
  device->global_mem_size = 1;
  pocl_topology_detect_device_info(device);
  pocl_cpuinfo_detect_device_info(device);
  pocl_basic_set_buffer_image_limits(device);

  /* The basic driver represents only one "compute unit" as
     it doesn't exploit multiple hardware threads. Multiple
     basic devices can be still used for task level parallelism 
     using multiple OpenCL devices. */
  device->max_compute_units = 1;

  if(!strcmp(device->llvm_cpu, "(unknown)"))
    device->llvm_cpu = NULL;

  // work-around LLVM bug where sizeof(long)=4
  #ifdef _CL_DISABLE_LONG
  device->has_64bit_long=0;
  #endif
}

static void *
pocl_basic_malloc (void *device_data, cl_mem_flags flags,
		    size_t size, void *host_ptr)
{
  void *b;

  if (flags & CL_MEM_COPY_HOST_PTR)
    {
      b = pocl_memalign_alloc(MAX_EXTENDED_ALIGNMENT, size);
      if (b != NULL)
        {
          memcpy(b, host_ptr, size);
          return b;
        }
      
      return NULL;
    }
  
  if (flags & CL_MEM_USE_HOST_PTR && host_ptr != NULL)
    {
      return host_ptr;
    }
  b = pocl_memalign_alloc(MAX_EXTENDED_ALIGNMENT, size);
  if (b != NULL)
    return b;
  
  return NULL;
}

cl_int
pocl_basic_alloc_mem_obj (cl_device_id device, cl_mem mem_obj)
{
  void *b = NULL;
  cl_mem_flags flags = mem_obj->flags;

  /* if memory for this global memory is not yet allocated -> do it */
  if (mem_obj->device_ptrs[device->global_mem_id].mem_ptr == NULL)
    {
      if (flags & CL_MEM_USE_HOST_PTR)
        {
          // mem_host_ptr must be non-NULL
          assert(mem_obj->mem_host_ptr != NULL);
          b = mem_obj->mem_host_ptr;
        }
      else
        {
          b = pocl_memalign_alloc(MAX_EXTENDED_ALIGNMENT, mem_obj->size);
          if (b == NULL)
            return CL_MEM_OBJECT_ALLOCATION_FAILURE;
        }

      if (flags & CL_MEM_COPY_HOST_PTR)
        {
          // mem_host_ptr must be non-NULL
          assert(mem_obj->mem_host_ptr != NULL);
          memcpy (b, mem_obj->mem_host_ptr, mem_obj->size);
        }

      mem_obj->device_ptrs[device->global_mem_id].mem_ptr = b;
      mem_obj->device_ptrs[device->global_mem_id].global_mem_id = 
        device->global_mem_id;
    }
  /* copy already allocated global mem info to devices own slot */
  mem_obj->device_ptrs[device->dev_id] = 
    mem_obj->device_ptrs[device->global_mem_id];

  return CL_SUCCESS;
}

void
pocl_basic_free (void *data, cl_mem_flags flags, void *ptr)
{
  if (flags & CL_MEM_USE_HOST_PTR)
    return;
  
  POCL_MEM_FREE(ptr);
}

void
pocl_basic_read (void *data, void *host_ptr, const void *device_ptr, 
                 size_t offset, size_t cb)
{
  if (host_ptr == device_ptr)
    return;

  memcpy (host_ptr, (char*)device_ptr + offset, cb);
}

void
pocl_basic_write (void *data, const void *host_ptr, void *device_ptr, 
                  size_t offset, size_t cb)
{
  if (host_ptr == device_ptr)
    return;

  memcpy ((char*)device_ptr + offset, host_ptr, cb);
}


void
pocl_basic_run 
(void *data, 
 _cl_command_node* cmd)
{
  struct data *d;
  struct pocl_argument *al;
  size_t x, y, z;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  struct pocl_context *pc = &cmd->command.run.pc;

  assert (data != NULL);
  d = (struct data *) data;

  d->current_kernel = kernel;

  void **arguments = (void**)malloc(
      sizeof(void*) * (kernel->num_args + kernel->num_locals)
    );

  /* Process the kernel arguments. Convert the opaque buffer
     pointers to real device pointers, allocate dynamic local 
     memory buffers, etc. */
  for (i = 0; i < kernel->num_args; ++i)
    {
      al = &(cmd->command.run.arguments[i]);
      if (kernel->arg_info[i].is_local)
        {
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = pocl_basic_malloc(data, 0, al->size, NULL);
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_POINTER)
        {
          /* It's legal to pass a NULL pointer to clSetKernelArguments. In 
             that case we must pass the same NULL forward to the kernel.
             Otherwise, the user must have created a buffer with per device
             pointers stored in the cl_mem. */
          if (al->value == NULL)
            {
              arguments[i] = malloc (sizeof (void *));
              *(void **)arguments[i] = NULL;
            }
          else
            arguments[i] = &((*(cl_mem *) (al->value))->device_ptrs[cmd->device->dev_id].mem_ptr);
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
          dev_image_t di;
          fill_dev_image_t (&di, al, cmd->device);

          void* devptr = pocl_basic_malloc (data, 0, sizeof(dev_image_t), NULL);
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = devptr; 
          pocl_basic_write (data, &di, devptr, 0, sizeof(dev_image_t));
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          dev_sampler_t ds;
          fill_dev_sampler_t(&ds, al);
          
          void* devptr = pocl_basic_malloc (data, 0, sizeof(dev_sampler_t), NULL);
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = devptr;
          pocl_basic_write (data, &ds, devptr, 0, sizeof(dev_sampler_t));
        }
      else
        {
          arguments[i] = al->value;
        }
    }
  for (i = kernel->num_args;
       i < kernel->num_args + kernel->num_locals;
       ++i)
    {
      al = &(cmd->command.run.arguments[i]);
      arguments[i] = malloc (sizeof (void *));
      *(void **)(arguments[i]) = pocl_basic_malloc (data, 0, al->size, NULL);
    }

  for (z = 0; z < pc->num_groups[2]; ++z)
    {
      for (y = 0; y < pc->num_groups[1]; ++y)
        {
          for (x = 0; x < pc->num_groups[0]; ++x)
            {
              pc->group_id[0] = x;
              pc->group_id[1] = y;
              pc->group_id[2] = z;

              cmd->command.run.wg (arguments, pc);

            }
        }
    }
  for (i = 0; i < kernel->num_args; ++i)
    {
      if (kernel->arg_info[i].is_local)
        {
          pocl_basic_free (data, 0, *(void **)(arguments[i]));
          POCL_MEM_FREE(arguments[i]);
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_IMAGE ||
                kernel->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          pocl_basic_free (data, 0, *(void **)(arguments[i]));
          POCL_MEM_FREE(arguments[i]);
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_POINTER && *(void**)arguments[i] == NULL)
        {
          POCL_MEM_FREE(arguments[i]);
        }
    }
  for (i = kernel->num_args;
       i < kernel->num_args + kernel->num_locals;
       ++i)
    {
      pocl_basic_free(data, 0, *(void **)(arguments[i]));
      POCL_MEM_FREE(arguments[i]);
    }
  free(arguments);
}

void
pocl_basic_run_native 
(void *data, 
 _cl_command_node* cmd)
{
  cmd->command.native.user_func(cmd->command.native.args);
}

void
pocl_basic_copy (void *data, const void *src_ptr, size_t src_offset, 
                 void *__restrict__ dst_ptr, size_t dst_offset, size_t cb)
{
  if (src_ptr == dst_ptr)
    return;
  
  memcpy ((char*)dst_ptr + dst_offset, (char*)src_ptr + src_offset, cb);
}

void
pocl_basic_copy_rect (void *data,
                      const void *__restrict const src_ptr,
                      void *__restrict__ const dst_ptr,
                      const size_t *__restrict__ const src_origin,
                      const size_t *__restrict__ const dst_origin, 
                      const size_t *__restrict__ const region,
                      size_t const src_row_pitch,
                      size_t const src_slice_pitch,
                      size_t const dst_row_pitch,
                      size_t const dst_slice_pitch)
{
  char const *__restrict const adjusted_src_ptr = 
    (char const*)src_ptr +
    src_origin[0] + src_row_pitch * src_origin[1] + src_slice_pitch * src_origin[2];
  char *__restrict__ const adjusted_dst_ptr = 
    (char*)dst_ptr +
    dst_origin[0] + dst_row_pitch * dst_origin[1] + dst_slice_pitch * dst_origin[2];
  
  size_t j, k;

  /* TODO: handle overlaping regions */
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      memcpy (adjusted_dst_ptr + dst_row_pitch * j + dst_slice_pitch * k,
              adjusted_src_ptr + src_row_pitch * j + src_slice_pitch * k,
              region[0]);
}

void
pocl_basic_write_rect (void *data,
                       const void *__restrict__ const host_ptr,
                       void *__restrict__ const device_ptr,
                       const size_t *__restrict__ const buffer_origin,
                       const size_t *__restrict__ const host_origin, 
                       const size_t *__restrict__ const region,
                       size_t const buffer_row_pitch,
                       size_t const buffer_slice_pitch,
                       size_t const host_row_pitch,
                       size_t const host_slice_pitch)
{
  char *__restrict const adjusted_device_ptr = 
    (char*)device_ptr +
    buffer_origin[0] + buffer_row_pitch * buffer_origin[1] + buffer_slice_pitch * buffer_origin[2];
  char const *__restrict__ const adjusted_host_ptr = 
    (char const*)host_ptr +
    host_origin[0] + host_row_pitch * host_origin[1] + host_slice_pitch * host_origin[2];
  
  size_t j, k;

  /* TODO: handle overlaping regions */
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      memcpy (adjusted_device_ptr + buffer_row_pitch * j + buffer_slice_pitch * k,
              adjusted_host_ptr + host_row_pitch * j + host_slice_pitch * k,
              region[0]);
}

void
pocl_basic_read_rect (void *data,
                      void *__restrict__ const host_ptr,
                      void *__restrict__ const device_ptr,
                      const size_t *__restrict__ const buffer_origin,
                      const size_t *__restrict__ const host_origin, 
                      const size_t *__restrict__ const region,
                      size_t const buffer_row_pitch,
                      size_t const buffer_slice_pitch,
                      size_t const host_row_pitch,
                      size_t const host_slice_pitch)
{
  char const *__restrict const adjusted_device_ptr = 
    (char const*)device_ptr +
    buffer_origin[2] * buffer_slice_pitch + buffer_origin[1] * buffer_row_pitch + buffer_origin[0];
  char *__restrict__ const adjusted_host_ptr = 
    (char*)host_ptr +
    host_origin[2] * host_slice_pitch + host_origin[1] * host_row_pitch + host_origin[0];
  
  size_t j, k;
  
  /* TODO: handle overlaping regions */
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      memcpy (adjusted_host_ptr + host_row_pitch * j + host_slice_pitch * k,
              adjusted_device_ptr + buffer_row_pitch * j + buffer_slice_pitch * k,
              region[0]);
}

/* origin and region must be in original shape unlike in copy/read/write_rect()
 */
void
pocl_basic_fill_rect (void *data,
                      void *__restrict__ const device_ptr,
                      const size_t *__restrict__ const buffer_origin,
                      const size_t *__restrict__ const region,
                      size_t const buffer_row_pitch,
                      size_t const buffer_slice_pitch,
                      void *fill_pixel,
                      size_t pixel_size)                    
{
  char *__restrict const adjusted_device_ptr = (char*)device_ptr 
    + buffer_origin[0] * pixel_size 
    + buffer_row_pitch * buffer_origin[1] 
    + buffer_slice_pitch * buffer_origin[2];
    
  size_t i, j, k;

  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      for (i = 0; i < region[0]; ++i)
        memcpy (adjusted_device_ptr + pixel_size * i 
                + buffer_row_pitch * j 
                + buffer_slice_pitch * k, fill_pixel, pixel_size);
}

void *
pocl_basic_map_mem (void *data, void *buf_ptr, 
                      size_t offset, size_t size,
                      void *host_ptr) 
{
  /* All global pointers of the pthread/CPU device are in 
     the host address space already, and up to date. */
  if (host_ptr != NULL) return host_ptr;
  return (char*)buf_ptr + offset;
}

void
pocl_basic_uninit (cl_device_id device)
{
  struct data *d = (struct data*)device->data;
  POCL_MEM_FREE(d);
  device->data = NULL;
}

cl_ulong
pocl_basic_get_timer_value (void *data) 
{
#ifndef _MSC_VER
  struct timeval current;
  gettimeofday(&current, NULL);  
  return (current.tv_sec * 1000000 + current.tv_usec)*1000;
#else
  FILETIME ft;
  cl_ulong tmpres = 0;
  GetSystemTimeAsFileTime(&ft);
  tmpres |= ft.dwHighDateTime;
  tmpres <<= 32;
  tmpres |= ft.dwLowDateTime;
  tmpres -= 11644473600000000Ui64;
  tmpres /= 10;
  return tmpres;
#endif
}

cl_int 
pocl_basic_get_supported_image_formats (cl_mem_flags flags,
                                        const cl_image_format **image_formats,
                                        cl_uint *num_img_formats)
{
    if (num_img_formats == NULL || image_formats == NULL)
      return CL_INVALID_VALUE;
  
    *num_img_formats = sizeof(supported_image_formats)/sizeof(cl_image_format);
    *image_formats = supported_image_formats;
    
    return CL_SUCCESS; 
}

typedef struct compiler_cache_item compiler_cache_item;
struct compiler_cache_item
{
  char *tmp_dir;
  char *function_name;
  pocl_workgroup wg;
  compiler_cache_item *next;
};

static compiler_cache_item *compiler_cache;
static pocl_lock_t compiler_cache_lock;

void check_compiler_cache (_cl_command_node *cmd)
{
  char workgroup_string[WORKGROUP_STRING_LENGTH];
  lt_dlhandle dlhandle;
  compiler_cache_item *ci = NULL;
  
  if (compiler_cache == NULL)
    POCL_INIT_LOCK (compiler_cache_lock);

  POCL_LOCK (compiler_cache_lock);
  LL_FOREACH (compiler_cache, ci)
    {
      if (strcmp (ci->tmp_dir, cmd->command.run.tmp_dir) == 0 &&
          strcmp (ci->function_name, 
                  cmd->command.run.kernel->name) == 0)
        {
          POCL_UNLOCK (compiler_cache_lock);
          cmd->command.run.wg = ci->wg;
          return;
        }
    }
  cl_program program = cmd->command.run.kernel->program;

  void* cache_lock = pocl_cache_acquire_writer_lock(program, cmd->device);
  assert(cache_lock);

  ci = (compiler_cache_item*) malloc (sizeof (compiler_cache_item));
  ci->next = NULL;
  ci->tmp_dir = strdup(cmd->command.run.tmp_dir);
  ci->function_name = strdup (cmd->command.run.kernel->name);
  const char* module_fn = llvm_codegen (cmd->command.run.tmp_dir,
                                        cmd->command.run.kernel,
                                        cmd->device);
  dlhandle = lt_dlopen (module_fn);
  if (dlhandle == NULL)
    {
      printf ("pocl error: lt_dlopen(\"%s\") failed with '%s'.\n", 
              module_fn, lt_dlerror());
      printf ("note: missing symbols in the kernel binary might be" 
              "reported as 'file not found' errors.\n");
      abort();
    }
  snprintf (workgroup_string, WORKGROUP_STRING_LENGTH,
            "_pocl_launcher_%s_workgroup", cmd->command.run.kernel->name);
  cmd->command.run.wg = ci->wg = 
    (pocl_workgroup) lt_dlsym (dlhandle, workgroup_string);

  pocl_cache_release_lock(cache_lock);
  LL_APPEND (compiler_cache, ci);
  POCL_UNLOCK (compiler_cache_lock);

}

void
pocl_basic_compile_submitted_kernels (_cl_command_node *cmd)
{
  if (cmd->type == CL_COMMAND_NDRANGE_KERNEL)
    check_compiler_cache (cmd);

}
