/* common.c - common code that can be reused between device driver 
              implementations

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos and
                           Pekka Jääskeläinen / Tampere Univ. of Technology
   
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
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _MSC_VER
#  include <sys/time.h>
#  include <sys/resource.h>
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include "config.h"

#include "pocl_image_util.h"
#include "pocl_file_util.h"
#include "pocl_cache.h"
#include "devices.h"
#include "pocl_mem_management.h"
#include "pocl_runtime_config.h"
#include "pocl_llvm.h"
#include "pocl_debug.h"

#include "_kernel_constants.h"

#define COMMAND_LENGTH 2048

/**
 * Generate code from the final bitcode using the LLVM
 * tools.
 *
 * Uses an existing (cached) one, if available.
 *
 * @param tmpdir The directory of the work-group function bitcode.
 * @param return the generated binary filename.
 */
const char*
llvm_codegen (const char* tmpdir, cl_kernel kernel, cl_device_id device) {

  char command[COMMAND_LENGTH];
  char bytecode[POCL_FILENAME_LENGTH];
  char objfile[POCL_FILENAME_LENGTH];

  char* module = malloc(strlen(tmpdir) + strlen(kernel->name) +
                        strlen("/.so") + 1);

  int error;

  error = snprintf(module, POCL_FILENAME_LENGTH,
                   "%s/%s.so", tmpdir, kernel->name);

  assert (error >= 0);

  error = snprintf(objfile, POCL_FILENAME_LENGTH,
                   "%s/%s.so.o", tmpdir, kernel->name);
  assert (error >= 0);

  if (pocl_exists(module))
    return module;

  void* write_lock = pocl_cache_acquire_writer_lock(kernel->program, device);
  assert(write_lock);

      error = snprintf (bytecode, POCL_FILENAME_LENGTH,
                        "%s%s", tmpdir, POCL_PARALLEL_BC_FILENAME);
      assert (error >= 0);
      
      error = pocl_llvm_codegen( kernel, device, bytecode, objfile);
      assert (error == 0);

      // clang is used as the linker driver in LINK_CMD
      error = snprintf (command, COMMAND_LENGTH,
#ifdef OCS_AVAILABLE
#ifndef POCL_ANDROID
            LINK_CMD " " HOST_CLANG_FLAGS " " HOST_LD_FLAGS " -o %s %s",
#else
            POCL_ANDROID_PREFIX"/bin/ld " HOST_LD_FLAGS " -o %s %s ",
#endif
#else
            "",
#endif
            module, objfile);
      assert (error >= 0);

      POCL_MSG_PRINT_INFO ("executing [%s]\n", command);
      error = system (command);
      assert (error == 0);

      /* Save space in kernel cache */
      if (!pocl_get_bool_option("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES", 0))
        {
          pocl_remove(objfile);
          pocl_remove(bytecode);
        }

  pocl_cache_release_lock(write_lock);

  return module;
}

/**
 * Populates the device specific image data structure used by kernel
 * from given kernel image argument
 */
void fill_dev_image_t (dev_image_t* di, struct pocl_argument* parg, 
                       cl_device_id device)
{
  cl_mem mem = *(cl_mem *)parg->value;
  di->data = (mem->device_ptrs[device->dev_id].mem_ptr);
  di->width = mem->image_width;
  di->height = mem->image_height;
  di->depth = mem->image_depth;
  di->row_pitch = mem->image_row_pitch;
  di->slice_pitch = mem->image_slice_pitch;
  di->order = mem->image_channel_order;
  di->data_type = mem->image_channel_data_type;
  pocl_get_image_information (mem->image_channel_order,
                              mem->image_channel_data_type, &(di->num_channels),
                              &(di->elem_size));
}

/**
 * Populates the device specific sampler data structure used by kernel
 * from given kernel sampler argument
 */
void fill_dev_sampler_t (dev_sampler_t *ds, struct pocl_argument *parg)
{
  cl_sampler_t sampler = *(cl_sampler_t *)parg->value;

  *ds = 0;
  *ds |= sampler.normalized_coords == CL_TRUE ? CLK_NORMALIZED_COORDS_TRUE :
      CLK_NORMALIZED_COORDS_FALSE;

  switch (sampler.addressing_mode) {
    case CL_ADDRESS_NONE:
      *ds |= CLK_ADDRESS_NONE; break;
    case CL_ADDRESS_CLAMP_TO_EDGE:
      *ds |= CLK_ADDRESS_CLAMP_TO_EDGE; break;
    case CL_ADDRESS_CLAMP:
      *ds |= CLK_ADDRESS_CLAMP; break;
    case CL_ADDRESS_REPEAT:
      *ds |= CLK_ADDRESS_REPEAT; break;
    case CL_ADDRESS_MIRRORED_REPEAT:
      *ds |= CLK_ADDRESS_MIRRORED_REPEAT; break;
  }

  switch (sampler.filter_mode) {
    case CL_FILTER_NEAREST:
      *ds |= CLK_FILTER_NEAREST; break;
    case CL_FILTER_LINEAR :
      *ds |= CLK_FILTER_LINEAR; break;
  }
}

void*
pocl_memalign_alloc(size_t align_width, size_t size)
{
  void *ptr;
  int status;

#ifndef POCL_ANDROID
  status = posix_memalign(&ptr, align_width, size);
  return ((status == 0)? ptr: (void*)NULL);
#else
  ptr = memalign(align_width, size);
  return ptr;
#endif
}


#define MIN_MAX_MEM_ALLOC_SIZE (128*1024*1024)

/* accounting object for the main memory */
static pocl_global_mem_t system_memory;

void pocl_setup_device_for_system_memory(cl_device_id device)
{
  /* set up system memory limits, if required */
  if (system_memory.total_alloc_limit == 0)
  {
      /* global_mem_size contains the entire memory size,
       * and we need to leave some available for OS & other programs
       * this sets it to 3/4 for systems with <=7gig mem,
       * for >7 it sets to (total-2gigs)
       */
      size_t alloc_limit = device->global_mem_size;
      if ((alloc_limit >> 20) > (7 << 10))
        system_memory.total_alloc_limit = alloc_limit - (size_t)(1 << 31);
      else
        {
          size_t temp = (alloc_limit >> 2);
          system_memory.total_alloc_limit = alloc_limit - temp;
        }

      system_memory.max_ever_allocated =
          system_memory.currently_allocated = 0;
  }

  device->global_mem_size = system_memory.total_alloc_limit;
  if (device->global_mem_size < MIN_MAX_MEM_ALLOC_SIZE)
    POCL_ABORT("Not enough memory to run on this device.\n");

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
  else
#endif
    alloc_limit = MIN_MAX_MEM_ALLOC_SIZE;

  if (alloc_limit > device->global_mem_size)
    alloc_limit = device->global_mem_size;

  if (alloc_limit < MIN_MAX_MEM_ALLOC_SIZE)
    alloc_limit = MIN_MAX_MEM_ALLOC_SIZE;

  // set up device properties..
  device->global_memory = &system_memory;
  device->max_mem_alloc_size = alloc_limit;

  // TODO in theory now if alloc_limit was > rlim_cur and < rlim_max
  // we should try and setrlimit to alloc_limit, or allocations might fail
}


/* set maximum allocation sizes for buffers and images */
void
pocl_set_buffer_image_limits(cl_device_id device)
{
  pocl_setup_device_for_system_memory(device);
  /* these aren't set up in pocl_setup_device_for_system_memory,
   * because some devices (HSA) set them up themselves */
  device->local_mem_size = device->max_constant_buffer_size =
      device->max_mem_alloc_size;

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

void* pocl_memalign_alloc_global_mem(cl_device_id device, size_t align, size_t size)
{
  pocl_global_mem_t *mem = device->global_memory;
  if ((mem->total_alloc_limit - mem->currently_allocated) < size)
    return NULL;

  void* ptr = pocl_memalign_alloc(align, size);
  if (!ptr)
    return NULL;

  mem->currently_allocated += size;
  if (mem->max_ever_allocated < mem->currently_allocated)
    mem->max_ever_allocated = mem->currently_allocated;

  assert(mem->currently_allocated <= mem->total_alloc_limit);
  return ptr;
}

void pocl_free_global_mem(cl_device_id device, void* ptr, size_t size)
{
  pocl_global_mem_t *mem = device->global_memory;

  assert(mem->currently_allocated >= size);
  mem->currently_allocated -= size;

  POCL_MEM_FREE(ptr);
}

void pocl_print_system_memory_stats()
{
  POCL_MSG_PRINT("MEM STATS:\n", "",
  "____ Total available system memory  : %10zu KB\n"
  " ____ Currently used system memory   : %10zu KB\n"
  " ____ Max used system memory         : %10zu KB\n",
  system_memory.total_alloc_limit >> 10,
  system_memory.currently_allocated >> 10,
  system_memory.max_ever_allocated >> 10);
}
