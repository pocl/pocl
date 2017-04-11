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
#include "pocl_shared.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <utlist.h>
#include <assert.h>

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
#include "pocl_util.h"
#include "pocl_cache.h"
#include "devices.h"
#include "pocl_mem_management.h"
#include "pocl_runtime_config.h"
#include "pocl_debug.h"

#ifdef OCS_AVAILABLE
#include "pocl_llvm.h"
#endif

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

#ifdef OCS_AVAILABLE
char*
llvm_codegen (const char* tmpdir, cl_kernel kernel, cl_device_id device) {

  char command[COMMAND_LENGTH];
  char bytecode[POCL_FILENAME_LENGTH];
  char objfile[POCL_FILENAME_LENGTH];
  /* strlen of / .so 4+1 */
  int file_name_alloc_size = 
    min(POCL_FILENAME_LENGTH, strlen(tmpdir) + strlen(kernel->name) + 5);
  char* module = (char*) malloc(file_name_alloc_size); 
  /* To avoid corrupted .so files, create a tmp file first
     and then rename it. */
  char tmp_module[file_name_alloc_size + 4]; /* .tmp postfix */
  int error;

  error = snprintf(module, POCL_FILENAME_LENGTH,
                   "%s/%s.so", tmpdir, kernel->name);

  assert (error >= 0);

  error = snprintf(objfile, POCL_FILENAME_LENGTH,
                   "%s/%s.so.o", tmpdir, kernel->name);
  assert (error >= 0);

  if (pocl_exists(module))
    return module;

  memcpy (tmp_module, module, file_name_alloc_size);
  strcat (tmp_module, ".tmp");

  void* write_lock = pocl_cache_acquire_writer_lock(kernel->program, device);
  assert(write_lock);

  error = snprintf (bytecode, POCL_FILENAME_LENGTH,
		    "%s%s", tmpdir, POCL_PARALLEL_BC_FILENAME);
  assert (error >= 0);

  error = pocl_llvm_codegen( kernel, device, bytecode, objfile);
  assert (error == 0);

  /* clang is used as the linker driver in LINK_CMD */
  error = snprintf (command, COMMAND_LENGTH,
#ifndef POCL_ANDROID
#ifdef OCS_AVAILABLE
                    CLANGXX " " HOST_CLANG_FLAGS " " HOST_LD_FLAGS " -o %s %s",
#else
                    LINK_COMMAND " " HOST_LD_FLAGS " -o %s %s",
#endif
#else
                    POCL_ANDROID_PREFIX"/bin/ld " HOST_LD_FLAGS " -o %s %s ",
#endif
                    tmp_module, objfile);
  assert (error >= 0);

  POCL_MSG_PRINT_INFO ("executing [%s]\n", command);
  error = system (command);
  assert (error == 0);

  error = snprintf (command, COMMAND_LENGTH, "mv %s %s", tmp_module, module);
  assert (error >= 0);
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
#endif


/**
 * Populates the device specific image data structure used by kernel
 * from given kernel image argument
 */
void
fill_dev_image_t (dev_image_t* di, struct pocl_argument* parg,
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

void
pocl_copy_mem_object (cl_device_id dest_dev, cl_mem dest,
                      size_t dest_offset,
                      cl_device_id source_dev, cl_mem source,
                      size_t source_offset, size_t cb)
{
  /* if source_dev is NULL -> src and dest dev must be the same */
  cl_device_id src_dev = (source_dev) ? source_dev : dest_dev;

  /* if source and destination are on the same global mem  */
  if (src_dev->global_mem_id == dest_dev->global_mem_id)
    {
      src_dev->ops->copy 
        (dest_dev->data, 
         source->device_ptrs[src_dev->dev_id].mem_ptr, source_offset,
         dest->device_ptrs[dest_dev->dev_id].mem_ptr, dest_offset, 
         cb);
    }
  else
    {
      void* tofree = NULL;
      void* tmp = NULL;
      if (source->flags & CL_MEM_USE_HOST_PTR)
        tmp = source->mem_host_ptr;
      else if (dest->flags & CL_MEM_USE_HOST_PTR)
        tmp = dest->mem_host_ptr;
      else
        {
          tmp = malloc (dest->size);
          tofree = tmp;
        }
      
      src_dev->ops->read 
        (src_dev->data, tmp, 
         source->device_ptrs[src_dev->dev_id].mem_ptr, source_offset, 
         cb);
      dest_dev->ops->write 
        (dest_dev->data, tmp, 
         dest->device_ptrs[dest_dev->dev_id].mem_ptr, dest_offset,
         cb);
      free (tofree);
    }
  return;
}

void
pocl_migrate_mem_objects (_cl_command_node * volatile node)
{
  size_t i;
  cl_mem *mem_objects = node->command.migrate.mem_objects;
  
  for (i = 0; i < node->command.migrate.num_mem_objects; ++i)
    {
      pocl_copy_mem_object (node->device,
                            mem_objects[i], 0,
                            node->command.migrate.source_devices[i], 
                            mem_objects[i], 0, mem_objects[i]->size);
      
      return;
    }
}

void
pocl_ndrange_node_cleanup(_cl_command_node *node)
{
  cl_uint i;
  free (node->command.run.tmp_dir);
  for (i = 0; i < node->command.run.kernel->num_args + 
       node->command.run.kernel->num_locals; ++i)
    {
      pocl_aligned_free (node->command.run.arguments[i].value);
    }
  free (node->command.run.arguments);

  POname(clReleaseKernel)(node->command.run.kernel);
}

void
pocl_native_kernel_cleanup(_cl_command_node *node)
{
  free (node->command.native.mem_list);
  free (node->command.native.args);
}

void
pocl_mem_objs_cleanup (cl_event event)
{
  int i;
  for (i = 0; i < event->num_buffers; ++i)
    {
      assert(event->mem_objs[i] != NULL);
      POCL_LOCK_OBJ (event->mem_objs[i]);
      if (event->mem_objs[i]->latest_event == event)
        event->mem_objs[i]->latest_event = NULL;
      POCL_UNLOCK_OBJ (event->mem_objs[i]);
      POname(clReleaseMemObject) (event->mem_objs[i]);
    }
  free (event->mem_objs);
  event->mem_objs = NULL;
}

/**
 * executes given command.
 */
void
pocl_exec_command (_cl_command_node * volatile node)
{
  unsigned i;
  /* because of POCL_UPDATE_EVENT_ */
  cl_event *event = &(node->event);
  switch (node->type)
    {
    case CL_COMMAND_READ_BUFFER:
      POCL_UPDATE_EVENT_RUNNING(event);
      node->device->ops->read
        (node->device->data, 
         node->command.read.host_ptr, 
         node->command.read.device_ptr,
         node->command.read.offset,
         node->command.read.cb); 
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "Read Buffer           ");
      break;
    case CL_COMMAND_WRITE_BUFFER:
      POCL_UPDATE_EVENT_RUNNING(event);
      node->device->ops->write
        (node->device->data, 
         node->command.write.host_ptr, 
         node->command.write.device_ptr,
         node->command.write.offset, 
         node->command.write.cb);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "Write Buffer          ");
      break;
    case CL_COMMAND_COPY_BUFFER:
      POCL_UPDATE_EVENT_RUNNING(event);
      pocl_copy_mem_object (node->command.copy.dst_dev, 
                            node->command.copy.dst_buffer,
                            node->command.copy.dst_offset,
                            node->command.copy.src_dev,
                            node->command.copy.src_buffer,
                            node->command.copy.src_offset, 
                            node->command.copy.cb);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "Copy Buffer           ");
      break;
    case CL_COMMAND_MIGRATE_MEM_OBJECTS:
      POCL_UPDATE_EVENT_RUNNING(event);
      pocl_migrate_mem_objects (node);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "Migrate Buffer        ");
      break;
    case CL_COMMAND_MAP_IMAGE:
    case CL_COMMAND_MAP_BUFFER: 
      POCL_UPDATE_EVENT_RUNNING(event);
      pocl_map_mem_cmd (node->device, node->command.map.buffer, 
                        node->command.map.mapping);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "Map Image/Buffer      ");
      break;
    case CL_COMMAND_WRITE_IMAGE:
      POCL_UPDATE_EVENT_RUNNING(event);
      node->device->ops->write_rect
        (node->device->data,
         node->command.write_image.host_ptr,
         node->command.write_image.device_ptr,
         node->command.write_image.origin,
         node->command.write_image.origin,
         node->command.write_image.region,
         node->command.write_image.b_rowpitch,
         node->command.write_image.b_slicepitch,
         node->command.write_image.b_rowpitch,
         node->command.write_image.b_slicepitch);
      POCL_UPDATE_EVENT_COMPLETE(event);
      break;
    case CL_COMMAND_WRITE_BUFFER_RECT:
      POCL_UPDATE_EVENT_RUNNING(event);
      node->device->ops->write_rect
        (node->device->data,
         node->command.write_image.host_ptr,
         node->command.write_image.device_ptr,
         node->command.write_image.origin,
         node->command.write_image.h_origin,
         node->command.write_image.region,
         node->command.write_image.b_rowpitch,
         node->command.write_image.b_slicepitch,
         node->command.write_image.h_rowpitch,
         node->command.write_image.h_slicepitch);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "Write Image           ");
      break;
    case CL_COMMAND_READ_IMAGE:
      POCL_UPDATE_EVENT_RUNNING(event);
      node->device->ops->read_rect
        (node->device->data, node->command.read_image.host_ptr,
         node->command.read_image.device_ptr,
         node->command.read_image.origin,
         node->command.read_image.origin,
         node->command.read_image.region,
         node->command.read_image.b_rowpitch,
         node->command.read_image.b_slicepitch,
         node->command.read_image.b_rowpitch,
         node->command.read_image.b_slicepitch);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "Read Image            ");
      break;
    case CL_COMMAND_READ_BUFFER_RECT:
      POCL_UPDATE_EVENT_RUNNING(event);
      node->device->ops->read_rect
        (node->device->data, node->command.read_image.host_ptr,
         node->command.read_image.device_ptr,
         node->command.read_image.origin,
         node->command.read_image.h_origin,
         node->command.read_image.region,
         node->command.read_image.b_rowpitch,
         node->command.read_image.b_slicepitch,
         node->command.read_image.h_rowpitch,
         node->command.read_image.h_slicepitch);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "Read Buffer Rect      ");
      break;
    case CL_COMMAND_COPY_BUFFER_RECT:
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
    case CL_COMMAND_COPY_IMAGE:
      POCL_UPDATE_EVENT_RUNNING(event);
      cl_device_id src_dev = node->command.copy_image.src_device;
      cl_mem src_buf = node->command.copy_image.src_buffer;
      cl_device_id dst_dev = node->command.copy_image.dst_device;
      cl_mem dst_buf = node->command.copy_image.dst_buffer;

      /* if source and destination are in the same global mem  */
      if (src_dev->global_mem_id == dst_dev->global_mem_id)
        {
          node->device->ops->copy_rect
            (node->device->data,
             src_buf->device_ptrs[src_dev->dev_id].mem_ptr,
             dst_buf->device_ptrs[dst_dev->dev_id].mem_ptr,
             node->command.copy_image.src_origin,
             node->command.copy_image.dst_origin,
             node->command.copy_image.region,
             node->command.copy_image.src_rowpitch,
             node->command.copy_image.src_slicepitch,
             node->command.copy_image.dst_rowpitch,
             node->command.copy_image.dst_slicepitch);
        }
      /* if source and destination are in different global mem
         data needs to be read to the host memory and then written to
         destination device */
      else
        {
          size_t size = node->command.copy_image.region[0]
            * node->command.copy_image.region[1]
            * node->command.copy_image.region[2];
          void *tmp = malloc (size);

          /* origin and slice pitch for tmp buffer */
          const size_t null_origin[3] = {0, 0, 0};
          size_t tmp_rowpitch =
            node->command.copy_image.region[0];
          size_t tmp_slicepitch =
            node->command.copy_image.region[0]
            * node->command.copy_image.region[1];

          src_dev->ops->read_rect
            (src_dev->data,
             tmp,
             src_buf->device_ptrs[src_dev->dev_id].mem_ptr,
             node->command.copy_image.src_origin,
             null_origin,
             node->command.copy_image.region,
             node->command.copy_image.src_rowpitch,
             node->command.copy_image.src_slicepitch,
             tmp_rowpitch,
             tmp_slicepitch);

          dst_dev->ops->write_rect
            (dst_dev->data,
             tmp,
             dst_buf->device_ptrs[dst_dev->dev_id].mem_ptr,
             node->command.copy_image.dst_origin,
             null_origin,
             node->command.copy_image.region,
             node->command.copy_image.dst_rowpitch,
             node->command.copy_image.dst_slicepitch,
             tmp_rowpitch,
             tmp_slicepitch);
          free (tmp);
        }
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "Copy Buffer Rect      ");
      break;
    case CL_COMMAND_UNMAP_MEM_OBJECT:
      POCL_UPDATE_EVENT_RUNNING(event);
      if ((node->command.unmap.memobj)->flags & 
          (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR))
        {
          /* TODO: should we ensure the device global region is updated from
             the host memory? How does the specs define it,
             can the host_ptr be assumed to point to the host and the
             device accessible memory or just point there until the
             kernel(s) get executed or similar? */
          /* Assume the region is automatically up to date. */
        } else 
        {
          if (node->device->ops->unmap_mem != NULL)        
            node->device->ops->unmap_mem
              (node->device->data, 
               (node->command.unmap.mapping)->host_ptr, 
               (node->command.unmap.memobj)->device_ptrs[node->device->dev_id].mem_ptr, 
               (node->command.unmap.mapping)->offset,
               (node->command.unmap.mapping)->size);
        }
      POCL_LOCK_OBJ (node->command.unmap.memobj);
      DL_DELETE((node->command.unmap.memobj)->mappings, 
                node->command.unmap.mapping);
      (node->command.unmap.memobj)->map_count--;
      POCL_UNLOCK_OBJ (node->command.unmap.memobj);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "Unmap Mem obj         ");
      break;
    case CL_COMMAND_NDRANGE_KERNEL:
      POCL_UPDATE_EVENT_RUNNING(event);
      assert (*event == node->event);
      node->device->ops->run(node->command.run.data, node);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "Enqueue NDRange       ");
      pocl_ndrange_node_cleanup(node);
      break;
    case CL_COMMAND_NATIVE_KERNEL:
      POCL_UPDATE_EVENT_RUNNING(event);
      node->device->ops->run_native(node->command.native.data, node);
      pocl_native_kernel_cleanup(node);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "Native Kernel         ");
      break;
    case CL_COMMAND_FILL_IMAGE:
      POCL_UPDATE_EVENT_RUNNING(event);
      node->device->ops->fill_rect 
        (node->command.fill_image.data, 
         node->command.fill_image.device_ptr,
         node->command.fill_image.buffer_origin,
         node->command.fill_image.region,
         node->command.fill_image.rowpitch, 
         node->command.fill_image.slicepitch,
         node->command.fill_image.fill_pixel,
         node->command.fill_image.pixel_size);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "Fill Image            ");
      free(node->command.fill_image.fill_pixel);
      break;
    case CL_COMMAND_FILL_BUFFER:
      POCL_UPDATE_EVENT_RUNNING(event);
      node->device->ops->memfill
        (node->command.memfill.ptr,
         node->command.memfill.size,
         node->command.memfill.offset,
         node->command.memfill.pattern,
         node->command.memfill.pattern_size);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "Fill Buffer           ");
      pocl_aligned_free(node->command.memfill.pattern);
      break;
    case CL_COMMAND_MARKER:
      POCL_UPDATE_EVENT_RUNNING(event);
      POCL_UPDATE_EVENT_COMPLETE(event);
      break;
    case CL_COMMAND_BARRIER:
      POCL_UPDATE_EVENT_RUNNING(event);
      POCL_UPDATE_EVENT_COMPLETE(event);
      break;
    case CL_COMMAND_SVM_FREE:
      POCL_UPDATE_EVENT_RUNNING(event);
      if (node->command.svm_free.pfn_free_func)
        node->command.svm_free.pfn_free_func(
           node->command.svm_free.queue,
           node->command.svm_free.num_svm_pointers,
           node->command.svm_free.svm_pointers,
           node->command.svm_free.data);
      else
        for (i=0; i < node->command.svm_free.num_svm_pointers; i++)
          node->device->ops->free_ptr(node->device,
                                      node->command.svm_free.svm_pointers[i]);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "SVM Free              ");
      break;
    case CL_COMMAND_SVM_MAP:
      POCL_UPDATE_EVENT_RUNNING(event);
      if (DEVICE_MMAP_IS_NOP(node->device))
        ; // no-op
      else
        node->device->ops->map_mem
          (node->device->data, node->command.svm_map.svm_ptr,
           0, node->command.svm_map.size, NULL);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "SVM Map              ");
      break;
    case CL_COMMAND_SVM_UNMAP:
      POCL_UPDATE_EVENT_RUNNING(event);
      if (DEVICE_MMAP_IS_NOP(node->device))
        ; // no-op
      else
        node->device->ops->unmap_mem
          (node->device->data, NULL,
           node->command.svm_unmap.svm_ptr, 0, 0);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "SVM Unmap             ");
      break;
    case CL_COMMAND_SVM_MEMCPY:
      POCL_UPDATE_EVENT_RUNNING(event);
      node->device->ops->copy(NULL,
                              node->command.svm_memcpy.src, 0,
                              node->command.svm_memcpy.dst, 0,
                              node->command.svm_memcpy.size);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "SVM Memcpy            ");
      break;
    case CL_COMMAND_SVM_MEMFILL:
      POCL_UPDATE_EVENT_RUNNING(event);
      node->device->ops->memfill(
                                 node->command.memfill.ptr,
                                 node->command.memfill.size, 0,
                                 node->command.memfill.pattern,
                                 node->command.memfill.pattern_size);
      POCL_UPDATE_EVENT_COMPLETE(event);
      POCL_DEBUG_EVENT_TIME(event, "SVM MemFill           ");
      pocl_aligned_free(node->command.memfill.pattern);
      break;
    default:
      POCL_ABORT_UNIMPLEMENTED("");
      break;
    }   
  pocl_mem_manager_free_command (node);
}

void
pocl_broadcast (cl_event brc_event)
{
  event_node *target;
  event_node *tmp;
  while ((target = brc_event->notify_list))
    {
      POCL_LOCK_OBJ (target->event);
      /* remove event from wait list */
      LL_FOREACH (target->event->wait_list, tmp)
        {
          if (tmp->event == brc_event)
            {
              LL_DELETE (target->event->wait_list, tmp);
              pocl_mem_manager_free_event_node (tmp);
              break;
            }
        }
      if (target->event->status == CL_SUBMITTED)
        {
          POCL_UNLOCK_OBJ (target->event);
          target->event->command->device->ops->notify 
            (target->event->command->device, target->event);
        }
      else 
        POCL_UNLOCK_OBJ (target->event);
      
      LL_DELETE (brc_event->notify_list, target);
      pocl_mem_manager_free_event_node (target);
    }
}

/**
 * Populates the device specific sampler data structure used by kernel
 * from given kernel sampler argument
 */
void
fill_dev_sampler_t (dev_sampler_t *ds, struct pocl_argument *parg)
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

/* CPU driver stuff */
typedef struct pocl_dlhandle_cache_item pocl_dlhandle_cache_item;
struct pocl_dlhandle_cache_item
{
  char *tmp_dir;
  char *function_name;
  pocl_workgroup wg;
  lt_dlhandle dlhandle;
  pocl_dlhandle_cache_item *next;
  pocl_dlhandle_cache_item *prev;
  volatile int ref_count;
};

static pocl_dlhandle_cache_item *pocl_dlhandle_cache;
static pocl_lock_t pocl_dlhandle_cache_lock;
static pocl_lock_t pocl_llvm_codegen_lock;
static pocl_lock_t pocl_dlhandle_lock;
static int pocl_dlhandle_cache_initialized;

/* only to be called in basic/pthread/<other cpu driver> init */
void
pocl_init_dlhandle_cache ()
{
  if (!pocl_dlhandle_cache_initialized)
    {
      POCL_INIT_LOCK (pocl_dlhandle_cache_lock);
      POCL_INIT_LOCK (pocl_llvm_codegen_lock);
      POCL_INIT_LOCK (pocl_dlhandle_lock);
      pocl_dlhandle_cache_initialized = 1;
   }
}

static int handle_count = 0;
void
pocl_check_dlhandle_cache (_cl_command_node *cmd)
{
  char workgroup_string[256];
  pocl_dlhandle_cache_item *ci = NULL;

  POCL_LOCK (pocl_dlhandle_cache_lock);
  DL_FOREACH (pocl_dlhandle_cache, ci)
    {
      if (strcmp (ci->tmp_dir, cmd->command.run.tmp_dir) == 0 &&
          strcmp (ci->function_name,
                  cmd->command.run.kernel->name) == 0)
        {
          /* move to the front of the line */
          DL_DELETE (pocl_dlhandle_cache, ci);
          DL_PREPEND (pocl_dlhandle_cache, ci);
          ++ci->ref_count;
          POCL_UNLOCK (pocl_dlhandle_cache_lock);
          cmd->command.run.wg = ci->wg;
          return;
        }
    }
  if (handle_count == 128)
    {
      ci = pocl_dlhandle_cache->prev;
      //assert (ci->ref_count == 0);
      DL_DELETE (pocl_dlhandle_cache, ci);
      free (ci->tmp_dir);
      free (ci->function_name);
      assert(!lt_dlclose (ci->dlhandle));
    }
  else
    {
      ++handle_count;
      ci = (pocl_dlhandle_cache_item*) malloc (sizeof (pocl_dlhandle_cache_item));
    }
  POCL_UNLOCK (pocl_dlhandle_cache_lock);
  ci->next = NULL;
  ci->tmp_dir = strdup (cmd->command.run.tmp_dir);
  ci->function_name = strdup (cmd->command.run.kernel->name);
  ci->ref_count = 1;

  char *module_fn = NULL;
  cl_kernel k = cmd->command.run.kernel;
  cl_program p = k->program;
  cl_device_id dev = cmd->device;
  int dev_i = pocl_cl_device_to_index(p, dev);

  if (p->binaries[dev_i] && !p->pocl_binaries[dev_i])
    {
#ifdef OCS_AVAILABLE
      POCL_LOCK (pocl_llvm_codegen_lock);
      module_fn = (char *)llvm_codegen (cmd->command.run.tmp_dir,
                                        cmd->command.run.kernel,
                                        cmd->device);
      POCL_UNLOCK (pocl_llvm_codegen_lock);
      POCL_MSG_PRINT_INFO("Using static WG size binary: %s\n", module_fn);
#else
      POCL_ABORT("pocl built without online compiler support "
                 "cannot compile LLVM IRs to machine code\n");
#endif
    }
  else
    {
      module_fn = malloc (POCL_FILENAME_LENGTH);
      /* First try to find a static WG binary for the local size as they
         are always more efficient than the dynamic ones.  Also, in case
         of reqd_wg_size, there might not be a dynamic sized one at all.  */
      pocl_cache_final_binary_path (module_fn, p, dev_i, k,
                                    cmd->command.run.local_x,
                                    cmd->command.run.local_y,
                                    cmd->command.run.local_z);
      if (!pocl_exists (module_fn))
        {
          pocl_cache_final_binary_path (module_fn, p, dev_i, k, 0, 0, 0);
          if (!pocl_exists (module_fn))
            POCL_ABORT("Dynamic WG size binary does not exist\n");
          POCL_MSG_PRINT_INFO("Using dynamic local size binary: %s\n", module_fn);
        }
      else
        POCL_MSG_PRINT_INFO("Using static local size binary: %s\n", module_fn);
    }

  POCL_LOCK (pocl_dlhandle_lock);
  ci->dlhandle = lt_dlopen (module_fn);
  POCL_UNLOCK (pocl_dlhandle_lock);
  
  if (ci->dlhandle == NULL)
    {
      printf ("pocl error: lt_dlopen(\"%s\") failed with '%s'.\n", 
              module_fn, lt_dlerror());
      printf ("note: missing symbols in the kernel binary might be" 
              " reported as 'file not found' errors.\n");
      abort();
    }
  free(module_fn);

  snprintf (workgroup_string, 256, "_pocl_launcher_%s_workgroup", 
            cmd->command.run.kernel->name);

  POCL_LOCK (pocl_dlhandle_lock);
  cmd->command.run.wg = ci->wg =
    (pocl_workgroup) lt_dlsym (ci->dlhandle, workgroup_string);
  POCL_UNLOCK (pocl_dlhandle_lock);

  assert (cmd->command.run.wg != NULL);

  POCL_LOCK (pocl_dlhandle_cache_lock);
  assert (handle_count <= 128);
  DL_PREPEND (pocl_dlhandle_cache, ci);
  POCL_UNLOCK (pocl_dlhandle_cache_lock);
}

/*
static void
pocl_free_dlhandle (_cl_command_node *cmd)
{
  pocl_dlhandle_cache_item *ci = NULL;
  POCL_LOCK (pocl_dlhandle_cache_lock);
  DL_FOREACH (pocl_dlhandle_cache, ci)
    {
      if (strcmp (ci->tmp_dir, cmd->command.run.tmp_dir) == 0 &&
          strcmp (ci->function_name, 
                  cmd->command.run.kernel->name) == 0)
        {
          if ((--ci->ref_count))
            break;
          --handle_count;
          DL_DELETE (pocl_dlhandle_cache, ci);
          POCL_UNLOCK (pocl_dlhandle_cache_lock);
          free (ci->tmp_dir);
          free (ci->function_name);
          POCL_LOCK (pocl_llvm_codegen_lock);
          assert(!lt_dlclose (ci->dlhandle));
          POCL_UNLOCK (pocl_llvm_codegen_lock);
          free (ci);
          return;
        }
    }
  POCL_UNLOCK (pocl_dlhandle_cache_lock);
}
*/

#define MIN_MAX_MEM_ALLOC_SIZE (128*1024*1024)

/* accounting object for the main memory */
static pocl_global_mem_t system_memory;

void
pocl_setup_device_for_system_memory(cl_device_id device)
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
        system_memory.total_alloc_limit = alloc_limit - (size_t)(1UL << 31);
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
   * because some devices (HSA) set them up themselves
   *
   * it's max mem alloc / 4 because some programs (conformance test)
   * try to allocate max size constant objects and run out of memory
   * while trying to fill them. */
  device->local_mem_size = device->max_constant_buffer_size =
      (device->max_mem_alloc_size / 4);

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

void*
pocl_memalign_alloc_global_mem(cl_device_id device, size_t align, size_t size)
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

void
pocl_free_global_mem(cl_device_id device, void* ptr, size_t size)
{
  pocl_global_mem_t *mem = device->global_memory;

  assert(mem->currently_allocated >= size);
  mem->currently_allocated -= size;

  POCL_MEM_FREE(ptr);
}

void
pocl_print_system_memory_stats()
{
  POCL_MSG_PRINT("MEM STATS:\n", "",
  "____ Total available system memory  : %10zu KB\n"
  " ____ Currently used system memory   : %10zu KB\n"
  " ____ Max used system memory         : %10zu KB\n",
  system_memory.total_alloc_limit >> 10,
  system_memory.currently_allocated >> 10,
  system_memory.max_ever_allocated >> 10);
}
