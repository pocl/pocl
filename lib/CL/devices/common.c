/* common.c - common code that can be reused between device driver
              implementations

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos
                 2011-2018 Pekka Jääskeläinen

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

/* for posix_memalign and strdup */
#define _BSD_SOURCE
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "common.h"
#include "pocl_shared.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <utlist.h>

#ifndef _MSC_VER
#  include <sys/time.h>
#  include <sys/resource.h>
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include "config.h"
#include "config2.h"
#include "devices.h"
#include "pocl_cache.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_image_util.h"
#include "pocl_mem_management.h"
#include "pocl_runtime_config.h"
#include "pocl_util.h"

#ifdef OCS_AVAILABLE
#include "pocl_llvm.h"
#endif

#include "_kernel_constants.h"

#if defined(__x86_64__) || defined(__i386__)
#define CPU_IS_X86 1
#endif

#define WORKGROUP_STRING_LENGTH 1024

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
int
llvm_codegen (char *output, unsigned device_i, cl_kernel kernel,
              cl_device_id device, size_t local_x, size_t local_y,
              size_t local_z)
{
  POCL_MEASURE_START (llvm_codegen);
  int error = 0;
  void *llvm_module = NULL;

  char tmp_module[POCL_FILENAME_LENGTH];
  char tmp_objfile[POCL_FILENAME_LENGTH];

  char *objfile = NULL;
  uint64_t objfile_size = 0;

  cl_program program = kernel->program;

  const char *kernel_name = kernel->name;

  /* $/parallel.bc */
  char parallel_bc_path[POCL_FILENAME_LENGTH];
  pocl_cache_work_group_function_path (parallel_bc_path, program, device_i,
                                       kernel, local_x, local_y, local_z);

  /* $/kernel.so */
  char final_binary_path[POCL_FILENAME_LENGTH];
  pocl_cache_final_binary_path (final_binary_path, program, device_i, kernel,
                                local_x, local_y, local_z);

  if (pocl_exists (final_binary_path))
    goto FINISH;

  assert (strlen (final_binary_path) < (POCL_FILENAME_LENGTH - 3));

  error = pocl_llvm_generate_workgroup_function_nowrite (
      device_i, device, kernel, local_x, local_y, local_z, &llvm_module);
  if (error)
    {
      POCL_MSG_PRINT_LLVM ("pocl_llvm_generate_workgroup_function() failed"
                           " for kernel %s\n",
                           kernel_name);
      goto FINISH;
    }
  assert (llvm_module != NULL);

  if (pocl_get_bool_option ("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES", 0))
    {
      POCL_MSG_PRINT_LLVM ("Writing parallel.bc to %s.\n", parallel_bc_path);
      error = pocl_cache_write_kernel_parallel_bc (
          llvm_module, program, device_i, kernel, local_x, local_y, local_z);
    }
  else
    {
      char kernel_parallel_path[POCL_FILENAME_LENGTH];
      pocl_cache_kernel_cachedir_path (kernel_parallel_path, program, device_i,
                                       kernel, "", local_x, local_y, local_z);
      error = pocl_mkdir_p (kernel_parallel_path);
    }
  if (error)
    {
      POCL_MSG_PRINT_GENERAL ("writing parallel.bc failed"
                              " for kernel %s\n",
                              kernel->name);
      goto FINISH;
    }

  /* May happen if another thread is building the same program & wins the llvm
     lock. */
  if (pocl_exists (final_binary_path))
    goto FINISH;

  error = pocl_llvm_codegen (device, llvm_module, &objfile, &objfile_size);
  if (error)
    {
      POCL_MSG_PRINT_LLVM ("pocl_llvm_codegen() failed for kernel %s\n",
                           kernel_name);
      goto FINISH;
    }

  if (pocl_exists (final_binary_path))
    goto FINISH;

  /* Write temporary kernel.so.o, required for the final linking step.
     Use append-write because tmp_objfile is already temporary, thus
     we don't need to create new temporary... */
  error = pocl_cache_write_kernel_objfile (tmp_objfile, objfile, objfile_size);
  if (error)
    {
      POCL_MSG_PRINT_LLVM ("writing %s failed for kernel %s\n",
                           tmp_objfile, kernel_name);
      goto FINISH;
    }
  else
    {
      POCL_MSG_PRINT_LLVM ("written %s size %zu\n",
                          tmp_objfile, (size_t)objfile_size);
    }

  /* temporary filename for kernel.so */
  if (pocl_cache_tempname (tmp_module, ".so", NULL))
    {
      POCL_MSG_PRINT_LLVM ("Creating temporary kernel.so file"
                           " for kernel %s FAILED\n",
                           kernel_name);
      goto FINISH;
    }
  else
    POCL_MSG_PRINT_LLVM ("Temporary kernel.so file"
                         " for kernel %s : %s\n",
                         kernel_name, tmp_module);

  POCL_MSG_PRINT_INFO ("Linking final module\n");

#ifdef LLVM_OLDER_THAN_5_0
  /* with older LLVMs, link by invoking ld or clang */
  char *const args1[]
#if defined(LINK_WITH_CLANG)
      = { CLANG,
#else
      = { LINK_COMMAND,
#endif
          "-o",
          tmp_module,
          tmp_objfile,
          HOST_LD_FLAGS_ARRAY,
          NULL };
  error = pocl_run_command (args1);

#else
  /* Link through Clang driver interface who knows the correct toolchains
     for all of its targets.  */
  const char *cmd_line[64] =
    {CLANG, "-o", tmp_module, tmp_objfile};
  const char **device_ld_arg = device->final_linkage_flags;
  const char **pos = &cmd_line[4];
  while ((*pos++ = *device_ld_arg++)) {}

  error = pocl_invoke_clang (device, cmd_line);
#endif

  if (error)
    {
      POCL_MSG_PRINT_LLVM ("Linking kernel.so.o -> kernel.so has failed\n");
      goto FINISH;
    }

  /* rename temporary kernel.so */
  error = pocl_rename (tmp_module, final_binary_path);
  if (error)
    {
      POCL_MSG_PRINT_LLVM ("Renaming temporary kernel.so to final has failed.\n");
      goto FINISH;
    }

  /* if LEAVE_COMPILER_FILES, rename temporary kernel.so.o, else delete it */
  if (pocl_get_bool_option ("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES", 0))
    {
      char objfile_path[POCL_FILENAME_LENGTH];
      strcpy (objfile_path, final_binary_path);
      strcat (objfile_path, ".o");
      error = pocl_rename (tmp_objfile, objfile_path);
      if (error)
        POCL_MSG_PRINT_LLVM ("Renaming temporary kernel.so.o to final .o has failed.\n");
    }
  else
    {
      error = pocl_remove (tmp_objfile);
      if (error)
        POCL_MSG_PRINT_LLVM ("Removing temporary kernel.so.o has failed.\n");
    }

FINISH:
  pocl_destroy_llvm_module (llvm_module);
  POCL_MEM_FREE (objfile);
  POCL_MEASURE_FINISH (llvm_codegen);

  if (error)
    return error;
  else
    {
      memcpy (output, final_binary_path, POCL_FILENAME_LENGTH);
      return 0;
    }
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
  di->_width = mem->image_width;
  di->_height = mem->image_height;
  di->_depth = mem->image_depth;
  di->_row_pitch = mem->image_row_pitch;
  di->_slice_pitch = mem->image_slice_pitch;
  di->_order = mem->image_channel_order;
  di->_image_array_size = mem->image_array_size;
  di->_data_type = mem->image_channel_data_type;
  pocl_get_image_information (mem->image_channel_order,
                              mem->image_channel_data_type,
                              &(di->_num_channels), &(di->_elem_size));

  IMAGE1D_TO_BUFFER (mem);
  di->_data = (mem->device_ptrs[device->dev_id].mem_ptr);
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
         &dest->device_ptrs[dest_dev->dev_id],
         dest,
         &source->device_ptrs[src_dev->dev_id],
         source,
         dest_offset, source_offset,
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
          &source->device_ptrs[src_dev->dev_id],
          source,
          source_offset, cb);
      dest_dev->ops->write 
        (dest_dev->data, tmp, 
         &dest->device_ptrs[dest_dev->dev_id],
          dest, dest_offset,
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
  for (i = 0; i < node->command.run.kernel->meta->num_args; ++i)
    {
      pocl_aligned_free (node->command.run.arguments[i].value);
    }
  free (node->command.run.arguments);

  POname(clReleaseKernel)(node->command.run.kernel);
}

void
pocl_mem_objs_cleanup (cl_event event)
{
  size_t i;
  for (i = 0; i < event->num_buffers; ++i)
    {
      assert(event->mem_objs[i] != NULL);
      POname(clReleaseMemObject) (event->mem_objs[i]);
    }
  free (event->mem_objs);
  event->mem_objs = NULL;
}

/**
 * executes given command. Call with node->event UNLOCKED.
 */
void
pocl_exec_command (_cl_command_node * volatile node)
{
  unsigned i;
  /* because of POCL_UPDATE_EVENT_ */
  cl_event event = node->event;
  cl_device_id dev = node->device;
  _cl_command_t *cmd = &node->command;
  switch (node->type)
    {
    case CL_COMMAND_READ_BUFFER:
      POCL_UPDATE_EVENT_RUNNING(event);
      assert (dev->ops->read);
      dev->ops->read
        (dev->data,
         cmd->read.dst_host_ptr,
         cmd->read.src_mem_id,
         event->mem_objs[0],
         cmd->read.offset,
         cmd->read.size);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Read Buffer           ");
      break;

    case CL_COMMAND_WRITE_BUFFER:
      POCL_UPDATE_EVENT_RUNNING(event);
      assert (dev->ops->write);
      dev->ops->write
        (dev->data,
         cmd->write.src_host_ptr,
         cmd->write.dst_mem_id,
         event->mem_objs[0],
         cmd->write.offset,
         cmd->write.size);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Write Buffer          ");
      break;

    case CL_COMMAND_COPY_BUFFER:
      POCL_UPDATE_EVENT_RUNNING(event);
      assert (dev->ops->copy);
      dev->ops->copy
        (dev->data,
         cmd->copy.dst_mem_id,
         event->mem_objs[1],
         cmd->copy.src_mem_id,
         event->mem_objs[0],
         cmd->copy.dst_offset,
         cmd->copy.src_offset,
         cmd->copy.size);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Copy Buffer           ");
      break;

    case CL_COMMAND_FILL_BUFFER:
      POCL_UPDATE_EVENT_RUNNING(event);
      assert (dev->ops->memfill);
      dev->ops->memfill
        (dev->data,
         cmd->memfill.dst_mem_id,
         event->mem_objs[0],
         cmd->memfill.size,
         cmd->memfill.offset,
         cmd->memfill.pattern,
         cmd->memfill.pattern_size);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Fill Buffer           ");
      pocl_aligned_free (cmd->memfill.pattern);
      break;

    case CL_COMMAND_READ_BUFFER_RECT:
      POCL_UPDATE_EVENT_RUNNING(event);
      assert (dev->ops->read_rect);
      dev->ops->read_rect
        (dev->data,
         cmd->read_rect.dst_host_ptr,
         cmd->read_rect.src_mem_id,
         event->mem_objs[0],
         cmd->read_rect.buffer_origin,
         cmd->read_rect.host_origin,
         cmd->read_rect.region,
         cmd->read_rect.buffer_row_pitch,
         cmd->read_rect.buffer_slice_pitch,
         cmd->read_rect.host_row_pitch,
         cmd->read_rect.host_slice_pitch);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Read Buffer Rect      ");
      break;

    case CL_COMMAND_COPY_BUFFER_RECT:
      POCL_UPDATE_EVENT_RUNNING(event);
      assert (dev->ops->copy_rect);
      dev->ops->copy_rect
        (dev->data,
         cmd->copy_rect.dst_mem_id,
         event->mem_objs[1],
         cmd->copy_rect.src_mem_id,
         event->mem_objs[0],
         cmd->copy_rect.dst_origin,
         cmd->copy_rect.src_origin,
         cmd->copy_rect.region,
         cmd->copy_rect.dst_row_pitch,
         cmd->copy_rect.dst_slice_pitch,
         cmd->copy_rect.src_row_pitch,
         cmd->copy_rect.src_slice_pitch);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Copy Buffer Rect      ");
      break;

    case CL_COMMAND_WRITE_BUFFER_RECT:
      POCL_UPDATE_EVENT_RUNNING(event);
      assert (dev->ops->write_rect);
      dev->ops->write_rect
        (dev->data,
         cmd->write_rect.src_host_ptr,
         cmd->write_rect.dst_mem_id,
         event->mem_objs[0],
         cmd->write_rect.buffer_origin,
         cmd->write_rect.host_origin,
         cmd->write_rect.region,
         cmd->write_rect.buffer_row_pitch,
         cmd->write_rect.buffer_slice_pitch,
         cmd->write_rect.host_row_pitch,
         cmd->write_rect.host_slice_pitch);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Write Buffer Rect     ");
      break;

    case CL_COMMAND_MIGRATE_MEM_OBJECTS:
      POCL_UPDATE_EVENT_RUNNING (event);
      pocl_migrate_mem_objects (node);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Migrate Buffer        ");
      break;

    case CL_COMMAND_MAP_BUFFER:
      POCL_UPDATE_EVENT_RUNNING (event);
      assert (dev->ops->map_mem);
      POCL_LOCK_OBJ (event->mem_objs[0]);
        dev->ops->map_mem (dev->data,
                           cmd->map.mem_id,
                           event->mem_objs[0],
                           cmd->map.mapping);
      (event->mem_objs[0])->map_count++;
      POCL_UNLOCK_OBJ (event->mem_objs[0]);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Map Buffer            ");
      break;

    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
      POCL_UPDATE_EVENT_RUNNING (event);
      assert (dev->ops->read_image_rect);
      dev->ops->read_image_rect (
          dev->data,
          event->mem_objs[0],
          cmd->read_image.src_mem_id,
          NULL,
          cmd->read_image.dst_mem_id,
          cmd->read_image.origin,
          cmd->read_image.region,
          cmd->read_image.dst_row_pitch,
          cmd->read_image.dst_slice_pitch,
          cmd->read_image.dst_offset);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "CopyImageToBuffer       ");
      break;

    case CL_COMMAND_READ_IMAGE:
      POCL_UPDATE_EVENT_RUNNING(event);
      assert (dev->ops->read_image_rect);
      dev->ops->read_image_rect (
          dev->data,
          event->mem_objs[0],
          cmd->read_image.src_mem_id,
          cmd->read_image.dst_host_ptr,
          NULL,
          cmd->read_image.origin,
          cmd->read_image.region,
          cmd->read_image.dst_row_pitch,
          cmd->read_image.dst_slice_pitch,
          cmd->read_image.dst_offset);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Read Image            ");
      break;

    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      POCL_UPDATE_EVENT_RUNNING(event);
      assert (dev->ops->write_image_rect);
      dev->ops->write_image_rect (
          dev->data,
          event->mem_objs[1],
          cmd->write_image.dst_mem_id,
          NULL,
          cmd->write_image.src_mem_id,
          cmd->write_image.origin,
          cmd->write_image.region,
          cmd->write_image.src_row_pitch,
          cmd->write_image.src_slice_pitch,
          cmd->write_image.src_offset);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "CopyBufferToImage       ");
      break;

    case CL_COMMAND_WRITE_IMAGE:
        POCL_UPDATE_EVENT_RUNNING(event);
        assert (dev->ops->write_image_rect);
        dev->ops->write_image_rect (
            dev->data,
            event->mem_objs[0],
            cmd->write_image.dst_mem_id,
            cmd->write_image.src_host_ptr,
            NULL,
            cmd->write_image.origin,
            cmd->write_image.region,
            cmd->write_image.src_row_pitch,
            cmd->write_image.src_slice_pitch,
            cmd->write_image.src_offset);
        POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Write Image           ");
        break;

    case CL_COMMAND_COPY_IMAGE:
        POCL_UPDATE_EVENT_RUNNING(event);
        assert (dev->ops->copy_image_rect);
        dev->ops->copy_image_rect(
              dev->data,
              event->mem_objs[0],
              event->mem_objs[1],
              cmd->copy_image.src_mem_id,
              cmd->copy_image.dst_mem_id,
              cmd->copy_image.src_origin,
              cmd->copy_image.dst_origin,
              cmd->copy_image.region);
        POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Copy Image            ");
        break;

    case CL_COMMAND_FILL_IMAGE:
      POCL_UPDATE_EVENT_RUNNING (event);
      assert (dev->ops->fill_image);
      dev->ops->fill_image
        (dev->data,
         event->mem_objs[0],
         cmd->fill_image.mem_id,
         cmd->fill_image.origin,
         cmd->fill_image.region,
         cmd->fill_image.fill_pixel,
         cmd->fill_image.pixel_size);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Fill Image            ");
      POCL_MEM_FREE (cmd->fill_image.fill_pixel);
      break;

    case CL_COMMAND_MAP_IMAGE:
      POCL_UPDATE_EVENT_RUNNING(event);
      POCL_LOCK_OBJ (event->mem_objs[0]);
      assert (dev->ops->map_image != NULL);
      dev->ops->map_image (dev->data,
                           cmd->map.mem_id,
                           event->mem_objs[0],
                           cmd->map.mapping);
     (event->mem_objs[0])->map_count++;
      POCL_UNLOCK_OBJ (event->mem_objs[0]);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Map Image             ");
      break;

    case CL_COMMAND_UNMAP_MEM_OBJECT:
      POCL_UPDATE_EVENT_RUNNING(event);
      POCL_LOCK_OBJ (event->mem_objs[0]);
      if (event->mem_objs[0]->is_image == CL_FALSE
          || IS_IMAGE1D_BUFFER (event->mem_objs[0]))
        {
          assert (dev->ops->unmap_mem != NULL);
          dev->ops->unmap_mem (dev->data,
                               cmd->unmap.mem_id,
                               event->mem_objs[0],
                               cmd->unmap.mapping);
        }
      else
        {
          assert (dev->ops->unmap_image != NULL);
          dev->ops->unmap_image (dev->data,
                                 cmd->unmap.mem_id,
                                 event->mem_objs[0],
                                 cmd->unmap.mapping);
        }
      assert ((cmd->unmap.mapping)->unmap_requested > 0);
      DL_DELETE((event->mem_objs[0])->mappings,
                cmd->unmap.mapping);
      (event->mem_objs[0])->map_count--;
      POCL_MEM_FREE (cmd->unmap.mapping);
      POCL_UNLOCK_OBJ (event->mem_objs[0]);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Unmap Mem obj         ");
      break;

    case CL_COMMAND_NDRANGE_KERNEL:
      POCL_UPDATE_EVENT_RUNNING(event);
      assert (event == node->event);
      assert (dev->ops->run);
      dev->ops->run (dev->data, node);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Enqueue NDRange       ");
      pocl_ndrange_node_cleanup(node);
      break;

    case CL_COMMAND_NATIVE_KERNEL:
      POCL_UPDATE_EVENT_RUNNING(event);
      assert (dev->ops->run_native);
      dev->ops->run_native (dev->data, node);
      POCL_MEM_FREE (node->command.native.args);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Native Kernel         ");
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
      if (cmd->svm_free.pfn_free_func)
        cmd->svm_free.pfn_free_func(
           cmd->svm_free.queue,
           cmd->svm_free.num_svm_pointers,
           cmd->svm_free.svm_pointers,
           cmd->svm_free.data);
      else
        for (i = 0; i < cmd->svm_free.num_svm_pointers; i++)
          dev->ops->svm_free (dev, cmd->svm_free.svm_pointers[i]);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "SVM Free              ");
      break;

    case CL_COMMAND_SVM_MAP:
      POCL_UPDATE_EVENT_RUNNING(event);
      if (DEVICE_MMAP_IS_NOP (dev))
        ; // no-op
      else
        {
          assert (dev->ops->svm_map);
          dev->ops->svm_map (dev, cmd->svm_map.svm_ptr);
        }
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "SVM Map              ");
      break;

    case CL_COMMAND_SVM_UNMAP:
      POCL_UPDATE_EVENT_RUNNING(event);
      if (DEVICE_MMAP_IS_NOP (dev))
        ; // no-op
      else
        {
          assert (dev->ops->svm_unmap);
          dev->ops->svm_unmap (dev, cmd->svm_unmap.svm_ptr);
        }
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "SVM Unmap             ");
      break;

    case CL_COMMAND_SVM_MEMCPY:
      POCL_UPDATE_EVENT_RUNNING(event);
      assert (dev->ops->svm_copy);
      dev->ops->svm_copy (dev,
                          cmd->svm_memcpy.dst,
                          cmd->svm_memcpy.src,
                          cmd->svm_memcpy.size);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "SVM Memcpy            ");
      break;

    case CL_COMMAND_SVM_MEMFILL:
      POCL_UPDATE_EVENT_RUNNING(event);
      assert (dev->ops->svm_fill);
      dev->ops->svm_fill (dev,
                          cmd->svm_fill.svm_ptr,
                          cmd->svm_fill.size,
                          cmd->svm_fill.pattern,
                          cmd->svm_fill.pattern_size);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "SVM MemFill           ");
      pocl_aligned_free (cmd->svm_fill.pattern);
      break;

    default:
      POCL_ABORT_UNIMPLEMENTED("");
      break;
    }   
  pocl_mem_manager_free_command (node);
}

/* call with brc_event UNLOCKED. */
void
pocl_broadcast (cl_event brc_event)
{
  event_node *target;
  event_node *tmp;

  while ((target = brc_event->notify_list))
    {
      pocl_lock_events_inorder (brc_event, target->event);
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

        if ((target->event->status == CL_SUBMITTED)
            || (target->event->status == CL_QUEUED))
          {
            target->event->command->device->ops->notify (
                target->event->command->device, target->event, brc_event);
          }

        LL_DELETE (brc_event->notify_list, target);
        pocl_unlock_events_inorder (brc_event, target->event);
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
  cl_sampler sampler = *(cl_sampler *)parg->value;

  *ds = (sampler->normalized_coords == CL_TRUE) ? CLK_NORMALIZED_COORDS_TRUE
                                                : CLK_NORMALIZED_COORDS_FALSE;

  switch (sampler->addressing_mode)
    {
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

  switch (sampler->filter_mode)
    {
    case CL_FILTER_NEAREST:
      *ds |= CLK_FILTER_NEAREST; break;
    case CL_FILTER_LINEAR :
      *ds |= CLK_FILTER_LINEAR; break;
  }
}

/* CPU driver stuff */
typedef struct pocl_dlhandle_cache_item pocl_dlhandle_cache_item;
struct pocl_dlhandle_cache_item
{
  pocl_kernel_hash_t hash;
  size_t local_wgs[3];
  void *wg;
  void* dlhandle;
  pocl_dlhandle_cache_item *next;
  pocl_dlhandle_cache_item *prev;
  unsigned ref_count;
};

static pocl_dlhandle_cache_item *pocl_dlhandle_cache;
static pocl_lock_t pocl_llvm_codegen_lock;
static pocl_lock_t pocl_dlhandle_lock;
static int pocl_dlhandle_cache_initialized;

/* only to be called in basic/pthread/<other cpu driver> init */
void
pocl_init_dlhandle_cache ()
{
  if (!pocl_dlhandle_cache_initialized)
    {
      POCL_INIT_LOCK (pocl_llvm_codegen_lock);
      POCL_INIT_LOCK (pocl_dlhandle_lock);
      pocl_dlhandle_cache_initialized = 1;
   }
}

static unsigned handle_count = 0;
#define MAX_CACHE_ITEMS 128

/* must be called with pocl_dlhandle_lock LOCKED */
static pocl_dlhandle_cache_item *
get_new_dlhandle_cache_item ()
{
  pocl_dlhandle_cache_item *ci = NULL;
  const char *dl_error = NULL;

  if (pocl_dlhandle_cache)
    {
      ci = pocl_dlhandle_cache->prev;
      while (ci->ref_count > 0 && ci != pocl_dlhandle_cache)
        ci = ci->prev;
    }

  if ((handle_count >= MAX_CACHE_ITEMS) && ci && (ci != pocl_dlhandle_cache))
    {
      DL_DELETE (pocl_dlhandle_cache, ci);
      dlclose (ci->dlhandle);
      dl_error = dlerror ();
      if (dl_error != NULL)
        POCL_ABORT ("dlclose() failed with error: %s\n", dl_error);
      memset (ci, 0, sizeof (pocl_dlhandle_cache_item));
    }
  else
    {
      ++handle_count;
      ci = (pocl_dlhandle_cache_item *)calloc (
          1, sizeof (pocl_dlhandle_cache_item));
    }

  return ci;
}

void
pocl_release_dlhandle_cache (_cl_command_node *cmd)
{
  pocl_dlhandle_cache_item *ci = NULL, *found = NULL;

  POCL_LOCK (pocl_dlhandle_lock);
  DL_FOREACH (pocl_dlhandle_cache, ci)
  {
    if ((memcmp (ci->hash, cmd->command.run.hash, sizeof (pocl_kernel_hash_t))
         == 0)
        && (ci->local_wgs[0] == cmd->command.run.local_x)
        && (ci->local_wgs[1] == cmd->command.run.local_y)
        && (ci->local_wgs[2] == cmd->command.run.local_z))
      {
        found = ci;
        break;
      }
  }

  assert (found != NULL);
  assert (found->ref_count > 0);
  --found->ref_count;
  POCL_UNLOCK (pocl_dlhandle_lock);
}

/**
 * Checks if a built binary is found in the disk for the given kernel command,
 * if not, builds the kernel, caches it, and returns the file name of the
 * end result.
 *
 * @param cmd The kernel command.
 * @returns The filename of the built binary in the disk.
 */
char *
pocl_check_kernel_disk_cache (_cl_command_node *cmd)
{
  char *module_fn = NULL;
  cl_kernel k = cmd->command.run.kernel;
  cl_program p = k->program;
  unsigned dev_i = cmd->command.run.device_i;

  size_t x = cmd->command.run.local_x;
  size_t y = cmd->command.run.local_y;
  size_t z = cmd->command.run.local_z;

  module_fn = malloc (POCL_FILENAME_LENGTH);
  pocl_cache_final_binary_path (module_fn, p, dev_i, k, x, y, z);
  if (pocl_exists (module_fn))
    {
      POCL_MSG_PRINT_INFO (
          "For %zu x %zu x %zu, using static WG size binary: %s\n", x, y, z,
          module_fn);
      return module_fn;
    }

  if (p->binaries[dev_i])
    {
#ifdef OCS_AVAILABLE

      POCL_LOCK (pocl_llvm_codegen_lock);
      int error = llvm_codegen (module_fn, dev_i, cmd->command.run.kernel,
                                cmd->device, x, y, z);
      POCL_UNLOCK (pocl_llvm_codegen_lock);
      if (error)
        {
          POCL_ABORT ("Final linking of kernel %s failed.\n", k->name);
        }
      POCL_MSG_PRINT_INFO (
          "For %zu x %zu x %zu, using static WG size binary: %s\n", x, y, z,
          module_fn);
      return module_fn;
#else
      if (!p->pocl_binaries[dev_i])
        POCL_ABORT ("pocl built without online compiler support"
                    " cannot compile LLVM IRs to machine code\n");
#endif
    }

  // pocl_binaries must exist
  assert (p->pocl_binaries[dev_i]);

  pocl_cache_final_binary_path (module_fn, p, dev_i, k, 0, 0, 0);
  if (!pocl_exists (module_fn))
    POCL_ABORT ("Can't find either static or dynamic WG size binary!\n");

  POCL_MSG_PRINT_INFO (
      "For %zu x %zu x %zu, using dynamic WG size binary: %s\n", x, y, z,
      module_fn);
  return module_fn;
}

/**
 * Checks if the kernel command has been built and has been loaded with
 * dlopen, and reuses its handle. If not, checks if a built binary is found
 * in the disk, if not, builds the kernel and puts it to respective
 * caches.
 *
 * The initial refcount may be 0, in case we're just pre-compiling kernels
 * (or compiling them for binaries), and not actually need them immediately. */
void
pocl_check_kernel_dlhandle_cache (_cl_command_node *cmd,
				  unsigned initial_refcount)
{
  char workgroup_string[WORKGROUP_STRING_LENGTH];
  pocl_dlhandle_cache_item *ci = NULL, *tmp = NULL;
  const char *dl_error = NULL;

  POCL_LOCK (pocl_dlhandle_lock);
  DL_FOREACH_SAFE (pocl_dlhandle_cache, ci, tmp)
  {
    if ((memcmp (ci->hash, cmd->command.run.hash, sizeof (pocl_kernel_hash_t))
         == 0)
        && (ci->local_wgs[0] == cmd->command.run.local_x)
        && (ci->local_wgs[1] == cmd->command.run.local_y)
        && (ci->local_wgs[2] == cmd->command.run.local_z))
      {
        /* move to the front of the line */
        DL_DELETE (pocl_dlhandle_cache, ci);
        DL_PREPEND (pocl_dlhandle_cache, ci);
        ++ci->ref_count;
        POCL_UNLOCK (pocl_dlhandle_lock);
        cmd->command.run.wg = ci->wg;
        return;
      }
  }

  ci = get_new_dlhandle_cache_item ();
  POCL_UNLOCK (pocl_dlhandle_lock);

  memcpy (ci->hash, cmd->command.run.hash, sizeof (pocl_kernel_hash_t));
  ci->local_wgs[0] = cmd->command.run.local_x;
  ci->local_wgs[1] = cmd->command.run.local_y;
  ci->local_wgs[2] = cmd->command.run.local_z;
  ci->ref_count = initial_refcount;

  char *module_fn = pocl_check_kernel_disk_cache (cmd);

  POCL_LOCK (pocl_dlhandle_lock);

  pocl_dlhandle_cache_item *ci2 = NULL;
  DL_FOREACH_SAFE (pocl_dlhandle_cache, ci2, tmp)
  {
    if ((memcmp (ci->hash, ci2->hash, sizeof (pocl_kernel_hash_t)) == 0)
        && (ci->local_wgs[0] == ci2->local_wgs[0])
        && (ci->local_wgs[1] == ci2->local_wgs[1])
        && (ci->local_wgs[2] == ci2->local_wgs[2]))
      {
        /* move to the front of the line */
        if (pocl_dlhandle_cache != ci2)
          {
            DL_DELETE (pocl_dlhandle_cache, ci2);
            DL_PREPEND (pocl_dlhandle_cache, ci2);
          }
        ++ci2->ref_count;
        cmd->command.run.wg = ci2->wg;

        POCL_MEM_FREE (ci);
        POCL_UNLOCK (pocl_dlhandle_lock);
        POCL_MEM_FREE (module_fn);
        return;
      }
    }

    ci->dlhandle = dlopen (module_fn, RTLD_NOW|RTLD_LOCAL);
    dl_error = dlerror ();

    if (ci->dlhandle != NULL && dl_error == NULL)
      {
        snprintf (workgroup_string, WORKGROUP_STRING_LENGTH,
                  "_pocl_kernel_%s_workgroup",
		  cmd->command.run.kernel->name);
        ci->wg = dlsym (ci->dlhandle, workgroup_string);
        dl_error = dlerror ();
        if (ci->wg == NULL)
          {
            // Older osx dyld APIs need the name without the underscore
            snprintf (workgroup_string, WORKGROUP_STRING_LENGTH,
                      "pocl_kernel_%s_workgroup",
		      cmd->command.run.kernel->name);
            ci->wg = dlsym (ci->dlhandle, workgroup_string);
            dl_error = dlerror ();
          }

        if (ci->wg != NULL && dl_error == NULL)
          {
            cmd->command.run.wg = ci->wg;
            DL_PREPEND (pocl_dlhandle_cache, ci);
          }
      }

    POCL_UNLOCK (pocl_dlhandle_lock);
    /***************************************************************************/
    POCL_MEM_FREE (module_fn);

    if (ci->dlhandle == NULL || ci->wg == NULL || dl_error != NULL)
      {
        POCL_ABORT (
            "pocl error: dlopen(\"%s\") or dlsym() failed with '%s'.\n"
            "note: missing symbols in the kernel binary might be"
            " reported as 'file not found' errors.\n",
            module_fn, dl_error);
      }
}


#define MIN_MAX_MEM_ALLOC_SIZE (128*1024*1024)

/* accounting object for the main memory */
static pocl_global_mem_t system_memory = {POCL_LOCK_INITIALIZER, 0, 0, 0};

void
pocl_setup_device_for_system_memory(cl_device_id device)
{
  int limit_memory_gb = pocl_get_int_option ("POCL_MEMORY_LIMIT", 0);

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

  if (limit_memory_gb > 0)
    {
      size_t limited_memory = (size_t)limit_memory_gb << 30;
      if (device->global_mem_size > limited_memory)
        device->global_mem_size = limited_memory;
      else
        POCL_MSG_WARN ("requested POCL_MEMORY_LIMIT %i GBs is larger than"
                       " physical memory size (%zu) GBs, ignoring\n",
                       limit_memory_gb,
                       (size_t) (device->global_mem_size >> 30));
    }

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
    alloc_limit = pocl_size_ceil2 (device->global_mem_size / 4);
  if (alloc_limit > (device->global_mem_size / 2))
    alloc_limit >>= 1;

  if (alloc_limit < MIN_MAX_MEM_ALLOC_SIZE)
    alloc_limit = MIN_MAX_MEM_ALLOC_SIZE;

  // set up device properties..
  device->global_memory = &system_memory;
  device->max_mem_alloc_size = alloc_limit;

  // TODO in theory now if alloc_limit was > rlim_cur and < rlim_max
  // we should try and setrlimit to alloc_limit, or allocations might fail
}

void
pocl_reinit_system_memory()
{
  system_memory.currently_allocated = 0;
  system_memory.max_ever_allocated = 0;
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

  size_t s;
  if (device->global_mem_cache_size > 0)
    s = pocl_size_ceil2 (device->global_mem_cache_size / 2);
  else
    s = pocl_size_ceil2 (device->global_mem_size / 256);

  device->local_mem_size = device->max_constant_buffer_size = s;

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
pocl_aligned_malloc_global_mem(cl_device_id device, size_t align, size_t size)
{
  pocl_global_mem_t *mem = device->global_memory;
  void *retval = NULL;

  POCL_LOCK (mem->pocl_lock);
  if ((mem->total_alloc_limit - mem->currently_allocated) < size)
    goto ERROR;

  retval = pocl_aligned_malloc (align, size);
  if (!retval)
    goto ERROR;

  mem->currently_allocated += size;
  if (mem->max_ever_allocated < mem->currently_allocated)
    mem->max_ever_allocated = mem->currently_allocated;
  assert(mem->currently_allocated <= mem->total_alloc_limit);

ERROR:
  POCL_UNLOCK (mem->pocl_lock);

  return retval;
}

void
pocl_free_global_mem(cl_device_id device, void* ptr, size_t size)
{
  pocl_global_mem_t *mem = device->global_memory;

  POCL_LOCK (mem->pocl_lock);
  assert(mem->currently_allocated >= size);
  mem->currently_allocated -= size;
  POCL_UNLOCK (mem->pocl_lock);

  POCL_MEM_FREE(ptr);
}


void
pocl_print_system_memory_stats()
{
  POCL_MSG_PRINT_F (MEMORY, INFO, "",
  "____ Total available system memory  : %10zu KB\n"
  " ____ Currently used system memory   : %10zu KB\n"
  " ____ Max used system memory         : %10zu KB\n",
  system_memory.total_alloc_limit >> 10,
  system_memory.currently_allocated >> 10,
  system_memory.max_ever_allocated >> 10);
}

/* Unique hash for a device + program build + kernel name combination.
 * NOTE: this does NOT take into account the local WG sizes. */
void
pocl_calculate_kernel_hash (cl_program program, unsigned kernel_i,
                            unsigned device_i)
{
  SHA1_CTX hash_ctx;
  pocl_SHA1_Init (&hash_ctx);

  char *n = program->kernel_meta[kernel_i].name;
  pocl_SHA1_Update (&hash_ctx, (uint8_t *)program->build_hash[device_i],
                    sizeof (SHA1_digest_t));
  pocl_SHA1_Update (&hash_ctx, (uint8_t *)n, strlen (n));

  uint8_t digest[SHA1_DIGEST_SIZE];
  pocl_SHA1_Final (&hash_ctx, digest);

  memcpy (program->kernel_meta[kernel_i].build_hash[device_i], digest,
          sizeof (pocl_kernel_hash_t));
}
