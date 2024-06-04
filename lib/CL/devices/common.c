/* common.c - common code that can be reused between device driver
              implementations

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos
                 2011-2021 Pekka Jääskeläinen
                 2022-2024 Pekka Jääskeläinen / Intel Finland Oy

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

/* for posix_memalign and strdup */
#define _BSD_SOURCE
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <utlist.h>

#ifdef _WIN32
#include "vccompat.hpp"
#endif

#include "common.h"
#include "pocl_shared.h"

#include "config.h"
#include "config2.h"
#include "devices.h"
#include "pocl_cache.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_image_util.h"
#include "pocl_mem_management.h"
#include "pocl_runtime_config.h"
#include "pocl_timing.h"
#include "pocl_util.h"
#include "common_driver.h"

#ifdef HAVE_GETRLIMIT
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#ifdef HAVE_DLFCN_H
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#endif
#include <dlfcn.h>
#endif

#ifdef ENABLE_LLVM
#include "pocl_llvm.h"
#endif

#include "_kernel_constants.h"

#define WORKGROUP_STRING_LENGTH 1024

uint64_t last_object_id = 0;

unsigned long buffer_c;
unsigned long svm_buffer_c;
unsigned long usm_buffer_c;
unsigned long queue_c;
unsigned long context_c;
unsigned long image_c;
unsigned long kernel_c;
unsigned long program_c;
unsigned long sampler_c;
unsigned long uevent_c;
unsigned long event_c;

/**
 * Generate code from the final bitcode using the LLVM
 * tools.
 *
 * Uses an existing (cached) one, if available.
 */

#ifdef ENABLE_LLVM
static int
llvm_codegen (char *output, unsigned device_i, cl_kernel kernel,
              cl_device_id device, _cl_command_node *command, int specialize)
{
  POCL_MEASURE_START (llvm_codegen);
  int error = 0;
  void *llvm_module = NULL;

  char tmp_module[POCL_MAX_PATHNAME_LENGTH];
  char tmp_objfile[POCL_MAX_PATHNAME_LENGTH];

  char *objfile = NULL;
  uint64_t objfile_size = 0;

  cl_program program = kernel->program;

  const char *kernel_name = kernel->name;

  /* $/parallel.bc */
  char parallel_bc_path[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_work_group_function_path (parallel_bc_path, program, device_i,
                                       kernel, command, specialize);

  /* $/kernel.so */
  char final_binary_path[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_final_binary_path (final_binary_path, program, device_i, kernel,
                                command, specialize);

  if (pocl_exists (final_binary_path))
    goto FINISH;

  assert (strlen (final_binary_path) < (POCL_MAX_PATHNAME_LENGTH - 3));

  error = pocl_llvm_generate_workgroup_function_nowrite (
      device_i, device, kernel, command, &llvm_module, specialize);
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
          llvm_module, program, device_i, kernel, command, specialize);
    }
  else
    {
      char kernel_parallel_path[POCL_MAX_PATHNAME_LENGTH];
      pocl_cache_kernel_cachedir_path (kernel_parallel_path, program,
                                       command->program_device_i,
                                       kernel, "", command, specialize);
      error = pocl_mkdir_p (kernel_parallel_path);
    }
  if (error)
    {
      POCL_MSG_PRINT_GENERAL ("writing parallel.bc failed for kernel %s\n",
                              kernel->name);
      goto FINISH;
    }

  /* May happen if another thread is building the same program & wins the llvm
     lock. */
  if (pocl_exists (final_binary_path))
    goto FINISH;

  error = pocl_llvm_codegen (device, program, llvm_module, &objfile,
                             &objfile_size);
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

  /* Link through Clang driver interface who knows the correct toolchains
     for all of its targets.  */
  const char *cmd_line[64] =
    {CLANG, "-o", tmp_module, tmp_objfile};
  const char **device_ld_arg = device->final_linkage_flags;
  const char **pos = &cmd_line[4];
  while ((*pos++ = *device_ld_arg++)) {}

  error = pocl_invoke_clang (device, cmd_line);

  if (error)
    {
      POCL_MSG_PRINT_LLVM ("Linking kernel.so.o -> kernel.so has failed\n");
      goto FINISH;
    }

  /* rename temporary kernel.so */
  error = pocl_rename (tmp_module, final_binary_path);
  if (error)
    {
      POCL_MSG_PRINT_LLVM (
          "Renaming temporary kernel.so to final ('%s') has failed.\n",
          final_binary_path);
      goto FINISH;
    }

  /* if LEAVE_COMPILER_FILES, rename temporary kernel.so.o, else delete it */
  if (pocl_get_bool_option ("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES", 0))
    {
      char objfile_path[POCL_MAX_PATHNAME_LENGTH];
      strcpy (objfile_path, final_binary_path);
      strcat (objfile_path, ".o");
      error = pocl_rename (tmp_objfile, objfile_path);
      if (error)
        POCL_MSG_PRINT_LLVM (
            "Renaming temporary kernel.so.o to final %s has failed.\n",
            objfile_path);
    }
  else
    {
      error = pocl_remove (tmp_objfile);
      if (error)
        POCL_MSG_PRINT_LLVM ("Removing temporary kernel.so.o has failed.\n");
    }

FINISH:
  pocl_destroy_llvm_module (llvm_module, kernel->context);
  POCL_MEM_FREE (objfile);
  POCL_MEASURE_FINISH (llvm_codegen);

  if (error)
    return error;
  else
    {
      memcpy (output, final_binary_path, POCL_MAX_PATHNAME_LENGTH);
      return 0;
    }
}
#endif


/**
 * Populates the device specific image data structure used by kernel
 * from given kernel image argument
 */
void
pocl_fill_dev_image_t (dev_image_t *di, struct pocl_argument *parg,
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
  di->_data = (mem->device_ptrs[device->global_mem_id].mem_ptr);
}

/**
 * executes given command. Call with node->sync.event.event UNLOCKED.
 */
void
pocl_exec_command (_cl_command_node *node)
{
  unsigned i;
  /* because of POCL_UPDATE_EVENT_ */
  cl_event event = node->sync.event.event;
  cl_device_id dev = node->device;
  _cl_command_t *cmd = &node->command;
  cl_mem mem = NULL;
  if (event->num_buffers > 0)
    mem = event->mem_objs[0];

  switch (node->type)
    {
    case CL_COMMAND_READ_BUFFER:
      pocl_update_event_running (event);
      assert (dev->ops->read);
      dev->ops->read
        (dev->data,
         cmd->read.dst_host_ptr,
         cmd->read.src_mem_id,
         event->mem_objs[0],
         cmd->read.offset,
         cmd->read.size);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Read Buffer           ");
      break;

    case CL_COMMAND_WRITE_BUFFER:
      pocl_update_event_running (event);
      assert (dev->ops->write);
      dev->ops->write
        (dev->data,
         cmd->write.src_host_ptr,
         cmd->write.dst_mem_id,
         event->mem_objs[0],
         cmd->write.offset,
         cmd->write.size);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Write Buffer          ");
      break;

    case CL_COMMAND_COPY_BUFFER:
      pocl_update_event_running (event);
      assert (dev->ops->copy);
      if (dev->ops->copy_with_size && cmd->copy.src_content_size != NULL)
          dev->ops->copy_with_size
            (dev->data,
             cmd->copy.dst_mem_id,
             cmd->copy.dst,
             cmd->copy.src_mem_id,
             cmd->copy.src,
             cmd->copy.src_content_size_mem_id,
             cmd->copy.src_content_size,
             cmd->copy.dst_offset,
             cmd->copy.src_offset,
             cmd->copy.size);
      else
          dev->ops->copy
            (dev->data,
             cmd->copy.dst_mem_id,
             cmd->copy.dst,
             cmd->copy.src_mem_id,
             cmd->copy.src,
             cmd->copy.dst_offset,
             cmd->copy.src_offset,
             cmd->copy.size);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Copy Buffer           ");
      break;

    case CL_COMMAND_FILL_BUFFER:
      pocl_update_event_running (event);
      assert (dev->ops->memfill);
      dev->ops->memfill
        (dev->data,
         cmd->memfill.dst_mem_id,
         event->mem_objs[0],
         cmd->memfill.size,
         cmd->memfill.offset,
         cmd->memfill.pattern,
         cmd->memfill.pattern_size);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Fill Buffer           ");
      break;

    case CL_COMMAND_READ_BUFFER_RECT:
      pocl_update_event_running (event);
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
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Read Buffer Rect      ");
      break;

    case CL_COMMAND_COPY_BUFFER_RECT:
      pocl_update_event_running (event);
      assert (dev->ops->copy_rect);
      dev->ops->copy_rect (
          dev->data, cmd->copy_rect.dst_mem_id, cmd->copy_rect.dst,
          cmd->copy_rect.src_mem_id, cmd->copy_rect.src,
          cmd->copy_rect.dst_origin, cmd->copy_rect.src_origin,
          cmd->copy_rect.region, cmd->copy_rect.dst_row_pitch,
          cmd->copy_rect.dst_slice_pitch, cmd->copy_rect.src_row_pitch,
          cmd->copy_rect.src_slice_pitch);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Copy Buffer Rect      ");
      break;

    case CL_COMMAND_WRITE_BUFFER_RECT:
      pocl_update_event_running (event);
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
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Write Buffer Rect     ");
      break;

    case CL_COMMAND_MIGRATE_MEM_OBJECTS:
      pocl_update_event_running (event);
      switch (cmd->migrate.type)
        {
        case ENQUEUE_MIGRATE_TYPE_D2H:
          {
            if (mem->is_image)
              {
                size_t region[3] = { mem->image_width, mem->image_height,
                                     mem->image_depth };
                if (region[2] == 0)
                  region[2] = 1;
                if (region[1] == 0)
                  region[1] = 1;
                size_t origin[3] = { 0, 0, 0 };
                assert (dev->ops->read_image_rect);
                dev->ops->read_image_rect (dev->data, mem, cmd->migrate.mem_id,
                                           mem->mem_host_ptr, NULL, origin,
                                           region, 0, 0, 0);
              }
            else
              {
                assert (dev->ops->read);
                dev->ops->read (dev->data, mem->mem_host_ptr,
                                cmd->migrate.mem_id, mem, 0, mem->size);
              }
            break;
          }
        case ENQUEUE_MIGRATE_TYPE_H2D:
          {
            if (mem->is_image)
              {
                size_t region[3] = { mem->image_width, mem->image_height,
                                     mem->image_depth };
                if (region[2] == 0)
                  region[2] = 1;
                if (region[1] == 0)
                  region[1] = 1;
                size_t origin[3] = { 0, 0, 0 };
                assert (dev->ops->write_image_rect);
                dev->ops->write_image_rect (
                    dev->data, mem, cmd->migrate.mem_id, mem->mem_host_ptr,
                    NULL, origin, region, 0, 0, 0);
              }
            else
              {
                assert (dev->ops->write);
                dev->ops->write (dev->data, mem->mem_host_ptr,
                                 cmd->migrate.mem_id, mem, 0, mem->size);
              }
            break;
          }
        case ENQUEUE_MIGRATE_TYPE_D2D:
          {
            assert (dev->ops->can_migrate_d2d);
            assert (dev->ops->migrate_d2d);
            dev->ops->migrate_d2d (cmd->migrate.src_device, dev, mem,
                                   cmd->migrate.src_id,
                                   cmd->migrate.dst_id);
            break;
          }
        case ENQUEUE_MIGRATE_TYPE_NOP:
          {
            break;
          }
        }

      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Migrate Buffer(s)     ");
      break;

    case CL_COMMAND_MAP_BUFFER:
      pocl_update_event_running (event);
      assert (dev->ops->map_mem);
        dev->ops->map_mem (dev->data,
                           cmd->map.mem_id,
                           event->mem_objs[0],
                           cmd->map.mapping);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Map Buffer            ");
      break;

    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
      pocl_update_event_running (event);
      assert (dev->ops->read_image_rect);
      dev->ops->read_image_rect (
          dev->data, cmd->read_image.src, cmd->read_image.src_mem_id, NULL,
          cmd->read_image.dst_mem_id, cmd->read_image.origin,
          cmd->read_image.region, cmd->read_image.dst_row_pitch,
          cmd->read_image.dst_slice_pitch, cmd->read_image.dst_offset);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event CopyImageToBuffer       ");
      break;

    case CL_COMMAND_READ_IMAGE:
      pocl_update_event_running (event);
      assert (dev->ops->read_image_rect);
      dev->ops->read_image_rect (
          dev->data, cmd->read_image.src, cmd->read_image.src_mem_id,
          cmd->read_image.dst_host_ptr, NULL, cmd->read_image.origin,
          cmd->read_image.region, cmd->read_image.dst_row_pitch,
          cmd->read_image.dst_slice_pitch, cmd->read_image.dst_offset);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Read Image            ");
      break;

    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      pocl_update_event_running (event);
      assert (dev->ops->write_image_rect);
      dev->ops->write_image_rect (
          dev->data, cmd->write_image.dst, cmd->write_image.dst_mem_id, NULL,
          cmd->write_image.src_mem_id, cmd->write_image.origin,
          cmd->write_image.region, cmd->write_image.src_row_pitch,
          cmd->write_image.src_slice_pitch, cmd->write_image.src_offset);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event CopyBufferToImage       ");
      break;

    case CL_COMMAND_WRITE_IMAGE:
      pocl_update_event_running (event);
      assert (dev->ops->write_image_rect);
      dev->ops->write_image_rect (
          dev->data, cmd->write_image.dst, cmd->write_image.dst_mem_id,
          cmd->write_image.src_host_ptr, NULL, cmd->write_image.origin,
          cmd->write_image.region, cmd->write_image.src_row_pitch,
          cmd->write_image.src_slice_pitch, cmd->write_image.src_offset);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Write Image           ");
      break;

    case CL_COMMAND_COPY_IMAGE:
      pocl_update_event_running (event);
      assert (dev->ops->copy_image_rect);
      dev->ops->copy_image_rect (
          dev->data, cmd->copy_image.src, cmd->copy_image.dst,
          cmd->copy_image.src_mem_id, cmd->copy_image.dst_mem_id,
          cmd->copy_image.src_origin, cmd->copy_image.dst_origin,
          cmd->copy_image.region);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Copy Image            ");
      break;

    case CL_COMMAND_FILL_IMAGE:
      pocl_update_event_running (event);
      assert (dev->ops->fill_image);
      dev->ops->fill_image (dev->data, event->mem_objs[0],
                            cmd->fill_image.mem_id, cmd->fill_image.origin,
                            cmd->fill_image.region, cmd->fill_image.orig_pixel,
                            cmd->fill_image.fill_pixel,
                            cmd->fill_image.pixel_size);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Fill Image            ");
      break;

    case CL_COMMAND_MAP_IMAGE:
      pocl_update_event_running (event);
      assert (dev->ops->map_image != NULL);
      dev->ops->map_image (dev->data,
                           cmd->map.mem_id,
                           event->mem_objs[0],
                           cmd->map.mapping);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Map Image             ");
      break;

    case CL_COMMAND_UNMAP_MEM_OBJECT:
      pocl_update_event_running (event);
      pocl_mem_identifier *mem_id = &mem->device_ptrs[dev->global_mem_id];
      if (mem->is_image == CL_FALSE || IS_IMAGE1D_BUFFER (mem))
        {
          assert (dev->ops->unmap_mem != NULL);
          dev->ops->unmap_mem (dev->data, cmd->unmap.mem_id, mem,
                               cmd->unmap.mapping);
        }
      else
        {
          assert (dev->ops->unmap_image != NULL);
          dev->ops->unmap_image (dev->data, cmd->unmap.mem_id, mem,
                                 cmd->unmap.mapping);
        }
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Unmap Mem obj         ");
      break;

    case CL_COMMAND_NDRANGE_KERNEL:
      pocl_update_event_running (event);
      assert (dev->ops->run);
      dev->ops->run (dev->data, node);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Enqueue NDRange       ");
      break;

    case CL_COMMAND_NATIVE_KERNEL:
      pocl_update_event_running (event);
      assert (dev->ops->run_native);
      dev->ops->run_native (dev->data, node);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Native Kernel         ");
      break;

    case CL_COMMAND_BARRIER:
    case CL_COMMAND_MARKER:
      pocl_update_event_running (event);
      POCL_UPDATE_EVENT_COMPLETE(event);
      break;

    case CL_COMMAND_SVM_FREE:
      pocl_update_event_running (event);
      if (cmd->svm_free.pfn_free_func)
        cmd->svm_free.pfn_free_func(
           cmd->svm_free.queue,
           cmd->svm_free.num_svm_pointers,
           cmd->svm_free.svm_pointers,
           cmd->svm_free.data);
      else
        for (i = 0; i < cmd->svm_free.num_svm_pointers; i++)
          {
            void *ptr = cmd->svm_free.svm_pointers[i];
            POCL_LOCK_OBJ (event->context);
            pocl_raw_ptr *tmp = NULL, *item = NULL;
            DL_FOREACH_SAFE (event->context->raw_ptrs, item, tmp)
            {
              if (item->vm_ptr == ptr)
                {
                  DL_DELETE (event->context->raw_ptrs, item);
                  break;
                }
            }
            POCL_UNLOCK_OBJ (event->context);
            assert (item);
            POCL_MEM_FREE (item);
            POname (clReleaseContext) (event->context);
            dev->ops->svm_free (dev, ptr);
          }
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event SVM Free              ");
      break;

    case CL_COMMAND_SVM_MAP:
      pocl_update_event_running (event);
      if (DEVICE_MMAP_IS_NOP (dev))
        ; // no-op
      else
        {
          assert (dev->ops->svm_map);
          dev->ops->svm_map (dev, cmd->svm_map.svm_ptr);
        }
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event SVM Map              ");
      break;

    case CL_COMMAND_SVM_UNMAP:
      pocl_update_event_running (event);
      if (DEVICE_MMAP_IS_NOP (dev))
        ; // no-op
      else
        {
          assert (dev->ops->svm_unmap);
          dev->ops->svm_unmap (dev, cmd->svm_unmap.svm_ptr);
        }
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event SVM Unmap             ");
      break;

    case CL_COMMAND_SVM_MEMCPY:
    case CL_COMMAND_MEMCPY_INTEL:
      pocl_update_event_running (event);
      assert (dev->ops->svm_copy);
      dev->ops->svm_copy (dev,
                          cmd->svm_memcpy.dst,
                          cmd->svm_memcpy.src,
                          cmd->svm_memcpy.size);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event SVM Memcpy            ");
      break;

    case CL_COMMAND_SVM_MEMFILL:
    case CL_COMMAND_MEMFILL_INTEL:
      pocl_update_event_running (event);
      assert (dev->ops->svm_fill);
      dev->ops->svm_fill (dev,
                          cmd->svm_fill.svm_ptr,
                          cmd->svm_fill.size,
                          cmd->svm_fill.pattern,
                          cmd->svm_fill.pattern_size);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event SVM MemFill           ");
      break;

    case CL_COMMAND_SVM_MIGRATE_MEM:
    case CL_COMMAND_MIGRATEMEM_INTEL:
      pocl_update_event_running (event);
      if (dev->ops->svm_migrate)
        dev->ops->svm_migrate (dev, cmd->svm_migrate.num_svm_pointers,
                               cmd->svm_migrate.svm_pointers,
                               cmd->svm_migrate.sizes);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event SVM Migrate_Mem       ");
      break;

    case CL_COMMAND_MEMADVISE_INTEL:
      pocl_update_event_running (event);
      if (dev->ops->svm_advise)
        dev->ops->svm_advise (dev, cmd->mem_advise.ptr, cmd->mem_advise.size,
                              cmd->mem_advise.advice);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event SVM Mem_Advise        ");
      break;

    case CL_COMMAND_SVM_MEMFILL_RECT_POCL:
      pocl_update_event_running (event);
      assert (dev->ops->svm_fill_rect);
      dev->ops->svm_fill_rect (dev,
                               cmd->svm_fill_rect.svm_ptr,
                               cmd->svm_fill_rect.origin,
                               cmd->svm_fill_rect.region,
                               cmd->svm_fill_rect.row_pitch,
                               cmd->svm_fill_rect.slice_pitch,
                               cmd->svm_fill_rect.pattern,
                               cmd->svm_fill_rect.pattern_size);
      break;

    case CL_COMMAND_SVM_MEMCPY_RECT_POCL:
      pocl_update_event_running (event);
      assert (dev->ops->svm_copy_rect);
      dev->ops->svm_copy_rect (dev,
                               cmd->svm_memcpy_rect.dst,
                               cmd->svm_memcpy_rect.src,
                               cmd->svm_memcpy_rect.dst_origin,
                               cmd->svm_memcpy_rect.src_origin,
                               cmd->svm_memcpy_rect.region,
                               cmd->svm_memcpy_rect.dst_row_pitch,
                               cmd->svm_memcpy_rect.dst_slice_pitch,
                               cmd->svm_memcpy_rect.src_row_pitch,
                               cmd->svm_memcpy_rect.src_slice_pitch);
      POCL_UPDATE_EVENT_COMPLETE_MSG (event, "Event Copy Buffer Rect      ");
      break;

    case CL_COMMAND_COMMAND_BUFFER_KHR:
      pocl_update_event_running (event);
      POCL_UPDATE_EVENT_COMPLETE (event);
      break;

    default:
      POCL_ABORT_UNIMPLEMENTED("");
      break;
    }
}

char *
pocl_cpu_build_hash (cl_device_id device)
{
  char *res = calloc(1000, sizeof(char));
  snprintf (res, 1000, "cpu-%s-%s", HOST_DEVICE_BUILD_HASH,
            device->llvm_cpu);
  return res;
}

/* call with brc_event UNLOCKED. */
void
pocl_broadcast (cl_event brc_event)
{
  event_node *target;
  event_node *tmp;

  POCL_LOCK_OBJ (brc_event);
  while ((target = brc_event->notify_list))
    {
      cl_event target_event = target->event;
      POCL_UNLOCK_OBJ (brc_event);
      POname (clRetainEvent) (target_event);

      pocl_lock_events_inorder (brc_event, target_event);
      if (target != brc_event->notify_list)
        {
          pocl_unlock_events_inorder (brc_event, target_event);
          POCL_LOCK_OBJ (brc_event);
          continue;
        }

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

        if (pocl_is_tracing_enabled() && target->event->meta_data)
          {
            pocl_event_md *md = target->event->meta_data;
            for (size_t i = 0; i < md->num_deps; ++i)
              if (md->dep_ids[i] == brc_event->id)
                {
                  md->dep_ts[i] = brc_event->time_end;
                  break;
                }
          }
        LL_DELETE (brc_event->notify_list, target);
        pocl_unlock_events_inorder (brc_event, target_event);
        POname (clReleaseEvent) (target->event);
        pocl_mem_manager_free_event_node (target);
        POCL_LOCK_OBJ (brc_event);
    }
  POCL_UNLOCK_OBJ (brc_event);
}

/**
 * Populates the device specific sampler data structure used by kernel
 * from given kernel sampler argument
 */
void
pocl_fill_dev_sampler_t (dev_sampler_t *ds, struct pocl_argument *parg)
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

/* Returns the width of the widest dimension in the grid of the given
   run command. */
size_t
pocl_cmd_max_grid_dim_width (_cl_command_run *cmd)
{
  return max (max (cmd->pc.local_size[0] * cmd->pc.num_groups[0],
                   cmd->pc.local_size[1] * cmd->pc.num_groups[1]),
              cmd->pc.local_size[2] * cmd->pc.local_size[2]);
}


/* CPU driver stuff */

#ifdef HAVE_DLFCN_H

typedef struct pocl_dlhandle_cache_item pocl_dlhandle_cache_item;
struct pocl_dlhandle_cache_item
{
  pocl_kernel_hash_t hash;

  /* The specialization properties. */
  /* The local dimensions. */
  size_t local_wgs[3];
  /* If global offset must be zero for this WG function version. */
  int goffs_zero;
  int specialize;
  /* Maximum grid dimension this WG function works with. */
  size_t max_grid_dim_width;

  void *wg;
  void *dlhandle;
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
        && (ci->local_wgs[0] == cmd->command.run.pc.local_size[0])
        && (ci->local_wgs[1] == cmd->command.run.pc.local_size[1])
        && (ci->local_wgs[2] == cmd->command.run.pc.local_size[2]))
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
 * @param command The kernel run command.
 * @param specialized 1 if should check the per-command specialized one instead
 * of the generic one.
 * @returns The filename of the built binary in the disk.
 */
char *
pocl_check_kernel_disk_cache (_cl_command_node *command, int specialized)
{
  char *module_fn = NULL;
  _cl_command_run *run_cmd = &command->command.run;
  cl_kernel k = run_cmd->kernel;
  cl_program p = k->program;
  unsigned dev_i = command->program_device_i;

  /* First try to find a static WG binary for the local size as they
     are always more efficient than the dynamic ones.  Also, in case
     of reqd_wg_size, there might not be a dynamic sized one at all.  */
  module_fn = malloc (POCL_MAX_PATHNAME_LENGTH);
  pocl_cache_final_binary_path (module_fn, p, dev_i, k, command, specialized);

  if (pocl_exists (module_fn))
    {
      POCL_MSG_PRINT_INFO ("Using a cached WG function: %s\n", module_fn);
      return module_fn;
    }

  /* static WG binary for the local size does not exist. If we have the LLVM IR
   * (program.bc), try to compile a new parallel.bc and static binary */
  if (p->binaries[dev_i])
    {
#ifdef ENABLE_LLVM
      POCL_LOCK (pocl_llvm_codegen_lock);
      int error = llvm_codegen (module_fn, dev_i, k, command->device, command,
                                specialized);
      POCL_UNLOCK (pocl_llvm_codegen_lock);
      if (error)
        POCL_ABORT ("Final linking of kernel %s failed.\n", k->name);
      POCL_MSG_PRINT_INFO ("Built a %sWG function: %s\n",
                           specialized ? "specialized " : "generic ",
                           module_fn);
      return module_fn;
#else
      /* TODO: This should be caught earlier. */
      if (!p->pocl_binaries[dev_i])
        POCL_ABORT ("pocl device without online compilation support"
                    " cannot compile LLVM IRs to machine code!\n");
#endif
    }
  else
    {
      module_fn = malloc (POCL_MAX_PATHNAME_LENGTH);
      /* First try to find a specialized WG binary, if allowed by the
         command. */
      if (!run_cmd->force_generic_wg_func)
        pocl_cache_final_binary_path (module_fn, p, dev_i, k, command, 1);

      if (run_cmd->force_generic_wg_func || !pocl_exists (module_fn))
        {
          /* Then check for a dynamic (non-specialized) kernel. */
          pocl_cache_final_binary_path (module_fn, p, dev_i, k, command, 0);
          if (!pocl_exists (module_fn))
            POCL_ABORT ("Generic WG function binary does not exist.\n");
          POCL_MSG_PRINT_INFO ("Using a cached generic WG function: %s\n",
                               module_fn);
        }
      else
        POCL_MSG_PRINT_INFO ("Using a cached specialized WG function: %s\n",
                             module_fn);
    }
  return module_fn;
}


/* Look for a dlhandle in the dlhandle cache for the given kernel command.
   If found, push the handle up in the cache to improve cache hit speed,
   and return it. Otherwise return NULL. The caller should hold
   pocl_dlhandle_lock. */
static pocl_dlhandle_cache_item *
fetch_dlhandle_cache_item (_cl_command_run *run_cmd, int specialize)
{
  pocl_dlhandle_cache_item *ci = NULL, *tmp = NULL;
  size_t max_grid_width = pocl_cmd_max_grid_dim_width (run_cmd);
  DL_FOREACH_SAFE (pocl_dlhandle_cache, ci, tmp)
  {
    if ((memcmp (ci->hash, run_cmd->hash, sizeof (pocl_kernel_hash_t)) == 0)
        && (ci->local_wgs[0] == run_cmd->pc.local_size[0])
        && (ci->local_wgs[1] == run_cmd->pc.local_size[1])
        && (ci->local_wgs[2] == run_cmd->pc.local_size[2])
        && (max_grid_width <= ci->max_grid_dim_width)
        && (ci->specialize == specialize)
        && (ci->goffs_zero == (run_cmd->pc.global_offset[0] == 0
                && run_cmd->pc.global_offset[1] == 0
                && run_cmd->pc.global_offset[2] == 0)))
      {
        /* move to the front of the line */
        DL_DELETE (pocl_dlhandle_cache, ci);
        DL_PREPEND (pocl_dlhandle_cache, ci);
        run_cmd->wg = ci->wg;
        return ci;
      }
  }
  return NULL;
}

/**
 * Checks if the kernel command has been built and has been loaded with
 * dlopen, and reuses its handle. If not, checks if a built binary is found
 * in the disk, if not, builds the kernel and puts it to respective
 * caches.
 *
 * if handle already exists: if the retain argument is given,
 * the refcount is increased, otherwise it's kept unchanged.
 * if handle doesn't exist: if the retain argument is given,
 * refcount is set to 1, otherwise it's set to 0.
 * This can be useful in case we're just pre-compiling kernels
 * (or compiling them for binaries), and not actually need them immediately.
 *
 * TODO: This function is really specific to CPU (host) drivers since dlhandles
 * imply program loading to the same process as the host. Move to basic.c? */
void
pocl_check_kernel_dlhandle_cache (_cl_command_node *command,
                                  int retain, int specialize)
{
  char workgroup_string[WORKGROUP_STRING_LENGTH];
  pocl_dlhandle_cache_item *ci = NULL;
  const char *dl_error = NULL;
  _cl_command_run *run_cmd = &command->command.run;

  /* Brute force mechanism to test relying on generic work-group functions
     only. */
  if (!pocl_get_bool_option("POCL_WORK_GROUP_SPECIALIZATION", 1))
    specialize = 0;

  POCL_LOCK (pocl_dlhandle_lock);
  ci = fetch_dlhandle_cache_item (run_cmd, specialize);
  if (ci != NULL)
    {
      if (retain) ++ci->ref_count;
      POCL_UNLOCK (pocl_dlhandle_lock);
      return;
    }

  /* Not found, build a new kernel and cache its dlhandle. */
  ci = get_new_dlhandle_cache_item ();
  memcpy (ci->hash, run_cmd->hash, sizeof (pocl_kernel_hash_t));
  ci->local_wgs[0] = run_cmd->pc.local_size[0];
  ci->local_wgs[1] = run_cmd->pc.local_size[1];
  ci->local_wgs[2] = run_cmd->pc.local_size[2];
  ci->ref_count = retain ? 1 : 0;
  ci->specialize = specialize;
  ci->goffs_zero = run_cmd->pc.global_offset[0] == 0
                   && run_cmd->pc.global_offset[1] == 0
                   && run_cmd->pc.global_offset[2] == 0;

  size_t max_grid_width = pocl_cmd_max_grid_dim_width (run_cmd);
  ci->max_grid_dim_width = max_grid_width;

  char *module_fn = pocl_check_kernel_disk_cache (command, specialize);

  // reset possibly existing error from calls from an ICD loader
  (void)dlerror();
  ci->dlhandle = dlopen (module_fn, RTLD_NOW | RTLD_LOCAL);
  dl_error = dlerror ();

  if (ci->dlhandle == NULL || dl_error != NULL)
    POCL_ABORT ("dlopen(\"%s\") failed with '%s'.\n"
                "note: missing symbols in the kernel binary might be"
                " reported as 'file not found' errors.\n",
                module_fn, dl_error);

  snprintf (workgroup_string, WORKGROUP_STRING_LENGTH,
            "_pocl_kernel_%s_workgroup", run_cmd->kernel->name);

  ci->wg = dlsym (ci->dlhandle, workgroup_string);
  dl_error = dlerror ();

  if (ci->wg == NULL || dl_error != NULL)
    {
      // Older OSX dyld APIs need the name without the underscore.
      snprintf (workgroup_string, WORKGROUP_STRING_LENGTH,
                "pocl_kernel_%s_workgroup", run_cmd->kernel->name);
      ci->wg = dlsym (ci->dlhandle, workgroup_string);
      dl_error = dlerror ();

      if (ci->wg == NULL || dl_error != NULL)
        POCL_ABORT ("dlsym(\"%s\", \"%s\") failed with '%s'.\n"
                    "note: missing symbols in the kernel binary might be"
                    " reported as 'file not found' errors.\n",
                    module_fn, workgroup_string, dl_error);
    }

  run_cmd->wg = ci->wg;
  DL_PREPEND (pocl_dlhandle_cache, ci);

  POCL_UNLOCK (pocl_dlhandle_lock);
  POCL_MEM_FREE (module_fn);
}

#endif


#define MIN_MAX_MEM_ALLOC_SIZE (128*1024*1024)

/* accounting object for the main memory */
static pocl_global_mem_t system_memory = {POCL_LOCK_INITIALIZER, 0, 0, 0};

void
pocl_setup_device_for_system_memory (cl_device_id device)
{
  /* set up system memory limits, if required */
  if (system_memory.total_alloc_limit == 0)
  {
      /* global_mem_size contains the entire memory size,
       * and we need to leave some available for OS & other programs
       * this sets it to 3/4 for systems with <=7gig mem,
       * for >7 it sets to (total-2gigs)
       */
      cl_ulong alloc_limit = device->global_mem_size;
      if (alloc_limit > ((cl_ulong)7 << 30))
        system_memory.total_alloc_limit = alloc_limit - ((cl_ulong)2 << 30);
      else
        {
          cl_ulong temp = (alloc_limit >> 2);
          system_memory.total_alloc_limit = alloc_limit - temp;
        }

      system_memory.max_ever_allocated =
          system_memory.currently_allocated = 0;

      /* in some cases (e.g. ARM32 pocl on ARM64 system with >4G ram),
       * global memory is correctly reported but larger than can be
       * used; limit to pointer size */
      if (system_memory.total_alloc_limit > UINTPTR_MAX)
        system_memory.total_alloc_limit = UINTPTR_MAX;

      /* apply rlimit settings */
#ifdef HAVE_GETRLIMIT
      struct rlimit limits;
      int ret = getrlimit (RLIMIT_DATA, &limits);
      if ((ret == 0) && (system_memory.total_alloc_limit > limits.rlim_cur))
        system_memory.total_alloc_limit = limits.rlim_cur;
#endif
  }

  device->global_mem_size = system_memory.total_alloc_limit;

  int limit_memory_gb = pocl_get_int_option ("POCL_MEMORY_LIMIT", 0);
  if (limit_memory_gb > 0)
    {
      cl_ulong limited_memory = (cl_ulong)limit_memory_gb << 30;
      if (device->global_mem_size > limited_memory)
        device->global_mem_size = limited_memory;
      else
        POCL_MSG_WARN ("requested POCL_MEMORY_LIMIT %i GBs is larger than"
                       " physical memory size (%u) GBs, ignoring\n",
                       limit_memory_gb,
                       (unsigned)(device->global_mem_size >> 30));
    }

  if (device->global_mem_size < MIN_MAX_MEM_ALLOC_SIZE)
    POCL_ABORT("Not enough memory to run on this device.\n");

  /* Maximum allocation size: we don't have hardware limits, so we
   * can potentially allocate the whole memory for a single buffer, unless
   * of course there are limits set at the operating system level. Of course
   * we still have to respect the OpenCL-commanded minimum */

  cl_ulong alloc_limit = pocl_size_ceil2_64 (device->global_mem_size / 4);

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

  assert (device->global_mem_size > 0);
  assert (device->max_compute_units > 0);
  assert (device->max_mem_alloc_size > 0);

  /* these should be ideally setup by hwloc or proc/cpuinfo;
   * if not, set them to some reasonable values
   */
  if (device->local_mem_size == 0)
    {
      cl_ulong s = pocl_size_ceil2_64 (device->global_mem_size / 1024);
      s = min (s, 512UL * 1024);
      device->local_mem_size = s;
      device->max_constant_buffer_size = s;
    }

  /* OpenCL 3.0 mandates at least 64KB for CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
   * and 32KB for CL_DEVICE_LOCAL_MEM_SIZE. pocl_topology tries to use size of
   * largest non-shared cache (usually L2), but some CPUs don't have L3
   * and the only non-shared cache is L1, which can be too small. */
  if (device->version_as_int > 299)
    {
      if (device->local_mem_size < 32 * 1024)
        device->local_mem_size = 32 * 1024;
      if (device->max_constant_buffer_size < 64 * 1024)
        device->max_constant_buffer_size = 64 * 1024;
    }

  /* set program scope variable device limits.
   * only the max_size is an actual limit.
   * for CPU devices there is no hardware limit.
   * TODO what should we set them to ?
   * setting this to >= 2^16 causes LLVM to crash in SDNode */
  if (device->run_program_scope_variables_pass)
    {
      device->global_var_max_size = 64 * 1000;
      device->global_var_pref_size = max(64 * 1000, device->max_constant_buffer_size);
    }

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
                    "____ Total available system memory  : %10" PRIu64 " KB\n"
                    " ____ Currently used system memory   : %10" PRIu64 " KB\n"
                    " ____ Max used system memory         : %10" PRIu64
                    " KB\n",
                    system_memory.total_alloc_limit >> 10,
                    system_memory.currently_allocated >> 10,
                    system_memory.max_ever_allocated >> 10);
}

/* default WG size in each dimension & total WG size.
 * this should be reasonable for CPU */
#define DEFAULT_WG_SIZE 4096

static const char *final_ld_flags[] =
  {"-lm", "-nostartfiles", HOST_LD_FLAGS_ARRAY, NULL};

static cl_device_partition_property basic_partition_properties[1] = { 0 };

#ifdef ENABLE_CONFORMANCE
static const cl_image_format supported_image_formats[] = {
  { CL_RGBA, CL_SNORM_INT8 },
  { CL_RGBA, CL_SNORM_INT16 },
  { CL_RGBA, CL_UNORM_INT8 },
  { CL_RGBA, CL_UNORM_INT16 },
  { CL_RGBA, CL_SIGNED_INT8 },
  { CL_RGBA, CL_SIGNED_INT16 },
  { CL_RGBA, CL_SIGNED_INT32 },
  { CL_RGBA, CL_UNSIGNED_INT8 },
  { CL_RGBA, CL_UNSIGNED_INT16 },
  { CL_RGBA, CL_UNSIGNED_INT32 },
  { CL_RGBA, CL_HALF_FLOAT },
  { CL_RGBA, CL_FLOAT },
  { CL_BGRA, CL_SNORM_INT8 },
  { CL_BGRA, CL_UNORM_INT8 },
  { CL_BGRA, CL_SIGNED_INT8 },
  { CL_BGRA, CL_UNSIGNED_INT8 }
};
#else
static const cl_image_format supported_image_formats[] = {
  { CL_A, CL_SNORM_INT8 },
  { CL_A, CL_SNORM_INT16 },
  { CL_A, CL_UNORM_INT8 },
  { CL_A, CL_UNORM_INT16 },
  { CL_A, CL_SIGNED_INT8 },
  { CL_A, CL_SIGNED_INT16 },
  { CL_A, CL_SIGNED_INT32 },
  { CL_A, CL_UNSIGNED_INT8 },
  { CL_A, CL_UNSIGNED_INT16 },
  { CL_A, CL_UNSIGNED_INT32 },
  { CL_A, CL_HALF_FLOAT },
  { CL_A, CL_FLOAT },
  { CL_R, CL_SNORM_INT8 },
  { CL_R, CL_SNORM_INT16 },
  { CL_R, CL_UNORM_INT8 },
  { CL_R, CL_UNORM_INT16 },
  { CL_R, CL_SIGNED_INT8 },
  { CL_R, CL_SIGNED_INT16 },
  { CL_R, CL_SIGNED_INT32 },
  { CL_R, CL_UNSIGNED_INT8 },
  { CL_R, CL_UNSIGNED_INT16 },
  { CL_R, CL_UNSIGNED_INT32 },
  { CL_R, CL_HALF_FLOAT },
  { CL_R, CL_FLOAT },
  { CL_RG, CL_SNORM_INT8 },
  { CL_RG, CL_SNORM_INT16 },
  { CL_RG, CL_UNORM_INT8 },
  { CL_RG, CL_UNORM_INT16 },
  { CL_RG, CL_SIGNED_INT8 },
  { CL_RG, CL_SIGNED_INT16 },
  { CL_RG, CL_SIGNED_INT32 },
  { CL_RG, CL_UNSIGNED_INT8 },
  { CL_RG, CL_UNSIGNED_INT16 },
  { CL_RG, CL_UNSIGNED_INT32 },
  { CL_RG, CL_HALF_FLOAT },
  { CL_RG, CL_FLOAT },
  { CL_RGBA, CL_SNORM_INT8 },
  { CL_RGBA, CL_SNORM_INT16 },
  { CL_RGBA, CL_UNORM_INT8 },
  { CL_RGBA, CL_UNORM_INT16 },
  { CL_RGBA, CL_SIGNED_INT8 },
  { CL_RGBA, CL_SIGNED_INT16 },
  { CL_RGBA, CL_SIGNED_INT32 },
  { CL_RGBA, CL_UNSIGNED_INT8 },
  { CL_RGBA, CL_UNSIGNED_INT16 },
  { CL_RGBA, CL_UNSIGNED_INT32 },
  { CL_RGBA, CL_HALF_FLOAT },
  { CL_RGBA, CL_FLOAT },
  { CL_ARGB, CL_SNORM_INT8 },
  { CL_ARGB, CL_UNORM_INT8 },
  { CL_ARGB, CL_SIGNED_INT8 },
  { CL_ARGB, CL_UNSIGNED_INT8 },
  { CL_BGRA, CL_SNORM_INT8 },
  { CL_BGRA, CL_UNORM_INT8 },
  { CL_BGRA, CL_SIGNED_INT8 },
  { CL_BGRA, CL_UNSIGNED_INT8 }
};
#endif

void
pocl_init_default_device_infos (cl_device_id dev,
                                const char *device_extensions)
{
  size_t i;

  dev->type = CL_DEVICE_TYPE_CPU;
  dev->max_work_item_dimensions = 3;
  dev->final_linkage_flags = final_ld_flags;
  dev->extensions = device_extensions;

  /*
    The hard restriction will be the context data which is
    stored in stack that can be as small as 8K in Linux.
    Thus, there should be enough work-items alive to fill up
    the SIMD lanes times the vector units, but not more than
    that to avoid stack overflow and cache trashing.
  */
  int max_wg
      = pocl_get_int_option ("POCL_MAX_WORK_GROUP_SIZE", DEFAULT_WG_SIZE);
  assert (max_wg > 0);
  max_wg = min (max_wg, DEFAULT_WG_SIZE);
  if (max_wg < 0)
    max_wg = DEFAULT_WG_SIZE;

  dev->max_work_item_sizes[0] = dev->max_work_item_sizes[1]
      = dev->max_work_item_sizes[2] = dev->max_work_group_size = max_wg;

  dev->preferred_wg_size_multiple = 8;
#ifdef ENABLE_LLVM
  cpu_setup_vector_widths (dev);
#else
  dev->preferred_vector_width_char = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR;
  dev->preferred_vector_width_short = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT;
  dev->preferred_vector_width_int = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT;
  dev->preferred_vector_width_long = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG;
  dev->preferred_vector_width_float = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT;
  /* TODO: figure out what the difference between preferred and native widths are */
  dev->native_vector_width_char = POCL_DEVICES_NATIVE_VECTOR_WIDTH_CHAR;
  dev->native_vector_width_short = POCL_DEVICES_NATIVE_VECTOR_WIDTH_SHORT;
  dev->native_vector_width_int = POCL_DEVICES_NATIVE_VECTOR_WIDTH_INT;
  dev->native_vector_width_long = POCL_DEVICES_NATIVE_VECTOR_WIDTH_LONG;
  dev->native_vector_width_float = POCL_DEVICES_NATIVE_VECTOR_WIDTH_FLOAT;

  if (strstr (device_extensions, "cl_khr_fp64") == NULL)
    {
      dev->native_vector_width_double = 0;
      dev->preferred_vector_width_double = 0;
    }
  else
    {
      dev->native_vector_width_double
          = POCL_DEVICES_NATIVE_VECTOR_WIDTH_DOUBLE;
      dev->preferred_vector_width_double
          = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE;
    }

  if (strstr (device_extensions, "cl_khr_fp16") == NULL)
    {
      dev->preferred_vector_width_half = 0;
      dev->native_vector_width_half = 0;
    }
  else
    {
      dev->preferred_vector_width_half
          = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF;
      dev->native_vector_width_half = POCL_DEVICES_NATIVE_VECTOR_WIDTH_HALF;
    }
#endif

  dev->grid_width_specialization_limit = USHRT_MAX;
  dev->address_bits = HOST_DEVICE_ADDRESS_BITS;
  dev->image_support = CL_TRUE;
  /* Use the minimum values until we get a more sensible upper limit from
     somewhere. */
  dev->max_read_image_args = dev->max_write_image_args = 128;
#ifdef ENABLE_CONFORMANCE
  dev->max_read_write_image_args = 0;
#else
  dev->max_read_write_image_args = 128;
#endif
  dev->image2d_max_width = dev->image2d_max_height = 8192;
  dev->image3d_max_width = dev->image3d_max_height = dev->image3d_max_depth = 2048;
  dev->max_samplers = 16;

  for (i = 0; i < NUM_OPENCL_IMAGE_TYPES; ++i)
    {
      dev->num_image_formats[i]
          = sizeof (supported_image_formats) / sizeof (cl_image_format);
      dev->image_formats[i] = supported_image_formats;
    }

  dev->image_max_buffer_size = 65536;
  dev->image_max_array_size = 2048;
  dev->max_constant_args = 8;
  dev->max_mem_alloc_size = 0;
  dev->max_parameter_size = 1024;
  dev->min_data_type_align_size = MAX_EXTENDED_ALIGNMENT;
  dev->mem_base_addr_align = MAX_EXTENDED_ALIGNMENT;
  dev->single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN;
#ifdef __x86_64__
  dev->single_fp_config |= (CL_FP_DENORM | CL_FP_ROUND_TO_INF
                            | CL_FP_ROUND_TO_ZERO
                            | CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT);
#ifdef ENABLE_LLVM
  if (cpu_has_fma())
    dev->single_fp_config |= CL_FP_FMA;
#endif
#endif

  if (strstr (device_extensions, "cl_khr_fp16") == NULL)
    {
      dev->half_fp_config = 0;
    }
  else
    {
      dev->half_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN;
    }

  if (strstr (device_extensions, "cl_khr_fp64") == NULL)
    {
      dev->double_fp_config = 0;
    }
  else
    {
      /* TODO: all of these are the minimum mandated, but not all CPUs may
       * actually support all of them. */
      dev->double_fp_config = CL_FP_FMA | CL_FP_ROUND_TO_NEAREST
                              | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF
                              | CL_FP_INF_NAN | CL_FP_DENORM;
      /* this is a workaround for issue 28 in https://github.com/Oblomov/clinfo
       * https://github.com/Oblomov/clinfo/issues/28 */
      dev->double_fp_config |= CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    }
  dev->global_mem_cache_type = CL_NONE;
  dev->global_mem_cacheline_size = 0;
  dev->global_mem_cache_size = 0;
  dev->global_mem_size = 0;
  dev->max_constant_buffer_size = 0;
  dev->max_constant_args = 8;
  dev->local_mem_type = CL_GLOBAL;
  dev->local_mem_size = 0;
  dev->error_correction_support = CL_FALSE;
  dev->host_unified_memory = CL_TRUE;

  dev->profiling_timer_resolution = pocl_timer_resolution;

  dev->endian_little = !(WORDS_BIGENDIAN);
  dev->compiler_available = CL_TRUE;
  dev->linker_available = CL_TRUE;
  dev->spmd = CL_FALSE;
  dev->arg_buffer_launcher = CL_FALSE;
  dev->grid_launcher = CL_FALSE;
  dev->run_workgroup_pass = CL_TRUE;
  dev->execution_capabilities = CL_EXEC_KERNEL | CL_EXEC_NATIVE_KERNEL;
  dev->platform = 0;

  dev->parent_device = NULL;
  /* These two are only used for subdevices.
   * Each subdevice has these two setup when created.
   * The subdevice will then use these CUs:
   *  [start, start+1, ..., start+count-1]
   * this may not work with more complicated partitioning schemes,
   * but is good enough for now. */
  dev->core_start = 0;
  dev->core_count = 0;
  /* basic does not support partitioning */
  dev->max_sub_devices = 1;
  dev->num_partition_properties = 1;
  dev->partition_properties = basic_partition_properties;
  dev->num_partition_types = 0;
  dev->partition_type = NULL;

  dev->device_side_printf = 1;
  dev->printf_buffer_size = PRINTF_BUFFER_SIZE * 1024;

  dev->vendor = "pocl";
  dev->profile = "FULL_PROFILE";
  /* Note: The specification describes identifiers being delimited by
     only a single space character. Some programs that check the device's
     extension  string assume this rule. Future extension additions should
     ensure that there is no more than a single space between
     identifiers. */
  dev->global_as_id = dev->local_as_id = dev->constant_as_id = 0;

  dev->svm_allocation_priority = 0;
  /* OpenCL 2.0 properties */
  dev->svm_caps = 0;
  /* TODO these are minimums, figure out whats a reasonable value */
  dev->max_events = 0;
  dev->max_queues = 0;

  /* Default pipe support for PoCL devices */
  dev->pipe_support = CL_FALSE;
  /* Specification requires pipe values to be 0, when pipes are not supported
   */
  dev->max_pipe_args = 0;
  dev->max_pipe_active_res = 0;
  dev->max_pipe_packet_size = 0;

  dev->dev_queue_pref_size = 0;
  dev->dev_queue_max_size = 0;
  dev->on_dev_queue_props = 0;
  dev->on_host_queue_props = CL_QUEUE_PROFILING_ENABLE;
  dev->has_64bit_long = 1;
  dev->autolocals_to_args = POCL_AUTOLOCALS_TO_ARGS_ALWAYS;
  dev->device_alloca_locals = 0;
  dev->global_var_max_size = 0;
  dev->global_var_pref_size = 0;
  dev->non_uniform_work_group_support = CL_FALSE;
  dev->max_num_sub_groups = 0;

#ifdef ENABLE_LLVM

  dev->llvm_target_triplet = OCL_KERNEL_TARGET;
  dev->kernellib_fallback_name = NULL;

  char kernellib[POCL_MAX_PATHNAME_LENGTH] = "kernel-";
  char kernellib_fallback[POCL_MAX_PATHNAME_LENGTH];
  strcat(kernellib, dev->llvm_target_triplet);

  strcat(kernellib, "-");
#ifdef KERNELLIB_HOST_DISTRO_VARIANTS
  const char* kernellib_variant = pocl_get_distro_kernellib_variant ();
  dev->llvm_cpu = pocl_get_distro_cpu_name (kernellib_variant);
  strcat(kernellib, kernellib_variant);
  if (!kernellib_variant)
    dev->available = CL_FALSE;
#elif defined(HOST_CPU_FORCED)
  dev->llvm_cpu = OCL_KERNEL_TARGET_CPU;
  strcat(kernellib, OCL_KERNEL_TARGET_CPU);
#else
  dev->llvm_cpu = pocl_get_llvm_cpu_name ();
  strcpy(kernellib_fallback, kernellib);
  strcat(kernellib_fallback, OCL_KERNEL_TARGET_CPU);
  strcat(kernellib, dev->llvm_cpu);
  dev->kernellib_fallback_name = strdup(kernellib_fallback);
#endif
  dev->kernellib_name = strdup(kernellib);
  dev->kernellib_subdir = "host";
  dev->llvm_abi = pocl_get_llvm_cpu_abi ();

#else /* No compiler, no CPU info */
  dev->kernellib_name = NULL;
  dev->kernellib_fallback_name = NULL;
  dev->kernellib_subdir = "host";
  dev->llvm_cpu = NULL;
  dev->llvm_abi = NULL;
  dev->llvm_target_triplet = "";
#endif

#ifdef ENABLE_SPIRV
  dev->supported_spir_v_versions = "SPIR-V_1.2 SPIR-V_1.1 SPIR-V_1.0";
#else
  dev->supported_spir_v_versions = "";
#endif

  /* OpenCL 3.0 properties */
  /* Minimum mandated capability */
  dev->atomic_memory_capabilities = CL_DEVICE_ATOMIC_ORDER_RELAXED
                                    | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP;
  dev->atomic_fence_capabilities = CL_DEVICE_ATOMIC_ORDER_RELAXED
                                    | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
                                    | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP;

  if (dev->llvm_cpu != NULL)
    {
      dev->builtin_kernel_list
          = strdup ("pocl.add.i8;"
                    "org.khronos.openvx.scale_image.nn.u8;"
                    "org.khronos.openvx.scale_image.bl.u8;"
                    "org.khronos.openvx.tensor_convert_depth.wrap.u8.f32");
      dev->num_builtin_kernels = 4;
    }
}

/*
 * Input: a space-delim string, and an array of cl_name_version items
 * (which should have all possible items that could occur in input string).
 * Then for each item in string, finds its corresponding cl_name_version
 * in the array, and copy it to the output array of cl_name_version items.
 * Returns: number of items found
 */

static unsigned
pocl_space_delim_string_to_cl_name_version_array (
    cl_name_version **output, const char *input_str,
    const cl_name_version *search_array, unsigned search_array_size)
{
  unsigned i, start = 0, end = 0, pos = 0, len = 0;
  const char *p = input_str;
  char item_name[256];
  unsigned num_output = 0;
  cl_name_version storage[256];

  while (p[pos])
    {
      while (p[pos] && isspace ((unsigned char)p[pos]))
        pos++;
      start = pos;
      if (p[pos] == 0)
        break;
      while (p[pos] && !isspace ((unsigned char)p[pos]))
        pos++;
      end = pos;
      len = end - start;
      if (len > 255)
        {
          POCL_MSG_WARN ("item too long: %u | %s", len, p + start);
          continue;
        }
      memcpy (item_name, p + start, len);
      item_name[len] = 0;

      int found = -1;
      for (i = 0; i < search_array_size; ++i)
        {
          if (strcmp (item_name, search_array[i].name) == 0)
            {
              found = i;
              break;
            }
        }
      if (found >= 0)
        {
          memcpy (&storage[num_output], &search_array[found],
                  sizeof (cl_name_version));
          ++num_output;
          if (num_output == 256) {
            POCL_MSG_WARN ("max items reached (256)\n");
            break;
          }
        }
      else
        POCL_MSG_WARN ("could not find item: %s\n", item_name);
    }

  if (num_output)
    {
      cl_name_version *tmp = malloc (num_output * sizeof (cl_name_version));
      memcpy (tmp, storage, (num_output * sizeof (cl_name_version)));
      *output = tmp;
    }
  else
    *output = NULL;

  return num_output;
}

static const cl_name_version OPENCL_C_VERSIONS[]
    = { { CL_MAKE_VERSION (1, 0, 0), "OpenCL C" },
        { CL_MAKE_VERSION (1, 1, 0), "OpenCL C" },
        { CL_MAKE_VERSION (1, 2, 0), "OpenCL C" },
        { CL_MAKE_VERSION (3, 0, 0), "OpenCL C" } };

void
pocl_setup_opencl_c_with_version (cl_device_id dev, int supports_30)
{
  dev->opencl_c_with_version = OPENCL_C_VERSIONS;
  dev->num_opencl_c_with_version = supports_30 ? 4 : 3;
}

/* this is a list of recognized extensions, not a list of reported extensions;
   the reported are stored in dev->extensions; this only for versioning */
static const cl_name_version OPENCL_EXTENSIONS[]
  = { { CL_MAKE_VERSION (1, 0, 0), "cl_intel_required_subgroup_size" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_intel_subgroups" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_intel_subgroups_short" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_intel_unified_shared_memory" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_byte_addressable_store" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_global_int32_base_atomics" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_global_int32_extended_atomics" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_local_int32_base_atomics" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_local_int32_extended_atomics" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_int64_base_atomics" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_int64_extended_atomics" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_subgroups" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_subgroup_extended_types" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_subgroup_non_uniform_vote" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_subgroup_ballot" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_subgroup_non_uniform_arithmetic" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_subgroup_shuffle" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_subgroup_shuffle_relative" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_subgroup_clustered_reduce" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_3d_image_writes" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_fp16" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_fp64" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_nv_device_attribute_query" },
      { CL_MAKE_VERSION (2, 0, 0), "cl_khr_depth_images" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_khr_image2d_from_buffer" },
      { CL_MAKE_VERSION (2, 1, 0), "cl_khr_il_program" },

      { CL_MAKE_VERSION (0, 9, 4), "cl_khr_command_buffer" },
      { CL_MAKE_VERSION (0, 9, 1), "cl_khr_command_buffer_multi_device" },
      { CL_MAKE_VERSION (1, 0, 0), "cl_ext_float_atomics" },
      { CL_MAKE_VERSION (0, 1, 0), "cl_ext_buffer_device_address" },
      { CL_MAKE_VERSION (0, 9, 0), "cl_pocl_svm_rect" },
      { CL_MAKE_VERSION (0, 9, 0), "cl_pocl_command_buffer_svm" },
      { CL_MAKE_VERSION (0, 9, 0), "cl_pocl_command_buffer_host_buffer" },
      { CL_MAKE_VERSION (0, 9, 0), "cl_pocl_command_buffer_host_exec" } };

const size_t OPENCL_EXTENSIONS_NUM
    = sizeof (OPENCL_EXTENSIONS) / sizeof (OPENCL_EXTENSIONS[0]);

void
pocl_setup_extensions_with_version (cl_device_id dev)
{
  cl_name_version *tmp = NULL;
  unsigned ret = pocl_space_delim_string_to_cl_name_version_array (
      &tmp, dev->extensions, OPENCL_EXTENSIONS, OPENCL_EXTENSIONS_NUM);

  dev->num_extensions_with_version = ret;
  dev->extensions_with_version = tmp;
}

static const cl_name_version OPENCL_SPIRV_VERSIONS[]
    = { { CL_MAKE_VERSION (1, 0, 0), "SPIR-V" },
        { CL_MAKE_VERSION (1, 1, 0), "SPIR-V" },
        { CL_MAKE_VERSION (1, 2, 0), "SPIR-V" },
        { CL_MAKE_VERSION (1, 3, 0), "SPIR-V" },
        { CL_MAKE_VERSION (1, 4, 0), "SPIR-V" },
        { CL_MAKE_VERSION (1, 5, 0), "SPIR-V" } };

const size_t OPENCL_SPIRV_VERSIONS_NUM
    = sizeof (OPENCL_SPIRV_VERSIONS) / sizeof (OPENCL_SPIRV_VERSIONS[0]);

void
pocl_setup_ils_with_version (cl_device_id dev)
{
  unsigned i, start = 0, end = 0, pos = 0, len = 0;
  const char *p = dev->supported_spir_v_versions;
  char item_name[256];
  unsigned num_output = 0;
  cl_name_version storage[256];

  while (p[pos])
    {
      while (p[pos] && isspace ((unsigned char)p[pos]))
        pos++;
      start = pos;
      if (p[pos] == 0)
        break;
      while (p[pos] && !isspace ((unsigned char)p[pos]))
        pos++;
      end = pos;
      len = end - start;
      if (len > 255)
        {
          POCL_MSG_WARN ("item too long: %u | %s", len, p + start);
          continue;
        }
      memcpy (item_name, p + start, len);
      item_name[len] = 0;

      cl_version V = CL_MAKE_VERSION (0, 0, 0);
      if (strncmp (item_name, "SPIR-V_1.", 9) == 0)
        {
          unsigned minor = item_name[9] - '0';
          V = CL_MAKE_VERSION (1, minor, 0);
        }
      int found = -1;
      for (i = 0; i < OPENCL_SPIRV_VERSIONS_NUM; ++i)
        {
          if (OPENCL_SPIRV_VERSIONS[i].version == V)
            {
              found = i;
              break;
            }
        }
      if (found >= 0)
        {
          memcpy (&storage[num_output], &OPENCL_SPIRV_VERSIONS[found],
                  sizeof (cl_name_version));
          ++num_output;
          if (num_output >= 256)
            break;
        }
      else
        POCL_MSG_WARN ("could not find item: %s\n", item_name);
    }

  if (num_output)
    {
      cl_name_version *tmp = malloc (num_output * sizeof (cl_name_version));
      memcpy (tmp, storage, (num_output * sizeof (cl_name_version)));
      dev->ils_with_version = tmp;
      dev->num_ils_with_version = num_output;
    }
}

static const cl_name_version OPENCL_C_FEATURES[] = {
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_3d_image_writes" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_images" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_read_write_images" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_atomic_order_acq_rel" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_atomic_order_seq_cst" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_atomic_scope_device" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_atomic_scope_all_devices" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_fp16" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_fp64" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_int64" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_program_scope_global_variables" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_generic_address_space" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_subgroups" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_work_group_collective_functions" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_ext_fp16_global_atomic_add" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_ext_fp32_global_atomic_add" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_ext_fp64_global_atomic_add" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_ext_fp16_local_atomic_add" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_ext_fp32_local_atomic_add" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_ext_fp64_local_atomic_add" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_ext_fp16_global_atomic_min_max" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_ext_fp32_global_atomic_min_max" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_ext_fp64_global_atomic_min_max" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_ext_fp16_local_atomic_min_max" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_ext_fp32_local_atomic_min_max" },
  { CL_MAKE_VERSION (3, 0, 0), "__opencl_c_ext_fp64_local_atomic_min_max" },
};

const size_t OPENCL_C_FEATURES_NUM
    = sizeof (OPENCL_C_FEATURES) / sizeof (OPENCL_C_FEATURES[0]);

void
pocl_setup_features_with_version (cl_device_id dev)
{
  cl_name_version *tmp = NULL;
  unsigned ret = pocl_space_delim_string_to_cl_name_version_array (
      &tmp, dev->features, OPENCL_C_FEATURES, OPENCL_C_FEATURES_NUM);

  dev->num_opencl_features_with_version = ret;
  dev->opencl_features_with_version = tmp;
}

void
pocl_setup_builtin_kernels_with_version (cl_device_id dev)
{
  if (dev->num_builtin_kernels == 0)
    return;

  assert (dev->builtin_kernel_list != NULL);

  dev->builtin_kernels_with_version
      = malloc (dev->num_builtin_kernels * sizeof (cl_name_version));
  assert (dev->builtin_kernels_with_version);

  char *temp = strdup (dev->builtin_kernel_list);
  char *token;
  char *rest = temp;

  unsigned i = 0;
  while ((token = strtok_r (rest, ";", &rest)))
    {
      // The builtin kernel name stored here can only be the
      // maximum of CL_NAME_VERSION_MAX_NAME_SIZE - 1.
      if (strlen (token) >= CL_NAME_VERSION_MAX_NAME_SIZE)
        {
          POCL_MSG_WARN ("Built-in kernel name cannot fit in to the "
                         "cl_name_version array. Length of built-in kernel "
                         "name is %zu, and the concatenated length is %d\n",
                         strlen (token), CL_NAME_VERSION_MAX_NAME_SIZE - 1);
          token[CL_NAME_VERSION_MAX_NAME_SIZE - 1] = '\0';
        }
      strncpy (dev->builtin_kernels_with_version[i].name, token,
               CL_NAME_VERSION_MAX_NAME_SIZE);

      /* proper versioning could use pocl_BIDescriptors.
       * For now, hardcode the version to 1.2 */
      dev->builtin_kernels_with_version[i].version = CL_MAKE_VERSION (1, 2, 0);
      i++;
    }
  free (temp);

  if (i != dev->num_builtin_kernels)
    {
      POCL_ABORT ("Builtin kernels with version list construction failed. "
                  "There are %u built-in kernels, but only %u were found\n",
                  dev->num_builtin_kernels, i);
    }
}
