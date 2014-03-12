/* OpenCL runtime library: clFinish()

   Copyright (c) 2011 Erik Schnetter
   
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

#include "pocl_cl.h"
#include "pocl_util.h"
#include "pocl_image_util.h"
#include "utlist.h"
#include "clEnqueueMapBuffer.h"

static void exec_commands (_cl_command_node *node_list);

CL_API_ENTRY cl_int CL_API_CALL
POname(clFinish)(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
  int i;
  _cl_command_node *node;
  cl_command_queue queue_at_hand;
  _cl_command_node *ready_list = NULL;
  cl_bool command_ready;
  cl_event *event;
  
  if (command_queue->properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    POCL_ABORT_UNIMPLEMENTED();

  /* loop until queue given as parameter is empty. If command in queue has 
     unsubmitted/uncomplete event move on to command_queue assosiated with that
     event */
  queue_at_hand = command_queue;
  while (command_queue->root != NULL)
    {
      node = queue_at_hand->root;
      if (node == NULL)
        {
          queue_at_hand = command_queue;
          continue;
        }
      command_ready = CL_TRUE;

      /* if command has no wait_list -> it's ready */
      if (node->event_wait_list != NULL)
        {
          for(i = 0; i < node->num_events_in_wait_list; ++i)
            {
              if (node->event_wait_list[i]->status != CL_COMPLETE &&
                  node->event_wait_list[i]->status != CL_SUBMITTED)
                {
                  /* move to handle command queue with unqueued/unfinished
                     prequisite event. Current node cannot be added to ready 
                     list */
                  queue_at_hand = node->event_wait_list[i]->queue;
                  command_ready = CL_FALSE;
                }
            }
        }
      if (command_ready == CL_TRUE)
        {
          if (node->event)
            {
              event = &(node->event);
              POCL_UPDATE_EVENT_SUBMITTED(event, command_queue);
            }
          /* remove command from queue and add to ready_list */
          LL_DELETE (queue_at_hand->root, node);
          LL_APPEND (ready_list, node);
        }
    }

  exec_commands(ready_list);

  return CL_SUCCESS;
}
POsym(clFinish)

static void exec_commands (_cl_command_node *node_list)
{
  int i;
  cl_event *event = NULL;
  _cl_command_node *node;
  cl_command_queue command_queue = NULL;
  event_callback_item* cb_ptr;
  
  LL_FOREACH (node_list, node)
    {
      event = &(node->event);
      /* Command queue is needed for POCL_UPDATE_EVENT macros */
      if (node->event)
        command_queue = node->event->queue;

      if (node->device->ops->compile_submitted_kernels)
        node->device->ops->compile_submitted_kernels (node);

      switch (node->type)
        {
        case CL_COMMAND_READ_BUFFER:
          POCL_UPDATE_EVENT_RUNNING(event, command_queue);
          node->device->ops->read
            (node->device->data, 
             node->command.read.host_ptr, 
             node->command.read.device_ptr, 
             node->command.read.cb); 
          POCL_UPDATE_EVENT_COMPLETE(event, command_queue);
          POname(clReleaseMemObject) (node->command.read.buffer);
          break;
        case CL_COMMAND_WRITE_BUFFER:
          POCL_UPDATE_EVENT_RUNNING(event, command_queue);
          node->device->ops->write
            (node->device->data, 
             node->command.write.host_ptr, 
             node->command.write.device_ptr, 
             node->command.write.cb);
          POCL_UPDATE_EVENT_COMPLETE(event, command_queue);
          POname(clReleaseMemObject) (node->command.write.buffer);
          break;
        case CL_COMMAND_COPY_BUFFER:
          POCL_UPDATE_EVENT_RUNNING(event, command_queue);
          node->device->ops->copy
            (node->command.copy.data, 
             node->command.copy.src_ptr, 
             node->command.copy.dst_ptr,
             node->command.copy.cb);
          POCL_UPDATE_EVENT_COMPLETE(event, command_queue);
          POname(clReleaseMemObject) (node->command.copy.src_buffer);
          POname(clReleaseMemObject) (node->command.copy.dst_buffer);
          break;
        case CL_COMMAND_MAP_IMAGE:
        case CL_COMMAND_MAP_BUFFER: 
          POCL_UPDATE_EVENT_RUNNING(event, command_queue);            
          pocl_map_mem_cmd (node->device, node->command.map.buffer, 
                            node->command.map.mapping);
          POCL_UPDATE_EVENT_COMPLETE(event, command_queue);
          break;
        case CL_COMMAND_WRITE_IMAGE:
          POCL_UPDATE_EVENT_RUNNING(event, command_queue); 
          node->device->ops->write_rect 
            (node->device->data, node->command.rw_image.host_ptr,
             node->command.rw_image.device_ptr, node->command.rw_image.origin,
             node->command.rw_image.origin, node->command.rw_image.region, 
             node->command.rw_image.rowpitch, 
             node->command.rw_image.slicepitch,
             node->command.rw_image.rowpitch,
             node->command.rw_image.slicepitch);
          POCL_UPDATE_EVENT_COMPLETE(event, command_queue);
          break;
        case CL_COMMAND_READ_IMAGE:
          POCL_UPDATE_EVENT_RUNNING(event, command_queue); 
          node->device->ops->read_rect 
            (node->device->data, node->command.rw_image.host_ptr,
             node->command.rw_image.device_ptr, node->command.rw_image.origin,
             node->command.rw_image.origin, node->command.rw_image.region, 
             node->command.rw_image.rowpitch, 
             node->command.rw_image.slicepitch,
             node->command.rw_image.rowpitch,
             node->command.rw_image.slicepitch);
          POCL_UPDATE_EVENT_COMPLETE(event, command_queue);
          break;
        case CL_COMMAND_UNMAP_MEM_OBJECT:
          POCL_UPDATE_EVENT_RUNNING(event, command_queue);
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
              /* TODO: fixme. The offset computation must be done at the device 
                 driver. */
              if (node->device->ops->unmap_mem != NULL)        
                node->device->ops->unmap_mem
                  (node->device->data, 
                   (node->command.unmap.mapping)->host_ptr, 
                   (node->command.unmap.memobj)->device_ptrs[node->device->dev_id].mem_ptr, 
                   (node->command.unmap.mapping)->size);
            }
          DL_DELETE((node->command.unmap.memobj)->mappings, 
                    node->command.unmap.mapping);
          (node->command.unmap.memobj)->map_count--;
          POCL_UPDATE_EVENT_COMPLETE(event, command_queue);
          break;
        case CL_COMMAND_NDRANGE_KERNEL:
          assert (*event == node->event);
          POCL_UPDATE_EVENT_RUNNING(event, command_queue);
          node->device->ops->run(node->command.run.data, node);
          POCL_UPDATE_EVENT_COMPLETE(event, command_queue);
          for (i = 0; i < node->command.run.arg_buffer_count; ++i)
            {
              cl_mem buf = node->command.run.arg_buffers[i];
              if (buf == NULL) continue;
              /*printf ("### releasing arg %d - the buffer %x of kernel %s\n", i, 
                buf,  node->command.run.kernel->function_name); */
              POname(clReleaseMemObject) (buf);
            }
          free (node->command.run.arg_buffers);
          free (node->command.run.tmp_dir);
          for (i = 0; i < node->command.run.kernel->num_args + 
                 node->command.run.kernel->num_locals; ++i)
            {
              pocl_aligned_free (node->command.run.arguments[i].value);
            }
          free (node->command.run.arguments);
      
          POname(clReleaseKernel)(node->command.run.kernel);
          break;
        case CL_COMMAND_NATIVE_KERNEL:
          POCL_UPDATE_EVENT_RUNNING(event, command_queue);
          node->device->ops->run_native(node->command.native.data, node);
          POCL_UPDATE_EVENT_COMPLETE(event, command_queue);
          for (i = 0; i < node->command.native.num_mem_objects; ++i)
            {
              cl_mem buf = node->command.native.mem_list[i];
              if (buf == NULL) continue;
              POname(clReleaseMemObject) (buf);
            }
          free (node->command.native.mem_list);
          free (node->command.native.args);
	      break;
        case CL_COMMAND_FILL_IMAGE:
          POCL_UPDATE_EVENT_RUNNING(event, command_queue);
          node->device->ops->fill_rect 
            (node->command.fill_image.data, 
             node->command.fill_image.device_ptr,
             node->command.fill_image.buffer_origin,
             node->command.fill_image.region,
             node->command.fill_image.rowpitch, 
             node->command.fill_image.slicepitch,
             node->command.fill_image.fill_pixel,
             node->command.fill_image.pixel_size);
          free(node->command.fill_image.fill_pixel);
          POCL_UPDATE_EVENT_COMPLETE(event, command_queue);
          break;
        case CL_COMMAND_MARKER:
          POCL_UPDATE_EVENT_RUNNING(event, command_queue);
          POCL_UPDATE_EVENT_COMPLETE(event, command_queue);
          break;
        default:
          POCL_ABORT_UNIMPLEMENTED();
          break;
        }   
    }

  /* event callback handling 
     just call functions in the same order they were added */
  if (event)
    {
      for (cb_ptr = (*event)->callback_list; cb_ptr; cb_ptr = cb_ptr->next)
        {
          cb_ptr->callback_function ((*event), cb_ptr->trigger_status, 
                                     cb_ptr->user_data);
        }
      clReleaseEvent(*event);
    }
  
  // free the queue contents
  node = node_list;
  node_list = NULL;
  while (node)
    {
      _cl_command_node *tmp;
      tmp = node->next;
      pocl_mem_manager_free_command (node);
      node = tmp;
    }
}
