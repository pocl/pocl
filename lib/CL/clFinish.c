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

static void exec_commands_in_queue_until_event(cl_command_queue queue, 
                                               cl_event event);

CL_API_ENTRY cl_int CL_API_CALL
POname(clFinish)(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
  int i;
  _cl_command_node *node;

  if (command_queue->properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    POCL_ABORT_UNIMPLEMENTED();

  exec_commands_in_queue_until_event(command_queue, NULL);

  // free the queue contents
  /*node = command_queue->root;
  command_queue->root = NULL;
  while (node)
    {
      _cl_command_node *tmp;
      tmp = node->next;
      free (node);
      node = tmp;
      }*/  

  return CL_SUCCESS;
}
POsym(clFinish)

static void exec_command (cl_command_queue command_queue, 
                          _cl_command_node *node)
{
  int i;
  cl_event *event = &node->event;
  
  switch (node->type)
    {
    case CL_COMMAND_READ_BUFFER:
      POCL_UPDATE_EVENT_SUBMITTED;
      POCL_UPDATE_EVENT_RUNNING;
      command_queue->device->read
        (node->command.read.data, 
         node->command.read.host_ptr, 
         node->command.read.device_ptr, 
         node->command.read.cb); 
      POCL_UPDATE_EVENT_COMPLETE;
      POname(clReleaseMemObject) (node->command.read.buffer);
      break;
    case CL_COMMAND_WRITE_BUFFER:
      POCL_UPDATE_EVENT_SUBMITTED;
      POCL_UPDATE_EVENT_RUNNING;
      command_queue->device->write
        (node->command.write.data, 
         node->command.write.host_ptr, 
         node->command.write.device_ptr, 
         node->command.write.cb);
      POCL_UPDATE_EVENT_COMPLETE;
      POname(clReleaseMemObject) (node->command.write.buffer);
      break;
    case CL_COMMAND_COPY_BUFFER:
      POCL_UPDATE_EVENT_SUBMITTED;
      POCL_UPDATE_EVENT_RUNNING;
      command_queue->device->copy
        (node->command.copy.data, 
         node->command.copy.src_ptr, 
         node->command.copy.dst_ptr,
         node->command.copy.cb);
      POCL_UPDATE_EVENT_COMPLETE;
      POname(clReleaseMemObject) (node->command.copy.src_buffer);
      POname(clReleaseMemObject) (node->command.copy.dst_buffer);
      break;
    case CL_COMMAND_MAP_BUFFER: 
      {
        POCL_UPDATE_EVENT_SUBMITTED;
        POCL_UPDATE_EVENT_RUNNING;            
        pocl_map_mem_cmd (command_queue->device, node->command.map.buffer, 
                          node->command.map.mapping);
        POCL_UPDATE_EVENT_COMPLETE;
        break;
      }
    case CL_COMMAND_MAP_IMAGE:
      {
        
        POCL_UPDATE_EVENT_SUBMITTED;
        POCL_UPDATE_EVENT_RUNNING; 
        command_queue->device->read_rect 
          (node->command.map_image.data, node->command.map_image.map_ptr,
           node->command.map_image.device_ptr, node->command.map_image.origin,
           node->command.map_image.origin, node->command.map_image.region, 
           node->command.map_image.rowpitch, 
           node->command.map_image.slicepitch,
           node->command.map_image.rowpitch,
           node->command.map_image.slicepitch);
        POCL_UPDATE_EVENT_COMPLETE;
        break;
      }
    case CL_COMMAND_UNMAP_MEM_OBJECT:
      POCL_UPDATE_EVENT_RUNNING;
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
          if (command_queue->device->unmap_mem != NULL)        
            command_queue->device->unmap_mem
              (command_queue->device->data, 
               (node->command.unmap.mapping)->host_ptr, 
               (node->command.unmap.memobj)->device_ptrs[command_queue->device->dev_id], 
               (node->command.unmap.mapping)->size);
        }
      DL_DELETE((node->command.unmap.memobj)->mappings, 
                node->command.unmap.mapping);
      (node->command.unmap.memobj)->map_count--;
      POCL_UPDATE_EVENT_COMPLETE;
      break;
    case CL_COMMAND_NDRANGE_KERNEL:
      assert (*event == node->event);
      POCL_UPDATE_EVENT_SUBMITTED;
      POCL_UPDATE_EVENT_RUNNING;
      command_queue->device->run(node->command.run.data, node);
      POCL_UPDATE_EVENT_COMPLETE;
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
    case CL_COMMAND_FILL_IMAGE:
      POCL_UPDATE_EVENT_SUBMITTED;
      POCL_UPDATE_EVENT_RUNNING;
      command_queue->device->fill_rect 
        (node->command.fill_image.data, 
         node->command.fill_image.device_ptr,
         node->command.fill_image.buffer_origin,
         node->command.fill_image.region,
         node->command.fill_image.rowpitch, 
         node->command.fill_image.slicepitch,
         node->command.fill_image.fill_pixel,
         node->command.fill_image.pixel_size);
      free(node->command.fill_image.fill_pixel);
      POCL_UPDATE_EVENT_COMPLETE;
      break;
    case CL_COMMAND_MARKER:
      POCL_UPDATE_EVENT_SUBMITTED;
      POCL_UPDATE_EVENT_RUNNING;
      POCL_UPDATE_EVENT_COMPLETE;
      break;
    default:
      POCL_ABORT_UNIMPLEMENTED();
      break;
    }   

   LL_DELETE(command_queue->root, node);
   free (node);
}
  

static void exec_commands_in_queue_until_event(cl_command_queue queue, 
                                               cl_event event)
{
  _cl_command_node *node;
  cl_event wait_event;
  int i;
  
  for (node = queue->root; (node != NULL && (event == NULL || node->event != event)); 
       node = node->next)
    {
      if (node->event != NULL && node->event->event_wait_list != NULL)
        {
          for (i = 0; i < node->event->num_events_in_wait_list; ++i)
            {
              wait_event = node->event->event_wait_list[i];
              if (wait_event->status != CL_COMPLETE || wait_event->status >= 0)
                {
                  exec_commands_in_queue_until_event(wait_event->queue, 
                                                     wait_event);
                }
            }
        }
      exec_command (queue, node);
    }
}

