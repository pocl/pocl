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
#include "utlist.h"

CL_API_ENTRY cl_int CL_API_CALL
clFinish(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
  _cl_command_node *node;

  assert((command_queue->properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) == 0
         && "we don't support out-of-order queues yet");
  
  /* Step through the list of commands in-order */  
  LL_FOREACH(command_queue->root,node)
  {
    switch( node->type )
    {
      case CL_COMMAND_TYPE_READ:
        command_queue->device->read(
          node->command.read.data, 
          node->command.read.host_ptr, 
          node->command.read.device_ptr, 
          node->command.read.cb); 
        break;
      case CL_COMMAND_TYPE_WRITE:
        command_queue->device->write(
          node->command.write.data, 
          node->command.write.host_ptr, 
          node->command.write.device_ptr, 
          node->command.write.cb);
        break;
      case CL_COMMAND_TYPE_RUN:
        command_queue->device->run(node->command.run.data,
			     node->command.run.file,
			     node->command.run.kernel,
			     &node->command.run.pc);
	free(node->command.run.file);
        break;
  
      default:
        POCL_ABORT_UNIMPLEMENTED();
        break;
    }
  } 
  
  // clear the queue 
  node = command_queue->root;
  command_queue->root = NULL;
  while( node )
    {
      _cl_command_node *tmp;
      tmp = node->next;
      free(node);
      node=tmp;
    }  

  return CL_SUCCESS;
}
