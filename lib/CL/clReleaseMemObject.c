/* OpenCL runtime library: clReleaseMemObject()

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
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

#include "utlist.h"
#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL
clReleaseMemObject(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0
{
  cl_device_id device_id;
  unsigned i;
  mem_mapping_t *mapping, *temp;

  POCL_RELEASE_OBJECT(memobj);

  /* OpenCL 1.2 Page 118:

     After the memobj reference count becomes zero and commands queued for execution on 
     a command-queue(s) that use memobj have finished, the memory object is deleted. If 
     memobj is a buffer object, memobj cannot be deleted until all sub-buffer objects associated 
     with memobj are deleted.
  */

  if (memobj->pocl_refcount == 0) 
    {
      if (memobj->parent == NULL) 
        {
          for (i = 0; i < memobj->context->num_devices; ++i)
            {
              device_id = memobj->context->devices[i];
              device_id->free(device_id->data, memobj->flags, memobj->device_ptrs[device_id->dev_id]);
              memobj->device_ptrs[device_id->dev_id] = NULL;
            }
        } else 
        {
          /* a sub buffer object does not free the memory from
             the device */
          POCL_RELEASE_OBJECT(memobj->parent);
        }
      POCL_RELEASE_OBJECT(memobj->context);
      DL_FOREACH_SAFE(memobj->mappings, mapping, temp)
        {
          free (mapping);
        }
      memobj->mappings = NULL;
      
      free(memobj->device_ptrs);
      free(memobj);
    }
  return CL_SUCCESS;
}
