/* OpenCL runtime library: clCreateSubDevices()

   Copyright (c) 2014 Lassi Koskinen
   
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

#include "pocl_util.h"
#include "pocl_cl.h"


/* Creates an array of sub-devices that each reference a non-intersecting 
   set of compute units within in_device, according to a partition scheme 
   given by properties. */
    

  
CL_API_ENTRY cl_int CL_API_CALL
POname(clCreateSubDevices)(cl_device_id in_device,
                           const cl_device_partition_property *properties,
                           cl_uint num_devices,
                           cl_device_id *out_devices,
                           cl_uint *num_devices_ret) CL_API_SUFFIX__VERSION_1_2
{
   int errcode;
   cl_device_id sub1 = in_device;
   cl_device_id sub2 = in_device;

   POCL_GOTO_ERROR_COND((in_device == NULL), CL_INVALID_DEVICE);

   POCL_INIT_OBJECT(sub1);
   sub1->parent_device = in_device;

   POCL_INIT_OBJECT(sub2);
   sub2->parent_device = in_device;
   
   out_devices[0] = sub1;
   out_devices[1] = sub2;

   POCL_RETAIN_OBJECT(in_device);
   POCL_RETAIN_OBJECT(in_device);
   
   return CL_SUCCESS;
    
ERROR: 
   if (&sub1 == &sub2) 
     {
        POCL_MEM_FREE(sub1);
     } 
   else 
     {
        POCL_MEM_FREE(sub1);
        POCL_MEM_FREE(sub2);
     }
   return errcode;

}
POsym(clCreateSubDevices)
