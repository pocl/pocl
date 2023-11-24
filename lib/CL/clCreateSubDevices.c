/* OpenCL runtime library: clCreateSubDevices()

   Copyright (c) 2014 Lassi Koskinen
                 2015 Giuseppe Bilotta

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
#include "pocl_debug.h"
#include "pocl_util.h"
#include <string.h>

/* Creates an array of sub-devices that each reference a non-intersecting
   set of compute units within in_device, according to a partition scheme
   given by properties.
   TODO we should actually pin the subdevices to specific cores,
   and possibly correct the cache information accordingly.
   */

CL_API_ENTRY cl_int CL_API_CALL
POname(clCreateSubDevices)(cl_device_id in_device,
                           const cl_device_partition_property *properties,
                           cl_uint num_devices,
                           cl_device_id *out_devices,
                           cl_uint *num_devices_ret) CL_API_SUFFIX__VERSION_1_2
{
   cl_int errcode = CL_SUCCESS;
   cl_uint count_devices = 0;
   cl_device_id *new_devs = NULL;
   // number of elements in (copies of) properties, including terminating null
   cl_uint num_props = 0;
   cl_uint i;

   //unsigned yo = offsetof(struct _cl_device_id, ops);

   POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (in_device)), CL_INVALID_DEVICE);
   POCL_RETURN_ERROR_COND ((*(in_device->available) == CL_FALSE),
                           CL_DEVICE_NOT_AVAILABLE);
   POCL_GOTO_ERROR_COND((properties == NULL), CL_INVALID_VALUE);
   POCL_GOTO_ERROR_COND((num_devices && !out_devices), CL_INVALID_VALUE);
   POCL_GOTO_ERROR_COND((!num_devices && out_devices), CL_INVALID_VALUE);

   POCL_GOTO_ERROR_ON (
       (in_device->max_sub_devices == 0), CL_DEVICE_PARTITION_FAILED,
       "Device %s cannot be further partitioned\n", in_device->short_name);

   /* check that the partition property is supported by the device */
   POCL_GOTO_ERROR_ON ((in_device->num_partition_properties == 0),
                       CL_INVALID_VALUE,
                       "Device %s does not support any partition property\n",
                       in_device->short_name);

   for (i = 0; i < in_device->num_partition_properties; ++i) {
     if (properties[0] == in_device->partition_properties[i]) {
       break;
     }
   }

   POCL_GOTO_ERROR_ON (
       (i == in_device->num_partition_properties), CL_INVALID_VALUE,
       "Device %s does not support the requested partition property\n",
       in_device->short_name);

   /* Ok, it's a supported partition property, count the number of devices; currently,
    * we only support EQUALLY and BY_COUNTS, which enumerate the number of devices
    * differently */
   if (properties[0] == CL_DEVICE_PARTITION_EQUALLY)
     {
       /* error out if the number of CUs per device is 0 or bigger than the
        * number of CUs of in_device */
       POCL_GOTO_ERROR_COND (
           (properties[1] == 0
            || (cl_uint)properties[1] > in_device->max_compute_units),
           CL_INVALID_VALUE);
       // error out if properties isn't zero-terminated
       POCL_GOTO_ERROR_COND (properties[2] != 0, CL_INVALID_VALUE);

       count_devices = in_device->max_compute_units / properties[1];
       num_props = 3; // partition type, CUs per device, terminating 0
     }
   else if (properties[0] == CL_DEVICE_PARTITION_BY_COUNTS)
     {
       cl_uint total_cus = 0;
       i = 1;
       while (properties[i] != 0)
         {
           count_devices++;
           total_cus += properties[i];
           ++i;
         }
       /* error out if the total number of CUs surpasses the number of device
        * CUs, or if we have to many subdevices */
       POCL_GOTO_ERROR_COND ((total_cus == 0),
                             CL_INVALID_DEVICE_PARTITION_COUNT);
       POCL_GOTO_ERROR_COND ((total_cus > in_device->max_compute_units),
                             CL_INVALID_DEVICE_PARTITION_COUNT);
       POCL_GOTO_ERROR_COND ((count_devices > in_device->max_sub_devices),
                             CL_INVALID_DEVICE_PARTITION_COUNT);
       num_props = count_devices
                   + 2; /* partition type, one spec per device, terminating 0 */
     }
   else
     {
       /* we end here if some of our devices claim to support a different
        * partition type, but this function was not updated accordingly */
       POCL_GOTO_ERROR_ON (1, CL_INVALID_VALUE,
                           "Device reported partition type 0x%x "
                           "is not supported by Pocl\n",
                           (unsigned int)properties[0]);
     }

   // num_devices must be greater than or equal to count_devices if non-zero
   POCL_GOTO_ERROR_COND((num_devices && num_devices < count_devices), CL_INVALID_VALUE);

   if (out_devices) {
     // we allocate our own array of devices to simplify management
     new_devs = (cl_device_id *)calloc (count_devices, sizeof (cl_device_id));
     POCL_GOTO_ERROR_COND((!new_devs), CL_OUT_OF_HOST_MEMORY);
     unsigned sum = 0;

     for (i = 0; i < count_devices; ++i) {
       new_devs[i] = (cl_device_id)calloc(1, sizeof(struct _cl_device_id));
       POCL_GOTO_ERROR_COND((new_devs[i] == NULL), CL_OUT_OF_HOST_MEMORY);

       // clone in_device
       memcpy(new_devs[i], in_device, sizeof(struct _cl_device_id));
       /* this must be done AFTER the clone, otherwise we end up with
        * lock states and refcounts copied from parent device */
       POCL_INIT_OBJECT (new_devs[i], in_device);

       new_devs[i]->parent_device = in_device;
       if (in_device->builtin_kernel_list)
         {
           new_devs[i]->builtin_kernel_list
               = strdup (in_device->builtin_kernel_list);
           new_devs[i]->builtin_kernels_with_version = malloc (
               in_device->num_builtin_kernels * sizeof (cl_name_version));
           memcpy (new_devs[i]->builtin_kernels_with_version,
                   in_device->builtin_kernels_with_version,
                   in_device->num_builtin_kernels * sizeof (cl_name_version));
         }

       new_devs[i]->max_sub_devices = new_devs[i]->max_compute_units
           = (properties[0] == CL_DEVICE_PARTITION_EQUALLY
                  ? properties[1]
                  : properties[i + 1]);

       /* for devices with 1 CU, report zero subdevices and
        * no partitioning support. */
       if (new_devs[i]->max_compute_units == 1)
         {
           new_devs[i]->max_sub_devices = 0;
           new_devs[i]->num_partition_properties = 0;
           new_devs[i]->partition_properties = NULL;
         }

       /* copy the partition type argument, for clGetDeviceInfo() */
       new_devs[i]->partition_type = (cl_device_partition_property *)calloc(num_props, sizeof(*properties));
       POCL_GOTO_ERROR_COND((new_devs[i]->partition_type == NULL),
         CL_OUT_OF_HOST_MEMORY);
       memcpy(new_devs[i]->partition_type, properties, num_props*sizeof(*properties));
       new_devs[i]->num_partition_types = num_props;

       new_devs[i]->core_count = new_devs[i]->max_compute_units;
       if (in_device->parent_device)
         new_devs[i]->core_start = in_device->core_start + sum;
       else
         new_devs[i]->core_start = sum;
       sum += new_devs[i]->core_count;
     }

     memcpy(out_devices, new_devs, count_devices*sizeof(cl_device_id));
     POCL_MEM_FREE(new_devs);
   }

   if (num_devices_ret)
     *num_devices_ret = count_devices;

   return errcode;

ERROR:
  if (new_devs) {
    // release all objects
    for (i = 0; i < count_devices; ++i) {
      int new_refcount = 0;
      // the first NULL new_devs indicates we're done
      // with the ones we actually managed to allocate
      if (new_devs[i] == NULL)
        break;
      POCL_RELEASE_OBJECT (new_devs[i], new_refcount);
      if (new_refcount == 0)
        {
          POCL_MEM_FREE (new_devs[i]);
          POCL_MEM_FREE (new_devs[i]->partition_type);
        }
    }

    POCL_MEM_FREE (new_devs);
  }
  return errcode;

}
POsym(clCreateSubDevices)
