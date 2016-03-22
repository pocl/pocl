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

#include "pocl_util.h"
#include "pocl_debug.h"
#include "pocl_cl.h"


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
   uint i;

   POCL_GOTO_ERROR_COND((in_device == NULL), CL_INVALID_DEVICE);
   POCL_GOTO_ERROR_COND((properties == NULL), CL_INVALID_VALUE);
   POCL_GOTO_ERROR_COND((num_devices && !out_devices), CL_INVALID_VALUE);
   POCL_GOTO_ERROR_COND((!num_devices && out_devices), CL_INVALID_VALUE);

   /* check that the partition property is supported by the device */
   errcode = CL_INVALID_VALUE;
   for (i = 0; i < in_device->num_partition_properties; ++i) {
     if (properties[0] == in_device->partition_properties[i]) {
       errcode = CL_SUCCESS;
       break;
     }
   }
   if (errcode != CL_SUCCESS)
     goto ERROR;

   /* Ok, it's a supported partition property, count the number of devices; currently,
    * we only support EQUALLY and BY_COUNTS, which enumerate the number of devices
    * differently */
   if (properties[0] == CL_DEVICE_PARTITION_EQUALLY) {
     // error out if the number of CUs per device is 0 or bigger than the number of
     // CUs of in_device
     POCL_GOTO_ERROR_COND(
       (properties[1] == 0 || properties[1] > in_device->max_compute_units),
       CL_INVALID_VALUE);
     // error out if properties isn't zero-terminated
     POCL_GOTO_ERROR_COND(properties[2] != 0, CL_INVALID_VALUE);

     count_devices = in_device->max_compute_units / properties[1];
     num_props = 3; // partition type, CUs per device, terminating 0
   } else if (properties[0] == CL_DEVICE_PARTITION_BY_COUNTS) {
     cl_uint total_cus = 0;
     i = 1;
     while (properties[i] != 0) {
       count_devices++;
       total_cus += properties[i];
       ++i;
     }
     // error out if the total number of CUs surpasses the number of device CUs,
     // or if we have to many subdevices
     POCL_GOTO_ERROR_COND(
       (total_cus == 0 || total_cus > in_device->max_compute_units ||
        count_devices > in_device->max_sub_devices),
       CL_INVALID_DEVICE_PARTITION_COUNT);
     num_props = count_devices + 2; // partition type, one spec per device, terminating 0
   } else {
     // we end here if some of our devices claim to support a different
     // partition type, but this function was not updated accordingly

     char what[1024];
     snprintf(what, 1024, "Device-reported partition type 0x%x", (unsigned int)properties[0]);
     POCL_ABORT_UNIMPLEMENTED(what);
   }

   // num_devices must match count_devices if non-zero
   POCL_GOTO_ERROR_COND((num_devices && count_devices != num_devices), CL_INVALID_VALUE);

   if (out_devices) {
     // we allocate our own array of devices to simplify management
     new_devs = calloc(count_devices, sizeof(cl_device_id));
     POCL_GOTO_ERROR_COND((!new_devs), CL_OUT_OF_HOST_MEMORY);

     for (i = 0; i < count_devices; ++i) {
       new_devs[i] = calloc(1, sizeof(struct _cl_device_id));
       POCL_GOTO_ERROR_COND((new_devs[i] == NULL), CL_OUT_OF_HOST_MEMORY);
       POCL_INIT_OBJECT(new_devs[i]);

       // clone in_device
       memcpy(new_devs[i], in_device, sizeof(struct _cl_device_id));

       // override the fields: partition type, parent, max_compute_units,
       // max_sub_devices
       new_devs[i]->partition_type = calloc(num_props, sizeof(*properties));
       POCL_GOTO_ERROR_COND((new_devs[i]->partition_type == NULL),
         CL_OUT_OF_HOST_MEMORY);
       memcpy(new_devs[i]->partition_type, properties, num_props*sizeof(*properties));
       new_devs[i]->num_partition_types = num_props;

       new_devs[i]->parent_device = in_device;
       new_devs[i]->max_sub_devices = new_devs[i]->max_compute_units =
         (properties[0] == CL_DEVICE_PARTITION_EQUALLY ? properties[1] :
          properties[i+1]);
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
      POCL_RELEASE_OBJECT(new_devs[i], new_refcount);
      if (new_refcount == 0)
        POCL_MEM_FREE(new_devs[i]);
    }

    free(new_devs);
  }
  return errcode;

}
POsym(clCreateSubDevices)
