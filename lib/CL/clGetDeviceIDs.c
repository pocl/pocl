/* OpenCL runtime library: clGetDeviceIDs()
 *
 * Copyright - Kalle Raiskila 2011.
 *
 * This is file is licencsed under a "Free Beer" type license:
 * You can do whatever you want with this stuff. If we meet some day,
 * and you think this stuff is worth it, you can buy me a beer in return.
 */

#include "pocl_cl.h"
#include "devices/devices.h"

/* Note: this is a kludge. This will require a thorough re-write when pocl
 * supports multiple devices
 */
CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDs(cl_platform_id   platform, 
               cl_device_type   device_type, 
               cl_uint          num_entries, 
               cl_device_id *   devices, 
               cl_uint *        num_devices) CL_API_SUFFIX__VERSION_1_0
{
  int num = 0;

  // TODO: OpenCL API specification allows implementation dependant behaviour
  // if platform == NULL. Should we just allow for it?	
  if (platform == NULL || ( platform->magic != 42 ))
    return CL_INVALID_PLATFORM;
	
  // Currently - POCL supports only the host device - i.e. a CPU
  if (device_type & CL_DEVICE_TYPE_CPU ||
      device_type & CL_DEVICE_TYPE_DEFAULT ||
      device_type & CL_DEVICE_TYPE_ALL)
    num = 1;
  else if (device_type == CL_DEVICE_TYPE_GPU ||
           device_type == CL_DEVICE_TYPE_ACCELERATOR )
    num = 0;
  else
    return CL_INVALID_DEVICE_TYPE;

	// no room for any response
  if (devices == NULL && num_devices == NULL)
    return CL_INVALID_VALUE;

  // user forgot to allocate space for response
  if (num_entries > 0 && devices == NULL )
    return CL_INVALID_VALUE;

	
  if (num_devices != NULL)
    *num_devices = num;
	
  if (num_entries > 0 && devices!= NULL)
	{
      if (num)
        devices[0] = &pocl_devices[0];
	}

  if (num > 0)
    return CL_SUCCESS;
  else
    return CL_DEVICE_NOT_FOUND;
}
