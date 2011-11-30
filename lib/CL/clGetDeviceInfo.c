/* OpenCL runtime library: clGetDeviceInfo()
 *
 * Copyright - Kalle Raiskila 2011.
 *
 * This is file is licencsed under a "Free Beer" type license:
 * You can do whatever you want with this stuff. If we meet some day,
 * and you think this stuff is worth it, you can buy me a beer in return.
 */

#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceInfo(cl_device_id     device,
                cl_device_info   param_name, 
                size_t           param_value_size, 
                void *           param_value,
                size_t *         param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0
{
	// TODO: dig up the info
	return CL_INVALID_VALUE;
}

