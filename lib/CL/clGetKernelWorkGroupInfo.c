/* OpenCL runtime library: clGetKernelWorkGroupInfo()
 * 
 */


#include "devices/devices.h"
#include "pocl_cl.h"

extern CL_API_ENTRY cl_int CL_API_CALL
clGetKernelWorkGroupInfo (cl_kernel kernel,
			  cl_device_id device,
			  cl_kernel_work_group_info param_name,
			  size_t param_value_size,
			  void *param_value, size_t * param_value_size_ret)
  CL_API_SUFFIX__VERSION_1_0
{

  /* check that kernel is associated with device, or that there is no risk of confusion */
  if (device != NULL)
    {
      int i;
      int found_it = 0;
      for (i = 0; i < kernel->context->num_devices; i++)
	if (device == kernel->context->devices[i])
	  {
	    found_it = 1;
	    break;
	  }
      if (!found_it)
	return CL_INVALID_DEVICE;

    }
  else if (kernel->context->num_devices > 1)
    return CL_INVALID_DEVICE;
  else
    device = kernel->context->devices[0];

  /* the wording of the specs is ambiguous - if param_value is NULL, do we need to check that param_name is valid? */
  if (param_value == NULL)
    switch (param_name)
      {
      case CL_KERNEL_WORK_GROUP_SIZE:
	if (param_value_size >= sizeof (size_t))
	  *(size_t *) param_value = device->max_work_group_size;
	else
	  return CL_INVALID_VALUE;
	if (param_value_size_ret)
	  *param_value_size_ret = sizeof (size_t);

	break;

      case CL_KERNEL_COMPILE_WORK_GROUP_SIZE:
	/* TODO: is this device->max_work_item_sizes? */
	return CL_INVALID_VALUE;	/* as there is no CL_UNIMPLEMENTED */
	break;

      case CL_KERNEL_LOCAL_MEM_SIZE:
	if (param_value_size >= sizeof (cl_ulong))
	  *(size_t *) param_value = device->local_mem_size;
	else
	  return CL_INVALID_VALUE;
	if (param_value_size_ret)
	  *param_value_size_ret = sizeof (cl_ulong);

	break;

      default:
	return CL_INVALID_VALUE;
      }

  return CL_SUCCESS;

}
