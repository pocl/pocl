/* OpenCL runtime library: clGetKernelWorkGroupInfo()
 * 
 */


#include "devices/devices.h"
#include "pocl_util.h"


extern CL_API_ENTRY cl_int CL_API_CALL
POname(clGetKernelWorkGroupInfo) 
(cl_kernel kernel,
 cl_device_id device,
 cl_kernel_work_group_info param_name,
 size_t param_value_size,
 void *param_value, 
 size_t * param_value_size_ret)
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

  switch (param_name)
    {
    case CL_KERNEL_WORK_GROUP_SIZE: 
      return POname(clGetDeviceInfo) 
        (device, CL_DEVICE_MAX_WORK_GROUP_SIZE, param_value_size,
         param_value, param_value_size_ret);
        
    case CL_KERNEL_COMPILE_WORK_GROUP_SIZE:
    {
        typedef struct { size_t size[3]; } size_t_3;
#if 0
        printf("### reqd wg sizes %d %d %d\n", 
               kernel->reqd_wg_size[0], 
               kernel->reqd_wg_size[1], 
               kernel->reqd_wg_size[2]);
#endif
        POCL_RETURN_GETINFO(size_t_3, *(size_t_3*)kernel->reqd_wg_size);
    }
      
    case CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:
      POCL_RETURN_GETINFO(size_t, device->preferred_wg_size_multiple);
      
    case CL_KERNEL_LOCAL_MEM_SIZE:
    {
      size_t local_size = 0, i;

      /* Count the host-allocated locals. */
      for (i = 0; i < kernel->num_args; ++i)
        {
          if (!kernel->arg_info[i].is_local) continue;
          local_size += kernel->dyn_arguments[i].size;
        }
      /* Count the automatic locals. */
      for (i = 0; i < kernel->num_locals; ++i)
        {
          local_size += kernel->dyn_arguments[kernel->num_args + i].size;
        }
#if 0
      printf("### local memory usage %d\n", local_size);
#endif
      POCL_RETURN_GETINFO(cl_ulong, local_size);
    }
      
    case CL_KERNEL_PRIVATE_MEM_SIZE:
      POCL_ABORT_UNIMPLEMENTED();

    default:
      return CL_INVALID_VALUE;
    }
  return CL_SUCCESS;
}
POsym(clGetKernelWorkGroupInfo)
