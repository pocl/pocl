/* OpenCL runtime library: clGetKernelWorkGroupInfo()
 * 
 */


#include "devices/devices.h"
#include "pocl_cl.h"
#include "pocl_util.h"

#define POCL_RETURN_KERNEL_WG_INFO(__TYPE__, __VALUE__)                \
  {                                                                 \
    size_t const value_size = sizeof(__TYPE__);                     \
    if (param_value)                                                \
      {                                                             \
        if (param_value_size < value_size) return CL_INVALID_VALUE; \
        *(__TYPE__*)param_value = __VALUE__;                        \
      }                                                             \
    if (param_value_size_ret)                                       \
      *param_value_size_ret = value_size;                           \
    return CL_SUCCESS;                                              \
  } 


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

  /* the wording of the specs is ambiguous - if param_value is NULL, do we need to check that param_name is valid? */
  if (param_value == NULL)
    return CL_INVALID_VALUE;

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
      POCL_RETURN_KERNEL_WG_INFO(size_t, device->preferred_wg_size_multiple);
      
    case CL_KERNEL_LOCAL_MEM_SIZE:
    {
      size_t local_size = 0, i;

      /* Count the host-allocated locals. */
      for (i = 0; i < kernel->num_args; ++i)
        {
          if (!kernel->arg_is_local[i]) continue;
          local_size += kernel->arguments[i].size;
        }
      /* Count the automatic locals. */
      for (i = 0; i < kernel->num_locals; ++i)
        {
          local_size += kernel->arguments[kernel->num_args + i].size;
        }
#if 0
      printf("### local memory usage %d\n", local_size);
#endif
      POCL_RETURN_KERNEL_WG_INFO(cl_ulong, local_size);
    }
      
    default:  
      POCL_ABORT_UNIMPLEMENTED();
    }
  return CL_SUCCESS;
}
POsym(clGetKernelWorkGroupInfo)
