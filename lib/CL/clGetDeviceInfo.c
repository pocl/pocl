/* OpenCL runtime library: clGetDeviceInfo()

   Copyright (c) 2011-2012 Kalle Raiskila and Pekka Jääskeläinen
   
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
#include <string.h>

#define POCL_RETURN_DEVICE_INFO(__TYPE__, __VALUE__)                \
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

#define POCL_RETURN_DEVICE_INFO_STR(__STR__)                        \
  {                                                                 \
    size_t const value_size = strlen(__STR__) + 1;                  \
    if (param_value)                                                \
      {                                                             \
        if (param_value_size < value_size) return CL_INVALID_VALUE; \
        memcpy(param_value, __STR__, value_size);                   \
      }                                                             \
    if (param_value_size_ret)                                       \
      *param_value_size_ret = value_size;                           \
    return CL_SUCCESS;                                              \
  }                                                                 \
    

  
CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceInfo(cl_device_id   device,
                cl_device_info param_name, 
                size_t         param_value_size, 
                void *         param_value,
                size_t *       param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  switch (param_name)
  {
  case CL_DEVICE_IMAGE_SUPPORT: 
    /* Return CL_FALSE until the APIs are implemented. */
    POCL_RETURN_DEVICE_INFO(cl_bool, CL_FALSE);
  case CL_DEVICE_TYPE:
    POCL_RETURN_DEVICE_INFO(cl_device_type, device->type);   
  case CL_DEVICE_VENDOR_ID                         : break;
  case CL_DEVICE_MAX_COMPUTE_UNITS:
    POCL_RETURN_DEVICE_INFO(cl_uint, device->max_compute_units);
  case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS          : break;
  case CL_DEVICE_MAX_WORK_GROUP_SIZE               : 
    /* There is no "preferred WG size" device query, so we probably should
       return something more sensible than the CL_INT_MAX that seems
       to be the default in the pthread device. It should be computed from 
       the machine's vector width or issue width.

       Some OpenCL programs (e.g. the Dijkstra book sample) seem to scale 
       the work groups using this. 

       There's a kernel query CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
       that can yield better heuristics for the good WG size and forms
       a basis for a higher level performance portability layer.
       
       Basically the size is now limited by the absence of work item
       loops. A huge unrolling factor explodes the instruction memory size with
       no benefits.

       16 should be large enough for anything (tm) for now ;) Let's 
       increase it when at least vectorization works and there's a better
       machine heuristics.
    */
    POCL_RETURN_DEVICE_INFO(cl_uint, 16); // device->max_work_group_size
  case CL_DEVICE_MAX_WORK_ITEM_SIZES               : break;
    
  case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:
    POCL_RETURN_DEVICE_INFO(cl_uint, device->preferred_vector_width_char);   
    
  case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:
    POCL_RETURN_DEVICE_INFO(cl_uint, device->preferred_vector_width_short);
    
  case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:
    POCL_RETURN_DEVICE_INFO(cl_uint, device->preferred_vector_width_int);

  case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:
    POCL_RETURN_DEVICE_INFO(cl_uint, device->preferred_vector_width_long);
    
  case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:
    POCL_RETURN_DEVICE_INFO(cl_uint, device->preferred_vector_width_float);
    
  case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:
    POCL_RETURN_DEVICE_INFO(cl_uint, device->preferred_vector_width_double);

  case CL_DEVICE_MAX_CLOCK_FREQUENCY               : break;
  case CL_DEVICE_ADDRESS_BITS                      : break;
  case CL_DEVICE_MAX_READ_IMAGE_ARGS               : break;
  case CL_DEVICE_MAX_WRITE_IMAGE_ARGS              : break;
  case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
    POCL_RETURN_DEVICE_INFO(cl_ulong, device->max_mem_alloc_size);
  case CL_DEVICE_IMAGE2D_MAX_WIDTH                 : break;
  case CL_DEVICE_IMAGE2D_MAX_HEIGHT                : break;
  case CL_DEVICE_IMAGE3D_MAX_WIDTH                 : break;
  case CL_DEVICE_IMAGE3D_MAX_HEIGHT                : break;
  case CL_DEVICE_IMAGE3D_MAX_DEPTH                 : break;
  case CL_DEVICE_MAX_PARAMETER_SIZE                : break;
  case CL_DEVICE_MAX_SAMPLERS                      : break;
  case CL_DEVICE_MEM_BASE_ADDR_ALIGN               : break;
  case CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE          : break;
  case CL_DEVICE_SINGLE_FP_CONFIG                  : break;
  case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE             : break;
  case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE         : break;
  case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE             : break;
  case CL_DEVICE_GLOBAL_MEM_SIZE:
    POCL_RETURN_DEVICE_INFO(cl_uint, device->global_mem_size);
  case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE          : break;
  case CL_DEVICE_MAX_CONSTANT_ARGS                 : break;
  case CL_DEVICE_LOCAL_MEM_TYPE                    : break;
  case CL_DEVICE_LOCAL_MEM_SIZE:
    POCL_RETURN_DEVICE_INFO(cl_ulong, device->local_mem_size);
  case CL_DEVICE_ERROR_CORRECTION_SUPPORT          : break;
  case CL_DEVICE_PROFILING_TIMER_RESOLUTION        : break;
  case CL_DEVICE_ENDIAN_LITTLE                     : break;
  case CL_DEVICE_AVAILABLE                         : break;
  case CL_DEVICE_COMPILER_AVAILABLE                : break;
  case CL_DEVICE_EXECUTION_CAPABILITIES            : break;
  case CL_DEVICE_QUEUE_PROPERTIES                  : break;
    
  case CL_DEVICE_NAME:
    POCL_RETURN_DEVICE_INFO_STR(device->name);
   
  case CL_DEVICE_VENDOR                            : break;
  case CL_DRIVER_VERSION                           : break;
  case CL_DEVICE_PROFILE                           : break;
  case CL_DEVICE_VERSION                           : break;
  case CL_DEVICE_EXTENSIONS                        : break;
  case CL_DEVICE_PLATFORM                          : break;
  case CL_DEVICE_DOUBLE_FP_CONFIG                  : break;
  case CL_DEVICE_HALF_FP_CONFIG                    : break;
  case CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF       : break;
  case CL_DEVICE_HOST_UNIFIED_MEMORY               : break;
  case CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR          : break;
  case CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT         : break;
  case CL_DEVICE_NATIVE_VECTOR_WIDTH_INT           : break;
  case CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG          : break;
  case CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT         : break;
  case CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE        : break;
  case CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF          : break;
  case CL_DEVICE_OPENCL_C_VERSION                  : break;
  }

  // remove me when everything *is* implemented, and param_name really is invalid
  POCL_WARN_INCOMPLETE();
  return CL_INVALID_VALUE;
}
