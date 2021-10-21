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

#include "pocl_util.h"

/* A version for querying the info and in case the device returns 
   a zero, assume the device info query hasn't been implemented 
   for the device driver at hand. Warns about an incomplete 
   implementation. */
#define POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(__TYPE__, __VALUE__)          \
  if (__VALUE__ == (__TYPE__)0)                                               \
    POCL_WARN_INCOMPLETE ();                                                  \
  POCL_RETURN_GETINFO (__TYPE__, __VALUE__);

#define POCL_RETURN_DEVICE_INFO_WITH_IMG_CHECK(__TYPE__, __VALUE__)           \
  if ((device->image_support) && (__VALUE__ == (__TYPE__)0))                  \
    POCL_WARN_INCOMPLETE ();                                                  \
  POCL_RETURN_GETINFO (__TYPE__, __VALUE__);

#define POCL_RETURN_DEVICE_INFO_WITH_EXT_CHECK(__TYPE__, __VALUE__, __EXT__)  \
  if ((strstr(#__EXT__, device->extensions)) && (__VALUE__ == (__TYPE__)0))     \
    POCL_WARN_INCOMPLETE ();                                                  \
  POCL_RETURN_GETINFO (__TYPE__, __VALUE__);

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_ (x)
#define HOST_DEVICE_CL_VERSION_MAJOR_STR                                      \
  STRINGIFY (HOST_DEVICE_CL_VERSION_MAJOR)
#define HOST_DEVICE_CL_VERSION_MINOR_STR                                      \
  STRINGIFY (HOST_DEVICE_CL_VERSION_MINOR)
#define HOST_CL_VERSION                                                       \
  "OpenCL C " HOST_DEVICE_CL_VERSION_MAJOR_STR                                \
  "." HOST_DEVICE_CL_VERSION_MINOR_STR " pocl"

CL_API_ENTRY cl_int CL_API_CALL
POname(clGetDeviceInfo)(cl_device_id   device,
                cl_device_info param_name, 
                size_t         param_value_size, 
                void *         param_value,
                size_t *       param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (device)), CL_INVALID_DEVICE);

  switch (param_name)
  {
  case CL_DEVICE_IMAGE_SUPPORT:
    POCL_RETURN_GETINFO(cl_bool, device->image_support);
  case CL_DEVICE_TYPE:
    POCL_RETURN_GETINFO(cl_device_type, device->type);   
  case CL_DEVICE_VENDOR_ID:
    POCL_RETURN_GETINFO(cl_uint, device->vendor_id);
  case CL_DEVICE_MAX_COMPUTE_UNITS:
    POCL_RETURN_GETINFO(cl_uint, device->max_compute_units);
  case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS          :
    POCL_RETURN_GETINFO(cl_uint, device->max_work_item_dimensions);
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
       loops. A huge unrolling factor explodes the instruction memory size (and
       compilation time) with usually no benefits.
    */
    {
      size_t max_wg_size = device->max_work_group_size;
      POCL_RETURN_GETINFO(size_t, max_wg_size);
    }
  case CL_DEVICE_MAX_WORK_ITEM_SIZES:
    {
      /* We allocate a 3-element array for this in pthread.c */
      typedef struct { size_t size[3]; } size_t_3;
      POCL_RETURN_GETINFO(size_t_3, *(size_t_3 const *)device->max_work_item_sizes);
    }
  case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK (cl_ulong,
                                             device->max_mem_alloc_size);
  case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_uint, device->preferred_vector_width_char);
  case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_uint, device->preferred_vector_width_short);
  case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_uint, device->preferred_vector_width_int);
  case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_uint, device->preferred_vector_width_long);
  case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_uint, device->preferred_vector_width_float);
  case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:
    POCL_RETURN_DEVICE_INFO_WITH_EXT_CHECK(cl_uint, device->preferred_vector_width_double, cl_khr_fp64);
  case CL_DEVICE_MAX_CLOCK_FREQUENCY               :
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_uint, device->max_clock_frequency);
  case CL_DEVICE_ADDRESS_BITS                      :
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_uint, device->address_bits);

  case CL_DEVICE_MAX_READ_IMAGE_ARGS:
    POCL_RETURN_DEVICE_INFO_WITH_IMG_CHECK (cl_uint,
                                            device->max_read_image_args);
  case CL_DEVICE_MAX_WRITE_IMAGE_ARGS              :
    POCL_RETURN_DEVICE_INFO_WITH_IMG_CHECK (cl_uint,
                                            device->max_write_image_args);
  case CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS         :
    POCL_RETURN_DEVICE_INFO_WITH_IMG_CHECK (cl_uint,
                                            device->max_read_write_image_args);
  case CL_DEVICE_IMAGE2D_MAX_WIDTH:
    POCL_RETURN_DEVICE_INFO_WITH_IMG_CHECK (size_t, device->image2d_max_width);
  case CL_DEVICE_IMAGE2D_MAX_HEIGHT                :
    POCL_RETURN_DEVICE_INFO_WITH_IMG_CHECK (size_t,
                                            device->image2d_max_height);
  case CL_DEVICE_IMAGE3D_MAX_WIDTH:
    POCL_RETURN_DEVICE_INFO_WITH_IMG_CHECK (size_t, device->image3d_max_width);
  case CL_DEVICE_IMAGE3D_MAX_HEIGHT:
    POCL_RETURN_DEVICE_INFO_WITH_IMG_CHECK (size_t,
                                            device->image3d_max_height);
  case CL_DEVICE_IMAGE3D_MAX_DEPTH                 :
    POCL_RETURN_DEVICE_INFO_WITH_IMG_CHECK (size_t, device->image3d_max_depth);
  case CL_DEVICE_IMAGE_MAX_BUFFER_SIZE             :
    POCL_RETURN_DEVICE_INFO_WITH_IMG_CHECK (size_t,
                                            device->image_max_buffer_size);
  case CL_DEVICE_IMAGE_MAX_ARRAY_SIZE              :
    POCL_RETURN_DEVICE_INFO_WITH_IMG_CHECK (size_t,
                                            device->image_max_array_size);
  case CL_DEVICE_MAX_SAMPLERS:
    POCL_RETURN_DEVICE_INFO_WITH_IMG_CHECK (cl_uint, device->max_samplers);

  case CL_DEVICE_MAX_PARAMETER_SIZE:
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(size_t, device->max_parameter_size);
  case CL_DEVICE_MEM_BASE_ADDR_ALIGN               :
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK (
        cl_uint, (device->mem_base_addr_align * 8));
  case CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE          : 
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_uint, device->min_data_type_align_size);
  case CL_DEVICE_SINGLE_FP_CONFIG                  :
    POCL_RETURN_GETINFO (cl_ulong, device->single_fp_config);
  case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE             :
    POCL_RETURN_GETINFO(cl_uint, device->global_mem_cache_type);
  case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE         : 
    POCL_RETURN_GETINFO(cl_uint, device->global_mem_cacheline_size);
  case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE             : 
    POCL_RETURN_GETINFO(cl_ulong, device->global_mem_cache_size);
  case CL_DEVICE_GLOBAL_MEM_SIZE:
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_ulong, device->global_mem_size);
  case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE          : 
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_ulong, device->max_constant_buffer_size);
  case CL_DEVICE_MAX_CONSTANT_ARGS                 : 
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_uint, device->max_constant_args);
  case CL_DEVICE_LOCAL_MEM_TYPE                    :
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_uint, device->local_mem_type);
  case CL_DEVICE_LOCAL_MEM_SIZE:
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_ulong, device->local_mem_size);
  case CL_DEVICE_ERROR_CORRECTION_SUPPORT          :
    POCL_RETURN_GETINFO(cl_bool, device->error_correction_support);
  case CL_DEVICE_PROFILING_TIMER_RESOLUTION        :
    POCL_RETURN_GETINFO(size_t, device->profiling_timer_resolution);
  case CL_DEVICE_ENDIAN_LITTLE                     :
    POCL_RETURN_GETINFO(cl_uint, device->endian_little);
  case CL_DEVICE_AVAILABLE                         :
    POCL_RETURN_GETINFO(cl_bool, device->available);
  case CL_DEVICE_COMPILER_AVAILABLE                :
    POCL_RETURN_GETINFO(cl_bool, device->compiler_available);
  case CL_DEVICE_LINKER_AVAILABLE                  :
    POCL_RETURN_GETINFO (cl_bool, device->linker_available);
  case CL_DEVICE_EXECUTION_CAPABILITIES            :
    POCL_RETURN_GETINFO(cl_device_exec_capabilities, device->execution_capabilities);

  case CL_DEVICE_NAME:
    POCL_RETURN_GETINFO_STR(device->long_name);
   
  case CL_DEVICE_VENDOR                            : 
    POCL_RETURN_GETINFO_STR(device->vendor);

  case CL_DRIVER_VERSION:
    POCL_RETURN_GETINFO_STR(device->driver_version);
  case CL_DEVICE_PROFILE                           : 
    POCL_RETURN_GETINFO_STR(device->profile);
  case CL_DEVICE_VERSION                           : 
    {
      char res[1000];
      char *hash = device->ops->build_hash(device);
      snprintf(res, 1000, "%s HSTR: %s", device->version, hash);
      free(hash);
      POCL_RETURN_GETINFO_STR(res);
    }
  case CL_DEVICE_EXTENSIONS                        : 
    POCL_RETURN_GETINFO_STR(device->extensions);
  case CL_DEVICE_PLATFORM                          :
    {
      /* Return the first platform id, assuming this is the only
         platform id (which is currently always the case for pocl) */
      cl_platform_id platform_id;
      POname(clGetPlatformIDs)(1, &platform_id, NULL);
      POCL_RETURN_GETINFO(cl_platform_id, platform_id);
    }
  case CL_DEVICE_DOUBLE_FP_CONFIG                  :
    POCL_RETURN_GETINFO (cl_ulong, device->double_fp_config);
  case CL_DEVICE_HALF_FP_CONFIG                    :
    POCL_RETURN_GETINFO (cl_ulong, device->half_fp_config);
  case CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF       :
    POCL_RETURN_DEVICE_INFO_WITH_EXT_CHECK(cl_uint, device->preferred_vector_width_half, cl_khr_fp16);
  case CL_DEVICE_HOST_UNIFIED_MEMORY               : 
    POCL_RETURN_GETINFO(cl_bool, device->host_unified_memory);
  case CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR          : 
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_uint, device->native_vector_width_char);
  case CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT         :
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_uint, device->native_vector_width_short);
  case CL_DEVICE_NATIVE_VECTOR_WIDTH_INT           : 
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_uint, device->native_vector_width_int);
  case CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG          : 
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_uint, device->native_vector_width_long);
  case CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT         : 
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_uint, device->native_vector_width_float);
  case CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE        : 
    POCL_RETURN_DEVICE_INFO_WITH_EXT_CHECK(cl_uint, device->native_vector_width_double, cl_khr_fp64);
  case CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF          : 
    POCL_RETURN_DEVICE_INFO_WITH_EXT_CHECK(cl_uint, device->native_vector_width_half, cl_khr_fp16);
  case CL_DEVICE_OPENCL_C_VERSION                  :
    POCL_RETURN_GETINFO_STR (HOST_CL_VERSION);
  case CL_DEVICE_BUILT_IN_KERNELS                  :
    if (device->builtin_kernel_list)
      POCL_RETURN_GETINFO_STR (device->builtin_kernel_list);
    else
      POCL_RETURN_GETINFO_STR ("");

  case CL_DEVICE_PARENT_DEVICE                     :
    POCL_RETURN_GETINFO(cl_device_id, device->parent_device);

  case CL_DEVICE_PARTITION_MAX_SUB_DEVICES         :
    POCL_RETURN_GETINFO(cl_uint, device->max_sub_devices);

  case CL_DEVICE_PARTITION_PROPERTIES              :
    if (device->num_partition_properties)
      POCL_RETURN_GETINFO_ARRAY (cl_device_partition_property,
                                 device->num_partition_properties,
                                 device->partition_properties);
    else
      POCL_RETURN_GETINFO (cl_device_partition_property, 0);

  case CL_DEVICE_PARTITION_TYPE                    :
    if (device->num_partition_types)
      POCL_RETURN_GETINFO_ARRAY (cl_device_partition_property,
                                 device->num_partition_types,
                                 device->partition_type);
    else
      POCL_RETURN_GETINFO (cl_device_partition_property, 0);

  case CL_DEVICE_PARTITION_AFFINITY_DOMAIN         :
    POCL_RETURN_GETINFO(cl_device_affinity_domain, 0);

  case CL_DEVICE_PREFERRED_INTEROP_USER_SYNC       :
    POCL_RETURN_GETINFO(cl_bool, CL_TRUE);

  case CL_DEVICE_PRINTF_BUFFER_SIZE                :
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(size_t, device->printf_buffer_size);

  case CL_DEVICE_REFERENCE_COUNT:
    POCL_RETURN_DEVICE_INFO_WITH_IMPL_CHECK(cl_uint, 
                                            (cl_uint)device->pocl_refcount)

  case CL_DEVICE_SVM_CAPABILITIES:
    POCL_RETURN_GETINFO(cl_device_svm_capabilities, device->svm_caps);
  case CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES:
    POCL_RETURN_GETINFO(cl_device_atomic_capabilities, device->atomic_memory_capabilities);
  case CL_DEVICE_ATOMIC_FENCE_CAPABILITIES:
    POCL_RETURN_GETINFO(cl_device_atomic_capabilities, device->atomic_fence_capabilities);
  case CL_DEVICE_MAX_ON_DEVICE_EVENTS:
    POCL_RETURN_GETINFO(cl_uint, device->max_events);
  case CL_DEVICE_MAX_ON_DEVICE_QUEUES:
    POCL_RETURN_GETINFO(cl_uint, device->max_queues);
  case CL_DEVICE_PIPE_SUPPORT:
    POCL_RETURN_GETINFO (cl_bool, device->pipe_support);
  case CL_DEVICE_MAX_PIPE_ARGS:
    POCL_RETURN_GETINFO(cl_uint, device->max_pipe_args);
  case CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS:
    POCL_RETURN_GETINFO(cl_uint, device->max_pipe_active_res);
  case CL_DEVICE_PIPE_MAX_PACKET_SIZE:
    POCL_RETURN_GETINFO(cl_uint, device->max_pipe_packet_size);
  case CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE:
    POCL_RETURN_GETINFO(cl_uint, device->dev_queue_pref_size);
  case CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE:
    POCL_RETURN_GETINFO(cl_uint, device->dev_queue_max_size);
  case CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT:
    POCL_RETURN_GETINFO(cl_uint, 0);
  case CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT:
    POCL_RETURN_GETINFO(cl_uint, 0);
  case CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT:
    POCL_RETURN_GETINFO(cl_uint, 0);
  case CL_DEVICE_SPIR_VERSIONS:
    if (strstr (device->extensions, "cl_khr_spir"))
      POCL_RETURN_GETINFO_STR ("1.2");
    else
      POCL_RETURN_GETINFO_STR ("");
  case CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES:
    POCL_RETURN_GETINFO(cl_command_queue_properties, device->on_dev_queue_props);
  case CL_DEVICE_QUEUE_ON_HOST_PROPERTIES:
    POCL_RETURN_GETINFO(cl_command_queue_properties, device->on_host_queue_props);

  case CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE:
    POCL_RETURN_GETINFO(size_t, device->global_var_pref_size);
  case CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE:
    POCL_RETURN_GETINFO(size_t, device->global_var_max_size);
  case CL_DEVICE_IL_VERSION:
    if (device->spirv_version)
      POCL_RETURN_GETINFO_STR (device->spirv_version);
    else
      POCL_RETURN_GETINFO_STR ("");
  }

  if(device->ops->get_device_info_ext != NULL) {
    return device->ops->get_device_info_ext(device, param_name, param_value_size,
                                            param_value, param_value_size_ret);
  }

  return CL_INVALID_VALUE;
}
POsym(clGetDeviceInfo)
