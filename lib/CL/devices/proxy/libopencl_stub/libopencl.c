/* libopencl.c - Stub libopencl that pocl_dynlib_symbol_address into actual
   library based on environment variable

   LIBOPENCL_SO_PATH      -- Path to the opencl .so that will be searched first
   LIBOPENCL_SO_PATH_2    -- Searched second
   LIBOPENCL_SO_PATH_3    -- Searched third
   LIBOPENCL_SO_PATH_4    -- Searched fourth

   If none of these are set, default system paths will be considered

   Copyright (c) 2023 PoCL Developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.


*/

#define stubname(name) stub##name

#include "libopencl.h"
#include <stdlib.h>
#include <sys/stat.h>

#include "pocl_dynlib.h"

#if defined(__APPLE__) || defined(__MACOSX)
static const char *default_so_paths[]
    = { "libOpenCL.so", "/System/Library/Frameworks/OpenCL.framework/OpenCL" };
#elif defined(__ANDROID__)
static const char *default_so_paths[]
    = { "/system/lib64/libOpenCL.so",
        "/system/vendor/lib64/libOpenCL.so",
        "/system/vendor/lib64/egl/libGLES_mali.so",
        "/system/vendor/lib64/libPVROCL.so",
        "/data/data/org.pocl.libs/files/lib64/libpocl.so",
        "/system/lib/libOpenCL.so",
        "/system/vendor/lib/libOpenCL.so",
        "/system/vendor/lib/egl/libGLES_mali.so",
        "/system/vendor/lib/libPVROCL.so",
        "/data/data/org.pocl.libs/files/lib/libpocl.so",
        "libOpenCL.so" };
#elif defined(_WIN32)
static const char *default_so_paths[] = { "OpenCL.dll" };
#elif defined(__linux__)
static const char *default_so_paths[]
    = { "/usr/lib/libOpenCL.so",     "/usr/local/lib/libOpenCL.so",
        "/usr/local/lib/libpocl.so", "/usr/lib64/libOpenCL.so",
        "/usr/lib32/libOpenCL.so",   "libOpenCL.so" };
#endif

static void *so_handle = NULL;

static int
access_file (const char *filename)
{
  struct stat buffer;
  return (stat (filename, &buffer) == 0);
}

static int
open_libopencl_so ()
{
  char *path = NULL, *str = NULL;
  int i;

  if ((str = getenv ("LIBOPENCL_SO_PATH")) && access_file (str))
    {
      path = str;
    }
  else if ((str = getenv ("LIBOPENCL_SO_PATH_2")) && access_file (str))
    {
      path = str;
    }
  else if ((str = getenv ("LIBOPENCL_SO_PATH_3")) && access_file (str))
    {
      path = str;
    }
  else if ((str = getenv ("LIBOPENCL_SO_PATH_4")) && access_file (str))
    {
      path = str;
    }

  if (!path)
    {
      for (i = 0; i < (sizeof (default_so_paths) / sizeof (char *)); i++)
        {
          if (access_file (default_so_paths[i]))
            {
              path = (char *)default_so_paths[i];
              break;
            }
        }
    }

  if (path)
    {
      so_handle = pocl_dynlib_open (path, RTLD_LAZY);
      return 0;
    }
  else
    {
      return -1;
    }
}

void
stubOpenclReset ()
{
  if (so_handle)
    pocl_dynlib_close (so_handle);

  so_handle = NULL;
}

cl_int
stubname (clGetPlatformIDs) (cl_uint num_entries, cl_platform_id *platforms,
                             cl_uint *num_platforms)
{
  f_clGetPlatformIDs func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetPlatformIDs)pocl_dynlib_symbol_address (so_handle,
                                                         "clGetPlatformIDs");
  if (func)
    {
      return func (num_entries, platforms, num_platforms);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetPlatformInfo) (cl_platform_id platform,
                              cl_platform_info param_name,
                              size_t param_value_size, void *param_value,
                              size_t *param_value_size_ret)
{
  f_clGetPlatformInfo func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetPlatformInfo)pocl_dynlib_symbol_address (so_handle,
                                                          "clGetPlatformInfo");
  if (func)
    {
      return func (platform, param_name, param_value_size, param_value,
                   param_value_size_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetDeviceIDs) (cl_platform_id platform, cl_device_type device_type,
                           cl_uint num_entries, cl_device_id *devices,
                           cl_uint *num_devices)
{
  f_clGetDeviceIDs func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetDeviceIDs)pocl_dynlib_symbol_address (so_handle,
                                                       "clGetDeviceIDs");
  if (func)
    {
      return func (platform, device_type, num_entries, devices, num_devices);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetDeviceInfo) (cl_device_id device, cl_device_info param_name,
                            size_t param_value_size, void *param_value,
                            size_t *param_value_size_ret)
{
  f_clGetDeviceInfo func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetDeviceInfo)pocl_dynlib_symbol_address (so_handle,
                                                        "clGetDeviceInfo");
  if (func)
    {
      return func (device, param_name, param_value_size, param_value,
                   param_value_size_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clCreateSubDevices) (cl_device_id in_device,
                               const cl_device_partition_property *properties,
                               cl_uint num_devices, cl_device_id *out_devices,
                               cl_uint *num_devices_ret)
{
  f_clCreateSubDevices func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateSubDevices)pocl_dynlib_symbol_address (
    so_handle, "clCreateSubDevices");
  if (func)
    {
      return func (in_device, properties, num_devices, out_devices,
                   num_devices_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clRetainDevice) (cl_device_id device)
{
  f_clRetainDevice func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clRetainDevice)pocl_dynlib_symbol_address (so_handle,
                                                       "clRetainDevice");
  if (func)
    {
      return func (device);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clReleaseDevice) (cl_device_id device)
{
  f_clReleaseDevice func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clReleaseDevice)pocl_dynlib_symbol_address (so_handle,
                                                        "clReleaseDevice");
  if (func)
    {
      return func (device);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_context
stubname (clCreateContext) (const cl_context_properties *properties,
                            cl_uint num_devices, const cl_device_id *devices,
                            void (*pfn_notify) (const char *, const void *,
                                                size_t, void *),
                            void *user_data, cl_int *errcode_ret)
{
  f_clCreateContext func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateContext)pocl_dynlib_symbol_address (so_handle,
                                                        "clCreateContext");
  if (func)
    {
      return func (properties, num_devices, devices, pfn_notify, user_data,
                   errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_context
stubname (clCreateContextFromType) (
    const cl_context_properties *properties, cl_device_type device_type,
    void (*pfn_notify) (const char *, const void *, size_t, void *),
    void *user_data, cl_int *errcode_ret)
{
  f_clCreateContextFromType func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateContextFromType)pocl_dynlib_symbol_address (
    so_handle, "clCreateContextFromType");
  if (func)
    {
      return func (properties, device_type, pfn_notify, user_data,
                   errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_int
stubname (clRetainContext) (cl_context context)
{
  f_clRetainContext func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clRetainContext)pocl_dynlib_symbol_address (so_handle,
                                                        "clRetainContext");
  if (func)
    {
      return func (context);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clReleaseContext) (cl_context context)
{
  f_clReleaseContext func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clReleaseContext)pocl_dynlib_symbol_address (so_handle,
                                                         "clReleaseContext");
  if (func)
    {
      return func (context);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetContextInfo) (cl_context context, cl_context_info param_name,
                             size_t param_value_size, void *param_value,
                             size_t *param_value_size_ret)
{
  f_clGetContextInfo func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetContextInfo)pocl_dynlib_symbol_address (so_handle,
                                                         "clGetContextInfo");
  if (func)
    {
      return func (context, param_name, param_value_size, param_value,
                   param_value_size_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_command_queue
stubname (clCreateCommandQueue) (cl_context context, cl_device_id device,
                                 cl_command_queue_properties properties,
                                 cl_int *errcode_ret)
{
  f_clCreateCommandQueue func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateCommandQueue)pocl_dynlib_symbol_address (
    so_handle, "clCreateCommandQueue");
  if (func)
    {
      return func (context, device, properties, errcode_ret);
    }
  else
    {
      return NULL;
    }
}

#ifdef CL_VERSION_2_0
cl_command_queue
clCreateCommandQueueWithProperties (cl_context context, cl_device_id device,
                                    const cl_queue_properties *properties,
                                    cl_int *errcode_ret)
{
  f_clCreateCommandQueueWithProperties func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateCommandQueueWithProperties)pocl_dynlib_symbol_address (
    so_handle, "clCreateCommandQueueWithProperties");
  if (func)
    {
      return func (context, device, properties, errcode_ret);
    }
  else
    {
      return NULL;
    }
}
#endif

cl_int
stubname (clRetainCommandQueue) (cl_command_queue command_queue)
{
  f_clRetainCommandQueue func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clRetainCommandQueue)pocl_dynlib_symbol_address (
    so_handle, "clRetainCommandQueue");
  if (func)
    {
      return func (command_queue);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clReleaseCommandQueue) (cl_command_queue command_queue)
{
  f_clReleaseCommandQueue func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clReleaseCommandQueue)pocl_dynlib_symbol_address (
    so_handle, "clReleaseCommandQueue");
  if (func)
    {
      return func (command_queue);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetCommandQueueInfo) (cl_command_queue command_queue,
                                  cl_command_queue_info param_name,
                                  size_t param_value_size, void *param_value,
                                  size_t *param_value_size_ret)
{
  f_clGetCommandQueueInfo func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetCommandQueueInfo)pocl_dynlib_symbol_address (
    so_handle, "clGetCommandQueueInfo");
  if (func)
    {
      return func (command_queue, param_name, param_value_size, param_value,
                   param_value_size_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_mem
stubname (clCreateBuffer) (cl_context context, cl_mem_flags flags, size_t size,
                           void *host_ptr, cl_int *errcode_ret)
{
  f_clCreateBuffer func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateBuffer)pocl_dynlib_symbol_address (so_handle,
                                                       "clCreateBuffer");
  if (func)
    {
      return func (context, flags, size, host_ptr, errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_mem
stubname (clCreateSubBuffer) (cl_mem buffer, cl_mem_flags flags,
                              cl_buffer_create_type buffer_create_type,
                              const void *buffer_create_info,
                              cl_int *errcode_ret)
{
  f_clCreateSubBuffer func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateSubBuffer)pocl_dynlib_symbol_address (so_handle,
                                                          "clCreateSubBuffer");
  if (func)
    {
      return func (buffer, flags, buffer_create_type, buffer_create_info,
                   errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_mem
stubname (clCreateImage) (cl_context context, cl_mem_flags flags,
                          const cl_image_format *image_format,
                          const cl_image_desc *image_desc, void *host_ptr,
                          cl_int *errcode_ret)
{
  f_clCreateImage func;

  if (!so_handle)
    open_libopencl_so ();

  func
    = (f_clCreateImage)pocl_dynlib_symbol_address (so_handle, "clCreateImage");
  if (func)
    {
      return func (context, flags, image_format, image_desc, host_ptr,
                   errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_int
stubname (clRetainMemObject) (cl_mem memobj)
{
  f_clRetainMemObject func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clRetainMemObject)pocl_dynlib_symbol_address (so_handle,
                                                          "clRetainMemObject");
  if (func)
    {
      return func (memobj);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clReleaseMemObject) (cl_mem memobj)
{
  f_clReleaseMemObject func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clReleaseMemObject)pocl_dynlib_symbol_address (
    so_handle, "clReleaseMemObject");
  if (func)
    {
      return func (memobj);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetSupportedImageFormats) (cl_context context, cl_mem_flags flags,
                                       cl_mem_object_type image_type,
                                       cl_uint num_entries,
                                       cl_image_format *image_formats,
                                       cl_uint *num_image_formats)
{
  f_clGetSupportedImageFormats func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetSupportedImageFormats)pocl_dynlib_symbol_address (
    so_handle, "clGetSupportedImageFormats");
  if (func)
    {
      return func (context, flags, image_type, num_entries, image_formats,
                   num_image_formats);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetMemObjectInfo) (cl_mem memobj, cl_mem_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret)
{
  f_clGetMemObjectInfo func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetMemObjectInfo)pocl_dynlib_symbol_address (
    so_handle, "clGetMemObjectInfo");
  if (func)
    {
      return func (memobj, param_name, param_value_size, param_value,
                   param_value_size_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetImageInfo) (cl_mem image, cl_image_info param_name,
                           size_t param_value_size, void *param_value,
                           size_t *param_value_size_ret)
{
  f_clGetImageInfo func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetImageInfo)pocl_dynlib_symbol_address (so_handle,
                                                       "clGetImageInfo");
  if (func)
    {
      return func (image, param_name, param_value_size, param_value,
                   param_value_size_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clSetMemObjectDestructorCallback) (
    cl_mem memobj, void (*pfn_notify) (cl_mem memobj, void *user_data),
    void *user_data)
{
  f_clSetMemObjectDestructorCallback func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clSetMemObjectDestructorCallback)pocl_dynlib_symbol_address (
    so_handle, "clSetMemObjectDestructorCallback");
  if (func)
    {
      return func (memobj, pfn_notify, user_data);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_sampler
stubname (clCreateSampler) (cl_context context, cl_bool normalized_coords,
                            cl_addressing_mode addressing_mode,
                            cl_filter_mode filter_mode, cl_int *errcode_ret)
{
  f_clCreateSampler func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateSampler)pocl_dynlib_symbol_address (so_handle,
                                                        "clCreateSampler");
  if (func)
    {
      return func (context, normalized_coords, addressing_mode, filter_mode,
                   errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_int
stubname (clRetainSampler) (cl_sampler sampler)
{
  f_clRetainSampler func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clRetainSampler)pocl_dynlib_symbol_address (so_handle,
                                                        "clRetainSampler");
  if (func)
    {
      return func (sampler);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clReleaseSampler) (cl_sampler sampler)
{
  f_clReleaseSampler func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clReleaseSampler)pocl_dynlib_symbol_address (so_handle,
                                                         "clReleaseSampler");
  if (func)
    {
      return func (sampler);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetSamplerInfo) (cl_sampler sampler, cl_sampler_info param_name,
                             size_t param_value_size, void *param_value,
                             size_t *param_value_size_ret)
{
  f_clGetSamplerInfo func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetSamplerInfo)pocl_dynlib_symbol_address (so_handle,
                                                         "clGetSamplerInfo");
  if (func)
    {
      return func (sampler, param_name, param_value_size, param_value,
                   param_value_size_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_program
stubname (clCreateProgramWithSource) (cl_context context, cl_uint count,
                                      const char **strings,
                                      const size_t *lengths,
                                      cl_int *errcode_ret)
{
  f_clCreateProgramWithSource func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateProgramWithSource)pocl_dynlib_symbol_address (
    so_handle, "clCreateProgramWithSource");
  if (func)
    {
      return func (context, count, strings, lengths, errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_program
stubname (clCreateProgramWithBinary) (cl_context context, cl_uint num_devices,
                                      const cl_device_id *device_list,
                                      const size_t *lengths,
                                      const unsigned char **binaries,
                                      cl_int *binary_status,
                                      cl_int *errcode_ret)
{
  f_clCreateProgramWithBinary func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateProgramWithBinary)pocl_dynlib_symbol_address (
    so_handle, "clCreateProgramWithBinary");
  if (func)
    {
      return func (context, num_devices, device_list, lengths, binaries,
                   binary_status, errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_program
stubname (clCreateProgramWithBuiltInKernels) (cl_context context,
                                              cl_uint num_devices,
                                              const cl_device_id *device_list,
                                              const char *kernel_names,
                                              cl_int *errcode_ret)
{
  f_clCreateProgramWithBuiltInKernels func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateProgramWithBuiltInKernels)pocl_dynlib_symbol_address (
    so_handle, "clCreateProgramWithBuiltInKernels");
  if (func)
    {
      return func (context, num_devices, device_list, kernel_names,
                   errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_int
stubname (clRetainProgram) (cl_program program)
{
  f_clRetainProgram func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clRetainProgram)pocl_dynlib_symbol_address (so_handle,
                                                        "clRetainProgram");
  if (func)
    {
      return func (program);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clReleaseProgram) (cl_program program)
{
  f_clReleaseProgram func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clReleaseProgram)pocl_dynlib_symbol_address (so_handle,
                                                         "clReleaseProgram");
  if (func)
    {
      return func (program);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clBuildProgram) (
    cl_program program, cl_uint num_devices, const cl_device_id *device_list,
    const char *options,
    void (*pfn_notify) (cl_program program, void *user_data), void *user_data)
{
  f_clBuildProgram func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clBuildProgram)pocl_dynlib_symbol_address (so_handle,
                                                       "clBuildProgram");
  if (func)
    {
      return func (program, num_devices, device_list, options, pfn_notify,
                   user_data);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clCompileProgram) (
    cl_program program, cl_uint num_devices, const cl_device_id *device_list,
    const char *options, cl_uint num_input_headers,
    const cl_program *input_headers, const char **header_include_names,
    void (*pfn_notify) (cl_program program, void *user_data), void *user_data)
{
  f_clCompileProgram func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCompileProgram)pocl_dynlib_symbol_address (so_handle,
                                                         "clCompileProgram");
  if (func)
    {
      return func (program, num_devices, device_list, options,
                   num_input_headers, input_headers, header_include_names,
                   pfn_notify, user_data);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_program
stubname (clLinkProgram) (cl_context context, cl_uint num_devices,
                          const cl_device_id *device_list, const char *options,
                          cl_uint num_input_programs,
                          const cl_program *input_programs,
                          void (*pfn_notify) (cl_program program,
                                              void *user_data),
                          void *user_data, cl_int *errcode_ret)
{
  f_clLinkProgram func;

  if (!so_handle)
    open_libopencl_so ();

  func
    = (f_clLinkProgram)pocl_dynlib_symbol_address (so_handle, "clLinkProgram");
  if (func)
    {
      return func (context, num_devices, device_list, options,
                   num_input_programs, input_programs, pfn_notify, user_data,
                   errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_int
stubname (clUnloadPlatformCompiler) (cl_platform_id platform)
{
  f_clUnloadPlatformCompiler func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clUnloadPlatformCompiler)pocl_dynlib_symbol_address (
    so_handle, "clUnloadPlatformCompiler");
  if (func)
    {
      return func (platform);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetProgramInfo) (cl_program program, cl_program_info param_name,
                             size_t param_value_size, void *param_value,
                             size_t *param_value_size_ret)
{
  f_clGetProgramInfo func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetProgramInfo)pocl_dynlib_symbol_address (so_handle,
                                                         "clGetProgramInfo");
  if (func)
    {
      return func (program, param_name, param_value_size, param_value,
                   param_value_size_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetProgramBuildInfo) (cl_program program, cl_device_id device,
                                  cl_program_build_info param_name,
                                  size_t param_value_size, void *param_value,
                                  size_t *param_value_size_ret)
{
  f_clGetProgramBuildInfo func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetProgramBuildInfo)pocl_dynlib_symbol_address (
    so_handle, "clGetProgramBuildInfo");
  if (func)
    {
      return func (program, device, param_name, param_value_size, param_value,
                   param_value_size_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_kernel
stubname (clCreateKernel) (cl_program program, const char *kernel_name,
                           cl_int *errcode_ret)
{
  f_clCreateKernel func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateKernel)pocl_dynlib_symbol_address (so_handle,
                                                       "clCreateKernel");
  if (func)
    {
      return func (program, kernel_name, errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_int
stubname (clCreateKernelsInProgram) (cl_program program, cl_uint num_kernels,
                                     cl_kernel *kernels,
                                     cl_uint *num_kernels_ret)
{
  f_clCreateKernelsInProgram func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateKernelsInProgram)pocl_dynlib_symbol_address (
    so_handle, "clCreateKernelsInProgram");
  if (func)
    {
      return func (program, num_kernels, kernels, num_kernels_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clRetainKernel) (cl_kernel kernel)
{
  f_clRetainKernel func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clRetainKernel)pocl_dynlib_symbol_address (so_handle,
                                                       "clRetainKernel");
  if (func)
    {
      return func (kernel);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clReleaseKernel) (cl_kernel kernel)
{
  f_clReleaseKernel func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clReleaseKernel)pocl_dynlib_symbol_address (so_handle,
                                                        "clReleaseKernel");
  if (func)
    {
      return func (kernel);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clSetKernelArg) (cl_kernel kernel, cl_uint arg_index,
                           size_t arg_size, const void *arg_value)
{
  f_clSetKernelArg func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clSetKernelArg)pocl_dynlib_symbol_address (so_handle,
                                                       "clSetKernelArg");
  if (func)
    {
      return func (kernel, arg_index, arg_size, arg_value);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetKernelInfo) (cl_kernel kernel, cl_kernel_info param_name,
                            size_t param_value_size, void *param_value,
                            size_t *param_value_size_ret)
{
  f_clGetKernelInfo func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetKernelInfo)pocl_dynlib_symbol_address (so_handle,
                                                        "clGetKernelInfo");
  if (func)
    {
      return func (kernel, param_name, param_value_size, param_value,
                   param_value_size_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetKernelArgInfo) (cl_kernel kernel, cl_uint arg_indx,
                               cl_kernel_arg_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret)
{
  f_clGetKernelArgInfo func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetKernelArgInfo)pocl_dynlib_symbol_address (
    so_handle, "clGetKernelArgInfo");
  if (func)
    {
      return func (kernel, arg_indx, param_name, param_value_size, param_value,
                   param_value_size_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetKernelWorkGroupInfo) (cl_kernel kernel, cl_device_id device,
                                     cl_kernel_work_group_info param_name,
                                     size_t param_value_size,
                                     void *param_value,
                                     size_t *param_value_size_ret)
{
  f_clGetKernelWorkGroupInfo func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetKernelWorkGroupInfo)pocl_dynlib_symbol_address (
    so_handle, "clGetKernelWorkGroupInfo");
  if (func)
    {
      return func (kernel, device, param_name, param_value_size, param_value,
                   param_value_size_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clWaitForEvents) (cl_uint num_events, const cl_event *event_list)
{
  f_clWaitForEvents func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clWaitForEvents)pocl_dynlib_symbol_address (so_handle,
                                                        "clWaitForEvents");
  if (func)
    {
      return func (num_events, event_list);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetEventInfo) (cl_event event, cl_event_info param_name,
                           size_t param_value_size, void *param_value,
                           size_t *param_value_size_ret)
{
  f_clGetEventInfo func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetEventInfo)pocl_dynlib_symbol_address (so_handle,
                                                       "clGetEventInfo");
  if (func)
    {
      return func (event, param_name, param_value_size, param_value,
                   param_value_size_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_event
stubname (clCreateUserEvent) (cl_context context, cl_int *errcode_ret)
{
  f_clCreateUserEvent func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateUserEvent)pocl_dynlib_symbol_address (so_handle,
                                                          "clCreateUserEvent");
  if (func)
    {
      return func (context, errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_int
stubname (clRetainEvent) (cl_event event)
{
  f_clRetainEvent func;

  if (!so_handle)
    open_libopencl_so ();

  func
    = (f_clRetainEvent)pocl_dynlib_symbol_address (so_handle, "clRetainEvent");
  if (func)
    {
      return func (event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clReleaseEvent) (cl_event event)
{
  f_clReleaseEvent func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clReleaseEvent)pocl_dynlib_symbol_address (so_handle,
                                                       "clReleaseEvent");
  if (func)
    {
      return func (event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clSetUserEventStatus) (cl_event event, cl_int execution_status)
{
  f_clSetUserEventStatus func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clSetUserEventStatus)pocl_dynlib_symbol_address (
    so_handle, "clSetUserEventStatus");
  if (func)
    {
      return func (event, execution_status);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clSetEventCallback) (cl_event event,
                               cl_int command_exec_callback_type,
                               void (*pfn_notify) (cl_event, cl_int, void *),
                               void *user_data)
{
  f_clSetEventCallback func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clSetEventCallback)pocl_dynlib_symbol_address (
    so_handle, "clSetEventCallback");
  if (func)
    {
      return func (event, command_exec_callback_type, pfn_notify, user_data);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetEventProfilingInfo) (cl_event event,
                                    cl_profiling_info param_name,
                                    size_t param_value_size, void *param_value,
                                    size_t *param_value_size_ret)
{
  f_clGetEventProfilingInfo func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetEventProfilingInfo)pocl_dynlib_symbol_address (
    so_handle, "clGetEventProfilingInfo");
  if (func)
    {
      return func (event, param_name, param_value_size, param_value,
                   param_value_size_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clFlush) (cl_command_queue command_queue)
{
  f_clFlush func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clFlush)pocl_dynlib_symbol_address (so_handle, "clFlush");
  if (func)
    {
      return func (command_queue);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clFinish) (cl_command_queue command_queue)
{
  f_clFinish func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clFinish)pocl_dynlib_symbol_address (so_handle, "clFinish");
  if (func)
    {
      return func (command_queue);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueReadBuffer) (cl_command_queue command_queue, cl_mem buffer,
                                cl_bool blocking_read, size_t offset,
                                size_t size, void *ptr,
                                cl_uint num_events_in_wait_list,
                                const cl_event *event_wait_list,
                                cl_event *event)
{
  f_clEnqueueReadBuffer func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueReadBuffer)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueReadBuffer");
  if (func)
    {
      return func (command_queue, buffer, blocking_read, offset, size, ptr,
                   num_events_in_wait_list, event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueReadBufferRect) (
    cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read,
    const size_t *buffer_offset, const size_t *host_offset,
    const size_t *region, size_t buffer_row_pitch, size_t buffer_slice_pitch,
    size_t host_row_pitch, size_t host_slice_pitch, void *ptr,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event)
{
  f_clEnqueueReadBufferRect func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueReadBufferRect)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueReadBufferRect");
  if (func)
    {
      return func (command_queue, buffer, blocking_read, buffer_offset,
                   host_offset, region, buffer_row_pitch, buffer_slice_pitch,
                   host_row_pitch, host_slice_pitch, ptr,
                   num_events_in_wait_list, event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueWriteBuffer) (cl_command_queue command_queue, cl_mem buffer,
                                 cl_bool blocking_write, size_t offset,
                                 size_t size, const void *ptr,
                                 cl_uint num_events_in_wait_list,
                                 const cl_event *event_wait_list,
                                 cl_event *event)
{
  f_clEnqueueWriteBuffer func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueWriteBuffer)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueWriteBuffer");
  if (func)
    {
      return func (command_queue, buffer, blocking_write, offset, size, ptr,
                   num_events_in_wait_list, event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueWriteBufferRect) (
    cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write,
    const size_t *buffer_offset, const size_t *host_offset,
    const size_t *region, size_t buffer_row_pitch, size_t buffer_slice_pitch,
    size_t host_row_pitch, size_t host_slice_pitch, const void *ptr,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event)
{
  f_clEnqueueWriteBufferRect func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueWriteBufferRect)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueWriteBufferRect");
  if (func)
    {
      return func (command_queue, buffer, blocking_write, buffer_offset,
                   host_offset, region, buffer_row_pitch, buffer_slice_pitch,
                   host_row_pitch, host_slice_pitch, ptr,
                   num_events_in_wait_list, event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueFillBuffer) (cl_command_queue command_queue, cl_mem buffer,
                                const void *pattern, size_t pattern_size,
                                size_t offset, size_t size,
                                cl_uint num_events_in_wait_list,
                                const cl_event *event_wait_list,
                                cl_event *event)
{
  f_clEnqueueFillBuffer func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueFillBuffer)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueFillBuffer");
  if (func)
    {
      return func (command_queue, buffer, pattern, pattern_size, offset, size,
                   num_events_in_wait_list, event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueCopyBuffer) (cl_command_queue command_queue,
                                cl_mem src_buffer, cl_mem dst_buffer,
                                size_t src_offset, size_t dst_offset,
                                size_t size, cl_uint num_events_in_wait_list,
                                const cl_event *event_wait_list,
                                cl_event *event)
{
  f_clEnqueueCopyBuffer func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueCopyBuffer)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueCopyBuffer");
  if (func)
    {
      return func (command_queue, src_buffer, dst_buffer, src_offset,
                   dst_offset, size, num_events_in_wait_list, event_wait_list,
                   event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueCopyBufferRect) (
    cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer,
    const size_t *src_origin, const size_t *dst_origin, const size_t *region,
    size_t src_row_pitch, size_t src_slice_pitch, size_t dst_row_pitch,
    size_t dst_slice_pitch, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event)
{
  f_clEnqueueCopyBufferRect func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueCopyBufferRect)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueCopyBufferRect");
  if (func)
    {
      return func (command_queue, src_buffer, dst_buffer, src_origin,
                   dst_origin, region, src_row_pitch, src_slice_pitch,
                   dst_row_pitch, dst_slice_pitch, num_events_in_wait_list,
                   event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueReadImage) (cl_command_queue command_queue, cl_mem image,
                               cl_bool blocking_read, const size_t *origin,
                               const size_t *region, size_t row_pitch,
                               size_t slice_pitch, void *ptr,
                               cl_uint num_events_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event)
{
  f_clEnqueueReadImage func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueReadImage)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueReadImage");
  if (func)
    {
      return func (command_queue, image, blocking_read, origin, region,
                   row_pitch, slice_pitch, ptr, num_events_in_wait_list,
                   event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueWriteImage) (cl_command_queue command_queue, cl_mem image,
                                cl_bool blocking_write, const size_t *origin,
                                const size_t *region, size_t input_row_pitch,
                                size_t input_slice_pitch, const void *ptr,
                                cl_uint num_events_in_wait_list,
                                const cl_event *event_wait_list,
                                cl_event *event)
{
  f_clEnqueueWriteImage func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueWriteImage)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueWriteImage");
  if (func)
    {
      return func (command_queue, image, blocking_write, origin, region,
                   input_row_pitch, input_slice_pitch, ptr,
                   num_events_in_wait_list, event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueFillImage) (cl_command_queue command_queue, cl_mem image,
                               const void *fill_color, const size_t *origin,
                               const size_t *region,
                               cl_uint num_events_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event)
{
  f_clEnqueueFillImage func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueFillImage)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueFillImage");
  if (func)
    {
      return func (command_queue, image, fill_color, origin, region,
                   num_events_in_wait_list, event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueCopyImage) (cl_command_queue command_queue,
                               cl_mem src_image, cl_mem dst_image,
                               const size_t *src_origin,
                               const size_t *dst_origin, const size_t *region,
                               cl_uint num_events_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event)
{
  f_clEnqueueCopyImage func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueCopyImage)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueCopyImage");
  if (func)
    {
      return func (command_queue, src_image, dst_image, src_origin, dst_origin,
                   region, num_events_in_wait_list, event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueCopyImageToBuffer) (cl_command_queue command_queue,
                                       cl_mem src_image, cl_mem dst_buffer,
                                       const size_t *src_origin,
                                       const size_t *region, size_t dst_offset,
                                       cl_uint num_events_in_wait_list,
                                       const cl_event *event_wait_list,
                                       cl_event *event)
{
  f_clEnqueueCopyImageToBuffer func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueCopyImageToBuffer)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueCopyImageToBuffer");
  if (func)
    {
      return func (command_queue, src_image, dst_buffer, src_origin, region,
                   dst_offset, num_events_in_wait_list, event_wait_list,
                   event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueCopyBufferToImage) (
    cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_image,
    size_t src_offset, const size_t *dst_origin, const size_t *region,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event)
{
  f_clEnqueueCopyBufferToImage func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueCopyBufferToImage)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueCopyBufferToImage");
  if (func)
    {
      return func (command_queue, src_buffer, dst_image, src_offset,
                   dst_origin, region, num_events_in_wait_list,
                   event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

void *
stubname (clEnqueueMapBuffer) (cl_command_queue command_queue, cl_mem buffer,
                               cl_bool blocking_map, cl_map_flags map_flags,
                               size_t offset, size_t size,
                               cl_uint num_events_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event, cl_int *errcode_ret)
{
  f_clEnqueueMapBuffer func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueMapBuffer)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueMapBuffer");
  if (func)
    {
      return func (command_queue, buffer, blocking_map, map_flags, offset,
                   size, num_events_in_wait_list, event_wait_list, event,
                   errcode_ret);
    }
  else
    {
      return NULL;
    }
}

void *
stubname (clEnqueueMapImage) (cl_command_queue command_queue, cl_mem image,
                              cl_bool blocking_map, cl_map_flags map_flags,
                              const size_t *origin, const size_t *region,
                              size_t *image_row_pitch,
                              size_t *image_slice_pitch,
                              cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list, cl_event *event,
                              cl_int *errcode_ret)
{
  f_clEnqueueMapImage func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueMapImage)pocl_dynlib_symbol_address (so_handle,
                                                          "clEnqueueMapImage");
  if (func)
    {
      return func (command_queue, image, blocking_map, map_flags, origin,
                   region, image_row_pitch, image_slice_pitch,
                   num_events_in_wait_list, event_wait_list, event,
                   errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_int
stubname (clEnqueueUnmapMemObject) (cl_command_queue command_queue,
                                    cl_mem memobj, void *mapped_ptr,
                                    cl_uint num_events_in_wait_list,
                                    const cl_event *event_wait_list,
                                    cl_event *event)
{
  f_clEnqueueUnmapMemObject func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueUnmapMemObject)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueUnmapMemObject");
  if (func)
    {
      return func (command_queue, memobj, mapped_ptr, num_events_in_wait_list,
                   event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueMigrateMemObjects) (cl_command_queue command_queue,
                                       cl_uint num_mem_objects,
                                       const cl_mem *mem_objects,
                                       cl_mem_migration_flags flags,
                                       cl_uint num_events_in_wait_list,
                                       const cl_event *event_wait_list,
                                       cl_event *event)
{
  f_clEnqueueMigrateMemObjects func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueMigrateMemObjects)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueMigrateMemObjects");
  if (func)
    {
      return func (command_queue, num_mem_objects, mem_objects, flags,
                   num_events_in_wait_list, event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueNDRangeKernel) (
    cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event)
{
  f_clEnqueueNDRangeKernel func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueNDRangeKernel)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueNDRangeKernel");
  if (func)
    {
      return func (command_queue, kernel, work_dim, global_work_offset,
                   global_work_size, local_work_size, num_events_in_wait_list,
                   event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueTask) (cl_command_queue command_queue, cl_kernel kernel,
                          cl_uint num_events_in_wait_list,
                          const cl_event *event_wait_list, cl_event *event)
{
  f_clEnqueueTask func;

  if (!so_handle)
    open_libopencl_so ();

  func
    = (f_clEnqueueTask)pocl_dynlib_symbol_address (so_handle, "clEnqueueTask");
  if (func)
    {
      return func (command_queue, kernel, num_events_in_wait_list,
                   event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueNativeKernel) (
    cl_command_queue command_queue, void (*user_func) (void *), void *args,
    size_t cb_args, cl_uint num_mem_objects, const cl_mem *mem_list,
    const void **args_mem_loc, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event)
{
  f_clEnqueueNativeKernel func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueNativeKernel)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueNativeKernel");
  if (func)
    {
      return func (command_queue, user_func, args, cb_args, num_mem_objects,
                   mem_list, args_mem_loc, num_events_in_wait_list,
                   event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueMarkerWithWaitList) (cl_command_queue command_queue,
                                        cl_uint num_events_in_wait_list,
                                        const cl_event *event_wait_list,
                                        cl_event *event)
{
  f_clEnqueueMarkerWithWaitList func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueMarkerWithWaitList)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueMarkerWithWaitList");
  if (func)
    {
      return func (command_queue, num_events_in_wait_list, event_wait_list,
                   event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueBarrierWithWaitList) (cl_command_queue command_queue,
                                         cl_uint num_events_in_wait_list,
                                         const cl_event *event_wait_list,
                                         cl_event *event)
{
  f_clEnqueueBarrierWithWaitList func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueBarrierWithWaitList)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueBarrierWithWaitList");
  if (func)
    {
      return func (command_queue, num_events_in_wait_list, event_wait_list,
                   event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

void *
stubname (clGetExtensionFunctionAddressForPlatform) (cl_platform_id platform,
                                                     const char *func_name)
{
  f_clGetExtensionFunctionAddressForPlatform func;

  if (!so_handle)
    open_libopencl_so ();

  func
    = (f_clGetExtensionFunctionAddressForPlatform)pocl_dynlib_symbol_address (
      so_handle, "clGetExtensionFunctionAddressForPlatform");
  if (func)
    {
      return func (platform, func_name);
    }
  else
    {
      return NULL;
    }
}

cl_mem
stubname (clCreateImage2D) (cl_context context, cl_mem_flags flags,
                            const cl_image_format *image_format,
                            size_t image_width, size_t image_height,
                            size_t image_row_pitch, void *host_ptr,
                            cl_int *errcode_ret)
{
  f_clCreateImage2D func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateImage2D)pocl_dynlib_symbol_address (so_handle,
                                                        "clCreateImage2D");
  if (func)
    {
      return func (context, flags, image_format, image_width, image_height,
                   image_row_pitch, host_ptr, errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_mem
stubname (clCreateImage3D) (cl_context context, cl_mem_flags flags,
                            const cl_image_format *image_format,
                            size_t image_width, size_t image_height,
                            size_t image_depth, size_t image_row_pitch,
                            size_t image_slice_pitch, void *host_ptr,
                            cl_int *errcode_ret)
{
  f_clCreateImage3D func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateImage3D)pocl_dynlib_symbol_address (so_handle,
                                                        "clCreateImage3D");
  if (func)
    {
      return func (context, flags, image_format, image_width, image_height,
                   image_depth, image_row_pitch, image_slice_pitch, host_ptr,
                   errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_int
stubname (clEnqueueMarker) (cl_command_queue command_queue, cl_event *event)
{
  f_clEnqueueMarker func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueMarker)pocl_dynlib_symbol_address (so_handle,
                                                        "clEnqueueMarker");
  if (func)
    {
      return func (command_queue, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueWaitForEvents) (cl_command_queue command_queue,
                                   cl_uint num_events,
                                   const cl_event *event_list)
{
  f_clEnqueueWaitForEvents func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueWaitForEvents)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueWaitForEvents");
  if (func)
    {
      return func (command_queue, num_events, event_list);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueBarrier) (cl_command_queue command_queue)
{
  f_clEnqueueBarrier func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueBarrier)pocl_dynlib_symbol_address (so_handle,
                                                         "clEnqueueBarrier");
  if (func)
    {
      return func (command_queue);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clUnloadCompiler) (void)
{
  f_clUnloadCompiler func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clUnloadCompiler)pocl_dynlib_symbol_address (so_handle,
                                                         "clUnloadCompiler");
  if (func)
    {
      return func ();
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

void *
stubname (clGetExtensionFunctionAddress) (const char *func_name)
{
  f_clGetExtensionFunctionAddress func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetExtensionFunctionAddress)pocl_dynlib_symbol_address (
    so_handle, "clGetExtensionFunctionAddress");
  if (func)
    {
      return func (func_name);
    }
  else
    {
      return NULL;
    }
}

cl_mem
stubname (clCreateFromGLBuffer) (cl_context context, cl_mem_flags flags,
                                 cl_GLuint bufobj, int *errcode_ret)
{
  f_clCreateFromGLBuffer func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateFromGLBuffer)pocl_dynlib_symbol_address (
    so_handle, "clCreateFromGLBuffer");
  if (func)
    {
      return func (context, flags, bufobj, errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_mem
stubname (clCreateFromGLTexture) (cl_context context, cl_mem_flags flags,
                                  cl_GLenum target, cl_GLint miplevel,
                                  cl_GLuint texture, cl_int *errcode_ret)
{
  f_clCreateFromGLTexture func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateFromGLTexture)pocl_dynlib_symbol_address (
    so_handle, "clCreateFromGLTexture");
  if (func)
    {
      return func (context, flags, target, miplevel, texture, errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_mem
stubname (clCreateFromGLRenderbuffer) (cl_context context, cl_mem_flags flags,
                                       cl_GLuint renderbuffer,
                                       cl_int *errcode_ret)
{
  f_clCreateFromGLRenderbuffer func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateFromGLRenderbuffer)pocl_dynlib_symbol_address (
    so_handle, "clCreateFromGLRenderbuffer");
  if (func)
    {
      return func (context, flags, renderbuffer, errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_int
stubname (clGetGLObjectInfo) (cl_mem memobj, cl_gl_object_type *gl_object_type,
                              cl_GLuint *gl_object_name)
{
  f_clGetGLObjectInfo func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetGLObjectInfo)pocl_dynlib_symbol_address (so_handle,
                                                          "clGetGLObjectInfo");
  if (func)
    {
      return func (memobj, gl_object_type, gl_object_name);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clGetGLTextureInfo) (cl_mem memobj, cl_gl_texture_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret)
{
  f_clGetGLTextureInfo func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetGLTextureInfo)pocl_dynlib_symbol_address (
    so_handle, "clGetGLTextureInfo");
  if (func)
    {
      return func (memobj, param_name, param_value_size, param_value,
                   param_value_size_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueAcquireGLObjects) (cl_command_queue command_queue,
                                      cl_uint num_objects,
                                      const cl_mem *mem_objects,
                                      cl_uint num_events_in_wait_list,
                                      const cl_event *event_wait_list,
                                      cl_event *event)
{
  f_clEnqueueAcquireGLObjects func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueAcquireGLObjects)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueAcquireGLObjects");
  if (func)
    {
      return func (command_queue, num_objects, mem_objects,
                   num_events_in_wait_list, event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_int
stubname (clEnqueueReleaseGLObjects) (cl_command_queue command_queue,
                                      cl_uint num_objects,
                                      const cl_mem *mem_objects,
                                      cl_uint num_events_in_wait_list,
                                      const cl_event *event_wait_list,
                                      cl_event *event)
{
  f_clEnqueueReleaseGLObjects func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clEnqueueReleaseGLObjects)pocl_dynlib_symbol_address (
    so_handle, "clEnqueueReleaseGLObjects");
  if (func)
    {
      return func (command_queue, num_objects, mem_objects,
                   num_events_in_wait_list, event_wait_list, event);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}

cl_mem
stubname (clCreateFromGLTexture2D) (cl_context context, cl_mem_flags flags,
                                    cl_GLenum target, cl_GLint miplevel,
                                    cl_GLuint texture, cl_int *errcode_ret)
{
  f_clCreateFromGLTexture2D func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateFromGLTexture2D)pocl_dynlib_symbol_address (
    so_handle, "clCreateFromGLTexture2D");
  if (func)
    {
      return func (context, flags, target, miplevel, texture, errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_mem
stubname (clCreateFromGLTexture3D) (cl_context context, cl_mem_flags flags,
                                    cl_GLenum target, cl_GLint miplevel,
                                    cl_GLuint texture, cl_int *errcode_ret)
{
  f_clCreateFromGLTexture3D func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clCreateFromGLTexture3D)pocl_dynlib_symbol_address (
    so_handle, "clCreateFromGLTexture3D");
  if (func)
    {
      return func (context, flags, target, miplevel, texture, errcode_ret);
    }
  else
    {
      return NULL;
    }
}

cl_int
stubname (clGetGLContextInfoKHR) (const cl_context_properties *properties,
                                  cl_gl_context_info param_name,
                                  size_t param_value_size, void *param_value,
                                  size_t *param_value_size_ret)
{
  f_clGetGLContextInfoKHR func;

  if (!so_handle)
    open_libopencl_so ();

  func = (f_clGetGLContextInfoKHR)pocl_dynlib_symbol_address (
    so_handle, "clGetGLContextInfoKHR");
  if (func)
    {
      return func (properties, param_name, param_value_size, param_value,
                   param_value_size_ret);
    }
  else
    {
      return CL_INVALID_PLATFORM;
    }
}
