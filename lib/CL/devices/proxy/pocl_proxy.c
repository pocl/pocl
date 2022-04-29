/* proxy.c - a pocl device driver which delegates to proxied OpenCL devices

   Copyright (c) 2021 Michal Babej / Tampere University

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

#include "config.h"
#include "pocl_proxy.h"
#include "common.h"
#include "devices.h"

#include <assert.h>
#include <alloca.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stddef.h>
#include <stdint.h>

#include <CL/cl.h>
#include <CL/cl_egl.h>
#include "pocl_cl.h"
#include "pocl_cache.h"
#include "pocl_timing.h"
#include "pocl_file_util.h"
#include "pocl_util.h"
#include "pocl_mem_management.h"
#include "pocl_image_util.h"
#include "common_driver.h"
#include "utlist.h"

#ifdef ENABLE_ICD
#error This driver cannot be built when pocl is to be linked against ICD
#endif

/*****************************************************************************/

typedef struct proxy_platform_data_s
{
  cl_platform_id id;
  cl_device_id *orig_devices;
  cl_device_id *pocl_devices;
  cl_uint num_devices;
  char has_gl_interop;
  char provides_metadata;
  char supports_binaries;
} proxy_platform_data_t;

typedef struct proxy_device_data_s
{
  proxy_platform_data_t *backend;
  // copy of backend->id
  cl_platform_id platform_id;
  // copy of backend->devices[device-index]
  cl_device_id device_id;

} proxy_device_data_t;

typedef struct proxy_queue_data_s
{
  // lock
  ALIGN_CACHE (pocl_lock_t wq_lock);
  // ready to launch queue
  _cl_command_node *work_queue;

  // for user threads that are waiting on clFinish
  ALIGN_CACHE (pocl_cond_t wait_cond);

  // for waking up the queue thread
  ALIGN_CACHE (pocl_cond_t wakeup_cond);

  // backend queue id
  cl_command_queue proxied_id;
  // pocl queue id
  cl_command_queue queue;

  // index of this queue's device in context->devices[]
  unsigned context_device_i;

  // ask the thread to exit
  int cq_thread_exit_requested;
  /* queue pthread */
  pocl_thread_t cq_thread_id;

} proxy_queue_data_t;

typedef struct pocl_proxy_event_data_s
{
  pocl_cond_t event_cond;
} pocl_proxy_event_data_t;

static const char proxy_device_name[] = "proxy";
static cl_uint num_platforms = 0;
static proxy_platform_data_t *platforms = NULL;

void
pocl_proxy_init_device_ops (struct pocl_device_ops *ops)
{
  memset (ops, 0, sizeof (struct pocl_device_ops));
  ops->device_name = proxy_device_name;

  ops->probe = pocl_proxy_probe;
  ops->init = pocl_proxy_init;
  ops->uninit = pocl_proxy_uninit;
  ops->reinit = pocl_proxy_reinit;

  ops->alloc_mem_obj = pocl_proxy_alloc_mem_obj;
  ops->free = pocl_proxy_free;
  ops->can_migrate_d2d = pocl_proxy_can_migrate_d2d;
  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;

  ops->create_kernel = pocl_proxy_create_kernel;
  ops->free_kernel = pocl_proxy_free_kernel;
  ops->init_queue = pocl_proxy_init_queue;
  ops->free_queue = pocl_proxy_free_queue;
  ops->init_context = pocl_proxy_init_context;
  ops->free_context = pocl_proxy_free_context;

  ops->build_source = pocl_proxy_build_source;
  ops->link_program = pocl_proxy_link_program;
  ops->build_binary = pocl_proxy_build_binary;
  ops->free_program = pocl_proxy_free_program;
  ops->setup_metadata = pocl_proxy_setup_metadata;
  ops->supports_binary = NULL;

  ops->join = pocl_proxy_join;
  ops->submit = pocl_proxy_submit;
  ops->broadcast = pocl_broadcast;
  ops->notify = pocl_proxy_notify;
  ops->flush = pocl_proxy_flush;
  ops->wait_event = pocl_proxy_wait_event;
  ops->free_event_data = pocl_proxy_free_event_data;
  ops->notify_cmdq_finished = pocl_proxy_notify_cmdq_finished;
  ops->notify_event_finished = pocl_proxy_notify_event_finished;
  ops->build_hash = pocl_proxy_build_hash;

  ops->create_sampler = pocl_proxy_create_sampler;
  ops->free_sampler = pocl_proxy_free_sampler;

#ifdef ENABLE_CL_GET_GL_CONTEXT
  ops->get_gl_context_assoc = pocl_proxy_get_gl_context_assoc;
#endif
}

char *
pocl_proxy_build_hash (cl_device_id device)
{
  char *res = (char *)calloc (1000, sizeof (char));
  snprintf (res, 1000, "pocl-proxy: %s", device->short_name);
  return res;
}

unsigned
pocl_proxy_probe (struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count (ops->device_name);
  if (env_count <= 0)
    return 0;
  else
    POCL_MSG_PRINT_PROXY ("Requested %i proxy devices.\n", env_count);

  assert (num_platforms == 0);
  cl_int err = clGetPlatformIDs (0, NULL, &num_platforms);

  if (err != CL_SUCCESS || (num_platforms == 0))
    {
      POCL_MSG_ERR ("clGetPlatformIDs returned 0 platforms! %i / %u \n", err,
                    num_platforms);
      return 0;
    }

  POCL_MSG_PRINT_PROXY ("clGetPlatformIDs found %u platforms.\n",
                        num_platforms);

  cl_uint i, total = 0;

  char platform_extensions[2048];
  char platform_vendor[256];
  size_t ext_len, ven_len;

  cl_platform_id *platform_ids
      = (cl_platform_id *)alloca (num_platforms * sizeof (cl_platform_id));
  err = clGetPlatformIDs (num_platforms, platform_ids, NULL);
  assert (err == CL_SUCCESS);

  platforms = (proxy_platform_data_t *)calloc (num_platforms,
                                               sizeof (proxy_platform_data_t));
  assert (platforms);

  for (i = 0; i < num_platforms; ++i)
    {
      platforms[i].id = platform_ids[i];
      cl_uint num;
      err = clGetDeviceIDs (platforms[i].id, CL_DEVICE_TYPE_ALL, 0, NULL,
                            &num);
      assert (err == CL_SUCCESS);
      assert (num > 0);
      platforms[i].num_devices = num;
      platforms[i].orig_devices
          = (cl_device_id *)calloc (num, sizeof (cl_device_id));
      platforms[i].pocl_devices
          = (cl_device_id *)calloc (num, sizeof (cl_device_id));
      platforms[i].has_gl_interop = 0;
      err = clGetDeviceIDs (platforms[i].id, CL_DEVICE_TYPE_ALL, num,
                            platforms[i].orig_devices, NULL);
      assert (err == CL_SUCCESS);

      err = clGetPlatformInfo (platforms[i].id, CL_PLATFORM_VENDOR, 256,
                               platform_vendor, &ven_len);
      assert (err == CL_SUCCESS);
      assert (ven_len > 0);
      platform_vendor[ven_len] = 0;

      err = clGetPlatformInfo (platforms[i].id, CL_PLATFORM_EXTENSIONS, 2048,
                               platform_extensions, &ext_len);
      assert (err == CL_SUCCESS);
      assert (ext_len > 0);
      platform_extensions[ext_len] = 0;

      /* ARM Mali SDK (and the Android clones) seems to have a faulty
       * implementation of clGetKernelArgInfo. this is a workaround to get the
       * kernel arg info from other device drivers (like remote). (obvious
       * drawback: there must be other platform devices in the context) */
#ifdef __ANDROID__
      platforms[i].provides_metadata = 0;
#else
      if (strstr (platform_vendor, "ARM"))
        platforms[i].provides_metadata = 0;
      else
        platforms[i].provides_metadata = 1;
#endif
      platforms[i].supports_binaries = platforms[i].provides_metadata;

#if defined(ENABLE_OPENGL_INTEROP) || defined(ENABLE_EGL_INTEROP)
      if (strstr (platform_extensions, "cl_khr_gl_sharing")
          || strstr (platform_extensions, "cl_khr_egl_image"))
        {
          platforms[i].has_gl_interop = CL_TRUE;
        }
#endif

      POCL_MSG_PRINT_PROXY (
          "Platform %u: %s Devices: %u OpenGL/EGL interop: %u Metadata: %u\n",
          i, platform_vendor, platforms[i].num_devices,
          platforms[i].has_gl_interop, platforms[i].provides_metadata);

      total += num;
    }

  if (total < (cl_uint)env_count)
    POCL_MSG_ERR ("Requested %i proxy devices, but only %u available.\n",
                  env_count, total);

  if ((cl_uint)env_count < total)
    {
      POCL_MSG_PRINT_PROXY (
          "Requested %i proxy devices, but more are available (%u).\n",
          env_count, total);
      return env_count;
    }
  else
    return total;
}

#ifdef ENABLE_CL_GET_GL_CONTEXT
cl_int
pocl_proxy_get_gl_context_assoc (cl_device_id device, cl_gl_context_info type,
                                 const cl_context_properties *properties)
{
  proxy_device_data_t *d = (proxy_device_data_t *)(device->data);
  cl_platform_id proxy_platform = d->platform_id;
  cl_device_id proxy_device = d->device_id;
  char has_gl = d->backend->has_gl_interop;

  POCL_RETURN_ERROR_ON ((has_gl == 0), CL_INVALID_OPERATION,
                        "device doesn't support CL-GL interop!\n");

  // find the platform index in the properties;
  // replace pocl platform with backend platform
  cl_context_properties fixed_props[1024];
  const cl_context_properties *p = properties;
  unsigned j, k, i = 0;
  unsigned platform_index = UINT32_MAX;
  while ((i < 1024) && (p[i] != 0))
    {
      fixed_props[i] = p[i];
      if (p[i] == CL_CONTEXT_PLATFORM)
        platform_index = i + 1;
      ++i;
    }
  // add closing 0
  assert (p[i] == 0);
  fixed_props[i] = 0;

  assert (i < 1024);
  assert (platform_index < i);
  assert (platform_index >= 0);
  assert (p[platform_index] != 0);

  fixed_props[platform_index] = (cl_context_properties)proxy_platform;

  cl_device_id retval = NULL;
  size_t retval_size = sizeof (cl_device_id);
  int err = clGetGLContextInfoKHR (fixed_props, type, sizeof (cl_device_id),
                                   &retval, &retval_size);
  if (err == CL_SUCCESS && retval_size > 0 && retval == proxy_device)
    return CL_SUCCESS;
  else
    return CL_INVALID_VALUE;
}
#endif

int
pocl_proxy_init_context (cl_device_id device, cl_context context)
{
  int err = CL_SUCCESS;
  unsigned i;
  context->proxied_context = NULL;

  proxy_device_data_t *d = (proxy_device_data_t *)(device->data);
  cl_platform_id proxy_platform = d->platform_id;
  cl_device_id proxy_device = d->device_id;

  /* replace PoCL platform in properties with the backend cl_platform */
  cl_context_properties *p = context->properties;
  cl_context_properties saved_platform = 0;
  while (p && p[0])
    {
      if (p[0] == CL_CONTEXT_PLATFORM)
        {
          saved_platform = p[1];
          p[1] = (cl_context_properties)proxy_platform;
        }
      p += 2;
    }

  cl_context proxy_ctx = clCreateContext (context->properties, 1,
                                          &proxy_device, NULL, NULL, &err);
  if (err != CL_SUCCESS)
    {
      POCL_MSG_ERR ("clCreateContext call in backing OpenCL implementation "
                    "failed! error: %i\n",
                    err);
      return err;
    }
  context->proxied_context = (cl_context)proxy_ctx;

  // restore platform
  p = context->properties;
  while (p && p[0])
    {
      if (p[0] == CL_CONTEXT_PLATFORM)
        p[1] = saved_platform;
      p += 2;
    }

  return err;
}

int
pocl_proxy_free_context (cl_device_id device, cl_context context)
{
  cl_context proxy_ctx = (cl_context)context->proxied_context;
  if (proxy_ctx == NULL)
    return CL_SUCCESS;

  int err = clReleaseContext (proxy_ctx);
  if (err != CL_SUCCESS)
    POCL_MSG_ERR ("clReleaseContext call in backing OpenCL failed!\n");

  context->proxied_context = NULL;
  return err;
}

static void *pocl_proxy_queue_pthread (void *ptr);

#define D(info, name)                                                         \
  err = clGetDeviceInfo (id, name, size, &device->info, NULL);                \
  assert (err == CL_SUCCESS);

#define DIuint(info, name)                                                    \
  err = clGetDeviceInfo (id, name, sizeof (cl_uint), &device->info, NULL);    \
  assert (err == CL_SUCCESS)
#define DIulong(info, name)                                                   \
  err = clGetDeviceInfo (id, name, sizeof (cl_ulong), &device->info, NULL);   \
  assert (err == CL_SUCCESS)
#define DIsizet(info, name)                                                   \
  err = clGetDeviceInfo (id, name, sizeof (size_t), &device->info, NULL);     \
  assert (err == CL_SUCCESS)
#define DIbool(info, name)                                                    \
  err = clGetDeviceInfo (id, name, sizeof (cl_bool), &device->info, NULL);    \
  assert (err == CL_SUCCESS)
#define DIflag(info, name)                                                    \
  err = clGetDeviceInfo (id, name, sizeof (cl_ulong), &device->info, NULL);   \
  assert (err == CL_SUCCESS)

#define DIstring(INFO, name)                                                  \
  {                                                                           \
    size_t len;                                                               \
    err = clGetDeviceInfo (id, name, 0, NULL, &len);                          \
    assert (err == CL_SUCCESS);                                               \
    device->INFO = (char *)malloc (len);                                      \
    assert (device->INFO);                                                    \
    err = clGetDeviceInfo (id, name, len, (void *)(device->INFO), NULL);      \
    assert (err == CL_SUCCESS);                                               \
  }

void
pocl_proxy_get_device_info (cl_device_id device, proxy_device_data_t *d)
{
  cl_device_id id = d->device_id;
  int err;

  DIbool (host_unified_memory, CL_DEVICE_HOST_UNIFIED_MEMORY);

  device->execution_capabilities = CL_EXEC_KERNEL;

  DIstring (opencl_c_version_as_opt, CL_DEVICE_OPENCL_C_VERSION);

  DIstring (short_name, CL_DEVICE_NAME);

  device->long_name = device->short_name;

  DIstring (version, CL_DEVICE_VERSION);

  DIstring (driver_version, CL_DRIVER_VERSION);

  DIstring (vendor, CL_DEVICE_VENDOR);

  // TODO strip extensions

  DIstring (extensions, CL_DEVICE_EXTENSIONS);

#ifdef ENABLE_OPENGL_INTEROP
  if (strstr (device->extensions, "cl_khr_gl_sharing"))
    {
      POCL_MSG_PRINT_PROXY ("Device %s has OpenGL interop\n",
                            device->long_name);
      device->has_gl_interop = CL_TRUE;
      d->backend->has_gl_interop = CL_TRUE;
    }
  else
    POCL_MSG_PRINT_PROXY ("Device %s doesn't have OpenGL interop\n",
                          device->long_name);
#endif

#ifdef ENABLE_EGL_INTEROP
  if (strstr (device->extensions, "cl_khr_egl_image"))
    {
      POCL_MSG_PRINT_PROXY ("Device %s has EGL interop\n", device->long_name);
      device->has_gl_interop = CL_TRUE;
      d->backend->has_gl_interop = CL_TRUE;
    }
  else
    POCL_MSG_PRINT_PROXY ("Device %s doesn't have EGL interop\n",
                          device->long_name);
#endif

  // This one is deprecated (and seems to be always 128)
  device->min_data_type_align_size = 128;

  DIuint (vendor_id, CL_DEVICE_VENDOR_ID);
  // TODO
  // D(device_id);

  DIuint (address_bits, CL_DEVICE_ADDRESS_BITS);
  DIuint (mem_base_addr_align, CL_DEVICE_MEM_BASE_ADDR_ALIGN);
  if (device->mem_base_addr_align < 4)
    device->mem_base_addr_align = MAX_EXTENDED_ALIGNMENT;

  DIulong (global_mem_cache_size, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
  DIflag (global_mem_cache_type, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE);
  DIulong (global_mem_size, CL_DEVICE_GLOBAL_MEM_SIZE);
  DIuint (global_mem_cacheline_size, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE);
  DIulong (max_mem_alloc_size, CL_DEVICE_MAX_MEM_ALLOC_SIZE);

  DIflag (double_fp_config, CL_DEVICE_DOUBLE_FP_CONFIG);
  DIflag (single_fp_config, CL_DEVICE_SINGLE_FP_CONFIG);
  // TODO
  // DIflag(half_fp_config, CL_DEVICE_HALF_FP_CONFIG);

  DIbool (image_support, CL_DEVICE_IMAGE_SUPPORT);
  DIbool (endian_little, CL_DEVICE_ENDIAN_LITTLE);
  DIbool (error_correction_support, CL_DEVICE_ERROR_CORRECTION_SUPPORT);

  DIflag (type, CL_DEVICE_TYPE);
  DIstring (profile, CL_DEVICE_PROFILE);

  // TODO queue properties
  device->queue_properties = CL_QUEUE_PROFILING_ENABLE;
  DIbool (available, CL_DEVICE_AVAILABLE);
  DIbool (compiler_available, CL_DEVICE_COMPILER_AVAILABLE);
  DIbool (linker_available, CL_DEVICE_LINKER_AVAILABLE);

  DIflag (local_mem_type, CL_DEVICE_LOCAL_MEM_TYPE);
  DIulong (local_mem_size, CL_DEVICE_LOCAL_MEM_SIZE);

  DIuint (max_clock_frequency, CL_DEVICE_MAX_CLOCK_FREQUENCY);
  DIuint (max_compute_units, CL_DEVICE_MAX_COMPUTE_UNITS);

  DIuint (max_constant_args, CL_DEVICE_MAX_CONSTANT_ARGS);
  DIulong (max_constant_buffer_size, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
  DIsizet (max_parameter_size, CL_DEVICE_MAX_PARAMETER_SIZE);

  DIuint (max_work_item_dimensions, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
  DIsizet (max_work_group_size, CL_DEVICE_MAX_WORK_GROUP_SIZE);

  size_t item_sizes[4] = {
    0,
  };
  err = clGetDeviceInfo (id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                         (sizeof (size_t) * device->max_work_item_dimensions),
                         &item_sizes, NULL);
  assert (err == CL_SUCCESS);

  device->max_work_item_sizes[0] = item_sizes[0];
  device->max_work_item_sizes[1] = item_sizes[1];
  device->max_work_item_sizes[2] = item_sizes[2];

  DIuint (native_vector_width_char, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR);
  DIuint (native_vector_width_short, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT);
  DIuint (native_vector_width_int, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT);
  DIuint (native_vector_width_long, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG);
  DIuint (native_vector_width_float, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT);
  DIuint (native_vector_width_double, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE);

  DIuint (preferred_vector_width_char, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
  DIuint (preferred_vector_width_short,
          CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
  DIuint (preferred_vector_width_int, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
  DIuint (preferred_vector_width_long, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
  DIuint (preferred_vector_width_float,
          CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
  DIuint (preferred_vector_width_double,
          CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);

  DIsizet (printf_buffer_size, CL_DEVICE_PRINTF_BUFFER_SIZE);
  DIsizet (profiling_timer_resolution, CL_DEVICE_PROFILING_TIMER_RESOLUTION);

  if (device->image_support == CL_FALSE)
    return;

  const cl_context_properties props[]
      = { CL_CONTEXT_PLATFORM, (cl_context_properties)d->platform_id, 0 };
  cl_context context
      = clCreateContext (props, 1, &d->device_id, NULL, NULL, &err);
  assert (err == CL_SUCCESS);

  DIuint (max_read_image_args, CL_DEVICE_MAX_READ_IMAGE_ARGS);
  DIuint (max_write_image_args, CL_DEVICE_MAX_WRITE_IMAGE_ARGS);
  DIuint (max_samplers, CL_DEVICE_MAX_SAMPLERS);
  DIsizet (image2d_max_height, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
  DIsizet (image2d_max_width, CL_DEVICE_IMAGE2D_MAX_WIDTH);
  DIsizet (image3d_max_height, CL_DEVICE_IMAGE3D_MAX_HEIGHT);
  DIsizet (image3d_max_width, CL_DEVICE_IMAGE3D_MAX_WIDTH);
  DIsizet (image3d_max_depth, CL_DEVICE_IMAGE3D_MAX_DEPTH);
  DIsizet (image_max_buffer_size, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE);
  DIsizet (image_max_array_size, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE);

  size_t i;
  cl_mem_object_type image_types[]
      = { CL_MEM_OBJECT_IMAGE1D,        CL_MEM_OBJECT_IMAGE1D_ARRAY,
          CL_MEM_OBJECT_IMAGE1D_BUFFER, CL_MEM_OBJECT_IMAGE2D,
          CL_MEM_OBJECT_IMAGE2D_ARRAY,  CL_MEM_OBJECT_IMAGE3D };
  for (i = 0; i < NUM_OPENCL_IMAGE_TYPES; ++i)
    {
      cl_mem_object_type type = image_types[i];
      cl_uint retval, num_image_formats = 0;
      err = clGetSupportedImageFormats (context, CL_MEM_READ_WRITE, type, 0,
                                        NULL, &num_image_formats);
      assert (err == CL_SUCCESS);

      if (num_image_formats == 0)
        continue;

      cl_image_format *formats = (cl_image_format *)calloc (
          num_image_formats, sizeof (cl_image_format));
      err = clGetSupportedImageFormats (context, CL_MEM_READ_WRITE, type,
                                        num_image_formats, formats, &retval);
      assert (err == CL_SUCCESS);
      assert (retval == num_image_formats);

      int type_index = opencl_image_type_to_index (type);
      device->image_formats[type_index] = formats;
      device->num_image_formats[type_index] = num_image_formats;
    }

  err = clReleaseContext (context);
  assert (err == CL_SUCCESS);
}

cl_int
pocl_proxy_init (unsigned j, cl_device_id dev, const char *parameters)
{
  cl_uint plat_i = 0, dev_i = j;
  while ((plat_i < num_platforms) && (dev_i >= platforms[plat_i].num_devices))
    {
      dev_i -= platforms[plat_i].num_devices;
      ++plat_i;
    }
  if (plat_i >= num_platforms)
    return CL_INVALID_DEVICE;

  proxy_device_data_t *d;
  d = (proxy_device_data_t *)calloc (1, sizeof (proxy_device_data_t));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;
  dev->data = d;

  d->backend = &platforms[plat_i];
  d->platform_id = platforms[plat_i].id;
  d->device_id = platforms[plat_i].orig_devices[dev_i];
  platforms[plat_i].pocl_devices[dev_i] = dev;

  pocl_proxy_get_device_info (dev, d);

  assert (dev->long_name);
  assert (dev->short_name);

  return CL_SUCCESS;
}

cl_int
pocl_proxy_uninit (unsigned j, cl_device_id device)
{
  return CL_SUCCESS;
}

cl_int
pocl_proxy_reinit (unsigned j, cl_device_id device)
{
  return CL_SUCCESS;
}

int
pocl_proxy_alloc_mem_obj (cl_device_id device, cl_mem mem, void *host_ptr)
{
  proxy_device_data_t *d = (proxy_device_data_t *)device->data;
  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];
  assert (p->mem_ptr == NULL);

  int r = 0;
  assert (mem->id > 0);
  cl_context proxy_ctx = mem->context->proxied_context;

  int ret = CL_MEM_OBJECT_ALLOCATION_FAILURE;

  cl_mem_flags reduced_flags
      = mem->flags
        & (CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_READ_WRITE
           | CL_MEM_HOST_WRITE_ONLY);

  /* proxy driver doesn't preallocate */
  if ((mem->flags & CL_MEM_ALLOC_HOST_PTR) && (mem->mem_host_ptr == NULL))
    goto ERROR;

  cl_mem buf = NULL;
  if (mem->is_image)
    {
#ifdef ENABLE_OPENGL_INTEROP
      if (mem->is_gl_texture)
        {
          assert (mem->context->gl_interop);
          buf = clCreateFromGLTexture (proxy_ctx, mem->flags, mem->target,
                                       mem->miplevel, mem->texture, &r);
        }
      else
#endif
#ifdef ENABLE_EGL_INTEROP
          if (mem->is_gl_texture)
        {
          assert (mem->context->gl_interop);
          buf = clCreateFromEGLImageKHR (proxy_ctx, mem->egl_display,
                                         mem->egl_image, mem->flags, NULL, &r);
        }
      else
#endif
        {
          assert (mem->is_gl_texture == 0);
          cl_image_format format;
          format.image_channel_data_type = mem->image_channel_data_type;
          format.image_channel_order = mem->image_channel_order;
          cl_image_desc desc;
          desc.image_array_size = mem->image_array_size;
          desc.image_width = mem->image_width;
          desc.image_height = mem->image_height;
          desc.image_depth = mem->image_depth;
          desc.image_row_pitch = mem->image_row_pitch;
          desc.image_slice_pitch = mem->image_slice_pitch;
          desc.image_type = mem->type;
          desc.buffer = NULL; // TODO FIXME
          desc.num_mip_levels = 0;
          desc.num_samples = 0;
          buf = clCreateImage (proxy_ctx, reduced_flags, &format, &desc, NULL,
                               &r);
        }

      if (r != CL_SUCCESS)
        POCL_MSG_ERR ("proxy: image alloc failed with %i\n", r);
    }
  else
    {
      buf = clCreateBuffer (proxy_ctx, reduced_flags, mem->size, NULL, &r);

      if (r != CL_SUCCESS)
        POCL_MSG_ERR ("proxy: mem alloc failed with %i\n", r);
    }
  if (buf == NULL)
    goto ERROR;

  POCL_MSG_PRINT_MEMORY ("proxy DEVICE ALLOC PTR %p "
                         "SIZE %zu IMAGE: %i" PRIu64 "\n",
                         p->mem_ptr, mem->size, mem->is_image);
  p->mem_ptr = buf;
  p->version = 0;

  ret = CL_SUCCESS;

ERROR:
  return ret;
}

void
pocl_proxy_free (cl_device_id device, cl_mem mem)
{
  proxy_device_data_t *d = (proxy_device_data_t *)device->data;
  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];
  assert (p->mem_ptr != NULL);

  assert (mem->id != 0);
  cl_mem m = (cl_mem)p->mem_ptr;
  assert (m);

  int err = clReleaseMemObject (m);
  if (err != CL_SUCCESS)
    return;

  p->mem_ptr = NULL;
  p->version = 0;

  POCL_MSG_PRINT_MEMORY ("PROXY DEVICE FREED PTR %p "
                         "SIZE %zu \n",
                         p->mem_ptr, mem->size);
}

#define MAX_TESTED_ARG_SIZE 128

static int
get_kernel_metadata (pocl_kernel_metadata_t *meta, cl_uint num_devices,
                     cl_program prog, cl_device_id device, cl_kernel kernel)
{
  char string_value[POCL_FILENAME_LENGTH];
  int err;
  size_t size;

  // device-specific
  assert (meta->data == NULL);
  meta->data = (void **)calloc (num_devices, sizeof (void *));
  meta->has_arg_metadata = (-1);

  err = clGetKernelInfo (kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &size);
  assert (err == CL_SUCCESS);
  assert (size > 0);
  assert (size < POCL_FILENAME_LENGTH);

  err = clGetKernelInfo (kernel, CL_KERNEL_FUNCTION_NAME, size, string_value,
                         NULL);
  assert (err == CL_SUCCESS);
  meta->name = (char *)malloc (size);
  memcpy (meta->name, string_value, size);

  err = clGetKernelInfo (kernel, CL_KERNEL_ATTRIBUTES, 0, NULL, &size);
  assert (err == CL_SUCCESS);
  assert (size < POCL_FILENAME_LENGTH);

  if (size > 0)
    {
      err = clGetKernelInfo (kernel, CL_KERNEL_ATTRIBUTES, size, string_value,
                             NULL);
      assert (err == CL_SUCCESS);
      meta->attributes = (char *)malloc (size);
      memcpy (meta->attributes, string_value, size);
    }
  else
    meta->attributes = NULL;

  cl_ulong local_mem_size = 0;
  err = clGetKernelWorkGroupInfo (kernel, device, CL_KERNEL_LOCAL_MEM_SIZE,
                                  sizeof (cl_ulong), &local_mem_size, NULL);
  // TODO
  assert (err == CL_SUCCESS);

  meta->num_locals = 1;
  meta->local_sizes = (size_t *)calloc (1, sizeof (size_t));
  meta->local_sizes[0] = local_mem_size;

  size_t reqd_wg_size[3];
  err = clGetKernelWorkGroupInfo (kernel, device,
                                  CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                  sizeof (reqd_wg_size), &reqd_wg_size, NULL);
  // TODO
  assert (err == CL_SUCCESS);

  meta->reqd_wg_size[0] = reqd_wg_size[0];
  meta->reqd_wg_size[1] = reqd_wg_size[1];
  meta->reqd_wg_size[2] = reqd_wg_size[2];

  cl_uint num_args;

  err = clGetKernelInfo (kernel, CL_KERNEL_NUM_ARGS, sizeof (num_args),
                         &num_args, NULL);
  assert (err == CL_SUCCESS);

  if (num_args == 0)
    {
      meta->arg_info = NULL;
      meta->num_args = 0;
      return CL_SUCCESS;
    }

  assert (num_args < 10000);

  meta->num_args = num_args;
  meta->arg_info
      = (pocl_argument_info *)calloc (num_args, sizeof (pocl_argument_info));

  char empty_buffer[MAX_TESTED_ARG_SIZE];

  cl_uint i;
  for (i = 0; i < num_args; ++i)
    {
      pocl_argument_info *pi = &meta->arg_info[i];
      err = clGetKernelArgInfo (kernel, i, CL_KERNEL_ARG_ACCESS_QUALIFIER,
                                sizeof (pi->access_qualifier),
                                &pi->access_qualifier, NULL);
      // TODO
      //      if (err == CL_KERNEL_ARG_INFO_NOT_AVAILABLE)
      //        pi->access_qualifier = CL_KERNEL_ARG_ACCESS_NONE;
      //      else
      assert (err == CL_SUCCESS);

      err = clGetKernelArgInfo (kernel, i, CL_KERNEL_ARG_ADDRESS_QUALIFIER,
                                sizeof (pi->address_qualifier),
                                &pi->address_qualifier, NULL);
      assert (err == CL_SUCCESS);

      err = clGetKernelArgInfo (kernel, i, CL_KERNEL_ARG_TYPE_QUALIFIER,
                                sizeof (pi->type_qualifier),
                                &pi->type_qualifier, NULL);
      assert (err == CL_SUCCESS);

      err = clGetKernelArgInfo (kernel, i, CL_KERNEL_ARG_TYPE_NAME, 0, NULL,
                                &size);
      assert (err == CL_SUCCESS);
      assert (size < POCL_FILENAME_LENGTH);
      err = clGetKernelArgInfo (kernel, i, CL_KERNEL_ARG_TYPE_NAME, size,
                                string_value, NULL);
      assert (err == CL_SUCCESS);
      pi->type_name = (char *)malloc (size);
      memcpy (pi->type_name, string_value, size);
      // 2 because size includes terminating NULL character
      int is_pointer = (pi->type_name[size - 2] == '*');

      err = clGetKernelArgInfo (kernel, i, CL_KERNEL_ARG_NAME, 0, NULL, &size);
      assert (err == CL_SUCCESS);
      assert (size < POCL_FILENAME_LENGTH);
      err = clGetKernelArgInfo (kernel, i, CL_KERNEL_ARG_NAME, size,
                                string_value, NULL);
      assert (err == CL_SUCCESS);
      pi->name = (char *)malloc (size + 1);
      memcpy (pi->name, string_value, size);
      pi->name[size] = 0;

      // can't get this directly through API, but we can workaround
      pi->type_size = 0;
      pi->type = POCL_ARG_TYPE_NONE;
      if (pi->access_qualifier != CL_KERNEL_ARG_ACCESS_NONE)
        {
          pi->type = POCL_ARG_TYPE_IMAGE;
          pi->type_size = sizeof (cl_mem);
        }
      if (strncmp (pi->type_name, "sampler_t", 9) == 0)
        {
          pi->type = POCL_ARG_TYPE_SAMPLER;
          pi->type_size = sizeof (cl_sampler);
        }
      if (is_pointer)
        {
          pi->type = POCL_ARG_TYPE_POINTER;
          pi->type_size = sizeof (cl_mem);
        }

      if (pi->type_size == 0)
        {
          size_t correct_size = 0;
          size_t successes = 0;
          for (size_t j = 1; j <= MAX_TESTED_ARG_SIZE; ++j)
            {
              err = clSetKernelArg (kernel, i, j, empty_buffer);
              if (err == CL_SUCCESS)
                {
                  correct_size = j;
                  successes += 1;
                }
            }
          /* Unfortunately it seems some of the crap vendor implementations
             will accept any argument size. */
          if (successes > 1)
            POCL_MSG_ERR ("More than one correct value reported"
                          " - can't figure it out\n");
          else
            pi->type_size = correct_size;
        }

      POCL_MSG_PRINT_PROXY ("KERNEL %s ARGUMENT %u NAME %s "
                            "TYPENAME %s ISPTR %u TYPE %u SIZE %u\n",
                            meta->name, i, pi->name, pi->type_name, is_pointer,
                            pi->type, pi->type_size);
    }
  return 0;
}

static void
set_build_log (cl_device_id proxy_dev, cl_program program,
               cl_program proxy_prog, unsigned device_i)
{
  assert (program->build_log[device_i] == NULL);
  size_t log_len = 0;
  int err = clGetProgramBuildInfo (proxy_prog, proxy_dev, CL_PROGRAM_BUILD_LOG,
                                   0, NULL, &log_len);
  assert (err == CL_SUCCESS);

  if (log_len > 0)
    {
      POCL_MSG_PRINT_PROXY ("build log length: %zu\n", log_len);
      char *tmp = (char *)malloc (log_len + 1);
      err = clGetProgramBuildInfo (proxy_prog, proxy_dev, CL_PROGRAM_BUILD_LOG,
                                   log_len, tmp, NULL);
      assert (err == CL_SUCCESS);
      tmp[log_len] = 0;
      program->build_log[device_i] = tmp;
      POCL_MSG_PRINT_PROXY ("build log:\n"
                            "*************************************************"
                            "*******************************\n"
                            "%s\n"
                            "*************************************************"
                            "*******************************\n",
                            tmp);
    }
  else
    program->build_log[device_i] = 0;
}

/******************************************************************************/

int
pocl_proxy_build_source (cl_program program, cl_uint device_i,
                         cl_uint num_input_headers,
                         const cl_program *input_headers,
                         const char **header_include_names, int link_program)
{
  cl_device_id device = program->devices[device_i];
  proxy_device_data_t *d = (proxy_device_data_t *)device->data;
  cl_context context = program->context->proxied_context;

  assert (program->data[device_i] == NULL);
  assert (program->id);
  int err;

  cl_program prog = NULL;
  size_t options_len
      = program->compiler_options ? strlen (program->compiler_options) : 0;
  char *options = (char *)alloca (options_len + 40);
  options[0] = 0;
  if (program->compiler_options)
    strcpy (options, program->compiler_options);
  strcat (options, " -cl-kernel-arg-info");
  POCL_MSG_PRINT_PROXY ("SOURCE BUILD: device %u ||| options %s \n", device_i,
                        options);

  assert (program->source);
  // context, num_sources, sources, source_lens, &err);
  size_t len = strlen (program->source);
  const char *source = program->source;
  prog = clCreateProgramWithSource (context, 1, &source, &len, &err);
  if (err)
    return err;

  if (link_program)
    {
      err = clBuildProgram (prog, 1, &d->device_id, options, NULL, NULL);
      set_build_log (d->device_id, program, prog, device_i);
      if (err)
        return err;
    }
  else
    {
      cl_uint i;
      cl_program *local_input_headers = NULL;
      if (num_input_headers > 0)
        {
          local_input_headers
              = (cl_program *)alloca (num_input_headers * sizeof (cl_program));
          for (i = 0; i < num_input_headers; ++i)
            local_input_headers[i] = NULL;
          for (i = 0; i < num_input_headers; ++i)
            {
              const char *p = input_headers[i]->source;
              cl_program tmp
                  = clCreateProgramWithSource (context, 1, &p, NULL, &err);
              if (err)
                goto FINISH;
              input_headers[i]->data[device_i] = local_input_headers[i] = tmp;
            }
        }
      err = clCompileProgram (
          prog, 1, &d->device_id, program->compiler_options, num_input_headers,
          local_input_headers, header_include_names, NULL, NULL);
    FINISH:
      if (err)
        {
          if (local_input_headers)
            {
              for (i = 0; i < num_input_headers; ++i)
                if (local_input_headers[i] != NULL)
                  {
                    clReleaseProgram (local_input_headers[i]);
                    input_headers[i]->data[device_i] = NULL;
                  }
            }
          return err;
        }
    }

  SHA1_CTX hash_ctx;
  pocl_SHA1_Init (&hash_ctx);

  size_t binary_size = 0;
  // some platforms are broken
  if (d->backend->supports_binaries)
    {
      err = clGetProgramInfo (prog, CL_PROGRAM_BINARY_SIZES,
                              sizeof (binary_size), &binary_size, NULL);
      assert (err == CL_SUCCESS);
      assert (binary_size > 0);

      char *binary = (char *)malloc (binary_size);
      assert (binary);
      err = clGetProgramInfo (prog, CL_PROGRAM_BINARIES, sizeof (binary),
                              &binary, NULL);
      assert (err == CL_SUCCESS);
      program->binary_sizes[device_i] = (size_t)binary_size;
      program->binaries[device_i] = (unsigned char *)binary;
      POCL_MSG_PRINT_PROXY ("BINARY SIZE [%u]: %zu \n", device_i,
                            program->binary_sizes[device_i]);

      // TODO program->binaries, program->binary_sizes set up, but caching on
      // them is wrong
      pocl_SHA1_Update (&hash_ctx, (uint8_t *)program->binaries[device_i],
                        program->binary_sizes[device_i]);
    }
  else
    {
      program->binary_sizes[device_i] = 0;
      program->binaries[device_i] = NULL;
      // TODO caching on source is unreliable, ignores includes
      pocl_SHA1_Update (&hash_ctx, (const uint8_t *)source, len);
      POCL_MSG_PRINT_PROXY ("This device does not support binaries.\n");
    }

  assert (program->build_hash[device_i][2] == 0);

  char *dev_hash = device->ops->build_hash (device);
  pocl_SHA1_Update (&hash_ctx, (const uint8_t *)dev_hash, strlen (dev_hash));
  free (dev_hash);

  uint8_t digest[SHA1_DIGEST_SIZE];
  pocl_SHA1_Final (&hash_ctx, digest);

  unsigned char *hashstr = program->build_hash[device_i];
  size_t i;
  for (i = 0; i < SHA1_DIGEST_SIZE; i++)
    {
      *hashstr++ = (digest[i] & 0x0F) + 65;
      *hashstr++ = ((digest[i] & 0xF0) >> 4) + 65;
    }
  *hashstr = 0;

  program->build_hash[device_i][2] = '/';

  if (d->backend->supports_binaries)
    {
      char program_bc_path[POCL_FILENAME_LENGTH];
      char temp_path[POCL_FILENAME_LENGTH];
      pocl_cache_create_program_cachedir (program, device_i, NULL, 0,
                                          program_bc_path);

      if (!pocl_exists (program_bc_path))
        {
          err = pocl_cache_write_generic_objfile (
              temp_path, (char *)program->binaries[device_i],
              program->binary_sizes[device_i]);
          assert (err == 0);
          pocl_rename (temp_path, program_bc_path);
        }
    }

  program->data[device_i] = (void *)prog;
  return CL_SUCCESS;
}

int
pocl_proxy_build_binary (cl_program program, cl_uint device_i,
                         int link_program, int spir_build)
{
  proxy_device_data_t *d
      = (proxy_device_data_t *)program->devices[device_i]->data;
  cl_context context = program->context->proxied_context;

  POCL_RETURN_ERROR_ON ((!d->backend->supports_binaries),
                        CL_BUILD_PROGRAM_FAILURE,
                        "This device does not support binaries.\n");

  assert (program->data[device_i] == NULL);
  assert (program->id);
  int err;

  cl_program prog = NULL;
  size_t options_len
      = program->compiler_options ? strlen (program->compiler_options) : 0;
  char *options = (char *)alloca (options_len + 40);
  options[0] = 0;
  if (program->compiler_options)
    strcpy (options, program->compiler_options);
  strcat (options, " -cl-kernel-arg-info");
  POCL_MSG_PRINT_PROXY ("BINARY BUILD: options %s \n", options);

  // TODO should binary be already loaded ?
  assert (program->pocl_binaries[device_i]);
  char program_bc_path[POCL_FILENAME_LENGTH];
  pocl_cache_program_bc_path (program_bc_path, program, device_i);

  assert (pocl_exists (program_bc_path));
  assert (program->binaries[device_i]);
  assert (program->binary_sizes[device_i] > 0);
  const unsigned char *binary = program->binaries[device_i];
  size_t size = program->binary_sizes[device_i];

  assert (link_program);

  // context, num_devices, devices, sizes, binaries, binary_statuses, &err);
  cl_int status = CL_INVALID_VALUE;
  size_t temp_size = (size_t)size;
  prog = clCreateProgramWithBinary (context, 1, &d->device_id, &temp_size,
                                    &binary, &status, &err);
  if (err)
    return err;

  err = clBuildProgram (prog, 1, &d->device_id, options, NULL, NULL);
  set_build_log (d->device_id, program, prog, device_i);
  if (err)
    return err;

  POCL_MSG_PRINT_PROXY ("Num kernels: %zu\n", program->num_kernels);

  program->data[device_i] = (void *)prog;
  return CL_SUCCESS;
}

int
pocl_proxy_link_program (cl_program program, cl_uint device_i,
                         cl_uint num_input_programs,
                         const cl_program *input_programs, int create_library)
{
  proxy_device_data_t *d
      = (proxy_device_data_t *)program->devices[device_i]->data;
  cl_context context = program->context->proxied_context;

  POCL_RETURN_ERROR_ON ((!d->backend->supports_binaries),
                        CL_BUILD_PROGRAM_FAILURE,
                        "This device does not support binaries.\n");

  assert (program->data[device_i] == NULL);
  assert (program->id);
  int err;
  cl_uint i;

  cl_program *proxy_input_programs
      = (cl_program *)alloca (num_input_programs * sizeof (cl_program));
  for (i = 0; i < num_input_programs; ++i)
    proxy_input_programs[i] = (cl_program)input_programs[i]->data[device_i];

  cl_program prog = NULL;
  size_t options_len = strlen (program->compiler_options);
  char *options = (char *)alloca (options_len + 40);
  options[0] = 0;
  strcpy (options, program->compiler_options);
  strcat (options, " -cl-kernel-arg-info");
  POCL_MSG_PRINT_PROXY ("proxy build with options %s \n", options);

  prog = clLinkProgram (context, 1, &d->device_id, program->compiler_options,
                        num_input_programs, proxy_input_programs, NULL, NULL,
                        &err);
  if (err)
    return err;

  if (!create_library)
    {
      err = clBuildProgram (prog, 1, &d->device_id, options, NULL, NULL);
      set_build_log (d->device_id, program, prog, device_i);
      if (err)
        return err;
    }

  POCL_MSG_PRINT_PROXY ("Num kernels: %zu\n", program->num_kernels);

  program->data[device_i] = (void *)prog;
  return CL_SUCCESS;
}

// TODO free also binaries ?
int
pocl_proxy_free_program (cl_device_id device, cl_program program,
                         unsigned program_device_i)
{
  proxy_device_data_t *d = (proxy_device_data_t *)device->data;

  // this can happen if the build fails
  if (program->data == NULL)
    return CL_SUCCESS;

  if (program->data[program_device_i] == NULL)
    return CL_SUCCESS;

  cl_program proxy_prog = (cl_program)program->data[program_device_i];
  int err = clReleaseProgram (proxy_prog);
  program->data[program_device_i] = NULL;

  return err;
}

int
pocl_proxy_setup_metadata (cl_device_id device, cl_program program,
                           unsigned program_device_i)
{
  cl_uint num_kernels = 0;

  proxy_device_data_t *d = (proxy_device_data_t *)device->data;
  cl_program proxy_prog = (cl_program)program->data[program_device_i];

  if (d->backend->provides_metadata)
    {
      assert (program->kernel_meta == NULL);
      POCL_MSG_PRINT_PROXY ("Setting up Kernel metadata\n");

      int err = clCreateKernelsInProgram (proxy_prog, 0, NULL, &num_kernels);
      if (err)
        {
          POCL_MSG_ERR ("proxy metadata setup error 1: %i", err);
          return err;
        }

      if (num_kernels > 0)
        {
          pocl_kernel_metadata_t *p = (pocl_kernel_metadata_t *)calloc (
              num_kernels, sizeof (pocl_kernel_metadata_t));
          cl_kernel *kernels
              = (cl_kernel *)alloca (num_kernels * sizeof (cl_kernel));
          assert (p);
          err = clCreateKernelsInProgram (proxy_prog, num_kernels, kernels,
                                          NULL);
          if (err)
            {
              POCL_MSG_ERR ("proxy metadata setup error 2: %i", err);
              return err;
            }
          cl_uint i;
          for (i = 0; i < num_kernels; ++i)
            {
              get_kernel_metadata (p + i, program->num_devices, proxy_prog,
                                   d->device_id, kernels[i]);
              err = clReleaseKernel (kernels[i]);
              assert (err == CL_SUCCESS);
            }

          program->kernel_meta = p;
        }
      else
        {
          POCL_MSG_WARN ("Program has zero kernels.\n");
          program->kernel_meta = NULL;
        }

      program->num_kernels = num_kernels;
      POCL_MSG_PRINT_PROXY ("Num kernels: %zu\n", program->num_kernels);
      return 1;
    }
  else
    return 0;
}

int
pocl_proxy_create_kernel (cl_device_id device, cl_program program,
                          cl_kernel kernel, unsigned device_i)
{
  assert (program->data[device_i] != NULL);
  cl_program proxy_prog = (cl_program)program->data[device_i];
  assert (proxy_prog);

  assert (kernel->data != NULL);
  assert (kernel->data[device_i] == NULL);

  int err = 0;
  cl_kernel proxy_ker = clCreateKernel (proxy_prog, kernel->name, &err);

  if (err == CL_SUCCESS)
    kernel->data[device_i] = (void *)proxy_ker;

  return err;
}

int
pocl_proxy_free_kernel (cl_device_id device, cl_program program,
                        cl_kernel kernel, unsigned device_i)
{
  assert (kernel->data != NULL);

  // may happen if creating kernel fails
  if (kernel->data[device_i] == NULL)
    return CL_SUCCESS;

  cl_kernel proxy_ker = (cl_kernel)kernel->data[device_i];

  int err = clReleaseKernel (proxy_ker);

  kernel->data[device_i] = NULL;

  return err;
}

int
pocl_proxy_init_queue (cl_device_id device, cl_command_queue queue)
{
  assert (queue->data == NULL);

  proxy_device_data_t *d = (proxy_device_data_t *)device->data;
  cl_context proxy_context = queue->context->proxied_context;

  proxy_queue_data_t *qd = (proxy_queue_data_t *)pocl_aligned_malloc (
      HOST_CPU_CACHELINE_SIZE, sizeof (proxy_queue_data_t));
  if (qd == NULL)
    goto ERROR;
  queue->data = qd;

  int err = CL_SUCCESS;
  cl_command_queue cq
      = clCreateCommandQueue (proxy_context, d->device_id,
                              (queue->properties & (~CL_QUEUE_HIDDEN)), &err);
  if (err != CL_SUCCESS)
    {
      goto ERROR;
    }

  POCL_INIT_COND (qd->wakeup_cond);
  POCL_INIT_COND (qd->wait_cond);
  POCL_INIT_LOCK (qd->wq_lock);
  qd->work_queue = NULL;

  qd->proxied_id = cq;
  qd->queue = queue;
  unsigned context_device_i = 0;
  for (context_device_i = 0; context_device_i < queue->context->num_devices;
       ++context_device_i)
    {
      if (queue->context->devices[context_device_i] == device)
        break;
    }
  assert (context_device_i < queue->context->num_devices);
  qd->context_device_i = context_device_i;

  qd->cq_thread_exit_requested = 0;
  POCL_CREATE_THREAD (qd->cq_thread_id, pocl_proxy_queue_pthread, qd);

  return CL_SUCCESS;

ERROR:
  // TODO
  pocl_aligned_free (queue->data);
  queue->data = NULL;
  return err;
}

int
pocl_proxy_free_queue (cl_device_id device, cl_command_queue queue)
{
  proxy_queue_data_t *qd = (proxy_queue_data_t *)queue->data;

  if (queue->data == NULL)
    return CL_SUCCESS;

  POCL_FAST_LOCK (qd->wq_lock);
  qd->cq_thread_exit_requested = 1;
  POCL_SIGNAL_COND (qd->wakeup_cond);
  POCL_FAST_UNLOCK (qd->wq_lock);

  if (pthread_self() != qd->cq_thread_id)
    POCL_JOIN_THREAD (qd->cq_thread_id);
  qd->cq_thread_id = 0;

  cl_command_queue cq = (cl_command_queue)qd->proxied_id;
  int err = clReleaseCommandQueue (cq);

  qd->work_queue = NULL;
  POCL_DESTROY_COND (qd->wakeup_cond);
  POCL_DESTROY_COND (qd->wait_cond);
  POCL_DESTROY_LOCK (qd->wq_lock);
  return err;
}

int
pocl_proxy_create_sampler (cl_device_id device, cl_sampler samp,
                           unsigned device_i)
{
  proxy_device_data_t *d = (proxy_device_data_t *)device->data;
  cl_context context = samp->context->proxied_context;
  uint32_t samp_id = (uint32_t)samp->id;
  assert (samp_id);
  assert (samp->device_data[device_i] == NULL);
  int err = 0;

  cl_sampler s
      = clCreateSampler (context, samp->normalized_coords,
                         samp->addressing_mode, samp->filter_mode, &err);
  if (err == CL_SUCCESS)
    samp->device_data[device_i] = (void *)s;

  return err;
}

int
pocl_proxy_free_sampler (cl_device_id device, cl_sampler samp,
                         unsigned device_i)
{
  uint32_t samp_id = (uint32_t)samp->id;
  assert (samp_id);
  assert (samp->device_data[device_i] != NULL);

  cl_sampler s = (cl_sampler)samp->device_data[device_i];
  int err = clReleaseSampler (s);
  samp->device_data[device_i] = NULL;
  return err;
}

/*****************************************************************************/
/*****************************************************************************/

static void
proxy_push_command (_cl_command_node *node)
{
  cl_command_queue cq = node->event->queue;
  proxy_queue_data_t *qd = (proxy_queue_data_t *)cq->data;

  POCL_FAST_LOCK (qd->wq_lock);
  DL_APPEND (qd->work_queue, node);
  POCL_SIGNAL_COND (qd->wakeup_cond);
  POCL_FAST_UNLOCK (qd->wq_lock);
}

void
pocl_proxy_submit (_cl_command_node *node, cl_command_queue cq)
{
  cl_event e = node->event;
  assert (e->data == NULL);

  pocl_proxy_event_data_t *e_d = NULL;
  e_d = calloc (1, sizeof (pocl_proxy_event_data_t));
  assert (e_d);

  POCL_INIT_COND (e_d->event_cond);
  e->data = (void *)e_d;

  node->ready = 1;
  if (pocl_command_is_ready (e))
    {
      pocl_update_event_submitted (e);
      proxy_push_command (node);
    }
  POCL_UNLOCK_OBJ (e);
  return;
}

void
pocl_proxy_notify_cmdq_finished (cl_command_queue cq)
{
  /* must be called with CQ already locked.
   * this must be a broadcast since there could be multiple
   * user threads waiting on the same command queue */
  proxy_queue_data_t *dd = (proxy_queue_data_t *)cq->data;
  POCL_BROADCAST_COND (dd->wait_cond);
}

void
pocl_proxy_join (cl_device_id device, cl_command_queue cq)
{
  POCL_LOCK_OBJ (cq);
  proxy_queue_data_t *dd = (proxy_queue_data_t *)cq->data;

  while (1)
    {
      if (cq->command_count == 0)
        {
          POCL_UNLOCK_OBJ (cq);
          return;
        }
      else
        {
          POCL_WAIT_COND (dd->wait_cond, cq->pocl_lock);
        }
    }
}

void
pocl_proxy_flush (cl_device_id device, cl_command_queue cq)
{
}

void
pocl_proxy_notify (cl_device_id device, cl_event event, cl_event finished)
{
  _cl_command_node *node = event->command;

  if (finished->status < CL_COMPLETE)
    {
      pocl_update_event_failed (event);
      return;
    }

  if (!node->ready)
    return;

  if (pocl_command_is_ready (node->event))
    {
      assert (event->status == CL_QUEUED);
      pocl_update_event_submitted (event);
      proxy_push_command (node);
    }

  return;

  POCL_MSG_PRINT_PROXY ("notify on event %zu \n", event->id);
}

void
pocl_proxy_wait_event (cl_device_id device, cl_event event)
{
  POCL_MSG_PRINT_PROXY ("device->wait_event on event %zu\n", event->id);
  pocl_proxy_event_data_t *e_d = (pocl_proxy_event_data_t *)event->data;

  POCL_LOCK_OBJ (event);
  while (event->status > CL_COMPLETE)
    {
      POCL_WAIT_COND (e_d->event_cond, event->pocl_lock);
    }
  POCL_UNLOCK_OBJ (event);

  POCL_MSG_PRINT_INFO ("event wait finished with status: %i\n", event->status);
  assert (event->status <= CL_COMPLETE);
}

void
pocl_proxy_free_event_data (cl_event event)
{
  assert (event->data != NULL);
  pocl_proxy_event_data_t *e_d = (pocl_proxy_event_data_t *)event->data;
  POCL_DESTROY_COND (e_d->event_cond);
  POCL_MEM_FREE (event->data);
}

void
pocl_proxy_notify_event_finished (cl_event event)
{
  pocl_proxy_event_data_t *e_d = (pocl_proxy_event_data_t *)event->data;
  POCL_BROADCAST_COND (e_d->event_cond);
}

/*****************************************************************************/

#define ENQUEUE(code)                                                         \
  int res = code;                                                             \
  assert (res == CL_SUCCESS);                                                 \
  clFinish (cq);

#if defined(ENABLE_OPENGL_INTEROP) || defined(ENABLE_EGL_INTEROP)
static void
pocl_proxy_enque_acquire_gl (void *data, cl_command_queue cq,
                             _cl_command_node *node, unsigned global_mem_id,
                             size_t num_objs, cl_mem *objs)
{
  cl_mem *proxy_objs = (cl_mem *)alloca (num_objs * sizeof (cl_mem));
  size_t i;
  for (i = 0; i < num_objs; ++i)
    proxy_objs[i] = (cl_mem)objs[i]->device_ptrs[global_mem_id].mem_ptr;

#ifdef ENABLE_EGL_INTEROP
  ENQUEUE (
      clEnqueueAcquireEGLObjectsKHR (cq, num_objs, proxy_objs, 0, NULL, NULL));
#else
  ENQUEUE (
      clEnqueueAcquireGLObjects (cq, num_objs, proxy_objs, 0, NULL, NULL));
#endif
}

static void
pocl_proxy_enque_release_gl (void *data, cl_command_queue cq,
                             _cl_command_node *node, unsigned global_mem_id,
                             size_t num_objs, cl_mem *objs)
{
  cl_mem *proxy_objs = (cl_mem *)alloca (num_objs * sizeof (cl_mem));
  size_t i;
  for (i = 0; i < num_objs; ++i)
    proxy_objs[i] = (cl_mem)objs[i]->device_ptrs[global_mem_id].mem_ptr;

#ifdef ENABLE_EGL_INTEROP
  ENQUEUE (
      clEnqueueReleaseEGLObjectsKHR (cq, num_objs, proxy_objs, 0, NULL, NULL));
#else
  ENQUEUE (
      clEnqueueReleaseGLObjects (cq, num_objs, proxy_objs, 0, NULL, NULL));
#endif
}
#endif

static void
pocl_proxy_enque_read (void *data, cl_command_queue cq, _cl_command_node *node,
                       void *__restrict__ host_ptr,
                       pocl_mem_identifier *src_mem_id, cl_mem unused,
                       size_t offset, size_t size)
{
  cl_mem mem = (cl_mem)src_mem_id->mem_ptr;

  ENQUEUE (clEnqueueReadBuffer (cq, mem, CL_FALSE, offset, size, host_ptr, 0,
                                NULL, NULL));
}

static void
pocl_proxy_enque_write (void *data, cl_command_queue cq,
                        _cl_command_node *node,
                        const void *__restrict__ host_ptr,
                        pocl_mem_identifier *dst_mem_id, cl_mem unused,
                        size_t offset, size_t size)
{
  cl_mem mem = (cl_mem)dst_mem_id->mem_ptr;

  ENQUEUE (clEnqueueWriteBuffer (cq, mem, CL_FALSE, offset, size, host_ptr, 0,
                                 NULL, NULL));
}

static int
pocl_proxy_enque_copy (void *data, cl_command_queue cq, _cl_command_node *node,
                       pocl_mem_identifier *dst_mem_id, cl_mem unused1,
                       pocl_mem_identifier *src_mem_id, cl_mem unused2,
                       size_t dst_offset, size_t src_offset, size_t size)
{
  cl_mem src = (cl_mem)src_mem_id->mem_ptr;
  cl_mem dst = (cl_mem)dst_mem_id->mem_ptr;

  if ((src == dst) && (src_offset == dst_offset))
    {
      return 1;
    }

  ENQUEUE (clEnqueueCopyBuffer (cq, src, dst, src_offset, dst_offset, size, 0,
                                NULL, NULL));
  return 0;
}

static void
pocl_proxy_enque_copy_rect (void *data, cl_command_queue cq,
                            _cl_command_node *node,
                            pocl_mem_identifier *dst_mem_id, cl_mem unused1,
                            pocl_mem_identifier *src_mem_id, cl_mem unused2,
                            const size_t *__restrict__ const dst_origin,
                            const size_t *__restrict__ const src_origin,
                            const size_t *__restrict__ const region,
                            size_t const dst_row_pitch,
                            size_t const dst_slice_pitch,
                            size_t const src_row_pitch,
                            size_t const src_slice_pitch)
{
  cl_mem src = (cl_mem)src_mem_id->mem_ptr;
  cl_mem dst = (cl_mem)dst_mem_id->mem_ptr;

  /*
    POCL_MSG_PRINT_PROXY ("ASYNC COPY: \nregion %zu %zu %zu\n"
                  "  src_origin %zu %zu %zu\n"
                  "  dst_origin %zu %zu %zu\n"
                  "  dst_row_pitch %zu, dst_slice_pitch %zu\n"
                  "  src_row_pitch %zu, src_slice_pitch %zu\n",

                  region[0], region[1], region[2],
                  src_origin[0], src_origin[1], src_origin[2],
                  dst_origin[0], dst_origin[1], dst_origin[2],
                  dst_row_pitch, dst_slice_pitch,
                  src_row_pitch, src_slice_pitch
                  );
  */

  ENQUEUE (clEnqueueCopyBufferRect (
      cq, src, dst, src_origin, dst_origin, region, src_row_pitch,
      src_slice_pitch, dst_row_pitch, dst_slice_pitch, 0, NULL, NULL));
}

static void
pocl_proxy_enque_write_rect (
    void *data, cl_command_queue cq, _cl_command_node *node,
    const void *__restrict__ const host_ptr, pocl_mem_identifier *dst_mem_id,
    cl_mem unused, const size_t *__restrict__ const buffer_origin,
    const size_t *__restrict__ const host_origin,
    const size_t *__restrict__ const region, size_t const buffer_row_pitch,
    size_t const buffer_slice_pitch, size_t const host_row_pitch,
    size_t const host_slice_pitch)
{
  cl_mem mem = (cl_mem)dst_mem_id->mem_ptr;

  /*
    POCL_MSG_PRINT_PROXY ("ASYNC WRITE: \nregion %zu %zu %zu\n"
                  "  buffer_origin %zu %zu %zu\n"
                  "  host_origin %zu %zu %zu\n"
                  "  offset %zu total_size %zu\n"
                  "  buffer_row_pitch %zu, buffer_slice_pitch %zu\n"
                  "  host_row_pitch %zu, host_slice_pitch %zu\n",
                  region[0], region[1], region[2],
                  buffer_origin[0], buffer_origin[1], buffer_origin[2],
                  host_origin[0], host_origin[1], host_origin[2],
                  (size_t)(adjusted_host_ptr - (const char*)host_ptr),
    total_size, buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
    host_slice_pitch
                  );
  */

  ENQUEUE (clEnqueueWriteBufferRect (
      cq, mem, CL_FALSE, buffer_origin, host_origin, region, buffer_row_pitch,
      buffer_slice_pitch, host_row_pitch, host_slice_pitch, host_ptr, 0, NULL,
      NULL));
}

static void
pocl_proxy_enque_read_rect (
    void *data, cl_command_queue cq, _cl_command_node *node,
    void *__restrict__ const host_ptr, pocl_mem_identifier *src_mem_id,
    cl_mem unused, const size_t *__restrict__ const buffer_origin,
    const size_t *__restrict__ const host_origin,
    const size_t *__restrict__ const region, size_t const buffer_row_pitch,
    size_t const buffer_slice_pitch, size_t const host_row_pitch,
    size_t const host_slice_pitch)
{
  cl_mem mem = (cl_mem)src_mem_id->mem_ptr;

  ENQUEUE (clEnqueueReadBufferRect (
      cq, mem, CL_FALSE, buffer_origin, host_origin, region, buffer_row_pitch,
      buffer_slice_pitch, host_row_pitch, host_slice_pitch, host_ptr, 0, NULL,
      NULL));
  /*
    POCL_MSG_PRINT_PROXY ("ASYNC READ: \nregion %zu %zu %zu\n"
                  "  buffer_origin %zu %zu %zu\n"
                  "  host_origin %zu %zu %zu\n"
                  "  offset %zu total_size %zu\n"
                  "  buffer_row_pitch %zu, buffer_slice_pitch %zu\n"
                  "  host_row_pitch %zu, host_slice_pitch %zu\n",
                  region[0], region[1], region[2],
                  buffer_origin[0], buffer_origin[1], buffer_origin[2],
                  host_origin[0], host_origin[1], host_origin[2],
                  (size_t)(adjusted_host_ptr - (char *)host_ptr), total_size,
                  buffer_row_pitch, buffer_slice_pitch,
                  host_row_pitch, host_slice_pitch
                  );
  */
}

static void
pocl_proxy_enque_memfill (void *data, cl_command_queue cq,
                          _cl_command_node *node,
                          pocl_mem_identifier *dst_mem_id, cl_mem unused,
                          size_t size, size_t offset,
                          const void *__restrict__ pattern,
                          size_t pattern_size)
{
  cl_mem mem = (cl_mem)dst_mem_id->mem_ptr;

  ENQUEUE (clEnqueueFillBuffer (cq, mem, pattern, pattern_size, offset, size,
                                0, NULL, NULL));
}

static int
pocl_proxy_enque_map_mem (void *data, cl_command_queue cq,
                          _cl_command_node *node,
                          pocl_mem_identifier *src_mem_id, cl_mem unused,
                          mem_mapping_t *map)
{

  cl_mem mem = (cl_mem)src_mem_id->mem_ptr;
  void *host_ptr = map->host_ptr;
  assert (host_ptr != NULL);
  size_t offset = map->offset;
  size_t size = map->size;
  /*
    POCL_MSG_PRINT_MEMORY ("PROXY: MAP memcpy() "
                           "src_id %p + offset %zu"
                           "to dst_host_ptr %p\n",
                           src_mem_id, offset, host_ptr);
  */
  ENQUEUE (clEnqueueReadBuffer (cq, mem, CL_FALSE, offset, size, host_ptr, 0,
                                NULL, NULL));

  return 0;
}

static int
pocl_proxy_enque_unmap_mem (void *data, cl_command_queue cq,
                            _cl_command_node *node,
                            pocl_mem_identifier *dst_mem_id, cl_mem unused,
                            mem_mapping_t *map)
{

  cl_mem dst = (cl_mem)dst_mem_id->mem_ptr;
  void *host_ptr = map->host_ptr;
  assert (host_ptr != NULL);
  size_t offset = map->offset;
  size_t size = map->size;

  /*
    POCL_MSG_PRINT_MEMORY ("PROXY: UNMAP memcpy() "
                         "host_ptr %p to mem_id %lu + offset %zu\n",
                         host_ptr, mem_id, offset);
  */

  /* equality test, because it could be CL_MAP_READ |
   * CL_MAP_WRITE(..invalidate) which has to be handled like a write */
  if (map->map_flags == CL_MAP_READ)
    return 1;
  else
    {
      ENQUEUE (clEnqueueWriteBuffer (cq, dst, CL_FALSE, offset, size, host_ptr,
                                     0, NULL, NULL));
    }
  return 0;
}

static void
pocl_proxy_enque_run (cl_device_id pocl_device, void *data, unsigned device_i,
                      cl_command_queue cq, _cl_command_node *node)
{
  struct pocl_argument *al = NULL;
  unsigned i;
  int err;
  cl_kernel pocl_kernel = node->command.run.kernel;
  assert (pocl_device == node->device);
  unsigned program_i = node->program_device_i;

  struct pocl_context *pc = &node->command.run.pc;

  if (pc->num_groups[0] == 0 || pc->num_groups[1] == 0 || pc->num_groups[2] == 0)
    return;

  pocl_kernel_metadata_t *kernel_md = pocl_kernel->meta;

  cl_kernel kernel = (cl_kernel)pocl_kernel->data[program_i];

  /* Process the kernel arguments. Find out what needs to be updated. */
  for (i = 0; i < kernel_md->num_args; ++i)
    {
      al = &(node->command.run.arguments[i]);
      assert (al->is_set > 0);
      if (ARG_IS_LOCAL (kernel_md->arg_info[i]))
        {
          err = clSetKernelArg (kernel, i, al->size, NULL);
          assert (err == CL_SUCCESS);
        }
      else if ((kernel_md->arg_info[i].type == POCL_ARG_TYPE_POINTER)
               || (kernel_md->arg_info[i].type == POCL_ARG_TYPE_IMAGE))
        {
          if (al->value)
            {
              cl_mem pocl_mem = (*(cl_mem *)(al->value));
              cl_mem mem
                  = (cl_mem)pocl_mem->device_ptrs[pocl_device->global_mem_id]
                        .mem_ptr;
              err = clSetKernelArg (kernel, i, sizeof (cl_mem), &mem);
              assert (err == CL_SUCCESS);
            }
          else
            {
              POCL_MSG_WARN ("NULL PTR ARG DETECTED: %s / ARG %i: %s \n",
                             kernel_md->name, i, kernel_md->arg_info[i].name);
              err = clSetKernelArg (kernel, i, sizeof (cl_mem), NULL);
              assert (err == CL_SUCCESS);
            }
        }
      else if (kernel_md->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          cl_sampler pocl_sampler = *(cl_sampler *)(al->value);
          cl_sampler samp = (cl_sampler)pocl_sampler->device_data[device_i];
          err = clSetKernelArg (kernel, i, sizeof (cl_sampler), &samp);
          assert (err == CL_SUCCESS);
        }
      else
        {
          assert (kernel_md->arg_info[i].type == POCL_ARG_TYPE_NONE);
          err = clSetKernelArg (kernel, i, al->size, al->value);
          assert (err == CL_SUCCESS);
        }
    }

  size_t local[] = { pc->local_size[0],
                     pc->local_size[1],
                     pc->local_size[2] };

  size_t *ptr = pc->num_groups;
  size_t global[]
      = { ptr[0] * local[0], ptr[1] * local[1], ptr[2] * local[2] };

  size_t offset[] = { pc->global_offset[0],
                      pc->global_offset[1],
                      pc->global_offset[2] };

  ENQUEUE (clEnqueueNDRangeKernel (cq, kernel, pc->work_dim,
                                   offset, global, local, 0, NULL, NULL));
}

static cl_int
pocl_proxy_enque_copy_image_rect (
    void *data, cl_command_queue cq, _cl_command_node *node, cl_mem src_image,
    cl_mem dst_image, pocl_mem_identifier *src_mem_id,
    pocl_mem_identifier *dst_mem_id, const size_t *src_origin,
    const size_t *dst_origin, const size_t *region)
{
  cl_mem src_img = (cl_mem)src_mem_id->mem_ptr;
  cl_mem dst_img = (cl_mem)dst_mem_id->mem_ptr;
  /*
    POCL_MSG_PRINT_PROXY (
        "PROXY COPY IMAGE RECT \n"
        "dst_image %p PROXIED id %u \n"
        "src_image %p PROXIED id %u \n"
        "dst_origin [0,1,2] %zu %zu %zu \n"
        "src_origin [0,1,2] %zu %zu %zu \n"
        "region [0,1,2] %zu %zu %zu \n",
        dst_image, dst_image_id,
        src_image, src_image_id,
        dst_origin[0], dst_origin[1], dst_origin[2],
        src_origin[0], src_origin[1], src_origin[2],
        region[0], region[1], region[2]);
  */

  ENQUEUE (clEnqueueCopyImage (cq, src_img, dst_img, src_origin, dst_origin,
                               region, 0, NULL, NULL));
  return 0;
}

static cl_int
pocl_proxy_enque_write_image_rect (void *data, cl_command_queue cq,
                                   _cl_command_node *node, cl_mem unused,
                                   pocl_mem_identifier *dst_mem_id,
                                   const void *__restrict__ src_host_ptr,
                                   pocl_mem_identifier *src_mem_id,
                                   const size_t *origin, const size_t *region,
                                   size_t src_row_pitch,
                                   size_t src_slice_pitch, size_t src_offset)
{
  int r;
  cl_mem dst_img = (cl_mem)dst_mem_id->mem_ptr;
  /*
    POCL_MSG_PRINT_PROXY (
        "PROXY WRITE IMAGE RECT \n"
        "dst_image %p PROXY id %u \n"
        "src_hostptr %p \n"
        "origin [0,1,2] %zu %zu %zu \n"
        "region [0,1,2] %zu %zu %zu \n"
        "row %zu slice %zu offset %zu \n",
        dst_image, dst_id,
        src_host_ptr,
        origin[0], origin[1], origin[2],
        region[0], region[1], region[2],
        src_row_pitch, src_slice_pitch, src_offset);
  */
  /* copies a region from host OR device buffer to device image.
   * clEnqueueCopyBufferToImage: src_mem_id = buffer,
   *     src_host_ptr = NULL, src_row_pitch = src_slice_pitch = 0
   * clEnqueueWriteImage: src_mem_id = NULL,
   *     src_host_ptr = host pointer, src_offset = 0 */

  if (src_host_ptr == NULL)
    {
      assert (src_mem_id);
      cl_mem src = (cl_mem)src_mem_id->mem_ptr;
      ENQUEUE (clEnqueueCopyBufferToImage (cq, src, dst_img, src_offset,
                                           origin, region, 0, NULL, NULL));
    }
  else
    {
      assert (src_mem_id == NULL);
      ENQUEUE (clEnqueueWriteImage (cq, dst_img, CL_FALSE, origin, region,
                                    src_row_pitch, src_slice_pitch,
                                    src_host_ptr, 0, NULL, NULL));
    }

  return 0;
}

static cl_int
pocl_proxy_enque_read_image_rect (void *data, cl_command_queue cq,
                                  _cl_command_node *node, cl_mem unused,
                                  pocl_mem_identifier *src_mem_id,
                                  void *__restrict__ dst_host_ptr,
                                  pocl_mem_identifier *dst_mem_id,
                                  const size_t *origin, const size_t *region,
                                  size_t dst_row_pitch, size_t dst_slice_pitch,
                                  size_t dst_offset)
{
  int r;
  cl_mem src_img = (cl_mem)src_mem_id->mem_ptr;
  /*
    POCL_MSG_PRINT_PROXY (
        "PROXY READ IMAGE RECT \n"
        "src_image %p PROXY id %u \n"
        "dst_hostptr %p \n"
        "origin [0,1,2] %zu %zu %zu \n"
        "region [0,1,2] %zu %zu %zu \n"
        "row %zu slice %zu offset %zu \n",
        src_image, src_id,
        dst_host_ptr,
        origin[0], origin[1], origin[2],
        region[0], region[1], region[2],
        dst_row_pitch, dst_slice_pitch, dst_offset);
  */
  /* copies a region from device image to host or device buffer
   * clEnqueueCopyImageToBuffer: dst_mem_id = buffer,
   *     dst_host_ptr = NULL, dst_row_pitch = dst_slice_pitch = 0
   * clEnqueueReadImage: dst_mem_id = NULL,
   *     dst_host_ptr = host pointer, dst_offset = 0
   */

  if (dst_host_ptr == NULL)
    {
      assert (dst_mem_id);
      cl_mem dst = (cl_mem)dst_mem_id->mem_ptr;
      ENQUEUE (clEnqueueCopyImageToBuffer (cq, src_img, dst, origin, region,
                                           dst_offset, 0, NULL, NULL));
    }
  else
    {
      assert (dst_mem_id == NULL);
      ENQUEUE (clEnqueueReadImage (cq, src_img, CL_FALSE, origin, region,
                                   dst_row_pitch, dst_slice_pitch,
                                   dst_host_ptr, 0, NULL, NULL));
    }

  return 0;
}

static cl_int
pocl_proxy_enque_map_image (void *data, cl_command_queue cq,
                            _cl_command_node *node,
                            pocl_mem_identifier *src_mem_id, cl_mem unused,
                            mem_mapping_t *map)
{
  if (map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION)
    {
      return 1;
    }

  cl_mem mem = (cl_mem)src_mem_id->mem_ptr;
  void *host_ptr = map->host_ptr;
  assert (host_ptr != NULL);

  ENQUEUE (clEnqueueReadImage (cq, mem, CL_FALSE, map->origin, map->region,
                               map->row_pitch, map->slice_pitch, map->host_ptr,
                               0, NULL, NULL));
  return 0;
}

static cl_int
pocl_proxy_enque_unmap_image (void *data, cl_command_queue cq,
                              _cl_command_node *node,
                              pocl_mem_identifier *dst_mem_id, cl_mem unused,
                              mem_mapping_t *map)
{
  if (map->map_flags == CL_MAP_READ)
    {
      return 1;
    }

  cl_mem mem = (cl_mem)dst_mem_id->mem_ptr;
  void *host_ptr = map->host_ptr;
  assert (host_ptr != NULL);

  ENQUEUE (clEnqueueWriteImage (cq, mem, CL_FALSE, map->origin, map->region,
                                map->row_pitch, map->slice_pitch,
                                map->host_ptr, 0, NULL, NULL));
  return 0;
}

static cl_int
pocl_proxy_enque_fill_image (void *data, cl_command_queue cq,
                             _cl_command_node *node, cl_mem unused,
                             pocl_mem_identifier *image_data,
                             const size_t *origin, const size_t *region,
                             cl_uint4 *fill_pixel)
{
  cl_mem mem = (cl_mem)image_data->mem_ptr;
  /*
    POCL_MSG_PRINT_PROXY ("PROXY FILL IMAGE \n"
                            "image ID %u data %p \n"
                            "origin [0,1,2] %zu %zu %zu \n"
                            "region [0,1,2] %zu %zu %zu \n",
                            image_id, image_data,
                            origin[0], origin[1], origin[2],
                            region[0], region[1], region[2]);
  */

  ENQUEUE (
      clEnqueueFillImage (cq, mem, fill_pixel, origin, region, 0, NULL, NULL));
  return 0;
}

/***********************************************************************************/

int
pocl_proxy_can_migrate_d2d (cl_device_id dest, cl_device_id source)
{
  /*
    proxy_device_data_t *src_d = (proxy_device_data_t *)(source->data);
    proxy_device_data_t *dst_d = (proxy_device_data_t *)(dest->data);
    return ((strncmp (dest->ops->device_name, proxy_device_name, 7) == 0)
            && (strncmp (source->ops->device_name, proxy_device_name, 7) == 0)
            && (src_d->platform_id == dst_d->platform_id));
  */
  return 0;
}

static void
pocl_proxy_enque_migrate_d2d (void *dest_data, void *source_data,
                              cl_command_queue cq, _cl_command_node *node,
                              cl_mem mem, pocl_mem_identifier *p)
{
  cl_mem actual_mem = (cl_mem)p->mem_ptr;

  POCL_MSG_PRINT_PROXY ("internal migrate D2D called\n");

  cl_mem_migration_flags flags = 0;
  ENQUEUE (
      clEnqueueMigrateMemObjects (cq, 1, &actual_mem, flags, 0, NULL, NULL));
}

/*****************************************************************************/

static void
proxy_exec_command (_cl_command_node *node, cl_device_id dev,
                    proxy_device_data_t *d, proxy_queue_data_t *qd)
{
  _cl_command_t *cmd = &node->command;
  cl_event event = node->event;
  const char *cstr = NULL;
  cl_command_queue cq_id = qd->proxied_id;
  unsigned context_device_i = qd->context_device_i;

  pocl_update_event_running (event);

  switch (node->type)
    {
    case CL_COMMAND_MIGRATE_MEM_OBJECTS:
      switch (cmd->migrate.type)
        {
        case ENQUEUE_MIGRATE_TYPE_D2H:
          {
            POCL_MSG_PRINT_PROXY ("export D2H, device %s\n", dev->long_name);

            cl_mem m = event->mem_objs[0];

            if (m->is_image)
              {
                size_t region[3]
                    = { m->image_width, m->image_height, m->image_depth };
                if (region[2] == 0)
                  region[2] = 1;
                if (region[1] == 0)
                  region[1] = 1;
                size_t origin[3] = { 0, 0, 0 };

                pocl_proxy_enque_read_image_rect (
                    d, cq_id, node, m, cmd->migrate.mem_id, m->mem_host_ptr,
                    NULL, origin, region, 0, 0, 0);
              }
            else
              {
                pocl_proxy_enque_read (d, cq_id, node, m->mem_host_ptr,
                                       cmd->migrate.mem_id, m, 0, m->size);
              }
            break;
          }
        case ENQUEUE_MIGRATE_TYPE_H2D:
          {
            POCL_MSG_PRINT_PROXY ("import H2D, device %s\n", dev->long_name);

            cl_mem m = event->mem_objs[0];

            if (m->is_image)
              {
                size_t region[3]
                    = { m->image_width, m->image_height, m->image_depth };
                if (region[2] == 0)
                  region[2] = 1;
                if (region[1] == 0)
                  region[1] = 1;
                size_t origin[3] = { 0, 0, 0 };

                pocl_proxy_enque_write_image_rect (
                    d, cq_id, node, m, cmd->migrate.mem_id, m->mem_host_ptr,
                    NULL, origin, region, 0, 0, 0);
              }
            else
              {
                pocl_proxy_enque_write (d, cq_id, node, m->mem_host_ptr,
                                        cmd->migrate.mem_id, m, 0, m->size);
              }
            break;
          }
        case ENQUEUE_MIGRATE_TYPE_D2D:
          {
            cl_device_id dev = cmd->migrate.src_device;
            assert (dev);
            pocl_proxy_enque_migrate_d2d (d, dev->data, cq_id, node,
                                          event->mem_objs[0],
                                          cmd->migrate.mem_id);
            break;
          }
        case ENQUEUE_MIGRATE_TYPE_NOP:
          {
          }
        }
      goto FINISH_COMMAND;

#if defined(ENABLE_OPENGL_INTEROP) || defined(ENABLE_EGL_INTEROP)
    case CL_COMMAND_ACQUIRE_GL_OBJECTS:
    case CL_COMMAND_ACQUIRE_EGL_OBJECTS_KHR:
      pocl_proxy_enque_acquire_gl (d, cq_id, node, dev->global_mem_id,
                                   event->num_buffers, event->mem_objs);
      goto FINISH_COMMAND;

    case CL_COMMAND_RELEASE_GL_OBJECTS:
    case CL_COMMAND_RELEASE_EGL_OBJECTS_KHR:
      pocl_proxy_enque_release_gl (d, cq_id, node, dev->global_mem_id,
                                   event->num_buffers, event->mem_objs);
      goto FINISH_COMMAND;
#endif

    case CL_COMMAND_READ_BUFFER:
      pocl_proxy_enque_read (d, cq_id, node, cmd->read.dst_host_ptr,
                             cmd->read.src_mem_id, event->mem_objs[0],
                             cmd->read.offset, cmd->read.size);
      goto FINISH_COMMAND;

    case CL_COMMAND_WRITE_BUFFER:
      pocl_proxy_enque_write (d, cq_id, node, cmd->write.src_host_ptr,
                              cmd->write.dst_mem_id, event->mem_objs[0],
                              cmd->write.offset, cmd->write.size);
      goto FINISH_COMMAND;

    case CL_COMMAND_COPY_BUFFER:
      pocl_proxy_enque_copy (d, cq_id, node, cmd->copy.dst_mem_id,
                             cmd->copy.dst, cmd->copy.src_mem_id,
                             cmd->copy.src, cmd->copy.dst_offset,
                             cmd->copy.src_offset, cmd->copy.size);
      goto FINISH_COMMAND;

    case CL_COMMAND_READ_BUFFER_RECT:
      pocl_proxy_enque_read_rect (
          d, cq_id, node, cmd->read_rect.dst_host_ptr,
          cmd->read_rect.src_mem_id, event->mem_objs[0],
          cmd->read_rect.buffer_origin, cmd->read_rect.host_origin,
          cmd->read_rect.region, cmd->read_rect.buffer_row_pitch,
          cmd->read_rect.buffer_slice_pitch, cmd->read_rect.host_row_pitch,
          cmd->read_rect.host_slice_pitch);
      goto FINISH_COMMAND;

    case CL_COMMAND_WRITE_BUFFER_RECT:
      pocl_proxy_enque_write_rect (
          d, cq_id, node, cmd->write_rect.src_host_ptr,
          cmd->write_rect.dst_mem_id, event->mem_objs[0],
          cmd->write_rect.buffer_origin, cmd->write_rect.host_origin,
          cmd->write_rect.region, cmd->write_rect.buffer_row_pitch,
          cmd->write_rect.buffer_slice_pitch, cmd->write_rect.host_row_pitch,
          cmd->write_rect.host_slice_pitch);
      goto FINISH_COMMAND;

    case CL_COMMAND_COPY_BUFFER_RECT:
      pocl_proxy_enque_copy_rect (
          d, cq_id, node, cmd->copy_rect.dst_mem_id, cmd->copy_rect.dst,
          cmd->copy_rect.src_mem_id, cmd->copy_rect.src,
          cmd->copy_rect.dst_origin, cmd->copy_rect.src_origin,
          cmd->copy_rect.region, cmd->copy_rect.dst_row_pitch,
          cmd->copy_rect.dst_slice_pitch, cmd->copy_rect.src_row_pitch,
          cmd->copy_rect.src_slice_pitch);
      goto FINISH_COMMAND;

    case CL_COMMAND_FILL_BUFFER:
      pocl_proxy_enque_memfill (d, cq_id, node, cmd->memfill.dst_mem_id,
                                event->mem_objs[0], cmd->memfill.size,
                                cmd->memfill.offset, cmd->memfill.pattern,
                                cmd->memfill.pattern_size);
      goto FINISH_COMMAND;

    case CL_COMMAND_MAP_BUFFER:
      pocl_proxy_enque_map_mem (d, cq_id, node, cmd->map.mem_id,
                                event->mem_objs[0], cmd->map.mapping);
      goto FINISH_COMMAND;

    case CL_COMMAND_UNMAP_MEM_OBJECT:

      if (event->mem_objs[0]->is_image == CL_FALSE
          || IS_IMAGE1D_BUFFER (event->mem_objs[0]))
        {
          pocl_proxy_enque_unmap_mem (d, cq_id, node, cmd->unmap.mem_id,
                                      event->mem_objs[0], cmd->unmap.mapping);
        }
      else
        {
          pocl_proxy_enque_unmap_image (d, cq_id, node, cmd->unmap.mem_id,
                                        event->mem_objs[0],
                                        cmd->unmap.mapping);
        }
      goto FINISH_COMMAND;

    case CL_COMMAND_NDRANGE_KERNEL:
      {
        pocl_proxy_enque_run (dev, d, context_device_i, cq_id, node);
        goto FINISH_COMMAND;
      }

    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
      pocl_proxy_enque_read_image_rect (
          d, cq_id, node, cmd->read_image.src, cmd->read_image.src_mem_id,
          NULL, cmd->read_image.dst_mem_id, cmd->read_image.origin,
          cmd->read_image.region, cmd->read_image.dst_row_pitch,
          cmd->read_image.dst_slice_pitch, cmd->read_image.dst_offset);
      goto FINISH_COMMAND;

    case CL_COMMAND_READ_IMAGE:
      pocl_proxy_enque_read_image_rect (
          d, cq_id, node, cmd->read_image.src, cmd->read_image.src_mem_id,
          cmd->read_image.dst_host_ptr, NULL, cmd->read_image.origin,
          cmd->read_image.region, cmd->read_image.dst_row_pitch,
          cmd->read_image.dst_slice_pitch, cmd->read_image.dst_offset);
      goto FINISH_COMMAND;

    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      pocl_proxy_enque_write_image_rect (
          d, cq_id, node, cmd->write_image.dst, cmd->write_image.dst_mem_id,
          NULL, cmd->write_image.src_mem_id, cmd->write_image.origin,
          cmd->write_image.region, cmd->write_image.src_row_pitch,
          cmd->write_image.src_slice_pitch, cmd->write_image.src_offset);
      goto FINISH_COMMAND;

    case CL_COMMAND_WRITE_IMAGE:
      pocl_proxy_enque_write_image_rect (
          d, cq_id, node, cmd->write_image.dst, cmd->write_image.dst_mem_id,
          cmd->write_image.src_host_ptr, NULL, cmd->write_image.origin,
          cmd->write_image.region, cmd->write_image.src_row_pitch,
          cmd->write_image.src_slice_pitch, cmd->write_image.src_offset);
      goto FINISH_COMMAND;

    case CL_COMMAND_COPY_IMAGE:
      pocl_proxy_enque_copy_image_rect (
          d, cq_id, node, cmd->copy_image.src, cmd->copy_image.dst,
          cmd->copy_image.src_mem_id, cmd->copy_image.dst_mem_id,
          cmd->copy_image.src_origin, cmd->copy_image.dst_origin,
          cmd->copy_image.region);
      goto FINISH_COMMAND;

    case CL_COMMAND_FILL_IMAGE:
      pocl_proxy_enque_fill_image (
          d, cq_id, node, event->mem_objs[0], cmd->fill_image.mem_id,
          cmd->fill_image.origin, cmd->fill_image.region,
          &cmd->fill_image.orig_pixel);
      goto FINISH_COMMAND;

    case CL_COMMAND_MAP_IMAGE:
      pocl_proxy_enque_map_image (d, cq_id, node, cmd->map.mem_id,
                                  event->mem_objs[0], cmd->map.mapping);
      goto FINISH_COMMAND;

    case CL_COMMAND_MARKER:
      clFlush (cq_id);
    case CL_COMMAND_BARRIER:
      goto FINISH_COMMAND;

    default:
      POCL_ABORT ("bug in code, unknown command type: %u\n", node->type);
    }

FINISH_COMMAND:

  cstr = pocl_command_to_str (node->type);
  char msg[128] = "Event ";
  strncat (msg, cstr, 127);

  POCL_UPDATE_EVENT_COMPLETE_MSG (event, msg);
}

static void *
pocl_proxy_queue_pthread (void *ptr)
{
  proxy_queue_data_t *qd = (proxy_queue_data_t *)ptr;

  _cl_command_node *cmd = NULL;
  cl_device_id device = qd->queue->device;
  proxy_device_data_t *d = (proxy_device_data_t *)device->data;
  POCL_FAST_LOCK (qd->wq_lock);

  while (1)
    {
      if (qd->cq_thread_exit_requested)
        {
          POCL_FAST_UNLOCK (qd->wq_lock);
          return NULL;
        }

      cmd = qd->work_queue;
      if (cmd)
        {
          DL_DELETE (qd->work_queue, cmd);
          POCL_FAST_UNLOCK (qd->wq_lock);

          assert (pocl_command_is_ready (cmd->event));
          assert (cmd->event->status == CL_SUBMITTED);

          proxy_exec_command (cmd, device, d, qd);
          /* if the proxy_exec_command called proxy_free_cmd_queue(),
           * return immediately */
          if (qd->cq_thread_exit_requested && qd->cq_thread_id==0)
            return NULL;

          POCL_FAST_LOCK (qd->wq_lock);
        }

      if ((qd->work_queue == NULL) && (qd->cq_thread_exit_requested == 0))
        {
          POCL_WAIT_COND (qd->wakeup_cond, qd->wq_lock);
        }
    }
}
