/* pocl_cl.h - local runtime library declarations.

   Copyright (c) 2011 Universidad Rey Juan Carlos
                 2011-2018 Pekka Jääskeläinen
   
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

#ifndef POCL_CL_H
#define POCL_CL_H

#include "config.h"

#include <assert.h>
#include <stdio.h>
#ifndef _MSC_VER
#  include <ltdl.h>
#else
#  include "vccompat.hpp"
#endif
#include <pthread.h>
#ifdef HAVE_CLOCK_GETTIME
#include <time.h>
#endif

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#ifdef BUILD_ICD
#  include "pocl_icd.h"
#endif
#include "pocl.h"
#include "pocl_tracing.h"
#include "pocl_debug.h"
#include "pocl_hash.h"
#include "pocl_runtime_config.h"
#include "common.h"

#if __STDC_VERSION__ < 199901L
# if __GNUC__ >= 2
#  define __func__ __PRETTY_FUNCTION__
# else
#  define __func__ UNKNOWN_FUNCTION
# endif
#endif

typedef pthread_mutex_t pocl_lock_t;
#define POCL_LOCK_INITIALIZER PTHREAD_MUTEX_INITIALIZER

/* Generic functionality for handling different types of 
   OpenCL (host) objects. */

#define POCL_LOCK(__LOCK__)                                                   \
  do                                                                          \
    {                                                                         \
      int r = pthread_mutex_lock (&(__LOCK__));                               \
      assert (r == 0);                                                        \
    }                                                                         \
  while (0)
#define POCL_UNLOCK(__LOCK__)                                                 \
  do                                                                          \
    {                                                                         \
      int r = pthread_mutex_unlock (&(__LOCK__));                             \
      assert (r == 0);                                                        \
    }                                                                         \
  while (0)
#define POCL_INIT_LOCK(__LOCK__)                                              \
  do                                                                          \
    {                                                                         \
      int r = pthread_mutex_init (&(__LOCK__), NULL);                         \
      assert (r == 0);                                                        \
    }                                                                         \
  while (0)
/* We recycle OpenCL objects by not actually freeing them until the
   very end. Thus, the lock should not be destoryed at the refcount 0. */
#define POCL_DESTROY_LOCK(__LOCK__)                                           \
  do                                                                          \
    {                                                                         \
      int r = pthread_mutex_destroy (&(__LOCK__));                            \
      assert (r == 0);                                                        \
    }                                                                         \
  while (0)

#define POCL_LOCK_OBJ(__OBJ__)                                                \
  do                                                                          \
    {                                                                         \
      POCL_LOCK ((__OBJ__)->pocl_lock);                                       \
      assert ((__OBJ__)->pocl_refcount > 0);                                  \
    }                                                                         \
  while (0)
#define POCL_UNLOCK_OBJ(__OBJ__)                                              \
  do                                                                          \
    {                                                                         \
      assert ((__OBJ__)->pocl_refcount >= 0);                                 \
      POCL_UNLOCK ((__OBJ__)->pocl_lock);                                     \
    }                                                                         \
  while (0)

#define POCL_RELEASE_OBJECT_UNLOCKED(__OBJ__, __NEW_REFCOUNT__)               \
  __NEW_REFCOUNT__ = --(__OBJ__)->pocl_refcount

#define POCL_RELEASE_OBJECT(__OBJ__, __NEW_REFCOUNT__)                        \
  do                                                                          \
    {                                                                         \
      POCL_LOCK_OBJ (__OBJ__);                                                \
      POCL_RELEASE_OBJECT_UNLOCKED (__OBJ__, __NEW_REFCOUNT__);               \
      POCL_UNLOCK_OBJ (__OBJ__);                                              \
    }                                                                         \
  while (0)

#define POCL_RETAIN_OBJECT_UNLOCKED(__OBJ__)    \
    ++((__OBJ__)->pocl_refcount);

#define POCL_RETAIN_OBJECT_REFCOUNT(__OBJ__, R) \
  do {                                          \
    POCL_LOCK_OBJ (__OBJ__);                    \
    R = POCL_RETAIN_OBJECT_UNLOCKED (__OBJ__);  \
    POCL_UNLOCK_OBJ (__OBJ__);                  \
  } while (0)

#define POCL_RETAIN_OBJECT(__OBJ__)             \
  do {                                          \
    POCL_LOCK_OBJ (__OBJ__);                    \
    POCL_RETAIN_OBJECT_UNLOCKED (__OBJ__);      \
    POCL_UNLOCK_OBJ (__OBJ__);                  \
  } while (0)

/* The reference counter is initialized to 1,
   when it goes to 0 object can be freed. */
#define POCL_INIT_OBJECT_NO_ICD(__OBJ__)         \
  do {                                           \
    POCL_INIT_LOCK ((__OBJ__)->pocl_lock);         \
    (__OBJ__)->pocl_refcount = 1;                  \
  } while (0)

#define POCL_MEM_FREE(F_PTR)                      \
  do {                                            \
      free((F_PTR));                              \
      (F_PTR) = NULL;                             \
  } while (0)

#ifdef BUILD_ICD
/* Most (all?) object must also initialize the ICD field */
#  define POCL_INIT_OBJECT(__OBJ__)                \
    do {                                           \
      POCL_INIT_OBJECT_NO_ICD(__OBJ__);            \
      POCL_INIT_ICD_OBJECT(__OBJ__);               \
    } while (0)
#else
#  define POCL_INIT_OBJECT(__OBJ__)                \
      POCL_INIT_OBJECT_NO_ICD(__OBJ__)
#endif

#define POCL_DESTROY_OBJECT(__OBJ__)                                          \
  do                                                                          \
    {                                                                         \
      POCL_DESTROY_LOCK ((__OBJ__)->pocl_lock);                               \
    }                                                                         \
  while (0);

/* Declares the generic pocl object attributes inside a struct. */
#define POCL_OBJECT \
  pocl_lock_t pocl_lock; \
  volatile int pocl_refcount 

#define POCL_OBJECT_INIT \
  POCL_LOCK_INITIALIZER, 0

#ifdef __APPLE__
/* Note: OSX doesn't support aliases because it doesn't use ELF */

#  ifdef BUILD_ICD
#    error "ICD not supported on OSX"
#  endif
#  define POname(name) name
#  define POdeclsym(name)
#  define POsym(name)
#  define POsymAlways(name)

#elif defined(_MSC_VER)
/* Visual Studio does not support this magic either */
#  define POname(name) name
#  define POdeclsym(name)
#  define POsym(name)
#  define POsymAlways(name)
#  define POdeclsym(name)

#else
/* Symbol aliases are supported */

#  define POname(name) PO##name
#  define POdeclsym(name)                      \
  __typeof__(name) PO##name __attribute__((visibility("hidden")));
#  define POCL_ALIAS_OPENCL_SYMBOL(name)                                \
  __typeof__(name) name __attribute__((alias ("PO" #name), visibility("default")));
#  define POsymAlways(name) POCL_ALIAS_OPENCL_SYMBOL(name)
#  if !defined(BUILD_ICD)
#    define POsym(name) POCL_ALIAS_OPENCL_SYMBOL(name)
#  else
#    define POsym(name)
#  endif

#endif

/* The ICD compatibility part. This must be first in the objects where
 * it is used (as the ICD loader assumes that)*/
#ifdef BUILD_ICD
#  define POCL_ICD_OBJECT struct _cl_icd_dispatch *dispatch;
#  define POCL_ICD_OBJECT_PLATFORM_ID POCL_ICD_OBJECT
#  define POsymICD(name) POsym(name)
#  define POdeclsymICD(name) POdeclsym(name)
#else
#  define POCL_ICD_OBJECT
#  define POCL_ICD_OBJECT_PLATFORM_ID unsigned long id;
#  define POsymICD(name)
#  define POdeclsymICD(name)
#endif

#include "pocl_intfn.h"

/* fields for cl_kernel -> has_arg_metadata */
#define POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER  1
#define POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER   2
#define POCL_HAS_KERNEL_ARG_TYPE_NAME          4
#define POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER     8
#define POCL_HAS_KERNEL_ARG_NAME               16

typedef struct pocl_argument {
  uint64_t size;
  void *value;
} pocl_argument;


typedef struct event_node event_node;

/**
 * Enumeration for kernel argument types
 */
typedef enum {
  POCL_ARG_TYPE_NONE = 0,
  POCL_ARG_TYPE_POINTER = 1,
  POCL_ARG_TYPE_IMAGE = 2,
  POCL_ARG_TYPE_SAMPLER = 3,
} pocl_argument_type;

typedef struct pocl_argument_info {
  char* type_name;
  char* name;
  cl_kernel_arg_address_qualifier address_qualifier;
  cl_kernel_arg_access_qualifier access_qualifier;
  cl_kernel_arg_type_qualifier type_qualifier;
  pocl_argument_type type;
  char is_local;
  char is_set;
  unsigned type_size;
} pocl_argument_info;


struct pocl_device_ops {
  const char *device_name;

  /* New driver api extension for out-of-order execution and
     asynchronous devices.
     See this for reference: http://URN.fi/URN:NBN:fi:tty-201412051583
     See basic and pthread driver for reference. */

  /* submit gives the command for the device. The command may be left in the cq
     or stored to the device driver owning the cq. submit is called
     with node->event locked, and must return with it unlocked. */
  void (*submit) (_cl_command_node *node, cl_command_queue cq);

  /* join is called by clFinish and this function blocks until all the enqueued
     commands are finished. */
  void (*join) (cl_device_id device, cl_command_queue cq);

  /* flush is called when clFlush is called. This function ensures that
     commands will be eventually executed. It is up to the device what happens
     here, if anything. See basic and pthread for reference.*/
  void (*flush) (cl_device_id device, cl_command_queue cq);

  /* notify is used to communicate to a device driver that an event, it has
     been waiting, has been completed. Upon call, both events are locked, and
     must be locked also on return.*/
  void (*notify) (cl_device_id device, cl_event event, cl_event finished);

  /* broadcast is(has to be) called by the device driver when a command is
     completed.
     It is used to broadcast notifications to device drivers waiting
     this event to complete.
     There is a default implementation for this. Use it if there is no need
     to do anything special here.
     The default implementation calls notify(event, target_event) for the
     list of events waiting on 'event'. */
  void (*broadcast) (cl_event event);

  /* wait_event is blocking the execution until the waited event is complete.
     Called (and must return) with unlocked event. */
  void (*wait_event) (cl_device_id device, cl_event event);

  /* update_event is an alternative way of handling event status changes if
     something device specific needs to be done when the status of the event
     changes.
     This function is called via POCL_UPDATE_EVENT_* macros if available.
     All POCL_UPDATE_EVENT_ (except COMPLETE) must be called with LOCKED event.
     may be NULL, no need to implement if not needed.
     Called (and must return) with locked event. */
  void (*update_event) (cl_device_id device, cl_event event, cl_int status);

  /* free_event_data may be called when event is freed. Event data may only be
     used by the device driver owning the corresponding command.
     No need to implement this if the device does not need any event data. */
  void (*free_event_data) (cl_event event);

  /* /New driver api extension */

  /* Detects & returns the number of available devices the driver finds on the system. */
  unsigned int (*probe) (struct pocl_device_ops *ops);
  /* Device initialization. Parameters:
   *  j : progressive index for the devices of the same type
   *  device : struct to initialize
   *  parameters : optional environment with device-specific parameters
   */
  cl_int (*init) (unsigned j, cl_device_id device, const char *parameters);
  /* uninitializes the driver for a particular device. May free hardware resources. */
  cl_int (*uninit) (unsigned j, cl_device_id device);
  /* reinitializes the driver for a particular device. Called after uninit;
   * the first initialization is done by 'init'. May be NULL */
  cl_int (*reinit) (unsigned j, cl_device_id device);

  /* if the driver needs to use hardware resources for command queues, use this */
  cl_int (*init_queue) (cl_command_queue queue);
  void (*free_queue) (cl_command_queue queue);

  /* allocate a buffer in device memory */
  cl_int (*alloc_mem_obj) (cl_device_id device, cl_mem mem_obj, void* host_ptr);
  void *(*create_sub_buffer) (void *data, void* buffer, size_t origin, size_t size);
  /* free a device buffer */
  void (*free) (cl_device_id device, cl_mem mem_obj);

  /* clEnqueSVMfree - free a SVM memory pointer. May be NULL if device doesn't
   * support SVM. */
  void (*svm_free) (cl_device_id dev, void *svm_ptr);
  void *(*svm_alloc) (cl_device_id dev, cl_svm_mem_flags flags, size_t size);
  void (*svm_map) (cl_device_id dev, void *svm_ptr);
  void (*svm_unmap) (cl_device_id dev, void *svm_ptr);
  /* we can use restrict here, because Spec says overlapping copy should return
   * with CL_MEM_COPY_OVERLAP error. */
  void (*svm_copy) (cl_device_id dev, void *__restrict__ dst,
                    const void *__restrict__ src, size_t size);
  void (*svm_fill) (cl_device_id dev, void *__restrict__ svm_ptr, size_t size,
                    void *__restrict__ pattern, size_t pattern_size);

  /* the following callbacks only deal with buffers (and IMAGE1D_BUFFER which
   * is backed by a buffer), not images.  */

  /* clEnqReadBuffer */
  void (*read) (void *data,
                void *__restrict__  dst_host_ptr,
                pocl_mem_identifier * src_mem_id,
                cl_mem src_buf,
                size_t offset,
                size_t size);
  /* clEnqReadBufferRect */
  void (*read_rect) (void *data,
                     void *__restrict__ dst_host_ptr,
                     pocl_mem_identifier * src_mem_id,
                     cl_mem src_buf,
                     const size_t *buffer_origin,
                     const size_t *host_origin, 
                     const size_t *region,
                     size_t buffer_row_pitch,
                     size_t buffer_slice_pitch,
                     size_t host_row_pitch,
                     size_t host_slice_pitch);
  /* clEnqWriteBuffer */
  void (*write) (void *data,
                 const void *__restrict__  src_host_ptr,
                 pocl_mem_identifier * dst_mem_id,
                 cl_mem dst_buf,
                 size_t offset,
                 size_t size);
  /* clEnqWriteBufferRect */
  void (*write_rect) (void *data,
                      const void *__restrict__ src_host_ptr,
                      pocl_mem_identifier * dst_mem_id,
                      cl_mem dst_buf,
                      const size_t *buffer_origin,
                      const size_t *host_origin, 
                      const size_t *region,
                      size_t buffer_row_pitch,
                      size_t buffer_slice_pitch,
                      size_t host_row_pitch,
                      size_t host_slice_pitch);
  /* clEnqCopyBuffer */
  void (*copy) (void *data,
                pocl_mem_identifier * dst_mem_id,
                cl_mem dst_buf,
                pocl_mem_identifier * src_mem_id,
                cl_mem src_buf,
                size_t dst_offset,
                size_t src_offset,
                size_t size);
  /* clEnqCopyBufferRect */
  void (*copy_rect) (void *data,
                     pocl_mem_identifier * dst_mem_id,
                     cl_mem dst_buf,
                     pocl_mem_identifier * src_mem_id,
                     cl_mem src_buf,
                     const size_t *dst_origin,
                     const size_t *src_origin,
                     const size_t *region,
                     size_t dst_row_pitch,
                     size_t dst_slice_pitch,
                     size_t src_row_pitch,
                     size_t src_slice_pitch);

  /* clEnqFillBuffer */
  void (*memfill) (void *data,
                   pocl_mem_identifier * dst_mem_id,
                   cl_mem dst_buf,
                   size_t size,
                   size_t offset,
                   const void *__restrict__  pattern,
                   size_t pattern_size);

  /* Maps 'size' bytes of device global memory at  + offset to
     host-accessible memory. This might or might not involve copying 
     the block from the device. */
  cl_int (*map_mem) (void *data,
                     pocl_mem_identifier * src_mem_id,
                     cl_mem src_buf,
                     mem_mapping_t *map);
  cl_int (*unmap_mem) (void *data,
                       pocl_mem_identifier * dst_mem_id,
                       cl_mem dst_buf,
                       mem_mapping_t *map);

  /* compile the fully linked LLVM IR to target-specific binaries. */
  void (*compile_kernel) (_cl_command_node* cmd, cl_kernel kernel, cl_device_id device);
  /* clEnqueueNDRangeKernel */
  void (*run) (void *data, _cl_command_node* cmd);
  /* for clEnqueueNativeKernel. may be NULL */
  void (*run_native) (void *data, _cl_command_node* cmd);

  /* The current device timer value in nanoseconds. */
  cl_ulong (*get_timer_value) (void *data);

  /* Perform initialization steps and can return additional
     build options that are required for the device. The caller
     owns the returned string. may be NULL */
  char* (*init_build) (void *data);

  /* may be NULL */
  void (*init_target_machine) (void *data, void *target_machine);

  /* returns a hash string that should identify the device. This string
   * is used when writing/loading pocl binaries to decide compatibility. */
  char* (*build_hash) (cl_device_id device);

  /* the following callbacks deal with images ONLY, with the exception of
   * IMAGE1D_BUFFER type (which is implemented as a buffer).
   * If the device does not support images, all of these may be NULL. */

  /* return supported image formats */
  cl_int (*get_supported_image_formats) (cl_mem_flags flags,
                                         const cl_image_format **image_formats,
                                         cl_uint *num_image_formats);

  /* returns a device specific pointer which may reference
   * a hardware resource. May be NULL */
  void* (*create_image) (void *data,
                         const cl_image_format * image_format,
                         const cl_image_desc *   image_desc,
                         cl_mem image,
                         cl_int *err);
  /* free the device-specific pointer (image_data) from create_image() */
  cl_int (*free_image) (void *data,
                        cl_mem image,
                        void *image_data);

  /* creates a device-specific hardware resource for sampler. May be NULL */
  void* (*create_sampler) (void *data,
                           cl_sampler samp,
                           cl_int *err);
  cl_int (*free_sampler) (void *data,
                          cl_sampler samp,
                          void *sampler_data);

  /* copies image to image, on the same device (or same global memory). */
  cl_int (*copy_image_rect) (void *data,
                             cl_mem src_image,
                             cl_mem dst_image,
                             pocl_mem_identifier *src_mem_id,
                             pocl_mem_identifier *dst_mem_id,
                             const size_t *src_origin,
                             const size_t *dst_origin,
                             const size_t *region);

  /* copies a region from host OR device buffer to device image.
   * clEnqueueCopyImageToBuffer: src_mem_id = buffer,
   *     src_host_ptr = NULL, src_row_pitch = src_slice_pitch = 0
   * clEnqueueWriteImage: src_mem_id = NULL,
   *     src_host_ptr = host pointer, src_offset = 0
   */
  cl_int (*write_image_rect ) (void *data,
                               cl_mem dst_image,
                               pocl_mem_identifier *dst_mem_id,
                               const void *__restrict__ src_host_ptr,
                               pocl_mem_identifier *src_mem_id,
                               const size_t *origin,
                               const size_t *region,
                               size_t src_row_pitch,
                               size_t src_slice_pitch,
                               size_t src_offset);

  /* copies a region from device image to host or device buffer
   * clEnqueueCopyBufferToImage: dst_mem_id = buffer,
   *     dst_host_ptr = NULL, dst_row_pitch = dst_slice_pitch = 0
   * clEnqueueReadImage: dst_mem_id = NULL,
   *     dst_host_ptr = host pointer, dst_offset = 0
   */
  cl_int (*read_image_rect) (void *data,
                             cl_mem src_image,
                             pocl_mem_identifier *src_mem_id,
                             void *__restrict__ dst_host_ptr,
                             pocl_mem_identifier *dst_mem_id,
                             const size_t *origin,
                             const size_t *region,
                             size_t dst_row_pitch,
                             size_t dst_slice_pitch,
                             size_t dst_offset);

  /* maps the entire image from device to host */
  cl_int (*map_image) (void *data,
                       pocl_mem_identifier *mem_id,
                       cl_mem src_image,
                       mem_mapping_t *map);

  /* unmaps the entire image from host to device */
  cl_int (*unmap_image) (void *data,
                         pocl_mem_identifier *mem_id,
                         cl_mem dst_image,
                         mem_mapping_t *map);

  /* fill image with pattern */
  cl_int (*fill_image)(void *data,
                       cl_mem image,
                       pocl_mem_identifier *mem_id,
                       const size_t *origin,
                       const size_t *region,
                       const void *__restrict__ fill_pixel,
                       size_t pixel_size);

};

typedef struct pocl_global_mem_t {
  pocl_lock_t pocl_lock;
  size_t max_ever_allocated;
  size_t currently_allocated;
  size_t total_alloc_limit;
} pocl_global_mem_t;

struct _cl_device_id {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  /* queries */
  cl_device_type type;
  cl_uint vendor_id;
  cl_uint max_compute_units;

  /* for subdevice support */
  cl_device_id parent_device;
  unsigned core_start;
  unsigned core_count;

  cl_uint max_work_item_dimensions;
  /* when enabled, Workgroup LLVM pass will replace all printf() calls
   * with calls to __pocl_printf and recursively change functions to
   * add printf buffer arguments from pocl_context.
   * Currently the pthread/basic devices require this; other devices
   * implement printf their own way. */
  int device_side_printf;
  size_t max_work_item_sizes[3];
  size_t max_work_group_size;
  size_t preferred_wg_size_multiple;
  cl_uint preferred_vector_width_char;
  cl_uint preferred_vector_width_short;
  cl_uint preferred_vector_width_int;
  cl_uint preferred_vector_width_long;
  cl_uint preferred_vector_width_float;
  cl_uint preferred_vector_width_double;
  cl_uint preferred_vector_width_half;
  cl_uint native_vector_width_char;
  cl_uint native_vector_width_short;
  cl_uint native_vector_width_int;
  cl_uint native_vector_width_long;
  cl_uint native_vector_width_float;
  cl_uint native_vector_width_double;
  cl_uint native_vector_width_half;
  cl_uint max_clock_frequency;
  cl_uint address_bits;
  cl_ulong max_mem_alloc_size;
  cl_bool image_support;
  cl_uint max_read_image_args;
  cl_uint max_write_image_args;
  cl_uint max_read_write_image_args;
  size_t image2d_max_width;
  size_t image2d_max_height;
  size_t image3d_max_width;
  size_t image3d_max_height;
  size_t image3d_max_depth;
  size_t image_max_buffer_size;
  size_t image_max_array_size;
  cl_uint max_samplers;
  size_t max_parameter_size;
  cl_uint mem_base_addr_align;
  cl_uint min_data_type_align_size;
  cl_device_fp_config half_fp_config;
  cl_device_fp_config single_fp_config;
  cl_device_fp_config double_fp_config;
  cl_device_mem_cache_type global_mem_cache_type;
  cl_uint global_mem_cacheline_size;
  cl_ulong global_mem_cache_size;
  cl_ulong global_mem_size;
  size_t global_var_pref_size;
  size_t global_var_max_size;
  cl_ulong max_constant_buffer_size;
  cl_uint max_constant_args;
  cl_device_local_mem_type local_mem_type;
  cl_ulong local_mem_size;
  cl_bool error_correction_support;
  cl_bool host_unified_memory;
  size_t profiling_timer_resolution;
  cl_bool endian_little;
  cl_bool available;
  cl_bool compiler_available;
  /* Is the target a Single Program Multiple Data machine? If not,
     we need to generate work-item loops to execute all the work-items
     in the WG. For SPMD machines, the hardware spawns the WIs. */
  cl_bool spmd;
  /* The device uses an HSA-like kernel ABI with a single argument buffer as
     an input. */
  cl_bool arg_buffer_launcher;
  /* The Workgroup pass creates launcher functions and replaces work-item
     placeholder global variables (e.g. _local_size_, _global_offset_ etc) with
     loads from the context struct passed as a kernel argument. This flag
     enables or disables this pass. */
  cl_bool workgroup_pass;
  cl_device_exec_capabilities execution_capabilities;
  cl_command_queue_properties queue_properties;
  cl_platform_id platform;
  cl_uint max_sub_devices;
  size_t num_partition_properties;
  cl_device_partition_property *partition_properties;
  size_t num_partition_types;
  cl_device_partition_property *partition_type;
  size_t printf_buffer_size;
  char *short_name;
  char *long_name;

  const char *vendor;
  const char *driver_version;
  const char *profile;
  const char *version;
  const char *extensions;
  const char *cl_version_std;  // "CL2.0"
  cl_ulong cl_version_int;     // 200

  void *data;
  const char* llvm_target_triplet; /* the llvm target triplet to use */
  const char* llvm_cpu; /* the llvm CPU variant to use */
  /* A running number (starting from zero) across all the device instances.
     Used for indexing arrays in data structures with device specific
     entries. */
  int dev_id;
  int global_mem_id; /* identifier for device global memory */
  /* pointer to an accounting struct for global memory */
  pocl_global_mem_t *global_memory;
  int has_64bit_long;  /* Does the device have 64bit longs */
  /* Convert automatic local variables to kernel arguments? */
  int autolocals_to_args;

  /* The target specific IDs for the different OpenCL address spaces. */
  unsigned global_as_id;
  unsigned local_as_id;
  unsigned constant_as_id;

  /* True if the device supports SVM. Then it has the responsibility of
     allocating shared buffers residing in Shared Virtual Memory areas. */
  cl_bool should_allocate_svm;
  /* OpenCL 2.0 properties */
  cl_device_svm_capabilities svm_caps;
  cl_uint max_events;
  cl_uint max_queues;
  cl_uint max_pipe_args;
  cl_uint max_pipe_active_res;
  cl_uint max_pipe_packet_size;
  cl_uint dev_queue_pref_size;
  cl_uint dev_queue_max_size;
  cl_command_queue_properties on_dev_queue_props;
  cl_command_queue_properties on_host_queue_props;

  /* Device operations, shared among devices of the same type */
  struct pocl_device_ops *ops;
};

#define DEVICE_SVM_FINEGR(dev) (dev->svm_caps & (CL_DEVICE_SVM_FINE_GRAIN_BUFFER \
                                              | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM))
#define DEVICE_SVM_ATOM(dev) (dev->svm_caps & CL_DEVICE_SVM_ATOMICS)

#define DEVICE_IS_SVM_CAPABLE(dev) (dev->svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)

#define DEVICE_MMAP_IS_NOP(dev) (DEVICE_SVM_FINEGR(dev) && DEVICE_SVM_ATOM(dev))

#define CHECK_DEVICE_AVAIL_RET(dev) if(!dev->available) { POCL_MSG_ERR("This cl_device is not available.\n"); return CL_INVALID_DEVICE; }
#define CHECK_DEVICE_AVAIL_RETV(dev) if(!dev->available) { POCL_MSG_ERR("This cl_device is not available.\n"); return; }


struct _cl_platform_id {
  POCL_ICD_OBJECT_PLATFORM_ID
}; 

struct _cl_context {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  /* queries */
  cl_device_id *devices;
  cl_context_properties *properties;
  /* implementation */
  unsigned num_devices;
  unsigned num_properties;
  /* some OpenCL apps (AMD OpenCL SDK at least) use a trial-error 
     approach for creating a context with a device type, and call 
     clReleaseContext for the result regardless if it failed or not. 
     Returns a valid = 0 context in that case.  */
  char valid;

  /* The minimal value of max_mem_alloc_size of all devices in context */
  size_t min_max_mem_alloc_size;
  /* The device that should allocate SVM (might be == host)
   * NULL if none of devices in the context is SVM capable */
  cl_device_id svm_allocdev;
};

typedef struct _pocl_data_sync_item pocl_data_sync_item;
struct _pocl_data_sync_item {
  unsigned int volatile event_id;
  cl_event volatile event;
  pocl_data_sync_item *volatile next;
};

struct _cl_event;
struct _cl_command_queue {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  /* queries */
  cl_context context;
  cl_device_id device;
  cl_command_queue_properties properties;
  /* implementation */
  cl_event events; /* events of the enqueued commands in enqueue order */
  struct _cl_event * volatile barrier;
  volatile int command_count; /* counter for unfinished command enqueued */
  volatile pocl_data_sync_item last_event;

  /* backend specific data */
  void *data;
};


typedef struct _cl_mem cl_mem_t;
struct _cl_mem {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  /* queries */
  cl_mem_object_type type;
  cl_mem_flags flags;
  size_t size;
  size_t origin; /* for sub-buffers */
  void *mem_host_ptr;
  cl_uint map_count;
  cl_context context;
  /* implementation */
  /* The device-specific pointers to the buffer for all
     device ids the buffer was allocated to. This can be a
     direct pointer to the memory of the buffer or a pointer to
     a book keeping structure. This always contains
     as many pointers as there are devices in the system, even
     though the buffer was not allocated for all.
     The location of the device's buffer ptr is determined by
     the device's dev_id. */
  pocl_mem_identifier *device_ptrs;
  /* device that allocated, and is going to free, 
     the shared system mem allocation */
  cl_device_id shared_mem_allocation_owner;
  /* device where this mem obj resides */
  volatile cl_device_id owning_device;
  /* A linked list of regions of the buffer mapped to the 
     host memory */
  mem_mapping_t *mappings;
  /* in case this is a sub buffer, this points to the parent
     buffer */
  cl_mem_t *parent;
  /* A linked list of destructor callbacks */
  mem_destructor_callback_t *destructor_callbacks;

  /* Image flags */
  cl_bool                 is_image;
  cl_channel_order        image_channel_order;
  cl_channel_type         image_channel_data_type;
  size_t                  image_width;
  size_t                  image_height;
  size_t                  image_depth;
  size_t                  image_array_size;
  size_t                  image_row_pitch;
  size_t                  image_slice_pitch;
  size_t                  image_elem_size;
  size_t                  image_channels;
  cl_uint                 num_mip_levels;
  cl_uint                 num_samples;
  cl_mem                  buffer;

  /* pipe flags */
  cl_bool                 is_pipe;
  size_t                  pipe_packet_size;
  size_t                  pipe_max_packets;
};

typedef uint8_t SHA1_digest_t[SHA1_DIGEST_SIZE * 2 + 1];

struct _cl_program {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  /* queries */
  cl_context context;
  cl_uint num_devices;
  /* bool flag, set to 1 when removing/adding default kernels to a program
   * This code needs to be eventually fixed by introducing kernel_metadata
   * struct, see Issue #390 */
  int operating_on_default_kernels;
  /* -cl-denorms-are-zero build option */
  unsigned flush_denorms;
  cl_device_id *devices;
  /* all the program sources appended together, terminated with a zero */
  char *source;
  /* The options in the last clBuildProgram call for this Program. */
  char *compiler_options;
  /* The binaries for each device.  Currently the binary is directly the
     sequential bitcode produced from the kernel sources.  */
  size_t *binary_sizes;
  unsigned char **binaries;

  /* Poclcc binary format.  */
  size_t *pocl_binary_sizes;
  unsigned char **pocl_binaries;

  /* "Default" kernels. See: https://github.com/pocl/pocl/issues/390  */
  size_t num_kernels;
  char **kernel_names;
  cl_kernel *default_kernels;

  /* implementation */
  cl_kernel kernels;
  /* Per-device program hash after build */
  SHA1_digest_t* build_hash;
  /* Per-device build logs, for the case when we don't yet have the program's cachedir */
  char** build_log;
  /* Per-program build log, for the case when we aren't yet building for devices */
  char main_build_log[640];
  /* Used to store the llvm IR of the build to save disk I/O. */
  void **llvm_irs;
  /* Use to store build status */
  cl_build_status build_status;
  /* Use to store binary type */
  cl_program_binary_type binary_type;
};

struct _cl_kernel {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  /* queries */
  char *name;
  cl_uint num_args;
  cl_context context;
  cl_program program;
  struct pocl_argument_info *arg_info;
  cl_bitfield has_arg_metadata;
  char *attributes;
  cl_uint num_locals;
  size_t *reqd_wg_size;
  /* The kernel arguments that are set with clSetKernelArg().
     These are copied to the command queue command at enqueue. */
  struct pocl_argument *dyn_arguments;
  struct _cl_kernel *next;

  /* backend specific data */
  void *data;
};

typedef struct event_callback_item event_callback_item;
struct event_callback_item
{
  void (*callback_function) (cl_event, cl_int, void*);
  void *user_data;
  cl_int trigger_status;
  struct event_callback_item *next;
};


struct event_node
{
  cl_event event;
  event_node * volatile next;
};

typedef struct _cl_event _cl_event;
struct _cl_event {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  cl_context context;
  cl_command_queue queue;
  cl_command_type command_type;
  _cl_command_node *command;
  unsigned int id;

  /* list of callback functions */
  event_callback_item *volatile callback_list;

  /* list of devices needing completion notification for this event */
  event_node * volatile notify_list;
  event_node * volatile wait_list;

  /* OoO doesn't use sync points -> put used buffers here */
  cl_mem *mem_objs;
  int num_buffers;

  /* The execution status of the command this event is monitoring. */
  volatile cl_int status;

  /* Profiling data: time stamps of the different phases of execution. */
  cl_ulong time_queue;  /* the enqueue time */
  cl_ulong time_submit; /* the time the command was submitted to the device */
  cl_ulong time_start;  /* the time the command actually started executing */
  cl_ulong time_end;    /* the finish time of the command */

  void *data; /* Device specific data */  

  /* impicit event = an event for pocl's internal use, not visible to user */
  int implicit_event;
  _cl_event * volatile next;
  _cl_event * volatile prev;
};

typedef struct _pocl_user_event_data
{
  pthread_cond_t wakeup_cond;
} pocl_user_event_data;

typedef struct _cl_sampler cl_sampler_t;
struct _cl_sampler {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  cl_context context;
  cl_bool             normalized_coords;
  cl_addressing_mode  addressing_mode;
  cl_filter_mode      filter_mode;
  void**              device_data;
};

#define POCL_UPDATE_EVENT_QUEUED(__event)                                     \
  do                                                                          \
    {                                                                         \
      if ((__event) != NULL)                                                  \
        {                                                                     \
          cl_command_queue __cq = (__event)->queue;                           \
          if ((__cq)->device->ops->update_event)                              \
            (__cq)->device->ops->update_event ((__cq)->device, (__event),     \
                                               CL_QUEUED);                    \
          else                                                                \
            {                                                                 \
              (__event)->status = CL_QUEUED;                                  \
              if (__cq && __cq->properties & CL_QUEUE_PROFILING_ENABLE)       \
                (__event)->time_queue = __cq->device->ops->get_timer_value (  \
                    __cq->device->data);                                      \
            }                                                                 \
          pocl_event_updated (__event, CL_QUEUED);                            \
        }                                                                     \
    }                                                                         \
  while (0)

#define POCL_UPDATE_EVENT_SUBMITTED(__event)                                  \
  do                                                                          \
    {                                                                         \
      if ((__event) != NULL)                                                  \
        {                                                                     \
          assert ((__event)->status == CL_QUEUED);                            \
          cl_command_queue __cq = (__event)->queue;                           \
          if ((__cq)->device->ops->update_event)                              \
            (__cq)->device->ops->update_event ((__cq)->device, (__event),     \
                                               CL_SUBMITTED);                 \
          else                                                                \
            {                                                                 \
              (__event)->status = CL_SUBMITTED;                               \
              if (__cq && __cq->properties & CL_QUEUE_PROFILING_ENABLE)       \
                (__event)->time_submit = __cq->device->ops->get_timer_value ( \
                    __cq->device->data);                                      \
            }                                                                 \
          pocl_event_updated (__event, CL_SUBMITTED);                         \
        }                                                                     \
    }                                                                         \
  while (0)

#define POCL_UPDATE_EVENT_RUNNING_UNLOCKED(__event)                           \
  do                                                                          \
    {                                                                         \
      if (__event != NULL)                                                    \
        {                                                                     \
          assert ((__event)->status == CL_SUBMITTED);                         \
          cl_command_queue __cq = (__event)->queue;                           \
          if ((__cq)->device->ops->update_event)                              \
            (__cq)->device->ops->update_event ((__cq)->device, (__event),     \
                                               CL_RUNNING);                   \
          else                                                                \
            {                                                                 \
              (__event)->status = CL_RUNNING;                                 \
              if (__cq && __cq->properties & CL_QUEUE_PROFILING_ENABLE)       \
                (__event)->time_start = __cq->device->ops->get_timer_value (  \
                    __cq->device->data);                                      \
            }                                                                 \
          pocl_event_updated (__event, CL_RUNNING);                           \
        }                                                                     \
    }                                                                         \
  while (0)

#define POCL_UPDATE_EVENT_RUNNING(__event)                                    \
  POCL_LOCK_OBJ (__event);                                                    \
  POCL_UPDATE_EVENT_RUNNING_UNLOCKED (__event);                               \
  POCL_UNLOCK_OBJ (__event);

#define POCL_UPDATE_EVENT_COMPLETE_INNER(__event, POST_EVENT)                 \
  do                                                                          \
    {                                                                         \
      if ((__event) != NULL)                                                  \
        {                                                                     \
          assert ((__event)->status == CL_RUNNING);                           \
          POCL_LOCK_OBJ (__event);                                            \
          cl_command_queue __cq = (__event)->queue;                           \
          if ((__cq)->device->ops->update_event)                              \
            (__cq)->device->ops->update_event ((__cq)->device, (__event),     \
                                               CL_COMPLETE);                  \
          else                                                                \
            {                                                                 \
              pocl_mem_objs_cleanup (__event);                                \
              (__event)->status = CL_COMPLETE;                                \
              if ((__cq)->properties & CL_QUEUE_PROFILING_ENABLE)             \
                {                                                             \
                  (__event)->time_end                                         \
                      = (__cq)->device->ops->get_timer_value (                \
                          (__cq)->device->data);                              \
                }                                                             \
              POCL_UNLOCK_OBJ (__event);                                      \
              pocl_update_command_queue (__event);                            \
              (__cq)->device->ops->broadcast (__event);                       \
              POCL_LOCK_OBJ (__event);                                        \
            }                                                                 \
          pocl_event_updated (__event, CL_COMPLETE);                          \
          POST_EVENT;                                                         \
          POCL_UNLOCK_OBJ (__event);                                          \
          POname (clReleaseEvent) (__event);                                  \
        }                                                                     \
    }                                                                         \
  while (0)

#define POCL_UPDATE_EVENT_COMPLETE(__event)                                   \
  POCL_UPDATE_EVENT_COMPLETE_INNER (__event, NULL)

#define POCL_UPDATE_EVENT_COMPLETE_MSG(__event, msg)                          \
  POCL_UPDATE_EVENT_COMPLETE_INNER (__event,                                  \
                                    POCL_DEBUG_EVENT_TIME ((__event), msg))

#define CL_FAILED (-1)

#define POCL_UPDATE_EVENT_FAILED(__event)                                     \
  do                                                                          \
    {                                                                         \
      if ((__event) != NULL)                                                  \
        {                                                                     \
          cl_command_queue __cq = (__event)->queue;                           \
          if ((__cq)->device->ops->update_event)                              \
            (__cq)->device->ops->update_event ((__cq)->device, (__event),     \
                                               CL_FAILED);                    \
          else                                                                \
            {                                                                 \
              pocl_mem_objs_cleanup (__event);                                \
              if ((__event)->status > CL_COMPLETE)                            \
                (__event)->status = CL_FAILED;                                \
              if ((__cq)->properties & CL_QUEUE_PROFILING_ENABLE)             \
                {                                                             \
                  (__event)->time_end                                         \
                      = (__cq)->device->ops->get_timer_value (                \
                          (__cq)->device->data);                              \
                }                                                             \
              POCL_UNLOCK_OBJ (__event);                                      \
              pocl_update_command_queue (__event);                            \
              (__cq)->device->ops->broadcast (__event);                       \
              POCL_LOCK_OBJ (__event);                                        \
            }                                                                 \
          pocl_event_updated (__event, CL_FAILED);                            \
          POCL_UNLOCK_OBJ (__event);                                          \
          POname (clReleaseEvent) (__event);                                  \
          POCL_LOCK_OBJ (__event);                                            \
        }                                                                     \
    }                                                                         \
  while (0)

#ifndef __cplusplus

#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) > (b)) ? (a) : (b))

#endif

#ifdef __APPLE__
  #include <libkern/OSByteOrder.h>
  #define htole16(x) OSSwapHostToLittleInt16(x)
  #define le16toh(x) OSSwapLittleToHostInt16(x)
  #define htole32(x) OSSwapHostToLittleInt32(x)
  #define le32toh(x) OSSwapLittleToHostInt32(x)
  #define htole64(x) OSSwapHostToLittleInt64(x)
  #define le64toh(x) OSSwapLittleToHostInt64(x)
#elif defined(__FreeBSD__)
  #include <sys/endian.h>
#else
  #include <endian.h>
#endif

#endif /* POCL_CL_H */
