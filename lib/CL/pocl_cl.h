/* pocl_cl.h - local runtime library declarations.

   Copyright (c) 2011 Universidad Rey Juan Carlos
                 2011-2019 Pekka Jääskeläinen

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

#ifndef POCL_CL_H
#define POCL_CL_H

#include "config.h"

#include <assert.h>
#include <stdio.h>

#ifdef _MSC_VER
#  include "vccompat.hpp"
#endif
/* To get adaptive mutex type */
#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <pthread.h>
#ifdef HAVE_CLOCK_GETTIME
#include <time.h>
#endif

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

typedef struct pocl_kernel_metadata_s pocl_kernel_metadata_t;
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


/* If available, use an Adaptive mutex for locking in the pthread driver,
   otherwise fallback to simple mutexes */
#define POCL_FAST_LOCK_T pthread_mutex_t
#define POCL_FAST_LOCK(l) POCL_LOCK(l)
#define POCL_FAST_UNLOCK(l) POCL_UNLOCK(l)
#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
  #define POCL_FAST_INIT(l) \
    do { \
      pthread_mutexattr_t attrs; \
      pthread_mutexattr_init (&attrs); \
      int r = pthread_mutexattr_settype (&attrs, PTHREAD_MUTEX_ADAPTIVE_NP); \
      assert (r == 0); \
      pthread_mutex_init(&l, &attrs); \
      pthread_mutexattr_destroy(&attrs);\
    } while (0)
#else
  #define POCL_FAST_INIT(l) pthread_mutex_init(&l, NULL);
#endif
#define POCL_FAST_DESTROY(l) POCL_DESTROY_LOCK(l)



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

#define POCL_RELEASE_OBJECT(__OBJ__, __NEW_REFCOUNT__)                        \
  do                                                                          \
    {                                                                         \
      POCL_LOCK_OBJ (__OBJ__);                                                \
      __NEW_REFCOUNT__ = --(__OBJ__)->pocl_refcount;                          \
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
  while (0)

/* Declares the generic pocl object attributes inside a struct. */
#define POCL_OBJECT                                                           \
  pocl_lock_t pocl_lock;                                                      \
  int pocl_refcount

#define POCL_OBJECT_INIT \
  POCL_LOCK_INITIALIZER, 0

#ifdef __APPLE__
/* Note: OSX doesn't support aliases because it doesn't use ELF */

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

#ifdef __cplusplus
extern "C"
{
  CL_API_ENTRY cl_int CL_API_CALL POname (clReleaseEvent) (cl_event event)
      CL_API_SUFFIX__VERSION_1_0;
}
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
  uint64_t offset;
  void *value;
  /* 1 if this argument has been set by clSetKernelArg */
  int is_set;
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

#define ARG_IS_LOCAL(a) (a.address_qualifier == CL_KERNEL_ARG_ADDRESS_LOCAL)
#define ARGP_IS_LOCAL(a) (a->address_qualifier == CL_KERNEL_ARG_ADDRESS_LOCAL)

typedef struct pocl_argument_info {
  char* type_name;
  char* name;
  cl_kernel_arg_address_qualifier address_qualifier;
  cl_kernel_arg_access_qualifier access_qualifier;
  cl_kernel_arg_type_qualifier type_qualifier;
  pocl_argument_type type;
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
     commands are finished. Called by the user thread; see notify_cmdq_finished
     for the driver thread counterpart. */
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

  /* wait_event is called by clWaitForEvents() and blocks the execution until
   * the waited event is complete or failed. Called by user threads; see the
   * notify_event_finished() callback for driver thread counterpart.
   * Called (and must return) with unlocked event. */
  void (*wait_event) (cl_device_id device, cl_event event);

  /* update_event is an extra callback called during handling of event status
   * changes, useful if something device specific needs to be done. May be
   * NULL; no need to implement if not needed.
   *
   * Called via pocl_update_event_* functions in pocl_util.c
   * All pocl_update_event_* (except COMPLETE) must be called (and return)
   * with LOCKED event.
   */
  void (*update_event) (cl_device_id device, cl_event event);

  /* free_event_data may be called when event is freed. Event data may only be
     used by the device driver owning the corresponding command.
     No need to implement this if the device does not need any event data. */
  void (*free_event_data) (cl_event event);

  /* Called from driver threads to notify every user thread waiting on
   * command queue finish. See join() for user counterpart.
   * Driver may chose to not implement this, which will result in
   * undefined behaviour in multi-threaded user programs. */
  void (*notify_cmdq_finished) (cl_command_queue cq);

  /* Called from driver threads to notify every user thread waiting on
   * a specific event. See wait_event() for user counterpart.
   * Driver may chose to not implement this, which will result in
   * undefined behaviour in multi-threaded user programs. */
  void (*notify_event_finished) (cl_event event);

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

  /* Compile the fully linked LLVM IR to target-specific binaries.
     If specialize is set to 1, allow specializing the final result
     according to the properites of the given cmd. Typically
     specialization is _not_ wanted only when creating a generic
     WG function for storing in a binary. Local size of all zeros
     forces dynamic local size work-group even if specializing
     according to other properties. */
  void (*compile_kernel) (_cl_command_node *cmd, cl_kernel kernel,
                          cl_device_id device, int specialize);
  /* clEnqueueNDRangeKernel */
  void (*run) (void *data, _cl_command_node *cmd);
  /* for clEnqueueNativeKernel. may be NULL */
  void (*run_native) (void *data, _cl_command_node *cmd);

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
   * clEnqueueCopyBufferToImage: src_mem_id = buffer,
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
   * clEnqueueCopyImageToBuffer: dst_mem_id = buffer,
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


  /* custom device functionality */

  /* Check if the device supports the builtin kernel with the given name. */
  cl_int (*supports_builtin_kernel) (void *data, const char *kernel_name);

  /* Loads the metadata for the kernel with the given name to the MD object
     allocated at the given target. */
  cl_int (*get_builtin_kernel_metadata) (void *data, const char *kernel_name,
                                         pocl_kernel_metadata_t *target);
};

typedef struct pocl_global_mem_t {
  pocl_lock_t pocl_lock;
  size_t max_ever_allocated;
  size_t currently_allocated;
  size_t total_alloc_limit;
} pocl_global_mem_t;

#define NUM_OPENCL_IMAGE_TYPES 6

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
  /* Does the device set the event times in update_event() callback ?
   * if zero, the default event change handlers set the event times based on
   * the host's system time (pocl_gettimemono_ns). */
  int has_own_timer;
  /* Convert automatic local variables to kernel arguments? */
  int autolocals_to_args;
  /* Allocate local buffers device side in the work-group launcher instead of
     having a disjoint physical local memory per work-group or having the
     runtime/driver allocate the local space. */
  int device_alloca_locals;

  /* If > 0, specialized versions of the work-group functions are generated
     which assume each grid dimension is of at most the given width. This
     assumption can be then taken in account in IR optimization and codegen
     to reduce address computation overheads etc. */
  size_t grid_width_specialization_limit;

  /* Device-specific linker flags that should be appended to the clang's
     argument list for a final linkage call when producing the final binary
     that can be uploaded to the device using the default LLVM-based
     codegen. The final entry in the list must be NULL.

     The flags will be added after the following command line:
     clang -o final.bin input.obj [flags]
  */

  const char **final_linkage_flags;

  /* Auxiliary functions required by the device binary which should
     be retained across the kernel compilation unused code pruning
     process. */
  const char **device_aux_functions;

  /* The target specific IDs for the different OpenCL address spaces. */
  unsigned global_as_id;
  unsigned local_as_id;
  unsigned constant_as_id;

  /* The address space where the argument data is passed. */
  unsigned args_as_id;

  /* The address space where the grid context data is passed. */
  unsigned context_as_id;


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
  /* OpenCL 2.1 */
  char *spirv_version;

  /* image formats supported by the device, per image type */
  const cl_image_format *image_formats[NUM_OPENCL_IMAGE_TYPES];
  cl_uint num_image_formats[NUM_OPENCL_IMAGE_TYPES];

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

#define OPENCL_MAX_DIMENSION 3

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

  /*********************************************************************/
  /* these values depend on which devices are in context;
   * they're calculated by pocl_setup_context() */

  /* The largest of max_mem_alloc_size of all devices in context */
  size_t max_mem_alloc_size;

  /* union of image formats supported by all of the devices in context,
   * per image-type (there are 6 image types)
     TODO the getSupportedImageFormats is supposed to also respect flags,
     but for now we ignore that. */
  cl_image_format *image_formats[NUM_OPENCL_IMAGE_TYPES];
  cl_uint num_image_formats[NUM_OPENCL_IMAGE_TYPES];

  /* The device that should allocate SVM (might be == host)
   * NULL if none of devices in the context is SVM capable */
  cl_device_id svm_allocdev;

  /* The minimal required buffer alignment for all devices in the context.
   * E.g. for clCreateSubBuffer:
   * CL_MISALIGNED_SUB_BUFFER_OFFSET is returned in errcode_ret if there are no
   * devices in context associated with buffer for which the origin value
   * is aligned to the CL_DEVICE_MEM_BASE_ADDR_ALIGN value.
   */
  size_t min_buffer_alignment;
};

typedef struct _pocl_data_sync_item pocl_data_sync_item;
struct _pocl_data_sync_item {
  cl_event event;
  pocl_data_sync_item *next;
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
  struct _cl_event *barrier;
  unsigned long command_count; /* counter for unfinished command enqueued */
  pocl_data_sync_item last_event;

  /* device specific data */
  void *data;
};

#define POCL_ON_SUB_MISALIGN(mem, que, operation)                             \
  do                                                                          \
    {                                                                         \
      if (mem->parent != NULL)  {                                             \
        operation (                                                           \
            (mem->origin % que->device->mem_base_addr_align != 0),            \
            CL_MISALIGNED_SUB_BUFFER_OFFSET,                                  \
            "SubBuffer is not "                                               \
            "properly aligned for this device");                              \
        }                                                                     \
    }                                                                         \
  while (0)

#define POCL_RETURN_ON_SUB_MISALIGN(mem, que)                                 \
  POCL_ON_SUB_MISALIGN(mem, que, POCL_RETURN_ERROR_ON)

#define POCL_GOTO_ON_SUB_MISALIGN(mem, que)                                   \
  POCL_ON_SUB_MISALIGN(mem, que, POCL_GOTO_ERROR_ON)

#define POCL_CONVERT_SUBBUFFER_OFFSET(mem, offset)                            \
  if (mem->parent != NULL)                                                    \
    {                                                                         \
      offset += mem->origin;                                                  \
      mem = mem->parent;                                                      \
    }

#define DEVICE_IMAGE_SIZE_SUPPORT 1
#define DEVICE_IMAGE_FORMAT_SUPPORT 2

#define DEVICE_DOESNT_SUPPORT_IMAGE(mem, dev_i)                               \
  (mem->device_supports_this_image[dev_i] == 0)

#define POCL_ON_UNSUPPORTED_IMAGE(mem, dev, operation)                        \
  do                                                                          \
    {                                                                         \
      unsigned dev_i;                                                         \
      for (dev_i = 0; dev_i < mem->context->num_devices; ++dev_i)             \
        if (mem->context->devices[dev_i] == dev)                              \
          break;                                                              \
      assert (dev_i < mem->context->num_devices);                             \
      operation (                                                  \
          (mem->context->devices[dev_i]->image_support == CL_FALSE),          \
          CL_INVALID_OPERATION, "Device %s does not support images\n",        \
          mem->context->devices[dev_i]->long_name);                           \
      operation (                                                  \
          ((mem->device_supports_this_image[dev_i]                            \
            & DEVICE_IMAGE_FORMAT_SUPPORT)                                    \
           == 0),                                                             \
          CL_IMAGE_FORMAT_NOT_SUPPORTED,                                      \
          "The image type is not supported by this device\n");                \
      operation (                                                  \
          ((mem->device_supports_this_image[dev_i]                            \
            & DEVICE_IMAGE_SIZE_SUPPORT)                                      \
           == 0),                                                             \
          CL_INVALID_IMAGE_SIZE,                                              \
          "The image size is not supported by this device\n");                \
    }                                                                         \
  while (0)


#define POCL_RETURN_ON_UNSUPPORTED_IMAGE(mem, dev)                            \
  POCL_ON_UNSUPPORTED_IMAGE(mem, dev, POCL_RETURN_ERROR_ON)

#define POCL_GOTO_ON_UNSUPPORTED_IMAGE(mem, dev)                              \
  POCL_ON_UNSUPPORTED_IMAGE(mem, dev, POCL_GOTO_ERROR_ON)



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
  cl_device_id owning_device;

  /* A linked list of regions of the buffer mapped to the
     host memory */
  mem_mapping_t *mappings;
  /* in case this is a sub buffer, this points to the parent
     buffer */
  cl_mem_t *parent;
  /* A linked list of destructor callbacks */
  mem_destructor_callback_t *destructor_callbacks;

  /* for images, a flag for each device in context,
   * whether that device supports this */
  int *device_supports_this_image;

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

typedef struct pocl_kernel_metadata_s
{
  cl_uint num_args;
  cl_uint num_locals;
  size_t *local_sizes;
  char *name;
  char *attributes;
  struct pocl_argument_info *arg_info;
  cl_bitfield has_arg_metadata;
  size_t reqd_wg_size[OPENCL_MAX_DIMENSION];

  /* array[program->num_devices] */
  pocl_kernel_hash_t *build_hash;

  /* If this is a BI kernel descriptor, they are statically defined in
     the custom device driver, thus should not be freed. */
  cl_bitfield builtin_kernel;
  /* device-specific data, void* array[program->num_devices] */
  void **data;

} pocl_kernel_metadata_t;

struct _cl_program {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  /* queries */
  cl_context context;
  cl_uint num_devices;
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

  /* If this is a program with built-in kernels, this is the list of kernel
     names it contains. */
  size_t num_builtin_kernels;
  char **builtin_kernel_names;

  /* Poclcc binary format.  */
  size_t *pocl_binary_sizes;
  unsigned char **pocl_binaries;

  /* kernel number and the metadata for each kernel */
  size_t num_kernels;
  pocl_kernel_metadata_t *kernel_meta;

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

  /* Store SPIR-V binary from clCreateProgramWithIL() */
  char *program_il;
  size_t program_il_size;
};

struct _cl_kernel {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  /* -------- */
  cl_context context;
  cl_program program;
  pocl_kernel_metadata_t *meta;
  /* just a convenience pointer to meta->name */
  const char *name;

  /* The kernel arguments that are set with clSetKernelArg().
     These are copied to the command queue command at enqueue. */
  struct pocl_argument *dyn_arguments;
  struct _cl_kernel *next;
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
  event_node *next;
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
  event_callback_item *callback_list;

  /* list of devices needing completion notification for this event */
  event_node *notify_list;
  event_node *wait_list;

  /* OoO doesn't use sync points -> put used buffers here */
  size_t num_buffers;
  cl_mem *mem_objs;

  /* Profiling data: time stamps of the different phases of execution. */
  cl_ulong time_queue;  /* the enqueue time */
  cl_ulong time_submit; /* the time the command was submitted to the device */
  cl_ulong time_start;  /* the time the command actually started executing */
  cl_ulong time_end;    /* the finish time of the command */

  void *data; /* Device specific data */

  /* The execution status of the command this event is monitoring. */
  cl_int status;
  /* impicit event = an event for pocl's internal use, not visible to user */
  int implicit_event;

  _cl_event *next;
  _cl_event *prev;
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

#define CL_FAILED (-1)

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
  #if defined(__GLIBC__) && __GLIBC__ == 2 && \
      defined(__GLIBC_MINOR__) && __GLIBC_MINOR__ < 9 && \
      defined(__x86_64__)
    #ifndef htole64
      #define htole64(x) (x)
    #endif
    #ifndef htole32
      #define htole32(x) (x)
    #endif
    #ifndef htole16
      #define htole16(x) (x)
    #endif
    #ifndef le64toh
      #define le64toh(x) (x)
    #endif
    #ifndef le32toh
      #define le32toh(x) (x)
    #endif
    #ifndef le16toh
      #define le16toh(x) (x)
    #endif
  #endif
#endif

#endif /* POCL_CL_H */
