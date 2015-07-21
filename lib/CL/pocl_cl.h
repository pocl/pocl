/* pocl_cl.h - local runtime library declarations.

   Copyright (c) 2011 Universidad Rey Juan Carlos
                 2011-2012 Pekka Jääskeläinen
   
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
#include "pocl_debug.h"
#include "pocl_hash.h"


#if __STDC_VERSION__ < 199901L
# if __GNUC__ >= 2
#  define __func__ __PRETTY_FUNCTION__
# else
#  define __func__ UNKNOWN_FUNCTION
# endif
#endif

#ifdef CLANGXX
#define LINK_CMD CLANGXX
#else
#define LINK_CMD CLANG
#endif

typedef pthread_mutex_t pocl_lock_t;
#define POCL_LOCK_INITIALIZER PTHREAD_MUTEX_INITIALIZER

/* Generic functionality for handling different types of 
   OpenCL (host) objects. */

#define POCL_LOCK(__LOCK__) pthread_mutex_lock (&(__LOCK__))
#define POCL_UNLOCK(__LOCK__) pthread_mutex_unlock (&(__LOCK__))
#define POCL_INIT_LOCK(__LOCK__) pthread_mutex_init (&(__LOCK__), NULL)
#define POCL_DESTROY_LOCK(__LOCK__) pthread_mutex_destroy (&(__LOCK__))

#define POCL_LOCK_OBJ(__OBJ__) POCL_LOCK((__OBJ__)->pocl_lock)
#define POCL_UNLOCK_OBJ(__OBJ__) POCL_UNLOCK((__OBJ__)->pocl_lock)

#define POCL_RELEASE_OBJECT(__OBJ__, __NEW_REFCOUNT__)  \
  do {                                                  \
    POCL_LOCK_OBJ (__OBJ__);                            \
    __NEW_REFCOUNT__ = --(__OBJ__)->pocl_refcount;      \
    POCL_UNLOCK_OBJ (__OBJ__);                          \
    if (__NEW_REFCOUNT__ == 0) POCL_DESTROY_LOCK ((__OBJ__)->pocl_lock); \
  } while (0)

#define POCL_RETAIN_OBJECT(__OBJ__)             \
  do {                                          \
    POCL_LOCK_OBJ (__OBJ__);                    \
    (__OBJ__)->pocl_refcount++;                   \
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

/* Declares the generic pocl object attributes inside a struct. */
#define POCL_OBJECT \
  pocl_lock_t pocl_lock; \
  int pocl_refcount 

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
#  define POdeclsym(name)			\
  __typeof__(name) PO##name __attribute__((visibility("hidden")));
#  define POCL_ALIAS_OPENCL_SYMBOL(name)                                \
  __typeof__(name) name __attribute__((alias ("PO" #name), visibility("default")));
#  define POsymAlways(name) POCL_ALIAS_OPENCL_SYMBOL(name)
#  ifdef DIRECT_LINKAGE
#    define POsym(name) POCL_ALIAS_OPENCL_SYMBOL(name)
#  else
#    define POsym(name)
#  endif

#endif

/* The ICD compatibility part. This must be first in the objects where
 * it is used (as the ICD loader assumes that)*/
#ifdef BUILD_ICD
#  define POCL_ICD_OBJECT struct _cl_icd_dispatch *dispatch;
#  define POsymICD(name) POsym(name)
#  define POdeclsymICD(name) POdeclsym(name)
#else
#  define POCL_ICD_OBJECT
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

struct pocl_argument {
  size_t size;
  void *value;
};

/**
 * Enumeration for kernel argument types
 */
typedef enum {
  POCL_ARG_TYPE_NONE = 0,
  POCL_ARG_TYPE_POINTER = 1,
  POCL_ARG_TYPE_IMAGE = 2,
  POCL_ARG_TYPE_SAMPLER = 3,
} pocl_argument_type;

struct pocl_argument_info {
  pocl_argument_type type;
  char is_local;
  char is_set;
  cl_kernel_arg_address_qualifier address_qualifier;
  cl_kernel_arg_access_qualifier access_qualifier;
  char* type_name;
  cl_kernel_arg_type_qualifier type_qualifier;
  char* name;
};

struct pocl_device_ops {
  const char *device_name;
  void (*init_device_infos) (struct _cl_device_id*);
  /* implementation */
  void (*uninit) (cl_device_id device);
  unsigned int (*probe) (struct pocl_device_ops *ops);
  void (*init) (cl_device_id device, const char *parameters);
  cl_int (*alloc_mem_obj) (cl_device_id device, cl_mem mem_obj);
  void *(*create_sub_buffer) (void *data, void* buffer, size_t origin, size_t size);
  void (*free) (void *data, cl_mem_flags flags, void *ptr);
  void (*read) (void *data, void *host_ptr, const void *device_ptr, 
                size_t offset, size_t cb);
  void (*read_rect) (void *data, void *host_ptr, void *device_ptr,
                     const size_t *buffer_origin,
                     const size_t *host_origin, 
                     const size_t *region,
                     size_t buffer_row_pitch,
                     size_t buffer_slice_pitch,
                     size_t host_row_pitch,
                     size_t host_slice_pitch);
  void (*write) (void *data, const void *host_ptr, void *device_ptr, 
                 size_t offset, size_t cb);
  void (*write_rect) (void *data, const void *host_ptr, void *device_ptr,
                      const size_t *buffer_origin,
                      const size_t *host_origin, 
                      const size_t *region,
                      size_t buffer_row_pitch,
                      size_t buffer_slice_pitch,
                      size_t host_row_pitch,
                      size_t host_slice_pitch);
  void (*copy) (void *data, const void *src_ptr, size_t src_offset, 
                void *__restrict__ dst_ptr, size_t dst_offset, size_t cb);
  void (*copy_rect) (void *data, const void *src_ptr, void *dst_ptr,
                     const size_t *src_origin,
                     const size_t *dst_origin, 
                     const size_t *region,
                     size_t src_row_pitch,
                     size_t src_slice_pitch,
                     size_t dst_row_pitch,
                     size_t dst_slice_pitch);

void (*fill_rect) (void *data,
                   void *__restrict__ const device_ptr,
                   const size_t *__restrict__ const buffer_origin,
                   const size_t *__restrict__ const region,
                   size_t const buffer_row_pitch,
                   size_t const buffer_slice_pitch,
                   void *fill_pixel,
                   size_t pixel_size);

  /* Maps 'size' bytes of device global memory at buf_ptr + offset to 
     host-accessible memory. This might or might not involve copying 
     the block from the device. */
  void* (*map_mem) (void *data, void *buf_ptr, size_t offset, size_t size, void *host_ptr);
  void* (*unmap_mem) (void *data, void *host_ptr, void *device_start_ptr, size_t size);
  
  void (*compile_submitted_kernels) (_cl_command_node* cmd);
  void (*run) (void *data, _cl_command_node* cmd);
  void (*run_native) (void *data, _cl_command_node* cmd);

  cl_ulong (*get_timer_value) (void *data); /* The current device timer value in nanoseconds. */

  /* Perform initialization steps and can return additional
     build options that are required for the device. The caller
     owns the returned string. */
  char* (*init_build) (void *data);

  void (*build_hash) (void *data, SHA1_CTX *build_hash);

    /* return supported image formats */
  cl_int (*get_supported_image_formats) (cl_mem_flags flags,
                                         const cl_image_format **image_formats,
                                         cl_uint *num_image_formats);
};

struct _cl_device_id {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  /* queries */
  cl_device_type type;
  cl_uint vendor_id;
  cl_uint max_compute_units;
  cl_uint max_work_item_dimensions;
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
     in the WG, otherwise the hardware spawns the WIs. */
  cl_bool spmd;
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
  char *cache_dir_name;
  cl_device_id parent_device;

  const char *vendor;
  const char *driver_version;
  const char *profile;
  const char *version;
  const char *extensions;
 
  void *data;
  const char* llvm_target_triplet; /* the llvm target triplet to use */
  const char* llvm_cpu; /* the llvm CPU variant to use */
  /* A running number (starting from zero) across all the device instances. Used for 
     indexing  arrays in data structures with device specific entries. */
  int dev_id;
  int global_mem_id; /* identifier for device global memory */
  int has_64bit_long;  /* Does the device have 64bit longs */
  /* Convert automatic local variables to kernel arguments? */
  int autolocals_to_args;

  struct pocl_device_ops *ops; /* Device operations, shared amongst same devices */
};

struct _cl_platform_id {
  POCL_ICD_OBJECT
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
};

struct _cl_command_queue {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  /* queries */
  cl_context context;
  cl_device_id device;
  cl_command_queue_properties properties;
  /* implementation */
  _cl_command_node *root;
};

/* memory identifier: id to point the global memory where memory resides 
                      + pointer to actual data */
typedef struct _pocl_mem_identifier{
  int global_mem_id;
  void* mem_ptr;
} pocl_mem_identifier;

typedef struct _cl_mem cl_mem_t;
struct _cl_mem {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  /* queries */
  cl_mem_object_type type;
  cl_mem_flags flags;
  size_t size;
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
  /* A linked list of regions of the buffer mapped to the 
     host memory */
  mem_mapping_t *mappings;
  /* in case this is a sub buffer, this points to the parent
     buffer */
  cl_mem_t *parent;
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
};

typedef uint8_t SHA1_digest_t[SHA1_DIGEST_SIZE * 2 + 1];

struct _cl_program {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  /* queries */
  cl_context context;
  cl_uint num_devices;
  cl_device_id *devices;
  /* all the program sources appended together, terminated with a zero */
  char *source;
  /* The options in the last clBuildProgram call for this Program. */
  char *compiler_options;
  /* The binaries for each device. Currently the binary is directly the
     sequential bitcode produced from the kernel sources.*/
  size_t *binary_sizes; 
  unsigned char **binaries; 
  /* implementation */
  cl_kernel kernels;
  /* Per-device program hash after build */
  SHA1_digest_t* build_hash;
  /* Per-device build logs, for the case when we don't yet have the program's cachedir */
  char** build_log;
  /* Used to store the llvm IR of the build to save disk I/O. */
  void **llvm_irs;
  /* Use to store build status */
  cl_build_status build_status;
};

struct _cl_kernel {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  /* queries */
  char *name;
  cl_uint num_args;
  cl_context context;
  cl_program program;
  /* implementation */
  lt_dlhandle dlhandle;
  struct pocl_argument_info *arg_info;
  cl_bitfield has_arg_metadata;
  cl_uint num_locals;
  int *reqd_wg_size;
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

typedef struct _cl_event _cl_event;
struct _cl_event {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  cl_context context;
  cl_command_queue queue;
  cl_command_type command_type;

  /* list of callback functions */
  event_callback_item* callback_list;

  /* The execution status of the command this event is monitoring. */
  cl_int status;

  /* Profiling data: time stamps of the different phases of execution. */
  cl_ulong time_queue;  /* the enqueue time */
  cl_ulong time_submit; /* the time the command was submitted to the device */
  cl_ulong time_start;  /* the time the command actually started executing */
  cl_ulong time_end;    /* the finish time of the command */   

  /* impicit event = an event for pocl's internal use, not visible to user */
  int implicit_event;
  _cl_event * volatile next;
};

typedef struct _cl_sampler cl_sampler_t;

struct _cl_sampler {
  POCL_ICD_OBJECT
  cl_bool             normalized_coords;
  cl_addressing_mode  addressing_mode;
  cl_filter_mode      filter_mode;
};

#define POCL_UPDATE_EVENT_QUEUED(__event)                               \
  do {                                                                  \
    if ((__event) != NULL && (*(__event)) != NULL)                      \
      {                                                                 \
        cl_command_queue __cq = (*(__event))->queue;                    \
        (*(__event))->status = CL_QUEUED;                               \
        if (__cq && __cq->properties & CL_QUEUE_PROFILING_ENABLE)       \
          (*(__event))->time_queue =                                    \
            __cq->device->ops->get_timer_value(__cq->device->data);     \
      }                                                                 \
  } while (0)                                                           \

#define POCL_UPDATE_EVENT_SUBMITTED(__event)                            \
  do {                                                                  \
    if ((__event) != NULL && (*(__event)) != NULL)                      \
      {                                                                 \
        assert((*(__event))->status == CL_QUEUED);                      \
        (*(__event))->status = CL_SUBMITTED;                            \
        cl_command_queue __cq = (*(__event))->queue;                    \
        if (__cq && __cq->properties & CL_QUEUE_PROFILING_ENABLE)       \
          (*(__event))->time_submit =                                   \
            __cq->device->ops->get_timer_value(__cq->device->data);     \
      }                                                                 \
  } while (0)                                                           \

#define POCL_UPDATE_EVENT_RUNNING(__event)                              \
  do {                                                                  \
    if (__event != NULL && (*(__event)) != NULL)                        \
      {                                                                 \
        assert((*(__event))->status == CL_SUBMITTED);                   \
        (*(__event))->status = CL_RUNNING;                              \
        cl_command_queue __cq = (*(__event))->queue;                    \
        if (__cq && __cq->properties & CL_QUEUE_PROFILING_ENABLE)       \
          (*(__event))->time_start =                                    \
            __cq->device->ops->get_timer_value(__cq->device->data);     \
      }                                                                 \
  } while (0)                                                           \

#define POCL_UPDATE_EVENT_COMPLETE(__event)                             \
  do {                                                                  \
    if ((__event) != NULL && (*(__event)) != NULL)                      \
      {                                                                 \
        assert((*(__event))->status == CL_RUNNING);                     \
        (*(__event))->status = CL_COMPLETE;                             \
        cl_command_queue __cq = (*(__event))->queue;                    \
        if (__cq && __cq->properties & CL_QUEUE_PROFILING_ENABLE)       \
          (*(__event))->time_end =                                      \
            __cq->device->ops->get_timer_value(__cq->device->data);     \
      }                                                                 \
  } while (0)                                                           \

#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) > (b)) ? (a) : (b))

#endif /* POCL_CL_H */
