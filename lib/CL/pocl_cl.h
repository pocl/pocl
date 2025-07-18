/* pocl_cl.h - local runtime library declarations.

   Copyright (c) 2011 Universidad Rey Juan Carlos
                 2011-2019 Pekka Jääskeläinen
                 2023-2024 Pekka Jääskeläinen / Intel Finland Oy

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
#include <errno.h>
#include <stdio.h>

#if defined(__FreeBSD__)
#include <stdlib.h>
#elif defined(_WIN32)
#include <malloc.h>
#else
#include <alloca.h>
#endif

#ifdef ENABLE_VALGRIND
#include <valgrind/helgrind.h>
#endif

#ifdef _WIN32
#  include "vccompat.hpp"
#endif

#include "pocl.h"

#include "pocl_debug.h"
#include "pocl_hash.h"
#include "pocl_runtime_config.h"
#include "pocl_threads.h"
#include "pocl_tracing.h"
#ifdef BUILD_ICD
#  include "pocl_icd.h"
#endif

#include "pocl_raw_ptr_set.h"

#include <CL/cl_egl.h>

#if defined(__STDC_VERSION__) && __STDC_VERSION__ < 199901L
# if __GNUC__ >= 2
#  define __func__ __PRETTY_FUNCTION__
# else
#  define __func__ UNKNOWN_FUNCTION
# endif
#endif

#ifdef ENABLE_VALGRIND
#define VG_REFC_ZERO(var)                                                     \
  ANNOTATE_HAPPENS_AFTER (&var->pocl_refcount);                               \
  ANNOTATE_HAPPENS_BEFORE_FORGET_ALL (&var->pocl_refcount)
#define VG_REFC_NONZERO(var) ANNOTATE_HAPPENS_BEFORE (&var->pocl_refcount)
#else
#define VG_REFC_ZERO(var) (void)0
#define VG_REFC_NONZERO(var) (void)0
#endif

/*
 * A workaround for a detection problem in Valgrind, which causes
 * false positives. Valgrind manual states:
 *
 * Helgrind only partially correctly handles POSIX condition variables. This is
 * because Helgrind can see inter-thread dependencies between a
 * pthread_cond_wait call and a pthread_cond_signal/ pthread_cond_broadcast
 * call only if the waiting thread actually gets to the rendezvous first (so
 * that it actually calls pthread_cond_wait). It can't see dependencies between
 * the threads if the signaller arrives first. In the latter case, POSIX
 * guidelines imply that the associated boolean condition still provides an
 * inter-thread synchronisation event, but one which is invisible to Helgrind.
 *
 * ... this macro explicitly associates the cond var with the mutex, by
 * calling pthread_cond_wait with a short timeout (10 usec)
 */

#ifdef ENABLE_VALGRIND
#define VG_ASSOC_COND_VAR(cond_var, mutex)                                    \
  do                                                                          \
    {                                                                         \
      POCL_TIMEDWAIT_COND (cond_var, mutex, 10);                              \
    }                                                                         \
  while (0)
#else
#define VG_ASSOC_COND_VAR(var, mutex) (void)0
#endif

#ifdef __linux__
#define ALIGN_CACHE(x) POCL_ALIGNAS(HOST_CPU_CACHELINE_SIZE) x
#else
#define ALIGN_CACHE(x) x
#endif

//############################################################################

#ifdef ENABLE_EXTRA_VALIDITY_CHECKS
// 13_729_512_230_397_775_139
#define POCL_MAGIC_1 0xBE8906A1A83D8D23ULL
// 511_941_616_703_887_367
#define POCL_MAGIC_2 0x071AC830215FD807ULL

#define IS_CL_OBJECT_VALID(__OBJ__)                                           \
  (((__OBJ__) != NULL) && ((__OBJ__)->magic_1 == POCL_MAGIC_1)                \
   && ((__OBJ__)->magic_2 == POCL_MAGIC_2))
#define CHECK_VALIDITY_MARKERS(__OBJ__)                                       \
      assert ((__OBJ__)->magic_1 == POCL_MAGIC_1);                            \
      assert ((__OBJ__)->magic_2 == POCL_MAGIC_2);
#define SET_VALIDITY_MARKERS(__OBJ__)                                         \
      (__OBJ__)->magic_1 = POCL_MAGIC_1;                                      \
      (__OBJ__)->magic_2 = POCL_MAGIC_2;
#define UNSET_VALIDITY_MARKERS(__OBJ__)                                       \
      (__OBJ__)->magic_1 = 0;                                                 \
      (__OBJ__)->magic_2 = 0
#else
#define IS_CL_OBJECT_VALID(__OBJ__)   ((__OBJ__) != NULL)
#define CHECK_VALIDITY_MARKERS(__OBJ__) do {} while(0)
#define SET_VALIDITY_MARKERS(__OBJ__) do {} while(0)
#define UNSET_VALIDITY_MARKERS(__OBJ__) do {} while(0)
#endif

#define POCL_LOCK_OBJ(__OBJ__)                                                \
  do                                                                          \
    {                                                                         \
      CHECK_VALIDITY_MARKERS(__OBJ__);                                        \
      POCL_LOCK ((__OBJ__)->pocl_lock);                                       \
      assert ((__OBJ__)->pocl_refcount > 0);                                  \
    }                                                                         \
  while (0)

#define POCL_LOCK_OBJ_NO_CHECK(__OBJ__)                                       \
  do                                                                          \
    {                                                                         \
      POCL_LOCK ((__OBJ__)->pocl_lock);                                       \
    }                                                                         \
  while (0)

#define POCL_UNLOCK_OBJ(__OBJ__)                                              \
  do                                                                          \
    {                                                                         \
      CHECK_VALIDITY_MARKERS (__OBJ__);                                       \
      assert ((__OBJ__)->pocl_refcount >= 0);                                 \
      POCL_UNLOCK ((__OBJ__)->pocl_lock);                                     \
    }                                                                         \
  while (0)

#define POCL_UNLOCK_OBJ_NO_CHECK(__OBJ__)                                     \
  do                                                                          \
    {                                                                         \
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

#define POCL_RELEASE_OBJECT_UNLOCKED(__OBJ__, __NEW_REFCOUNT__)               \
      __NEW_REFCOUNT__ = --(__OBJ__)->pocl_refcount;

#define POCL_RETAIN_OBJECT_UNLOCKED(__OBJ__)    \
    ++((__OBJ__)->pocl_refcount)

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

extern pocl_obj_id_t last_object_id;

/* The reference counter is initialized to 1,
   when it goes to 0 object can be freed. */
#define POCL_INIT_OBJECT_NO_ICD(__OBJ__)                                      \
  do                                                                          \
    {                                                                         \
      SET_VALIDITY_MARKERS(__OBJ__);                                          \
      (__OBJ__)->pocl_refcount = 1;                                           \
      POCL_INIT_LOCK ((__OBJ__)->pocl_lock);                                  \
      (__OBJ__)->id = POCL_ATOMIC_INC (last_object_id);                       \
    }                                                                         \
  while (0)

#define POCL_MEM_FREE(F_PTR)                      \
  do {                                            \
      free((F_PTR));                              \
      (F_PTR) = NULL;                             \
  } while (0)

#ifdef BUILD_ICD
/* Most (all?) object must also initialize the ICD field */
#define POCL_INIT_OBJECT(__OBJ__, __PARENT__)                                 \
  do                                                                          \
    {                                                                         \
      POCL_INIT_OBJECT_NO_ICD (__OBJ__);                                      \
      POCL_INIT_ICD_OBJECT (__OBJ__, __PARENT__);                             \
    }                                                                         \
  while (0)
#else
#define POCL_INIT_OBJECT(__OBJ__, __PARENT__) POCL_INIT_OBJECT_NO_ICD (__OBJ__)
#endif

#define POCL_DESTROY_OBJECT(__OBJ__)                                          \
  do                                                                          \
    {                                                                         \
      UNSET_VALIDITY_MARKERS(__OBJ__);                                        \
      POCL_DESTROY_LOCK ((__OBJ__)->pocl_lock);                               \
    }                                                                         \
  while (0)

/* Declares the generic pocl object attributes inside a struct. */
#ifdef ENABLE_EXTRA_VALIDITY_CHECKS
#define POCL_OBJECT                                                           \
  uint64_t magic_1;                                                           \
  uint64_t id;                                                                \
  pocl_lock_t pocl_lock;                                                      \
  uint64_t magic_2;                                                           \
  int pocl_refcount
#else
#define POCL_OBJECT                                                           \
  uint64_t id;                                                                \
  pocl_lock_t pocl_lock;                                                      \
  int pocl_refcount
#endif

#ifdef __APPLE__
/* Note: OSX doesn't support aliases because it doesn't use ELF */

#  define POname(name) name
#  define POdeclsym(name)
#  define POdeclsymExport(name)
#  define POsym(name)
#  define POsymAlways(name)

#elif defined(_WIN32) && !defined(__MINGW32__)
/* Visual Studio does not support this magic either */
#  define POname(name) name
#  define POdeclsym(name)
#  define POdeclsymExport(name)
#  define POsym(name)
#  define POsymAlways(name)

#else
/* Symbol aliases are supported */

#  define POname(name) PO##name

#if defined(RENAME_POCL) && !defined(BUILD_ICD)
#define POdeclsym(name) POCL_EXPORT __typeof__ (name) PO##name;
#define POdeclsymExport(name) POdeclsym(name)
#else
#define POdeclsym(name) __typeof__ (name) PO##name;
#define POdeclsymExport(name) POCL_EXPORT POdeclsym(name)
#endif

#  define POCL_ALIAS_OPENCL_SYMBOL(name)                                \
  __typeof__(name) name __attribute__((alias ("PO" #name), visibility("default")));

#if !defined(BUILD_ICD) && !defined(RENAME_POCL)
#    define POsym(name) POCL_ALIAS_OPENCL_SYMBOL(name)
#  else
#    define POsym(name)
#  endif

#if !defined(RENAME_POCL)
#    define POsymAlways(name) POCL_ALIAS_OPENCL_SYMBOL(name)
#  else
#    define POsymAlways(name)
#  endif

#endif

/* The ICD compatibility part. This must be first in the objects where
 * it is used (as the ICD loader assumes that)*/

/* This block allows building using outdated headers that do not contain
 * ICD 2 definitions. */
#ifndef CL_ICD2_TAG_KHR
/* Defines a unique tag that signals an implementation is ICD 2 compatible
 * when set in the clGetPlatformIDs and clUnloadCompiler of the dispatch
 * table. */
#if INTPTR_MAX == INT32_MAX
#define CL_ICD2_TAG_KHR ((intptr_t)0x434C3331)
#else
#define CL_ICD2_TAG_KHR ((intptr_t)0x4F50454E434C3331)
#endif

typedef void *CL_API_CALL clIcdGetFunctionAddressForPlatformKHR_t (
  cl_platform_id platform, const char *function_name);

typedef clIcdGetFunctionAddressForPlatformKHR_t
  *clIcdGetFunctionAddressForPlatformKHR_fn;

extern CL_API_ENTRY void *CL_API_CALL clIcdGetFunctionAddressForPlatformKHR (
  cl_platform_id platform, const char *func_name);

typedef cl_int CL_API_CALL
clIcdSetPlatformDispatchDataKHR_t (cl_platform_id platform, void *disp_data);

typedef clIcdSetPlatformDispatchDataKHR_t *clIcdSetPlatformDispatchDataKHR_fn;

extern CL_API_ENTRY cl_int CL_API_CALL
clIcdSetPlatformDispatchDataKHR (cl_platform_id platform, void *dispatch_data);
#endif /* !defined(CL_ICD2_TAG_KHR) */

#ifdef BUILD_ICD
#  define POCL_ICD_OBJECT struct _cl_icd_dispatch *dispatch; void *disp_data;
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

/* pocl specific flag, for "hidden" default queues allocated in each context */
#define CL_QUEUE_HIDDEN (1 << 10)

#define POCL_GVAR_INIT_KERNEL_NAME "pocl.gvar.init"

/* Stores kernel argument data defined by clSetKernelArg(). */
typedef struct pocl_argument {
  size_t size;
  /* The "offset" is used to simplify subbuffer handling.
   * At enqueue time, subbuffers are converted to buffers + offset into them.
   */
  size_t offset;
  void *value;
  /* 1 if this argument has been set by clSetKernelArg. */
  char is_set;
  /* 1 if the argument is read-only according to kernel metadata. So either
   * a buffer with "const" qualifier, or an image with read_only qualifier.  */
  char is_readonly;
  /* 1 if the argument pointer is a raw pointer (SVM, USM or device),
     not a cl_mem handle. */
  char is_raw_ptr;
} pocl_argument;

typedef struct event_node event_node;

/**
 * Enumeration for kernel argument types
 */
typedef enum
{
  POCL_ARG_TYPE_NONE = 0,
  POCL_ARG_TYPE_POINTER = 1,
  POCL_ARG_TYPE_IMAGE = 2,
  POCL_ARG_TYPE_SAMPLER = 3,
  POCL_ARG_TYPE_PIPE = 4,
  POCL_ARG_TYPE_MUTABLE = 0, /* POD type with mutable size, only used by DBKs */
} pocl_argument_type;

#define ARG_IS_LOCAL(a) (a.address_qualifier == CL_KERNEL_ARG_ADDRESS_LOCAL)
#define ARGP_IS_LOCAL(a) (a->address_qualifier == CL_KERNEL_ARG_ADDRESS_LOCAL)

/* Kernel argument metadata. */
typedef struct pocl_argument_info {
  char *type_name;
  char *name;
  cl_kernel_arg_address_qualifier address_qualifier;
  cl_kernel_arg_access_qualifier access_qualifier;
  cl_kernel_arg_type_qualifier type_qualifier;
  pocl_argument_type type;
  unsigned type_size;
} pocl_argument_info;

/* Struct for storing information of a cl_mem that should
   be migrated to the device before executing a kernel. */
/* typedef is in pocl.h */
struct _pocl_buffer_migration_info
{
  /* The buffer to migrate (can be a sub-buffer). */
  cl_mem buffer;
  /* If the buffer is declared read-only at creation or in kernel argument
   * list. */
  char read_only;
  /* For utlist.h linked lists. */
  struct _pocl_buffer_migration_info *prev, *next;
};

/* The device driver layer operations. The device implementations override
   these hooks for their device-specific functionality. */
struct pocl_device_ops {
  const char *device_name;

  /****** The API for the out-of-order execution API and asynchronous devices.

     See this master's thesis for reference documentation:
     http://URN.fi/URN:NBN:fi:tty-201412051583
     See basic and pthread drivers for reference implementations. */

  /**
   * Passes a command for the device.
   *
   * The command may be left in the cq or stored to the device driver owning
   * the cq. submit is called with node->sync.event.event locked, and must
   * return with it unlocked. */
  void (*submit) (_cl_command_node *node, cl_command_queue cq);

  /**
   * Called by clFinish and this function blocks until all the enqueued
   * commands are finished.
   *
   * Called by the user thread; see notify_cmdq_finished
   * for the driver thread counterpart. */
  void (*join) (cl_device_id device, cl_command_queue cq);

  /**
   * Called when clFlush is called.
   *
   * This function ensures that
   * commands will be eventually executed. It is up to the device what happens
   * here, if anything. See basic and pthread for reference.*/
  void (*flush) (cl_device_id device, cl_command_queue cq);

  /**
   * Used to communicate to a device driver that an event, it has
   * been waiting, has been completed.
   *
   * Upon call, both events are locked, and
   * must be locked also on return. */
  void (*notify) (cl_device_id device, cl_event event, cl_event finished);

  /**
   * Must be called by the device driver when a command is completed.
   *
   * It is used to broadcast notifications to device drivers waiting
   * this event to complete. There is a default implementation for this. Use it
   * if there is no need to do anything special here. The default
   * implementation calls notify (event, target_event) for the list of events
   * waiting on 'event'. */
  void (*broadcast) (cl_event event);

  /**
   * Called by clWaitForEvents() and blocks the execution until
   * the waited event is complete or failed.
   *
   * Called by user threads; see the
   * notify_event_finished() callback for driver thread counterpart.
   * Called (and must return) with unlocked event. */
  void (*wait_event) (cl_device_id device, cl_event event);

  /**
   * Optional: An extra callback called during handling of event status
   * changes, useful if something device specific needs to be done.
   *
   * Called via pocl_update_event_* functions in pocl_util.c
   * All pocl_update_event_* (except COMPLETE) must be called (and return)
   * with LOCKED event.
   */
  void (*update_event) (cl_device_id device, cl_event event);

  /**
   * May be called when event is freed.
   *
   * Event data may only be
   * used by the device driver owning the corresponding command.
   * No need to implement this if the device does not need any event data. */
  void (*free_event_data) (cl_event event);

  /**
   * Called from driver threads to notify every user thread waiting on
   * command queue finish.
   *
   * See join() for user counterpart.
   * The river may chose to not implement this, which will result in
   * undefined behaviour in multi-threaded user programs. */
  void (*notify_cmdq_finished) (cl_command_queue cq);

  /**
   * Called from driver threads to notify every user thread waiting on
   * a specific event.
   *
   * See wait_event() for user counterpart.
   * Driver may chose to not implement this, which will result in
   * undefined behaviour in multi-threaded user programs. */
  void (*notify_event_finished) (cl_event event);

  /****** Device init/uninit APIs. */

  /**
   * Detects & returns the number of available devices the driver finds on the
   * system. */
  unsigned int (*probe) (struct pocl_device_ops *ops);

  /**
   * Device initialization.
   *
   * @param j progressive index for the devices of the same type
   * @param device struct to initialize
   * @param parameters optional environment with device-specific parameters
   */
  cl_int (*init) (unsigned j, cl_device_id device, const char *parameters);

  /**
   * Device type initialization after all devices have been initialized
   */
  cl_int (*post_init) (struct pocl_device_ops *ops);

  /** Uninitializes the driver for a particular device.
   *
   * May free hardware resources. */
  cl_int (*uninit) (unsigned j, cl_device_id device);

  /**
   * Reinitializes the driver for a particular device.
   *
   * Called after uninit; the first initialization is done by 'init'. May be
   * NULL */
  cl_int (*reinit) (unsigned j, cl_device_id device, const char *parameters);

  /**
   * Initialize discovery mechanism in the driver to dynamically find new
   * devices
   * \param add_discovered_device callback in runtime to init found devices
   * \param pocl_dev_type_idx index for the device type in runtime's device
   *                          handle.
   */
  cl_int (*init_discovery) (cl_int (*add_discovered_device) (const char *,
                                                             unsigned,
                                                             cl_platform_id),
                            unsigned pocl_dev_type_idx,
                            cl_platform_id pocl_dev_platform);

  /****** Memory management APIs. */

  /** Allocate a buffer in the device's global memory. */
  cl_int (*alloc_mem_obj) (cl_device_id device, cl_mem mem_obj, void* host_ptr);

  /** Free a device buffer. */
  void (*free) (cl_device_id device, cl_mem mem_obj);

  /** Optional: Allocate/register a sub-buffer in the device's global memory.
   *
   * If defined, this function should set the mem_ptr for the global mem of the
   * device to the subbuffer position/id. The mem_ptr of the memory is
   * pre-initialized to the parent buffer's starting address + the sub-buffer
   * offset.
   *
   * @param sub_buf is initialized in clCreateSubBuffer(). */
  cl_int (*alloc_subbuffer) (cl_device_id device, cl_mem sub_buf);

  /** Optional: Free a sub-buffer in the device's global memory. */
  void (*free_subbuffer) (cl_device_id device, cl_mem mem_obj);

  /** Return >0 if the driver can migrate directly between devices.
   *
   * Priority between devices signalled by larger numbers. */
  int (*can_migrate_d2d) (cl_device_id dest, cl_device_id source);

  /** Migrate buffer content directly between devices. */
  int (*migrate_d2d) (cl_device_id src_dev,
                      cl_device_id dst_dev,
                      cl_mem mem,
                      pocl_mem_identifier *src_mem_id,
                      pocl_mem_identifier *dst_mem_id);

  /** Shared Virtual Memory operations. */

  void (*svm_free) (cl_device_id dev, void *svm_ptr);
  void *(*svm_alloc) (cl_device_id dev, cl_svm_mem_flags flags, size_t size);
  void (*svm_map) (cl_device_id dev, void *svm_ptr);
  void (*svm_unmap) (cl_device_id dev, void *svm_ptr);

  /** These are optional. If the driver needs to do anything to be able
   * to use host memory, it should do it (and undo it) in these callbacks.
   * Currently used by HSA.
   * See pocl_driver_alloc_mem_obj and pocl_driver_free for details. */
  void (*svm_register) (cl_device_id dev, void *host_ptr, size_t size);
  void (*svm_unregister) (cl_device_id dev, void *host_ptr, size_t size);

  /** We can use restrict here, because Spec says overlapping copy should
   * return with CL_MEM_COPY_OVERLAP error. */
  void (*svm_copy) (cl_device_id dev, void *__restrict__ dst,
                    const void *__restrict__ src, size_t size);
  void (*svm_fill) (cl_device_id dev, void *__restrict__ svm_ptr, size_t size,
                    void *__restrict__ pattern, size_t pattern_size);
  void (*svm_migrate) (cl_device_id dev, size_t num_svm_pointers,
                       void *__restrict__ svm_pointers,
                       size_t *__restrict__ sizes);
  void (*svm_advise) (cl_device_id dev, const void *svm_ptr, size_t size,
                      cl_mem_advice_intel advice);

  /** Required for PoCL's command buffer extensions */
  void (*svm_copy_rect) (cl_device_id dev,
                         void *__restrict__ dst,
                         const void *__restrict__ src,
                         const size_t *dst_origin,
                         const size_t *src_origin,
                         const size_t *region,
                         size_t dst_row_pitch,
                         size_t dst_slice_pitch,
                         size_t src_row_pitch,
                         size_t src_slice_pitch);
  void (*svm_fill_rect) (cl_device_id dev,
                         void *__restrict__ svm_ptr,
                         const size_t *origin,
                         const size_t *region,
                         size_t row_pitch,
                         size_t slice_pitch,
                         void *__restrict__ pattern,
                         size_t pattern_size);

  /** Intel Unified Shared Memory operations. */
  void *(*usm_alloc) (cl_device_id dev, unsigned alloc_type,
                      cl_mem_alloc_flags_intel flags, size_t size, cl_int *errcode);
  void (*usm_free) (cl_device_id dev, void *svm_ptr);

  /* This one is separate, because the device might choose to not support it in
   * the driver. In that case, the runtime will create an event of usm_free
   * type, which has a wait on all CQs in the context. */
  void (*usm_free_blocking) (cl_device_id dev, void *svm_ptr);

  /* The following callbacks only deal with buffers (and IMAGE1D_BUFFER which
   * is backed by a buffer), not images.  */

  /* clEnqReadBuffer */
  void (*read) (void *data,
                void *__restrict__  dst_host_ptr,
                pocl_mem_identifier * src_mem_id,
                cl_mem src_buf,
                size_t offset,
                size_t size);
  /* clEnqReadBufferRect */
  void (*read_rect) (void *data, void *__restrict__ dst_host_ptr,
                     pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                     const size_t *buffer_origin, const size_t *host_origin,
                     const size_t *region, size_t buffer_row_pitch,
                     size_t buffer_slice_pitch, size_t host_row_pitch,
                     size_t host_slice_pitch);
  /* clEnqWriteBuffer */
  void (*write) (void *data,
                 const void *__restrict__  src_host_ptr,
                 pocl_mem_identifier * dst_mem_id,
                 cl_mem dst_buf,
                 size_t offset,
                 size_t size);
  /* clEnqWriteBufferRect */
  void (*write_rect) (void *data, const void *__restrict__ src_host_ptr,
                      pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                      const size_t *buffer_origin, const size_t *host_origin,
                      const size_t *region, size_t buffer_row_pitch,
                      size_t buffer_slice_pitch, size_t host_row_pitch,
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

  /* clEnqCopyBuffer with the cl_pocl_content_size extension. This callback is optional */
  void (*copy_with_size) (void *data,
                          pocl_mem_identifier *dst_mem_id,
                          cl_mem dst_buf,
                          pocl_mem_identifier *src_mem_id,
                          cl_mem src_buf,
                          pocl_mem_identifier *content_size_buf_mem_id,
                          cl_mem content_size_buf,
                          size_t dst_offset,
                          size_t src_offset,
                          size_t size);

  /* clEnqFillBuffer */
  void (*memfill) (void *data,
                   pocl_mem_identifier * dst_mem_id,
                   cl_mem dst_buf,
                   size_t size,
                   size_t offset,
                   const void *__restrict__  pattern,
                   size_t pattern_size);

  /**
   * Maps 'size' bytes of device global memory at  + offset to
   * host-accessible memory.
   *
   * This might or might not involve copying the block from the device. */
  cl_int (*map_mem) (void *data,
                     pocl_mem_identifier * src_mem_id,
                     cl_mem src_buf,
                     mem_mapping_t *map);

  cl_int (*unmap_mem) (void *data,
                       pocl_mem_identifier * dst_mem_id,
                       cl_mem dst_buf,
                       mem_mapping_t *map);

  /** These don't actually do the mapping, only return a pointer
   * where the driver will map in future.
   *
   * Separate API from map/unmap
   * because 1) unlike other driver ops, this is called from the user thread,
   * so it can be called in parallel with map/unmap or any command executing
   * in the driver; 2) most drivers can share the code for these */
  cl_int (*get_mapping_ptr) (void *data, pocl_mem_identifier *mem_id,
                             cl_mem mem, mem_mapping_t *map);
  cl_int (*free_mapping_ptr) (void *data, pocl_mem_identifier *mem_id,
                              cl_mem mem, mem_mapping_t *map);

  /****** Kernel build/management APIs. */

  /** Optional: If the driver needs to do something at kernel create/destroy
   * time. */
  int (*create_kernel) (cl_device_id device, cl_program program,
                        cl_kernel kernel, unsigned program_device_i);
  int (*free_kernel) (cl_device_id device, cl_program program,
                      cl_kernel kernel, unsigned program_device_i);

  /** Program building callbacks. */
  int (*build_source) (
      cl_program program, cl_uint device_i,

      /* these are filled by clCompileProgram(), otherwise NULLs */
      cl_uint num_input_headers, const cl_program *input_headers,
      const char **header_include_names,

      /* 1 = compile & link, 0 = compile only, linked later via clLinkProgram*/
      int link_program);

  int (*build_binary) (
      cl_program program, cl_uint device_i,

      /* 1 = compile & link, 0 = compile only, linked later via clLinkProgram*/
      int link_program, int spir_build);

  /** Build a program with builtin kernels. */
  int (*build_builtin) (cl_program program, cl_uint device_i);

  /* build a program with defined builtin kernels (DBKs). */
  int (*build_defined_builtin) (cl_program program, cl_uint device_i);

  int (*link_program) (cl_program program, cl_uint device_i,

                       cl_uint num_input_programs,
                       const cl_program *input_programs,

                       /* 1 = create library, 0 = create executable*/
                       int create_library);

  /** Optional: Called after build/link and after metadata setup. */
  int (*post_build_program) (cl_program program, cl_uint device_i);

  /** Optional: Ensures that everything is built for returning a poclbinary
   * to the user.
   *
   * E.g. for CPU driver this means building a dynamic WG sized parallel.bc */
  int (*build_poclbinary) (cl_program program, cl_uint device_i);

  /** Optional: If the driver uses the default build_poclbinary implementation
   * from common_driver.c, that implementation calls this to compile a
   * "dynamic WG size" kernel. */
  int (*compile_kernel) (_cl_command_node *cmd,
                         cl_kernel kernel,
                         cl_device_id device,
                         int specialize);

  /** Optional: If the target can utilize the basic Clang-driven steps for
   * other compilation steps, but the final linkage step, this function can be
   * used to define them.
   *
   * \param final_binary The target filename for the finalized binary.
   * \param wg_func_obj The binary for the generated work-group function.
   * \return Non-zero on error.
   */
  int (*finalize_binary) (const char *final_binary, const char *wg_func_obj);

  /** Optional: The driver should free the content of "program->data" here,
   * if it fills it. */
  int (*free_program) (cl_device_id device, cl_program program,
                       unsigned program_device_i);

  /**
   * The driver should setup kernel metadata here, if it can, and return
   * non-zero on success.
   *
   * This is called after compilation/build/link. E.g. CPU
   * driver parses the LLVM metadata. */
  int (*setup_metadata) (cl_device_id device, cl_program program,
                         unsigned program_device_i);

  /** The driver should examine the binary and return non-zero if it can load
   * it.
   *
   * Note that it's never called with pocl-binaries; those are
   * automatically accepted if device-hash in the binary's header matches the
   * device. */
  int (*supports_binary) (cl_device_id device, const size_t length,
                          const char *binary);

  /* determine DefinedBuiltinKernel support. Driver should examine the
   * kernel_id and the content of kernel_attributes and return CL_SUCCESS
   * if it supports the required kernel+attributes combination. If it does not,
   * it should return an error indicating which attribute is the problem.
   * Note: the attributes have been already validated by runtime at this point
   */
  int (*supports_dbk) (cl_device_id device,
                       cl_dbk_id_exp kernel_id,
                       const void *kernel_attributes);

  /** Optional: If the driver needs to use hardware resources
   * for command queues, it should use these callbacks */
  int (*init_queue) (cl_device_id device, cl_command_queue queue);
  int (*free_queue) (cl_device_id device, cl_command_queue queue);

  /** Optional: If the driver needs to use per-context resources,
   * it should use these callbacks for management. */
  int (*init_context) (cl_device_id device, cl_context context);
  int (*free_context) (cl_device_id device, cl_context context);

  /* clEnqueueNDRangeKernel */
  void (*run) (void *data, _cl_command_node *cmd);

  /* For clEnqueueNativeKernel. May be NULL. */
  void (*run_native) (void *data, _cl_command_node *cmd);

  /** Perform initialization steps and can return additional
   * build options that are required for the device.
   *
   * The caller owns the returned string. May be NULL. */
  char* (*init_build) (void *data);

  /**
   * Optional: Called from the LLVM-based work-group function builder to
   * initialize the LLVM TargetMachine, if needed. */
  void (*init_target_machine) (void *data, void *target_machine);

  /** Returns a hash string that should uniquely identify the device.
   *
   * This string is used when writing/loading pocl binaries to decide
   * compatibility.
   */
  char* (*build_hash) (cl_device_id device);

  /****** Image-related APIs (all optional). */

  /* The following callbacks deal with images ONLY, with the exception of
   * IMAGE1D_BUFFER type (which is implemented as a buffer).
   * If the device does not support images, all of these may be NULL. */

  /** Creates/frees a device-specific hardware resource for sampler. May be
   * NULL
   */
  int (*create_sampler) (cl_device_id device,
                         cl_sampler samp,
                         unsigned context_device_i);
  int (*free_sampler) (cl_device_id device,
                       cl_sampler samp,
                       unsigned context_device_i);

  /** Copies image to image, on the same device (or same global memory). */
  cl_int (*copy_image_rect) (void *data,
                             cl_mem src_image,
                             cl_mem dst_image,
                             pocl_mem_identifier *src_mem_id,
                             pocl_mem_identifier *dst_mem_id,
                             const size_t *src_origin,
                             const size_t *dst_origin,
                             const size_t *region);

  /** Copies a region from host OR device buffer to device image.
   *
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

  /** Copies a region from device image to host or device buffer
   *
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

  /** Maps the entire image from device to host. */
  cl_int (*map_image) (void *data,
                       pocl_mem_identifier *mem_id,
                       cl_mem src_image,
                       mem_mapping_t *map);

  /** Unmaps the entire image from host to device. */
  cl_int (*unmap_image) (void *data,
                         pocl_mem_identifier *mem_id,
                         cl_mem dst_image,
                         mem_mapping_t *map);

  /** Fill image with pattern. */
  cl_int (*fill_image) (void *data, cl_mem image, pocl_mem_identifier *mem_id,
                        const size_t *origin, const size_t *region,
                        cl_uint4 orig_pixel, pixel_t fill_pixel,
                        size_t pixel_size);

  /****** Custom device functionality APIs (all optional). */

  /**
   * The device can override this function to perform driver-specific
   * optimizations to the local size dimensions, whenever the decision
   * is left to the runtime.
   *
   * @param max_group_size The maximum total size of the WG; either
   *        device->max_work_group_size, or if present:
   *        kernel->meta->max_workgroup_size[device_i] (this can be smaller)
   * */
  void (*compute_local_size) (cl_device_id dev,
                              cl_kernel kernel,
                              unsigned device_i,
                              size_t max_group_size,
                              size_t global_x,
                              size_t global_y,
                              size_t global_z,
                              size_t *local_x,
                              size_t *local_y,
                              size_t *local_z);

  /* verifies that the device can run the requested WG sizes/offsets.
   * better to do this at enqueueNDRange time, than handling
   * the error later in the driver */
  int (*verify_ndrange_sizes) (const size_t *global_work_offset,
                               const size_t *global_work_size,
                               const size_t *local_work_size);

  /** If the device implements an extension that introduces new
   * clGetDeviceInfo() types, it can override this function. */
  cl_int (*get_device_info_ext) (cl_device_id dev,
                                 cl_device_info param_name,
                                 size_t param_value_size,
                                 void *param_value,
                                 size_t *param_value_size_ret);

  cl_int (*get_subgroup_info_ext) (cl_device_id dev,
                                   cl_kernel kernel,
                                   unsigned program_device_i,
                                   cl_kernel_sub_group_info param_name,
                                   size_t input_value_size,
                                   const void *input_value,
                                   size_t param_value_size,
                                   void *param_value,
                                   size_t *param_value_size_ret);

  /** If the device implements an extension to the clSetKernelExecInfo,
   * it can override this function. */
  cl_int (*set_kernel_exec_info_ext) (cl_device_id dev,
                                      unsigned program_device_i,
                                      cl_kernel kernel,
                                      cl_uint param_name,
                                      size_t param_value_size,
                                      const void *param_value);

  /** Optional: Returns synchronized Device & Host timestamps. */
  cl_int (*get_synchronized_timestamps) (cl_device_id dev,
                                         cl_ulong *dev_timestamp,
                                         cl_ulong *host_timestamp);

  /** Return CL_SUCCESS if the device can be, or is associated with
   * the GL context described in properties. */
  cl_int (*get_gl_context_assoc) (cl_device_id device, cl_gl_context_info type,
                                  const cl_context_properties *properties);

  /****** cl_khr_command_buffer extension APIs (all optional). */

  cl_int (*create_finalized_command_buffer) (
      cl_device_id device, cl_command_buffer_khr command_buffer);

  cl_int (*free_command_buffer) (cl_device_id device,
                                 cl_command_buffer_khr command_buffer);

};

typedef struct pocl_global_mem_t {
  cl_ulong max_ever_allocated;
  cl_ulong currently_allocated;
  cl_ulong total_alloc_limit;
} pocl_global_mem_t;

#define NUM_OPENCL_IMAGE_TYPES 6

/* typedef for intrinsics replacement function. TODO this is a hack,
 * remove when OpenASIP has memcpy implemented.*/
typedef const char *(*llvm_intrin_replace_fn) (const char *intrin_name,
                                               size_t size);

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
   * with calls to pocl_printf_alloc_stub and recursively change functions to
   * add printf buffer arguments from pocl_context.
   * Currently the pthread/basic devices require this; other devices
   * implement printf their own way. */
  int device_side_printf;
  size_t max_work_item_sizes[3];
  size_t max_work_group_size;
  size_t preferred_wg_size_multiple;
  cl_bool non_uniform_work_group_support;
  cl_bool generic_as_support;
  cl_bool wg_collective_func_support;
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
  cl_device_fp_atomic_capabilities_ext single_fp_atomic_caps;
  cl_device_fp_atomic_capabilities_ext half_fp_atomic_caps;
  cl_device_fp_atomic_capabilities_ext double_fp_atomic_caps;
  cl_device_integer_dot_product_capabilities_khr dot_product_caps;
  cl_device_integer_dot_product_acceleration_properties_khr
    dot_product_accel_props_8bit;
  cl_device_integer_dot_product_acceleration_properties_khr
    dot_product_accel_props_4x8bit;
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
  cl_bool *available;
  cl_bool compiler_available;
  cl_bool linker_available;
  /* Is the target a Single Program Multiple Data machine? If not,
     we need to generate work-item loops to execute all the work-items
     in the WG. For SPMD machines, the hardware spawns the WIs. */
  cl_bool spmd;

  /**
   * The device uses an HSA-like kernel ABI with a single argument buffer as
   * an input.
   */
  cl_bool arg_buffer_launcher;

  /**
   * The device uses a grid launcher which iterates through the whole
   * index space.
   */
  cl_bool grid_launcher;

  /* The Workgroup pass creates launcher functions and replaces work-item
     placeholder global variables (e.g. _local_size_, _global_offset_ etc) with
     loads from the context struct passed as a kernel argument. This flag
     enables or disables this pass. */
  cl_bool run_workgroup_pass;
  /* The program scope variable pass takes program-scope variables and replaces
     them by references into a buffer, and creates an initializer kernel. */
  cl_bool run_program_scope_variables_pass;
  /* Some architectures (x86) trap when encountering undefined behavior (UB)
   * of div/rem, and have no way to disable this behavior. ARM has optional
   * trapping, and RISC-V has mandatory non-trapping. OpenCL explicitly
   * forbids raising exceptions on division for any values that trigger UB.
   * This pass adds checks of input operands to div/rem so that UB is
   * never triggered. */
  cl_bool run_sanitize_divrem_pass;

  /* If CL_TRUE, pocl_llvm_build_program will ignore pocl's OpenCL headers
   * that perform built-in renames during OpenCL C build and relies on
   * Clang's OpenCL header augmented with extra declarations in
   * _clang_opencl.h. For most drivers, this should default to CL_FALSE. */
  cl_bool use_only_clang_opencl_headers;
  /* device supports command buffer execution natively, meaning pocl does
   * not need to split the command buffer into individual commands in the
   * clEnqueueCommandBuffer time. */
  cl_bool native_command_buffers;
  cl_device_exec_capabilities execution_capabilities;
  cl_platform_id platform;
  cl_uint max_sub_devices;
  size_t num_partition_properties;
  cl_device_partition_property *partition_properties;
  size_t num_partition_types;
  cl_device_partition_property *partition_type;
  size_t printf_buffer_size;
  const char *short_name;
  const char *long_name;

  const char *vendor;
  const char *driver_version;
  const char *profile;
  const char *extensions;

  /* these are Device versions, not OpenCL C versions */
  const char *version;
  unsigned version_as_int;  /* e.g. 200 */
  cl_version version_as_cl; /* cl_version format */

  /* highest OpenCL C version supported by the compiler */
  const char *opencl_c_version_as_opt;
  cl_version opencl_c_version_as_cl;
  /* Holds data specifically unique to each device type,
   * needed for internal device functions. */
  void *data;

  const char* llvm_target_triplet; /* the llvm target triplet to use */
  const char* kernellib_name;      /* bitcode kernel library name */
  const char* kernellib_fallback_name; /* bitcode kernel fallback library name */
  const char* kernellib_subdir; /* bitcode kernel library subdir */
  const char* llvm_cpu; /* the llvm CPU variant to use */
  const char *llvm_abi; /* the ABI to use */
  const char* llvm_fp_contract_mode; /* the floating point contract mde to use */
  /* function to replace intrinsic at linking stage */
  llvm_intrin_replace_fn llvm_intrin_replace;

  /* A running number (starting from zero) across all the device instances.
     Used for indexing arrays in data structures with device specific
     entries. */
  int dev_id;
  /* Identifier for a physical device global memory. */
  int global_mem_id;
  /* Pointer to an accounting struct for global memory */
  pocl_global_mem_t *global_memory;
  /* Does the device have 64bit longs */
  int has_64bit_long;
  /* Does the device set the event times in update_event() callback ?
   * if zero, the default event change handlers set the event times based on
   * the host's system time (pocl_gettimemono_ns). */
  int has_own_timer;

  /* whether this device supports OpenGL / EGL interop */
  int has_gl_interop;

  /* Convert automatic local variables to kernel arguments? */
  pocl_autolocals_to_args_strategy autolocals_to_args;
  /* Allocate local buffers device side in the work-group launcher instead of
     having a disjoint physical local memory per work-group or having the
     runtime/driver allocate the local space. */
  int device_alloca_locals;

  /* Optional property. If the device uses stack for work-item context data and
   * has limited stack size, this property can be used to guide the work-group
   * size computation selection to take the stack size in account. It should
   * be set to the maximum number of bytes that can be stored on the stack.
   *
   * If the property is zero, the work-group size computation is only limited
   * by device's max_work_group_size property, and PoCL assumes the device's
   * compiler can always handle the work-item context data correctly. */
  size_t work_group_stack_size;

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

  /* semicolon separated list of builtin kernels*/
  char *builtin_kernel_list;
  unsigned num_builtin_kernels;
  /* relative path to file with OpenCL sources of the builtin kernels */
  const char* builtins_sources_path;

  /* list of extra filenames or directories to serialize, from
   * the program's directory in pocl cache. By default
   * "program.bc" is serialized so that shouldn't be included here.
   * Useful for adding extra files not related to any particular kernel
   * (which have their own subdirectories in program's cache dir). */
  const char **serialize_entries;
  unsigned num_serialize_entries;

  /* The target specific IDs for the different OpenCL address spaces. */
  unsigned global_as_id;
  unsigned local_as_id;
  unsigned constant_as_id;

  /* optional device property for devices using PoCL's LLVM stack.
   * if nonzero, functions with arguments larger than
   * this will be force-inlined.
   */
  unsigned native_vector_width_in_bits;

  /* The address space where the argument data is passed. */
  unsigned args_as_id;

  /* The address space where the grid context data is passed. */
  unsigned context_as_id;

  /* Set to >0 if the device supports SVM.
   * When creating context with multiple devices, the device with
   * largest priority will have the responsibility of allocating
   * shared buffers residing in Shared Virtual Memory areas.
   * This allows using both CPU and HSA for SVM allocations,
   * with HSA having priority in multi-device context */
  cl_uint svm_allocation_priority;

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

  cl_uint max_num_sub_groups;
  cl_bool sub_group_independent_forward_progress;

  /* image formats supported by the device, per image type */
  const cl_image_format *image_formats[NUM_OPENCL_IMAGE_TYPES];
  cl_uint num_image_formats[NUM_OPENCL_IMAGE_TYPES];

  /* Device operations, shared among devices of the same type */
  struct pocl_device_ops *ops;

  /* cl_khr_il_program / CL_DEVICE_IL_VERSION, this only includes SPIR-V
   * NOTE: this must be kept ordered from highest to lowest version */
  const char *supported_spir_v_versions;
  /* list of supported SPIRV extensions in the form:
   * +SPV_KHR_linkonce_odr,+SPV_KHR_shader_clock,... */
  const char *supported_spirv_extensions;

  /* OpenCL 3.0 properties */

  /* list of compiler features, e.g. __opencl_c_fp64 */
  const char *features;

  cl_device_atomic_capabilities atomic_memory_capabilities;
  cl_device_atomic_capabilities atomic_fence_capabilities;

  const char *version_of_latest_passed_cts;

  cl_bool pipe_support;

  /* extensions as listed in device->extensions,
   * but with version */
  size_t num_extensions_with_version;
  const cl_name_version *extensions_with_version;

  /* ILs and their versions supported by
   * the compiler of the device */
  size_t num_ils_with_version;
  const cl_name_version *ils_with_version;

  /* list of builtin kernels as in device->builtin_kernel_list,
   * but with their versions */
  cl_name_version *builtin_kernels_with_version;

  /* OpenCL C language versions supported by the device compiler */
  size_t num_opencl_c_with_version;
  const cl_name_version *opencl_c_with_version;

  /* OpenCL C features supported by the device compiler */
  size_t num_opencl_features_with_version;
  const cl_name_version *opencl_features_with_version;

  /* cl_khr_spirv_queries */
  size_t num_spirv_extended_instruction_sets;
  const char **spirv_extended_instruction_sets;

  size_t num_spirv_extensions;
  const char **spirv_extensions;

  size_t num_spirv_capabilities;
  const cl_uint *spirv_capabilities;

  /* cl_intel_unified_shared_memory */
  cl_device_unified_shared_memory_capabilities_intel host_usm_capabs;
  cl_device_unified_shared_memory_capabilities_intel device_usm_capabs;
  cl_device_unified_shared_memory_capabilities_intel single_shared_usm_capabs;
  cl_device_unified_shared_memory_capabilities_intel cross_shared_usm_capabs;
  cl_device_unified_shared_memory_capabilities_intel system_shared_usm_capabs;

  // cl_khr_device_uuid
  cl_uchar device_uuid[CL_UUID_SIZE_KHR];
  cl_uchar driver_uuid[CL_UUID_SIZE_KHR];
  cl_bool luid_is_valid;
  cl_uint device_node_mask;
  cl_uchar device_luid[CL_LUID_SIZE_KHR];

  /* cl_khr_pci_bus_info */
  cl_device_pci_bus_info_khr pci_bus_info;

  /* command buffer related properties */
  cl_mutable_dispatch_fields_khr cmdbuf_mutable_dispatch_capabilities;
  cl_command_queue_properties cmdbuf_supported_properties;
  cl_command_queue_properties cmdbuf_required_properties;
  cl_device_command_buffer_capabilities_khr cmdbuf_capabilities;

  struct _cl_device_id *next;
};

#define DEVICE_SVM_FINEGR(dev) (dev->svm_caps & (CL_DEVICE_SVM_FINE_GRAIN_BUFFER \
                                              | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM))
#define DEVICE_SVM_ATOM(dev) (dev->svm_caps & CL_DEVICE_SVM_ATOMICS)

#define DEVICE_IS_SVM_CAPABLE(dev)                                            \
  (dev->svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)

#define DEVICE_MMAP_IS_NOP(dev) (DEVICE_SVM_FINEGR(dev) && DEVICE_SVM_ATOM(dev))

#define CHECK_DEVICE_AVAIL_RET(dev)                                           \
  if (*(dev->available) == CL_FALSE)                                          \
    {                                                                         \
      POCL_MSG_ERR ("This cl_device is not available.\n");                    \
      return CL_INVALID_DEVICE;                                               \
    }
#define CHECK_DEVICE_AVAIL_RETV(dev)                                          \
  if (*(dev->available) == CL_FALSE)                                          \
    {                                                                         \
      POCL_MSG_ERR ("This cl_device is not available.\n");                    \
      return;                                                                 \
    }

#define OPENCL_MAX_DIMENSION 3
#ifndef HAVE_SIZE_T_3
#define HAVE_SIZE_T_3
typedef struct
{
  size_t size[3];
} size_t_3;
#endif

struct _cl_platform_id {
  POCL_ICD_OBJECT_PLATFORM_ID
};

typedef struct _context_destructor_callback context_destructor_callback_t;
struct _context_destructor_callback
{
  void (CL_CALLBACK *pfn_notify) (cl_context, void *);
  void *user_data;
  context_destructor_callback_t *next;
};

struct _cl_context {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  /* queries */
  cl_device_id *devices;
  cl_context_properties *properties;
  cl_bool gl_interop;
  /* implementation */
  unsigned num_devices;
  unsigned num_properties;

  /* the original device list given to clCreateContext,
   * required for */
  cl_device_id *create_devices;
  unsigned num_create_devices;
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

  /* The device that should allocate USM memory
   * NULL if none of devices in the context is USM capable */
  cl_device_id usm_allocdev;

  /* for enqueueing migration commands. Two reasons:
   * 1) since migration commands can execute in parallel
   * to other commands, we can increase parallelism
   * 2) in some cases (migration between 2 devices through
   * host memory), we need to put two commands in two queues,
   * and the clEnqueueX only gives us one (on the destination
   * device). */
  cl_command_queue *default_queues;

  /* The minimal required buffer alignment for all devices in the context.
   * E.g. for clCreateSubBuffer:
   * CL_MISALIGNED_SUB_BUFFER_OFFSET is returned in errcode_ret if there are no
   * devices in context associated with buffer for which the origin value
   * is aligned to the CL_DEVICE_MEM_BASE_ADDR_ALIGN value.
   */
  size_t min_buffer_alignment;

  /* List of destructor callbacks */
  context_destructor_callback_t *destructor_callbacks;

  /* List of allocations with raw host-side accessible pointers associated
   * with them (SVM, USM, DEV). */
  pocl_raw_ptr_set *raw_ptrs;

  /* list of command queues created for the context.
   * required for clMemBlockingFreeINTEL */
  struct _cl_command_queue *command_queues;

  /* The maximum of CL_DEVICE_MEM_BASE_ADDR_ALIGN across the devices in the
   * context. */
  cl_uint mem_base_addr_align;

  /* True if none of devices support cl_ext_buffer_device_address */
  cl_bool no_devices_support_bda;

#ifdef ENABLE_LLVM
  void *llvm_context_data;
#endif

#ifdef BUILD_PROXY
  cl_context proxied_context;
#endif
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

  /* Events of the enqueued commands in enqueue order. */
  cl_event events;
  struct _cl_event *barrier;

  /* Number of unfinished command enqueued. */
  unsigned long command_count;

  /* The event of the last command pushed to the queue. */
  pocl_data_sync_item last_event;

  cl_queue_properties queue_properties[10];
  unsigned num_queue_properties;

  cl_queue_priority_khr priority;
  cl_queue_throttle_khr throttle;

  /* number of user threads (via clFinish) awaiting
   * cmd-queue-finished notification (via ops->notify_cmdq_finished) */
  unsigned notification_waiting_threads;

  /* device specific data */
  void *data;

  /* list of CQs stored in cl_context */
  struct _cl_command_queue *prev, *next;
};

struct _cl_command_buffer_khr
{
  POCL_ICD_OBJECT
  POCL_OBJECT;
  pocl_lock_t mutex;

  /** Queues that this command buffer was created for */
  cl_command_queue *queues;
  cl_uint num_queues;

  /** Helper flag indicating whether the queues of this command buffer belong
   * to different devices */
  cl_int is_multi_device;

  /** List of flags that this command buffer was created with */
  cl_uint num_properties;
  cl_command_buffer_properties_khr *properties;

  /** recording / ready / pending (executing) / invalid */
  cl_command_buffer_state_khr state;
  /** Number of currently in-flight instances of this command buffer */
  cl_uint pending;

  /** Number of currently allocated sync points in this command buffer.
   * Used for generating the next sync point id and for validating sync point
   * wait lists when recording commands. */
  cl_uint num_syncpoints;

  _cl_command_node *cmds;
  cl_bool is_mutable;
  cl_bool assert_no_more_wgs;
  /* device-specific data */
  void **data;

  /** List of mem objects that have to be migrated before the buffer can be
   * safely run. Does not account for migrations that need to happen between
   * commands in the same buffer. */
  pocl_buffer_migration_info *migr_infos;
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

#define DEVICE_IMAGE_SIZE_SUPPORT 1
#define DEVICE_IMAGE_FORMAT_SUPPORT 2
#define DEVICE_IMAGE_INTEROP_SUPPORT 4

#define DEVICE_DOESNT_SUPPORT_IMAGE(mem, dev_i)                               \
  (mem->device_supports_this_image[dev_i]                                     \
   != (DEVICE_IMAGE_SIZE_SUPPORT | DEVICE_IMAGE_FORMAT_SUPPORT                \
       | DEVICE_IMAGE_INTEROP_SUPPORT))

#define POCL_ON_UNSUPPORTED_IMAGE(mem, dev, operation)                        \
  do                                                                          \
    {                                                                         \
      unsigned dev_i;                                                         \
      for (dev_i = 0; dev_i < mem->context->num_devices; ++dev_i)             \
        if (mem->context->devices[dev_i] == dev)                              \
          break;                                                              \
      assert (dev_i < mem->context->num_devices);                             \
      operation ((mem->context->devices[dev_i]->image_support == CL_FALSE),   \
                 CL_INVALID_OPERATION, "Device %s does not support images\n", \
                 mem->context->devices[dev_i]->long_name);                    \
      operation (((mem->device_supports_this_image[dev_i]                     \
                   & DEVICE_IMAGE_FORMAT_SUPPORT)                             \
                  == 0),                                                      \
                 CL_IMAGE_FORMAT_NOT_SUPPORTED,                               \
                 "The image type is not supported by this device\n");         \
      operation (((mem->device_supports_this_image[dev_i]                     \
                   & DEVICE_IMAGE_SIZE_SUPPORT)                               \
                  == 0),                                                      \
                 CL_INVALID_IMAGE_SIZE,                                       \
                 "The image size is not supported by this device\n");         \
      operation (                                                             \
          ((mem->device_supports_this_image[dev_i]                            \
            & DEVICE_IMAGE_INTEROP_SUPPORT)                                   \
           == 0),                                                             \
          CL_INVALID_GL_OBJECT,                                               \
          "OpenGL/EGL/other interop is not supported by this device\n");      \
    }                                                                         \
  while (0)


#define POCL_RETURN_ON_UNSUPPORTED_IMAGE(mem, dev)                            \
  POCL_ON_UNSUPPORTED_IMAGE(mem, dev, POCL_RETURN_ERROR_ON)

#define POCL_GOTO_ON_UNSUPPORTED_IMAGE(mem, dev)                              \
  POCL_ON_UNSUPPORTED_IMAGE(mem, dev, POCL_GOTO_ERROR_ON)

typedef struct _cl_mem_list_item_t cl_mem_list_item_t;

typedef struct _cl_mem cl_mem_t;
struct _cl_mem {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  cl_context context;
  cl_mem_object_type type;
  cl_mem_flags flags;

  cl_mem_properties properties[5];
  unsigned num_properties;

  size_t size;

  /* Sub-buffer related data: */

  /* In case this is a sub-buffer, this points to the parent buffer. */
  cl_mem_t *parent;
  /* The starting offset to the parent buffer. */
  size_t origin;

  /* For utlist.h linked lists. */
  struct _cl_mem *prev, *next;

  /* The sub-buffers of this buffer, if any. */
  cl_mem_list_item_t *sub_buffers;

  /* The implicit sub-buffers covering the empty region.  These will be added
     on-demand when optimizing the migrations. */
  cl_mem_list_item_t *implicit_sub_buffers;

  /* Set to 1 for implicit sub-buffers, to differentiate from user-defined
     sub-buffers. */
  int implicit_sub_buffer;

  /* cl_pocl_content_size: If set to nonzero, it defines the size of the
     defined content in bytes, which can be used to avoid transferring
     untouched data when migrating / reading buffers. */
  size_t content_size;

  /** The host backing memory for a buffer.
   *
   * This is either a user-provided host pointer, a driver-allocated,
   * or a temporary allocation by a migration command. Since it
   * can have multiple users, it's refcounted. */
  void *mem_host_ptr;

  /** The version of the buffer content in mem_host_ptr. */
  uint64_t mem_host_ptr_version;

  /* reference count; when it reaches 0,
   * the mem_host_ptr is automatically freed */
  uint mem_host_ptr_refcount;
  int mem_host_ptr_is_svm;

  /* Array of device-specific memory allocation bookkeeping structs.
     The location of some device's struct is determined by
     the device's global_mem_id. */
  pocl_mem_identifier *device_ptrs;

  /* For content tracking:
   *
   * This is the valid (highest) version of the buffer's content.
   * If any device has a lower version in its device copy,
   * the buffer content on that device is invalid and should be
   * updated before used. Sub-buffers' latest_version follows the parent's:
   * when the parent buffer is updated by a command, all its sub-buffers
   * get their latest_version synchronized to the parent version. */
  uint64_t latest_version;

  /* The event (denotes a command here) that last wrote to the buffer,
   * this is used as the dependency source for migration commands. */
  cl_event last_updater;

  /* A linked list of regions of the buffer mapped to the
     host memory */
  mem_mapping_t *mappings;
  size_t map_count;

  /* A linked list of destructor callbacks */
  mem_destructor_callback_t *destructor_callbacks;

  /* These two are for cl_pocl_content_size extension.
   * They link two buffers together, like this:
   * mem->size_buffer->content_buffer = mem
   * mem->content_buffer->size_buffer = mem
   */
  cl_mem size_buffer;
  cl_mem content_buffer;

  /* OpenGL data */
  cl_GLenum               target;
  cl_GLint                miplevel;
  cl_GLuint               texture;
  CLeglDisplayKHR egl_display;
  CLeglImageKHR egl_image;

  /* for images, a flag for each device in context,
   * whether that device supports this */
  int *device_supports_this_image;

  /* if the memory backing mem_host_ptr is "permanent" =
   * valid through the entire lifetime of the buffer,
   * we can make some assumptions and optimizations */
  cl_bool mem_host_ptr_is_permanent;

  /* If the allocation was requested to have a permanent device global memory
     address (until freed). This is set via CL_MEM_DEVICE_ADDRESS flag of
     the cl_ext_buffer_device_address extension. */
  cl_bool has_device_address;

  /* Image flags */
  cl_bool                 is_image;
  cl_bool                 is_gl_texture;
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
  /* This points to the backing storage cl_mem in case of images. */
  cl_mem                  buffer;
  cl_uint                 is_gl_acquired;

  /* pipe flags */
  cl_bool                 is_pipe;
  size_t                  pipe_packet_size;
  size_t                  pipe_max_packets;

  /* Tensor Properties */
  cl_uint tensor_rank;
  cl_tensor_shape_exp tensor_shape[CL_MEM_MAX_TENSOR_RANK_EXP];
  cl_tensor_datatype_exp tensor_dtype;
  cl_tensor_layout_type_exp tensor_layout_type;
  void *tensor_layout;
  // properties
  char is_tensor;
};

/** Returns the backing store cl_mem for an image, otherwise the cl_mem
    itself (for regular buffers). */
#define POCL_MEM_BS(BUF) (BUF->buffer != NULL ? BUF->buffer : BUF)

struct _cl_mem_list_item_t
{
  cl_mem mem;
  cl_mem_list_item_t *prev, *next;
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
  size_t wg_size_hint[OPENCL_MAX_DIMENSION];
  char vectypehint[16];

  /* if we know the size of _every_ kernel argument, we store
   * the total size here. see struct _cl_kernel on why */
  size_t total_argument_storage_size;

  /****** subgroups *******/
  /* per-device value for CL_KERNEL_MAX_NUM_SUB_GROUPS */
  size_t *max_subgroups;
  /* per-device value for CL_KERNEL_COMPILE_NUM_SUB_GROUPS */
  size_t *compile_subgroups;

  /****** workgroups *******/
  /* per-device value for CL_KERNEL_WORK_GROUP_SIZE */
  size_t *max_workgroup_size;
  /* per-device value for CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE */
  size_t *preferred_wg_multiple;
  /* per-device value for CL_KERNEL_LOCAL_MEM_SIZE */
  cl_ulong *local_mem_size;
  /* per-device value for CL_KERNEL_PRIVATE_MEM_SIZE */
  cl_ulong *private_mem_size;
  /* per-device value for CL_KERNEL_SPILL_MEM_SIZE_INTEL */
  cl_ulong *spill_mem_size;

  /* per-device array of hashes */
  pocl_kernel_hash_t *build_hash;

  /* enum cl_dbk_id_exp */
  unsigned builtin_kernel_id;
  /* only for defined builtin kernels */
  void *builtin_kernel_attrs;
  /* maximum global work size usable with the kernel.
   * Only applies to builtin kernels */
  size_t_3 builtin_max_global_work;

  /* device-specific METAdata, void* array[program->num_devices] */
  void **data;
} pocl_kernel_metadata_t;

#define MAIN_PROGRAM_LOG_SIZE 6400

struct _cl_program {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  /* queries */
  cl_context context;
  /* -cl-denorms-are-zero build option */
  unsigned flush_denorms;

  /* list of devices "associated with the program" (quote from Specs)
   * ... IOW for which we *can* build the program.
   * this is setup once, at clCreateProgramWith{Source,Binaries,...} time */
  cl_device_id *associated_devices;
  cl_uint associated_num_devices;
  /* list of devices for which we actually did build the program.
   * this changes on every rebuild to device arguments given to clBuildProgram
   */
  cl_uint num_devices;
  cl_device_id *devices;

  /* all the program sources appended together, terminated with a zero */
  char *source;
  /* The options in the last clBuildProgram call for this Program. */
  char *compiler_options;

  /* per-device binaries, in device-specific format */
  size_t *binary_sizes;
  unsigned char **binaries;

  /* If this is a program with built-in kernels, this is the list of kernel
     names it contains. */
  size_t num_builtin_kernels;
  /* Names of builtin kernels, as array of char*,
   * and also as semicolon-separated string. */
  char **builtin_kernel_names;
  char *concated_builtin_names;
  // relevant only for DefinedBuiltinKernels:
  cl_dbk_id_exp *builtin_kernel_ids;
  void **builtin_kernel_attributes;

  /* Poclcc binary format.  */
  /* per-device poclbinary-format binaries.  */
  size_t *pocl_binary_sizes;
  unsigned char **pocl_binaries;
  /* device-specific data, per each device */
  void **data;

  /* kernel number and the metadata for each kernel */
  size_t num_kernels;
  pocl_kernel_metadata_t *kernel_meta;

  /* list of attached cl_kernel instances */
  cl_kernel kernels;
  /* Per-device program hash after build */
  SHA1_digest_t* build_hash;
  /* Per-device build logs, for the case when we don't yet have the program's cachedir */
  char** build_log;
  /* Per-program build log, for the case when we aren't yet building for devices */
  char main_build_log[MAIN_PROGRAM_LOG_SIZE];
  /* Use to store build status */
  cl_build_status build_status;
  /* Use to store binary type */
  cl_program_binary_type binary_type;
  /* total size of program-scope variables. This depends on alignments
   * & type sizes, hence it is device-dependent */
  size_t *global_var_total_size;
  /* per-device pointer to a llmv::Module instance;
   * optional - for devices which use PoCL's LLVM passes */
  void** llvm_irs;
  /* per-device buffers for storing program-scope vars. Allocated lazily.
   * these are not cl_mem because the Specs explicitly say these are not
   * migrated between devices. */
  void** gvar_storage;

  /* Store SPIR-V binary from clCreateProgramWithIL() */
  char *program_il;
  size_t program_il_size;
  /* for SPIR-V store also specialization constants */
  size_t num_spec_consts;
  cl_uint *spec_const_ids;
  cl_uint *spec_const_sizes;
  uint64_t *spec_const_values;
  char *spec_const_is_set;
};

typedef struct _pocl_ptr_list_node pocl_ptr_list;
struct _pocl_ptr_list_node
{
  void *ptr;
  struct _pocl_ptr_list_node *prev, *next;
};


struct _cl_kernel {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  /* -------- */
  cl_context context;
  cl_program program;
  pocl_kernel_metadata_t *meta;
  /* device-specific data, per each device. This is different from meta->data,
   * as this is per-instance of cl_kernel, while there is just one meta->data
   * for all instances of the kernel of the same name. */
  void **data;
  /* just a convenience pointer to meta->name */
  const char *name;

  /* The kernel arguments that are set with clSetKernelArg().
     These are copied to the command queue command at enqueue. */
  struct pocl_argument *dyn_arguments;

  /* if total_argument_storage_size is known, we preallocate storage for
   * actual kernel arguments here, instead of allocating it by one for
   * each argument separately. The "offsets" store pointers calculated as
   * "dyn_argument_storage + offset-of-argument-N".
   *
   * The pointer to actual value for argument N, used by drivers, is stored
   * in dyn_arguments[N].value; if total_argument_storage_size is not known,
   * the .value must be allocated separately for every argument in
   * clSetKernelArg; if it is known, clSetKernelArg sets the .value to
   * dyn_argument_offsets[N] and copies the value there.
   *
   * We must keep both ways, because not every driver can know kernel
   * argument sizes beforehand.
   */
  char *dyn_argument_storage;
  void **dyn_argument_offsets;

  /* The SVM, USM or device allocations accessed by the kernel which were set
     explicitly using clSetKernelExecInfo(). */
  pocl_ptr_list *indirect_raw_ptrs;

  /* Set to true, in case the kernel might access any of the raw buffers
     indirectly. All USM indirect access flags will set this currently.
     We should ensure at enqueue time that all of the known raw buffers
     will be synchronized to the device. */
  char can_access_all_raw_buffers_indirectly;

  /* for program's linked list of kernels */
  struct _cl_kernel *next;
};

typedef struct event_callback_item event_callback_item;
struct event_callback_item
{
  void(CL_CALLBACK *callback_function) (cl_event, cl_int, void *);
  void *user_data;
  cl_int trigger_status;
  struct event_callback_item *next;
};


struct event_node
{
  cl_event event;
  event_node *next;
};

#define MAX_EVENT_DEPS 60

/* Optional metadata for events for improved profile data readability etc. */
typedef struct _pocl_event_md
{
  /* The kernel executed by the NDRange command associated with the event,
     if any. */
  cl_kernel kernel;

  size_t num_deps;
  // event IDs on which this event depends
  uint64_t dep_ids[MAX_EVENT_DEPS];
  // the finish time of those ^^^ event IDs
  cl_ulong dep_ts[MAX_EVENT_DEPS];
} pocl_event_md;


typedef struct _cl_event _cl_event;
struct _cl_event {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  cl_context context;
  cl_command_queue queue;
  /* note: this is not necessarily the same as command->type. User events
   * do not have a *command, and in some cases the actual command is different
   * than the one enqueued by the user (e.g. SVMMemcpy can be translated
   * to other commands) */
  cl_command_type command_type;
  _cl_command_node *command;

  /* list of callback functions */
  event_callback_item *callback_list;

  /* list of devices needing completion notification of this event */
  event_node *notify_list;
  /* events this event is dependent on */
  event_node *wait_list;

  /* OoO doesn't use sync points -> put used buffers here */
  size_t num_used_buffers;
  /* Mem object arguments of the command the event is associated with. */
  /* The code seemed to assume that the mem_objs that are passed as the command
     arguments are the first ones in the list. This has not been the
     ever since the list has been sorted by the mem object id. */
  /* cl_mem mem_objs; */
  /* The buffers that should be migrated when this event/command is launched.
   */
  /* Moved to the _cl_command struct */
  /* pocl_buffer_migration_info *migrated_bufs; */

  /* Profiling data: time stamps of the different phases of execution. */
  cl_ulong time_queue;  /* the enqueue time */
  cl_ulong time_submit; /* the time the command was submitted to the device */
  cl_ulong time_start;  /* the time the command actually started executing */
  cl_ulong time_end;    /* the finish time of the command */

  /* Device specific data */
  void *data;

  /* Additional (optional data) used to make profile data more readable etc. */
  pocl_event_md *meta_data;

  /* The execution status of the command this event is monitoring. */
  cl_int status;
  /* implicit event = an event for pocl's internal use, not visible to user */
  char implicit_event;

  /* if set, at the completion of event, the mem_host_ptr_refcount should be
   * lowered and memory freed if it's 0 */
  char release_mem_host_ptr_after;

  /* for command buffers, profiling info is only available if
   * profiling is enabled on all queues of the cmdbuffer */
  char profiling_available;

  /* reset command buffer when the event is finished */
  char reset_command_buffer;
  cl_command_buffer_khr command_buffer;

  _cl_event *next;
  _cl_event *prev;
};

typedef struct _pocl_user_event_data
{
  pocl_cond_t wakeup_cond;
} pocl_user_event_data;

typedef struct _cl_sampler cl_sampler_t;
struct _cl_sampler {
  POCL_ICD_OBJECT
  POCL_OBJECT;
  cl_context context;
  cl_bool             normalized_coords;
  cl_addressing_mode  addressing_mode;
  cl_filter_mode      filter_mode;
  cl_sampler_properties properties[10];
  cl_uint             num_properties;
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
#elif defined (_WIN32)
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

#ifdef HAVE_LTTNG_UST

#include "pocl_lttng.h"

#define TP_CREATE_QUEUE(context_id, queue_id)                                 \
  tracepoint (pocl_trace, create_queue, context_id, queue_id);
#define TP_FREE_QUEUE(context_id, queue_id)                                   \
  tracepoint (pocl_trace, free_queue, context_id, queue_id);

#define TP_CREATE_BUFFER(context_id, buffer_id)                               \
  tracepoint (pocl_trace, create_buffer, context_id, buffer_id);
#define TP_FREE_BUFFER(context_id, buffer_id)                                 \
  tracepoint (pocl_trace, free_buffer, context_id, buffer_id);

#define TP_CREATE_PROGRAM(context_id, program_id)                             \
  tracepoint (pocl_trace, create_program, context_id, program_id);
#define TP_BUILD_PROGRAM(context_id, program_id)                              \
  tracepoint (pocl_trace, build_program, context_id, program_id);
#define TP_FREE_PROGRAM(context_id, program_id)                               \
  tracepoint (pocl_trace, free_program, context_id, program_id);

#define TP_CREATE_KERNEL(context_id, kernel_id, kernel_name)                  \
  tracepoint (pocl_trace, create_kernel, context_id, kernel_id, kernel_name);
#define TP_FREE_KERNEL(context_id, kernel_id, kernel_name)                    \
  tracepoint (pocl_trace, free_kernel, context_id, kernel_id, kernel_name);

#define TP_CREATE_IMAGE(context_id, image_id)                                 \
  tracepoint (pocl_trace, create_image, context_id, image_id);
#define TP_FREE_IMAGE(context_id, image_id)                                   \
  tracepoint (pocl_trace, free_image, context_id, image_id);

#define TP_CREATE_SAMPLER(context_id, sampler_id)                             \
  tracepoint (pocl_trace, create_sampler, context_id, sampler_id);
#define TP_FREE_SAMPLER(context_id, sampler_id)                               \
  tracepoint (pocl_trace, free_sampler, context_id, sampler_id);

#else

#define TP_CREATE_QUEUE(context_id, queue_id)
#define TP_FREE_QUEUE(context_id, queue_id)

#define TP_CREATE_BUFFER(context_id, buffer_id)
#define TP_FREE_BUFFER(context_id, buffer_id)

#define TP_CREATE_PROGRAM(context_id, program_id)
#define TP_BUILD_PROGRAM(context_id, program_id)
#define TP_FREE_PROGRAM(context_id, program_id)

#define TP_CREATE_KERNEL(context_id, kernel_id, kernel_name)
#define TP_FREE_KERNEL(context_id, kernel_id, kernel_name)

#define TP_CREATE_IMAGE(context_id, image_id)
#define TP_FREE_IMAGE(context_id, image_id)

#define TP_CREATE_SAMPLER(context_id, sampler_id)
#define TP_FREE_SAMPLER(context_id, sampler_id)

#endif

#endif /* POCL_CL_H */
