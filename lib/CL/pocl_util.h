/* OpenCL runtime library: pocl_util utility functions

   Copyright (c) 2012-2023 pocl developers

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

#ifndef POCL_UTIL_H
#define POCL_UTIL_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "pocl_cl.h"

#ifdef __cplusplus
extern "C" {
#endif

POCL_EXPORT
uint32_t pocl_byteswap_uint32_t (uint32_t word, char should_swap);
float byteswap_float (float word, char should_swap);

/* set rounding mode */
POCL_EXPORT
void pocl_restore_rm (unsigned rm);
/* get current rounding mode */
POCL_EXPORT
unsigned pocl_save_rm ();
/* set OpenCL's default (round to nearest) rounding mode */
POCL_EXPORT
void pocl_set_default_rm ();


/* sets the flush-denorms-to-zero flag on the CPU, if supported */
POCL_EXPORT
void pocl_set_ftz (unsigned ftz);
/* saves / restores cpu flags*/
POCL_EXPORT
unsigned pocl_save_ftz (void);
POCL_EXPORT
void pocl_restore_ftz (unsigned ftz);

void pocl_install_sigfpe_handler ();
void pocl_install_sigusr2_handler ();
#if defined(__linux__) && defined(__x86_64__)
POCL_EXPORT
void pocl_ignore_sigfpe_for_thread (pthread_t thr);
#endif

void bzero_s (void *v, size_t n);

/* Finds the next highest power of two of the given value. */
POCL_EXPORT
size_t pocl_size_ceil2 (size_t x);

POCL_EXPORT
uint64_t pocl_size_ceil2_64 (uint64_t x);

POCL_EXPORT
size_t pocl_align_value (size_t value, size_t alignment);

/* Allocates aligned blocks of memory.
 *
 * Uses posix_memalign when available. Otherwise, uses
 * malloc to allocate a block of memory on the heap of the desired
 * size which is then aligned with the given alignment. The resulting
 * pointer must be freed with a call to pocl_aligned_free. Alignment
 * must be a non-zero power of 2.
 */

POCL_EXPORT
void *pocl_aligned_malloc(size_t alignment, size_t size);
#define pocl_aligned_free(x) POCL_MEM_FREE(x)

/* locks / unlocks two events in order of their event-id.
 * This avoids any potential deadlocks of threads should
 * they try to lock events in opposite order. */
void pocl_lock_events_inorder (cl_event ev1, cl_event ev2);
void pocl_unlock_events_inorder (cl_event ev1, cl_event ev2);

/* Function for creating events */
cl_int pocl_create_event (cl_event *event, cl_command_queue command_queue,
                          cl_command_type command_type, size_t num_buffers,
                          const cl_mem* buffers, cl_context context);

cl_int pocl_create_command (_cl_command_node **cmd,
                            cl_command_queue command_queue,
                            cl_command_type command_type, cl_event *event,
                            cl_uint num_events, const cl_event *wait_list,
                            size_t num_buffers, cl_mem *buffers,
                            char *readonly_flags);

cl_int pocl_create_command_migrate (_cl_command_node **cmd,
                                    cl_command_queue command_queue,
                                    cl_mem_migration_flags flags,
                                    cl_event *event_p,
                                    cl_uint num_events,
                                    const cl_event *wait_list,
                                    size_t num_buffers,
                                    cl_mem *buffers,
                                    char *readonly_flags);

cl_int pocl_command_record (cl_command_buffer_khr command_buffer,
                            _cl_command_node *cmd,
                            cl_sync_point_khr *sync_point);

cl_int pocl_create_recorded_command (
    _cl_command_node **cmd, cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue, cl_command_type command_type,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list, size_t num_buffers,
    cl_mem *buffers, char *readonly_flags);

void pocl_command_enqueue (cl_command_queue command_queue,
                          _cl_command_node *node);

cl_int
pocl_cmdbuf_choose_recording_queue (cl_command_buffer_khr command_buffer,
                                    cl_command_queue *command_queue);

POCL_EXPORT
int pocl_alloc_or_retain_mem_host_ptr (cl_mem mem);

POCL_EXPORT
int pocl_release_mem_host_ptr (cl_mem mem);

void pocl_ndrange_node_cleanup (_cl_command_node *node);

/* does several sanity checks on buffer & given memory region */
int pocl_buffer_boundcheck(cl_mem buffer, size_t offset, size_t size);
/* same as above just 2 buffers */
int pocl_buffers_boundcheck(cl_mem src_buffer, cl_mem dst_buffer,
                            size_t src_offset, size_t dst_offset, size_t size);
/* checks for overlapping regions in buffers, including overlapping subbuffers */
int pocl_buffers_overlap(cl_mem src_buffer, cl_mem dst_buffer,
                            size_t src_offset, size_t dst_offset, size_t size);

int pocl_buffer_boundcheck_3d(const size_t buffer_size, const size_t *origin,
                              const size_t *region, size_t *row_pitch,
                              size_t *slice_pitch, const char* prefix);

/**
 * Finds an SVM/USM allocation where the host pointer is in.
 *
 * @return an allocation (info) where it is found, NULL if not found.
 */
pocl_raw_ptr *pocl_find_raw_ptr_with_vm_ptr (cl_context context,
                                             const void *host_ptr);

/**
 * Finds a cl_mem allocation where the device pointer is mapped.
 *
 * The cl_mem allocation should be allocated with CL_MEM_BUFFER_DEVICE_ADDRESS.
 *
 * @return an allocation where the device pointer is in, NULL if not found.
 */
POCL_EXPORT
pocl_raw_ptr *pocl_find_raw_ptr_with_dev_ptr (cl_context context,
                                              const void *dev_ptr);

int
check_copy_overlap(const size_t src_offset[3],
                   const size_t dst_offset[3],
                   const size_t region[3],
                   const size_t row_pitch, const size_t slice_pitch);

/**
 * Push a command into ready list if all previous events are completed or
 * in pending_list if the command still has pending dependencies
 */
POCL_EXPORT
void
pocl_command_push (_cl_command_node *node,
                   _cl_command_node **ready_list,
                   _cl_command_node **pending_list);

/**
 * Return true if a command is ready to execute (no more event in wait list
 * or false if not
 */
static inline int
pocl_command_is_ready(cl_event event)
{
  return event->wait_list == NULL;
}

typedef void (*empty_queue_callback) (cl_command_queue cq);

void pocl_cl_mem_inherit_flags (cl_mem mem, cl_mem from_buffer,
                                cl_mem_flags flags);

int pocl_setup_context (cl_context context);

/* Helpers for dealing with devices / subdevices */

cl_device_id pocl_real_dev (const cl_device_id);
cl_device_id * pocl_unique_device_list(const cl_device_id * in, cl_uint num, cl_uint *real);
int pocl_device_supports_builtin_kernel (cl_device_id dev,
                                         const char *kernel_name);

#define POCL_CHECK_DEV_IN_CMDQ                                                \
  do                                                                          \
    {                                                                         \
      device = pocl_real_dev (command_queue->device);                         \
      for (i = 0; i < command_queue->context->num_devices; ++i)               \
        {                                                                     \
          if (command_queue->context->devices[i] == device)                   \
            break;                                                            \
        }                                                                     \
      assert (i < command_queue->context->num_devices);                       \
    }                                                                         \
  while (0)

int pocl_check_event_wait_list(cl_command_queue     command_queue,
                               cl_uint              num_events_in_wait_list,
                               const cl_event *     event_wait_list);

int
pocl_check_syncpoint_wait_list (cl_command_buffer_khr command_buffer,
                                cl_uint num_sync_points_in_wait_list,
                                const cl_sync_point_khr *sync_point_wait_list);

void pocl_update_event_queued (cl_event event);

POCL_EXPORT
void pocl_update_event_submitted (cl_event event);

POCL_EXPORT
void pocl_update_event_running_unlocked (cl_event event);

POCL_EXPORT
void pocl_update_event_running (cl_event event);

POCL_EXPORT
void pocl_update_event_complete (const char *func, unsigned line,
                                 cl_event event, const char *msg);

POCL_EXPORT
int pocl_copy_event_node (_cl_command_node *dst_node,
                          _cl_command_node *src_node);

#define POCL_UPDATE_EVENT_COMPLETE_MSG(__event, msg)                          \
  pocl_update_event_complete (__func__, __LINE__, (__event), msg)

#define POCL_UPDATE_EVENT_COMPLETE(__event)                                   \
  pocl_update_event_complete (__func__, __LINE__, (__event), NULL)

POCL_EXPORT
void pocl_update_event_failed (cl_event event);

POCL_EXPORT
void pocl_update_event_device_lost (cl_event event);

const char*
pocl_status_to_str (int status);

POCL_EXPORT
const char *
pocl_command_to_str (cl_command_type cmd);

POCL_EXPORT
int
pocl_run_command(char * const *args);

POCL_EXPORT
int pocl_run_command_capture_output (char *capture_string,
                                     size_t *captured_bytes,
                                     char *const *args);

uint16_t float_to_half (float value);

float half_to_float (uint16_t value);

void pocl_free_kernel_metadata (cl_program program, unsigned kernel_i);

POCL_EXPORT
int pocl_svm_check_pointer (cl_context context, const void *svm_ptr,
                            size_t size, size_t *buffer_size);

/* returns !0 if binary is SPIR-V bitcode with OpCapability Kernel
 * OpenCL-style bitcode produced by e.g. llvm-spirv */
POCL_EXPORT
int pocl_bitcode_is_spirv_execmodel_kernel (const char *bitcode, size_t size);

/* returns !0 if binary is SPIR-V bitcode with OpCapability Shader
 * these are produced by e.g. GLSL and clspv compilation */
POCL_EXPORT
int pocl_bitcode_is_spirv_execmodel_shader (const char *bitcode, size_t size);

int pocl_device_is_associated_with_kernel (cl_device_id device,
                                           cl_kernel kernel);
POCL_EXPORT
int pocl_escape_quoted_whitespace (char *temp_options, char *replace_me);

POCL_EXPORT
int pocl_fill_aligned_buf_with_pattern (void *__restrict__ ptr, size_t offset,
                                        size_t size,
                                        const void *__restrict__ pattern,
                                        size_t pattern_size);

POCL_EXPORT
int pocl_get_private_datadir (char* private_datadir);

POCL_EXPORT
int pocl_get_srcdir_or_datadir (char* path,
                                const char* srcdir_suffix,
                                const char *datadir_suffix,
                                const char* filename);

POCL_EXPORT
void pocl_str_toupper (char *out, const char *in);

POCL_EXPORT
void pocl_str_tolower (char *out, const char *in);

#ifdef __cplusplus
}
#endif

/* Common macro for cleaning up "*GetInfo" API call implementations.
 * All the *GetInfo functions have been specified to look alike,
 * and have been implemented to use the same variable names, so this
 * code can be shared.
 */

#define POCL_RETURN_GETINFO_INNER(__SIZE__, MEMASSIGN)                        \
  do                                                                          \
    {                                                                         \
      if (param_value)                                                        \
        {                                                                     \
          POCL_RETURN_ERROR_ON (                                              \
              (param_value_size < __SIZE__), CL_INVALID_VALUE,                \
              "param_value_size (%zu) smaller than actual size (%zu)\n",      \
              param_value_size, __SIZE__);                                    \
          MEMASSIGN;                                                          \
        }                                                                     \
      if (param_value_size_ret)                                               \
        *param_value_size_ret = __SIZE__;                                     \
      return CL_SUCCESS;                                                      \
    }                                                                         \
  while (0)

/* Version for handling only the size checking and returning.
   Assigning the data and returning is left to the caller. */
#define POCL_RETURN_GETINFO_SIZE_CHECK(__SIZE__)                              \
do                                                                            \
  {                                                                           \
    if (param_value)                                                          \
      {                                                                       \
        POCL_RETURN_ERROR_ON (                                                \
            (param_value_size < __SIZE__), CL_INVALID_VALUE,                  \
            "param_value_size (%zu) smaller than actual size (%zu)\n",        \
            param_value_size, __SIZE__);                                      \
      }                                                                       \
    if (param_value_size_ret)                                                 \
      {                                                                       \
        *param_value_size_ret = __SIZE__;                                     \
        if (param_value == NULL)                                              \
          return CL_SUCCESS;                                                  \
      }                                                                       \
  }                                                                           \
while (0)

#define POCL_RETURN_GETINFO_SIZE(__SIZE__, __POINTER__)                 \
  POCL_RETURN_GETINFO_INNER(__SIZE__,                                   \
    memcpy(param_value, __POINTER__, __SIZE__))

#define POCL_RETURN_GETINFO_STR(__STR__)                                \
  do                                                                    \
    {                                                                   \
      size_t const value_size = strlen(__STR__) + 1;                    \
      POCL_RETURN_GETINFO_INNER(value_size,                             \
                  memcpy(param_value, __STR__, value_size));            \
    }                                                                   \
  while (0)

#define POCL_RETURN_GETINFO_STR_FREE(__STR__)                                 \
  do                                                                          \
    {                                                                         \
      size_t const value_size = strlen (__STR__) + 1;                         \
      if (param_value)                                                        \
        {                                                                     \
          if (param_value_size >= value_size)                                 \
            memcpy (param_value, __STR__, value_size);                        \
          POCL_MEM_FREE (__STR__);                                            \
          if (param_value_size < value_size)                                  \
            return CL_INVALID_VALUE;                                          \
        }                                                                     \
      else                                                                    \
        POCL_MEM_FREE (__STR__);                                              \
      if (param_value_size_ret)                                               \
        *param_value_size_ret = value_size;                                   \
      return CL_SUCCESS;                                                      \
    }                                                                         \
  while (0)

#define POCL_RETURN_GETINFO(__TYPE__, __VALUE__)                        \
  do                                                                    \
    {                                                                   \
      size_t const value_size = sizeof(__TYPE__);                       \
      POCL_RETURN_GETINFO_INNER(value_size,                             \
                  *(__TYPE__*)param_value=__VALUE__);                   \
    }                                                                   \
  while (0)

#define POCL_RETURN_GETINFO_ARRAY(__TYPE__, __NUM__, __VALUE__)         \
  do                                                                    \
    {                                                                   \
      size_t const value_size = __NUM__*sizeof(__TYPE__);               \
      POCL_RETURN_GETINFO_INNER(value_size,                             \
                  memcpy(param_value, __VALUE__, value_size));          \
    }                                                                   \
  while (0)

#define IMAGE1D_TO_BUFFER(mem)                                                \
  mem = ((mem->is_image && (mem->type == CL_MEM_OBJECT_IMAGE1D_BUFFER))       \
             ? mem->buffer                                                    \
             : mem);

#define IS_IMAGE1D_BUFFER(mem)                                        \
  (mem && mem->is_image && (mem->type == CL_MEM_OBJECT_IMAGE1D_BUFFER))

#define IMAGE1D_ORIG_REG_TO_BYTES(mem, o, r)                                  \
  size_t px = (mem->image_elem_size * mem->image_channels);                   \
  size_t i1d_origin[3] = { o[0] * px, o[1], o[2] };                           \
  size_t i1d_region[3] = { r[0] * px, r[1], r[2] };

#define CMDBUF_VALIDATE_COMMON_HANDLES                                        \
  do                                                                          \
    {                                                                         \
      POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_buffer)),         \
                              CL_INVALID_COMMAND_BUFFER_KHR);                 \
      POCL_RETURN_ERROR_COND (                                                \
        (command_queue == NULL && command_buffer->num_queues > 1),            \
        CL_INVALID_COMMAND_QUEUE);                                            \
      int queue_in_buffer = 0;                                                \
      for (int ii = 0; ii < command_buffer->num_queues; ++ii)                 \
        {                                                                     \
          queue_in_buffer |= (command_queue == command_buffer->queues[ii]);   \
        }                                                                     \
      POCL_RETURN_ERROR_COND ((command_queue != NULL && !queue_in_buffer),    \
                              CL_INVALID_COMMAND_QUEUE);                      \
      POCL_RETURN_ERROR_COND ((mutable_handle != NULL), CL_INVALID_VALUE);    \
      errcode = pocl_cmdbuf_choose_recording_queue (command_buffer,           \
                                                    &command_queue);          \
      if (errcode != CL_SUCCESS)                                              \
        return errcode;                                                       \
    }                                                                         \
  while (0)

/*  A class implemented with C macros, resembling LLVM's SmallVector
 *  includes locking for access
 *
 *  Put the define in a struct; will add these members to the struct:
 *  2 pointers + 2x uint + TYPE array[STATIC_CAPACITY]
 *
 *  if inserting would overflow the static array,
 *  it moves all items to malloc'ed memory
 */

#define SMALL_VECTOR_DEFINE(TYPE, PREFIX, STATIC_CAPACITY)                    \
  TYPE PREFIX##_backing_array[STATIC_CAPACITY];                               \
  TYPE *PREFIX##_backing_malloc;                                              \
  TYPE *PREFIX##_ptr;                                                         \
  unsigned PREFIX##_capacity;                                                 \
  unsigned PREFIX##_used;                                                     \
  pocl_lock_t PREFIX##_handling_lock;

#define SMALL_VECTOR_INIT(STRUCT, TYPE, PREFIX, STATIC_CAPACITY)              \
  do                                                                          \
    {                                                                         \
      memset (STRUCT->PREFIX##_backing_array, 0,                              \
              (STATIC_CAPACITY * sizeof (TYPE)));                             \
      POCL_INIT_LOCK (STRUCT->PREFIX##_handling_lock);                        \
      STRUCT->PREFIX##_backing_malloc = NULL;                                 \
      STRUCT->PREFIX##_ptr = STRUCT->PREFIX##_backing_array;                  \
      STRUCT->PREFIX##_capacity = STATIC_CAPACITY;                            \
      STRUCT->PREFIX##_used = 0;                                              \
    }                                                                         \
  while (0)

#define SMALL_VECTOR_DESTROY(STRUCT, PREFIX, STATIC_CAPACITY)                 \
  do                                                                          \
    {                                                                         \
      if (STRUCT->PREFIX##_ptr != STRUCT->PREFIX##_backing_array)             \
        pocl_aligned_free (STRUCT->PREFIX##_backing_malloc);                  \
      STRUCT->PREFIX##_backing_malloc = NULL;                                 \
      STRUCT->PREFIX##_ptr = STRUCT->PREFIX##_backing_array;                  \
      STRUCT->PREFIX##_capacity = STATIC_CAPACITY;                            \
      STRUCT->PREFIX##_used = 0;                                              \
      POCL_DESTROY_LOCK (STRUCT->PREFIX##_handling_lock);                     \
    }                                                                         \
  while (0)

#define SMALL_VECTOR_HELPERS_EXTRA(SUFFIX, STRUCT, TYPE, PREFIX)              \
  static void small_vector_set_##SUFFIX (STRUCT *s, unsigned index,           \
                                         TYPE item)                           \
  {                                                                           \
    POCL_LOCK (s->PREFIX##_handling_lock);                                    \
    assert (index < s->PREFIX##_capacity);                                    \
    memcpy (&s->PREFIX##_ptr[index], &item, sizeof (TYPE));                   \
    POCL_UNLOCK (s->PREFIX##_handling_lock);                                  \
  }                                                                           \
  static TYPE small_vector_get_##SUFFIX (S *s, unsigned index)                \
  {                                                                           \
    POCL_LOCK (s->PREFIX##_handling_lock);                                    \
    assert (index < s->PREFIX##_used);                                        \
    TYPE temp = s->PREFIX##_ptr[index];                                       \
    POCL_UNLOCK (s->PREFIX##_handling_lock);                                  \
    return temp;                                                              \
  }

#define SMALL_VECTOR_HELPERS(SUFFIX, STRUCT, TYPE, PREFIX)                    \
  static int small_vector_find_##SUFFIX (STRUCT *s, TYPE item)                \
  {                                                                           \
    POCL_LOCK (s->PREFIX##_handling_lock);                                    \
    unsigned i;                                                               \
    int retval = -1;                                                          \
    for (i = 0; i < s->PREFIX##_used; ++i)                                    \
      {                                                                       \
        if (memcmp (&s->PREFIX##_ptr[i], &item, sizeof (TYPE)) == 0)          \
          {                                                                   \
            retval = i;                                                       \
            break;                                                            \
          }                                                                   \
      }                                                                       \
    POCL_UNLOCK (s->PREFIX##_handling_lock);                                  \
    return retval;                                                            \
  }                                                                           \
  static unsigned small_vector_append_##SUFFIX (STRUCT *s, TYPE item)         \
  {                                                                           \
    POCL_LOCK (s->PREFIX##_handling_lock);                                    \
    if (s->PREFIX##_used == s->PREFIX##_capacity)                             \
      {                                                                       \
        if (s->PREFIX##_ptr == s->PREFIX##_backing_array)                     \
          {                                                                   \
            s->PREFIX##_backing_malloc                                        \
                = malloc (s->PREFIX##_capacity * 2 * sizeof (TYPE));          \
            s->PREFIX##_ptr = s->PREFIX##_backing_malloc;                     \
            memcpy (s->PREFIX##_ptr, s->PREFIX##_backing_array,               \
                    s->PREFIX##_capacity * sizeof (TYPE));                    \
            s->PREFIX##_capacity *= 2;                                        \
          }                                                                   \
        else                                                                  \
          {                                                                   \
            s->PREFIX##_backing_malloc                                        \
                = realloc (s->PREFIX##_backing_malloc,                        \
                           s->PREFIX##_capacity * 2 * sizeof (TYPE));         \
            s->PREFIX##_capacity *= 2;                                        \
          }                                                                   \
      }                                                                       \
    memcpy (&s->PREFIX##_ptr[s->PREFIX##_used], &item, sizeof (TYPE));        \
    unsigned retval = ++s->PREFIX##_used;                                     \
    POCL_UNLOCK (s->PREFIX##_handling_lock);                                  \
    return retval;                                                            \
  }                                                                           \
  static unsigned small_vector_remove_##SUFFIX (STRUCT *s, TYPE item)         \
  {                                                                           \
    POCL_LOCK (s->PREFIX##_handling_lock);                                    \
    unsigned index;                                                           \
    for (index = 0; index < s->PREFIX##_used; ++index)                        \
      {                                                                       \
        if (memcmp (&s->PREFIX##_ptr[index], &item, sizeof (TYPE)) == 0)      \
          break;                                                              \
      }                                                                       \
    assert (index < s->PREFIX##_used);                                        \
    unsigned last = (s->PREFIX##_used - 1);                                   \
    if (index != last)                                                        \
      {                                                                       \
        memcpy (&s->PREFIX##_ptr[index], &s->PREFIX##_ptr[last],              \
                sizeof (TYPE));                                               \
      }                                                                       \
    memset (&s->PREFIX##_ptr[last], 0, sizeof (TYPE));                        \
    unsigned retval = --s->PREFIX##_used;                                     \
    POCL_UNLOCK (s->PREFIX##_handling_lock);                                  \
    return retval;                                                            \
  }

#endif
