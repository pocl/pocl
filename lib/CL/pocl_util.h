/* OpenCL runtime library: pocl_util utility functions

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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

#ifndef POCL_UTIL_H
#define POCL_UTIL_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "pocl_cl.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

#ifdef __cplusplus
extern "C" {
#endif

uint32_t byteswap_uint32_t (uint32_t word, char should_swap);
float byteswap_float (float word, char should_swap);

/* Finds the next highest power of two of the given value. */
size_t pocl_size_ceil2(size_t x);

/* Allocates aligned blocks of memory.
 *
 * Uses posix_memalign when available. Otherwise, uses
 * malloc to allocate a block of memory on the heap of the desired
 * size which is then aligned with the given alignment. The resulting
 * pointer must be freed with a call to pocl_aligned_free. Alignment
 * must be a non-zero power of 2.
 */
#if defined HAVE_POSIX_MEMALIGN
void *pocl_aligned_malloc(size_t alignment, size_t size);
# define pocl_aligned_free free
#else
void *pocl_aligned_malloc(size_t alignment, size_t size);
void pocl_aligned_free(void* ptr);
#endif

/* Function for creating events */
cl_int pocl_create_event (cl_event *event, cl_context context,
                          cl_command_queue command_queue,
                          cl_command_type command_type);

cl_int pocl_create_command (_cl_command_node **cmd,
                            cl_command_queue command_queue,
                            cl_command_type command_type, cl_event *event,
                            cl_int num_events, const cl_event *wait_list);


void pocl_command_enqueue (cl_command_queue command_queue,
                          _cl_command_node *node);


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

int
check_copy_overlap(const size_t src_offset[3],
                   const size_t dst_offset[3],
                   const size_t region[3],
                   const size_t row_pitch, const size_t slice_pitch);

/* Finds a device which should allocate the SVM memory. */
cl_device_id find_svm_device(cl_context context);


/* Helpers for dealing with devices / subdevices */

#define POCL_REAL_DEV(dev) (dev->parent_device ? dev->parent_device : dev)

cl_device_id * pocl_unique_device_list(const cl_device_id * in, cl_uint num, cl_uint *real);

#define POCL_CHECK_DEV_IN_CMDQ                                               \
  device = command_queue->device;                                            \
  for (i = 0; i < command_queue->context->num_devices; ++i)                  \
    {                                                                        \
      if (command_queue->context->devices[i] == POCL_REAL_DEV(device))       \
        break;                                                               \
    }                                                                        \
  assert(i < command_queue->context->num_devices);


#ifdef __cplusplus
}
#endif

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

/* Common macro for cleaning up "*GetInfo" API call implementations.
 * All the *GetInfo functions have been specified to look alike, 
 * and have been implemented to use the same variable names, so this
 * code can be shared.
 */

#define POCL_RETURN_GETINFO_INNER(__SIZE__, MEMASSIGN)                  \
    if (param_value) {                                                  \
      if (param_value_size < __SIZE__) return CL_INVALID_VALUE;         \
      MEMASSIGN;                                                        \
    }                                                                   \
    if (param_value_size_ret)                                           \
      *param_value_size_ret = __SIZE__;                                 \
    return CL_SUCCESS;                                                  \

#define POCL_RETURN_GETINFO_SIZE(__SIZE__, __POINTER__)                 \
  {                                                                     \
    POCL_RETURN_GETINFO_INNER(__SIZE__,                                 \
                memcpy(param_value, __POINTER__, __SIZE__))             \
  }

#define POCL_RETURN_GETINFO_STR(__STR__)                                \
  {                                                                     \
    size_t const value_size = strlen(__STR__) + 1;                      \
    POCL_RETURN_GETINFO_INNER(value_size,                               \
                memcpy(param_value, __STR__, value_size))               \
  }

#define POCL_RETURN_GETINFO(__TYPE__, __VALUE__)                        \
  {                                                                     \
    size_t const value_size = sizeof(__TYPE__);                         \
    POCL_RETURN_GETINFO_INNER(value_size,                               \
                *(__TYPE__*)param_value=__VALUE__)                      \
  }

#define POCL_RETURN_GETINFO_ARRAY(__TYPE__, __NUM__, __VALUE__)         \
  {                                                                     \
    size_t const value_size = __NUM__*sizeof(__TYPE__);                 \
    POCL_RETURN_GETINFO_INNER(value_size,                               \
                memcpy(param_value, __VALUE__, value_size))             \
  }

#endif
