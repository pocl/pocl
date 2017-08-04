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

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

#ifndef _MSC_VER
#include <dirent.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unistd.h>
#include <utime.h>
#else
#  include "vccompat.hpp"
#endif

#include "pocl_util.h"
#include "pocl_llvm.h"
#include "utlist.h"
#include "common.h"
#include "pocl_mem_management.h"
#include "devices.h"
#include "pocl_runtime_config.h"

/* required for setting SSE/AVX flush denorms to zero flag */
#if defined(__x86_64__) && defined(__GNUC__)
#include <x86intrin.h>
#endif

struct list_item;

typedef struct list_item
{
  void *value;
  struct list_item *next;
} list_item;



void
pocl_restore_ftz (unsigned ftz)
{
#if defined(__x86_64__) && defined(__GNUC__)

#ifdef _MM_FLUSH_ZERO_ON
  if (ftz & _MM_FLUSH_ZERO_ON)
    _MM_SET_FLUSH_ZERO_MODE (_MM_FLUSH_ZERO_ON);
  else
    _MM_SET_FLUSH_ZERO_MODE (_MM_FLUSH_ZERO_OFF);
#endif
#ifdef _MM_DENORMALS_ZERO_ON
  if (ftz & _MM_DENORMALS_ZERO_ON)
    _MM_SET_DENORMALS_ZERO_MODE (_MM_DENORMALS_ZERO_ON);
  else
    _MM_SET_DENORMALS_ZERO_MODE (_MM_DENORMALS_ZERO_OFF);
#endif

#endif
}

unsigned
pocl_save_ftz ()
{
#if defined(__x86_64__) && defined(__GNUC__)

  unsigned s = 0;
#ifdef _MM_FLUSH_ZERO_ON
  if (_MM_GET_FLUSH_ZERO_MODE ())
    s |= _MM_FLUSH_ZERO_ON;
  else
    s &= (~_MM_FLUSH_ZERO_ON);
#endif
#ifdef _MM_DENORMALS_ZERO_ON
  if (_MM_GET_DENORMALS_ZERO_MODE ())
    s |= _MM_DENORMALS_ZERO_ON;
  else
    s &= (~_MM_DENORMALS_ZERO_ON);
#endif
  return s;

#else
  return 0;
#endif
}

void
pocl_set_ftz (unsigned ftz)
{
#if defined(__x86_64__) && defined(__GNUC__)
  if (ftz)
    {
#ifdef _MM_FLUSH_ZERO_ON
      _MM_SET_FLUSH_ZERO_MODE (_MM_FLUSH_ZERO_ON);
#endif

#ifdef _MM_DENORMALS_ZERO_ON
      _MM_SET_DENORMALS_ZERO_MODE (_MM_DENORMALS_ZERO_ON);
#endif
    }
  else
    {
#ifdef _MM_FLUSH_ZERO_OFF
      _MM_SET_FLUSH_ZERO_MODE (_MM_FLUSH_ZERO_OFF);
#endif

#ifdef _MM_DENORMALS_ZERO_OFF
      _MM_SET_DENORMALS_ZERO_MODE (_MM_DENORMALS_ZERO_OFF);
#endif
    }
#endif
}


void
pocl_set_default_rm ()
{
#if defined(__x86_64__) && defined(__GNUC__) && defined(_MM_ROUND_NEAREST)
  unsigned rm = _MM_GET_ROUNDING_MODE ();
  if (rm != _MM_ROUND_NEAREST)
    _MM_SET_ROUNDING_MODE (_MM_ROUND_NEAREST);
#endif
}

unsigned
pocl_save_rm ()
{
#if defined(__x86_64__) && defined(__GNUC__) && defined(_MM_ROUND_NEAREST)
  return _MM_GET_ROUNDING_MODE ();
#else
  return 0;
#endif
}

void
pocl_restore_rm (unsigned rm)
{
#if defined(__x86_64__) && defined(__GNUC__) && defined(_MM_ROUND_NEAREST)
  _MM_SET_ROUNDING_MODE (rm);
#endif
}

uint32_t
byteswap_uint32_t (uint32_t word, char should_swap)
{
    union word_union
    {
        uint32_t full_word;
        unsigned char bytes[4];
    } old, neww;
    if (!should_swap) return word;

    old.full_word = word;
    neww.bytes[0] = old.bytes[3];
    neww.bytes[1] = old.bytes[2];
    neww.bytes[2] = old.bytes[1];
    neww.bytes[3] = old.bytes[0];
    return neww.full_word;
}

float
byteswap_float (float word, char should_swap)
{
    union word_union
    {
        float full_word;
        unsigned char bytes[4];
    } old, neww;
    if (!should_swap) return word;

    old.full_word = word;
    neww.bytes[0] = old.bytes[3];
    neww.bytes[1] = old.bytes[2];
    neww.bytes[2] = old.bytes[1];
    neww.bytes[3] = old.bytes[0];
    return neww.full_word;
}

size_t
pocl_size_ceil2(size_t x) {
  /* Rounds up to the next highest power of two without branching and
   * is as fast as a BSR instruction on x86, see:
   *
   * http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
   */
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
#if SIZE_MAX > 0xFFFFFFFF
  x |= x >> 32;
#endif
  return ++x;
}

#ifndef HAVE_ALIGNED_ALLOC
void *
pocl_aligned_malloc (size_t alignment, size_t size)
{
# ifdef HAVE_POSIX_MEMALIGN

  /* make sure that size is a multiple of alignment, as posix_memalign
   * does not perform this test, whereas aligned_alloc does */
  if ((size & (alignment - 1)) != 0)
    {
      errno = EINVAL;
      return NULL;
    }

  /* posix_memalign requires alignment to be at least sizeof(void *) */
  if (alignment < sizeof(void *))
    alignment = sizeof(void* );

  void* result;

  result = pocl_memalign_alloc(alignment, size);
  if (result == NULL)
    {
      errno = -1;
      return NULL;
    }

  return result;

# else

  /* allow zero-sized allocations, force alignment to 1 */
  if (!size)
    alignment = 1;

  /* make sure alignment is a non-zero power of two and that
   * size is a multiple of alignment */
  size_t mask = alignment - 1;
  if (!alignment || ((alignment & mask) != 0) || ((size & mask) != 0))
    {
      errno = EINVAL;
      return NULL;
    }

  /* allocate memory plus space for alignment header */
  uintptr_t address = (uintptr_t)malloc(size + mask + sizeof(void *));
  if (!address)
    return NULL;

  /* align the address, and store original pointer for future use
   * with free in the preceding bytes */
  uintptr_t aligned_address = (address + mask + sizeof(void *)) & ~mask;
  void** address_ptr = (void **)(aligned_address - sizeof(void *));
  *address_ptr = (void *)address;
  return (void *)aligned_address;

#endif
}
#endif

#if !defined HAVE_ALIGNED_ALLOC && !defined HAVE_POSIX_MEMALIGN
void
pocl_aligned_free (void *ptr)
{
  /* extract pointer from original allocation and free it */
  if (ptr)
    free(*(void **)((uintptr_t)ptr - sizeof(void *)));
}
#endif

void
pocl_lock_events_inorder (cl_event ev1, cl_event ev2)
{
  if (ev1->id < ev2->id)
    {
      POCL_LOCK_OBJ (ev1);
      POCL_LOCK_OBJ (ev2);
    }
  else
    {
      POCL_LOCK_OBJ (ev2);
      POCL_LOCK_OBJ (ev1);
    }
}

void
pocl_unlock_events_inorder (cl_event ev1, cl_event ev2)
{
  if (ev1->id < ev2->id)
    {
      POCL_UNLOCK_OBJ (ev1);
      POCL_UNLOCK_OBJ (ev2);
    }
  else
    {
      POCL_UNLOCK_OBJ (ev2);
      POCL_UNLOCK_OBJ (ev1);
    }
}

cl_int pocl_create_event (cl_event *event, cl_command_queue command_queue, 
                          cl_command_type command_type, int num_buffers,
                          const cl_mem *buffers, cl_context context)
{
  static unsigned int event_id_counter = 0;

  POCL_MSG_PRINT_EVENTS ("creating event\n");

  if (context == NULL || !(context->valid))
    return CL_INVALID_CONTEXT;
  if (event != NULL)
    {
      *event = pocl_mem_manager_new_event ();
      if (*event == NULL)
        return CL_OUT_OF_HOST_MEMORY;

      (*event)->context = context;
      POname(clRetainContext) (context);
      (*event)->queue = command_queue;

      /* user events have a NULL command queue, don't retain it */
      if (command_queue)
        POname(clRetainCommandQueue) (command_queue);

      (*event)->command_type = command_type;
      (*event)->command =  NULL;
      (*event)->callback_list = NULL;
      (*event)->id = event_id_counter++;
      (*event)->notify_list = NULL;
      (*event)->wait_list = NULL;
      (*event)->data = NULL;
      (*event)->num_buffers = num_buffers;
      if (num_buffers > 0)
        {
          (*event)->mem_objs = malloc (num_buffers * sizeof(cl_mem));
          memcpy ((*event)->mem_objs, buffers, num_buffers * sizeof(cl_mem));
        }
      else
        (*event)->mem_objs = NULL;
      (*event)->status = CL_QUEUED;
      (*event)->implicit_event = 0;
      (*event)->next = NULL;
      (*event)->prev = NULL;

      /* user events do not have cq */
      if (!command_queue)
        return CL_SUCCESS;

    }
  return CL_SUCCESS;
}

static int
pocl_create_event_sync(cl_event waiting_event,
                       cl_event notifier_event, cl_mem mem)
{
  event_node * volatile notify_target = NULL;
  event_node * volatile wait_list_item = NULL;

  if (notifier_event == NULL)
    return CL_SUCCESS;

  assert(notifier_event->pocl_refcount != 0);
  POCL_MSG_PRINT_INFO("create event sync: waiting %d, notifier %d\n", waiting_event->id, notifier_event->id);
  if (waiting_event == notifier_event)
    {
      printf("waiting id %d, notifier id = %d\n", waiting_event->id, 
             notifier_event->id);
      assert(waiting_event != notifier_event);
    }

  pocl_lock_events_inorder (waiting_event, notifier_event);

  LL_FOREACH (waiting_event->wait_list, wait_list_item)
    {
      if (wait_list_item->event == notifier_event)
        goto FINISH;
    }

  if (notifier_event->status == CL_COMPLETE)
    goto FINISH;
  notify_target = pocl_mem_manager_new_event_node();
  wait_list_item = pocl_mem_manager_new_event_node();
  if (!notify_target || !wait_list_item)
    return CL_OUT_OF_HOST_MEMORY;
    
  notify_target->event = waiting_event;
  wait_list_item->event = notifier_event;
  LL_PREPEND (notifier_event->notify_list, notify_target);
  LL_PREPEND (waiting_event->wait_list, wait_list_item);

FINISH:
  pocl_unlock_events_inorder (waiting_event, notifier_event);
  return CL_SUCCESS;
}

cl_int pocl_create_command (_cl_command_node **cmd,
                            cl_command_queue command_queue, 
                            cl_command_type command_type, cl_event *event_p, 
                            cl_int num_events, const cl_event *wait_list,
                            int num_buffers, const cl_mem *buffers)
{
  int i;
  int err;
  cl_event *event = NULL;
  
  if ((wait_list == NULL && num_events != 0) ||
      (wait_list != NULL && num_events == 0))
    {
      assert(0);
      return CL_INVALID_EVENT_WAIT_LIST;
    }

  for (i = 0; i < num_events; ++i)
    {
      if (wait_list[i] == NULL)
        return CL_INVALID_EVENT_WAIT_LIST;
    }

  *cmd = pocl_mem_manager_new_command ();
  if (*cmd == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  (*cmd)->type = command_type;

  /* Even if user does not provide event pointer, create event anyway */
  event = &((*cmd)->event);
  err = pocl_create_event(event, command_queue, 0, num_buffers, 
                          buffers, command_queue->context);

  if (err != CL_SUCCESS)
    {
      POCL_MEM_FREE(*cmd);
      return err;
    }
  (*event)->command_type = command_type;

  /* if host application wants this commands event
     one reference for the host and one for the runtime/driver */
  if (event_p)
    {
      POCL_MSG_PRINT_EVENTS ("event pointer provided\n");
      *event_p = *event;
      (*event)->implicit_event = 0;
      (*event)->pocl_refcount = 2;
    }
  else
    {
      (*event)->implicit_event = 1;
      (*event)->pocl_refcount = 1;
    }

  (*cmd)->next = NULL;
  (*cmd)->prev = NULL;
  (*cmd)->device = command_queue->device;
  (*cmd)->event->command = (*cmd);
  (*cmd)->ready = 0;

  /* in case of in-order queue, synchronize to previously enqueued command
     if available */
  if (!(command_queue->properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE))
    {
      POCL_LOCK_OBJ (command_queue);
      if (command_queue->last_event.event)
        {
          pocl_create_event_sync ((*cmd)->event,
                                  command_queue->last_event.event,
                                  NULL);
        }
      POCL_UNLOCK_OBJ (command_queue);
    }
  /* Form event synchronizations based on the given wait list */
  for (i = 0; i < num_events; ++i)
    {
      cl_event wle = wait_list[i];
      pocl_create_event_sync ((*cmd)->event, wle, NULL);
    }
  POCL_MSG_PRINT_EVENTS ("Created command struct (event %d, type %X)\n",
                         (*cmd)->event->id, command_type);
  return CL_SUCCESS;
}

void pocl_command_enqueue (cl_command_queue command_queue,
                          _cl_command_node *node)
{
  cl_event event;

  POCL_LOCK_OBJ (node->event);
  assert(node->event->status == CL_QUEUED);
  POCL_UNLOCK_OBJ (node->event);

  assert (command_queue == node->event->queue);

  POCL_LOCK_OBJ (command_queue);
  ++command_queue->command_count;
  if ((node->type == CL_COMMAND_BARRIER || node->type == CL_COMMAND_MARKER) &&
      node->command.barrier.has_wait_list == 0)
    {
      DL_FOREACH (command_queue->events, event)
        {
          pocl_create_event_sync (node->event, event, NULL);
        }
    }
  if (node->type == CL_COMMAND_BARRIER)
    command_queue->barrier = node->event;
  else
    {
      if (command_queue->barrier)
        {
          pocl_create_event_sync (node->event, command_queue->barrier, NULL);
        }
    }
  DL_APPEND (command_queue->events, node->event);
  command_queue->last_event.event = node->event;
  command_queue->last_event.event_id = node->event->id;
  POCL_UNLOCK_OBJ (command_queue);

  POCL_UPDATE_EVENT_QUEUED (node->event);

  command_queue->device->ops->submit(node, command_queue);
#ifdef POCL_DEBUG_BUILD
  if (pocl_is_option_set("POCL_IMPLICIT_FINISH"))
    POclFinish (command_queue);
#endif
}


void
pocl_command_push (_cl_command_node *node, 
                   _cl_command_node *volatile *ready_list, 
                   _cl_command_node *volatile *pending_list)
{
  assert (node != NULL);

  /* If the last command inserted is a barrier,
     command is necessary not ready */
  if ((*ready_list) != NULL && (*ready_list)->prev
      && (*ready_list)->prev->type == CL_COMMAND_BARRIER)
    {
      CDL_PREPEND ((*pending_list), node);
      return;
    }
  POCL_LOCK_OBJ (node->event);
  if (pocl_command_is_ready(node->event))
    {
      CDL_PREPEND ((*ready_list), node);
    }
  else
    {
      CDL_PREPEND ((*pending_list), node);
    }
  POCL_UNLOCK_OBJ (node->event);
}

int pocl_update_command_queue (cl_event event)
{
  int cq_ready;

  POCL_LOCK_OBJ (event->queue);
  --event->queue->command_count;
  if (event->queue->barrier == event)
    event->queue->barrier = NULL;

  if (event->queue->last_event.event == event)
    event->queue->last_event.event = NULL;
  assert (event->queue->command_count >= 0);

  DL_DELETE (event->queue->events, event);

  cq_ready = (event->queue->command_count) ? 0: 1;
  POCL_UNLOCK_OBJ (event->queue);
  return cq_ready;
}

void
pocl_cl_mem_inherit_flags (cl_mem mem, cl_mem from_buffer, cl_mem_flags flags)
{
  if ((flags & CL_MEM_READ_WRITE) | (flags & CL_MEM_READ_ONLY)
      | (flags & CL_MEM_WRITE_ONLY))
    {
      mem->flags = (flags & CL_MEM_READ_WRITE) | (flags & CL_MEM_READ_ONLY)
                   | (flags & CL_MEM_WRITE_ONLY);
    }
  else
    {
      mem->flags = (from_buffer->flags & CL_MEM_READ_WRITE)
                   | (from_buffer->flags & CL_MEM_READ_ONLY)
                   | (from_buffer->flags & CL_MEM_WRITE_ONLY);
    }

  if ((flags & CL_MEM_HOST_NO_ACCESS) | (flags & CL_MEM_HOST_READ_ONLY)
      | (flags & CL_MEM_HOST_WRITE_ONLY))
    {
      mem->flags = mem->flags | ((flags & CL_MEM_HOST_NO_ACCESS)
                                 | (flags & CL_MEM_HOST_READ_ONLY)
                                 | (flags & CL_MEM_HOST_WRITE_ONLY));
    }
  else
    {
      mem->flags
          = mem->flags | ((from_buffer->flags & CL_MEM_HOST_NO_ACCESS)
                          | (from_buffer->flags & CL_MEM_HOST_READ_ONLY)
                          | (from_buffer->flags & CL_MEM_HOST_WRITE_ONLY));
    }

  mem->flags = mem->flags | (from_buffer->flags & CL_MEM_USE_HOST_PTR)
               | (from_buffer->flags & CL_MEM_ALLOC_HOST_PTR)
               | (from_buffer->flags & CL_MEM_COPY_HOST_PTR);
}

cl_int pocl_update_mem_obj_sync (cl_command_queue cq, _cl_command_node *cmd, 
                                 cl_mem mem, char operation)
{
  int i;
  POCL_LOCK_OBJ (mem);
  mem->owning_device = cmd->device;
  mem->latest_event = cmd->event;
  POCL_UNLOCK_OBJ (mem);

  for (i = 0; i < cmd->event->num_buffers; ++i)
    {
      if (cmd->event->mem_objs[i] == NULL)
        {
          cmd->event->mem_objs[i] = mem;
          break;
        }
    }
  return CL_SUCCESS;
}

int pocl_buffer_boundcheck(cl_mem buffer, size_t offset, size_t size) {
  POCL_RETURN_ERROR_ON ((offset > buffer->size), CL_INVALID_VALUE,
                        "offset(%zu) > buffer->size(%zu)\n", offset,
                        buffer->size);
  POCL_RETURN_ERROR_ON ((size > buffer->size), CL_INVALID_VALUE,
                        "size(%zu) > buffer->size(%zu)\n", size, buffer->size);
  POCL_RETURN_ERROR_ON ((offset + size > buffer->size), CL_INVALID_VALUE,
                        "offset + size (%zu) > buffer->size(%zu)\n",
                        (offset + size), buffer->size);
  return CL_SUCCESS;
}

int pocl_buffer_boundcheck_3d(const size_t buffer_size,
                              const size_t *origin,
                              const size_t *region,
                              size_t *row_pitch,
                              size_t *slice_pitch,
                              const char* prefix)
{
  size_t rp = *row_pitch;
  size_t sp = *slice_pitch;

  /* CL_INVALID_VALUE if row_pitch is not 0 and is less than region[0]. */
  POCL_RETURN_ERROR_ON((rp != 0 && rp<region[0]),
    CL_INVALID_VALUE, "%srow_pitch is not 0 and is less than region[0]\n", prefix);

  if (rp == 0) rp = region[0];

  /* CL_INVALID_VALUE if slice_pitch is not 0 and is less than region[1] * row_pitch
   * or if slice_pitch is not 0 and is not a multiple of row_pitch.
   */
  POCL_RETURN_ERROR_ON((sp != 0 && sp < (region[1] * rp)),
    CL_INVALID_VALUE, "%sslice_pitch is not 0 and is less than "
      "region[1] * %srow_pitch\n", prefix, prefix);
  POCL_RETURN_ERROR_ON((sp != 0 && (sp % rp != 0)),
    CL_INVALID_VALUE, "%sslice_pitch is not 0 and is not a multiple "
      "of %srow_pitch\n", prefix, prefix);

  if (sp == 0) sp = region[1] * rp;

  *row_pitch = rp;
  *slice_pitch = sp;

  size_t byte_offset_begin = origin[2] * sp +
               origin[1] * rp +
               origin[0];

  size_t byte_offset_end = origin[0] + region[0]-1 +
       rp * (origin[1] + region[1]-1) +
       sp * (origin[2] + region[2]-1);


  POCL_RETURN_ERROR_ON((byte_offset_begin > buffer_size), CL_INVALID_VALUE,
            "%sorigin is outside the %sbuffer", prefix, prefix);
  POCL_RETURN_ERROR_ON((byte_offset_end > buffer_size), CL_INVALID_VALUE,
            "%sorigin+region is outside the %sbuffer", prefix, prefix);
  return CL_SUCCESS;
}



int pocl_buffers_boundcheck(cl_mem src_buffer,
                            cl_mem dst_buffer,
                            size_t src_offset,
                            size_t dst_offset,
                            size_t size) {
  POCL_RETURN_ERROR_ON((src_offset > src_buffer->size), CL_INVALID_VALUE,
            "src_offset(%zu) > src_buffer->size(%zu)", src_offset, src_buffer->size);
  POCL_RETURN_ERROR_ON((size > src_buffer->size), CL_INVALID_VALUE,
            "size(%zu) > src_buffer->size(%zu)", size, src_buffer->size);
  POCL_RETURN_ERROR_ON((src_offset + size > src_buffer->size), CL_INVALID_VALUE,
            "src_offset + size (%zu) > src_buffer->size(%zu)", (src_offset+size), src_buffer->size);

  POCL_RETURN_ERROR_ON((dst_offset > dst_buffer->size), CL_INVALID_VALUE,
            "dst_offset(%zu) > dst_buffer->size(%zu)", dst_offset, dst_buffer->size);
  POCL_RETURN_ERROR_ON((size > dst_buffer->size), CL_INVALID_VALUE,
            "size(%zu) > dst_buffer->size(%zu)", size, dst_buffer->size);
  POCL_RETURN_ERROR_ON((dst_offset + size > dst_buffer->size), CL_INVALID_VALUE,
            "dst_offset + size (%zu) > dst_buffer->size(%zu)", (dst_offset+size), dst_buffer->size);
  return CL_SUCCESS;
}

int pocl_buffers_overlap(cl_mem src_buffer,
                         cl_mem dst_buffer,
                         size_t src_offset,
                         size_t dst_offset,
                         size_t size) {
  /* The regions overlap if src_offset ≤ to dst_offset ≤ to src_offset + size - 1,
   * or if dst_offset ≤ to src_offset ≤ to dst_offset + size - 1.
   */
  if (src_buffer == dst_buffer) {
    POCL_RETURN_ERROR_ON(((src_offset <= dst_offset) && (dst_offset <=
      (src_offset + size - 1))), CL_MEM_COPY_OVERLAP, "dst_offset lies inside \
      the src region and the src_buffer == dst_buffer");
    POCL_RETURN_ERROR_ON(((dst_offset <= src_offset) && (src_offset <=
      (dst_offset + size - 1))), CL_MEM_COPY_OVERLAP, "src_offset lies inside \
      the dst region and the src_buffer == dst_buffer");
  }

  /* sub buffers overlap check  */
  if (src_buffer->parent && dst_buffer->parent &&
        (src_buffer->parent == dst_buffer->parent)) {
      src_offset = src_buffer->origin + src_offset;
      dst_offset = dst_buffer->origin + dst_offset;

      POCL_RETURN_ERROR_ON (((src_offset <= dst_offset)
                             && (dst_offset <= (src_offset + size - 1))),
                            CL_MEM_COPY_OVERLAP, "dst_offset lies inside \
      the src region and src_buffer + dst_buffer are subbuffers of the same buffer");
      POCL_RETURN_ERROR_ON (((dst_offset <= src_offset)
                             && (src_offset <= (dst_offset + size - 1))),
                            CL_MEM_COPY_OVERLAP, "src_offset lies inside \
      the dst region and src_buffer + dst_buffer are subbuffers of the same buffer");

  }

  return CL_SUCCESS;
}

/*
 * Copyright (c) 2011 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and /or associated documentation files (the "Materials "), to deal in the Materials
 * without restriction, including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Materials, and to permit persons to
 * whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE USE OR OTHER DEALINGS IN
 * THE MATERIALS.
 */

int
check_copy_overlap(const size_t src_offset[3],
                   const size_t dst_offset[3],
                   const size_t region[3],
                   const size_t row_pitch, const size_t slice_pitch)
{
  const size_t src_min[] = {src_offset[0], src_offset[1], src_offset[2]};
  const size_t src_max[] = {src_offset[0] + region[0],
                            src_offset[1] + region[1],
                            src_offset[2] + region[2]};
  const size_t dst_min[] = {dst_offset[0], dst_offset[1], dst_offset[2]};
  const size_t dst_max[] = {dst_offset[0] + region[0],
                            dst_offset[1] + region[1],
                            dst_offset[2] + region[2]};
  int overlap = 1;
  unsigned i;
  for (i=0; i != 3; ++i)
  {
    overlap = overlap && (src_min[i] < dst_max[i])
                      && (src_max[i] > dst_min[i]);
  }

  size_t dst_start =  dst_offset[2] * slice_pitch +
                      dst_offset[1] * row_pitch + dst_offset[0];
  size_t dst_end = dst_start + (region[2] * slice_pitch +
                                region[1] * row_pitch + region[0]);
  size_t src_start =  src_offset[2] * slice_pitch +
                      src_offset[1] * row_pitch + src_offset[0];
  size_t src_end = src_start + (region[2] * slice_pitch +
                                region[1] * row_pitch + region[0]);

  if (!overlap)
  {
    size_t delta_src_x = (src_offset[0] + region[0] > row_pitch) ?
                          src_offset[0] + region[0] - row_pitch : 0;
    size_t delta_dst_x = (dst_offset[0] + region[0] > row_pitch) ?
                          dst_offset[0] + region[0] - row_pitch : 0;
    if ( (delta_src_x > 0 && delta_src_x > dst_offset[0]) ||
          (delta_dst_x > 0 && delta_dst_x > src_offset[0]) )
      {
        if ( (src_start <= dst_start && dst_start < src_end) ||
          (dst_start <= src_start && src_start < dst_end) )
          overlap = 1;
      }

    if (region[2] > 1)
    {
      size_t src_height = slice_pitch / row_pitch;
      size_t dst_height = slice_pitch / row_pitch;

      size_t delta_src_y = (src_offset[1] + region[1] > src_height) ?
                            src_offset[1] + region[1] - src_height : 0;
      size_t delta_dst_y = (dst_offset[1] + region[1] > dst_height) ?
                            dst_offset[1] + region[1] - dst_height : 0;

      if ( (delta_src_y > 0 && delta_src_y > dst_offset[1]) ||
            (delta_dst_y > 0 && delta_dst_y > src_offset[1]) )
      {
        if ( (src_start <= dst_start && dst_start < src_end) ||
              (dst_start <= src_start && src_start < dst_end) )
              overlap = 1;
      }
    }
  }

  return overlap;
}

/* For a subdevice parameter, return the actual device it belongs to. */
cl_device_id
pocl_real_dev (const cl_device_id dev)
{
  cl_device_id ret = dev;
  while (ret->parent_device)
    ret = ret->parent_device;
  return ret;
}

/* Make a list of unique devices. If any device is a subdevice,
 * replace with parent, then remove duplicate parents. */
cl_device_id * pocl_unique_device_list(const cl_device_id * in, cl_uint num, cl_uint *real)
{
  cl_uint real_num = num;
  cl_device_id * out = calloc(num, sizeof(cl_device_id));
  if (!out)
    return NULL;

  unsigned i;
  for (i=0; i < num; ++i)
    out[i] = (in[i] ? pocl_real_dev (in[i]) : NULL);

  i=1;
  unsigned device_i=0;
  while (i < real_num)
    {
      device_i=0;
      while (device_i < i)
        {
          if (out[device_i] == out[i])
            {
              out[device_i] = out[--real_num];
              out[real_num] = NULL;
            }
          else
            device_i++;
        }
      i++;
    }

  *real = real_num;
  return out;
}

/* Setup certain info about context that comes up later in API calls */
void pocl_setup_context(cl_context context)
{
  unsigned i;
  context->min_max_mem_alloc_size = SIZE_MAX;
  context->svm_allocdev = NULL;
  for(i=0; i<context->num_devices; i++)
    {
      if (context->devices[i]->should_allocate_svm)
        context->svm_allocdev = context->devices[i];
      if (context->devices[i]->max_mem_alloc_size < context->min_max_mem_alloc_size)
        context->min_max_mem_alloc_size =
            context->devices[i]->max_mem_alloc_size;
    }
  if (context->svm_allocdev == NULL)
    for(i=0; i<context->num_devices; i++)
      if (DEVICE_IS_SVM_CAPABLE(context->devices[i]))
        {
          context->svm_allocdev = context->devices[i];
          break;
        }
}

int
pocl_check_event_wait_list (cl_command_queue command_queue,
                            cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list)
{
  POCL_RETURN_ERROR_COND (
      (event_wait_list == NULL && num_events_in_wait_list > 0),
      CL_INVALID_EVENT_WAIT_LIST);

  POCL_RETURN_ERROR_COND (
      (event_wait_list != NULL && num_events_in_wait_list == 0),
      CL_INVALID_EVENT_WAIT_LIST);

  if (event_wait_list)
    {
      unsigned i;
      for (i = 0; i < num_events_in_wait_list; i++)
        {
          POCL_RETURN_ERROR_COND ((event_wait_list[i] == NULL),
                                  CL_INVALID_EVENT_WAIT_LIST);
          POCL_RETURN_ERROR_COND (
              (event_wait_list[i]->context != command_queue->context),
              CL_INVALID_CONTEXT);
        }
    }

  return CL_SUCCESS;
}

const char*
pocl_status_to_str (int status)
{
  static const char *status_to_str[] = {
  "complete",
  "running",
  "submitted",
  "queued"};
  return status_to_str[status];
};


/* Convert a command type to its representation string
 */
const char *
pocl_command_to_str (cl_command_type cmd)
{
  switch (cmd)
    {
    case CL_COMMAND_NDRANGE_KERNEL:
      return "ndrange_kernel";
    case CL_COMMAND_TASK:
      return "task_kernel";
    case CL_COMMAND_NATIVE_KERNEL:
      return "native_kernel";
    case CL_COMMAND_READ_BUFFER:
      return "read_buffer";
    case CL_COMMAND_WRITE_BUFFER:
      return "write_buffer";
    case CL_COMMAND_COPY_BUFFER:
      return "copy_buffer";
    case CL_COMMAND_READ_IMAGE:
      return "read_image";
    case CL_COMMAND_WRITE_IMAGE:
      return "write_image";
    case CL_COMMAND_COPY_IMAGE:
      return "copy_image";
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
      return "copy_image_to_buffer";
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      return "copy_buffer_to_image";
    case CL_COMMAND_MAP_BUFFER:
      return "map_buffer";
    case CL_COMMAND_MAP_IMAGE:
      return "map_image";
    case CL_COMMAND_UNMAP_MEM_OBJECT:
      return "unmap_mem_object";
    case CL_COMMAND_MARKER:
      return "marker";
    case CL_COMMAND_ACQUIRE_GL_OBJECTS:
      return "acquire_gl_objects";
    case CL_COMMAND_RELEASE_GL_OBJECTS:
      return "release_gl_objects";
    case CL_COMMAND_READ_BUFFER_RECT:
      return "read_buffer_rect";
    case CL_COMMAND_WRITE_BUFFER_RECT:
      return "write_buffer_rect";
    case CL_COMMAND_COPY_BUFFER_RECT:
      return "copy_buffer_rect";
    case CL_COMMAND_USER:
      return "user";
    case CL_COMMAND_BARRIER:
      return "barrier";
    case CL_COMMAND_MIGRATE_MEM_OBJECTS:
      return "migrate_mem_objects";
    case CL_COMMAND_FILL_BUFFER:
      return "fill_buffer";
    case CL_COMMAND_FILL_IMAGE:
      return "fill_image";
    case CL_COMMAND_SVM_FREE:
      return "svm_free";
    case CL_COMMAND_SVM_MEMCPY:
      return "svm_memcpy";
    case CL_COMMAND_SVM_MEMFILL:
      return "svm_memfill";
    case CL_COMMAND_SVM_MAP:
      return "svm_map";
    case CL_COMMAND_SVM_UNMAP:
      return "svm_unmap";
    }

  return "unknown";
}

/*
 * This replaces a simple system(), because:
 *
 * 1) system() was causing issues (gpu lockups) with HSA when
 * compiling code (via compile_parallel_bc_to_brig)
 * with OpenCL 2.0 atomics (like CalcPie from AMD SDK).
 * The reason of lockups is unknown (yet).
 *
 * 2) system() uses fork() which copies page table maps, and runs
 * out of AS when pocl has already allocated huge buffers in memory.
 * this happened in llvm_codegen()
 *
 * vfork() does not copy pagetables.
 */
int
pocl_run_command (char *const *args)
{
  POCL_MSG_PRINT_INFO ("Launching: %s\n", args[0]);
#ifdef HAVE_VFORK
  pid_t p = vfork ();
#elif defined(HAVE_FORK)
  pid_t p = fork ();
#else
#error Must have fork() or vfork() system calls for HSA
#endif
  if (p == 0)
    {
      return execv (args[0], args);
    }
  else
    {
      if (p < 0)
        return EXIT_FAILURE;
      int status;
      if (waitpid (p, &status, 0) < 0)
        POCL_ABORT ("pocl: waitpid() failed.\n");
      if (WIFEXITED (status))
        return WEXITSTATUS (status);
      else if (WIFSIGNALED (status))
        return WTERMSIG (status);
      else
        return EXIT_FAILURE;
    }
}

/*
 * float 2 half / half 2 float
 */

static int const shift = 13;
static int const shiftSign = 16;

static int32_t const infN = 0x7F800000;  /* flt32 infinity */
static int32_t const maxN = 0x477FE000;  /* max flt16 normal as a flt32 */
static int32_t const minN = 0x38800000;  /* min flt16 normal as a flt32 */
static int32_t const signN = 0x80000000; /* flt32 sign bit */

/* static int32_t const infC = infN >> shift;
 * static int32_t const infC = 0x3FC00;
 * static int32_t const nanN = (infC + 1) << shift; // minimum flt16 nan as a flt32
 */
static int32_t const nanN = 0x7f802000;
/* static int32_t const maxC = maxN >> shift; */
static int32_t const maxC = 0x23bff;
/* static int32_t const minC = minN >> shift;
 * static int32_t const minC = 0x1c400;
 * static int32_t const signC = signN >> shiftSign; // flt16 sign bit
 */
static int32_t const signC = 0x40000; /* flt16 sign bit */

static int32_t const mulN = 0x52000000; /* (1 << 23) / minN */
static int32_t const mulC = 0x33800000; /* minN / (1 << (23 - shift)) */

static int32_t const subC = 0x003FF; /* max flt32 subnormal down shifted */
static int32_t const norC = 0x00400; /* min flt32 normal down shifted */

/* static int32_t const maxD = infC - maxC - 1; */
static int32_t const maxD = 0x1c000;
/* static int32_t const minD = minC - subC - 1; */
static int32_t const minD = 0x1c000;

typedef union
{
  float f;
  int32_t si;
  uint32_t ui;
} H2F_Bits;

float
half_to_float (uint16_t value)
{
  H2F_Bits v;
  v.ui = value;
  int32_t sign = v.si & signC;
  v.si ^= sign;
  sign <<= shiftSign;
  v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
  v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
  H2F_Bits s;
  s.si = mulC;
  s.f *= v.si;
  int32_t mask = -(norC > v.si);
  v.si <<= shift;
  v.si ^= (s.si ^ v.si) & mask;
  v.si |= sign;
  return v.f;
}

uint16_t
float_to_half (float value)
{
  H2F_Bits v, s;
  v.f = value;
  uint32_t sign = v.si & signN;
  v.si ^= sign;
  sign >>= shiftSign;
  s.si = mulN;
  s.si = s.f * v.f;
  v.si ^= (s.si ^ v.si) & -(minN > v.si);
  v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
  v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
  v.ui >>= shift;
  v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
  v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
  return v.ui | sign;
}
