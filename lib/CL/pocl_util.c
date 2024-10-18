/* OpenCL runtime library: pocl_util utility functions

   Copyright (c) 2012-2019 Pekka Jääskeläinen
                 2020-2024 PoCL Developers
                 2024 Pekka Jääskeläinen / Intel Finland Oy
                 2024 Henry Linjamäki / Intel Finland Oy

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

#define _BSD_SOURCE
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#if defined(__FreeBSD__)
#include <stdlib.h>
#elif defined(_WIN32)
#include <malloc.h>
#else
#include <alloca.h>
#endif

#include <time.h>

#ifndef _WIN32
#include <dirent.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utime.h>
#else
#  include "vccompat.hpp"
#endif

#ifdef __MINGW32__
#include <process.h>
#endif

#include "common.h"
#include "devices.h"
#include "pocl_cache.h"
#include "pocl_dynlib.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_local_size.h"
#include "pocl_mem_management.h"
#include "pocl_runtime_config.h"
#include "pocl_shared.h"
#include "pocl_timing.h"
#include "pocl_util.h"
#include "utlist.h"
#include "utlist_addon.h"


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
pocl_byteswap_uint32_t (uint32_t word, char should_swap)
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

void
bzero_s (void *v, size_t n)
{
  if (v == NULL)
    return;

  volatile unsigned char *p = v;
  while (n--)
    {
      *p++ = 0;
    }

  return;
}

size_t
pocl_size_ceil2(size_t x) {
  /* Rounds up to the next highest power of two without branching and
   * is as fast as a BSR instruction on x86, see:
   *
   * https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
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

uint64_t
pocl_size_ceil2_64 (uint64_t x)
{
  /* Rounds up to the next highest power of two without branching and
   * is as fast as a BSR instruction on x86, see:
   *
   * https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
   */
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  return ++x;
}

/* Rounds up to the (power of two) alignment */
size_t pocl_align_value (size_t value, size_t alignment)
{
  if (value & (alignment-1))
    {
      value |= (alignment-1);
      ++value;
    }
  return value;
}

#if defined(_WIN32) || defined(HAVE_POSIX_MEMALIGN) || defined(__ANDROID__)    \
    || (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L))
#define HAVE_ALIGNED_ALLOC
#else
#error aligned malloc unavailable
#endif

static void*
pocl_memalign_alloc(size_t align_width, size_t size)
{
  void *ptr;
  int status;

#ifdef __ANDROID__
  return memalign (align_width, size);
#elif defined(HAVE_POSIX_MEMALIGN)
  status = posix_memalign (&ptr, align_width, size);
  return ((status == 0) ? ptr : NULL);
#elif defined(_MSC_VER)
  return _aligned_malloc(size, align_width);
#elif defined(__MINGW32__)
  return __mingw_aligned_malloc(size, align_width);
#elif (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L))
  return aligned_alloc (align_width, size);
#else
#error Cannot find aligned malloc
#endif
}

static void
pocl_memalign_free (void *ptr)
{
#ifdef __ANDROID__
  free (ptr);
#elif defined(HAVE_POSIX_MEMALIGN)
  free (ptr);
#elif defined(_MSC_VER)
  _aligned_free (ptr);
#elif defined(__MINGW32__)
  __mingw_aligned_free (ptr);
#elif (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L))
  free (ptr);
#else
#error Cannot find aligned malloc
#endif
}

void *
pocl_aligned_malloc (size_t alignment, size_t size)
{
  assert (alignment > 0);

  /* posix_memalign requires alignment to be at least sizeof(void *) */
  if (alignment < sizeof(void *))
    alignment = sizeof(void* );

  /* make sure that size is a multiple of alignment, as posix_memalign
   * does not perform this test, whereas aligned_alloc does */
  if ((size & (alignment - 1)) != 0)
    {
      size = size | (alignment - 1);
      size += 1;
    }

  return pocl_memalign_alloc (alignment, size);
}

void
pocl_aligned_free (void *ptr)
{
  pocl_memalign_free (ptr);
}

void
pocl_lock_events_inorder (cl_event ev1, cl_event ev2)
{
  assert (ev1 != ev2);
  assert (ev1->id != ev2->id);
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
  assert (ev1 != ev2);
  assert (ev1->id != ev2->id);
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


cl_int
pocl_create_event (cl_event *event,
                   cl_command_queue command_queue,
                   cl_command_type command_type,
                   pocl_buffer_migration_info *migration_infos,
                   cl_context context)
{
  static uint64_t event_id_counter = 0;

  if (context == NULL)
    return CL_INVALID_CONTEXT;

  assert (event != NULL);
  *event = pocl_mem_manager_new_event ();
  if (*event == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  (*event)->context = context;
  (*event)->queue = command_queue;
  if (command_queue)
    (*event)->profiling_available
      = (command_queue->properties & CL_QUEUE_PROFILING_ENABLE) ? 1 : 0;

  /* user events have a NULL command queue, don't retain it */
  if (command_queue)
    POname (clRetainCommandQueue) (command_queue);
  else
    POname (clRetainContext) (context);

  (*event)->command_type = command_type;
  (*event)->id = POCL_ATOMIC_INC (event_id_counter);
  (*event)->status = CL_QUEUED;

  if (command_type == CL_COMMAND_USER)
    POCL_ATOMIC_INC (uevent_c);
  else
    POCL_ATOMIC_INC (event_c);

  POCL_MSG_PRINT_EVENTS ("Created event %" PRIu64 " (%p) for Command %s\n",
                         (*event)->id, (*event),
                         pocl_command_to_str (command_type));

  return CL_SUCCESS;
}

static int
check_for_circular_dep (cl_event waiting_event, cl_event notifier_event)
{
  event_node *wait_list_item = NULL;
  LL_FOREACH (notifier_event->wait_list, wait_list_item)
  {
    if (wait_list_item->event == waiting_event)
      {
        POCL_MSG_ERR ("Circular event dependency detected!\n");
        abort ();
        return 1;
      }
    else if (check_for_circular_dep (waiting_event, wait_list_item->event))
      return 1;
  }
  return 0;
}

int
pocl_create_event_sync (cl_event waiting_event, cl_event notifier_event)
{
  event_node *notify_target = NULL;
  event_node *wait_list_item = NULL;

  if (notifier_event == NULL)
    return CL_SUCCESS;

  POCL_MSG_PRINT_EVENTS ("create event sync: waiting %" PRIu64
                         " , notifier %" PRIu64 "\n",
                         waiting_event->id, notifier_event->id);

  pocl_lock_events_inorder (waiting_event, notifier_event);

  assert (notifier_event->pocl_refcount != 0);
  assert (waiting_event != notifier_event);

  LL_FOREACH (waiting_event->wait_list, wait_list_item)
    {
      if (wait_list_item->event == notifier_event)
        {
          POCL_MSG_PRINT_EVENTS ("Skipping event sync creation \n");
          goto FINISH;
        }
    }

  if (notifier_event->status == CL_COMPLETE)
    goto FINISH;
  notify_target = pocl_mem_manager_new_event_node();
  wait_list_item = pocl_mem_manager_new_event_node();
  if (!notify_target || !wait_list_item)
    return CL_OUT_OF_HOST_MEMORY;

  /* check_for_circular_dep (waiting_event, notifier_event); */

  notify_target->event = waiting_event;
  wait_list_item->event = notifier_event;
  LL_PREPEND (notifier_event->notify_list, notify_target);
  LL_PREPEND (waiting_event->wait_list, wait_list_item);

  if (pocl_is_tracing_enabled ())
    {
      if (waiting_event->meta_data == NULL)
        waiting_event->meta_data = (pocl_event_md *) calloc (1, sizeof (pocl_event_md));
      pocl_event_md *md = waiting_event->meta_data;
      if (md->num_deps < MAX_EVENT_DEPS)
        md->dep_ids[md->num_deps++] = notifier_event->id;
    }

FINISH:
  pocl_unlock_events_inorder (waiting_event, notifier_event);
  return CL_SUCCESS;
}

/** Preallocate the buffers in the migration_infos lists on the destination
 * device.
 *
 * @return 0 If any of the allocations fails (can't run the command). */
static int
preallocate_buffers (cl_device_id dev,
                     pocl_buffer_migration_info *migration_infos)
{
  size_t i;
  int errcode;

  pocl_buffer_migration_info *migr_info = NULL;
  LL_FOREACH (migration_infos, migr_info)
  {
    cl_mem obj = migr_info->buffer;
    POCL_LOCK_OBJ (obj);
    pocl_mem_identifier *p = &obj->device_ptrs[dev->global_mem_id];
    /* Skip already allocated. */
    if (p->mem_ptr != NULL)
      {
        POCL_UNLOCK_OBJ (obj);
        continue;
      }

      assert (dev->ops->alloc_mem_obj);
      errcode = dev->ops->alloc_mem_obj (dev, obj, NULL);
      if (errcode != CL_SUCCESS) {
        POCL_MSG_ERR ("Failed to allocate %zu bytes on device %s\n", obj->size,
                      dev->short_name);
      }

      POCL_UNLOCK_OBJ (obj);
      if (errcode != CL_SUCCESS)
        return CL_FALSE;
    }

  return CL_TRUE;
}

cl_int
pocl_create_command_struct (_cl_command_node **cmd,
                            cl_command_queue command_queue,
                            cl_command_type command_type,
                            cl_event *event_p,
                            cl_uint num_events,
                            const cl_event *wait_list,
                            pocl_buffer_migration_info *migration_infos)
{
  unsigned i;
  cl_event *event = NULL;
  cl_int errcode = CL_SUCCESS;

  *cmd = pocl_mem_manager_new_command ();
  POCL_RETURN_ERROR_COND ((*cmd == NULL), CL_OUT_OF_HOST_MEMORY);

  (*cmd)->type = command_type;

  event = &((*cmd)->sync.event.event);
  errcode = pocl_create_event (event, command_queue, command_type,
                               migration_infos, command_queue->context);

  if (errcode != CL_SUCCESS)
    goto ERROR;
  (*event)->command_type = command_type;

  /* If host application wants this commands event
     one reference for the host and one for the runtime/driver. */
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

  (*cmd)->device = command_queue->device;
  (*cmd)->sync.event.event->command = (*cmd);

  /* Form event synchronizations based on the given wait list. */
  for (i = 0; i < num_events; ++i)
    {
      cl_event wle = wait_list[i];
      pocl_create_event_sync ((*event), wle);
    }
  POCL_MSG_PRINT_EVENTS (
      "Created immediate command struct: CMD %p (event %" PRIu64
      " / %p, type: %s)\n",
      *cmd, (*event)->id, *event, pocl_command_to_str (command_type));
  return CL_SUCCESS;

ERROR:
  pocl_mem_manager_free_command (*cmd);
  return errcode;
}

/**
 * Creates a command node for immediate execution and adds implicit data
 * migrations required by it.
 *
 * @param buffer_usage A linked list of buffer migration data for
 *                     the buffers accessed by the command.
 */
static cl_int
pocl_create_command_full (_cl_command_node **cmd,
                          cl_command_queue command_queue,
                          cl_command_type command_type,
                          cl_event *event_p,
                          cl_uint num_events,
                          const cl_event *wait_list,
                          pocl_buffer_migration_info *buffer_usage,
                          cl_mem_migration_flags mig_flags)
{
  cl_device_id dev = pocl_real_dev (command_queue->device);
  int err = CL_SUCCESS;
  size_t i;

  POCL_RETURN_ERROR_ON ((*dev->available == CL_FALSE), CL_INVALID_DEVICE,
                        "device is not available\n");

  if (buffer_usage != NULL)
    {
      /* If the buffer is an image backed by buffer storage,
         replace with actual storage. */
      pocl_buffer_migration_info *migr_info = NULL;
      LL_FOREACH (buffer_usage, migr_info)
        {
          migr_info->buffer = POCL_MEM_BS (migr_info->buffer);
        }

      if (!preallocate_buffers (dev, buffer_usage))
        return CL_OUT_OF_RESOURCES;
    }

  /* Waitlist here only contains the user-provided events.
     Migration events are added to the waitlist later. */
  err = pocl_create_command_struct (cmd, command_queue, command_type, event_p,
                                    num_events, wait_list, buffer_usage);

  if (err)
    return err;

  buffer_usage = pocl_convert_to_subbuffer_migrations (buffer_usage, &err);

  if (err)
    return err;

  (*cmd)->migr_infos = buffer_usage;

  /* The event (command) that will be set as the destination to all the
     migration commands' dependencies. */
  cl_event final_event = (*cmd)->sync.event.event;

  /* Retain once for every buffer. This is because we set every buffer's
   * "last event" to this, and then a next command enqueue that changes
   * the last event to something else or clReleaseMemObject will release it.
   */
  size_t num_buffers = 0;
  pocl_buffer_migration_info *mi = NULL;
  LL_FOREACH (buffer_usage, mi)
    {
      ++num_buffers;
    }

  if (num_buffers > 0)
    {
      POCL_LOCK_OBJ (final_event);
      final_event->pocl_refcount += num_buffers;
      POCL_MSG_PRINT_REFCOUNTS (
        "Event %zu refcount now %d due to %zu buffer(s) referring to it.\n",
        final_event->id, final_event->pocl_refcount, num_buffers);
      POCL_UNLOCK_OBJ (final_event);
    }

  cl_event *size_events = NULL;
  /* Temporary copy of the buffer list just for keeping track of which buffers
   * were migrated in the size buffer migration phase */
  int *already_migrated = NULL;
  if (num_buffers > 0)
    {
      size_events = alloca (sizeof (cl_event) * num_buffers);
      memset (size_events, 0, sizeof (cl_event) * num_buffers);

      already_migrated = alloca (sizeof (int) * num_buffers);
      memset (already_migrated, 0, sizeof (int) * num_buffers);
    }

  i = 0;
  /* Always migrate content size buffers first, if they exist */
  LL_FOREACH (buffer_usage, mi)
    {
      if (mi->buffer->size_buffer != NULL)
        {
          /* Bump "last event" refcount for content size buffers that weren't
           * explicitly given as dependencies. */
          int explicit = 0;
          pocl_buffer_migration_info *mi_j = NULL;
          int j = 0;
          LL_FOREACH (buffer_usage, mi_j)
            {
              if (mi_j->buffer == mi->buffer->size_buffer)
                {
                  already_migrated[j] = 1;
                  explicit = 1;
                  break;
                }
              ++j;
            }
          if (!explicit)
            {
              POname (clRetainEvent) (final_event);
            }

          pocl_create_migration_commands (
            dev, &size_events[i], final_event, mi->buffer->size_buffer,
            &(mi->buffer->size_buffer)->device_ptrs[dev->global_mem_id],
            mi->read_only, command_type, mig_flags,
            mi->buffer->size_buffer->size, NULL);
        }
      ++i;
    }

  cl_event prev_migr_event = NULL;

  i = 0;
  LL_FOREACH (buffer_usage, mi)
    {
      uint64_t migration_size = mi->buffer->size;

      /* If both a content buffer and its associated size buffer are
       * explicitly listed in buffers, the size buffer was already migrated
       * above. */
      if (already_migrated[i])
        {
          ++i;
          continue;
        }

      if (mi->buffer->size_buffer != NULL)
        {
          /* BLOCK until size buffer has been imported to host mem! No event
           * exists if host import is not needed. */
          if (size_events[i] != NULL)
            {
              cl_device_id d = size_events[i]->queue->device;
              POname (clWaitForEvents) (1, &size_events[i]);
              if (mi->buffer->size_buffer->mem_host_ptr != NULL)
                {
                  migration_size
                    = *(uint64_t *)mi->buffer->size_buffer->mem_host_ptr;
                }
              pocl_release_mem_host_ptr (mi->buffer->size_buffer);
              POname (clReleaseEvent) (size_events[i]);
              size_events[i] = NULL;
            }
        }

      cl_int err;
      /* Capture the last created migration command for chaining. */
      pocl_create_migration_commands (
        dev, NULL, final_event, mi->buffer,
        &mi->buffer->device_ptrs[dev->global_mem_id], mi->read_only,
        command_type, mig_flags, migration_size, &prev_migr_event);

      /* Hold the last updater events of the parent buffers so we can refer
         to the event in potential implicit sub-buffer migrations. */
      if (mi->buffer->parent == NULL && mi->buffer->sub_buffers != NULL
          && mi->buffer->last_updater != NULL)
        POname (clRetainEvent) (mi->buffer->last_updater);
      ++i;
    }

    LL_FOREACH (buffer_usage, mi)
      {
        /* The last events of the parent buffers can be now released as the
           sub-buffer references to the events should hold them alive. */
        if (mi->buffer->parent == NULL && mi->buffer->sub_buffers != NULL
            && mi->buffer->parent->last_updater != NULL)
          {
            POname (clReleaseEvent) (mi->buffer->last_updater);
            mi->buffer->last_updater = NULL;
          }
      }
    if (prev_migr_event != NULL)
      POname (clReleaseEvent) (prev_migr_event);
    return err;
}

cl_int
pocl_create_command_migrate (_cl_command_node **cmd,
                             cl_command_queue command_queue,
                             cl_mem_migration_flags flags,
                             cl_event *event_p,
                             cl_uint num_events,
                             const cl_event *wait_list,
                             pocl_buffer_migration_info *migration_infos)

{
  return pocl_create_command_full (
    cmd, command_queue, CL_COMMAND_MIGRATE_MEM_OBJECTS, event_p, num_events,
    wait_list, migration_infos, flags);
}

/**
 * Create a command node which as multiple buffers associated with it.
 *
 * The buffers can be due to the command taking them as arguments or
 * due to the need to implicitly migrate them for another reason.
 */
cl_int
pocl_create_command (_cl_command_node **cmd,
                     cl_command_queue command_queue,
                     cl_command_type command_type,
                     cl_event *event_p,
                     cl_uint num_events,
                     const cl_event *wait_list,
                     pocl_buffer_migration_info *migration_infos)
{
  return pocl_create_command_full (cmd, command_queue, command_type, event_p,
                                   num_events, wait_list, migration_infos, 0);
}

cl_int
pocl_cmdbuf_validate_queue_list (cl_uint num_queues,
                                 const cl_command_queue *queues)
{
  POCL_RETURN_ERROR_COND ((num_queues == 0), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND ((queues == NULL), CL_INVALID_VALUE);

  /* All queues must have the same OpenCL context */
  cl_context ref_ctx = queues[0]->context;

  for (unsigned i = 0; i < num_queues; ++i)
    {
      /* All queues must be valid Command queue objects */
      POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (queues[i])),
                              CL_INVALID_COMMAND_QUEUE);

      POCL_RETURN_ERROR_COND ((queues[i]->device == NULL),
                              CL_INVALID_COMMAND_QUEUE);

      POCL_RETURN_ERROR_COND ((queues[i]->context == NULL),
                              CL_INVALID_COMMAND_QUEUE);

      POCL_RETURN_ERROR_COND ((queues[i]->context != ref_ctx),
                              CL_INVALID_COMMAND_QUEUE);
    }

  return CL_SUCCESS;
}

cl_int
pocl_cmdbuf_choose_recording_queue (cl_command_buffer_khr command_buffer,
                                    cl_command_queue *command_queue)
{
  assert (command_queue != NULL);
  cl_command_queue q = *command_queue;

  POCL_RETURN_ERROR_COND ((q == NULL && command_buffer->num_queues != 1),
                          CL_INVALID_COMMAND_QUEUE);

  if (q)
    {
      POCL_RETURN_ERROR_COND (
          (command_buffer->queues[0]->context != q->context),
          CL_INVALID_CONTEXT);
      int queue_in_buffer = 0;
      for (unsigned i = 0; i < command_buffer->num_queues; ++i)
        {
          if (q == command_buffer->queues[i])
            queue_in_buffer = 1;
        }
      POCL_RETURN_ERROR_COND ((!queue_in_buffer), CL_INVALID_COMMAND_QUEUE);
    }
  else
    q = command_buffer->queues[0];

  *command_queue = q;
  return CL_SUCCESS;
}

cl_command_buffer_properties_khr
pocl_cmdbuf_get_property (cl_command_buffer_khr command_buffer,
                          cl_command_buffer_properties_khr name)
{
  for (unsigned i = 0; i < command_buffer->num_properties; ++i)
    {
      if (command_buffer->properties[2 * i] == name)
        return command_buffer->properties[2 * i + 1];
    }
  return 0;
}

/**
 * Create a command buffered command node.
 *
 * The node contains the minimum information to "clone" launchable
 * commands in clEnqueueCommandBufferKHR.c.
 */
cl_int
pocl_create_recorded_command (_cl_command_node **cmd,
                              cl_command_buffer_khr command_buffer,
                              cl_command_queue command_queue,
                              cl_command_type command_type,
                              cl_uint num_sync_points_in_wait_list,
                              const cl_sync_point_khr *sync_point_wait_list,
                              pocl_buffer_migration_info *buffer_usage)
{
  cl_int errcode = pocl_check_syncpoint_wait_list (
    command_buffer, num_sync_points_in_wait_list, sync_point_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  if (buffer_usage != NULL)
    {
      /* If the buffer is an image backed by buffer storage,
         replace with actual storage. */
      pocl_buffer_migration_info *migr_info = NULL;
      LL_FOREACH (buffer_usage, migr_info)
        if (migr_info->buffer->buffer)
          migr_info->buffer = migr_info->buffer->buffer;

      if (!preallocate_buffers (command_queue->device, buffer_usage))
        return CL_OUT_OF_RESOURCES;
    }

  *cmd = pocl_mem_manager_new_command ();
  POCL_RETURN_ERROR_COND ((*cmd == NULL), CL_OUT_OF_HOST_MEMORY);
  (*cmd)->type = command_type;
  (*cmd)->buffered = 1;

  /* pocl_cmdbuf_choose_recording_queue should have been called to ensure we
   * have a valid command queue, usually via CMDBUF_VALIDATE_COMMON_HANDLES
   * but at that time *cmd was not allocated at that time, so find the queue
   * index again here */
  for (unsigned i = 0; i < command_buffer->num_queues; ++i)
    {
      if (command_buffer->queues[i] == command_queue)
        (*cmd)->queue_idx = i;
    }

  (*cmd)->sync.syncpoint.num_sync_points_in_wait_list
    = num_sync_points_in_wait_list;
  if (num_sync_points_in_wait_list > 0)
    {
      cl_sync_point_khr *wait_list
        = malloc (sizeof (cl_sync_point_khr) * num_sync_points_in_wait_list);
      if (wait_list == NULL)
        {
          POCL_MEM_FREE (*cmd);
          return CL_OUT_OF_HOST_MEMORY;
        }
      memcpy (wait_list, sync_point_wait_list,
              sizeof (cl_sync_point_khr) * num_sync_points_in_wait_list);
      (*cmd)->sync.syncpoint.sync_point_wait_list = wait_list;
    }

  (*cmd)->migr_infos = buffer_usage;
  pocl_buffer_migration_info *migr_info = NULL;

  /* We need to retain the buffers as we expect them to be executed
     later. They are retained again for each executed instance in
     pocl_create_migration_commands() and those references are freed
     after the executed instance is freed.  This one is freed at
     command buffer free time. */
  LL_FOREACH (buffer_usage, migr_info)
    POname (clRetainMemObject) (migr_info->buffer);

  return CL_SUCCESS;
}

cl_int
pocl_command_record (cl_command_buffer_khr command_buffer,
                     _cl_command_node *cmd, cl_sync_point_khr *sync_point)
{
  POCL_LOCK (command_buffer->mutex);
  if (command_buffer->state != CL_COMMAND_BUFFER_STATE_RECORDING_KHR)
    {
      POCL_UNLOCK (command_buffer->mutex);
      return CL_INVALID_OPERATION;
    }
  LL_APPEND (command_buffer->cmds, cmd);
  if (sync_point != NULL)
    *sync_point = command_buffer->num_syncpoints + 1;
  command_buffer->num_syncpoints++;
  POCL_UNLOCK (command_buffer->mutex);
  return CL_SUCCESS;
}

/* call with node->sync.event.event UNLOCKED */
void pocl_command_enqueue (cl_command_queue command_queue,
                          _cl_command_node *node)
{
  cl_event event;

  POCL_LOCK_OBJ (command_queue);

  ++command_queue->command_count;

  /* In case of in-order queue, synchronize to the previously enqueued command,
     if available. */
  if (!(command_queue->properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE))
    {
      POCL_MSG_PRINT_EVENTS ("In-order Q; adding event syncs\n");
      if (command_queue->last_event.event)
        {
          pocl_create_event_sync (node->sync.event.event,
                                  command_queue->last_event.event);
        }
    }
  else if ((node->type == CL_COMMAND_BARRIER
            || node->type == CL_COMMAND_MARKER)
           && node->command.barrier.has_wait_list == 0)
    /* Command queue is out-of-order queue. If command type is a barrier, then
       synchronize to all previously enqueued commands to make sure they are
       executed before the barrier. */
    {
      POCL_MSG_PRINT_EVENTS ("Barrier; adding event syncs\n");
      DL_FOREACH (command_queue->events, event)
        {
          pocl_create_event_sync (node->sync.event.event, event);
        }
    }

  if (node->type == CL_COMMAND_BARRIER)
    command_queue->barrier = node->sync.event.event;
  else
    {
      if (command_queue->barrier)
        {
          pocl_create_event_sync (node->sync.event.event,
                                  command_queue->barrier);
        }
    }
  DL_APPEND (command_queue->events, node->sync.event.event);

  POCL_MSG_PRINT_EVENTS ("Pushed Event %" PRIu64 " to CQ %" PRIu64 ".\n",
                         node->sync.event.event->id, command_queue->id);
  command_queue->last_event.event = node->sync.event.event;
  POCL_UNLOCK_OBJ (command_queue);

  POCL_LOCK_OBJ (node->sync.event.event);
  assert (node->sync.event.event->status == CL_QUEUED);
  assert (command_queue == node->sync.event.event->queue);
  pocl_update_event_queued (node->sync.event.event);
  command_queue->device->ops->submit(node, command_queue);
  /* node->sync.event.event is unlocked by device_ops->submit */
}

int
pocl_alloc_or_retain_mem_host_ptr (cl_mem mem)
{
  if (mem->mem_host_ptr == NULL)
    {
      size_t align = max (mem->context->min_buffer_alignment, 16);
      mem->mem_host_ptr = pocl_aligned_malloc (align, mem->size);
      if (mem->mem_host_ptr == NULL)
        return -1;
      mem->mem_host_ptr_version = 0;
      mem->mem_host_ptr_refcount = 0;
    }
  ++mem->mem_host_ptr_refcount;
  return 0;
}

int
pocl_release_mem_host_ptr (cl_mem mem)
{
  assert (mem->mem_host_ptr_refcount > 0);
  --mem->mem_host_ptr_refcount;
  if (mem->mem_host_ptr_refcount == 0 && mem->mem_host_ptr != NULL)
    {
      pocl_aligned_free (mem->mem_host_ptr);
      mem->mem_host_ptr = NULL;
      mem->mem_host_ptr_version = 0;
    }
  return 0;
}

/* call (and return) with node->sync.event.event locked */
void
pocl_command_push (_cl_command_node *node,
                   _cl_command_node **ready_list,
                   _cl_command_node **pending_list)
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
  if (pocl_command_is_ready (node->sync.event.event))
    {
      pocl_update_event_submitted (node->sync.event.event);
      CDL_PREPEND ((*ready_list), node);
    }
  else
    {
      CDL_PREPEND ((*pending_list), node);
    }
}

static void
pocl_unmap_command_finished (cl_device_id dev, _cl_command_t *cmd)
{
  pocl_mem_identifier *mem_id = NULL;
  cl_mem mem = NULL;
  mem = POCL_MEM_BS (cmd->unmap.buffer);
  mem_id = &POCL_MEM_BS (mem)->device_ptrs[dev->global_mem_id];

  mem_mapping_t *map = cmd->unmap.mapping;
  POCL_LOCK_OBJ (mem);
  assert (map->unmap_requested > 0);
  if (dev->ops->free_mapping_ptr)
    dev->ops->free_mapping_ptr (dev->data, mem_id, mem, map);
  DL_DELETE (mem->mappings, map);
  mem->map_count--;
  POCL_MEM_FREE (map);
  POCL_UNLOCK_OBJ (mem);
}

void
pocl_ndrange_node_cleanup (_cl_command_node *node)
{
  cl_uint i;
  for (i = 0; i < node->command.run.kernel->meta->num_args; ++i)
    {
      pocl_aligned_free (node->command.run.arguments[i].value);
    }
  POCL_MEM_FREE (node->command.run.arguments);
  POname(clReleaseKernel)(node->command.run.kernel);
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
  POCL_RETURN_ERROR_ON((byte_offset_end >= buffer_size), CL_INVALID_VALUE,
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
  cl_device_id *out = (cl_device_id *)calloc (num, sizeof (cl_device_id));
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

int
pocl_device_supports_builtin_kernel (cl_device_id dev, const char *kernel_name)
{
  if (kernel_name == NULL)
    return 0;

  if (dev->builtin_kernel_list == NULL)
    return 0;

  for (unsigned i = 0; i < dev->num_builtin_kernels; ++i)
    {
      if (strcmp (dev->builtin_kernels_with_version[i].name, kernel_name) == 0)
        {
          return 1;
        }
    }

  return 0;
}

static void
image_format_union (const cl_image_format *dev_formats,
                    cl_uint               num_dev_formats,
                    cl_image_format       **context_formats,
                    cl_uint               *num_context_formats)
{
  if ((dev_formats == NULL) || (num_dev_formats == 0))
    return;

  if ((*num_context_formats == 0) || (*context_formats == NULL))
    {
      // alloc & copy
      *context_formats = (cl_image_format *)malloc (sizeof (cl_image_format)
                                                    * num_dev_formats);
      memcpy (*context_formats, dev_formats,
              sizeof (cl_image_format) * num_dev_formats);
      *num_context_formats = num_dev_formats;
    }
  else
    {
      // realloc & merge
      cl_uint i, j;
      cl_uint ncf = *num_context_formats;
      size_t size = sizeof (cl_image_format) * (num_dev_formats + ncf);
      cl_image_format *ctf
          = (cl_image_format *)realloc (*context_formats, size);
      assert (ctf);
      for (i = 0; i < num_dev_formats; ++i)
        {
          for (j = 0; j < ncf; ++j)
            if (memcmp (ctf + j, dev_formats + i, sizeof (cl_image_format))
                == 0)
              break;
          if (j < ncf)
            {
              // format already in context, skip
              continue;
            }
          else
            {
              memcpy (ctf + ncf, dev_formats + i, sizeof (cl_image_format));
              ++ncf;
            }
        }
      *context_formats = ctf;
      *num_context_formats = ncf;
    }
}

/* Setup certain info about context that comes up later in API calls */
int
pocl_setup_context (cl_context context)
{
  unsigned i, j;
  int err;
  size_t alignment = context->devices[0]->mem_base_addr_align;
  context->max_mem_alloc_size = 0;
  context->svm_allocdev = NULL;
  assert (context->default_queues);

  memset (context->image_formats, 0, sizeof (void *) * NUM_OPENCL_IMAGE_TYPES);
  memset (context->num_image_formats, 0,
          sizeof (cl_uint) * NUM_OPENCL_IMAGE_TYPES);

  for(i=0; i<context->num_devices; i++)
    {
      cl_device_id dev = context->devices[i];
      if (dev->svm_allocation_priority > 0)
        {
          if (context->svm_allocdev == NULL
              || context->svm_allocdev->svm_allocation_priority
                     < dev->svm_allocation_priority)
            {
              context->svm_allocdev = dev;
              if (dev->ops->usm_alloc && dev->ops->usm_free)
              {
                  context->usm_allocdev = dev;
              }
            }
        }

      if (dev->mem_base_addr_align < alignment)
        alignment = dev->mem_base_addr_align;

      if (dev->max_mem_alloc_size
          > context->max_mem_alloc_size)
        context->max_mem_alloc_size =
            dev->max_mem_alloc_size;

      if (dev->image_support == CL_TRUE)
        {
          for (j = 0; j < NUM_OPENCL_IMAGE_TYPES; ++j)
            image_format_union (
                dev->image_formats[j],
                dev->num_image_formats[j],
                &context->image_formats[j], &context->num_image_formats[j]);
        }

      if (dev->ops->init_context)
        dev->ops->init_context (dev, context);

      cl_command_queue_properties props
        = CL_QUEUE_HIDDEN | CL_QUEUE_PROFILING_ENABLE;
      if (dev->on_host_queue_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
        props |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
      context->default_queues[i]
        = POname (clCreateCommandQueue) (context, dev, props, &err);
      assert (err == CL_SUCCESS);
      assert (context->default_queues[i]);
    }

  assert (alignment > 0);
  context->min_buffer_alignment = alignment;
  return CL_SUCCESS;
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
          POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (event_wait_list[i])),
                                  CL_INVALID_EVENT_WAIT_LIST);
          POCL_RETURN_ERROR_COND (
              (event_wait_list[i]->context != command_queue->context),
              CL_INVALID_CONTEXT);
        }
    }

  return CL_SUCCESS;
}

int
pocl_check_syncpoint_wait_list (cl_command_buffer_khr command_buffer,
                                cl_uint num_sync_points_in_wait_list,
                                const cl_sync_point_khr *sync_point_wait_list)
{
  POCL_RETURN_ERROR_COND (
      (num_sync_points_in_wait_list > 0 && sync_point_wait_list == NULL),
      CL_INVALID_SYNC_POINT_WAIT_LIST_KHR);
  POCL_RETURN_ERROR_COND (
      (num_sync_points_in_wait_list == 0 && sync_point_wait_list != NULL),
      CL_INVALID_SYNC_POINT_WAIT_LIST_KHR);

  POCL_LOCK (command_buffer->mutex);
  cl_uint next_syncpoint = command_buffer->num_syncpoints + 1;
  POCL_UNLOCK (command_buffer->mutex);

  POCL_RETURN_ERROR_ON ((next_syncpoint == 0), CL_OUT_OF_RESOURCES,
                        "Too many commands in buffer\n");

  for (unsigned i = 0; i < num_sync_points_in_wait_list; ++i)
    {
      POCL_RETURN_ERROR_COND ((sync_point_wait_list[i] == 0),
                              CL_INVALID_SYNC_POINT_WAIT_LIST_KHR);
      POCL_RETURN_ERROR_COND ((sync_point_wait_list[i] >= next_syncpoint),
                              CL_INVALID_SYNC_POINT_WAIT_LIST_KHR);
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
}

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
    case CL_COMMAND_COMMAND_BUFFER_KHR:
      return "command_buffer_khr";
    }

  return "unknown";
}

int
pocl_run_command (const char **args)
{
  POCL_MSG_PRINT_INFO ("Launching: %s\n", args[0]);
#if defined(HAVE_FORK)
  pid_t p = fork ();
  if (p == 0)
    {
      return execv (args[0], (char *const *)args);
    }
  else
    {
      if (p < 0)
        return EXIT_FAILURE;
      int status;
      int ret;
      do
        {
          ret = waitpid (p, &status, 0);
        }
      while (ret == -1 && errno == EINTR);
      if (ret < 0)
        {
          POCL_MSG_ERR ("pocl: waitpid() failed.\n");
          return EXIT_FAILURE;
        }
      if (WIFEXITED (status))
        return WEXITSTATUS (status);
      else if (WIFSIGNALED (status))
        return WTERMSIG (status);
      else
        return EXIT_FAILURE;
    }
#elif _WIN32
  STARTUPINFO si;
  ZeroMemory(&si, sizeof(si));
  si.cb = sizeof(si);
  PROCESS_INFORMATION pi;
  ZeroMemory(&pi, sizeof(pi));
  DWORD dwProcessFlags = 0;
  char * cmd = strdup(args[0]);
  int p = CreateProcess(NULL, cmd, NULL, NULL, 1, dwProcessFlags, NULL, NULL, &si, &pi) != 0;
  if (!p)
    return EXIT_FAILURE;
  DWORD waitRc = WaitForSingleObject(pi.hProcess, INFINITE);
  if (waitRc == WAIT_FAILED)
    return EXIT_FAILURE;
  DWORD exit_code = 0;
  p = GetExitCodeProcess(pi.hProcess, &exit_code) != 0;
  if (!p)
    return EXIT_FAILURE;
  return exit_code;
#else
#error Must have fork() or vfork() or Win32 CreateProcess
#endif
}

#if defined(HAVE_FORK)
int
pocl_run_command_capture_output (char *capture_string,
                                 size_t *captured_bytes,
                                 const char **args)
{
  POCL_MSG_PRINT_INFO ("Launching: %s\n", args[0]);

  int in[2];
  int out[2];
  pipe (in);
  pipe (out);

  pid_t p = fork ();
  if (p == 0)
    {
      close (in[1]);
      close (out[0]);

      dup2 (in[0], STDIN_FILENO);
      dup2 (out[1], STDOUT_FILENO);
      dup2 (out[1], STDERR_FILENO);

      return execv (args[0], (char *const *)args);
    }
  else
    {
      if (p < 0)
        return EXIT_FAILURE;

      close (in[0]);
      close (out[1]);

      ssize_t r = 0;
      size_t total_bytes = 0;
      size_t capture_limit = *captured_bytes;
      char buf[4096];

      while ((r = read (out[0], buf, 4096)) > 0)
        {
          if (total_bytes + r > capture_limit)
            break;
          memcpy (capture_string + total_bytes, buf, r);
          total_bytes += r;
        }
      if (total_bytes > capture_limit)
        total_bytes = capture_limit;

      capture_string[total_bytes] = 0;
      *captured_bytes = total_bytes;

      int status;
      int ret;
      do {
        ret = waitpid (p, &status, 0);
      } while (ret == -1 && errno == EINTR);
      if (ret < 0)
        {
          POCL_MSG_ERR ("pocl: waitpid() failed.\n");
          return EXIT_FAILURE;
        }

      close (out[0]);
      close (in[1]);

      if (WIFEXITED (status))
        return WEXITSTATUS (status);
      else if (WIFSIGNALED (status))
        return WTERMSIG (status);
      else
        return EXIT_FAILURE;
    }
}
#endif // fork

void
pocl_update_event_queued (cl_event event)
{
  assert (event != NULL);

  event->status = CL_QUEUED;
  cl_command_queue cq = event->queue;
  if ((cq->properties & CL_QUEUE_PROFILING_ENABLE)
      && (cq->device->has_own_timer == 0))
    event->time_queue = pocl_gettimemono_ns ();

  POCL_MSG_PRINT_EVENTS ("Event queued: %" PRIu64 "\n", event->id);

  if (cq->device->ops->update_event)
    cq->device->ops->update_event (cq->device, event);
  pocl_event_updated (event, CL_QUEUED);
}

// event locked
void
pocl_update_event_submitted (cl_event event)
{
  assert (event != NULL);
  assert (event->status == CL_QUEUED);

  cl_command_queue cq = event->queue;
  event->status = CL_SUBMITTED;
  if ((cq->properties & CL_QUEUE_PROFILING_ENABLE)
      && (cq->device->has_own_timer == 0))
    event->time_submit = pocl_gettimemono_ns ();

  POCL_MSG_PRINT_EVENTS ("Event submitted: %" PRIu64 "\n", event->id);

  if (cq->device->ops->update_event)
    cq->device->ops->update_event (cq->device, event);
  pocl_event_updated (event, CL_SUBMITTED);
}

void
pocl_update_event_running_unlocked (cl_event event)
{
  assert (event != NULL);
  assert (event->status == CL_SUBMITTED);

  cl_command_queue cq = event->queue;
  event->status = CL_RUNNING;
  if ((cq->properties & CL_QUEUE_PROFILING_ENABLE)
      && (cq->device->has_own_timer == 0))
    event->time_start = pocl_gettimemono_ns ();

  POCL_MSG_PRINT_EVENTS ("Event running: %" PRIu64 "\n", event->id);

  if (cq->device->ops->update_event)
    cq->device->ops->update_event (cq->device, event);
  pocl_event_updated (event, CL_RUNNING);
}

void
pocl_update_event_running (cl_event event)
{
  POCL_LOCK_OBJ (event);
  pocl_update_event_running_unlocked (event);
  POCL_UNLOCK_OBJ (event);
}

/* Note: this must be kept in sync with pocl_copy_command_node */
static void pocl_free_event_node (_cl_command_node *node)
{
  switch (node->type)
    {
    case CL_COMMAND_NDRANGE_KERNEL:
    case CL_COMMAND_TASK:
      pocl_ndrange_node_cleanup (node);
      break;

    case CL_COMMAND_FILL_BUFFER:
      pocl_aligned_free (node->command.memfill.pattern);
      break;

    case CL_COMMAND_SVM_MEMFILL:
      pocl_aligned_free (node->command.svm_fill.pattern);
      break;

    case CL_COMMAND_SVM_MEMFILL_RECT_POCL:
      pocl_aligned_free (node->command.svm_fill_rect.pattern);
      break;

    case CL_COMMAND_NATIVE_KERNEL:
      POCL_MEM_FREE (node->command.native.args);
      break;

    case CL_COMMAND_UNMAP_MEM_OBJECT:
      pocl_unmap_command_finished (node->device, &node->command);
      break;

    case CL_COMMAND_SVM_MIGRATE_MEM:
      POCL_MEM_FREE (node->command.svm_migrate.sizes);
      POCL_MEM_FREE (node->command.svm_migrate.svm_pointers);
      break;

    case CL_COMMAND_SVM_FREE:
      POCL_MEM_FREE (node->command.svm_free.svm_pointers);
      break;
    }
  pocl_mem_manager_free_command (node);
}

/**
 * Copies relevant parts of a command node for command buffer execution
 * purposes.
 *
 * The "relevant parts" include the information needed for execution and
 * independent freeing of the command node resources after finishing. Doesn't
 * touch the next/prev pointers, for instance. The command buffer (default)
 * execution happens in clEnqueueCommandBufferKHR.
 */
int
pocl_copy_command_node (_cl_command_node *dst_node, _cl_command_node *src_node)
{
  memcpy (&dst_node->command, &src_node->command, sizeof (_cl_command_t));
  dst_node->program_device_i = src_node->program_device_i;

  /* Copy variables that are freed when the command finishes. */
  switch (src_node->type)
    {
    case CL_COMMAND_NDRANGE_KERNEL:
    case CL_COMMAND_TASK:
      POname (clRetainKernel) (src_node->command.run.kernel);
      /* Note: this must use the arguments stored in the src_node,
       * NOT the ones in kernel->dyn_arguments; these might differ,
       * because the user could clSetKernelArg() right after
       * clCommandNDRangeKernelKHR(). */
      int errcode = pocl_kernel_copy_args (src_node->command.run.kernel,
                                           src_node->command.run.arguments,
                                           &dst_node->command.run);
      if (errcode != CL_SUCCESS)
        return CL_OUT_OF_HOST_MEMORY;
      break;

    case CL_COMMAND_FILL_BUFFER:
      dst_node->command.memfill.pattern
          = pocl_aligned_malloc (src_node->command.memfill.pattern_size,
                                 src_node->command.memfill.pattern_size);
      if (dst_node->command.memfill.pattern == NULL)
        return CL_OUT_OF_HOST_MEMORY;
      memcpy (dst_node->command.memfill.pattern,
              src_node->command.memfill.pattern,
              src_node->command.memfill.pattern_size);
      break;

    case CL_COMMAND_SVM_MEMFILL:
      dst_node->command.svm_fill.pattern
          = pocl_aligned_malloc (src_node->command.svm_fill.pattern_size,
                                 src_node->command.svm_fill.pattern_size);
      if (dst_node->command.svm_fill.pattern == NULL)
        return CL_OUT_OF_HOST_MEMORY;
      memcpy (dst_node->command.svm_fill.pattern,
              src_node->command.svm_fill.pattern,
              src_node->command.svm_fill.pattern_size);
      break;

    /* These cases are currently not handled in pocl_copy_event_node,
     * because there is no command buffer equivalent of these nodes. */
    case CL_COMMAND_NATIVE_KERNEL:
    case CL_COMMAND_UNMAP_MEM_OBJECT:
    case CL_COMMAND_SVM_MIGRATE_MEM:
    case CL_COMMAND_SVM_FREE:
      assert (0 && "Unimplemented.");

    default:
      break;
    }

  return CL_SUCCESS;
}

/* Status can be complete or failed (<0). */
void
pocl_update_event_finished (cl_int status, const char *func, unsigned line,
                            cl_event event, const char *msg)
{
  assert (event != NULL);
  assert (event->queue != NULL);
  assert (event->status > CL_COMPLETE);
  int notify_cmdq = CL_FALSE;
  cl_command_buffer_khr command_buffer = NULL;
  _cl_command_node *node = NULL;

  cl_command_queue cq = event->queue;
  POCL_LOCK_OBJ (cq);
  POCL_LOCK_OBJ (event);
  if ((cq->properties & CL_QUEUE_PROFILING_ENABLE)
      && (cq->device->has_own_timer == 0))
    event->time_end = pocl_gettimemono_ns ();

  struct pocl_device_ops *ops = cq->device->ops;
  event->status = status;
  if (cq->device->ops->update_event)
    ops->update_event (cq->device, event);

  if (status == CL_COMPLETE)
    POCL_MSG_PRINT_EVENTS ("%s: Command complete, event %" PRIu64 "\n",
                           cq->device->short_name, event->id);
  else
    POCL_MSG_PRINT_EVENTS ("%s: Command FAILED, event %" PRIu64 "\n",
                           cq->device->short_name, event->id);

  assert (cq->command_count > 0);
  --cq->command_count;
  if (cq->barrier == event)
    cq->barrier = NULL;
  if (cq->last_event.event == event)
    cq->last_event.event = NULL;
  DL_DELETE (cq->events, event);

  if (ops->notify_cmdq_finished && (cq->command_count == 0) && cq->notification_waiting_threads) {
    notify_cmdq = CL_TRUE;
  }

  POCL_UNLOCK_OBJ (cq);
  /* note that we must unlock the CmqQ before calling pocl_event_updated,
   * because it calls event callbacks, which can have calls to
   * clEnqueueSomething() */
  pocl_event_updated (event, status);
  command_buffer = event->command_buffer;
  node = event->command;
  event->command = NULL;
  POCL_UNLOCK_OBJ (event);

  /* NOTE: this must be called before we call broadcast.
   * Reason: pocl_ndrange_node_cleanup releases kernel; broadcast makes the next
   * event runnable. Assume pocl_ndrange_node_cleanup is not called before
   * pocl_broadcast; then with this sequence of calls:
   * clBuildProgram(p)
   * kernel = clCreateKernel(p)
   * clEnqueueNDRange(kernel)
   * clFinish()
   *   ... pocl_update_event_finished()
   *      ... pocl_broadcast
   *      <this cpu thread gets descheduled here, but next events are launched>
   *      ... pocl_ndrange_node_cleanup
   * clReleaseKernel(kernel)
   * clBuildProgram(rebuild the same program again)
   * ...
   * since cleanup is still not called at clBuildProgram, this will cause the
   * clBuildProgram to fail with CL_INVALID_OPERATION(program still has kernels)
   * this happens with CTS test "compiler", subtests: options_build_macro,
   * options_build_macro_existence, options_denorm_cache */
  if (node)
  {
    pocl_free_event_node (node);
  }

  /* NOTE this must be called before we call broadcast, see above */
  if (event->reset_command_buffer)
  {
    assert (command_buffer);
    POCL_LOCK (command_buffer->mutex);
    command_buffer->pending -= 1;
    if (command_buffer->pending == 0)
        command_buffer->state = CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR;
    POCL_UNLOCK (command_buffer->mutex);
    POname (clReleaseCommandBufferKHR) (command_buffer);
  }

  ops->broadcast (event);

#ifdef ENABLE_REMOTE_CLIENT
  /* With remote being asynchronous it is possible that an event completion
   * signal is received before some of its dependencies. Therefore this event
   * has to be removed from the notify lists of any remaining events in the
   * wait list.
   *
   * Mind the acrobatics of trying to avoid races with pocl_broadcast and
   * pocl_create_event_sync. */
  event_node *tmp;
  POCL_LOCK_OBJ (event);
  while ((tmp = event->wait_list))
    {
      cl_event notifier = tmp->event;
      POCL_UNLOCK_OBJ (event);
      pocl_lock_events_inorder (notifier, event);
      if (tmp != event->wait_list)
        {
          pocl_unlock_events_inorder (notifier, event);
          POCL_LOCK_OBJ (event);
          continue;
        }
      event_node *tmp2;
      LL_FOREACH (notifier->notify_list, tmp2)
      {
        if (tmp2->event == event)
          {
            LL_DELETE (notifier->notify_list, tmp2);
            pocl_mem_manager_free_event_node (tmp2);
            break;
          }
      }
      LL_DELETE (event->wait_list, tmp);
      pocl_unlock_events_inorder (notifier, event);
      pocl_mem_manager_free_event_node (tmp);
      POCL_LOCK_OBJ (event);
    }
  POCL_UNLOCK_OBJ (event);
#endif

#ifdef POCL_DEBUG_MESSAGES
  if (msg != NULL)
    {
      pocl_debug_print_duration (
          func, line, msg, (uint64_t) (event->time_end - event->time_start));
    }
#endif

  POCL_LOCK_OBJ (event);
  if (ops->notify_event_finished)
    ops->notify_event_finished (event);
  POCL_UNLOCK_OBJ (event);
  POname (clReleaseEvent) (event);

  if (notify_cmdq) {
    POCL_LOCK_OBJ (cq);
    ops->notify_cmdq_finished (cq);
    POCL_UNLOCK_OBJ (cq);
  }
}

void
pocl_update_event_failed (const char *func,
                          unsigned line,
                          cl_event event,
                          const char *msg)
{
  pocl_update_event_finished (CL_FAILED, func, line, event, msg);
}

void
pocl_update_event_failed_locked (cl_event event)
{
  POCL_UNLOCK_OBJ (event);
  pocl_update_event_finished (CL_FAILED, NULL, 0, event, NULL);
  POCL_LOCK_OBJ (event);
}

void
pocl_update_event_device_lost (cl_event event)
{
  POCL_UNLOCK_OBJ (event);
  pocl_update_event_finished (CL_DEVICE_NOT_AVAILABLE, NULL, 0, event, NULL);
  POCL_LOCK_OBJ (event);
}

void
pocl_update_event_complete (const char *func, unsigned line,
                            cl_event event, const char *msg)
{
  pocl_update_event_finished (CL_COMPLETE, func, line, event, msg);
}


/* SPIR-V magic header */
#define SPIRV_MAGIC 0x07230203U
/* Opcode for capability used by module */
#define OpCapab 0x00020011
/* execution model = Kernel is used by OpenCL SPIR-V modules */
#define KernelExecModel 0x6
/* execution model = Shader is used by Vulkan SPIR-V modules */
#define ShaderExecModel 0x1

static int
bitcode_is_spirv_execmodel (const char *bitcode, size_t size, uint32_t type)
{
  const uint32_t *bc32 = (const uint32_t *)bitcode;
  unsigned location = 0;
  uint32_t header_magic = htole32 (bc32[location++]);

  if ((size < 20) || (header_magic != SPIRV_MAGIC))
    return 0;

  // skip version, generator, bound, schema
  location += 4;
  int is_type = 0;
  uint32_t value, instruction;
  instruction = htole32 (bc32[location++]);
  value = htole32 (bc32[location++]);
  while (instruction == OpCapab && location < (size / 4))
    {
      if (value == type)
        return 1;
      instruction = htole32 (bc32[location++]);
      value = htole32 (bc32[location++]);
    }

  return 0;
}

int
pocl_bitcode_is_spirv_execmodel_kernel (const char *bitcode, size_t size)
{
  return bitcode_is_spirv_execmodel (bitcode, size, KernelExecModel);
}

int
pocl_bitcode_is_spirv_execmodel_shader (const char *bitcode, size_t size)
{
  return bitcode_is_spirv_execmodel (bitcode, size, ShaderExecModel);
}

int
pocl_device_is_associated_with_kernel (cl_device_id device, cl_kernel kernel)
{
  unsigned i;
  int found_it = 0;
  for (i = 0; i < kernel->context->num_devices; i++)
    if (pocl_real_dev (device) == kernel->context->devices[i])
      {
        found_it = 1;
        break;
      }

  return found_it;
}

/*
 * search for an unused ASCII character in options,
 * to be used to replace whitespaces within double quoted substrings
 */
static int
pocl_find_unused_char (const char *options, char *replace_me)
{
  for (int y = 35; y < 128; y++)
  {
    if (strchr (options, (char) y) == NULL)
    {
      *replace_me = (char) y;
      return 0;
    }
  }

  return -1;
}

int
pocl_escape_quoted_whitespace (char *temp_options, char *replace_me)
{
  /* searching for double quote in temp_options */
  if (strchr (temp_options, '"') != NULL)
  {
    size_t replace_cnt = 0;

    int in_substring = -1;

    /* scan for double quoted substring */
    for (size_t x = 0; x < strlen (temp_options); x++)
    {
      if (temp_options[x] == '"')
      {
        if (in_substring == -1)
        {
          /* enter in double quoted substring */
          in_substring = 0;
          continue;
        }

        /* exit from double quoted substring */
        in_substring = -1;
        continue;
      }

      /* search for whitespaces in substring */
      if (in_substring == 0)
      {
        if (temp_options[x] == ' ')
        {
          /* at first need, get an unused char */
          if (replace_cnt == 0)
          {
            if (pocl_find_unused_char (temp_options, replace_me) == -1)
            {
              /* no replace, no party */
              return -1;
            }
          }

          /* replace whitespace with unused char */
          temp_options[x] = *replace_me;
          replace_cnt++;
        }
      }
    }
  }

  return 0;
}

/* returns private datadir, possibly using relative path to libpocl sharedlib */
int pocl_get_private_datadir(char* private_datadir)
{
/* pocl_dynlib_pathname() is not implemented for LLVM dynlib */
#ifndef ENABLE_LLVM_PLATFORM_SUPPORT
  const char *Path = pocl_dynlib_pathname ((void *)pocl_get_private_datadir);
  if (Path)
    {
      strncpy (private_datadir, Path, POCL_MAX_PATHNAME_LENGTH);
      char *last_slash = strrchr (private_datadir, '/');
      if (last_slash)
        {
          ++last_slash;
          *last_slash = 0;
          strcat (private_datadir, POCL_INSTALL_PRIVATE_DATADIR_REL);
          return 0;
        }
        else
          return -1;
    }
#endif
    strcpy (private_datadir, POCL_INSTALL_PRIVATE_DATADIR);
    return 0;
}

/* returns path to a file from either the PoCL's source directory
 * (if POCL_BUILDING=1), or PoCL's private datadir (if POCL_BUILDING=0)
 * each arg (if not empty) should start with '/' but NOT end with it
 */
int pocl_get_srcdir_or_datadir (char* path,
                                const char* srcdir_suffix,
                                const char* datadir_suffix,
                                const char* filename)
{
  path[0] = 0;
#ifdef ENABLE_POCL_BUILDING
  if (pocl_get_bool_option ("POCL_BUILDING", 0))
    {
      strcat(path, SRCDIR);
      strcat(path, srcdir_suffix);
      strcat(path, filename);
    }
  else
#endif
    {
      if (pocl_get_private_datadir(path)) return -1;
      strcat(path, datadir_suffix);
      strcat(path, filename);
    }

  return 0;
}


void
pocl_str_toupper(char *out, const char *in)
{
  int i;

  for (i = 0; in[i] != '\0'; i++)
    out[i] = toupper(in[i]);
  out[i] = '\0';
}

char *
pocl_strcatdup_v (size_t num_strs, const char **strs)
{
  assert ((strs || !num_strs) && "strs is NULL while num_strs > 0!");
  switch (num_strs)
    {
    default:
      break;
    case 0:
      return NULL;
    case 1:
      return strdup (strs[0]);
    }

  size_t new_size = 1; /* Place for NULL. */
  for (size_t i = 0; i < num_strs; i++)
    {
      assert (strs[i]);
      new_size += strlen (strs[i]);
    }

  char *new_str = calloc (new_size, 1);
  if (new_str == NULL)
    return NULL;
  for (size_t i = 0; i < num_strs; i++)
    strcat (new_str, strs[i]);
  return new_str;
}

void
pocl_str_tolower(char *out, const char *in)
{
  int i;

  for (i = 0; in[i] != '\0'; i++)
    out[i] = tolower(in[i]);
  out[i] = '\0';
}

const char *
pocl_str_append (const char **dst, const char *src)
{
  assert (src);
  assert (dst && *dst);
  unsigned src_len = strlen (src);
  unsigned dst_len = strlen (*dst);
  char *new_dst = calloc (dst_len + src_len + 1, 1);
  if (new_dst == NULL)
    return NULL;
  strncpy (new_dst, *dst, dst_len);
  strncpy (new_dst + dst_len, src, src_len);
  const char *old_dst = *dst;
  *dst = new_dst;
  return old_dst;
}


int
pocl_fill_aligned_buf_with_pattern (void *__restrict__ ptr, size_t offset,
                                    size_t size,
                                    const void *__restrict__ pattern,
                                    size_t pattern_size)
{
  size_t i;
  unsigned j;

  /* memfill size is in bytes, we wanto make it into elements */
  size /= pattern_size;
  offset /= pattern_size;

  switch (pattern_size)
    {
    case 1:
      {
        uint8_t *p = (uint8_t *)ptr + offset;
        for (i = 0; i < size; i++)
          p[i] = *(uint8_t *)pattern;
      }
      break;
    case 2:
      {
        uint16_t *p = (uint16_t *)ptr + offset;
        for (i = 0; i < size; i++)
          p[i] = *(uint16_t *)pattern;
      }
      break;
    case 4:
      {
        uint32_t *p = (uint32_t *)ptr + offset;
        for (i = 0; i < size; i++)
          p[i] = *(uint32_t *)pattern;
      }
      break;
    case 8:
      {
        uint64_t *p = (uint64_t *)ptr + offset;
        for (i = 0; i < size; i++)
          p[i] = *(uint64_t *)pattern;
      }
      break;
    case 16:
      {
        uint64_t *p = (uint64_t *)ptr + (offset << 1);
        for (i = 0; i < size; i++)
          for (j = 0; j < 2; j++)
            p[(i << 1) + j] = *((uint64_t *)pattern + j);
      }
      break;
    case 32:
      {
        uint64_t *p = (uint64_t *)ptr + (offset << 2);
        for (i = 0; i < size; i++)
          for (j = 0; j < 4; j++)
            p[(i << 2) + j] = *((uint64_t *)pattern + j);
      }
      break;
    case 64:
      {
        uint64_t *p = (uint64_t *)ptr + (offset << 3);
        for (i = 0; i < size; i++)
          for (j = 0; j < 8; j++)
            p[(i << 3) + j] = *((uint64_t *)pattern + j);
      }
      break;
    case 128:
      {
        uint64_t *p = (uint64_t *)ptr + (offset << 4);
        for (i = 0; i < size; i++)
          for (j = 0; j < 16; j++)
            p[(i << 4) + j] = *((uint64_t *)pattern + j);
      }
      break;
    default:
      assert (0 && "Invalid pattern size");
      return -1;
    }
  return 0;
}

void
pocl_free_kernel_metadata (cl_program program, unsigned kernel_i)
{
  pocl_kernel_metadata_t *meta = &program->kernel_meta[kernel_i];
  unsigned j;
  POCL_MEM_FREE (meta->attributes);
  POCL_MEM_FREE (meta->name);
  for (j = 0; j < meta->num_args; ++j)
    {
      POCL_MEM_FREE (meta->arg_info[j].name);
      POCL_MEM_FREE (meta->arg_info[j].type_name);
    }
  POCL_MEM_FREE (meta->max_subgroups);
  POCL_MEM_FREE (meta->compile_subgroups);
  POCL_MEM_FREE (meta->max_workgroup_size);
  POCL_MEM_FREE (meta->preferred_wg_multiple);
  POCL_MEM_FREE (meta->local_mem_size);
  POCL_MEM_FREE (meta->private_mem_size);
  POCL_MEM_FREE (meta->spill_mem_size);
  POCL_MEM_FREE (meta->arg_info);
  if (meta->data != NULL)
    for (j = 0; j < program->num_devices; ++j)
      if (meta->data[j] != NULL)
        {
          POCL_MSG_WARN ("kernel metadata not freed\n");
          meta->data[j] = NULL; // TODO free data in driver callback
        }
  POCL_MEM_FREE (meta->data);
  if (program->builtin_kernel_names == NULL)
    POCL_MEM_FREE (meta->local_sizes);
  POCL_MEM_FREE (meta->build_hash);
}

int
pocl_svm_check_get_pointer (cl_context context, const void *svm_ptr, size_t size,
                            size_t *buffer_size, void** actual_ptr)
{

  /* TODO we need a better data structure than linked list,
   * right now it does a linear scan of all SVM allocations. */
  POCL_LOCK_OBJ (context);
  pocl_raw_ptr *found = NULL, *item = NULL;
  char *svm_alloc_end = NULL;
  char *svm_alloc_start = NULL;
  DL_FOREACH (context->raw_ptrs, item)
  {
    svm_alloc_start = (char *)item->vm_ptr;
    svm_alloc_end = svm_alloc_start + item->size;
    if (((char *)svm_ptr >= svm_alloc_start)
        && ((char *)svm_ptr < svm_alloc_end))
      {
        found = item;
        break;
      }
  }
  POCL_UNLOCK_OBJ (context);

  /* if the device does not support system allocation,
   * then the pointer must be found in the context's SVM alloc list */
  if (found == NULL) {
    if (context->svm_allocdev->svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)
    {
      return CL_SUCCESS;
    } else {
      POCL_MSG_ERR (
          "Can't find the pointer %p in list of allocated SVM pointers\n",
            svm_ptr);
      return CL_INVALID_OPERATION;
    }
  } else {
    assert (found != NULL);
    if (((char *)svm_ptr + size) > svm_alloc_end)
    {
      POCL_MSG_ERR ("The pointer+size exceeds the size of the allocation\n");
      return CL_INVALID_OPERATION;
    }

    if (buffer_size != NULL)
      *buffer_size = found->size;

    if (actual_ptr != NULL)
      *actual_ptr = found->vm_ptr;
    return CL_SUCCESS;
  }
}

int pocl_svm_check_pointer (cl_context context, const void *svm_ptr,
                            size_t size, size_t *buffer_size)
{
  return pocl_svm_check_get_pointer(context, svm_ptr, size, buffer_size, NULL);
}

typedef enum
{
  POCL_CB_TYPE_EVENT,
  POCL_CB_TYPE_CONTEXT,
  POCL_CB_TYPE_MEM
} pocl_cb_type_t;

typedef struct _pocl_async_callback_item pocl_async_callback_item;
struct _pocl_async_callback_item
{
  union
  {
    struct
    {
      int status;
      cl_event event;
      event_callback_item *cb;
    } event_cb;
    struct
    {
      context_destructor_callback_t *cb;
      cl_context ctx;
    } context_cb;
    struct
    {
      mem_destructor_callback_t *cb;
      cl_mem mem;
    } mem_cb;
  } data;
  pocl_cb_type_t type;
  pocl_async_callback_item *next;
};

static pocl_async_callback_item *async_callback_list = NULL;
static pocl_cond_t async_cb_wake_cond
  __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
static pocl_lock_t async_cb_lock
  __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
static int exit_pocl_async_callback_thread = CL_FALSE;
static pocl_thread_t async_callback_thread_id = 0;

static void
pocl_async_cb_push (pocl_async_callback_item *it)
{
  POCL_LOCK (async_cb_lock);
  LL_APPEND (async_callback_list, it);
  POCL_SIGNAL_COND (async_cb_wake_cond);
  POCL_UNLOCK (async_cb_lock);
}

void
pocl_event_cb_push (cl_event event, int status)
{
  pocl_async_callback_item *it = malloc (sizeof (pocl_async_callback_item));
  it->data.event_cb.event = event;
  it->data.event_cb.status = status;
  it->data.event_cb.cb = NULL;

  event_callback_item *tmp = NULL, *cb = NULL;
  LL_FOREACH_SAFE (event->callback_list, cb, tmp)
    {
      if ((cb->trigger_status == status)
          || (cb->trigger_status == CL_COMPLETE && status < CL_COMPLETE))
        {
          assert (event->callback_list);
          LL_DELETE (event->callback_list, cb);
          LL_APPEND (it->data.event_cb.cb, cb);
        }
    }

  it->type = POCL_CB_TYPE_EVENT;
  it->next = NULL;

  if (it->data.event_cb.cb)
    {
      POCL_RETAIN_OBJECT_UNLOCKED (event);
      pocl_async_cb_push (it);
    }
  else
    {
      free (it);
    }
}

void
pocl_context_cb_push (cl_context ctx)
{
  POCL_RETAIN_OBJECT_UNLOCKED (ctx);
  pocl_async_callback_item *it = malloc (sizeof (pocl_async_callback_item));
  it->data.context_cb.cb = ctx->destructor_callbacks;
  ctx->destructor_callbacks = NULL;
  it->data.context_cb.ctx = ctx;
  it->type = POCL_CB_TYPE_CONTEXT;
  it->next = NULL;
  pocl_async_cb_push (it);
}

void
pocl_mem_cb_push (cl_mem mem)
{
  POCL_RETAIN_OBJECT_UNLOCKED (mem);
  pocl_async_callback_item *it = malloc (sizeof (pocl_async_callback_item));
  it->data.mem_cb.cb = mem->destructor_callbacks;
  mem->destructor_callbacks = NULL;
  it->data.mem_cb.mem = mem;
  it->type = POCL_CB_TYPE_MEM;
  it->next = NULL;
  pocl_async_cb_push (it);
}

void
pocl_async_callback_finish ()
{
  POCL_LOCK (async_cb_lock);
  exit_pocl_async_callback_thread = CL_TRUE;
  POCL_SIGNAL_COND (async_cb_wake_cond);
  POCL_UNLOCK (async_cb_lock);
  if (async_callback_thread_id)
    POCL_JOIN_THREAD (async_callback_thread_id);
  POCL_DESTROY_COND (async_cb_wake_cond);
  POCL_DESTROY_LOCK (async_cb_lock);
}

static void
process_event_cb (pocl_async_callback_item *it)
{
  cl_event event = it->data.event_cb.event;
  event_callback_item *cb = it->data.event_cb.cb;
  event_callback_item *next_cb = NULL;

  while (cb)
    {
      next_cb = cb->next;
      assert ((cb->trigger_status == it->data.event_cb.status)
              || (cb->trigger_status == CL_COMPLETE
                  && it->data.event_cb.status < CL_COMPLETE));
      cb->callback_function (event, cb->trigger_status, cb->user_data);
      free (cb);
      cb = next_cb;
    }
  POname (clReleaseEvent) (event);
}

static void
process_mem_cb (pocl_async_callback_item *it)
{
  cl_mem mem = it->data.mem_cb.mem;
  mem_destructor_callback_t *cb = it->data.mem_cb.cb;
  mem_destructor_callback_t *next_cb = NULL;
  while (cb)
    {
      next_cb = cb->next;
      cb->pfn_notify (mem, cb->user_data);
      free (cb);
      cb = next_cb;
    }
  POname (clReleaseMemObject) (mem);
}

static void
process_context_cb (pocl_async_callback_item *it)
{
  cl_context ctx = it->data.context_cb.ctx;
  context_destructor_callback_t *cb = it->data.context_cb.cb;
  context_destructor_callback_t *next_cb = NULL;
  while (cb)
    {
      next_cb = cb->next;
      cb->pfn_notify (ctx, cb->user_data);
      free (cb);
      cb = next_cb;
    }
  POname (clReleaseContext) (ctx);
}

static void *
pocl_async_callback_thread (void *data)
{
  POCL_LOCK (async_cb_lock);
  while (exit_pocl_async_callback_thread == CL_FALSE)
    {
      /* Event callback handling calls functions in the same order
         they were added if the status matches the specified one. */
      pocl_async_callback_item *it = NULL;
      if (async_callback_list != NULL)
        {
          it = async_callback_list;
          LL_DELETE (async_callback_list, it);
        }
      else
        {
          POCL_WAIT_COND (async_cb_wake_cond, async_cb_lock);
        }

      if (it)
        {
          POCL_UNLOCK (async_cb_lock);
          switch (it->type)
            {
            case POCL_CB_TYPE_EVENT:
              process_event_cb (it);
              break;
            case POCL_CB_TYPE_MEM:
              process_mem_cb (it);
              break;
            case POCL_CB_TYPE_CONTEXT:
              process_context_cb (it);
              break;
            }
          free (it);
          POCL_LOCK (async_cb_lock);
        }
    }

  POCL_UNLOCK (async_cb_lock);
  return NULL;
}

void
pocl_async_callback_init ()
{
  POCL_INIT_LOCK (async_cb_lock);
  POCL_INIT_COND (async_cb_wake_cond);
  exit_pocl_async_callback_thread = CL_FALSE;
  async_callback_thread_id = 0;
  async_callback_list = NULL;
  POCL_CREATE_THREAD (async_callback_thread_id, pocl_async_callback_thread,
                      NULL);
}
