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

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

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

#include "common.h"
#include "devices.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_local_size.h"
#include "pocl_mem_management.h"
#include "pocl_runtime_config.h"
#include "pocl_shared.h"
#include "pocl_timing.h"
#include "pocl_util.h"
#include "utlist.h"

#ifdef ENABLE_RELOCATION
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#endif
#ifdef __linux__
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#endif

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
  ptr = memalign (align_width, size);
  return ptr;
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

void *
pocl_aligned_malloc (size_t alignment, size_t size)
{
#ifdef HAVE_ALIGNED_ALLOC
  assert (alignment > 0);
  /* make sure that size is a multiple of alignment, as posix_memalign
   * does not perform this test, whereas aligned_alloc does */
  if ((size & (alignment - 1)) != 0)
    {
      size = size | (alignment - 1);
      size += 1;
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

#else
#error Cannot find aligned malloc
#endif

#if 0
  /* this code works in theory, but there many places in pocl
   * where aligned memory is used in the same pointers
   * as memory allocated by other means */
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

#if 0
void
pocl_aligned_free (void *ptr)
{
#ifdef HAVE_ALIGNED_ALLOC
  POCL_MEM_FREE (ptr);
#else
#error Cannot find aligned malloc
  /* extract pointer from original allocation and free it */
  if (ptr)
    free(*(void **)((uintptr_t)ptr - sizeof(void *)));
#endif
}
#endif

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

/* This is required because e.g. NDRange commands could have the same buffer
 * multiple times as argument, or CopyBuffer could have src == dst buffer.
 *
 * If the buffer that appears multiple times in the list, is on another device,
 * we don't want to enqueue >1 migrations for the same buffer.
 */
static void
sort_and_uniq (cl_mem *objs, char *readonly_flags, size_t *num_objs)
{
  size_t i;
  ssize_t j;
  size_t n = *num_objs;
  assert (n > 1);

  /* if the buffer is an image backed by buffer storage,
   * replace with actual storage */
  for (i = 0; i < n; ++i)
    if (objs[i]->buffer)
      objs[i] = objs[i]->buffer;

  /* sort by obj id */
  for (i = 1; i < n; ++i)
    {
      cl_mem buf = objs[i];
      char c = readonly_flags[i];
      for (j = (i - 1); ((j >= 0) && (objs[j]->id > buf->id)); --j)
        {
          objs[j + 1] = objs[j];
          readonly_flags[j + 1] = readonly_flags[j];
        }
      objs[j + 1] = buf;
      readonly_flags[j + 1] = c;
    }

  /* skip the first i objects which are different */
  for (i = 1; i < n; ++i)
    if (objs[i - 1] == objs[i])
      break;

  /* uniq */
  size_t k = i;
  while (i < n)
    {
      if (objs[k] != objs[i])
        {
          objs[k] = objs[i];
          readonly_flags[k] = readonly_flags[i];
          ++k;
        }
      else
        {
          readonly_flags[k] = readonly_flags[k] & readonly_flags[i];
        }
      ++i;
    }

  *num_objs = k;
}

extern unsigned long event_c;
extern unsigned long uevent_c;

cl_int
pocl_create_event (cl_event *event, cl_command_queue command_queue,
                   cl_command_type command_type, size_t num_buffers,
                   const cl_mem *buffers, cl_context context)
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
  (*event)->num_buffers = num_buffers;
  if (num_buffers > 0)
    {
      (*event)->mem_objs = (cl_mem *)malloc (num_buffers * sizeof (cl_mem));
      memcpy ((*event)->mem_objs, buffers, num_buffers * sizeof (cl_mem));
    }
  (*event)->status = CL_QUEUED;

  if (command_type == CL_COMMAND_USER)
    POCL_ATOMIC_INC (uevent_c);
  else
    POCL_ATOMIC_INC (event_c);

  POCL_MSG_PRINT_EVENTS ("Created event %" PRIu64 " (%p) Command %s\n",
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

static int
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

/* preallocate the buffers on destination device.
 * if any allocation fails, we can't run this command. */
static int
can_run_command (cl_device_id dev, size_t num_objs, cl_mem *objs)
{
  size_t i;
  int errcode;

  for (i = 0; i < num_objs; ++i)
    {
      POCL_LOCK_OBJ (objs[i]);
      pocl_mem_identifier *p = &objs[i]->device_ptrs[dev->global_mem_id];
      // skip already allocated
      if (p->mem_ptr) {
        POCL_UNLOCK_OBJ (objs[i]);
        continue;
      }

      assert (dev->ops->alloc_mem_obj);
      errcode = dev->ops->alloc_mem_obj (dev, objs[i], NULL);
      if (errcode != CL_SUCCESS) {
        POCL_MSG_ERR("Failed to allocate %zu bytes on device %s\n",
                     objs[i]->size, dev->short_name);
      }

      POCL_UNLOCK_OBJ (objs[i]);
      if (errcode != CL_SUCCESS)
        return CL_FALSE;
    }

  return CL_TRUE;
}

static cl_int
pocl_create_command_struct (_cl_command_node **cmd,
                            cl_command_queue command_queue,
                            cl_command_type command_type, cl_event *event_p,
                            cl_uint num_events, const cl_event *wait_list,
                            size_t num_buffers, const cl_mem *buffers)
{
  unsigned i;
  cl_event *event = NULL;
  cl_int errcode = CL_SUCCESS;

  *cmd = pocl_mem_manager_new_command ();
  POCL_RETURN_ERROR_COND ((*cmd == NULL), CL_OUT_OF_HOST_MEMORY);

  (*cmd)->type = command_type;

  event = &((*cmd)->sync.event.event);
  errcode = pocl_create_event (event, command_queue, command_type, num_buffers,
                               buffers, command_queue->context);

  if (errcode != CL_SUCCESS)
    goto ERROR;
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

  (*cmd)->device = command_queue->device;
  (*cmd)->sync.event.event->command = (*cmd);

  /* Form event synchronizations based on the given wait list */
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
 * Creates the necessary implicit migration commands to ensure data is
 * where it's supposed to be according to the semantics of the program
 * defined using commands, buffers, command queues and events.
 *
 * @param dev Destination device
 * @param ev_export_p Optional output parameter for the export event
 * @param migration_size Max number of bytes to migrate (caller has to read
 *                       content size from mem->size_buffer if applicable)
 */
static int
pocl_create_migration_commands (cl_device_id dev, cl_event *ev_export_p,
                                cl_event final_event, cl_mem mem,
                                pocl_mem_identifier *p, const char readonly,
                                cl_command_type command_type,
                                cl_mem_migration_flags mig_flags,
                                uint64_t migration_size)
{
  int errcode = CL_SUCCESS;

  cl_event ev_export = NULL, ev_import = NULL, previous_last_event = NULL,
           last_migration_event = NULL;
  _cl_command_node *cmd_export = NULL, *cmd_import = NULL;
  cl_device_id ex_dev = NULL;
  cl_command_queue ex_cq = NULL, dev_cq = NULL;
  int can_directly_mig = 0;
  size_t i;

  /* "export" means copy buffer content from source device to mem_host_ptr;
   *
   * "import" means copy mem_host_ptr content to destination device,
   * or copy directly between devices
   *
   * "need_hostptr" if set, increase the mem_host_ptr_refcount,
   * to keep the mem_host_ptr backing memory around */
  int do_import = 0, do_export = 0, do_need_hostptr = 0;

  /*****************************************************************/

  /* this part only:
   *   sets up the buffer content versions according to requested migration type;
   *   sets the buffer->last_event pointer to the final_event;
   *   decides what needs to be actually done (import, export) but not do it;
   *
   * ... so that any following command sees a correct buffer state.
   * The actual migration commands are enqueued after. */
  POCL_LOCK_OBJ (mem);

  /* Retain the buffer for the duration of the command, except Unmaps,
   * because corresponding Maps retain twice. */
  if (command_type != CL_COMMAND_UNMAP_MEM_OBJECT)
    POCL_RETAIN_OBJECT_UNLOCKED (mem);

  /* save buffer's current last_event as previous last_event,
   * then set the last_event pointer to the actual command's event
   * (final_event).
   *
   * We'll need the "previous" event to properly chain commands, but
   * will release it after we've enqueued the required commands. */
  previous_last_event = mem->last_event;
  mem->last_event = final_event;

  /* Find the device/gmem with the latest copy of the data and that has the
   * fastest migration route.
   * ex_dev = device with the latest copy _other than dev_
   * dev_cq = default command queue for destination dev */
  int highest_d2d_mig_priority = 0;
  for (i = 0; i < mem->context->num_devices; ++i)
    {
      cl_device_id d = mem->context->devices[i];
      cl_command_queue cq = mem->context->default_queues[i];
      if (d == dev)
        dev_cq = cq;
      else if (mem->device_ptrs[d->global_mem_id].version == mem->latest_version)
        {
          int cur_d2d_mig_priority = 0;
          if (d->ops->can_migrate_d2d)
            cur_d2d_mig_priority = d->ops->can_migrate_d2d (dev, d);

          // if we can directly migrate, and we found a better device, use it
          if (cur_d2d_mig_priority > highest_d2d_mig_priority)
            {
              ex_dev = d;
              ex_cq = cq;
              highest_d2d_mig_priority = cur_d2d_mig_priority;
            }

          // if we can't migrate D2D, just use plain old through-host migration
          if (highest_d2d_mig_priority == 0)
            {
              ex_dev = d;
              ex_cq = cq;
            }
        }
    }

  assert (dev);
  assert (dev_cq);
  /* ex_dev can be NULL, or non-NULL != dev */
  assert (ex_dev != dev);

  /* if mem_host_ptr_version < latest_version, one of devices must have it;
   *
   * could be latest_version == mem_host_ptr_version == some p->version
   * for some p, and so i < ndev; in that case,
   * we leave ex_dev set since D2D is preferred migration way;
   *
   * otherwise must be
   * mem_host_ptr_version == latest_version & > all p->version */

  if ((mem->mem_host_ptr_version < mem->latest_version) && (p->version != mem->latest_version))
    assert ((ex_dev != NULL) && (mem->device_ptrs[ex_dev->global_mem_id].version == mem->latest_version));

  /* if ex_dev is NULL, either we have the latest or it's in mem_host_ptr */
  if (ex_dev == NULL)
    assert ((p->version == mem->latest_version) ||
            (mem->mem_host_ptr_version == mem->latest_version));

  /*****************************************************************/

  /* buffer must be already allocated on this device's globalmem */
  assert (p->mem_ptr != NULL);

  /* we're migrating to host mem only: clEnqueueMigMemObjs() with HOST flag */
  if (mig_flags & CL_MIGRATE_MEM_OBJECT_HOST)
    {
      do_import = 0;
      do_export = 0;
      do_need_hostptr = 1;
      if (mem->mem_host_ptr_version < mem->latest_version)
        {
          mem->mem_host_ptr_version = mem->latest_version;
          /* migrate content only if needed */
          if ((mig_flags & CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED) == 0
              || migration_size == 0)
            {
              /* Could be that destination dev has the latest version,
               * we still need to migrate to host mem */
              if (ex_dev == NULL)
                {
                  ex_dev = dev; ex_cq = dev_cq;
                }
              do_export = 1;
              POCL_RETAIN_OBJECT_UNLOCKED (mem);
            }
        }

      goto FINISH_VER_SETUP;
    }

  /* otherwise, we're migrating to a device memory. */
  /* check if we can migrate to the device associated with command_queue
   * without incurring the overhead of migrating their contents */
  if (mig_flags & CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED
      || migration_size == 0)
    p->version = mem->latest_version;

  can_directly_mig = highest_d2d_mig_priority > 0;

  /* Set the flag so the hostptr refcount gets incremented to keep the buffer
   * alive until it has been read for the associated content buffer's
   * migration. */
  if ((mem->content_buffer != NULL) && !can_directly_mig)
    do_need_hostptr = 1;

  /* if we don't need to migrate, skip to end */
  if (p->version >= mem->latest_version)
    {
      do_import = 0;
      do_export = 0;
      goto FINISH_VER_SETUP;
    }

  /* if mem_host_ptr is outdated AND the devices can't migrate
   * between each other, we need an export command */
  if ((mem->mem_host_ptr_version != mem->latest_version)
      && (can_directly_mig == 0))
    {
      /* we need two migration commands; one on the "source" device's hidden
       * queue, and one on the destination device. */
      do_import = 1;
      do_export = 1;
      do_need_hostptr = 1;

      /* because the two migrate commands will clRelease the buffer */
      POCL_RETAIN_OBJECT_UNLOCKED (mem);
      POCL_RETAIN_OBJECT_UNLOCKED (mem);
      mem->mem_host_ptr_version = mem->latest_version;
      p->version = mem->latest_version;
    }
  /* otherwise either:
   * 1) mem_host_ptr is latest, and we need to migrate mem-host-ptr to device, or
   * 2) mem_host_ptr is not latest, but devices can migrate directly between each other,
   * For both cases we only need one migration command on the destination device. */
  else
    {
      do_import = 1;
      do_export = 0;
      do_need_hostptr = 1;

      /* because the corresponding migrate command will clRelease the buffer */
      POCL_RETAIN_OBJECT_UNLOCKED (mem);
      p->version = mem->latest_version;
    }

FINISH_VER_SETUP:
  /* if the command is a write-use, increase the version. */
  if (!readonly)
    {
      ++p->version;
      mem->latest_version = p->version;
    }

  if (do_need_hostptr)
    {
      /* increase refcount for the two mig commands and for the caller
       * if this is a size buffer needed for content size -aware migration */
      if (do_export)
        ++mem->mem_host_ptr_refcount;
      if (do_import)
        ++mem->mem_host_ptr_refcount;
      if (ev_export_p)
        ++mem->mem_host_ptr_refcount;

      /* allocate mem_host_ptr here if needed... */
      if (mem->mem_host_ptr == NULL)
        {
          size_t align = max (mem->context->min_buffer_alignment, 16);
          /* Always allocate mem_host_ptr for the full size of the buffer to
           * guard against applications forgetting to check content size */
          mem->mem_host_ptr = pocl_aligned_malloc (align, mem->size);
          assert ((mem->mem_host_ptr != NULL)
                  && "Cannot allocate backing memory for mem_host_ptr!\n");
        }
    }

  /*****************************************************************/

  /* Enqueue a command for export.
   * Put the previous last event into its waitlist. */
  if (do_export)
    {
      assert (ex_cq);
      assert (ex_dev);
      errcode = pocl_create_command_struct (
          &cmd_export, ex_cq, CL_COMMAND_MIGRATE_MEM_OBJECTS,
          &ev_export, // event_p
          (previous_last_event ? 1 : 0),
          (previous_last_event ? &previous_last_event : NULL), // waitlist
          1, &mem                                              // buffer list
      );
      assert (errcode == CL_SUCCESS);
      if (do_need_hostptr)
        ev_export->release_mem_host_ptr_after = 1;

      cmd_export->command.migrate.mem_id
          = &mem->device_ptrs[ex_dev->global_mem_id];
      cmd_export->command.migrate.type = ENQUEUE_MIGRATE_TYPE_D2H;
      cmd_export->command.migrate.migration_size = migration_size;

      last_migration_event = ev_export;

      if (ev_export_p)
        {
          POname (clRetainEvent) (ev_export);
          *ev_export_p = ev_export;
        }
    }

  /* enqueue a command for import.
   * Put either the previous last event, or export ev, into its waitlist. */
  if (do_import)
    {
      /* the import command must depend on (wait for) either the export
       * command, or the buffer's previous last event. Can be NULL if there's
       * no last event or export command */
      cl_event import_wait_ev = (ev_export ? ev_export : previous_last_event);

      errcode = pocl_create_command_struct (
          &cmd_import, dev_cq, CL_COMMAND_MIGRATE_MEM_OBJECTS,
          &ev_import, // event_p
          (import_wait_ev ? 1 : 0),
          (import_wait_ev ? &import_wait_ev : NULL), // waitlist
          1, &mem                                    // buffer list
      );
      assert (errcode == CL_SUCCESS);
      if (do_need_hostptr)
        ev_import->release_mem_host_ptr_after = 1;

      if (can_directly_mig)
        {
          cmd_import->command.migrate.type = ENQUEUE_MIGRATE_TYPE_D2D;
          cmd_import->command.migrate.src_device = ex_dev;
          cmd_import->command.migrate.src_id
              = &mem->device_ptrs[ex_dev->global_mem_id];
          cmd_import->command.migrate.dst_id
              = &mem->device_ptrs[dev->global_mem_id];
          if (mem->size_buffer != NULL)
            cmd_import->command.migrate.src_content_size_mem_id
                = &mem->size_buffer->device_ptrs[ex_dev->global_mem_id];
        }
      else
        {
          cmd_import->command.migrate.type = ENQUEUE_MIGRATE_TYPE_H2D;
          cmd_import->command.migrate.mem_id
              = &mem->device_ptrs[dev->global_mem_id];
          cmd_import->command.migrate.migration_size = migration_size;
        }

      /* because explicit event */
      if (ev_export)
        POname (clReleaseEvent) (ev_export);

      last_migration_event = ev_import;
    }

  /* we don't need it anymore. */
  if (previous_last_event)
    POname (clReleaseEvent (previous_last_event));

  /* the final event must depend on the export/import commands */
  if (last_migration_event)
    {
      pocl_create_event_sync (final_event, last_migration_event);
      /* if the event itself only reads from the buffer,
       * set the last buffer event to last_mig_event,
       * instead of the actual command event;
       * this avoids unnecessary waits e.g on kernels
       * which only read from buffers */
      if (readonly)
        {
          mem->last_event = last_migration_event;
          POname (clReleaseEvent) (final_event);
        }
      else /* because explicit event */
        POname (clReleaseEvent) (last_migration_event);
    }
  POCL_UNLOCK_OBJ (mem);

  if (do_export) {
    POCL_MSG_PRINT_MEMORY (
      "Queuing a %zu-byte device-to-host migration for buf %zu%s\n",
      migration_size, mem->id, mem->parent != NULL ? " (sub-buffer)" : "");

    pocl_command_enqueue (ex_cq, cmd_export);
  }

  if (do_import) {
    POCL_MSG_PRINT_MEMORY (
      "Queuing a %zu-byte host-to-device migration for buf %zu%s\n",
      migration_size, mem->id, mem->parent != NULL ? " (sub-buffer)" : "");

    pocl_command_enqueue (dev_cq, cmd_import);
  }

  return CL_SUCCESS;
}

/**
 * Creates a command node and adds implicit data migrations.
 *
 * @param num_buffers Number of buffers the command depends on.
 * @param buffers The buffers the command depends on.
 */
static cl_int
pocl_create_command_full (_cl_command_node **cmd,
                          cl_command_queue command_queue,
                          cl_command_type command_type, cl_event *event_p,
                          cl_uint num_events, const cl_event *wait_list,
                          size_t num_buffers, cl_mem *buffers,
                          char *readonly_flags,
                          cl_mem_migration_flags mig_flags)
{
  cl_device_id dev = pocl_real_dev (command_queue->device);
  int err = CL_SUCCESS;
  size_t i;

  POCL_RETURN_ERROR_ON ((*dev->available == CL_FALSE), CL_INVALID_DEVICE,
                        "device is not available\n");

  if (num_buffers >= 1)
    {
      assert (buffers);
      assert (readonly_flags);

      if (num_buffers > 1)
        sort_and_uniq (buffers, readonly_flags, &num_buffers);

      if (can_run_command (dev, num_buffers, buffers) == CL_FALSE)
        return CL_OUT_OF_RESOURCES;
    }

  /* Waitlist here only contains the user-provided events.
   * migration events are added to waitlist later. */
  err = pocl_create_command_struct (cmd, command_queue, command_type, event_p,
                                    num_events, wait_list, num_buffers,
                                    buffers);
  if (err)
    return err;
  cl_event final_event = (*cmd)->sync.event.event;

  /* Retain once for every buffer. This is because we set every buffer's
   * "last event" to this, and then some next command enqueue
   * (or clReleaseMemObject) will release it.
   */
  POCL_LOCK_OBJ (final_event);
  final_event->pocl_refcount += num_buffers;
  POCL_UNLOCK_OBJ (final_event);

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

  /* Always migrate content size buffers first, if they exist */
  for (i = 0; i < num_buffers; ++i)
    {
      if (buffers[i]->size_buffer != NULL)
        {
          /* Bump "last event" refcount for content size buffers that weren't
           * explicitly given as dependencies */
          int explicit = 0;
          for (size_t j = 0; j < num_buffers; ++j)
            {
              if (buffers[j] == buffers[i]->size_buffer)
                {
                  already_migrated[j] = 1;
                  explicit = 1;
                  break;
                }
            }
          if (!explicit)
            {
              POname (clRetainEvent) (final_event);
            }

          pocl_create_migration_commands (
              dev, &size_events[i], final_event, buffers[i]->size_buffer,
              &(buffers[i]->size_buffer)->device_ptrs[dev->global_mem_id],
              readonly_flags[i], command_type, mig_flags,
              buffers[i]->size_buffer->size);
        }
    }

  for (i = 0; i < num_buffers; ++i)
    {
      uint64_t migration_size = buffers[i]->size;

      /* If both a content buffer and its associated size buffer are explicitly
       * listed in buffers, the size buffer was already migrated above. */
      if (already_migrated[i])
        continue;

      if (buffers[i]->size_buffer != NULL)
        {
          /* BLOCK until size buffer has been imported to host mem! No event
           * exists if host import is not needed. */
          if (size_events[i] != NULL)
            {
              cl_device_id d = size_events[i]->queue->device;
              POname (clWaitForEvents) (1, &size_events[i]);
              if (buffers[i]->size_buffer->mem_host_ptr != NULL)
                {
                  migration_size
                      = *(uint64_t *)buffers[i]->size_buffer->mem_host_ptr;
                }
              pocl_release_mem_host_ptr (buffers[i]->size_buffer);
              POname (clReleaseEvent) (size_events[i]);
              size_events[i] = NULL;
            }
        }

      pocl_create_migration_commands (
          dev, NULL, final_event, buffers[i],
          &buffers[i]->device_ptrs[dev->global_mem_id], readonly_flags[i],
          command_type, mig_flags, migration_size);
    }

  return err;
}

cl_int
pocl_create_command_migrate (_cl_command_node **cmd,
                             cl_command_queue command_queue,
                             cl_mem_migration_flags flags, cl_event *event_p,
                             cl_uint num_events, const cl_event *wait_list,
                             size_t num_buffers, cl_mem *buffers,
                             char *readonly_flags)
{
  return pocl_create_command_full (
      cmd, command_queue, CL_COMMAND_MIGRATE_MEM_OBJECTS, event_p, num_events,
      wait_list, num_buffers, buffers, readonly_flags, flags);
}

cl_int
pocl_create_command (_cl_command_node **cmd, cl_command_queue command_queue,
                     cl_command_type command_type, cl_event *event_p,
                     cl_uint num_events, const cl_event *wait_list,
                     size_t num_buffers, cl_mem *buffers, char *readonly_flags)
{
  return pocl_create_command_full (cmd, command_queue, command_type, event_p,
                                   num_events, wait_list, num_buffers, buffers,
                                   readonly_flags, 0);
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

cl_int
pocl_create_recorded_command (_cl_command_node **cmd,
                              cl_command_buffer_khr command_buffer,
                              cl_command_queue command_queue,
                              cl_command_type command_type, cl_uint num_deps,
                              const cl_sync_point_khr *sync_point_wait_list,
                              size_t num_buffers, cl_mem *buffers,
                              char *readonly_flags)
{
  cl_int errcode = pocl_check_syncpoint_wait_list (command_buffer, num_deps,
                                                   sync_point_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  POCL_RETURN_ERROR_COND (
    (can_run_command (command_queue->device, num_buffers, buffers) != CL_TRUE),
    CL_OUT_OF_RESOURCES);

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

  (*cmd)->sync.syncpoint.num_sync_points_in_wait_list = num_deps;
  if (num_deps > 0)
    {
      cl_sync_point_khr *wait_list
          = malloc (sizeof (cl_sync_point_khr) * num_deps);
      if (wait_list == NULL)
        {
          POCL_MEM_FREE (*cmd);
          return CL_OUT_OF_HOST_MEMORY;
        }
      memcpy (wait_list, sync_point_wait_list,
              sizeof (cl_sync_point_khr) * num_deps);
      (*cmd)->sync.syncpoint.sync_point_wait_list = wait_list;
    }

  (*cmd)->memobj_count = num_buffers;
  if (num_buffers > 0)
    {
      cl_mem *memobjs = (cl_mem *)malloc (sizeof (cl_mem) * num_buffers);
      (*cmd)->memobj_list = memobjs;
      POCL_GOTO_ERROR_COND ((memobjs == NULL), CL_OUT_OF_HOST_MEMORY);

      char *flags = (char *)malloc (sizeof (char) * num_buffers);
      (*cmd)->readonly_flag_list = flags;
      POCL_GOTO_ERROR_COND ((flags == NULL), CL_OUT_OF_HOST_MEMORY);

      memcpy (memobjs, buffers, sizeof (cl_mem) * num_buffers);
      memcpy (flags, readonly_flags, sizeof (char) * num_buffers);
      for (unsigned i = 0; i < num_buffers; ++i)
        {
          POname (clRetainMemObject) (memobjs[i]);
        }
    }

  return CL_SUCCESS;

ERROR:
  pocl_mem_manager_free_command (*cmd);
  return errcode;
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
pocl_unmap_command_finished (cl_event event, _cl_command_t *cmd)
{
  cl_device_id dev = event->queue->device;
  pocl_mem_identifier *mem_id = NULL;
  cl_mem mem = NULL;
  mem = event->mem_objs[0];
  mem_id = &mem->device_ptrs[dev->global_mem_id];

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

  char *temp = strdup (dev->builtin_kernel_list);
  char *token;
  char *rest = temp;

  while ((token = strtok_r (rest, ";", &rest)))
    {
      if (strcmp (token, kernel_name) == 0)
        {
          free (temp);
          return 1;
        }
    }

  free (temp);
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

      context->default_queues[i] = POname (clCreateCommandQueue) (
          context, dev,
          (CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_HIDDEN
           | CL_QUEUE_PROFILING_ENABLE),
          &err);
      assert (err == CL_SUCCESS);
      assert (context->default_queues[i]);
    }

  assert (alignment > 0);
  context->min_buffer_alignment = alignment;
  return CL_SUCCESS;
}

pocl_raw_ptr *
pocl_find_raw_ptr_with_vm_ptr (cl_context context, const void *host_ptr)
{
  POCL_LOCK_OBJ (context);
  pocl_raw_ptr *item = NULL;
  DL_FOREACH (context->raw_ptrs, item)
  {
    if (item->vm_ptr == NULL)
      continue;
    if (item->vm_ptr <= host_ptr && item->vm_ptr + item->size > host_ptr)
      {
        break;
      }
  }
  POCL_UNLOCK_OBJ (context);
  return item;
}

pocl_raw_ptr *
pocl_find_raw_ptr_with_dev_ptr (cl_context context, const void *dev_ptr)
{
  POCL_LOCK_OBJ (context);
  pocl_raw_ptr *item = NULL;
  DL_FOREACH (context->raw_ptrs, item)
  {
    if (item->dev_ptr == NULL)
      continue;
    if (item->dev_ptr <= dev_ptr && item->dev_ptr + item->size > dev_ptr)
      break;
  }
  POCL_UNLOCK_OBJ (context);
  return item;
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

int
pocl_run_command_capture_output (char *capture_string, size_t *captured_bytes,
                                 char *const *args)
{
  POCL_MSG_PRINT_INFO ("Launching: %s\n", args[0]);

  int in[2];
  int out[2];
  pipe (in);
  pipe (out);

#ifdef HAVE_VFORK
  pid_t p = vfork ();
#elif defined(HAVE_FORK)
  pid_t p = fork ();
#else
#error Must have fork() or vfork() system calls
#endif
  if (p == 0)
    {
      close (in[1]);
      close (out[0]);

      dup2 (in[0], STDIN_FILENO);
      dup2 (out[1], STDOUT_FILENO);
      dup2 (out[1], STDERR_FILENO);

      return execv (args[0], args);
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
      if (waitpid (p, &status, 0) < 0)
        POCL_ABORT ("pocl: waitpid() failed.\n");

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

// event locked
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

/* Note: this must be kept in sync with pocl_copy_event_node */
static void pocl_free_event_node (cl_event event)
{
  _cl_command_node *node = event->command;
  switch (node->type)
    {
    case CL_COMMAND_NDRANGE_KERNEL:
    case CL_COMMAND_TASK:
      pocl_ndrange_node_cleanup (node);
      break;

    case CL_COMMAND_FILL_BUFFER:
      POCL_MEM_FREE (node->command.memfill.pattern);
      break;

    case CL_COMMAND_SVM_MEMFILL:
      POCL_MEM_FREE (node->command.svm_fill.pattern);
      break;

    case CL_COMMAND_NATIVE_KERNEL:
      POCL_MEM_FREE (node->command.native.args);
      break;

    case CL_COMMAND_UNMAP_MEM_OBJECT:
      pocl_unmap_command_finished (event, &node->command);
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
  event->command = NULL;
}

/**
 * Copies a command node (mostly) for command buffering purposes.
 *
 * Doesn't touch the next/prev pointers, for instance.
 */
int
pocl_copy_event_node (_cl_command_node *dst_node, _cl_command_node *src_node)
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

#if 0
    /* these cases are currently not handled in pocl_copy_event_node,
     * because there is no command buffer equivalent of these nodes. */
    case CL_COMMAND_NATIVE_KERNEL:
    case CL_COMMAND_UNMAP_MEM_OBJECT:
    case CL_COMMAND_SVM_MIGRATE_MEM:
    case CL_COMMAND_SVM_FREE:
#endif

    default:
      break;
    }
  return CL_SUCCESS;
}

static void pocl_free_event_memobjs (cl_event event)
{
  size_t i;
  for (i = 0; i < event->num_buffers; ++i)
    {
      cl_mem mem = event->mem_objs[i];
      if (event->release_mem_host_ptr_after)
        {
          POCL_LOCK_OBJ (mem);
          pocl_release_mem_host_ptr (mem);
          POCL_UNLOCK_OBJ (mem);
        }
      POname (clReleaseMemObject) (mem);
    }
  POCL_MEM_FREE (event->mem_objs);
}

// status can be complete or failed (<0)
void
pocl_update_event_finished (cl_int status, const char *func, unsigned line,
                            cl_event event, const char *msg)
{
  assert (event != NULL);
  assert (event->queue != NULL);
  assert (event->status > CL_COMPLETE);
  int notify_cmdq = CL_FALSE;

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
  POCL_UNLOCK_OBJ (event);
  ops->broadcast (event);

  /* With remote being asynchronous it is possible that an event is signaled as
   * complete before some of its dependencies. Therefore this event has to be
   * removed from the notify lists of any remaining events in the wait list.
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

#ifdef POCL_DEBUG_MESSAGES
  if (msg != NULL)
    {
      pocl_debug_print_duration (
          func, line, msg, (uint64_t) (event->time_end - event->time_start));
    }
#endif

  pocl_free_event_node (event);
  pocl_free_event_memobjs (event);

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
pocl_update_event_failed (cl_event event)
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
#ifdef ENABLE_RELOCATION
    Dl_info info;
    if (dladdr((void*)pocl_get_private_datadir, &info))
    {
        char const *soname = info.dli_fname;
        strcpy(private_datadir, soname);
        char* last_slash = strrchr (private_datadir,'/');
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
  assert (strs || !num_strs && "strs is NULL while num_strs > 0!");
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
