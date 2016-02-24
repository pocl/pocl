/* Command queue management functions

   Copyright (c) 2015 Giuseppe Bilotta
   
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

/* We keep a global list of all 'live' command queues in order to be able
 * to force a clFinish on all of them before this is triggered by the destructors
 * at program end, which happen in unspecified order and might cause all sorts
 * of issues. This header defines the signatures of the available functions
 */

#include <stdlib.h>
#include <string.h>
#include "pocl_debug.h"
#include "pocl_queue_util.h"
#include "common.h"

static pocl_lock_t queue_lock = POCL_LOCK_INITIALIZER;
static size_t queue_size = 0;
static size_t queue_alloc = 0;
static cl_command_queue *queue_list = NULL;

#define QUEUE_ALLOC_SIZE 256

int pocl_aborting;

void pocl_finish_all_queues()
{
  size_t i;
  if (pocl_aborting)
    return;
  for (i = 0; i < queue_size; ++i) {
    if (queue_list[i])
      POname(clFinish)(queue_list[i]);
  }
  pocl_print_system_memory_stats();
}

void pocl_init_queue_list()
{
  POCL_INIT_LOCK(queue_lock);

  POCL_LOCK(queue_lock);
  // will probably never need a realloc, but still
  queue_alloc = QUEUE_ALLOC_SIZE;

  queue_list = calloc(queue_alloc, sizeof(cl_command_queue));

  if (!queue_list)
    POCL_ABORT("unable to allocate queue list!");

  //atexit(pocl_finish_all_queues);

  POCL_UNLOCK(queue_lock);

}

// walk the queue list, 
void pocl_compact_queue_list() {
  size_t i; // walking index
  size_t compact = 0; // number of non-NULL elements
  for (i = 0; i < queue_size; ++i) {
    if (queue_list[i])
      compact++;
    else {
      // look for the first next non-NULL
      while (i < queue_size && queue_list[i] == NULL)
        ++i;
      if (i == queue_size)
        break; // no more entries
      // move stuff over
      memmove(queue_list + compact + 1, queue_list + i,
        (queue_size - i + 1)*sizeof(*queue_list));
      queue_size -= i - compact; // number of NULLs compacted
      i = compact + 1;
    }
  }
  queue_size = compact + 1;
}

void pocl_queue_list_insert(cl_command_queue q)
{
  POCL_LOCK(queue_lock);
  if (queue_size == queue_alloc) {
    // queue is full, try and compact it by removing the deleted queues
    pocl_compact_queue_list();
  }

  if (queue_size == queue_alloc) {
    // compaction failed to give us room
    cl_command_queue *resized = realloc(queue_list, queue_alloc + 256);
    if (!resized)
      POCL_ABORT("failed to enlarge queue list!");
    queue_list = resized;
    queue_alloc += 256;
  }

  queue_list[queue_size++] = q;
  POCL_UNLOCK(queue_lock);
}

void pocl_queue_list_delete(cl_command_queue q)
{
  POCL_LOCK(queue_lock);
  size_t i;
  for (i = 0; i < queue_size; ++i) {
    if (queue_list[i] == q) {
      queue_list[i] = NULL;
      goto unlock;
    }
  }
  // not found (?)
  POCL_MSG_WARN("command queue %p not found during deletion\n", q);

unlock:
  POCL_UNLOCK(queue_lock);
  return;
}

