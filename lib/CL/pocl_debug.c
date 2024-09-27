/* OpenCL runtime library: PoCL debug functions

   Copyright (c) 2015-2023 PoCL developers
                 2024 Pekka Jääskeläinen / Intel Finland Oy

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
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pocl_debug.h"
#include "pocl_threads.h"
#include "pocl_timing.h"
#include "pocl_cl.h"

#include "utlist.h"

#ifdef POCL_DEBUG_MESSAGES

uint64_t pocl_debug_messages_filter; /* Bitfield */
int pocl_stderr_is_a_tty;

static pocl_lock_t console_mutex;

static pocl_lock_t pocl_tg_dump_lock = POCL_LOCK_INITIALIZER;
static pocl_cond_t pocl_tg_dump_cond;

void pocl_debug_output_lock(void) { POCL_LOCK(console_mutex); }

void pocl_debug_output_unlock(void) { POCL_UNLOCK(console_mutex); }

void pocl_debug_messages_setup(const char *debug) {
  POCL_INIT_LOCK(console_mutex);
  pocl_debug_messages_filter = 0;
  if (strlen(debug) == 1) {
    if (debug[0] == '1')
      pocl_debug_messages_filter = POCL_DEBUG_FLAG_GENERAL |
                                   POCL_DEBUG_FLAG_WARNING |
                                   POCL_DEBUG_FLAG_ERROR;
    return;
  }
  /* else parse */
  char *tokenize = strdup(debug);
  for (size_t i = 0; i < strlen (tokenize); i++)
  {
    tokenize[i] = tolower(tokenize[i]);
  }
  char *ptr = NULL;
  ptr = strtok(tokenize, ",");

  while (ptr != NULL) {
    if (strncmp(ptr, "general", 7) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_GENERAL;
    else if (strncmp(ptr, "level0", 6) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_LEVEL0;
    else if (strncmp(ptr, "vulkan", 6) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_VULKAN;
    else if (strncmp(ptr, "remote", 6) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_REMOTE;
    else if (strncmp(ptr, "event", 5) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_EVENTS;
    else if (strncmp(ptr, "cache", 5) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_CACHE;
    else if (strncmp(ptr, "proxy", 5) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_PROXY;
    else if (strncmp(ptr, "llvm", 4) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_LLVM;
    else if (strncmp(ptr, "refc", 4) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_REFCOUNTS;
    else if (strncmp(ptr, "lock", 4) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_LOCKING;
    else if (strncmp(ptr, "cuda", 4) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_CUDA;
    else if (strncmp(ptr, "almaif", 6) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_ALMAIF;
    else if (strncmp(ptr, "mmap", 4) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_ALMAIF_MMAP;
    else if (strncmp(ptr, "warn", 4) == 0)
      pocl_debug_messages_filter |=
          (POCL_DEBUG_FLAG_WARNING | POCL_DEBUG_FLAG_ERROR);
    else if (strncmp(ptr, "hsa", 3) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_HSA;
    else if (strncmp(ptr, "tce", 3) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_TCE;
    else if (strncmp(ptr, "mem", 3) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_MEMORY;
    else if (strncmp(ptr, "tim", 3) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_TIMING;
    else if (strncmp(ptr, "all", 3) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_ALL;
    else if (strncmp(ptr, "err", 3) == 0)
      pocl_debug_messages_filter |= POCL_DEBUG_FLAG_ERROR;
    else
      POCL_MSG_WARN("Unknown token in POCL_DEBUG env var: %s", ptr);

    ptr = strtok(NULL, ",");
  }

  free(tokenize);
}

void pocl_debug_print_header(const char *func, unsigned line,
                             const char *filter, int filter_type) {

  int year, mon, day, hour, min, sec, nanosec;
  pocl_gettimereal(&year, &mon, &day, &hour, &min, &sec, &nanosec);

  const char *filter_type_str;
  const char *formatstring;

  if (filter_type == POCL_FILTER_TYPE_ERR)
    filter_type_str =
        (pocl_stderr_is_a_tty ? POCL_COLOR_RED : " *** ERROR *** ");
  else if (filter_type == POCL_FILTER_TYPE_WARN)
    filter_type_str =
        (pocl_stderr_is_a_tty ? POCL_COLOR_YELLOW : " *** WARNING *** ");
  else if (filter_type == POCL_FILTER_TYPE_INFO)
    filter_type_str =
        (pocl_stderr_is_a_tty ? POCL_COLOR_GREEN : " *** INFO *** ");
  else
    filter_type_str =
        (pocl_stderr_is_a_tty ? POCL_COLOR_GREEN : " *** UNKNOWN *** ");

#ifndef POCL_DEBUG_LOG_PREFIX
#define POCL_DEBUG_LOG_PREFIX "PoCL"
#endif

  if (pocl_stderr_is_a_tty)
    formatstring = POCL_COLOR_BLUE
        "[%04i-%02i-%02i %02i:%02i:%02i.%09li] " POCL_COLOR_RESET
        "" POCL_DEBUG_LOG_PREFIX ": in fn %s " POCL_COLOR_RESET
        "at line %u:\n%s | %9s | ";
  else
    /* Print the log entries to a single line to enable merging of
       PoCL-R and PoCL-D logs with 'sort client.log server.log'. */
    formatstring
        = "[%04i-%02i-%02i %02i:%02i:%02i.%09i] " POCL_DEBUG_LOG_PREFIX
          ": in fn %s at line %u: %s | %9s | ";

  log_printf(formatstring, year, mon, day, hour, min, sec, nanosec, func, line,
             filter_type_str, filter);
}

void pocl_debug_measure_start(uint64_t *start) {
  *start = pocl_gettimemono_ns();
}

#define PRINT_DURATION(func, line, ...)                                        \
  do {                                                                         \
    pocl_debug_output_lock();                                                  \
    pocl_debug_print_header(func, line, "TIMING", POCL_FILTER_TYPE_INFO);      \
    log_printf(__VA_ARGS__);                                                   \
    pocl_debug_output_unlock();                                                \
  } while (0)

void pocl_debug_print_duration(const char *func, unsigned line, const char *msg,
                               uint64_t nanosecs) {
  if (!(pocl_debug_messages_filter & POCL_DEBUG_FLAG_TIMING))
    return;
  const char *formatstring;
  if (pocl_stderr_is_a_tty)
    formatstring = "      >>>  " POCL_COLOR_MAGENTA "     %3" PRIu64
                   ".%03" PRIu64 " " POCL_COLOR_RESET " %s    %s\n";
  else
    formatstring = "      >>>       %3" PRIu64 ".%03" PRIu64 "  %s    %s\n";

  uint64_t nsec = nanosecs % 1000000000;
  uint64_t sec = nanosecs / 1000000000;
  uint64_t a, b;

  if ((sec == 0) && (nsec < 1000)) {
    b = nsec % 1000;
    if (pocl_stderr_is_a_tty)
      formatstring = "      >>>      " POCL_COLOR_MAGENTA "     %3" PRIu64
                     " " POCL_COLOR_RESET " ns    %s\n";
    else
      formatstring = "      >>>           %3" PRIu64 "  ns    %s\n";
    PRINT_DURATION(func, line, formatstring, b, msg);
  } else if ((sec == 0) && (nsec < 1000000)) {
    a = nsec / 1000;
    b = nsec % 1000;
    PRINT_DURATION(func, line, formatstring, a, b, "us", msg);
  } else if (sec == 0) {
    a = nsec / 1000000;
    b = (nsec % 1000000) / 1000;
    PRINT_DURATION(func, line, formatstring, a, b, "ms", msg);
  } else {
    if (pocl_stderr_is_a_tty)
      formatstring = "      >>>  " POCL_COLOR_MAGENTA "     %3" PRIu64
                     ".%09" PRIu64 " " POCL_COLOR_RESET " %s    %s\n";
    else
      formatstring = "      >>>       %3" PRIu64 ".%09" PRIu64 "  %s    %s\n";

    PRINT_DURATION(func, line, formatstring, sec, nsec, "s", msg);
  }
}

void pocl_debug_measure_finish(uint64_t *start, uint64_t *finish,
                               const char *msg, const char *func,
                               unsigned line) {
  *finish = pocl_gettimemono_ns();
  pocl_debug_print_duration(func, line, msg, (*finish - *start));
}

/* Some of the device drivers can wait until clFinish() is called to allow
   a full task graph dumped to disk of the accumulated commands. The lock and
   the condition variable are used to synchronize the
   "clFinish() -> dump -> asynch execution in drivers" cycle with the condition
   variable holding the asynch execution until after we have dumped the
   graph. */

static void
dump_dot_command_queue (FILE *f,
                        struct _cl_command_queue *q,
                        size_t *sg_ids,
                        const char *extra_str)
{
  fprintf (f, "\tsubgraph cluster%zu {\n", (*sg_ids)++);
  fprintf (f, "\t\tlabel=\"CQ #%zu%s\";\n", q->id, extra_str);

  struct _cl_event *e;
  LL_FOREACH (q->events, e)
    {
      fprintf (f, "\t\tevent%zu [label=\"%zu: %s\\n", e->id, e->id,
               pocl_command_type_to_str (e->command_type, 1));
      if (e->command_type == CL_COMMAND_NDRANGE_KERNEL)
        {
          fprintf (f, "%s\\n", e->command->command.run.kernel->name);
        }

      pocl_buffer_migration_info *mi;
      LL_FOREACH (e->command->migr_infos, mi)
        {
          if (mi->buffer->parent != NULL)
            {
              fprintf (f, "sbuf#%zu/#%zu", mi->buffer->id,
                       mi->buffer->parent->id);
              if (e->command_type == CL_COMMAND_MIGRATE_MEM_OBJECTS)
                fprintf (f, "\\ns:%zu\\ne:%zu", mi->buffer->origin,
                         mi->buffer->origin + mi->buffer->size);
            }
          else
            fprintf (f, "buf#%zu", mi->buffer->id);
          if (mi->read_only)
            fprintf (f, " ro");

          if (mi->next != NULL)
            fprintf (f, "\\n");
        }
      fprintf (f, "\", color=");

      if (e->command_type == CL_COMMAND_NDRANGE_KERNEL)
        fprintf (f, "blue");
      else if (e->command_type == CL_COMMAND_MIGRATE_MEM_OBJECTS
               && e->command->command.migrate.implicit)
        fprintf (f, "red,style=\"dotted\"");
      else if (e->command_type == CL_COMMAND_READ_BUFFER)
        fprintf (f, "green");
      else if (e->command_type == CL_COMMAND_WRITE_BUFFER)
        fprintf (f, "red");
      else
        fprintf (f, "black");

      fprintf (f, "];\n");
    }
  fprintf (f, "\t}\n\n");
}

void
pocl_dump_dot_task_graph (cl_context context, const char *file_name)
{
  FILE *f = fopen (file_name, "w+");
  if (!f)
    {
      fprintf (stderr, "Unable to write to '%s'\n", file_name);
      fclose (f);
      return;
    }

  fprintf (f, "digraph {\n");
  struct _cl_context *ctx = (struct _cl_context *)context;

  size_t sg_ids = 0;
  /* Dump subgraphs (devices, command queues) and their nodes
   * (commands/events). */
  for (int dev = 0; dev < ctx->num_devices; ++dev)
    {
      cl_device_id device = ctx->devices[dev];

      fprintf (f, "subgraph cluster%zu {\n", sg_ids++);
      fprintf (f, "\tlabel=\"Device %d: %s\";\n", dev, device->short_name);
      struct _cl_command_queue *q;
      LL_FOREACH (ctx->command_queues, q)
        {
          if (q->device != device)
            continue;
          dump_dot_command_queue (f, q, &sg_ids, "");
        }
      LL_FOREACH (ctx->default_queues[dev], q)
        {
          assert (q->device == device);
          dump_dot_command_queue (f, q, &sg_ids, " (default)");
        }
      fprintf (f, "}\n");
    }

  /* Dump the event dependencies. */

  struct _cl_command_queue *q;
  LL_FOREACH (ctx->command_queues, q)
    {
      struct _cl_event *e;
      LL_FOREACH (q->events, e)
        {
          event_node *evn;
          LL_FOREACH (e->wait_list, evn)
            {
              fprintf (f,
                       "\t\tevent%zu -> event%zu [labelfontsize=8.0, "
                       "headlabel=\"%zu\"",
                       evn->event->id, e->id, evn->event->id);

              if (evn->event->command_type == CL_COMMAND_MIGRATE_MEM_OBJECTS
                  && evn->event->command->command.migrate.implicit)
                fprintf (f, ", color=\"red\", style=\"dotted\"");
              fprintf (f, "];\n");
            }
        }
    }
  for (int dev = 0; dev < ctx->num_devices; ++dev)
    {
      LL_FOREACH (ctx->default_queues[dev], q)
        {
          struct _cl_event *e;
          LL_FOREACH (q->events, e)
            {
              event_node *evn;
              LL_FOREACH (e->wait_list, evn)
                {
                  fprintf (f, "\t\tevent%zu -> event%zu;\n", evn->event->id,
                           e->id);
                }
            }
        }
    }

  fprintf (f, "}\n");

  fclose (f);
}

void
pocl_dump_dot_task_graph_wait ()
{
  POCL_LOCK (pocl_tg_dump_lock);
  POCL_WAIT_COND (pocl_tg_dump_cond, pocl_tg_dump_lock);
  POCL_UNLOCK (pocl_tg_dump_lock);
}

void
pocl_dump_dot_task_graph_signal ()
{
  /* Snapshot dumped. Now let the drivers supporting task graph dumping
   execute finish executing their queues asynchronously. The condition
   wait is per clFinish(). The drivers should take care of waiting for it
   in the correct spot. */
  POCL_LOCK (pocl_tg_dump_lock);
  POCL_BROADCAST_COND (pocl_tg_dump_cond);
  POCL_UNLOCK (pocl_tg_dump_lock);
}

const char *
pocl_command_type_to_str (cl_command_type cmd, int shortened)
{
  switch (cmd)
    {
    case CL_COMMAND_NDRANGE_KERNEL:
      return shortened ? "nd" : "ndrange_kernel";
    case CL_COMMAND_TASK:
      return "task_kernel";
    case CL_COMMAND_NATIVE_KERNEL:
      return "native_kernel";
    case CL_COMMAND_READ_BUFFER:
      return shortened ? "read" : "read_buffer";
    case CL_COMMAND_WRITE_BUFFER:
      return shortened ? "write" : "write_buffer";
    case CL_COMMAND_COPY_BUFFER:
      return shortened ? "copy" : "copy_buffer";
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
      return shortened ? "map" : "map_buffer";
    case CL_COMMAND_MAP_IMAGE:
      return "map_image";
    case CL_COMMAND_UNMAP_MEM_OBJECT:
      return shortened ? "unmap" : "unmap_mem_object";
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
      return shortened ? "migrate" : "migrate_mem_objects";
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

#endif
