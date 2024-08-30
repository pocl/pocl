/* pocl_tracing.c: event update and tracing system

   Copyright (c) 2015 Clément Léger / Kalray
   Copyright (c) 2016-2021 Tampere University

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

#define _DEFAULT_SOURCE

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <netdb.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#include "pocl_cq_profiling.h"
#include "pocl_util.h"
#include "pocl_tracing.h"
#include "pocl_timing.h"
#include "pocl_runtime_config.h"

#include "devices.h"

#ifdef HAVE_LTTNG_UST
#include "pocl_lttng.h"
static const struct pocl_event_tracer lttng_tracer;
#endif

static int tracing_initialized = 0;
static uint8_t event_trace_filter = 0xF;

static const struct pocl_event_tracer *event_tracer = NULL;

/* Called with event locked, and must also return with a locked event. */
void
pocl_event_updated (cl_event event, int status)
{
  if (event_tracer && event_tracer->event_updated
      && ((1 << status) & event_trace_filter))
    event_tracer->event_updated (event, status);

  if (event->callback_list)
    pocl_event_cb_push (event, status);
}

static void
pocl_parse_event_filter ()
{
  const char *trace_filter;
  char *tmp_str, *save_ptr, *token;

  trace_filter = pocl_get_string_option ("POCL_TRACING_FILTER", NULL);
  if (trace_filter == NULL)
    return;

  tmp_str = strdup (trace_filter);
  if (tmp_str == NULL)
    return;

  event_trace_filter = 0;
  while (1)
    {
      token = strtok_r (tmp_str, ",", &save_ptr);
      if (token == NULL)
        goto PARSE_OUT;
      if (strcmp (token, "queued") == 0)
        event_trace_filter |= (1 << CL_QUEUED);
      else if (strcmp (token, "submitted") == 0)
        event_trace_filter |= (1 << CL_SUBMITTED);
      else if (strcmp (token, "running") == 0)
        event_trace_filter |= (1 << CL_RUNNING);
      else if (strcmp (token, "complete") == 0)
        event_trace_filter |= (1 << CL_COMPLETE);

      tmp_str = NULL;
    }

PARSE_OUT:
  free (tmp_str);
}

//#################################################################
/* Basic text logger, useful for grep/cut/sed operations
 */
static FILE *text_tracer_file = NULL;
static pocl_lock_t text_tracer_lock;
static const char *text_tracer_output = NULL;

static void
text_tracer_init ()
{
  POCL_INIT_LOCK (text_tracer_lock);

  text_tracer_output = pocl_get_string_option ("POCL_TRACING_OPT",
                                          "pocl_trace_events.log");
  text_tracer_file = fopen (text_tracer_output, "w");
  if (!text_tracer_file)
    POCL_ABORT ("Failed to open text tracer output\n");
}

static void
text_tracer_destroy ()
{
  if (text_tracer_file)
    {
      int r = fclose (text_tracer_file);
      assert (r == 0);
    }
}

static void
text_tracer_event_updated (cl_event event, int status)
{
  if (!text_tracer_file)
    {
      POCL_MSG_ERR ("TEXT TRACER: log file doesn't exist\n");
      return;
    }

  // #######################################################
  // don't write until events are finished, since remote driver
  // sets up timestamps only at finish
  if (status > CL_COMPLETE)
    return;

  _cl_command_node *node = event->command;
  if (node == NULL)
    {
      POCL_MSG_ERR ("TEXT TRACER: node null\n");
      return;
    }

  cl_ulong ts;
  int statuses[] = { CL_QUEUED, CL_SUBMITTED, CL_RUNNING, CL_COMPLETE };
  cl_ulong times[] = { event->time_queue, event->time_submit,
                       event->time_start, event->time_end };

  cl_ulong ev_id = event->id;
  assert (ev_id && "No EV ID");
  cl_ulong cq_id = event->queue->id;
  assert (cq_id && "No CQ ID");
  cl_ulong dev_id = event->queue->device->id;
  assert (dev_id && "No DEV ID");
  char tmp_buffer[4096];
  char *cur_buf = tmp_buffer;
  int text_size = 0;

  for (unsigned i = 0; i < 4; ++i)
    {
      ts = times[i];
      status = statuses[i];

      const char *cmd_type = pocl_command_to_str (event->command_type);
      assert (cmd_type);
      const char *cmd_stat = pocl_status_to_str (status);
      assert (cmd_stat);

      text_size = sprintf (cur_buf,
                           "%" PRIu64 " | EV ID %" PRIu64 " | DEV %" PRIu64
                           " | CQ %" PRIu64 " | %s | %s | ",
                           ts, ev_id, dev_id, cq_id, cmd_type, cmd_stat);
      assert (text_size > 0);
      cur_buf += text_size;

      /* Print more informations for some commonly used commands */
      // TODO print only once
      switch (event->command_type)
        {
        case CL_COMMAND_NDRANGE_KERNEL:
          text_size = sprintf (cur_buf, "KERNEL ID %" PRIu64 " | name=%s\n",
                               node->command.run.kernel->id,
                               node->command.run.kernel->name);
          break;

        case CL_COMMAND_READ_BUFFER:
          text_size = sprintf (
            cur_buf, "MEM ID %" PRIu64 " | size=%" PRIuS " | host_ptr=%p\n",
            node->command.read.src->id, node->command.read.size,
            node->command.read.dst_host_ptr);
          break;
        case CL_COMMAND_WRITE_BUFFER:
          text_size = sprintf (
            cur_buf, "MEM ID %" PRIu64 " | size=%" PRIuS " | host_ptr=%p\n",
            node->command.write.dst->id, node->command.write.size,
            node->command.write.src_host_ptr);
          break;
        case CL_COMMAND_COPY_BUFFER:
          text_size
              = sprintf (cur_buf,
                         "MEM ID FROM %" PRIu64 " | MEM ID TO %" PRIu64
                         " | size=%" PRIuS "\n",
                         node->command.copy.src->id,
                         node->command.copy.dst->id, node->command.copy.size);
          break;
        case CL_COMMAND_FILL_BUFFER:
          text_size = sprintf (
            cur_buf, "MEM ID %" PRIu64 " | size=%" PRIuS "\n",
            node->command.memfill.dst->id, node->command.memfill.size);
          break;

        case CL_COMMAND_MIGRATE_MEM_OBJECTS:
          switch (node->command.migrate.type)
            {

            case ENQUEUE_MIGRATE_TYPE_H2D:
              text_size = sprintf (cur_buf,
                                   " # MEMS %" PRIuS " | MEM 0 ID %" PRIu64
                                   " | FROM DEV HOST | TO DEV %" PRIu64 " |\n",
                                   node->command.migrate.num_buffers,
                                   node->migr_infos->buffer->id, dev_id);
              break;

            case ENQUEUE_MIGRATE_TYPE_D2H:
              text_size = sprintf (cur_buf,
                                   " # MEMS %" PRIuS " | MEM 0 ID %" PRIu64
                                   " | FROM DEV %" PRIu64 " | TO DEV HOST |\n",
                                   node->command.migrate.num_buffers,
                                   node->migr_infos->buffer->id, dev_id);
              break;

            case ENQUEUE_MIGRATE_TYPE_D2D:
              text_size
                = sprintf (cur_buf,
                           " # MEMS %" PRIuS " | MEM 0 ID %" PRIu64
                           " | FROM DEV %" PRIu64 " | TO DEV %" PRIu64 " |\n",
                           node->command.migrate.num_buffers,
                           node->migr_infos->buffer->id,
                           node->command.migrate.src_device->id, dev_id);
              break;

            case ENQUEUE_MIGRATE_TYPE_NOP:
              text_size = sprintf (cur_buf,
                                   " # MEMS %" PRIuS " | MEM 0 ID %" PRIu64
                                   " | NOP MIGRATION DEV %" PRIu64 " |\n",
                                   node->command.migrate.num_buffers,
                                   node->migr_infos->buffer->id, dev_id);
              break;
            }
          break;

        case CL_COMMAND_MAP_BUFFER:
          text_size = sprintf (
            cur_buf, "MEM ID %" PRIu64 " | size=%" PRIuS "\n",
            node->command.map.buffer->id, node->command.map.mapping->size);
          break;

        case CL_COMMAND_UNMAP_MEM_OBJECT:
          text_size = sprintf (cur_buf, "MEM ID %" PRIu64 "\n",
                               node->command.unmap.buffer->id);
          break;

        case CL_COMMAND_READ_BUFFER_RECT:
          text_size = sprintf (
            cur_buf, "MEM ID %" PRIu64 " | size=%" PRIuS " | host_ptr=%p\n",
            node->command.read_rect.src->id, 0UL,
            node->command.read_rect.dst_host_ptr);
          break;

        case CL_COMMAND_WRITE_BUFFER_RECT:
          text_size = sprintf (
            cur_buf, "MEM ID %" PRIu64 " | size=%" PRIuS " | host_ptr=%p\n",
            node->command.write_rect.dst->id, 0UL,
            node->command.write_rect.src_host_ptr);
          break;

        case CL_COMMAND_COPY_BUFFER_RECT:
          text_size = sprintf (cur_buf,
                               "MEM ID FROM %" PRIu64 " | MEM ID TO %" PRIu64
                               " | size=%" PRIuS "\n",
                               node->command.copy_rect.src->id,
                               node->command.copy_rect.dst->id, 0UL);
          break;

        default:
          cur_buf[0] = '\n';
          text_size = 1;
        }
      assert (text_size > 0);
      cur_buf += text_size;
    }

  pocl_event_md *md = event->meta_data;
  if (md && md->num_deps > 0)
    {
      for (size_t i = 0; i < md->num_deps; ++i)
        {
          text_size = sprintf (
              cur_buf, "DEP | EV ID %" PRIu64 " -> EV ID %" PRIu64 "\n",
              md->dep_ids[i], ev_id);
          assert (text_size > 0);
          cur_buf += text_size;
        }
    }

  /* TODO: Make text_logger less intrusive by merging it with cq_profiler.
     Now it actually sprintfs after every event change which has a significant
     footprint. It could just use the collected events and sprintf the log
     after N events or atexit() to avoid the printing affecting the profile. */

  POCL_LOCK (text_tracer_lock);
  fwrite (tmp_buffer, (cur_buf - tmp_buffer), 1, text_tracer_file);
  POCL_UNLOCK (text_tracer_lock);
}

static const struct pocl_event_tracer text_logger = {
  "text",
  text_tracer_init,
  text_tracer_destroy,
  text_tracer_event_updated,
};

static const struct pocl_event_tracer cq_profiler
    = { "cq", pocl_cq_profiling_init,
        /* Avoid a callback after every event change to minimize impact to the
           profiling run. Instead just store the event timestamps with as
           little impact as possible for later collection/analysis. */
        NULL, NULL };

//#################################################################

#ifdef HAVE_LTTNG_UST
/* LTTNG tracer */

static void
lttng_tracer_init ()
{

}

static void
lttng_tracer_event_updated (cl_event event, int status)
{
  _cl_command_node *node = event->command;
  cl_command_queue cq = event->queue;
  cl_device_id dev = cq->device;

  if (node == NULL)
    return;

  switch (event->command_type)
    {
    case CL_COMMAND_NDRANGE_KERNEL:
      tracepoint (pocl_trace, ndrange_kernel,
                  event->id, status,
                  dev->dev_id, cq->id,
                  node->command.run.kernel->id,
                  node->command.run.kernel->name);
      break;

    case CL_COMMAND_READ_BUFFER:
      tracepoint (pocl_trace, read_buffer, event->id, status, dev->dev_id,
                  cq->id, node->command.read.src->id);
      break;

    case CL_COMMAND_WRITE_BUFFER:
      tracepoint (pocl_trace, write_buffer, event->id, status, dev->dev_id,
                  cq->id, node->command.write.dst->id);
      break;

    case CL_COMMAND_COPY_BUFFER:
      tracepoint (pocl_trace, copy_buffer,
                  event->id, status,
                  dev->dev_id, cq->id,
                  node->command.copy.src->id,
                  node->command.copy.dst->id);
      break;

    case CL_COMMAND_FILL_BUFFER:
      tracepoint (pocl_trace, fill_buffer, event->id, status, dev->dev_id,
                  cq->id, node->command.memfill.dst->id);
      break;

    case CL_COMMAND_READ_BUFFER_RECT:
      tracepoint (pocl_trace, read_buffer_rect, event->id, status, dev->dev_id,
                  cq->id, node->command.read_rect.src->id);
      break;

    case CL_COMMAND_WRITE_BUFFER_RECT:
      tracepoint (pocl_trace, write_buffer_rect, event->id, status,
                  dev->dev_id, cq->id, node->command.write_rect.dst->id);
      break;

    case CL_COMMAND_COPY_BUFFER_RECT:
      tracepoint (pocl_trace, copy_buffer_rect,
                  event->id, status,
                  dev->dev_id,
                  cq->id,
                  node->command.copy_rect.src->id,
                  node->command.copy_rect.dst->id);
      break;

    case CL_COMMAND_READ_IMAGE:
      tracepoint (pocl_trace, read_image_rect, event->id, status, dev->dev_id,
                  cq->id, node->command.read_image.dst->id);
      break;
    case CL_COMMAND_WRITE_IMAGE:
      tracepoint (pocl_trace, write_image_rect, event->id, status, dev->dev_id,
                  cq->id, node->command.write_image.dst->id);
      break;

    case CL_COMMAND_COPY_IMAGE:
      tracepoint (pocl_trace, copy_image_rect,
                  event->id, status,
                  dev->dev_id,
                  cq->id,
                  node->command.copy_image.src->id,
                  node->command.copy_image.dst->id);
      break;

    case CL_COMMAND_FILL_IMAGE:
      tracepoint (pocl_trace, fill_image, event->id, status, dev->dev_id,
                  cq->id, node->command.fill_image.dst->id);
      break;

    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
      tracepoint (pocl_trace, copy_image2buf,
                  event->id, status,
                  dev->dev_id,
                  cq->id,
                  node->command.read_image.src->id,
                  node->command.read_image.dst->id);
      break;

    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      tracepoint (pocl_trace, copy_buf2image,
                  event->id, status,
                  dev->dev_id,
                  cq->id,
                  node->command.write_image.src->id,
                  node->command.write_image.dst->id);
      break;

    case CL_COMMAND_MAP_BUFFER:
      tracepoint (pocl_trace, map_buffer, event->id, status, dev->dev_id,
                  cq->id, node->command.map.buffer->id);
      break;

    case CL_COMMAND_MAP_IMAGE:
      tracepoint (pocl_trace, map_image, event->id, status, dev->dev_id,
                  cq->id, node->command.map.buffer->id);
      break;

    case CL_COMMAND_UNMAP_MEM_OBJECT:
      tracepoint (pocl_trace, unmap_memobj, event->id, status, dev->dev_id,
                  cq->id, node->command.unmap.buffer->id);
      break;

    case CL_COMMAND_MIGRATE_MEM_OBJECTS:
      {
        assert (node->command.migrate.num_buffers > 0);
        switch (node->command.migrate.type)
          {

          case ENQUEUE_MIGRATE_TYPE_H2D:
            tracepoint (pocl_trace, migrate_mem_obj, event->id, status,
                        dev->dev_id, cq->id, node->command.migrate.num_buffers,
                        node->migr_infos->buffer->id, 0, "H2D");
            break;

          case ENQUEUE_MIGRATE_TYPE_D2H:
            tracepoint (pocl_trace, migrate_mem_obj, event->id, status,
                        dev->dev_id, cq->id, node->command.migrate.num_buffers,
                        node->migr_infos->buffer->id, 0, "D2H");
            break;

          case ENQUEUE_MIGRATE_TYPE_D2D:
            tracepoint (pocl_trace, migrate_mem_obj, event->id, status,
                        dev->dev_id, cq->id, node->command.migrate.num_buffers,
                        node->migr_infos->buffer->id,
                        node->command.migrate.src_device->id, "D2D");
            break;

          case ENQUEUE_MIGRATE_TYPE_NOP:
            tracepoint (pocl_trace, migrate_mem_obj, event->id, status,
                        dev->dev_id, cq->id, node->command.migrate.num_buffers,
                        node->migr_infos->buffer->id, 0, "NOP");
            break;
          }
        break;
      }
    }
}

static const struct pocl_event_tracer lttng_tracer = {
  "lttng",
  lttng_tracer_init,
  NULL,
  lttng_tracer_event_updated,
};

#endif

// #################################################################

/* List of tracers
 */
static const struct pocl_event_tracer *pocl_event_tracers[]
    = { &text_logger,
#ifdef HAVE_LTTNG_UST
        &lttng_tracer,
#endif
        &cq_profiler };

#define POCL_TRACER_COUNT                                                     \
  (sizeof (pocl_event_tracers) / sizeof ((pocl_event_tracers)[0]))

void
pocl_event_tracing_init ()
{
  const char *trace_env;
  unsigned i;

  if (tracing_initialized)
    return;

  trace_env = pocl_get_string_option ("POCL_TRACING", NULL);
  if (trace_env == NULL)
    goto EVENT_INIT_OUT;

  /* Check if a tracer has a name matching the supplied one */
  for (i = 0; i < POCL_TRACER_COUNT; i++)
    {
      if (strcmp (trace_env, pocl_event_tracers[i]->name) == 0)
        {
          event_tracer = pocl_event_tracers[i];
          break;
        }
    }
  if (event_tracer == NULL)
    goto EVENT_INIT_OUT;

  pocl_parse_event_filter ();

  event_tracer->init ();

EVENT_INIT_OUT:
  tracing_initialized = 1;
}

int pocl_is_tracing_enabled ()
{
  return event_tracer != NULL;
}

void
pocl_event_tracing_finish ()
{
  if (event_tracer && event_tracer->destroy)
    event_tracer->destroy ();
}
