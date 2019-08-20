/* pocl_tracing.c: event update and tracing system

   Copyright (c) 2015 Clément Léger / Kalray

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

#include <stdio.h>
#include <string.h>
#include "pocl_cq_profiling.h"
#include "pocl_util.h"
#include "pocl_tracing.h"
#include "pocl_runtime_config.h"

#ifdef HAVE_LTTNG_UST
#include "pocl_lttng.h"
static const struct pocl_event_tracer lttng_tracer;
#endif

static int tracing_initialized = 0;
static uint8_t event_trace_filter = 0xF;

static const struct pocl_event_tracer text_logger;
static const struct pocl_event_tracer cq_profiler;

/* List of tracers
 */
static const struct pocl_event_tracer *pocl_event_tracers[] = {
  &text_logger,
#ifdef HAVE_LTTNG_UST
  &lttng_tracer,
#endif
  &cq_profiler
};

#define POCL_TRACER_COUNT (sizeof(pocl_event_tracers) / sizeof((pocl_event_tracers)[0]))

static const struct pocl_event_tracer *event_tracer = NULL;

/* Called with event locked, and must also return with a locked event. */
void
pocl_event_updated (cl_event event, int status)
{
  event_callback_item *cb_ptr;

  /* Event callback handling calls functions in the same order
     they were added if the status matches the specified one. */
  for (cb_ptr = event->callback_list; cb_ptr; cb_ptr = cb_ptr->next)
    {
      if (cb_ptr->trigger_status == status)
        {
          POCL_UNLOCK_OBJ (event);
          cb_ptr->callback_function (event, cb_ptr->trigger_status,
                                     cb_ptr->user_data);
          POCL_LOCK_OBJ (event);
        }
    }

  if (event_tracer && event_tracer->event_updated &&
      ((1 << status) & event_trace_filter))
    event_tracer->event_updated (event, status);
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

void
pocl_tracing_init ()
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


/* Basic text logger, useful for grep/cut/sed operations
 */
static FILE *text_tracer_file = NULL;
static pocl_lock_t text_tracer_lock = POCL_LOCK_INITIALIZER;

static void
text_tracer_init ()
{
  const char *text_tracer_output;

  text_tracer_output = pocl_get_string_option ("POCL_TRACING_OPT",
                                          "pocl_trace_events.log");
  text_tracer_file = fopen (text_tracer_output, "w");
  if (!text_tracer_file)
    POCL_ABORT ("Failed to open text tracer output\n");
}

static void
text_tracer_event_updated (cl_event event, int status)
{
  cl_ulong ts;
  switch (status)
    {
    case CL_QUEUED:
      ts = event->time_queue;
      break;
    case CL_SUBMITTED:
      ts = event->time_submit;
      break;
    case CL_RUNNING:
      ts = event->time_start;
      break;
    case CL_COMPLETE:
    default:
      ts = event->time_end;
    }
  _cl_command_node *node = event->command;
  char tmp_buffer[512];
  char *cur_buf = tmp_buffer;
  int text_size = 0;

  /* Some less integrated commands (clEnqueueReadBufferRect) do not use
   * standard mecanism, so check node to be non null */
  if (node == NULL)
    return;

  text_size = sprintf (cur_buf, "%"PRIu64" %s %s ", ts,
                       pocl_command_to_str (event->command_type),
                       pocl_status_to_str(event->status));
  cur_buf += text_size;
  /* Print more informations for some commonly used commands */
  switch (event->command_type)
    {
    case CL_COMMAND_READ_BUFFER:
      text_size += sprintf (cur_buf, "size=%" PRIuS ", host_ptr=%p\n",
                            node->command.read.size,
                            node->command.read.dst_host_ptr);
      break;
    case CL_COMMAND_WRITE_BUFFER:
      text_size += sprintf (cur_buf, "size=%" PRIuS ", host_ptr=%p\n\n",
                            node->command.write.size,
                            node->command.write.src_host_ptr);
      break;
    case CL_COMMAND_COPY_BUFFER:
      text_size
          += sprintf (cur_buf, "size=%" PRIuS "\n", node->command.copy.size);
      break;
    case CL_COMMAND_NDRANGE_KERNEL:
      text_size += sprintf (cur_buf, "name=%s\n",
                            node->command.run.kernel->name);
      break;
    case CL_COMMAND_FILL_BUFFER:
      text_size += sprintf (cur_buf, "size=%"PRIuS"\n", 
                            node->command.memfill.size);
      break;
    default:
      cur_buf[0] = '\n';
      text_size++;
    }

  /* TODO: Make text_logger less intrusive by merging it with cq_profiler.
     Now it actually sprintfs after every event change which has a significant
     footprint. It could just use the collected events and sprintf the log
     after N events or atexit() to avoid the printing affecting the profile. */

  POCL_LOCK (text_tracer_lock);
  fwrite (tmp_buffer, text_size, 1, text_tracer_file);
  POCL_UNLOCK (text_tracer_lock);
}

static const struct pocl_event_tracer text_logger = {
  "text",
  text_tracer_init,
  text_tracer_event_updated,
};

static const struct pocl_event_tracer cq_profiler = {
  "cq",
  pocl_cq_profiling_init,
  /* Avoid a callback after every event change to minimize impact to the profiling run.
     Instead just store the event timestamps with as little impact as possible for
     later collection/analysis. */
  NULL
};

#ifdef HAVE_LTTNG_UST

/* LTTNG tracer
 */

static void
lttng_tracer_init ()
{

}

static void
lttng_tracer_event_updated (cl_event event, int status)
{
  _cl_command_node *node = event->command;

  if (node == NULL)
    return;

  switch (event->command_type)
    {
    case CL_COMMAND_NDRANGE_KERNEL:
      tracepoint (pocl_trace, ndrange_kernel, event->id, status,
                  node->command.run.kernel->name);
      break;
    case CL_COMMAND_READ_BUFFER:
      tracepoint (pocl_trace, read_buffer, event->id, status,
                  node->command.read.dst_host_ptr, node->command.read.size);
      break;
    case CL_COMMAND_WRITE_BUFFER:
      tracepoint (pocl_trace, write_buffer, event->id, status,
                  node->command.write.src_host_ptr, node->command.write.size);
      break;
    case CL_COMMAND_COPY_BUFFER:
      tracepoint (pocl_trace, copy_buffer, event->id, status, node->command.copy.size);
      break;
    case CL_COMMAND_FILL_BUFFER:
      tracepoint (pocl_trace, fill_buffer, event->id, status, node->command.copy.size);
      break;
    case CL_COMMAND_MAP_BUFFER:
    case CL_COMMAND_MAP_IMAGE:
      tracepoint (pocl_trace, map, event->id, status, node->command.map.mapping->size);
      break;
    default:
      tracepoint (pocl_trace, command, event->id, status,
                  pocl_command_to_str (event->command_type));
      break;
    }
}

static const struct pocl_event_tracer lttng_tracer = {
  "lttng",
  lttng_tracer_init,
  lttng_tracer_event_updated,
};


#endif
