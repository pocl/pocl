#include <stdio.h>
#include <string.h>
#include "pocl_event.h"
#include "pocl_runtime_config.h"

static int tracing_initialized = 0;

static const char *status_to_str[] = {
  "complete",
  "running",
  "submitted",
  "queued"
};

static const struct pocl_event_tracer text_logger;

/**
 * List of tracers
 */
static const struct pocl_event_tracer *pocl_event_tracers[] = {
  &text_logger,
};

#define POCL_TRACER_COUNT (sizeof(pocl_event_tracers) / sizeof((pocl_event_tracers)[0]))

static const struct pocl_event_tracer *event_tracer = NULL;

int pocl_event_updated(cl_event event, int status)
{
  event_callback_item* cb_ptr;

  /* event callback handling just call functions in the same order
   * they were added if the status match the specified one */
  for (cb_ptr = event->callback_list; cb_ptr; cb_ptr = cb_ptr->next)
    {
      if (cb_ptr->trigger_status == status)
        cb_ptr->callback_function (event, cb_ptr->trigger_status,
						  cb_ptr->user_data);
    }

  if (event_tracer)
    event_tracer->event_updated(event, status);
}

void pocl_event_tracing_init()
{
  const char *trace_env;
  int i;

  if (tracing_initialized)
    return;

  trace_env = pocl_get_string_option("POCL_TRACE_EVENT", NULL);
  if (trace_env == NULL)
    goto EVENT_INIT_OUT;

  /* Check if a tracer has a name matching the supplied one */
  for (i = 0; i < POCL_TRACER_COUNT; i++)
    {
	    if (strcmp(trace_env, pocl_event_tracers[i]->name) == 0)
	      {
					event_tracer = pocl_event_tracers[i];
					break;
				}
	  }
  if (event_tracer == NULL)
    goto EVENT_INIT_OUT;

  event_tracer->init();

EVENT_INIT_OUT:
  tracing_initialized = 1;
}

/**
 *  Convert a command type to it's equivalent string
 */
static const char *command_to_str(cl_command_type cmd)
{
  switch (cmd)
    {
	case CL_COMMAND_NDRANGE_KERNEL: return "ndrange_kernel";
	case CL_COMMAND_TASK: return "task_kernel";
	case CL_COMMAND_NATIVE_KERNEL: return "native_kernel";
	case CL_COMMAND_READ_BUFFER: return "read_buffer";
	case CL_COMMAND_WRITE_BUFFER: return "write_buffer";
	case CL_COMMAND_COPY_BUFFER: return "copy_buffer";
	case CL_COMMAND_READ_IMAGE: return "read_image";
	case CL_COMMAND_WRITE_IMAGE: return "write_image";
	case CL_COMMAND_COPY_IMAGE: return "copy_image";
	case CL_COMMAND_COPY_IMAGE_TO_BUFFER: return "copy_image_to_buffer";
	case CL_COMMAND_COPY_BUFFER_TO_IMAGE: return "copy_buffer_to_image";
	case CL_COMMAND_MAP_BUFFER: return "map_buffer";
	case CL_COMMAND_MAP_IMAGE: return "map_image";
	case CL_COMMAND_UNMAP_MEM_OBJECT: return "unmap_mem_object";
	case CL_COMMAND_MARKER: return "marker";
	case CL_COMMAND_ACQUIRE_GL_OBJECTS: return "acquire_gl_objects";
	case CL_COMMAND_RELEASE_GL_OBJECTS: return "release_gl_objects";
	case CL_COMMAND_READ_BUFFER_RECT: return "read_buffer_rect";
	case CL_COMMAND_WRITE_BUFFER_RECT: return "write_buffer_rect";
	case CL_COMMAND_COPY_BUFFER_RECT: return "copy_buffer_rect";
	case CL_COMMAND_USER: return "user";
	case CL_COMMAND_BARRIER: return "barrier";
	case CL_COMMAND_MIGRATE_MEM_OBJECTS: return "migrate_mem_objects";
	case CL_COMMAND_FILL_BUFFER: return "fill_buffer";
	case CL_COMMAND_FILL_IMAGE: return "fill_image";
	case CL_COMMAND_SVM_FREE: return "svm_free";
	case CL_COMMAND_SVM_MEMCPY: return "svm_memcpy";
	case CL_COMMAND_SVM_MEMFILL: return "svm_memfill";
	case CL_COMMAND_SVM_MAP: return "svm_map";
	case CL_COMMAND_SVM_UNMAP: return "svm_unmap";
    }

  return "unknown";
}

/**
 * Basic text logger
 */
static FILE *text_tracer_file = NULL;
static pocl_lock_t text_tracer_lock = POCL_LOCK_INITIALIZER;

static void text_tracer_init()
{
  const char *text_tracer_output;

  text_tracer_output = pocl_get_string_option("POCL_TRACE_EVENT_OPT", 
											"pocl_trace_events.log");
  text_tracer_file = fopen(text_tracer_output, "w");
  if (!text_tracer_file)
    POCL_ABORT("Failed to open text tracer output\n");
}

static void text_tracer_event_updated(cl_event event, int status)
{
  cl_command_queue cq = event->queue;
  cl_ulong ts = cq->device->ops->get_timer_value(cq->device->data);
  _cl_command_node *node = event->command;
  char tmp_buffer[512];
  char *cur_buf = tmp_buffer;
  int text_size = 0;
  
  /* Some less integrated commands (clEnqueueReadBufferRect) do not use
   * standard mecanism, so check node to be non null */
  if (node == NULL)
	  return;

  text_size = sprintf(cur_buf, "%lld %s %s ", ts,
												command_to_str(event->command_type),
												status_to_str[event->status]);
	cur_buf += text_size;
	/* Print more informations for some commonly used commands */
  switch (event->command_type)
    {
    case CL_COMMAND_READ_BUFFER:
      text_size += sprintf(cur_buf, "size=%d, host_ptr=%p\n",
														node->command.read.cb,
														node->command.read.host_ptr);
      break;
    case CL_COMMAND_WRITE_BUFFER:
      text_size += sprintf(cur_buf, "size=%d\n", node->command.write.cb);
      break;
    case CL_COMMAND_COPY_BUFFER:
      text_size += sprintf(cur_buf, "size=%d\n", node->command.copy.cb);
      break;
    case CL_COMMAND_NDRANGE_KERNEL:
      text_size += sprintf(cur_buf, "name=%s\n",
														node->command.run.kernel->name);
      break;
    case CL_COMMAND_FILL_BUFFER:
      text_size += sprintf(cur_buf, "size=%d\n",
														node->command.memfill.size);
      break;
    default:
      cur_buf[0] = '\n';
      text_size++;
    }

  POCL_LOCK(text_tracer_lock);
  fwrite(tmp_buffer, text_size, 1, text_tracer_file);
  POCL_UNLOCK(text_tracer_lock);
}

static const struct pocl_event_tracer text_logger = {
  "text",
  text_tracer_init,
  text_tracer_event_updated,
};
