#include "pocl_cl.h"
#include "utlist.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clSetEventCallback) (cl_event     event ,
                    cl_int       command_exec_callback_type ,
                    void (CL_CALLBACK *  pfn_notify )(cl_event, cl_int, void *),
                    void *       user_data ) CL_API_SUFFIX__VERSION_1_1
{
  event_callback_item *cb_ptr = NULL;

  if (event == NULL)
    return CL_INVALID_EVENT;

  if (pfn_notify == NULL || 
      (command_exec_callback_type != CL_SUBMITTED &&
       command_exec_callback_type != CL_RUNNING && 
       command_exec_callback_type != CL_COMPLETE))
    return CL_INVALID_VALUE;

  cb_ptr = malloc (sizeof (event_callback_item));
  if (cb_ptr == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  cb_ptr->callback_function = pfn_notify;
  cb_ptr->user_data = user_data;
  cb_ptr->trigger_status = command_exec_callback_type;
  cb_ptr->next = NULL;

  POCL_LOCK_OBJ (event);
  LL_APPEND (event->callback_list, cb_ptr);
  POCL_UNLOCK_OBJ (event);

  return CL_SUCCESS;
}

POsym(clSetEventCallback)
