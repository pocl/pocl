#include "pocl_cl.h"
#include "utlist.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clSetEventCallback) (cl_event     event ,
                    cl_int       command_exec_callback_type ,
                    void (CL_CALLBACK *  pfn_notify )(cl_event, cl_int, void *),
                    void *       user_data ) CL_API_SUFFIX__VERSION_1_1
{
  event_callback_item *cb_ptr = NULL;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (event)), CL_INVALID_EVENT);

  POCL_RETURN_ERROR_COND((pfn_notify == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON((command_exec_callback_type != CL_SUBMITTED &&
       command_exec_callback_type != CL_RUNNING && 
       command_exec_callback_type != CL_COMPLETE), CL_INVALID_VALUE,
       "callback type must be CL_SUBMITTED, CL_RUNNING or CL_COMPLETE");

  cb_ptr = (event_callback_item*) malloc (sizeof (event_callback_item));
  if (cb_ptr == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  cb_ptr->callback_function = pfn_notify;
  cb_ptr->user_data = user_data;
  cb_ptr->trigger_status = command_exec_callback_type;
  cb_ptr->next = NULL;

  POCL_LOCK_OBJ (event);
  if (event->status > command_exec_callback_type)
    {
      LL_APPEND (event->callback_list, cb_ptr);
      POCL_UNLOCK_OBJ (event);
    }
  else
    {
      /* The event is already at or past the requested status, so fire the
         callback synchronously. We have just dropped the event lock but still
         pass the event to callback_function while holding no reference of our
         own. The OpenCL spec requires that "all callbacks registered for an
         event object must be called before the event object is destroyed", so
         the implementation must keep the event alive until the callback has
         run -- it cannot let a concurrent release free it mid-callback. Such a
         release is possible: the callback itself may release the event (calling
         a non-blocking API such as clReleaseEvent from a callback is allowed),
         or another thread holding a reference (e.g. the command's completion on
         a device worker) may release it at the same time. Retain across the
         call and release after it returns, so the event lives for the full
         duration of callback_function. This mirrors the asynchronous path,
         where pocl_event_cb_push retains before queuing the callback and
         process_event_cb releases after it runs. */
      POCL_RETAIN_OBJECT_UNLOCKED (event);
      POCL_UNLOCK_OBJ (event);
      cb_ptr->callback_function (event, cb_ptr->trigger_status,
                                 cb_ptr->user_data);
      free (cb_ptr);
      POname (clReleaseEvent) (event);
    }
  

  return CL_SUCCESS;
}

POsym(clSetEventCallback)
