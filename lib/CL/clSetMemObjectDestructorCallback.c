#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clSetMemObjectDestructorCallback)(  cl_mem  memobj , 
                                    void (CL_CALLBACK * pfn_notify)( cl_mem /* memobj */, void* /*user_data*/), 
                                    void * user_data  )             CL_API_SUFFIX__VERSION_1_1
{
  mem_destructor_callback_t *callback;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (memobj)),
                          CL_INVALID_MEM_OBJECT);
  POCL_RETURN_ERROR_COND((pfn_notify == NULL), CL_INVALID_VALUE);

  callback = (mem_destructor_callback_t *)malloc (
      sizeof (mem_destructor_callback_t));
  if (callback == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  callback->pfn_notify = pfn_notify;
  callback->user_data  = user_data;
  callback->next       = memobj->destructor_callbacks;
  memobj->destructor_callbacks = callback;

  return CL_SUCCESS;
}
POsym(clSetMemObjectDestructorCallback)
