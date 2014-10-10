#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clSetMemObjectDestructorCallback)(  cl_mem  memobj , 
                                    void (CL_CALLBACK * pfn_notify)( cl_mem /* memobj */, void* /*user_data*/), 
                                    void * user_data  )             CL_API_SUFFIX__VERSION_1_1
{
  POCL_ABORT_UNIMPLEMENTED("The entire clSetMemObjectDestructorCallback call");
  return CL_SUCCESS;
}
POsym(clSetMemObjectDestructorCallback)
