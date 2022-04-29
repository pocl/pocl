/* OpenCL runtime library: clSetContextDestructorCallback()

   Copyright (c) 2022 Michal Babej / Tampere University

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

#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL
POname (clSetContextDestructorCallback) (
    cl_context context, void (CL_CALLBACK *pfn_notify) (cl_context, void *),
    void *user_data) CL_API_SUFFIX__VERSION_3_0
{
  context_destructor_callback_t *callback;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);
  POCL_RETURN_ERROR_COND ((pfn_notify == NULL), CL_INVALID_VALUE);

  callback = (context_destructor_callback_t *)malloc (
      sizeof (context_destructor_callback_t));
  if (callback == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  callback->pfn_notify = pfn_notify;
  callback->user_data = user_data;
  callback->next = context->destructor_callbacks;
  context->destructor_callbacks = callback;

  return CL_SUCCESS;
}
POsym (clSetContextDestructorCallback)
