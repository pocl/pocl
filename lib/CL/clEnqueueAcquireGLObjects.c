/* OpenCL runtime library: clEnqueueAcquireGLObjects()

   Copyright (c) 2021 Michal Babej / Tampere University

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

#include <assert.h>
#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL POname (clEnqueueAcquireGLObjects) (
    cl_command_queue command_queue, cl_uint num_mem_objects,
    const cl_mem *mem_objects, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event)
{
#ifdef ENABLE_OPENGL_INTEROP

  unsigned i, acquired = 0;
  int errcode;
  _cl_command_node *cmd = NULL;

  POCL_RETURN_ERROR_COND ((num_mem_objects == 0), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND ((mem_objects == NULL), CL_INVALID_VALUE);

  char *rdonly = (char *)alloca (num_mem_objects * sizeof (char));
  cl_mem *copy = (cl_mem *)alloca (num_mem_objects * sizeof (cl_mem));

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  for (i = 0; i < num_mem_objects; ++i)
    {
      POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (mem_objects[i])),
                            CL_INVALID_MEM_OBJECT);

      POCL_LOCK_OBJ (mem_objects[i]);

      POCL_GOTO_ERROR_COND (
          (mem_objects[i]->context != command_queue->context),
          CL_INVALID_CONTEXT);

      POCL_GOTO_ERROR_ON ((mem_objects[i]->is_gl_texture == 0),
                          CL_INVALID_MEM_OBJECT,
                          "mem_obj is NOT a GL texture\n");

      POCL_GOTO_ERROR_ON ((mem_objects[i]->is_gl_acquired != 0),
                          CL_INVALID_MEM_OBJECT,
                          "GL texture has ALREADY been acquired\n");

      assert (mem_objects[i]->parent == NULL);

      mem_objects[i]->is_gl_acquired += 1;
      ++acquired;

      // TODO
      rdonly[i] = (mem_objects[i]->flags & CL_MEM_READ_ONLY) ? 1 : 0;
      copy[i] = mem_objects[i];

      POCL_UNLOCK_OBJ (mem_objects[i]);
    }

  errcode = pocl_create_command (
      &cmd, command_queue, CL_COMMAND_ACQUIRE_GL_OBJECTS, event,
      num_events_in_wait_list, event_wait_list, num_mem_objects, copy, rdonly);

  if (errcode != CL_SUCCESS)
    goto ERROR;

  pocl_command_enqueue (command_queue, cmd);

  return CL_SUCCESS;

ERROR:
  for (i = 0; i < acquired; ++i)
    {
      POCL_LOCK_OBJ (mem_objects[i]);
      mem_objects[i]->is_gl_acquired -= 1;
      POCL_UNLOCK_OBJ (mem_objects[i]);
    }
  return errcode;

#else

  POCL_MSG_WARN (
      "CL-GL interop is only implemented by proxy device at this point\n");
  return CL_INVALID_CONTEXT;
#endif
}
POsym (clEnqueueAcquireGLObjects)
