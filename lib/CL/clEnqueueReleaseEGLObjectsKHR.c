/* OpenCL runtime library: clEnqueueReleaseEGLObjectsKHR()

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

#include "CL/cl_egl.h"
#include "pocl_cl.h"
#include "pocl_util.h"
#include "pocl_mem_management.h"

CL_API_ENTRY cl_int CL_API_CALL POname (clEnqueueReleaseEGLObjectsKHR) (
    cl_command_queue command_queue, cl_uint num_mem_objects,
    const cl_mem *mem_objects, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event)
{
#ifdef ENABLE_EGL_INTEROP

  unsigned i, released = 0;
  int errcode;
  _cl_command_node *cmd = NULL;

  POCL_RETURN_ERROR_COND ((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);
  POCL_RETURN_ERROR_COND ((num_mem_objects == 0), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND ((mem_objects == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);
  POCL_RETURN_ERROR_COND ((*(command_queue->device->available) == CL_FALSE),
                          CL_DEVICE_NOT_AVAILABLE);

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  pocl_buffer_migration_info *migr_infos = NULL;

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

      POCL_GOTO_ERROR_ON ((mem_objects[i]->is_gl_acquired == 0),
                          CL_INVALID_MEM_OBJECT,
                          "GL texture has NOT been acquired\n");

      assert (mem_objects[i]->parent == NULL);

      mem_objects[i]->is_gl_acquired -= 1;
      ++released;

      // TODO
      char rdonly = (mem_objects[i]->flags & CL_MEM_READ_ONLY) ? 1 : 0;
      cl_mem m = mem_objects[i];

      pocl_append_unique_migration_info (migr_infos, m, rdonly);

      POCL_UNLOCK_OBJ (mem_objects[i]);
    }

  errcode = pocl_create_command (
      &cmd, command_queue, CL_COMMAND_RELEASE_EGL_OBJECTS_KHR, event,
      num_events_in_wait_list, event_wait_list, migr_infos);

  if (errcode != CL_SUCCESS)
    goto ERROR;

  pocl_command_enqueue (command_queue, cmd);

  return CL_SUCCESS;

ERROR:
  for (i = 0; i < released; ++i)
    {
      POCL_LOCK_OBJ (mem_objects[i]);
      mem_objects[i]->is_gl_acquired += 1;
      POCL_UNLOCK_OBJ (mem_objects[i]);
    }
  return errcode;

#else

  POCL_MSG_WARN (
      "EGL interop is only implemented by proxy device at this point\n");
  return CL_INVALID_CONTEXT;
#endif
}

POsym (clEnqueueReleaseEGLObjectsKHR)
