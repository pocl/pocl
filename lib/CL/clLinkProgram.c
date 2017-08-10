/* OpenCL runtime library: clLinkProgram()

   Copyright (c) 2017 Michal Babej / Tampere University of Technology

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

#include "pocl_cl.h"
#include "pocl_shared.h"
#include "pocl_util.h"

CL_API_ENTRY cl_program CL_API_CALL
POname (clLinkProgram) (cl_context context,
                        cl_uint num_devices,
                        const cl_device_id *device_list,
                        const char *options,
                        cl_uint num_input_programs,
                        const cl_program *input_programs,
                        void (CL_CALLBACK *pfn_notify) (cl_program program, void *user_data),
                        void *user_data,
                        cl_int *errcode_ret)
CL_API_SUFFIX__VERSION_1_2
{
  int errcode; unsigned i;
  cl_program program = NULL;
  cl_device_id *unique_devlist = NULL;

  POCL_GOTO_ERROR_COND ((context == NULL), CL_INVALID_CONTEXT);

  POCL_GOTO_ERROR_COND ((num_input_programs == 0), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND ((input_programs == NULL), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND ((num_devices > 0 && device_list == NULL),
                        CL_INVALID_VALUE);
  POCL_GOTO_ERROR_COND ((num_devices == 0 && device_list != NULL),
                        CL_INVALID_VALUE);

  for (i = 0; i < num_input_programs; i++)
    {
      cl_program p = input_programs[i];
      POCL_GOTO_ERROR_ON (
          ((p->binary_type != CL_PROGRAM_BINARY_TYPE_LIBRARY)
           && (p->binary_type != CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT)),
          CL_INVALID_OPERATION,
          "clLinkProgram called for !library && !compiled_obj\n");
    }

  if (num_devices == 0)
    {
      num_devices = context->num_devices;
      device_list = context->devices;
    }
  else
    {
      /* convert subdevices to devices and remove duplicates */
      cl_uint real_num_devices = 0;
      unique_devlist = pocl_unique_device_list (device_list,
                                                num_devices,
                                                &real_num_devices);
      num_devices = real_num_devices;
      device_list = unique_devlist;
    }

  program = create_program_skeleton (context, num_devices, device_list,
                                     NULL, NULL, NULL, &errcode, 1);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  assert (num_devices == program->num_devices);

  /* link the program */
  errcode = compile_and_link_program (0, 1, program,
                                      num_devices, device_list, options,
                                      0, NULL, NULL,
                                      num_input_programs, input_programs,
                                      pfn_notify, user_data);

ERROR:
  POCL_MEM_FREE (unique_devlist);

  if (errcode_ret)
    *errcode_ret = errcode;

  if (errcode == CL_SUCCESS)
    {
      return program;
    }
  else
    {
      POname (clReleaseProgram) (program);
      return NULL;
    }
}
POsym (clLinkProgram)
