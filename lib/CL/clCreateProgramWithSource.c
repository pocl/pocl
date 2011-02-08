/* OpenCL runtime library: clCreateProgramWithSource()

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
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

#include "locl_cl.h"
#include <string.h>

CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource(cl_context context,
                          cl_uint count,
                          const char **strings,
                          const size_t *lengths,
                          cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  cl_program program;
  unsigned size;
  char *source;
  unsigned i;

  if (count == 0)
    LOCL_ERROR(CL_INVALID_VALUE);

  program = (cl_program) malloc(sizeof(struct _cl_program));
  if (program == NULL)
    LOCL_ERROR(CL_OUT_OF_HOST_MEMORY);

  size = 0;
  for (i = 0; i < count; ++i)
    {
      if (strings[i] == NULL)
	LOCL_ERROR(CL_INVALID_VALUE);

      if (lengths == NULL)
	size += strlen(strings[i]);
      else if (lengths[i] == 0)
	size += strlen(strings[i]);
      else
	size += lengths[i];
    }
  
  source = (char *) malloc(size + 1);
  if (source == NULL)
    LOCL_ERROR(CL_OUT_OF_HOST_MEMORY);

  program->source = source;

  for (i = 0; i < count; ++i)
    {
      if (lengths == NULL)
	{
	  memcpy(source, strings[i], strlen(strings[i]));
	  source += strlen(strings[i]);
	}
      else if (lengths[i] == 0)
	{
	  memcpy(source, strings[i], strlen(strings[i]));
	  source += strlen(strings[i]);
	}
      else
	{
	  memcpy(source, strings[i], lengths[i]);
	  source += lengths[i];
	}
    }

  *source = '\0';

  program->reference_count = 1;
  program->context = context;
  program->num_devices = context->num_devices;
  program->devices = context->devices;
  
  return program;
}
