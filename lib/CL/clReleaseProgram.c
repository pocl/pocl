/* OpenCL runtime library: clReleaseProgram()

   Copyright (c) 2011 Universidad Rey Juan Carlos, 
   Pekka Jääskeläinen / Tampere University of Technology
   
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

#include <string.h>
#include <unistd.h>

#include "pocl_cl.h"
#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseProgram)(cl_program program) CL_API_SUFFIX__VERSION_1_0
{
  int new_refcount;
  cl_kernel k;
  char *env = NULL;

  POCL_RELEASE_OBJECT (program, new_refcount);

  if (new_refcount == 0)
    {

      /* Mark all kernels as having no program.
         FIXME: this should not be needed if the kernels
         retain the parent program (and release when the kernel
         is released). */
      for (k=program->kernels; k!=NULL; k=k->next)
        {          
          k->program = NULL;
        }

      POCL_RELEASE_OBJECT (program->context, new_refcount);
      free (program->source);
      if (program->binaries != NULL)
        {
          if (program->binaries[0])
            free (program->binaries[0]);
          free (program->binaries);
        }
      free (program->binary_sizes);

      env = getenv ("POCL_LEAVE_TEMP_DIRS");
      if (!(env != NULL && strcmp (env, "1") == 0) &&
          getenv("POCL_TEMP_DIR") == NULL)
        {
          remove_directory (program->temp_dir);
        }
      free (program->temp_dir);
      free (program);
    }

  return CL_SUCCESS;
}
POsym(clReleaseProgram)
