/* OpenCL runtime library: clBuildProgram()

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
#include <assert.h>
#include <string.h>
#include <sys/stat.h>

#define COMMAND_LENGTH 256

CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram(cl_program program,
               cl_uint num_devices,
               const cl_device_id *device_list,
               const char *options,
               void (CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
               void *user_data) CL_API_SUFFIX__VERSION_1_0
{
  const char source_file_name[L_tmpnam], binary_file_name[L_tmpnam];
  FILE *source_file, *binary_file;
  size_t n;
  struct stat buf;
  char command[COMMAND_LENGTH];
  int error;
  unsigned char *binary;

  if (program == NULL)
    return CL_INVALID_PROGRAM;

  if (program->source == NULL && program->binary == NULL)
    return CL_INVALID_PROGRAM;

  if (program->binary == NULL)
    {
      tmpnam(source_file_name);
      source_file = fopen(source_file_name, "w+");
      if (source_file == NULL)
	return CL_OUT_OF_HOST_MEMORY;

      n = fwrite(program->source, 1,
		 strlen(program->source), source_file);
      if (n < strlen(program->source))
	return CL_OUT_OF_HOST_MEMORY;

      fclose(source_file);
      
      tmpnam(binary_file_name);

      if (stat(BUILDDIR "/scripts/" LOCL_BUILD, &buf) == 0)
	error = snprintf(command, COMMAND_LENGTH,
			 BUILDDIR "/scripts/" LOCL_BUILD " -o %s %s",
			 binary_file_name, source_file_name);
      else
	error = snprintf(command, COMMAND_LENGTH, LOCL_BUILD " -o %s %s",
			 binary_file_name, source_file_name);
      if (error < 0)
	return CL_OUT_OF_HOST_MEMORY;

      error = system(command);
      if (error != 0)
	return CL_BUILD_PROGRAM_FAILURE;

      binary_file = fopen(binary_file_name, "r");
      if (binary_file == NULL)
	return CL_OUT_OF_HOST_MEMORY;

      fseek(binary_file, 0, SEEK_END);
      program->binary_size = ftell(binary_file);
      fseek(binary_file, 0, SEEK_SET);

      binary = (unsigned char *) malloc(program->binary_size);
      if (binary == NULL)
	return CL_OUT_OF_HOST_MEMORY;

      n = fread(binary, 1, program->binary_size, binary_file);
      if (n < program->binary_size)
	{
	  free(binary);
	  return CL_OUT_OF_HOST_MEMORY;
	}

      program->binary = binary;
    }

  //assert(program->num_devices == num_devices);

  return CL_SUCCESS;
}
