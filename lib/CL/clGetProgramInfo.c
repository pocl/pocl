/* OpenCL runtime library: clGetProgramInfo()

   Copyright (c) 2011 Erik Schnetter
   
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
#include "pocl_llvm.h"
#include "pocl_util.h"
#include "pocl_cache.h"
#include "pocl_binary_format.h"


cl_int compileKernels(cl_program program, cl_device_id device, poclcc_device *devicecc)
{
  cl_int errcode = CL_SUCCESS;

  cl_command_queue command_queue = clCreateCommandQueue(program->context, 
                                                        device, 0, &errcode);
  POCL_RETURN_ERROR_COND(errcode != CL_SUCCESS, errcode);

  _cl_command_node *command_node;
  errcode = pocl_create_command (&command_node, command_queue, 
                               CL_COMMAND_NDRANGE_KERNEL,
                               NULL, 0, NULL);

  cl_kernel kernel[32];
  cl_uint num_kernels = 0;
  errcode = clCreateKernelsInProgram(program, 32, kernel, &num_kernels);
  POCL_RETURN_ERROR_COND(errcode != CL_SUCCESS, errcode);

  poclcc_kernel *kernel_tab;
  POCL_RETURN_ERROR_COND(
    (kernel_tab = malloc(sizeof(poclcc_kernel)*num_kernels)) == NULL,
    CL_OUT_OF_HOST_MEMORY);
    
  int i;
  for (i=0; i<num_kernels; i++){

    //COMPILATION
    char cachedir[POCL_FILENAME_LENGTH];
    pocl_cache_make_kernel_cachedir_path(cachedir, program,
                                         device, kernel[i],
                                         0,0,0);

    POCL_GOTO_ERROR_COND(
      (errcode = pocl_llvm_generate_workgroup_function(device,
                                                       kernel[i],
                                                       0,0,0)) != CL_SUCCESS,
      errcode);

    command_node->command.run.kernel = kernel[i];
    command_node->command.run.tmp_dir = strdup(cachedir);
    device->ops->compile_submitted_kernels(command_node);

    //READ OBJFILE
    char objfile[POCL_FILENAME_LENGTH];
    errcode = snprintf(objfile, POCL_FILENAME_LENGTH, 
                     "%s/%s.so", 
                     command_node->command.run.tmp_dir, 
                     kernel[i]->name);
    assert(errcode >= 0);
    errcode = CL_SUCCESS;

    void *lock =pocl_cache_acquire_reader_lock(program, device);

    FILE *f = fopen(objfile, "r");
    fseek(f, 0, SEEK_END);
    int fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
      
    unsigned char *binary;
    POCL_GOTO_ERROR_COND(
      (binary = malloc(fsize)) == NULL, CL_OUT_OF_HOST_MEMORY);

    fread(binary, fsize, 1, f);
    fclose(f);

    pocl_cache_release_lock(lock);

    //ALLOC KERNEL INFOS
    char *kernel_name;
    struct pocl_argument *dyn_arguments;
    struct pocl_argument_info *arg_info;
    
    int sizeofKernelName = strlen(kernel[i]->name);
    POCL_GOTO_ERROR_COND(
      (kernel_name = malloc(sizeofKernelName)) 
      == NULL,
      CL_OUT_OF_HOST_MEMORY);
    memcpy(kernel_name, kernel[i]->name, sizeofKernelName);

    int num_args = kernel[i]->num_args;
    int num_locals = kernel[i]->num_locals;

    POCL_GOTO_ERROR_COND(
      (dyn_arguments = malloc((num_args+num_locals)*sizeof(struct pocl_argument))) 
      == NULL,
      CL_OUT_OF_HOST_MEMORY);
    memcpy(dyn_arguments, kernel[i]->dyn_arguments, 
           (num_args+num_locals)*sizeof(struct pocl_argument));

    POCL_GOTO_ERROR_COND(
      (arg_info = malloc((num_args)*sizeof(struct pocl_argument_info))) 
      == NULL,
      CL_OUT_OF_HOST_MEMORY);
    memcpy(arg_info, kernel[i]->arg_info, 
           (num_args)*sizeof(struct pocl_argument_info));
    
    //INITIALIZE POCLCC KERNEL  
    poclcc_kernel *kernelcc = &(kernel_tab[i]);
    poclcc_init_kernel(kernelcc, kernel_name, sizeofKernelName, binary, fsize, 
                       num_args, num_locals, dyn_arguments, arg_info);
  }
  
  poclcc_init_device(devicecc, device, num_kernels, kernel_tab);

  return errcode;

ERROR:
  POCL_MEM_FREE(kernel_tab);
  return errcode;
}

cl_int compileForDevices(cl_program program)
{
  cl_int errcode = CL_SUCCESS;
  int num_devices = program->num_devices;

  if (program->BF != NULL && program->BF_sizes != NULL)
    return errcode;
  assert(program->BF == NULL);
  assert(program->BF_sizes == NULL);

  poclcc_global poclcc;
  poclcc_device *devicecc;
  POCL_RETURN_ERROR_COND(
    (devicecc = malloc(sizeof(poclcc_device)*num_devices)) == NULL,
    CL_OUT_OF_HOST_MEMORY);
  poclcc_init_global(&poclcc, num_devices, devicecc);

  int i;
  for (i=0; i<num_devices; i++){
    cl_device_id device = program->devices[i];
    errcode = compileKernels(program, device, &(poclcc.devices[i]));
    POCL_GOTO_ERROR_COND(errcode != CL_SUCCESS, errcode);
  }

  errcode = binaryFormat2ProgramInfos(&(program->BF), &(program->BF_sizes), &poclcc);
  POCL_GOTO_ERROR_COND(errcode != CL_SUCCESS, errcode);

  return errcode;

ERROR:
  poclcc_free(&poclcc);
  return errcode;
}

CL_API_ENTRY cl_int CL_API_CALL
POname(clGetProgramInfo)(cl_program program,
                 cl_program_info param_name,
                 size_t param_value_size,
                 void *param_value,
                 size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  unsigned i;
  cl_int errcode = CL_SUCCESS;

  POCL_RETURN_ERROR_COND((program == NULL), CL_INVALID_PROGRAM);

  switch (param_name)
  {
  case CL_PROGRAM_REFERENCE_COUNT:
    POCL_RETURN_GETINFO(cl_uint, (cl_uint)program->pocl_refcount);
  case CL_PROGRAM_CONTEXT:
    POCL_RETURN_GETINFO(cl_context, program->context);

  case CL_PROGRAM_SOURCE:
    {
      const char *source = program->source;
      if (source == NULL)
        source = "";

      POCL_RETURN_GETINFO_STR(source);
    }
    
  case CL_PROGRAM_BINARY_SIZES:
    {
      size_t const value_size = sizeof(size_t) * program->num_devices;
      if (param_value)
        errcode = compileForDevices(program);
      
      if (errcode != CL_SUCCESS)
        return errcode;

      POCL_RETURN_GETINFO_SIZE(value_size, program->BF_sizes);
    }

  case CL_PROGRAM_BINARIES:
    {
      size_t const value_size = sizeof(unsigned char *) * program->num_devices;
      if (param_value)
      {
        errcode = compileForDevices(program);

        if (errcode != CL_SUCCESS)
          return errcode;

        if (param_value_size < value_size) return CL_INVALID_VALUE;
        unsigned char **target = (unsigned char**) param_value;
        for (i = 0; i < program->num_devices; ++i)
          {
            if (target[i] == NULL) continue;
            memcpy (target[i], program->BF[i], program->BF_sizes[i]);
          }
      }
      if (param_value_size_ret)
        *param_value_size_ret = value_size;
      return CL_SUCCESS;
    }
  case CL_PROGRAM_NUM_DEVICES:
    POCL_RETURN_GETINFO(cl_uint, program->num_devices);

  case CL_PROGRAM_DEVICES:
    {
      size_t const value_size = sizeof(cl_device_id) * program->num_devices;
      if (param_value)
        {
          if (param_value_size < value_size) return CL_INVALID_VALUE;
          memcpy(param_value, (void*)program->devices, value_size);
        }
      if (param_value_size_ret)
        *param_value_size_ret = value_size;
      return CL_SUCCESS;
    }
  case CL_PROGRAM_NUM_KERNELS:
    {
      size_t num_kernels = pocl_llvm_get_kernel_count(program);
      POCL_RETURN_GETINFO(size_t, num_kernels);
    }
  case CL_PROGRAM_KERNEL_NAMES:
    {
      /* Note: In the specification (2.0) of the other XXXInfo
         functions, param_value_size_ret is described as follows:

         > param_value_size_ret returns the actual size in bytes of data
         > being *queried* by param_value.

         while in GetProgramInfo and APIs defined later in the
         documentation, it is:

         > param_value_size_ret returns the actual size in bytes of data
         > *copied* to param_value.

         it reads as if the spec allows the implementation to stop copying
         the string at an arbitrary point where the limit
         (param_value_size) is reached, but that's not the case. When it
         happens, it should instead raise an error CL_INVALID_VALUE.

         Also note the specification of the param_value_size_ret to param_name
         CL_PROGRAM_SOURCE.  It says "The actual number of characters that
         represents[sic] the program source code including the null terminator
         is returned in param_value_size_ret." By an analogy, it is sane to
         return the size of entire concatenated string, not the size of
         bytes copied (partially).

         Also note the specification of GetPlatformInfo +
         CL_PLATFORM_EXTENSIONS.  it refers to "param_value_size_ret" as
         the actual size in bytes of data being *queried*, and its
         description of param_value_size is the same.

         --- guicho271828
      */

      const char *kernel_names[32];
      unsigned num_kernels = 0;
      size_t size = 0;
      num_kernels = pocl_llvm_get_kernel_names(program, kernel_names, 32);

      /* optimized for clarity */
      for (i = 0; i < num_kernels; ++i)
        {
          size += strlen (kernel_names[i]) ;
          if (i != num_kernels - 1)
            size += 1;          /* a semicolon */
        }
      size += 1;                /* a NULL */
      if (param_value_size_ret)
        *param_value_size_ret = size;
      if (param_value)
        {
          /* only when param_value is non-NULL */
          if (size > param_value_size)
            return CL_INVALID_VALUE;
          /* should not break from the switch clause
             because of POCL_ABORT_UNIMPLEMENTED */
          for (i = 0; i < num_kernels; ++i)
            {
              if (i == 0)
                strcpy (param_value, kernel_names[i]); /* copy including NULL */
              else
                strcat ((char*)param_value, kernel_names[i]);
              if (i != num_kernels - 1)
                strcat ((char*)param_value, ";");
            }
        }
      return CL_SUCCESS;
    }
  default:
    break;
  }
  
  char error_str[64];
  sprintf(error_str, "clGetProgramInfo: %X", param_name);
  POCL_ABORT_UNIMPLEMENTED(error_str);
  return CL_INVALID_VALUE;
}
POsym(clGetProgramInfo)
