/* Pocl tool: poclcc

   Copyright (c) 2016 pocl developers

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

#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include <assert.h>
#include <string.h>
#include <errno.h>

#include "poclu.h"

#define DEVICE_INFO_MAX_LENGTH 2048
#define NUM_OF_DEVICE_ID 32
#define NUM_OPTIONS 6

#define ERRNO_EXIT(filename) do { \
    printf("IO error on file %s: %s\n", filename, strerror(errno)); \
    exit(2); \
  } while(0)

char *kernel_source = NULL;
char *output_file = NULL;
cl_uint opencl_device = CL_DEVICE_TYPE_DEFAULT;
unsigned opencl_device_id = 0;
int list_devices = 0;
int list_devices_only = 0;
char *build_options = NULL;

/**********************************************************/

typedef int(*poclcc_process)(int, char **, int);

typedef struct _poclcc_option {
  poclcc_process fct;
  char *id;
  char *helper;
  int num_args_read;
} poclcc_option;

/**********************************************************/

poclcc_option *options_help;

static int
print_help()
{
  printf("USAGE: poclcc [OPTION]... [FILE]\n");
  printf("\n");
  printf("OPTIONS:\n");
  int i;
  for (i=0; i<NUM_OPTIONS; i++)
    printf("%s", options_help[i].helper);

  return -1;
}

static int
poclcc_error(char *msg)
{
  printf("ERROR: %s", msg);
  return print_help();
}

/**********************************************************
 * MANAGE INPUT KERNEL FILE */

static int
process_kernel_file(int arg, char **argv, int argc)
{
  if (arg >= argc)
    return poclcc_error("Incomplete argument for input file!\n");

  char *filename = argv[arg];
  char *ext = ".pocl";
  kernel_source = poclu_read_file(filename);
  if (!kernel_source)
    ERRNO_EXIT(filename);
  if (output_file == NULL)
    {
      output_file = malloc (strlen (filename) + strlen (ext) + 2);
      strcpy(output_file, filename);
      strcat(output_file, ext);
    }
  return 0;
}

/**********************************************************
 * OPTIONS PROCESS FUNCTIONS*/

static int
process_help(int arg, char **argv, int argc)
{
  print_help();
  return 0;
}

static int
process_output(int arg, char **argv, int argc)
{
  if (arg >= argc)
    return poclcc_error("Incomplete argument for output file!\n");

  output_file = argv[arg];
  return 0;
}

static int
process_opencl_device(int arg, char **argv, int argc)
{
  if (arg >= argc)
    return poclcc_error("Incomplete argument for device_type!\n");

  char *opencl_string = argv[arg];
  if (!strcmp(opencl_string, "CL_DEVICE_TYPE_CPU"))
    opencl_device = CL_DEVICE_TYPE_CPU;
  else if (!strcmp(opencl_string, "CL_DEVICE_TYPE_GPU"))
    opencl_device = CL_DEVICE_TYPE_GPU;
  else if (!strcmp(opencl_string, "CL_DEVICE_TYPE_ACCELERATOR"))
    opencl_device = CL_DEVICE_TYPE_ACCELERATOR;
  else if (!strcmp(opencl_string, "CL_DEVICE_TYPE_DEFAULT"))
    opencl_device = CL_DEVICE_TYPE_DEFAULT;
  else if (!strcmp(opencl_string, "CL_DEVICE_TYPE_ALL"))
    opencl_device = CL_DEVICE_TYPE_ALL;
  else
    {
      printf("Invalid argument for device_type!\n");
      return print_help();
    }
  return 0;
}

static int
process_build_options(int arg, char **argv, int argc)
{
  if (arg >= argc)
    return poclcc_error("Incomplete argument for build_options!\n");

  build_options = argv[arg];
  return 0;
}

static int
process_device_id(int arg, char **argv, int argc)
{
  if (arg >= argc)
    return poclcc_error("Incomplete argument for build_options!\n");

  opencl_device_id = atoi(argv[arg]);
  return 0;
}

static int
process_list_devices(int arg, char **argv, int argc)
{
  list_devices = 1;
  opencl_device = CL_DEVICE_TYPE_ALL;
  return 0;
}

/**********************************************************/

static poclcc_option options[NUM_OPTIONS] =
{
  {process_help, "-h",
   "\t-h\n"
   "\t\tDisplay the help\n",
   1},
  {process_build_options, "-b",
   "\t-b <options>\n"
   "\t\tBuild the program with <options> options\n",
   2},
  {process_opencl_device, "-d",
   "\t-d <device_type>\n"
   "\t\tSelect <device_type> as the device_type for clGetDeviceIDs.\n"
   "\t\tDefault: CL_DEVICE_TYPE_DEFAULT\n",
   2},
  {process_list_devices, "-l",
   "\t-l\n"
   "\t\tList the opencl device found (that match the <device_type>\n",
   1},
  {process_device_id, "-i",
   "\t-i <device_id>\n"
   "\t\tSelect the <device_id> opencl device to generate the pocl binary file\n"
   "\t\tDefault: 0\n",
   2},
  {process_output, "-o",
   "\t-o <file>\n"
   "\t\tWrite output to <file>\n",
   2}
};

/**********************************************************/

static int
search_process(char *arg)
{
  int i;
  for (i=0; i<NUM_OPTIONS; i++)
    {
      if (!strcmp(options[i].id, arg))
        return i;
    }
  return -1;
}

static int
process_arg(int *arg, char **argv, int argc)
{
  int prev_arg = *arg;
  char *current_arg = argv[*arg];
  int current_process = search_process(current_arg);
  if (current_process == -1)
    return poclcc_error("Unknown argument!\n");
  else
    {
      poclcc_option * current_option = &options[current_process];
      int num_args_read = current_option->num_args_read;
      *arg = prev_arg + num_args_read;
      return current_option->fct(prev_arg + 1, argv, argc);
    }
}

/**********************************************************/

int
main(int argc, char **argv)
{
//MANAGEMENT OF ARGUMENTS
  options_help = options;
  int arg_num=1;
  if (argc < 2)
    return poclcc_error("Invalid argument!\n");

  while (arg_num < argc-1)
    if (process_arg(&arg_num, argv, argc))
      return -1;

  if (arg_num >= argc && list_devices)
    list_devices_only = 1;
  else if (arg_num >= argc)
    poclcc_error("Invalid arguments!\n");
  else
    {
      int current_process = search_process(argv[arg_num]);
      if (current_process == -1 && process_kernel_file(arg_num, argv, argc))
        return -1;
      else if (current_process != -1)
        {
          process_arg(&arg_num, argv, argc);
          list_devices_only = 1;
        }
    }

//OPENCL STUFF
  cl_platform_id cpPlatform;
  cl_device_id device_ids[NUM_OF_DEVICE_ID];
  cl_context context;
  cl_program program;
  cl_int err;
  cl_uint num_devices, i;

  CHECK_CL_ERROR(clGetPlatformIDs(1, &cpPlatform, NULL));

  CHECK_CL_ERROR(clGetDeviceIDs(cpPlatform, opencl_device,
                                NUM_OF_DEVICE_ID, device_ids, &num_devices));

  if (opencl_device_id >= num_devices)
    return poclcc_error("Invalid opencl device_id!\n");

  if (list_devices)
    {
      context = clCreateContext(0, num_devices, device_ids, NULL, NULL, &err);
      CHECK_OPENCL_ERROR_IN("clCreateContext");

      printf("LIST OF DEVICES:\n");
      for (i=0; i<num_devices; i++)
        {
          char str[DEVICE_INFO_MAX_LENGTH];
          CHECK_CL_ERROR(clGetDeviceInfo(device_ids[i], CL_DEVICE_VENDOR,
                                         DEVICE_INFO_MAX_LENGTH, str, NULL));

          printf("%i:\n  Vendor:   %s\n", i, str);

          CHECK_CL_ERROR(clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME,
                                         DEVICE_INFO_MAX_LENGTH, str, NULL));
          printf("    Name:   %s\n", str);

          // to print device->poclbin_hash_string
          CHECK_CL_ERROR(clGetDeviceInfo(device_ids[i], CL_DEVICE_VERSION,
                                         DEVICE_INFO_MAX_LENGTH, str, NULL));
          printf(" Version:   %s\n", str);
        }

      clReleaseContext(context);
    }
  if (list_devices_only)
    return 0;

  context = clCreateContext(0, 1, &device_ids[opencl_device_id], NULL, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateContext");

  program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

  CHECK_CL_ERROR(clBuildProgram(program, 0, NULL, build_options, NULL, NULL));

  size_t binary_sizes;
  char *binary;

  CHECK_CL_ERROR(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                                  sizeof(size_t), &binary_sizes, NULL));

  binary = malloc(sizeof(char)*binary_sizes);
  if (!binary)
    {
      printf("malloc(binary) failed\n");
      exit(1);
    }

  CHECK_CL_ERROR(clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                                  sizeof(unsigned char*), &binary, NULL));

  CHECK_CL_ERROR(clReleaseProgram(program));
  CHECK_CL_ERROR(clReleaseContext(context));

  if (poclu_write_file(output_file, binary, binary_sizes))
    ERRNO_EXIT(output_file);

  free(binary);

  return 0;
}
