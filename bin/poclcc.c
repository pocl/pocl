#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include <assert.h>
#include <string.h>

#define DEVICE_INFO_MAX_LENGTH 128
#define NUM_OF_DEVICE_ID 32
#define NUM_OPTIONS 6

char *kernel_source = NULL;
char *output_file = NULL;
int opencl_device = CL_DEVICE_TYPE_DEFAULT;
int opencl_device_id = 0;
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
int print_help()
{
  printf("USAGE: poclcc [OPTION]... [FILE]\n");
  printf("\n");
  printf("OPTIONS:\n");
  int i;
  for (i=0; i<NUM_OPTIONS; i++)
    printf("%s", options_help[i].helper);

  return -1;
}

int poclcc_error(char *msg)
{
  printf("ERROR: %s", msg);
  return print_help();
}

/**********************************************************
 * MANAGE INPUT KERNEL FILE */

char *kernel_load_file(const char * filename)
{
  char *buffer;
  size_t size, read_size;
  FILE *kern_file;

  kern_file = fopen(filename, "r");
  if (kern_file == NULL) 
    return NULL;

  fseek(kern_file, 0, SEEK_END);
  size = ftell(kern_file);
  rewind(kern_file);

  buffer = malloc(size + 1);
  read_size = fread(buffer, 1, size, kern_file);
  if (read_size != size) 
    {
      free(buffer);
      fclose(kern_file);
      return NULL;
    }
  fclose(kern_file);
  buffer[size] = '\0';

  return buffer;
}

int process_kernel_file(int arg, char **argv, int argc)
{
  if (arg >= argc)
    return poclcc_error("Incomplete argument for input file!\n");
     
  char *filename = argv[arg];
  char *ext = ".pocl";
  kernel_source = kernel_load_file(filename);
  if (output_file == NULL)
    {
      output_file = malloc(strlen(filename)+strlen(ext));
      strcpy(output_file, filename);
      strcat(output_file, ext);
    }
  return 0;
}

/**********************************************************
 * OPTIONS PROCESS FUNCTIONS*/

int process_help(int arg, char **argv, int argc)
{
  print_help();
  return 0;
}

int process_output(int arg, char **argv, int argc)
{
  if (arg >= argc)
    return poclcc_error("Incomplete argument for output file!\n");

  output_file = argv[arg];
  return 0;
}

int process_opencl_device(int arg, char **argv, int argc)
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

int process_build_options(int arg, char **argv, int argc)
{
  if (arg >= argc)
    return poclcc_error("Incomplete argument for build_options!\n");

  build_options = argv[arg];
  return 0;
}

int process_device_id(int arg, char **argv, int argc)
{
  if (arg >= argc)
    return poclcc_error("Incomplete argument for build_options!\n");

  opencl_device_id = atoi(argv[arg]);  
  return 0;
}

int process_list_devices(int arg, char **argv, int argc)
{
  list_devices = 1;
  return 0;
}

/**********************************************************/

poclcc_option options[NUM_OPTIONS]=
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

int search_process(char *arg)
{
  int i;
  for (i=0; i<NUM_OPTIONS; i++)
    {
      if (!strcmp(options[i].id, arg))
        return i;
    }
  return -1;
}

int process_arg(int *arg, char **argv, int argc)
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

int main(int argc, char **argv) 
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
  cl_device_id device_id[NUM_OF_DEVICE_ID];
  cl_context context;
  cl_program program;
  cl_int err;
  cl_uint num_devices, i;

  err = clGetPlatformIDs(1, &cpPlatform, NULL);

  err = clGetDeviceIDs(cpPlatform, opencl_device, NUM_OF_DEVICE_ID, device_id, &num_devices);
  assert(!err && "clGetDeviceIDs failed");
 
  if (opencl_device_id >= num_devices)
    return poclcc_error("Invalid opencl device_id!\n");
     
  if (list_devices)
    {
      context = clCreateContext(0, num_devices, device_id, NULL, NULL, &err);
      assert(context && "clCreateContext failed");
  
      printf("LIST OF DEVICES:\n");
      for (i=0; i<num_devices; i++)
        {
          printf("%i: ", i);

          char str[DEVICE_INFO_MAX_LENGTH];
          err = clGetDeviceInfo(device_id[i], CL_DEVICE_VENDOR, 
                                DEVICE_INFO_MAX_LENGTH, str, NULL);
          assert(!err);
          printf("%s --- ", str);

          err = clGetDeviceInfo(device_id[i], CL_DEVICE_NAME, 
                                DEVICE_INFO_MAX_LENGTH, str, NULL);
          assert(!err);
          printf("%s\n", str);
        }

      clReleaseContext(context);
    }
  if (list_devices_only)
    return 0;

  context = clCreateContext(0, 1, &device_id[opencl_device_id], NULL, NULL, &err);
  assert(context && "clCreateContext failed");  

  program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
  assert(program && "clCreateProgramWithSource failed");

  err = clBuildProgram(program, 0, NULL, build_options, NULL, NULL);  
  assert(!err && "clBuildProgram failed");

  size_t binary_sizes;
  unsigned char *binary;

  err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_sizes, NULL);
  assert(!err);
  
  binary = malloc(sizeof(unsigned char)*binary_sizes);
  assert(binary);
  
  err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &binary, NULL);
  assert(!err);

//GENERATE FILE
  FILE *fp=fopen(output_file, "w"); 
  fwrite(binary, 1, binary_sizes, fp);
  fclose(fp);

//RELEASE OPENCL STUFF
  clReleaseProgram(program);
  clReleaseContext(context);

  free(binary);

  return 0;
}
