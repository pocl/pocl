/* OpenCL runtime library: compile_and_link_program()

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos,
                 2011-2023 Pekka Jääskeläinen

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal
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
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include <assert.h>
#include <fcntl.h>
#include <stdarg.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef _WIN32
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include "pocl_cl.h"
#ifdef ENABLE_LLVM
#include "pocl_llvm.h"
#endif
#include "pocl_util.h"
#include "pocl_file_util.h"
#include "pocl_cache.h"
#include "config.h"
#include "pocl_runtime_config.h"
#include "pocl_binary.h"
#include "pocl_shared.h"

#define REQUIRES_CR_SQRT_DIV_ERR                                              \
  "-cl-fp32-correctly-rounded-divide-sqrt build option "                      \
  "was specified, but CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT "                   \
  "is not set for device"

/* supported compiler parameters which should pass to the frontend directly
   by using -Xclang */
static const char cl_parameters[] =
  "-cl-single-precision-constant "
  "-cl-fp32-correctly-rounded-divide-sqrt "
  "-cl-opt-disable "
  "-cl-mad-enable "
  "-cl-unsafe-math-optimizations "
  "-cl-finite-math-only "
  "-cl-fast-relaxed-math "
  "-cl-std=CL1.2 "
  "-cl-std=CL1.1 "
  "-cl-std=CL2.0 "
  "-cl-std=CL2.1 "
  "-cl-std=CL2.2 "
  "-cl-std=CL3.0 "
  "-cl-kernel-arg-info "
  "-cl-strict-aliasing "
  "-cl-denorms-are-zero "
  "-cl-no-signed-zeros "
  "-w "
  "-g "
  "-Werror ";

/*
static const char cl_library_link_options[] =
  "-create-library "
  "-enable-link-options ";
*/

static const char cl_program_link_options[] =
  "-cl-denorms-are-zero "
  "-cl-no-signed-zeros "
  "-cl-unsafe-math-optimizations "
  "-cl-finite-math-only "
  "-cl-fast-relaxed-math ";

/* TODO: In case of a PoCL-R, we should pass on unhandled
   target-specific/extension-specific options to the
   native driver's clBuildProgram() call. Might be difficult
   to filter the kept ones as the set of target devices can
   be diverse. */
static const char cl_parameters_not_yet_supported_by_clang[]
    = "-cl-uniform-work-group-size "
      "-cl-no-subgroup-ifp "
      "-cl-intel-no-prera-scheduling";

#define MEM_ASSERT(x, err_jmp) do{ if (x){errcode = CL_OUT_OF_HOST_MEMORY;goto err_jmp;}} while(0)

// append token, growing modded_options, if necessary, by max(strlen(token)+1, 256)
#define APPEND_TOKEN()                                                        \
  do                                                                          \
    {                                                                         \
      needed = strlen (token) + 1;                                            \
      assert (size > (i + needed));                                           \
      i += needed;                                                            \
      strcat (modded_options, token);                                         \
      strcat (modded_options, " ");                                           \
    }                                                                         \
  while (0)

#define APPEND_TO_OPTION_BUILD_LOG(...)                                       \
  do                                                                          \
    {                                                                         \
      POCL_MSG_ERR (__VA_ARGS__);                                             \
      size_t l = strlen (program->main_build_log);                            \
      if (l < 640)                                                            \
        snprintf (program->main_build_log + l, (640 - l), __VA_ARGS__);       \
    }                                                                         \
  while (0)

static void
append_to_build_log (cl_program program, unsigned device_i, const char *format,
                     ...)
{
  char temp[4096];
  va_list args;
  va_start (args, format);
  ssize_t written = vsnprintf (temp, 4096, format, args);
  va_end (args);
  size_t l = 0;
  if (written > 0)
    {
      if (written > 4096)
        written = 4096;
      if (program->build_log[device_i])
        l = strlen (program->build_log[device_i]);
      size_t newl = l + (size_t)written;
      char *newp = (char *)realloc (program->build_log[device_i], newl + 1);
      assert (newp);
      memcpy (newp + l, temp, (size_t)written);
      newp[newl] = 0;
      program->build_log[device_i] = newp;
    }
}

#define APPEND_TO_BUILD_LOG_GOTO(err, ...)                                    \
  do                                                                          \
    {                                                                         \
      append_to_build_log (program, device_i, __VA_ARGS__);                   \
      if (err == CL_COMPILE_PROGRAM_FAILURE)                                  \
        POCL_MSG_ERR2 ("CL_COMPILE_PROGRAM_FAILURE", __VA_ARGS__);            \
      if (err == CL_BUILD_PROGRAM_FAILURE)                                    \
        POCL_MSG_ERR2 ("CL_BUILD_PROGRAM_FAILURE", __VA_ARGS__);              \
      else                                                                    \
        POCL_MSG_ERR2 (#err, __VA_ARGS__);                                    \
      errcode = err;                                                          \
      goto ERROR;                                                             \
    }                                                                         \
  while (0)

/* options must be non-NULL.
 * modded_options[size] + link_options are preallocated outputs
 */
static cl_int
process_options (const char *options, char *modded_options, char *link_options,
                 cl_program program, int compiling, int linking,
                 int *create_library, unsigned *flush_denorms,
                 int *requires_correctly_rounded_sqrt_div, int *spir_build,
                 cl_version *cl_c_version, size_t size)
{
  cl_int error;
  char *token = NULL;
  char *saveptr = NULL;

  *create_library = 0;
  *flush_denorms = 0;
  *cl_c_version = 0;
  *requires_correctly_rounded_sqrt_div = 0;
  *spir_build = 0;
  int enable_link_options = 0;
  link_options[0] = 0;
  modded_options[0] = 0;
  int ret_error = (linking ? (compiling ? CL_INVALID_BUILD_OPTIONS
                                        : CL_INVALID_LINKER_OPTIONS)
                           : CL_INVALID_COMPILER_OPTIONS);

  assert (options);
  assert (modded_options);
  assert (compiling || linking);

  char replace_me = 0;

  size_t i = 1; /* terminating char */
  size_t needed = 0;
  char *temp_options = (char *) malloc (strlen (options) + 1);

  memset (temp_options, 0, strlen (options) + 1);
  strncpy (temp_options, options, strlen (options));

  if (pocl_escape_quoted_whitespace (temp_options, &replace_me) == -1)
  {
    error = CL_INVALID_BUILD_OPTIONS;
    goto ERROR;
  }

  token = strtok_r (temp_options, " ", &saveptr);
  while (token != NULL)
    {
      /* check if parameter is supported compiler parameter */
      if (strncmp (token, "-cl", 3) == 0 || strncmp (token, "-w", 2) == 0
          || strncmp (token, "-Werror", 7) == 0)
        {
          if (strstr (cl_program_link_options, token))
            {
              /* when linking, only a subset of -cl* options are valid,
               * and only with -enable-link-options */
              if (linking && (!compiling))
                {
                  if (!enable_link_options)
                    {
                      APPEND_TO_OPTION_BUILD_LOG (
                          "Not compiling but link options were not enabled, "
                          "therefore %s is an invalid option\n",
                          token);
                      error = ret_error;
                      goto ERROR;
                    }
                  strcat (link_options, token);
                }
              if (strstr (token, "-cl-denorms-are-zero"))
                {
                  *flush_denorms = 1;
                }
              if (strstr (token, "-cl-fp32-correctly-rounded-divide-sqrt"))
                {
                  *requires_correctly_rounded_sqrt_div = 1;
                }
            }

          if (strstr (cl_parameters, token))
            {
              /* the LLVM API call pushes the parameters directly to the
                 frontend without using -Xclang */

            // LLVM 11 has removed "-cl-denorms-are-zero" option
            // https://reviews.llvm.org/D69878
            if (strncmp(token, "-cl-denorms-are-zero", 20) == 0) {
                token = "-fdenormal-fp-math=positive-zero";
            }

            if (strncmp (token, "-cl-std=CL", 10) == 0)
            {
                unsigned major = token[10] - '0';
                unsigned minor = token[12] - '0';
                *cl_c_version = CL_MAKE_VERSION (major, minor, 0);
            }
            }
          else if (strstr (cl_parameters_not_yet_supported_by_clang, token))
            {
              APPEND_TO_OPTION_BUILD_LOG (
                  "This build option is not yet supported by clang: %s\n",
                  token);
              token = strtok_r (NULL, " ", &saveptr);
              continue;
            }
          else
            {
              APPEND_TO_OPTION_BUILD_LOG ("Invalid build option: %s\n", token);
              error = ret_error;
              goto ERROR;
            }
        }
      else if (strncmp (token, "-g", 2) == 0)
        {
          token = "-debug-info-kind=limited " \
          "-dwarf-version=4 -debugger-tuning=gdb";
        }
      else if (strncmp (token, "-D", 2) == 0 || strncmp (token, "-I", 2) == 0)
        {
          APPEND_TOKEN();
          /* if there is a space in between, then next token is part
             of the option */
          if (strlen (token) == 2)
            token = strtok_r (NULL, " ", &saveptr);
          else
            {
              token = strtok_r (NULL, " ", &saveptr);
              continue;
            }
        }
      else if (strncmp (token, "-x", 2) == 0 && strlen (token) == 2)
        {
          /* only "-x spir" is valid for the "-x" option */
          token = strtok_r (NULL, " ", &saveptr);
          if (!token || strncmp (token, "spir", 4) != 0)
            {
              APPEND_TO_OPTION_BUILD_LOG (
                  "Invalid parameter to -x build option\n");
              error = ret_error;
              goto ERROR;
            }
          /* "-x spir" is not valid if we are building from source */
          else if (program->source)
            {
              APPEND_TO_OPTION_BUILD_LOG (
                  "\"-x spir\" is not valid when building from source\n");
              error = ret_error;
              goto ERROR;
            }
          else
            *spir_build = 1;
          token = strtok_r (NULL, " ", &saveptr);
          continue;
        }
      else if (strncmp (token, "-spir-std=1.2", 13) == 0)
        {
          /* "-spir-std=" flags are not valid when building from source */
          if (program->source)
            {
              APPEND_TO_OPTION_BUILD_LOG ("\"-spir-std=\" flag is not valid "
                                          "when building from source\n");
              error = ret_error;
              goto ERROR;
            }
          else
            *spir_build = 1;
          token = strtok_r (NULL, " ", &saveptr);
          continue;
        }
      else if (strncmp (token, "-create-library", 15) == 0)
        {
          if (!linking)
            {
              APPEND_TO_OPTION_BUILD_LOG (
                  "\"-create-library\" flag is only valid when linking\n");
              error = ret_error;
              goto ERROR;
            }
          *create_library = 1;
          token = strtok_r (NULL, " ", &saveptr);
          continue;
        }
      else if (strncmp (token, "-enable-link-options", 20) == 0)
        {
          if (!linking)
            {
              APPEND_TO_OPTION_BUILD_LOG ("\"-enable-link-options\" flag is "
                                          "only valid when linking\n");
              error = ret_error;
              goto ERROR;
            }
          if (!(*create_library))
            {
              APPEND_TO_OPTION_BUILD_LOG ("\"-enable-link-options\" flag is "
                                          "only valid when -create-library "
                                          "option was given\n");
              error = ret_error;
              goto ERROR;
            }
          enable_link_options = 1;
          token = strtok_r (NULL, " ", &saveptr);
          continue;
        }
      else
        {
          APPEND_TO_OPTION_BUILD_LOG ("Invalid build option: %s\n", token);
          error = ret_error;
          goto ERROR;
        }
      APPEND_TOKEN ();
      token = strtok_r (NULL, " ", &saveptr);
    }

  error = CL_SUCCESS;

  /* remove trailing whitespace */
  i = strlen (modded_options);
  if ((i > 0) && (modded_options[i - 1] == ' '))
    modded_options[i - 1] = 0;

  /* put back replaced whitespaces if needed */
  if (replace_me != 0)
  {
    for (size_t x = 0; x < i; x++)
    {
      if (modded_options[x] == replace_me) modded_options[x] = ' ';
    }
  }

ERROR:
  POCL_MEM_FREE (temp_options);
  return error;
}

/* Unique hash for a device + program build + kernel name combination.
   NOTE: this does NOT take into account the local WG sizes or other
   specialization properties. */
static void
pocl_calculate_kernel_hash (cl_program program, unsigned kernel_i,
                            unsigned device_i)
{
  SHA1_CTX hash_ctx;
  pocl_SHA1_Init (&hash_ctx);

  char *n = program->kernel_meta[kernel_i].name;
  assert (n != NULL && program->build_hash[device_i] != NULL);
  pocl_SHA1_Update (&hash_ctx, (uint8_t *)program->build_hash[device_i],
                    sizeof (SHA1_digest_t));
  pocl_SHA1_Update (&hash_ctx, (uint8_t *)n, strlen (n));

  uint8_t digest[SHA1_DIGEST_SIZE];
  pocl_SHA1_Final (&hash_ctx, digest);

  memcpy (program->kernel_meta[kernel_i].build_hash[device_i], digest,
          sizeof (pocl_kernel_hash_t));
}

static void
free_meta (cl_program program)
{
  size_t i;
  unsigned j;

  if (program->num_kernels)
    {
      for (i = 0; i < program->num_kernels; i++)
        {
          pocl_kernel_metadata_t *meta = &program->kernel_meta[i];
          if (meta->builtin_kernel)
            continue;
          pocl_free_kernel_metadata (program, i);
        }
      POCL_MEM_FREE (program->kernel_meta);
    }
}

static void
clean_program_on_rebuild (cl_program program, int from_error)
{
  /* if we're rebuilding the program, release the kernels and reset log/status
   */
  cl_uint i;
  if (!from_error && (program->build_status == CL_BUILD_NONE))
    return;

  /* CL_INVALID_OPERATION if there are kernel objects attached to program.
     ...and we check for that earlier.
   */
  assert (program->kernels == NULL);

  free_meta (program);

  program->num_kernels = 0;
  program->build_status = CL_BUILD_NONE;
  program->binary_type = CL_PROGRAM_BINARY_TYPE_NONE;

  for (i = 0; i < program->num_devices;
       ++i) // TODO associated_num_devices or not ???
    {
      cl_device_id dev = program->devices[i];
      if (!from_error)
        POCL_MEM_FREE (program->build_log[i]);
      memset (program->build_hash[i], 0, sizeof (SHA1_digest_t));
      if (program->source)
        {
          if (dev->ops->free_program)
            dev->ops->free_program (dev, program, i);
          POCL_MEM_FREE (program->binaries[i]);
          program->binary_sizes[i] = 0;
          POCL_MEM_FREE (program->pocl_binaries[i]);
          program->pocl_binary_sizes[i] = 0;
        }
      program->global_var_total_size[i] = 0;
    }

  if (!from_error)
    {
      if (program->devices != program->context->devices
          && program->devices != program->associated_devices)
        {
          POCL_MEM_FREE (program->devices);
        }
      program->num_devices = 0;
      program->main_build_log[0] = 0;
    }
}

static int
setup_kernel_metadata (cl_program program)
{
  size_t i, j;
  cl_uint device_i;
  assert (program->kernel_meta == NULL);
  assert (program->num_kernels == 0);
  int setup_successful = 0;

  /* Get the kernel metadata, either from pocl binaries or device drivers */
  for (device_i = 0; device_i < program->num_devices; device_i++)
    {
      cl_device_id device = program->devices[device_i];
      if (program->pocl_binaries[device_i])
        {
          program->num_kernels
              = pocl_binary_get_kernel_count (program, device_i);
          if (program->num_kernels)
            {
              program->kernel_meta = (pocl_kernel_metadata_t *)calloc (
                  program->num_kernels, sizeof (pocl_kernel_metadata_t));
              pocl_binary_get_kernels_metadata (program, device_i);
            }
          setup_successful = 1;
          break;
        }
      else
        {
          if (device->ops->setup_metadata
              && device->ops->setup_metadata (device, program, device_i))
            {
              setup_successful = 1;
              break;
            }
        }
    }

  POCL_RETURN_ERROR_ON (
      (setup_successful == 0), CL_INVALID_BINARY,
      "Could not find kernel metadata in the built program\n");

  /* calculate argument storage size */
  for (i = 0; i < program->num_kernels; ++i)
    {
      program->kernel_meta[i].total_argument_storage_size = 0;
      if (program->kernel_meta[i].num_args > 0)
        {
          size_t total = 0;
          for (j = 0; j < program->kernel_meta[i].num_args; ++j)
            {
              /* if one of the arguments have size 0,
                 the driver couldn't figure it out. In that case,
                 leave total_argument_storage_size == zero, and use
                 the old way of setting arguments. */
              if (program->kernel_meta[i].arg_info[j].type_size == 0)
                break;
              total += program->kernel_meta[i].arg_info[j].type_size;
            }
          if (j >= program->kernel_meta[i].num_args)
            program->kernel_meta[i].total_argument_storage_size = total;
        }
    }

  return CL_SUCCESS;
}

static void
setup_device_kernel_hashes (cl_program program)
{
  cl_uint i, device_i;

  if ((program->num_kernels == 0) || (program->num_devices == 0))
    return;

  assert (program->kernel_meta);
  for (i = 0; i < program->num_kernels; ++i)
    {
      assert (program->kernel_meta[i].build_hash == NULL);
      program->kernel_meta[i].build_hash = (pocl_kernel_hash_t *)calloc (
          program->num_devices, sizeof (pocl_kernel_hash_t));
    }

  for (device_i = 0; device_i < program->num_devices; device_i++)
    {
      for (i = 0; i < program->num_kernels; ++i)
        {
          /* calculate device-specific kernel hashes. */
          pocl_calculate_kernel_hash (program, i, device_i);
        }
    }
}

static int
check_device_supports (cl_device_id device, cl_version cl_c_version)
{
  if (device->num_opencl_c_with_version > 0)
    {
      for (size_t i = 0; i < device->num_opencl_c_with_version; ++i)
        {
          if (device->opencl_c_with_version[i].version == cl_c_version)
            return 0;
        }
      return -1;
    }
  else
    {
      return cl_c_version > device->opencl_c_version_as_cl;
    }
}

cl_int
compile_and_link_program(int compile_program,
                         int link_program,
                         cl_program program,
                         cl_uint num_devices,
                         const cl_device_id *device_list,
                         const char *options,
                         cl_uint num_input_headers,
                         const cl_program *input_headers,
                         const char **header_include_names,
                         cl_uint num_input_programs,
                         const cl_program *input_programs,
                         void (CL_CALLBACK *pfn_notify) (cl_program program,
                                                         void *user_data),
                         void *user_data)
{
  char link_options[512];
  int errcode, error;
  int create_library = 0;
  int requires_cr_sqrt_div = 0;
  int spir_build = 0;
  cl_version cl_c_version = 0;
  unsigned flush_denorms = 0;
  cl_device_id *unique_devlist = NULL;
  unsigned device_i = 0, actually_built = 0;
  size_t i;
  char *temp_options = NULL;

  const char *extra_build_options =
    pocl_get_string_option ("POCL_EXTRA_BUILD_FLAGS", NULL);

  int build_error_code
      = (link_program ? CL_BUILD_PROGRAM_FAILURE : CL_COMPILE_PROGRAM_FAILURE);

  POCL_GOTO_LABEL_COND (PFN_NOTIFY, (!IS_CL_OBJECT_VALID (program)),
                        CL_INVALID_PROGRAM);

  POCL_GOTO_LABEL_COND (PFN_NOTIFY, (num_devices > 0 && device_list == NULL),
                        CL_INVALID_VALUE);
  POCL_GOTO_LABEL_COND (PFN_NOTIFY, (num_devices == 0 && device_list != NULL),
                        CL_INVALID_VALUE);

  POCL_GOTO_LABEL_COND (PFN_NOTIFY, (pfn_notify == NULL && user_data != NULL),
                        CL_INVALID_VALUE);

  POCL_LOCK_OBJ (program);

  POCL_GOTO_LABEL_ON (FINISH, program->kernels, CL_INVALID_OPERATION,
                      "Program already has kernels\n");

  POCL_GOTO_LABEL_ON (
      FINISH,
      (program->source == NULL && program->binaries == NULL
       && program->builtin_kernel_names == NULL),
      CL_INVALID_PROGRAM,
      "Program doesn't have sources, binaries nor builtin-kernel names. You "
      "need "
      "to call clCreateProgramWith{Binary|Source|BuiltinKernels} first\n");

  POCL_GOTO_LABEL_ON (FINISH,
                      ((program->source == NULL) &&
                        (program->program_il == NULL) && (link_program == 0)),
                      CL_INVALID_OPERATION,
                      "Cannot clCompileProgram when program has no source\n");

  program->main_build_log[0] = 0;

  TP_BUILD_PROGRAM (program->context->id, program->id);

  /* TODO this should be somehow utilized at linking */
  POCL_MEM_FREE (program->compiler_options);

  if (extra_build_options)
    {
      size_t len = (options != NULL) ? strlen (options) : 0;
      len += strlen (extra_build_options) + 2;
      temp_options = (char *)malloc (len);
      temp_options[0] = 0;
      if (options != NULL)
        {
          strcpy (temp_options, options);
          strcat (temp_options, " ");
        }
      strcat (temp_options, extra_build_options);
    }
  else
    temp_options = (char*) options;

  if (temp_options)
    {
      i = strlen (temp_options);
      size_t size = i + 512; /* add some space for pocl-added options */
      program->compiler_options = (char *)malloc (size);
      errcode = process_options (
          temp_options, program->compiler_options, link_options, program,
          compile_program, link_program, &create_library, &flush_denorms,
          &requires_cr_sqrt_div, &spir_build, &cl_c_version, size);
      if (errcode != CL_SUCCESS)
        goto ERROR_CLEAN_OPTIONS;
    }

  /* The build option -x spir is only needed for the old SPIR format.
     When creating a SPIR-V program via clCreateProgramWithIL, it's not
     needed and we just assume if the program_il blob is there, we want
     to also build it. */
  spir_build = spir_build || program->program_il != NULL;

  POCL_MSG_PRINT_LLVM ("building program with options %s\n",
                       program->compiler_options);

  program->flush_denorms = flush_denorms;
  clean_program_on_rebuild (program, 0);

  /* adjust device list to what we're building for */
  if (num_devices == 0)
    {
      program->num_devices = program->associated_num_devices;
      program->devices = program->associated_devices;
    }
  else
    {
      // convert subdevices to devices and remove duplicates
      cl_uint real_num_devices = 0;
      unique_devlist = pocl_unique_device_list (device_list, num_devices,
                                                &real_num_devices);
      program->num_devices = real_num_devices;
      program->devices = unique_devlist;
    }

  /* if program will be compiled using clCompileProgram its binary_type
   * will be set to CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT.
   *
   * if program was created by clLinkProgram which is called
   * with the –createlibrary link option its binary_type will be set to
   * CL_PROGRAM_BINARY_TYPE_LIBRARY.
   */
  program->binary_type = CL_PROGRAM_BINARY_TYPE_EXECUTABLE;
  if (create_library)
    program->binary_type = CL_PROGRAM_BINARY_TYPE_LIBRARY;
  if (compile_program && !link_program)
    program->binary_type = CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT;
  if (program->num_builtin_kernels > 0)
    program->binary_type = CL_PROGRAM_BINARY_TYPE_NONE;

  POCL_MSG_PRINT_LLVM ("building program for %u devs with options %s\n",
                       program->num_devices, program->compiler_options);

  for (device_i = 0; device_i < program->num_devices; ++device_i)
    POCL_MSG_PRINT_LLVM ("   BUILDING for device: %s\n",
                         program->devices[device_i]->short_name);

  /* check the devices in the supplied devices-to-build-for list */
  cl_uint num_found = 0;
  for (i = 0; i < program->num_devices; ++i)
    {
      cl_device_id dev = program->devices[i];
      POCL_GOTO_ERROR_COND ((*dev->available == CL_FALSE),
                            CL_DEVICE_NOT_AVAILABLE);
      for (cl_uint j = 0; j < program->associated_num_devices; ++j)
        if (program->associated_devices[j] == dev)
          ++num_found;
    }
  POCL_GOTO_ERROR_ON (
      (num_found < program->num_devices), build_error_code,
      "Some of the devices on the argument-supplied list are "
      "not available for the program, or do not exist: %u < %u\n",
      actually_built, program->num_devices);

  /* Build the program for all requested devices. */
  for (device_i = 0; device_i < program->num_devices; ++device_i)
    {
      cl_device_id device = program->devices[device_i];

      if (cl_c_version && check_device_supports (device, cl_c_version))
        {
          APPEND_TO_BUILD_LOG_GOTO (
              build_error_code,
              "Build option -cl-std specified OpenCL C version %u.%u,"
              "but device %s doesn't support that OpenCL C version.\n",
              CL_VERSION_MAJOR (cl_c_version), CL_VERSION_MINOR (cl_c_version),
              device->short_name);
        }

      if (requires_cr_sqrt_div
          && !(device->single_fp_config & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT))
        APPEND_TO_BUILD_LOG_GOTO (build_error_code,
                                  REQUIRES_CR_SQRT_DIV_ERR " %s\n",
                                  device->short_name);

      /* clCreateProgramWithBuiltinKernels */
      if (program->builtin_kernel_names)
        {
          if (device->ops->build_builtin)
            {
              error = device->ops->build_builtin (program, device_i);
              if (error != CL_SUCCESS)
                APPEND_TO_BUILD_LOG_GOTO (CL_BUILD_PROGRAM_FAILURE,
                                          "Device %s failed to build the "
                                          "program with builtin kernels\n",
                                          device->long_name);
            }
        }
      /* only link the program/library */
      else if (!compile_program && link_program)
        {
          assert (num_input_programs > 0);

          if (device->ops->link_program == NULL)
            APPEND_TO_BUILD_LOG_GOTO (CL_LINK_PROGRAM_FAILURE,
                                      "%s device's driver does "
                                      "not support linking programs\n",
                                      device->long_name);

          error = device->ops->link_program (program, device_i,
                                             num_input_programs,
                                             input_programs, create_library);
          if (error != CL_SUCCESS)
            APPEND_TO_BUILD_LOG_GOTO (CL_LINK_PROGRAM_FAILURE,
                                      "Device %s failed to link the program\n",
                                      device->long_name);
        }
      /* compile and/or link from source */
      else if (program->source)
        {
          if (device->ops->build_source == NULL)
            APPEND_TO_BUILD_LOG_GOTO (
                build_error_code,
                "%s device's driver does not "
                "support building programs from source\n",
                device->long_name);

          error = device->ops->build_source (
              program, device_i, num_input_headers, input_headers,
              header_include_names, (create_library ? 0 : link_program));

          if (error != CL_SUCCESS)
            {
              if (program->build_log[device_i])
                POCL_MSG_ERR ("Build log for device %s:\n%s\n",
                              device->long_name, program->build_log[device_i]);
              APPEND_TO_BUILD_LOG_GOTO (build_error_code,
                                        "Device %s failed to build"
                                        " the program\n",
                                        device->long_name);
            }
        }
      /* compile and/or link from binary */
      else
        {
          if (device->ops->build_binary == NULL)
            APPEND_TO_BUILD_LOG_GOTO (build_error_code,
                                      "%s device's driver does not support "
                                      "building programs from binaries\n",
                                      device->long_name);

          if ((program->binary_sizes[device_i] == 0)
              && (program->pocl_binary_sizes[device_i] == 0)
              && (program->program_il_size == 0))
            APPEND_TO_BUILD_LOG_GOTO (CL_INVALID_BINARY,
                                      "No poclbinaries nor binaries "
                                      "for device %s - can't build "
                                      "the program\n",
                                      device->short_name);

          error = device->ops->build_binary (
              program, device_i, (create_library ? 0 : link_program),
              spir_build);

          if (error != CL_SUCCESS)
            {
              if (program->build_log[device_i])
                POCL_MSG_ERR ("Build log for device %s:\n%s\n",
                              device->long_name, program->build_log[device_i]);
              APPEND_TO_BUILD_LOG_GOTO (build_error_code,
                                        "Device %s failed to build"
                                        " the program\n",
                                        device->long_name);
            }
        }

      /* Maintain a 'last_accessed' file in every program's
       * cache directory. Will be useful for a cache pruning script
       * that flushes old directories based on LRU */
      if (!program->builtin_kernel_names)
        pocl_cache_update_program_last_access (program, device_i);

      ++actually_built;
    }
  assert (actually_built == program->num_devices);
  assert(program->num_kernels == 0);

  /* for executables & programs with builtin kernels,
   * setup the kernel metadata */
  /* if the program is not a finished executable, we don't need
   * to setup kernel metadata */
  if (program->binary_type == CL_PROGRAM_BINARY_TYPE_EXECUTABLE
      || program->binary_type == CL_PROGRAM_BINARY_TYPE_NONE)
    {
      errcode = setup_kernel_metadata (program);
      if (errcode != CL_SUCCESS)
        {
          POCL_MSG_ERR ("Program build: kernel metadata setup failed\n");
          goto ERROR;
        }

      setup_device_kernel_hashes (program);
    }

  for (device_i = 0; device_i < program->num_devices; device_i++)
    {
      cl_device_id device = program->devices[device_i];

      if (device->ops->post_build_program)
        {
          errcode = device->ops->post_build_program (program, device_i);
          if (errcode != CL_SUCCESS)
            {
              POCL_MSG_ERR ("Program build: post-build-program failed\n");
              goto ERROR;
            }
        }
    }

  TP_BUILD_PROGRAM (program->context->id, program->id);

  program->build_status = CL_BUILD_SUCCESS;
  errcode = CL_SUCCESS;
  goto FINISH;

ERROR:
  clean_program_on_rebuild (program, 1);

ERROR_CLEAN_OPTIONS:
  if (temp_options != options)
    free (temp_options);

  program->build_status = CL_BUILD_ERROR;

FINISH:
  POCL_UNLOCK_OBJ (program);

PFN_NOTIFY:
  if (pfn_notify)
    pfn_notify (program, user_data);

  return errcode;
}
