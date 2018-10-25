/* OpenCL runtime library: compile_and_link_program()

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos,
                 2011-2018 Pekka Jääskeläinen

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

#include "pocl_cl.h"
#include <assert.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif
#ifdef OCS_AVAILABLE
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

#define REQUIRES_SPIR_SUPPORT "SPIR support is not available for device "

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
  "-cl-kernel-arg-info "
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

static const char cl_parameters_supported_after_clang_3_9[] =
  "-cl-strict-aliasing " /* deprecated after OCL1.0 */
  "-cl-denorms-are-zero "
  "-cl-no-signed-zeros ";

static const char cl_parameters_not_yet_supported_by_clang[] =
  "-cl-uniform-work-group-size ";

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

#define APPEND_TO_MAIN_BUILD_LOG(...)  \
  POCL_MSG_ERR(__VA_ARGS__);   \
  {                            \
    size_t l = strlen(program->main_build_log); \
    snprintf(program->main_build_log + l, (640 - l), __VA_ARGS__); \
  }

#ifdef OCS_AVAILABLE
/* if 'only_spmd_devices' is set, prebuild the kernel binaries only for the
 * SPMD devices in the context, otherwise build binaries for all devices.
 * The former is useful for clBuildProgram(), the latter for
 * clGetProgramInfo(CL_PROGRAM_BINARIES). */
cl_int
program_compile_dynamic_wg_binaries (cl_program program, int only_spmd_devices)
{
  unsigned i, device_i;
  _cl_command_node cmd;

  assert(program->build_status == CL_BUILD_SUCCESS);
  if (program->num_kernels == 0)
    return CL_SUCCESS;

  memset(&cmd, 0, sizeof(_cl_command_node));
  cmd.type = CL_COMMAND_NDRANGE_KERNEL;

  POCL_LOCK_OBJ (program);

  /* Build the dynamic WG sized parallel.bc and device specific code,
     for each kernel & device combo.  */
  for (device_i = 0; device_i < program->num_devices; ++device_i)
    {
      cl_device_id device = program->devices[device_i];

      /* program may not be built for some of its devices */
      if (program->pocl_binaries[device_i] || (!program->binaries[device_i]))
        continue;

      if (only_spmd_devices && (device->spmd == CL_FALSE))
        continue;

      cmd.device = device;
      cmd.command.run.device_i = device_i;

      struct _cl_kernel fake_k;
      memset (&fake_k, 0, sizeof (fake_k));
      fake_k.context = program->context;
      fake_k.program = program;
      fake_k.next = NULL;
      cl_kernel kernel = &fake_k;

      for (i=0; i < program->num_kernels; i++)
        {
          fake_k.meta = &program->kernel_meta[i];
          fake_k.name = fake_k.meta->name;
          cmd.command.run.hash = fake_k.meta->build_hash[device_i];

          size_t local_x = 0, local_y = 0, local_z = 0;

          if (kernel->meta->reqd_wg_size[0] > 0
              && kernel->meta->reqd_wg_size[1] > 0
              && kernel->meta->reqd_wg_size[2] > 0)
            {
              local_x = kernel->meta->reqd_wg_size[0];
              local_y = kernel->meta->reqd_wg_size[1];
              local_z = kernel->meta->reqd_wg_size[2];
            }

          cmd.command.run.local_x = local_x;
          cmd.command.run.local_y = local_y;
          cmd.command.run.local_z = local_z;

          cmd.command.run.kernel = kernel;

          device->ops->compile_kernel (&cmd, kernel, device);
        }
    }

  POCL_UNLOCK_OBJ (program);

  return CL_SUCCESS;
}

#endif

/* options must be non-NULL.
 * modded_options[size] + link_options are preallocated outputs
 */
static cl_int
process_options (const char *options, char *modded_options, char *link_options,
                 cl_program program, int compiling, int linking,
                 int *create_library, unsigned *flush_denorms,
                 int *requires_correctly_rounded_sqrt_div,
                 int *spir_build, size_t size)
{
  cl_int error;
  char *token = NULL;
  char *saveptr = NULL;

  *create_library = 0;
  *flush_denorms = 0;
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

  size_t i = 1; /* terminating char */
  size_t needed = 0;
  char *temp_options = (char*) malloc (strlen (options) + 1);
  strcpy (temp_options, options);

  token = strtok_r (temp_options, " ", &saveptr);
  while (token != NULL)
    {
      /* check if parameter is supported compiler parameter */
      if (memcmp (token, "-cl", 3) == 0 || memcmp (token, "-w", 2) == 0
          || memcmp (token, "-Werror", 7) == 0)
        {
          if (strstr (cl_program_link_options, token))
            {
              /* when linking, only a subset of -cl* options are valid,
               * and only with -enable-link-options */
              if (linking && (!compiling))
                {
                  if (!enable_link_options)
                    {
                      APPEND_TO_MAIN_BUILD_LOG (
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
            }
          else if (strstr (cl_parameters_supported_after_clang_3_9, token))
            {
#ifndef LLVM_OLDER_THAN_3_9
/* the LLVM API call pushes the parameters directly to the
 * frontend without using -Xclang*/
#else
              APPEND_TO_MAIN_BUILD_LOG (
                  "This build option is supported after clang3.9: %s\n",
                  token);
              token = strtok_r (NULL, " ", &saveptr);
              continue;
#endif
            }
          else if (strstr (cl_parameters_not_yet_supported_by_clang, token))
            {
              APPEND_TO_MAIN_BUILD_LOG (
                  "This build option is not yet supported by clang: %s\n",
                  token);
              token = strtok_r (NULL, " ", &saveptr);
              continue;
            }
          else
            {
              APPEND_TO_MAIN_BUILD_LOG("Invalid build option: %s\n", token);
              error = ret_error;
              goto ERROR;
            }
        }
      else if (memcmp (token, "-g", 2) == 0)
        {
#ifndef LLVM_OLDER_THAN_3_8
          token = "-dwarf-column-info -debug-info-kind=limited " \
	    "-dwarf-version=4 -debugger-tuning=gdb";
#endif
        }
      else if (memcmp (token, "-D", 2) == 0 || memcmp (token, "-I", 2) == 0)
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
      else if (memcmp (token, "-x", 2) == 0 && strlen (token) == 2)
        {
          /* only "-x spir" is valid for the "-x" option */
          token = strtok_r (NULL, " ", &saveptr);
          if (!token || memcmp (token, "spir", 4) != 0)
            {
              APPEND_TO_MAIN_BUILD_LOG (
                  "Invalid parameter to -x build option\n");
              error = ret_error;
              goto ERROR;
            }
          /* "-x spir" is not valid if we are building from source */
          else if (program->source)
            {
              APPEND_TO_MAIN_BUILD_LOG (
                  "\"-x spir\" is not valid when building from source\n");
              error = ret_error;
              goto ERROR;
            }
          else
            *spir_build = 1;
          token = strtok_r (NULL, " ", &saveptr);
          continue;
        }
      else if (memcmp (token, "-spir-std=1.2", 13) == 0)
        {
          /* "-spir-std=" flags are not valid when building from source */
          if (program->source)
            {
              APPEND_TO_MAIN_BUILD_LOG ("\"-spir-std=\" flag is not valid "
                                        "when building from source\n");
              error = ret_error;
              goto ERROR;
            }
          else
            *spir_build = 1;
          token = strtok_r (NULL, " ", &saveptr);
          continue;
        }
      else if (memcmp (token, "-create-library", 15) == 0)
        {
          if (!linking)
            {
              APPEND_TO_MAIN_BUILD_LOG (
                  "\"-create-library\" flag is only valid when linking\n");
              error = ret_error;
              goto ERROR;
            }
          *create_library = 1;
          token = strtok_r (NULL, " ", &saveptr);
          continue;
        }
      else if (memcmp (token, "-enable-link-options", 20) == 0)
        {
          if (!linking)
            {
              APPEND_TO_MAIN_BUILD_LOG ("\"-enable-link-options\" flag is "
                                        "only valid when linking\n");
              error = ret_error;
              goto ERROR;
            }
          if (!(*create_library))
            {
              APPEND_TO_MAIN_BUILD_LOG ("\"-enable-link-options\" flag is "
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
          APPEND_TO_MAIN_BUILD_LOG ("Invalid build option: %s\n", token);
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
ERROR:
  POCL_MEM_FREE (temp_options);
  return error;
}

/*****************************************************************************/

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
          POCL_MEM_FREE (meta->attributes);
          POCL_MEM_FREE (meta->name);
          POCL_MEM_FREE (meta->arg_info);
          for (j = 0; j < program->num_devices; ++j)
            if (meta->data[j] != NULL)
              meta->data[j] = NULL; // TODO free data in driver callback
          POCL_MEM_FREE (meta->data);
          POCL_MEM_FREE (meta->local_sizes);
          POCL_MEM_FREE (meta->build_hash);
        }
      POCL_MEM_FREE (program->kernel_meta);
    }
}

static void
clean_program_on_rebuild (cl_program program)
{
  /* if we're rebuilding the program, release the kernels and reset log/status
   */
  size_t i;
  if ((program->build_status != CL_BUILD_NONE) || program->num_kernels > 0)
    {
      /* Spec says:
         CL_INVALID_OPERATION if there are kernel objects attached to program.
         ...and we check for that earlier.
       */
      assert (program->kernels == NULL);

      free_meta (program);

      program->num_kernels = 0;
      program->build_status = CL_BUILD_NONE;

      for (i = 0; i < program->num_devices; ++i)
        {
          POCL_MEM_FREE (program->build_log[i]);
          memset (program->build_hash[i], 0, sizeof (SHA1_digest_t));
          if (program->source)
            {
              POCL_MEM_FREE (program->binaries[i]);
              program->binary_sizes[i] = 0;
              POCL_MEM_FREE (program->llvm_irs[i]);
            }
        }
      program->main_build_log[0] = 0;
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
  char program_bc_path[POCL_FILENAME_LENGTH];
  char link_options[512];
  int errcode, error;
  int create_library = 0;
  int requires_cr_sqrt_div = 0;
  int spir_build = 0;
  unsigned flush_denorms = 0;
  uint64_t fsize;
  cl_device_id *unique_devlist = NULL;
  char *binary = NULL;
  unsigned device_i = 0, actually_built = 0;
  size_t i, j;
  char *temp_options = NULL;
  const char *extra_build_options =
    pocl_get_string_option ("POCL_EXTRA_BUILD_FLAGS", NULL);
  int build_error_code
      = (link_program ? CL_BUILD_PROGRAM_FAILURE : CL_COMPILE_PROGRAM_FAILURE);

  POCL_GOTO_LABEL_COND (PFN_NOTIFY, (program == NULL), CL_INVALID_PROGRAM);

  POCL_GOTO_LABEL_COND (PFN_NOTIFY, (num_devices > 0 && device_list == NULL),
                        CL_INVALID_VALUE);
  POCL_GOTO_LABEL_COND (PFN_NOTIFY, (num_devices == 0 && device_list != NULL),
                        CL_INVALID_VALUE);

  POCL_GOTO_LABEL_COND (PFN_NOTIFY, (pfn_notify == NULL && user_data != NULL),
                        CL_INVALID_VALUE);

  POCL_GOTO_LABEL_ON (PFN_NOTIFY, program->kernels, CL_INVALID_OPERATION,
                      "Program already has kernels\n");

  POCL_GOTO_LABEL_ON (PFN_NOTIFY,
                      (program->source == NULL && program->binaries == NULL),
                      CL_INVALID_PROGRAM,
                      "Program doesn't have sources or binaries! You need "
                      "to call clCreateProgramWith{Binary|Source} first\n");

  POCL_GOTO_LABEL_ON (PFN_NOTIFY,
                      ((program->source == NULL) && (link_program == 0)),
                      CL_INVALID_OPERATION,
                      "Cannot clCompileProgram when program has no source\n");

  POCL_LOCK_OBJ (program);

  program->main_build_log[0] = 0;

  /* TODO this should be somehow utilized at linking */
  POCL_MEM_FREE (program->compiler_options);

  if (extra_build_options)
    {
      temp_options =
	(char*) malloc (options != NULL ? strlen (options) : 0
			+ strlen (extra_build_options) + 2);
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
      errcode = process_options (temp_options, program->compiler_options,
                                 link_options, program, compile_program,
                                 link_program, &create_library, &flush_denorms,
                                 &requires_cr_sqrt_div, &spir_build, size);
      if (errcode != CL_SUCCESS)
        goto ERROR_CLEAN_OPTIONS;
    }

  POCL_MSG_PRINT_LLVM ("building program with options %s\n",
                       program->compiler_options);


  program->flush_denorms = flush_denorms;
#if !(defined(__x86_64__) && defined(__GNUC__))
  if (flush_denorms)
    {
      POCL_MSG_WARN ("flush to zero is currently only implemented for "
                     "x86-64 & gcc/clang, ignoring flag\n");
    }
#endif

  /* DEVICE LIST */
  if (num_devices == 0)
    {
      num_devices = program->num_devices;
      device_list = program->devices;
    }
  else
    {
      // convert subdevices to devices and remove duplicates
      cl_uint real_num_devices = 0;
      unique_devlist = pocl_unique_device_list (device_list, num_devices,
                                                &real_num_devices);
      num_devices = real_num_devices;
      device_list = unique_devlist;
    }

  clean_program_on_rebuild (program);

  /* Build the fully linked non-parallel bitcode for all
         devices. */
  for (device_i = 0; device_i < program->num_devices; ++device_i)
    {
      cl_device_id device = program->devices[device_i];

      /* find the device in the supplied devices-to-build-for list */
      int found = 0;
      for (i = 0; i < num_devices; ++i)
          if (device_list[i] == device) found = 1;
      if (!found) continue;

      if (requires_cr_sqrt_div
          && !(device->single_fp_config & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT))
        {
          APPEND_TO_MAIN_BUILD_LOG (REQUIRES_CR_SQRT_DIV_ERR);
          POCL_GOTO_ERROR_ON (1, build_error_code,
                              REQUIRES_CR_SQRT_DIV_ERR " %s\n",
                              device->short_name);
        }
      actually_built++;

      /* clCreateProgramWithSource */
      if (program->source)
        {
          POCL_MSG_PRINT_INFO("building from sources for device %d\n", device_i);
#ifdef OCS_AVAILABLE
          error = pocl_llvm_build_program(
              program, device_i, program->compiler_options, program_bc_path,
              num_input_headers, input_headers, header_include_names,
              (create_library ? 0 : link_program));
          POCL_GOTO_ERROR_ON ((error != 0), build_error_code,
                              "pocl_llvm_build_program() failed\n");
#else
          APPEND_TO_MAIN_BUILD_LOG (
              "Cannot build a program from sources with pocl "
              "that does not have online compiler support\n");
          POCL_GOTO_ERROR_ON(1, CL_COMPILER_NOT_AVAILABLE,
                             "%s", program->main_build_log);
#endif
        }
      /* clCreateProgramWithBinaries */
      else if (program->binaries[device_i]
               && (program->pocl_binaries[device_i] == NULL))
        {
#ifdef OCS_AVAILABLE
          /* bitcode is now either plain LLVM IR or SPIR IR */
          int spir_binary = bitcode_is_spir ((char*)program->binaries[device_i],
                                             program->binary_sizes[device_i]);
          if (spir_binary)
            POCL_MSG_PRINT_LLVM ("LLVM-SPIR binary detected\n");
          else
            POCL_MSG_PRINT_LLVM ("building from a BC binary for device %d\n",
                                 device_i);

          if (spir_binary)
            {
#ifdef ENABLE_SPIR
              if (!strstr (device->extensions, "cl_khr_spir"))
                {
                  APPEND_TO_MAIN_BUILD_LOG (REQUIRES_SPIR_SUPPORT);
                  POCL_GOTO_ERROR_ON (1, build_error_code,
                                      REQUIRES_SPIR_SUPPORT " %s\n",
                                      device->short_name);
                }
              if (!spir_build)
                POCL_MSG_WARN (
                    "SPIR binary provided, but no spir in build options\n");
              /* SPIR binaries need to be explicitly linked to the kernel
               * library. for non-SPIR binaries this happens as part of build
               * process when program.bc is generated. */
              error
                  = pocl_llvm_link_program (program, device_i, program_bc_path,
                                            0, NULL, NULL, NULL, 0, 1);

              POCL_GOTO_ERROR_ON (error, CL_LINK_PROGRAM_FAILURE,
                                  "Failed to link SPIR program.bc\n");
#else
              APPEND_TO_MAIN_BUILD_LOG (REQUIRES_SPIR_SUPPORT);
              POCL_GOTO_ERROR_ON (1, build_error_code,
                                  REQUIRES_SPIR_SUPPORT " %s\n",
                                  device->short_name);
#endif
            }

#else
          APPEND_TO_MAIN_BUILD_LOG (
              "Cannot build program from LLVM IR binaries with "
              "pocl that does not have online compiler support\n");
          POCL_GOTO_ERROR_ON (1, CL_COMPILER_NOT_AVAILABLE, "%s",
                              program->main_build_log);
#endif
        }
      else if (program->pocl_binaries[device_i])
        {
          POCL_MSG_PRINT_INFO("having a poclbinary for device %d\n", device_i);
#ifdef OCS_AVAILABLE
          if (program->binaries[device_i] == NULL)
            {
              POCL_MSG_WARN (
                  "pocl-binary for this device doesn't contain "
                  "program.bc - you won't be able to rebuild/link it\n");
              /* do not try to read program.bc or LLVM IRs
               * TODO maybe read LLVM IRs ?*/
              continue;
            }
#else
          continue;
#endif
        }
      else if (link_program && (num_input_programs > 0))
        {
#ifdef OCS_AVAILABLE
          /* just link binaries. */
          unsigned char *cur_device_binaries[num_input_programs];
          size_t cur_device_binary_sizes[num_input_programs];
          void *cur_llvm_irs[num_input_programs];
          for (j = 0; j < num_input_programs; j++)
            {
              assert (device == input_programs[j]->devices[device_i]);
              cur_device_binaries[j] = input_programs[j]->binaries[device_i];

              assert (cur_device_binaries[j]);
              cur_device_binary_sizes[j]
                  = input_programs[j]->binary_sizes[device_i];

              if (input_programs[j]->llvm_irs[device_i] == NULL)
                pocl_update_program_llvm_irs (input_programs[j], device_i);

              cur_llvm_irs[j] = input_programs[j]->llvm_irs[device_i];
              assert (cur_llvm_irs[j]);
            }
          error = pocl_llvm_link_program (
              program, device_i, program_bc_path, num_input_programs,
              cur_device_binaries, cur_device_binary_sizes, cur_llvm_irs,
              create_library, 0);
          POCL_GOTO_ERROR_ON ((error != CL_SUCCESS), CL_LINK_PROGRAM_FAILURE,
                              "pocl_llvm_link_program() failed\n");
#else
          POCL_GOTO_ERROR_ON ((1), CL_LINK_PROGRAM_FAILURE,
                              "clCompileProgram/clLinkProgram/clBuildProgram"
                              " require a pocl built with LLVM\n");

#endif
        }
      else
        {
          POCL_GOTO_ERROR_ON (1, CL_INVALID_BINARY,
                              "No sources nor binaries for device %s - can't "
                              "build the program\n", device->short_name);
        }

#ifdef OCS_AVAILABLE
      /* Read binaries from program.bc to memory */
      if (program->binaries[device_i] == NULL)
        {
          errcode = pocl_read_file(program_bc_path, &binary, &fsize);
          POCL_GOTO_ERROR_ON(errcode, CL_BUILD_ERROR,
                             "Failed to read binaries from program.bc to "
                             "memory: %s\n", program_bc_path);

          program->binary_sizes[device_i] = (size_t)fsize;
          program->binaries[device_i] = (unsigned char *)binary;
        }

      if (program->llvm_irs[device_i] == NULL)
        {
          pocl_update_program_llvm_irs(program, device_i);
        }
      /* Maintain a 'last_accessed' file in every program's
       * cache directory. Will be useful for cache pruning script
       * that flushes old directories based on LRU */
      pocl_cache_update_program_last_access(program, device_i);
#endif

    }

  POCL_GOTO_ERROR_ON ((actually_built < num_devices), build_error_code,
                      "Some of the devices on the argument-supplied list are"
                      "not available for the program, or do not exist\n");

  program->build_status = CL_BUILD_SUCCESS;
  program->binary_type = CL_PROGRAM_BINARY_TYPE_EXECUTABLE;
  /* if program will be compiled using clCompileProgram its binary_type
   * will be set to CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT.
   *
   * if program was created by clLinkProgram which is called
   * with the –createlibrary link option its binary_type will be set to
   * CL_PROGRAM_BINARY_TYPE_LIBRARY.
   */
  if (create_library)
    program->binary_type = CL_PROGRAM_BINARY_TYPE_LIBRARY;
  if (compile_program && !link_program)
    program->binary_type = CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT;

  assert(program->num_kernels == 0);

  /* get non-device-specific kernel metadata. We can stop after finding
   * the first method that works.*/
  for (device_i = 0; device_i < program->num_devices; device_i++)
    {
#ifdef OCS_AVAILABLE
      if (program->binaries[device_i])
        {
          program->num_kernels
              = pocl_llvm_get_kernel_count (program, device_i);
          if (program->num_kernels)
            {
              program->kernel_meta = calloc (program->num_kernels,
                                             sizeof (pocl_kernel_metadata_t));
              pocl_llvm_get_kernels_metadata (program, device_i);
            }
          break;
        }
#endif
      if (program->pocl_binaries[device_i])
        {
          program->num_kernels
              = pocl_binary_get_kernel_count (program, device_i);
          if (program->num_kernels)
            {
              program->kernel_meta = calloc (program->num_kernels,
                                             sizeof (pocl_kernel_metadata_t));
              pocl_binary_get_kernels_metadata (program, device_i);
            }
          break;
        }
    }

  POCL_GOTO_ERROR_ON ((device_i >= program->num_devices), CL_INVALID_BINARY,
                      "Could find kernel metadata in the built program\n");

  /* calculate device-specific kernel hashes. */
  for (j = 0; j < program->num_kernels; ++j)
    {
      program->kernel_meta[j].build_hash
          = calloc (program->num_devices, sizeof (pocl_kernel_hash_t));

      for (device_i = 0; device_i < program->num_devices; device_i++)
        {
          pocl_calculate_kernel_hash (program, j, device_i);
        }
    }

#ifdef OCS_AVAILABLE
  /* for SPMD devices, prebuild the kernel binaries here. */
  if (program->binary_type == CL_PROGRAM_BINARY_TYPE_EXECUTABLE) {
    POCL_UNLOCK_OBJ (program);
    program_compile_dynamic_wg_binaries (program, 1);
    POCL_LOCK_OBJ (program);
  }
#endif

  errcode = CL_SUCCESS;
  goto FINISH;

ERROR:
  free_meta (program);

  program->kernels = NULL;

  for (device_i = 0; device_i < program->num_devices; device_i++)
    {
      if (program->source)
        {
          POCL_MEM_FREE (program->binaries[device_i]);
          program->binary_sizes[device_i] = 0;
        }
    }

ERROR_CLEAN_OPTIONS:
  if (temp_options != options)
    free (temp_options);

  program->build_status = CL_BUILD_ERROR;

FINISH:
  POCL_UNLOCK_OBJ (program);
  POCL_MEM_FREE (unique_devlist);

PFN_NOTIFY:
  if (pfn_notify)
    pfn_notify (program, user_data);

  return errcode;
}
