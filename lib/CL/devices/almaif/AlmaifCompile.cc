/* AlmaifCompile.cc - compiler support for custom devices

   Copyright (c) 2022 Topi Leppänen / Tampere University

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


#include "pocl_cache.h"
#include "pocl_file_util.h"

#include "AlmaifShared.hh"
#include "almaif.h"
#include "AlmaifCompile.hh"

#include "common_driver.h"

// TODO TCE SPECIFIC
#if defined(TCEMC_AVAILABLE) || defined(TCE_AVAILABLE)
#define ENABLE_COMPILER
#endif

#ifdef ENABLE_COMPILER
#include "AlmaifCompileTCE.hh"
#endif

extern int pocl_offline_compile;

int pocl_almaif_compile_init(unsigned j, cl_device_id dev, const char *parameters) {
  AlmaifData *d = (AlmaifData *)dev->data;

  d->compilationData = (compilation_data_t *)pocl_aligned_malloc(
      HOST_CPU_CACHELINE_SIZE, sizeof(compilation_data_t));

  if (d->compilationData == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  if (!pocl_offline_compile) {
    d->compilationData->pocl_context =
        pocl_alloc_buffer(d->Dev->AllocRegions, sizeof(pocl_context32));
    assert(d->compilationData->pocl_context &&
           "Failed to allocate pocl context on device\n");
  }

  /**********************************************************/

  /* setup device info */
  dev->image_support = CL_FALSE;
  dev->half_fp_config = CL_FP_ROUND_TO_ZERO;
  dev->single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN;
  dev->double_fp_config = 0;
  dev->llvm_target_triplet = NULL;
  dev->llvm_cpu = NULL;
  dev->has_64bit_long = 0;
  dev->spmd = CL_FALSE;
  dev->workgroup_pass = CL_TRUE;

  dev->arg_buffer_launcher = CL_TRUE;
  dev->autolocals_to_args = POCL_AUTOLOCALS_TO_ARGS_ALWAYS;

  dev->global_as_id = TTA_ASID_GLOBAL;
  dev->local_as_id = TTA_ASID_LOCAL;
  dev->constant_as_id = TTA_ASID_CONSTANT;
  dev->args_as_id = TTA_ASID_GLOBAL;
  dev->context_as_id = TTA_ASID_GLOBAL;

  dev->final_linkage_flags = NULL;

  d->compilationData->current_kernel = NULL;
  SETUP_DEVICE_CL_VERSION(1, 2);

  // dev->available = CL_TRUE;
  dev->available = pocl_offline_compile ? CL_FALSE : CL_TRUE;

  dev->compiler_available = true;
  dev->linker_available = true;

  compilation_data_t *adi = (compilation_data_t *)d->compilationData;

#ifdef ENABLE_COMPILER
  // TODO tce specific
  adi->initialize_device = pocl_almaif_tce_initialize;
  adi->cleanup_device = pocl_almaif_tce_cleanup;
  adi->compile_kernel = pocl_almaif_tce_compile;

  // backend specific init
  POCL_MSG_PRINT_ALMAIF("Starting device specific initializion\n");
  adi->initialize_device(dev, parameters);

  POCL_MSG_PRINT_ALMAIF("Device specific initializion done\n");

  SHA1_digest_t digest;
  pocl_almaif_tce_device_hash(parameters, dev->llvm_target_triplet,
                              (char *)digest);
  POCL_MSG_PRINT_ALMAIF("ALMAIF TCE DEVICE HASH=%s", (char *)digest);
  adi->build_hash = strdup((char *)digest);

#else
  char option_str[256];
  snprintf(option_str, 256, "POCL_ALMAIF%u_HASH", j);
  if (pocl_is_option_set(option_str)) {
    adi->build_hash = (char *)pocl_get_string_option(option_str, NULL);
    assert(adi->build_hash);
  } else
    adi->build_hash = strdup(DEFAULT_BUILD_HASH);
#endif

  dev->ops->build_hash = pocl_almaif_compile_build_hash;
  dev->ops->build_source = pocl_driver_build_source;
  dev->ops->setup_metadata = pocl_driver_setup_metadata;
  dev->ops->create_kernel = pocl_almaif_create_kernel;
  dev->ops->free_kernel = pocl_almaif_free_kernel;
  dev->ops->build_poclbinary = pocl_driver_build_poclbinary;
  dev->ops->build_binary = pocl_almaif_build_binary;
#ifdef ENABLE_COMPILER
  dev->ops->compile_kernel = pocl_almaif_tce_compile;
  dev->ops->init_build = pocl_tce_init_build;
#endif
  return CL_SUCCESS;
}

cl_int pocl_almaif_compile_uninit(unsigned j, cl_device_id dev) {
  AlmaifData *d = (AlmaifData *)dev->data;

#ifdef ENABLE_COMPILER
  d->compilationData->cleanup_device(dev);
#endif

  pocl_free_chunk(d->compilationData->pocl_context);
  pocl_aligned_free(d->compilationData);

  return CL_SUCCESS;
}

void pocl_almaif_compile_kernel(_cl_command_node *cmd, cl_kernel kernel,
                                cl_device_id device, int specialize) {
  if (cmd->type != CL_COMMAND_NDRANGE_KERNEL) {
    return;
  }

  if (kernel == NULL)
    kernel = cmd->command.run.kernel;
  cl_program program = kernel->program;
  cl_device_id dev = (device ? device : cmd->device);
  AlmaifData *d = (AlmaifData *)dev->data;
  unsigned dev_i = cmd->program_device_i;

  POCL_MSG_PRINT_ALMAIF("Current kernel %p, new kernel %p\n",
                       (void*)d->compilationData->current_kernel, (void*)kernel);

  /* if (d->compilationData->current_kernel == kernel) {
     POCL_MSG_PRINT_ALMAIF(
         "kernel %s is the currently loaded kernel, nothing to do\n",
         kernel->name);
     return;
   }*/

#ifdef ENABLE_COMPILER
  if (!program->pocl_binaries[dev_i]) {
    POCL_MSG_PRINT_ALMAIF("Compiling kernel %s to poclbinary\n", kernel->name);

    d->compilationData->compile_kernel(cmd, kernel, device, specialize);
  }
#endif

  if (pocl_offline_compile) {
    return;
  }
  almaif_kernel_data_t *kd =
      (almaif_kernel_data_t *)kernel->data[cmd->program_device_i];

  POCL_MSG_PRINT_ALMAIF("Loading program to device\n");
  d->Dev->loadProgramToDevice(kd, kernel, cmd);

  POCL_MSG_PRINT_ALMAIF("Loaded program to device\n");
  d->compilationData->current_kernel = kernel;
}

int pocl_almaif_create_kernel(cl_device_id device, cl_program program,
                              cl_kernel kernel, unsigned device_i) {
  assert(kernel->data != NULL);
  assert(kernel->data[device_i] == NULL);

  kernel->data[device_i] = (void *)calloc(1, sizeof(almaif_kernel_data_t));

  return CL_SUCCESS;
}

int pocl_almaif_free_kernel(cl_device_id device, cl_program program,
                            cl_kernel kernel, unsigned device_i) {
  assert(kernel->data != NULL);
  // may happen if creating kernel fails
  if (kernel->data[device_i] == NULL)
    return CL_SUCCESS;

  almaif_kernel_data_t *p = (almaif_kernel_data_t *)kernel->data[device_i];
  POCL_MEM_FREE(p->dmem_img);
  POCL_MEM_FREE(p->imem_img);
  POCL_MEM_FREE(p->pmem_img);
  POCL_MEM_FREE(p);
  kernel->data[device_i] = NULL;

  return CL_SUCCESS;
}

int pocl_almaif_build_binary(cl_program program, cl_uint device_i,
                             int link_program, int spir_build) {
  assert(program->pocl_binaries[device_i] != NULL);
  assert(program->pocl_binary_sizes[device_i] > 0);
  assert(link_program != 0);
  assert(spir_build == 0);
  return CL_SUCCESS;
}

char *pocl_almaif_compile_build_hash(cl_device_id device) {
  AlmaifData *d = (AlmaifData *)device->data;
  assert(d->compilationData->build_hash);
  return strdup(d->compilationData->build_hash);
}
