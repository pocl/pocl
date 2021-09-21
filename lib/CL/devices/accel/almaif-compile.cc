

#include "pocl_cache.h"
#include "pocl_file_util.h"

#include "accel-shared.h"
#include "accel.h"
#include "almaif-compile.h"

// TODO TCE SPECIFIC
#include "almaif-compile-tce.h"

int pocl_almaif_init(unsigned j, cl_device_id dev, const char *parameters) {
  AccelData *d = (AccelData *)dev->data;
  unsigned device_number = j;

  d->compilationData = (compilation_data_t *)pocl_aligned_malloc(
      HOST_CPU_CACHELINE_SIZE, sizeof(compilation_data_t));

  if (d->compilationData == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  d->compilationData->pocl_context =
      alloc_buffer(&d->AllocRegion, sizeof(pocl_context32));
  assert(d->compilationData->pocl_context &&
         "Failed to allocate pocl context on device\n");

  /**********************************************************/

  /* setup device info */
  dev->image_support = CL_FALSE;
  dev->half_fp_config = CL_FP_ROUND_TO_ZERO;
  dev->single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN;
  dev->double_fp_config = 0;
  dev->profiling_timer_resolution = 1000;
  dev->profile = "EMBEDDED_PROFILE";
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

  // TODO not sure about these 2
  dev->max_constant_buffer_size = 32768;
  dev->local_mem_size = 16384;

  dev->max_work_item_dimensions = 3;
  dev->final_linkage_flags = NULL;

  // kernel param size. this is a bit arbitrary
  dev->max_parameter_size = 64;
  dev->address_bits = 32;
  dev->mem_base_addr_align = 16;
  d->compilationData->current_kernel = NULL;

  dev->max_work_item_sizes[0] = dev->max_work_item_sizes[1] =
      dev->max_work_item_sizes[2] = dev->max_work_group_size = 64;

  dev->preferred_wg_size_multiple = 8;

  SETUP_DEVICE_CL_VERSION(1, 2);

  dev->available = CL_TRUE;
  // dev->available = pocl_offline_compile ? CL_FALSE : CL_TRUE;

  dev->compiler_available = true;
  dev->linker_available = true;

  POCL_MSG_PRINT_INFO("ASDF\n");
  compilation_data_t *adi = (compilation_data_t *)d->compilationData;

  // TODO tce specific
  adi->initialize_device = pocl_almaif_tce_initialize;
  adi->cleanup_device = pocl_almaif_tce_cleanup;
  adi->compile_kernel = pocl_almaif_tce_compile;

  // backend specific init
  POCL_MSG_PRINT_INFO("Starting device specific initializion\n");
  adi->initialize_device(dev, parameters);

  POCL_MSG_PRINT_INFO("Device specific initializion done\n");

  if (1) {
    SHA1_digest_t digest;
    pocl_almaif_tce_device_hash(parameters, dev->llvm_target_triplet,
                                (char *)digest);
    adi->build_hash = strdup((char *)digest);
  } else {
    char option_str[256];
    snprintf(option_str, 256, "POCL_ALMAIF%u_HASH", j);
    if (pocl_is_option_set(option_str)) {
      adi->build_hash = (char *)pocl_get_string_option(option_str, NULL);
      assert(adi->build_hash);
    } else
      adi->build_hash = DEFAULT_BUILD_HASH;
  }

  /*
    // must be run AFTER initialize, since it changes little_endian
  #if defined(WORDS_BIGENDIAN) && WORDS_BIGENDIAN == 1
    d->requires_bswap = dev->endian_little;
  #else
    d->requires_bswap = !dev->endian_little;
  #endif

  //  POCL_MSG_PRINT_INFO ("LITTLE_ENDIAN: %u, requires BSWAP: %u
  \n",dev->endian_little,  d->requires_bswap);
  */
  return CL_SUCCESS;
}

cl_int pocl_almaif_uninit(unsigned j, cl_device_id dev) {
  AccelData *d = (AccelData *)dev->data;

  d->compilationData->cleanup_device(dev);

  free_chunk(d->compilationData->pocl_context);
  pocl_aligned_free(d->compilationData);

  return CL_SUCCESS;
}

void pocl_almaif_compile_kernel(_cl_command_node *cmd, cl_kernel kernel,
                                cl_device_id device, int specialize) {
  if (cmd->type != CL_COMMAND_NDRANGE_KERNEL)
    return;

  if (kernel == NULL)
    kernel = cmd->command.run.kernel;
  cl_program program = kernel->program;
  cl_device_id dev = (device ? device : cmd->device);
  AccelData *d = (AccelData *)dev->data;
  unsigned dev_i = cmd->device_i;

  POCL_MSG_PRINT_INFO("Current kernel %p, new kernel %p\n",
                      d->compilationData->current_kernel, kernel);

  if (d->compilationData->current_kernel == kernel) {
    POCL_MSG_PRINT_INFO(
        "kernel %s is the currently loaded kernel, nothing to do\n",
        kernel->name);
    return;
  }

  if (!program->pocl_binaries[dev_i]) {
    /*      if (!pocl_offline_compile)
            POCL_ABORT ("Compiler not available for this device.\n");
    */
    POCL_MSG_PRINT_INFO("Compiling kernel to poclbinary\n");

    if (d->compilationData->compile_kernel(cmd, kernel, device, specialize))
      POCL_ABORT("Kernel compilation failed\n");
  }

  //  if (pocl_offline_compile)
  //    return;

  almaif_kernel_data_t *kd =
      (almaif_kernel_data_t *)kernel->data[cmd->device_i];
  assert(kd);

  if (kd->imem_img_size == 0) {
    char img_file[POCL_FILENAME_LENGTH];
    char cachedir[POCL_FILENAME_LENGTH];
    // first try specialized
    pocl_cache_kernel_cachedir_path(img_file, kernel->program, cmd->device_i,
                                    kernel, "/parallel.img", cmd, 1);
    if (pocl_exists(img_file)) {
      pocl_cache_kernel_cachedir_path(cachedir, kernel->program, cmd->device_i,
                                      kernel, "", cmd, 1);
      preread_images(cachedir, d, kd);
    } else {
      // if it doesn't exist, try specialized with local sizes 0-0-0
      // should pick either 0-0-0 or 0-0-0-goffs0
      _cl_command_node cmd_copy;
      memcpy(&cmd_copy, cmd, sizeof(_cl_command_node));
      cmd_copy.command.run.pc.local_size[0] = 0;
      cmd_copy.command.run.pc.local_size[1] = 0;
      cmd_copy.command.run.pc.local_size[2] = 0;
      pocl_cache_kernel_cachedir_path(cachedir, kernel->program, cmd->device_i,
                                      kernel, "", &cmd_copy, 1);
      POCL_MSG_PRINT_INFO("Specialized kernel not found, using %s\n", cachedir);
      preread_images(cachedir, d, kd);
    }
  }

  assert(kd->imem_img_size > 0);

  d->ControlMemory->Write32(ACCEL_CONTROL_REG_COMMAND, ACCEL_RESET_CMD);

  d->InstructionMemory->CopyToMMAP(d->InstructionMemory->PhysAddress,
                                   kd->imem_img, kd->imem_img_size);
  POCL_MSG_PRINT_INFO("IMEM image written: %p / %zu B\n",
                      d->InstructionMemory->PhysAddress, kd->imem_img_size);

  d->ControlMemory->Write32(ACCEL_CONTROL_REG_COMMAND, ACCEL_CONTINUE_CMD);
  /*
  if (kd->dmem_img_size > 0)
    {
      POCL_MSG_PRINT_INFO ("Scratchpad image written: %p / %zu B\n",
  d->ScratchpadMemory->PhysAddress, kd->dmem_img_size);
    }

  if (kd->pmem_img_size > 0)
    {
      d->DataMemory->CopyToMMAP(d->DataMemory->PhysAddress, kd->pmem_img,
  kd->pmem_img_size); POCL_MSG_PRINT_INFO ("Data image written: %p / %zu B\n",
  d->DataMemory->PhysAddress, kd->pmem_img_size);
    }
*/
  d->compilationData->current_kernel = kernel;
}

void preread_images(const char *kernel_cachedir, void *d_void,
                    almaif_kernel_data_t *kd) {
  AccelData *d = (AccelData *)d_void;
  POCL_MSG_PRINT_INFO("Reading image files\n");
  uint64_t temp = 0;
  size_t size = 0;
  char *content = NULL;

  char module_fn[POCL_FILENAME_LENGTH];
  snprintf(module_fn, POCL_FILENAME_LENGTH, "%s/parallel.img", kernel_cachedir);

  if (pocl_exists(module_fn)) {
    int res = pocl_read_file(module_fn, &content, &temp);
    size = (size_t)temp;
    assert(res == 0);
    assert(size > 0);
    assert(size < d->InstructionMemory->Size);
    kd->imem_img = content;
    kd->imem_img_size = size;
    content = NULL;
  } else
    POCL_ABORT("ALMAIF: %s for this kernel does not exist.\n", module_fn);

  /* dmem/pmem images which contains also struct kernel_metadata;
   * should be already byteswapped for the device */
  snprintf(module_fn, POCL_FILENAME_LENGTH, "%s/parallel_data.img",
           kernel_cachedir);
  if (pocl_exists(module_fn)) {
    int res = pocl_read_file(module_fn, &content, &temp);
    assert(res == 0);
    size = (size_t)temp;
    if (size == 0)
      POCL_MEM_FREE(content);
    kd->dmem_img = content;
    kd->dmem_img_size = size;

    /* There seems to be a bug in TCE, where the actual kernel address is
        not byteswappped. This is a workaround. */
    uint32_t kernel_addr = 0;
    if (size) {
      void *p = content + size - 4;
      uint32_t *up = (uint32_t *)p;
      kernel_addr = *up;
      if (kernel_addr > d->InstructionMemory->Size) {
        POCL_MSG_PRINT_INFO(
            "Incorrect kernel address (%0x) detected, byteswapping\n",
            kernel_addr);
        *up = byteswap_uint32_t((*up), 1);
        kernel_addr = *up;
      }
    }
    POCL_MSG_PRINT_INFO("Kernel address (%0x) found\n", kernel_addr);
    kd->kernel_address = kernel_addr;
    content = NULL;
  } else
    POCL_ABORT("ALMAIF: %s for this kernel does not exist.\n", module_fn);

  snprintf(module_fn, POCL_FILENAME_LENGTH, "%s/parallel_param.img",
           kernel_cachedir);
  if (pocl_exists(module_fn)) {
    int res = pocl_read_file(module_fn, &content, &temp);
    assert(res == 0);
    size = (size_t)temp;
    if (size == 0)
      POCL_MEM_FREE(content);
    kd->pmem_img = content;
    kd->pmem_img_size = size;
    content = NULL;
  } else
    POCL_ABORT("ALMAIF: %s for this kernel does not exist.\n", module_fn);
  /*
    snprintf(module_fn, POCL_FILENAME_LENGTH,
             "%s/kernel_metadata.txt", kernel_cachedir);
    if (pocl_exists(module_fn))
      {
        int res = pocl_read_file(module_fn, &content, &temp);
        assert(res == 0);
        size = (size_t)temp;
        assert(size > 0);

        uint32_t metadata_offset = 0;
        sscanf(content, "kernel_md address = %d", &metadata_offset);
        assert(metadata_offset != 0);
        kd->kernel_md_offset = metadata_offset;
        content = NULL;
      }
    else
      POCL_ABORT("ALMAIF: %s for this kernel does not exist.\n", module_fn);
  */
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

char *pocl_almaif_build_hash(cl_device_id device) {
  AccelData *d = (AccelData *)device->data;
  assert(d->compilationData->build_hash);
  return d->compilationData->build_hash;
}
