
#ifndef POCL_ALMAIFCOMPILE_H
#define POCL_ALMAIFCOMPILE_H

#include "bufalloc.h"
#include "pocl_util.h"
//#include "accel-shared.h"

// TODO copied from TCE
/* The address space ids in the ADFs. */
#define TTA_ASID_PRIVATE 0
#define TTA_ASID_GLOBAL 1
#define TTA_ASID_LOCAL 3
#define TTA_ASID_CONSTANT 2

typedef struct compilation_data_s
{
  /* used in pocl_llvm_build_program */
  const char *llvm_triplet;
  const char *llvm_cpu;
  /* does device support 64bit integers */
  int has_64bit_long;
  /* see comment below on scratchpad_size */
  int has_scratchpad_mem;

  /* Currently loaded kernel. */
  cl_kernel current_kernel;

  char *build_hash;

  chunk_info_t *pocl_context;

  /* device-specific callbacks */
  int (*compile_kernel) (_cl_command_node *cmd, cl_kernel kernel,
                         cl_device_id device, int specialize);
  int (*initialize_device) (cl_device_id device, const char *parameters);
  int (*cleanup_device) (cl_device_id device);

  /* backend-specific data */
  void *backend_data;
};

typedef struct almaif_kernel_data_s
{
  /* Binaries of kernel */
  char *dmem_img;
  char *pmem_img;
  char *imem_img;
  size_t dmem_img_size;
  size_t imem_img_size;
  size_t pmem_img_size;
  uint32_t kernel_address;
  uint32_t kernel_md_offset;
} almaif_kernel_data_t;

int pocl_almaif_init (unsigned j, cl_device_id dev, const char *parameters);
cl_int pocl_almaif_uninit (unsigned j, cl_device_id dev);

extern "C"
{
  void pocl_almaif_compile_kernel (_cl_command_node *cmd, cl_kernel kernel,
                                   cl_device_id device, int specialize);
  int pocl_almaif_create_kernel (cl_device_id device, cl_program,
                                 cl_kernel kernel, unsigned device_i);
  int pocl_almaif_free_kernel (cl_device_id device, cl_program program,
                               cl_kernel kernel, unsigned device_i);
}

void preread_images (const char *kernel_cachedir, void *d_void,
                     almaif_kernel_data_t *kd);
char *pocl_almaif_build_hash (cl_device_id device);

#endif
