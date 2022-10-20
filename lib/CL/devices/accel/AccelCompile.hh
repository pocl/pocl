/* AccelCompile.hh - compiler support for custom devices

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

#ifndef POCL_ALMAIFCOMPILE_H
#define POCL_ALMAIFCOMPILE_H

#include "AccelShared.hh"
#include "bufalloc.h"
#include "pocl_util.h"

typedef struct compilation_data_s {
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
  void (*compile_kernel)(_cl_command_node *cmd, cl_kernel kernel,
                         cl_device_id device, int specialize);
  int (*initialize_device)(cl_device_id device, const char *parameters);
  int (*cleanup_device)(cl_device_id device);

  /* backend-specific data */
  void *backend_data;
} compilation_data_t;

typedef struct almaif_kernel_data_s {
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

int pocl_almaif_init(unsigned j, cl_device_id dev, const char *parameters);
cl_int pocl_almaif_uninit(unsigned j, cl_device_id dev);

extern "C" {
void pocl_almaif_compile_kernel(_cl_command_node *cmd, cl_kernel kernel,
                                cl_device_id device, int specialize);
int pocl_almaif_create_kernel(cl_device_id device, cl_program, cl_kernel kernel,
                              unsigned device_i);
int pocl_almaif_free_kernel(cl_device_id device, cl_program program,
                            cl_kernel kernel, unsigned device_i);
}

int pocl_almaif_build_binary(cl_program program, cl_uint device_i,
                             int link_program, int spir_build);

void preread_images(const char *kernel_cachedir, void *d_void,
                    almaif_kernel_data_t *kd);
char *pocl_almaif_build_hash(cl_device_id device);

#endif
