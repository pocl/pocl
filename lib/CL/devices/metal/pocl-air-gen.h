/* pocl-air-gen.h - lowering of PoCL's LLVM IR to Apple's AIR, and packaging
   of the result as a Metal library

   Copyright (c) 2026 PoCL developers

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

#ifndef POCL_AIR_GEN_H
#define POCL_AIR_GEN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The AIR / Metal versions and limits to generate code for. */
typedef struct
{
  unsigned air_major, air_minor;
  unsigned metal_major, metal_minor;
  /* the macOS version used for the triple and the metallib header */
  unsigned macos_major, macos_minor;
  /* threadgroup memory available to a threadgroup, in bytes */
  unsigned local_mem_size;
} pocl_metal_air_target_t;

/* Kernel calling convention produced by the lowering, which the host side of
   the driver has to follow when binding arguments (all indices are Metal
   buffer indices):

   - kernel argument i is bound at buffer index i:
     - __global / __constant pointers as buffers (with the sub-buffer offset);
     - plain-old-data arguments (scalars, vectors, structs) as bytes;
     - __local pointers as a 32-bit byte offset (as bytes) into a static
       threadgroup region of *local_region_size bytes; each offset
       must be aligned to POCL_METAL_LOCAL_ARG_ALIGN.
   - buffer index num_args holds the launch parameters, a
     pocl_metal_launch_params_t passed as bytes.
   - buffer index num_args + 1 holds the printf buffer: a uint32 write
     position (zeroed before the launch), a uint32 data capacity, followed by
     the data, which the host decodes with pocl_write_printf_buffer().
   - buffer index num_args + 2 holds the program-scope __global variables,
     a per-program buffer of *gvar_size bytes initialized from *gvar_init. */
#define POCL_METAL_PRINTF_HEADER_SIZE 8
#define POCL_METAL_MAX_KERNEL_ARGS 28
#define POCL_METAL_LOCAL_ARG_ALIGN 16

typedef struct
{
  unsigned offset;
  unsigned target;
} pocl_metal_gvar_reloc_t;

typedef struct
{
  unsigned work_dim;
  unsigned global_offset[3];
  /* bit i set: buffer argument i is a NULL pointer (Metal doesn't guarantee
     that an unbound buffer reads as a null pointer) */
  unsigned null_args;
} pocl_metal_launch_params_t;

/* Lowers the (linked, PoCL-pass-processed) LLVM module of program/device_i
   to AIR and returns a Metal library containing all of its kernels in
   *metallib (malloc'd; the caller frees it).

   *local_region_size is set to the size of the static threadgroup region
   that backs __local pointer kernel arguments (0 if no kernel has any).

   *gvar_size is set to the size of the program-scope variable storage, and
   *gvar_init to its initial contents (malloc'd; NULL if *gvar_size is 0).
   Pointers to program-scope variables in the initial contents are listed
   in *gvar_relocs (malloc'd, *num_gvar_relocs entries): the host stores
   the device address of the storage plus 'target' as a 64-bit pointer at
   offset 'offset'.

   On failure returns nonzero and, if error_log is non-NULL, stores a
   malloc'd message in it. */
int pocl_metal_generate_metallib (void *program, unsigned device_i,
                                  const pocl_metal_air_target_t *target,
                                  char **metallib, size_t *metallib_size,
                                  unsigned *local_region_size,
                                  unsigned *gvar_size, char **gvar_init,
                                  pocl_metal_gvar_reloc_t **gvar_relocs,
                                  unsigned *num_gvar_relocs,
                                  char **error_log);

#ifdef __cplusplus
}
#endif

#endif /* POCL_AIR_GEN_H */
