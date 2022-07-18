/* common_driver.c - common code that can be reused between device driver
   implementations

   Copyright (c) 2011-2021 pocl developers

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

#include "config.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "pocl_cl.h"
#include "utlist.h"

// for pocl_aligned_malloc
#include "pocl_util.h"

int pocl_setup_builtin_metadata (cl_device_id device, cl_program program,
                                 unsigned program_device_i);

#ifdef ENABLE_LLVM
#include "pocl_llvm.h"
#endif

#include "common_driver.h"

#define APPEND_TO_BUILD_LOG_RET(err, ...)                                     \
  do                                                                          \
    {                                                                         \
      char temp[1024];                                                        \
      ssize_t written = snprintf (temp, 1024, __VA_ARGS__);                   \
      if (written > 0)                                                        \
        {                                                                     \
          size_t l = strlen (program->build_log[device_i]);                   \
          size_t newl = l + (size_t)written;                                  \
          char *newp = realloc (program->build_log[device_i], newl);          \
          assert (newp);                                                      \
          memcpy (newp + l, temp, (size_t)written);                           \
          newp[newl] = 0;                                                     \
          program->build_log[device_i] = newp;                                \
        }                                                                     \
      POCL_RETURN_ERROR_ON (1, err, __VA_ARGS__);                             \
    }                                                                         \
  while (0)

void
pocl_driver_read (void *data, void *__restrict__ host_ptr,
                  pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                  size_t offset, size_t size)
{
  void *__restrict__ device_ptr = src_mem_id->mem_ptr;
  if (host_ptr == device_ptr)
    return;

  memcpy (host_ptr, (char *)device_ptr + offset, size);
}

void
pocl_driver_write (void *data, const void *__restrict__ host_ptr,
                   pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                   size_t offset, size_t size)
{
  void *__restrict__ device_ptr = dst_mem_id->mem_ptr;
  if (host_ptr == device_ptr)
    return;

  memcpy ((char *)device_ptr + offset, host_ptr, size);
}

void
pocl_driver_copy (void *data, pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                  pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                  size_t dst_offset, size_t src_offset, size_t size)
{
  char *__restrict__ src_ptr = (char *)src_mem_id->mem_ptr;
  char *__restrict__ dst_ptr = (char *)dst_mem_id->mem_ptr;
  if ((src_ptr + src_offset) == (dst_ptr + dst_offset))
    return;

  memcpy (dst_ptr + dst_offset, src_ptr + src_offset, size);
}

void
pocl_driver_copy_with_size (void *data, pocl_mem_identifier *dst_mem_id,
                            cl_mem dst_buf, pocl_mem_identifier *src_mem_id,
                            cl_mem src_buf,
                            pocl_mem_identifier *content_size_buf_mem_id,
                            cl_mem content_size_buf, size_t dst_offset,
                            size_t src_offset, size_t size)
{
  char *__restrict__ src_ptr = (char *)src_mem_id->mem_ptr;
  char *__restrict__ dst_ptr = (char *)dst_mem_id->mem_ptr;
  if ((src_ptr + src_offset) == (dst_ptr + dst_offset))
    return;

  uint64_t *content_size = (uint64_t *)content_size_buf_mem_id->mem_ptr;
  if (*content_size < (src_offset + size))
    {
      if (*content_size > src_offset)
        {
          size_t real_bytes = *content_size - src_offset;
          size_t to_copy = real_bytes < size ? real_bytes : size;
          memcpy (dst_ptr + dst_offset, src_ptr + src_offset, to_copy);
        }
    }
  else
    memcpy (dst_ptr + dst_offset, src_ptr + src_offset, size);
}

void
pocl_driver_copy_rect (void *data, pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf, pocl_mem_identifier *src_mem_id,
                       cl_mem src_buf,
                       const size_t *__restrict__ const dst_origin,
                       const size_t *__restrict__ const src_origin,
                       const size_t *__restrict__ const region,
                       size_t const dst_row_pitch,
                       size_t const dst_slice_pitch,
                       size_t const src_row_pitch,
                       size_t const src_slice_pitch)
{

  void *__restrict__ src_ptr = src_mem_id->mem_ptr;
  void *__restrict__ dst_ptr = dst_mem_id->mem_ptr;
  char const *__restrict const adjusted_src_ptr
      = (char const *)src_ptr + src_origin[0] + src_row_pitch * src_origin[1]
        + src_slice_pitch * src_origin[2];
  char *__restrict__ const adjusted_dst_ptr
      = (char *)dst_ptr + dst_origin[0] + dst_row_pitch * dst_origin[1]
        + dst_slice_pitch * dst_origin[2];

  POCL_MSG_PRINT_MEMORY (
      "COPY RECT \n"
      "SRC %p DST %p SIZE %zu\n"
      "src origin %u %u %u dst origin %u %u %u \n"
      "src_row_pitch %lu src_slice pitch %lu\n"
      "dst_row_pitch %lu dst_slice_pitch %lu\n"
      "reg[0] %lu reg[1] %lu reg[2] %lu\n",
      adjusted_src_ptr, adjusted_dst_ptr, region[0] * region[1] * region[2],
      (unsigned)src_origin[0], (unsigned)src_origin[1],
      (unsigned)src_origin[2], (unsigned)dst_origin[0],
      (unsigned)dst_origin[1], (unsigned)dst_origin[2],
      (unsigned long)src_row_pitch, (unsigned long)src_slice_pitch,
      (unsigned long)dst_row_pitch, (unsigned long)dst_slice_pitch,
      (unsigned long)region[0], (unsigned long)region[1],
      (unsigned long)region[2]);

  size_t j, k;

  /* TODO: handle overlaping regions */
  if ((src_row_pitch == dst_row_pitch && dst_row_pitch == region[0])
      && (src_slice_pitch == dst_slice_pitch
          && dst_slice_pitch == (region[1] * region[0])))
    {
      memcpy (adjusted_dst_ptr, adjusted_src_ptr,
              region[2] * region[1] * region[0]);
    }
  else
    {
      for (k = 0; k < region[2]; ++k)
        for (j = 0; j < region[1]; ++j)
          memcpy (adjusted_dst_ptr + dst_row_pitch * j + dst_slice_pitch * k,
                  adjusted_src_ptr + src_row_pitch * j + src_slice_pitch * k,
                  region[0]);
    }
}

void
pocl_driver_write_rect (void *data, const void *__restrict__ const host_ptr,
                        pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                        const size_t *__restrict__ const buffer_origin,
                        const size_t *__restrict__ const host_origin,
                        const size_t *__restrict__ const region,
                        size_t const buffer_row_pitch,
                        size_t const buffer_slice_pitch,
                        size_t const host_row_pitch,
                        size_t const host_slice_pitch)
{
  void *__restrict__ device_ptr = dst_mem_id->mem_ptr;

  char *__restrict const adjusted_device_ptr
      = (char *)device_ptr + buffer_origin[0]
        + buffer_row_pitch * buffer_origin[1]
        + buffer_slice_pitch * buffer_origin[2];
  char const *__restrict__ const adjusted_host_ptr
      = (char const *)host_ptr + host_origin[0]
        + host_row_pitch * host_origin[1] + host_slice_pitch * host_origin[2];

  POCL_MSG_PRINT_MEMORY (
      "WRITE RECT \n"
      "SRC HOST %p DST DEV %p SIZE %zu\n"
      "borigin %u %u %u horigin %u %u %u \n"
      "row_pitch %lu slice pitch \n"
      "%lu host_row_pitch %lu host_slice_pitch %lu\n"
      "reg[0] %lu reg[1] %lu reg[2] %lu\n",
      adjusted_host_ptr, adjusted_device_ptr,
      region[0] * region[1] * region[2], (unsigned)buffer_origin[0],
      (unsigned)buffer_origin[1], (unsigned)buffer_origin[2],
      (unsigned)host_origin[0], (unsigned)host_origin[1],
      (unsigned)host_origin[2], (unsigned long)buffer_row_pitch,
      (unsigned long)buffer_slice_pitch, (unsigned long)host_row_pitch,
      (unsigned long)host_slice_pitch, (unsigned long)region[0],
      (unsigned long)region[1], (unsigned long)region[2]);

  size_t j, k;

  /* TODO: handle overlaping regions */
  if ((buffer_row_pitch == host_row_pitch && host_row_pitch == region[0])
      && (buffer_slice_pitch == host_slice_pitch
          && host_slice_pitch == (region[1] * region[0])))
    {
      memcpy (adjusted_device_ptr, adjusted_host_ptr,
              region[2] * region[1] * region[0]);
    }
  else
    {
      for (k = 0; k < region[2]; ++k)
        for (j = 0; j < region[1]; ++j)
          memcpy (adjusted_device_ptr + buffer_row_pitch * j
                      + buffer_slice_pitch * k,
                  adjusted_host_ptr + host_row_pitch * j
                      + host_slice_pitch * k,
                  region[0]);
    }
}

void
pocl_driver_read_rect (void *data, void *__restrict__ const host_ptr,
                       pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                       const size_t *__restrict__ const buffer_origin,
                       const size_t *__restrict__ const host_origin,
                       const size_t *__restrict__ const region,
                       size_t const buffer_row_pitch,
                       size_t const buffer_slice_pitch,
                       size_t const host_row_pitch,
                       size_t const host_slice_pitch)
{
  void *__restrict__ device_ptr = src_mem_id->mem_ptr;

  char const *__restrict const adjusted_device_ptr
      = (char const *)device_ptr + buffer_origin[2] * buffer_slice_pitch
        + buffer_origin[1] * buffer_row_pitch + buffer_origin[0];
  char *__restrict__ const adjusted_host_ptr
      = (char *)host_ptr + host_origin[2] * host_slice_pitch
        + host_origin[1] * host_row_pitch + host_origin[0];

  POCL_MSG_PRINT_MEMORY (
      "READ RECT \n"
      "SRC DEV %p DST HOST %p SIZE %zu\n"
      "borigin %u %u %u horigin %u %u %u row_pitch %lu slice pitch "
      "%lu host_row_pitch %lu host_slice_pitch %lu\n"
      "reg[0] %lu reg[1] %lu reg[2] %lu\n",
      adjusted_device_ptr, adjusted_host_ptr,
      region[0] * region[1] * region[2], (unsigned)buffer_origin[0],
      (unsigned)buffer_origin[1], (unsigned)buffer_origin[2],
      (unsigned)host_origin[0], (unsigned)host_origin[1],
      (unsigned)host_origin[2], (unsigned long)buffer_row_pitch,
      (unsigned long)buffer_slice_pitch, (unsigned long)host_row_pitch,
      (unsigned long)host_slice_pitch, (unsigned long)region[0],
      (unsigned long)region[1], (unsigned long)region[2]);

  size_t j, k;

  /* TODO: handle overlaping regions */
  if ((buffer_row_pitch == host_row_pitch && host_row_pitch == region[0])
      && (buffer_slice_pitch == host_slice_pitch
          && host_slice_pitch == (region[1] * region[0])))
    {
      memcpy (adjusted_host_ptr, adjusted_device_ptr,
              region[2] * region[1] * region[0]);
    }
  else
    {
      for (k = 0; k < region[2]; ++k)
        for (j = 0; j < region[1]; ++j)
          memcpy (adjusted_host_ptr + host_row_pitch * j
                      + host_slice_pitch * k,
                  adjusted_device_ptr + buffer_row_pitch * j
                      + buffer_slice_pitch * k,
                  region[0]);
    }
}

void
pocl_driver_memfill (void *data, pocl_mem_identifier *dst_mem_id,
                     cl_mem dst_buf, size_t size, size_t offset,
                     const void *__restrict__ pattern, size_t pattern_size)
{
  void *__restrict__ ptr = dst_mem_id->mem_ptr;
  pocl_fill_aligned_buf_with_pattern (ptr, offset, size, pattern,
                                      pattern_size);
}

cl_int
pocl_driver_map_mem (void *data, pocl_mem_identifier *src_mem_id,
                     cl_mem src_buf, mem_mapping_t *map)
{
  char *__restrict__ src_device_ptr = (char *)src_mem_id->mem_ptr;
  assert (map->host_ptr);

  if (map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION)
    {
      return CL_SUCCESS;
    }

  if (map->host_ptr == (src_device_ptr + map->offset))
    NULL;
  else
    memcpy (map->host_ptr, src_device_ptr + map->offset, map->size);

  return CL_SUCCESS;
}

cl_int
pocl_driver_unmap_mem (void *data, pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf, mem_mapping_t *map)
{
  char *__restrict__ dst_device_ptr = (char *)dst_mem_id->mem_ptr;
  assert (map->host_ptr);

  if (map->host_ptr == (dst_device_ptr + map->offset))
    NULL;
  else
    {
      if (map->map_flags != CL_MAP_READ)
        memcpy (dst_device_ptr + map->offset, map->host_ptr, map->size);
    }

  return CL_SUCCESS;
}

cl_int
pocl_driver_get_mapping_ptr (void *data, pocl_mem_identifier *mem_id,
                             cl_mem mem, mem_mapping_t *map)
{
  char *__restrict__ src_device_ptr = (char *)mem_id->mem_ptr;
  assert (mem->size > 0);
  assert (map->size > 0);

  if (mem->mem_host_ptr != NULL)
    {
      map->host_ptr = mem->mem_host_ptr + map->offset;
    }
  else
    {
      map->host_ptr = pocl_aligned_malloc (16, map->size);
    }

  assert (map->host_ptr);
  return CL_SUCCESS;
}

cl_int
pocl_driver_free_mapping_ptr (void *data, pocl_mem_identifier *mem_id,
                              cl_mem mem, mem_mapping_t *map)
{
  char *__restrict__ src_device_ptr = (char *)mem_id->mem_ptr;
  if (map->host_ptr == NULL)
    return CL_SUCCESS;

  if ((mem->mem_host_ptr != NULL)
      && map->host_ptr != (mem->mem_host_ptr + map->offset))
    pocl_aligned_free (map->host_ptr);

  map->host_ptr = NULL;
  return CL_SUCCESS;
}

cl_int
pocl_driver_alloc_mem_obj (cl_device_id device, cl_mem mem, void *host_ptr)
{
  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];

  /* let other drivers preallocate */
  if ((mem->flags & CL_MEM_ALLOC_HOST_PTR) && (mem->mem_host_ptr == NULL))
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;

  /* malloc mem_host_ptr then increase refcount */
  pocl_alloc_or_retain_mem_host_ptr (mem);

  cl_device_id svm_dev = mem->context->svm_allocdev;
  /* if we have a device which shares global memory with host,
   * and it needs to do anything to make allocations accessible
   * to itself, do it here */
  if (svm_dev && svm_dev->global_mem_id == 0 && svm_dev->ops->svm_register)
    svm_dev->ops->svm_register (svm_dev, mem->mem_host_ptr, mem->size);

  p->version = mem->mem_host_ptr_version;
  p->mem_ptr = mem->mem_host_ptr;

  POCL_MSG_PRINT_MEMORY ("Basic device ALLOC %p / size %zu \n", p->mem_ptr,
                         mem->size);

  return CL_SUCCESS;
}

void
pocl_driver_free (cl_device_id device, cl_mem mem)
{
  cl_device_id svm_dev = mem->context->svm_allocdev;
  if (svm_dev && svm_dev->global_mem_id == 0 && svm_dev->ops->svm_unregister)
    svm_dev->ops->svm_unregister (svm_dev, mem->mem_host_ptr, mem->size);

  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];
  pocl_release_mem_host_ptr (mem);
  p->mem_ptr = NULL;
  p->version = 0;
}

void
pocl_driver_svm_fill (cl_device_id dev, void *__restrict__ svm_ptr,
                      size_t size, void *__restrict__ pattern,
                      size_t pattern_size)
{
  pocl_mem_identifier temp;
  temp.mem_ptr = svm_ptr;
  pocl_driver_memfill (dev->data, &temp, NULL, size, 0, pattern, pattern_size);
}

/* These are implementations of compilation callbacks for all devices
 * that support compilation via LLVM. They take care of compilation/linking
 * of source/binary/spir down to parallel.bc level.
 *
 * The driver only has to provide the "device->ops->compile_kernel" callback,
 * which compiles parallel.bc to whatever final binary format is needed.
 *
 * Devices that support compilation by other means than LLVM,
 * must reimplement these callbacks.
 */

#ifdef ENABLE_LLVM
/* Converts SPIR to LLVM IR, and links it to pocl's kernel library. */
static int
pocl_llvm_link_and_convert_spir (cl_program program, cl_uint device_i,
                                 int link_program, int spir_build)
{
  cl_device_id device = program->devices[device_i];
  int error;

  /* SPIR-V was handled; bitcode is now either plain LLVM IR or SPIR IR */
  int spir_binary
      = bitcode_is_triple ((char *)program->binaries[device_i],
                           program->binary_sizes[device_i], "spir");
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
          APPEND_TO_BUILD_LOG_RET (CL_LINK_PROGRAM_FAILURE,
                                   "SPIR support is not available"
                                   "for device %s\n",
                                   device->short_name);
        }
      if (!spir_build)
        POCL_MSG_WARN ("SPIR binary provided, but no spir in build options\n");

      /* SPIR binaries need to be explicitly linked to the kernel
       * library. For non-SPIR binaries this happens as part of build
       * process when program.bc is generated. */
      error = pocl_llvm_link_program (
          program, device_i, 1, &program->binaries[device_i],
          &program->binary_sizes[device_i], NULL, link_program, 1);

      POCL_RETURN_ERROR_ON (error, CL_LINK_PROGRAM_FAILURE,
                            "Failed to link SPIR program.bc\n");
#else
      APPEND_TO_BUILD_LOG_RET (CL_LINK_PROGRAM_FAILURE,
                               "SPIR support is not available"
                               "for device %s\n",
                               device->short_name);
#endif
    }
  return CL_SUCCESS;
}
#endif

int
pocl_driver_build_source (cl_program program, cl_uint device_i,
                          cl_uint num_input_headers,
                          const cl_program *input_headers,
                          const char **header_include_names, int link_program)
{
  assert (program->devices[device_i]->compiler_available == CL_TRUE);
  assert (program->devices[device_i]->linker_available == CL_TRUE);

#ifdef ENABLE_LLVM

  POCL_MSG_PRINT_LLVM ("building from sources for device %d\n", device_i);

  return pocl_llvm_build_program (program, device_i, num_input_headers,
                                  input_headers, header_include_names,
                                  link_program);

#else
  POCL_RETURN_ERROR_ON (1, CL_BUILD_PROGRAM_FAILURE,
                        "This device requires LLVM to build from sources\n");
#endif
}

int
pocl_driver_build_binary (cl_program program, cl_uint device_i,
                          int link_program, int spir_build)
{

#ifdef ENABLE_LLVM
  /* poclbinary doesn't need special handling */
  if (program->pocl_binaries[device_i])
    {
      /* program.bc must be either NULL or unpacked by now */
      if (program->binaries[device_i] == NULL)
        POCL_MSG_WARN ("pocl-binary for this device doesn't contain "
                       "program.bc - you won't be able to rebuild it\n");
      else
        pocl_llvm_read_program_llvm_irs (program, device_i, NULL);
    }
  else /* program->binaries but not poclbinary */
    {
      assert (program->binaries[device_i]);
      int err = pocl_llvm_link_and_convert_spir (program, device_i,
                                                 link_program, spir_build);
      if (err != CL_SUCCESS)
        return err;
      pocl_llvm_read_program_llvm_irs (program, device_i, NULL);
    }
  return CL_SUCCESS;
#else
  POCL_RETURN_ERROR_ON ((program->pocl_binaries[device_i] == NULL),
                        CL_BUILD_PROGRAM_FAILURE,
                        "This device requires LLVM to "
                        "build from SPIR/LLVM bitcode\n");
  return CL_SUCCESS;
#endif
}

int
pocl_driver_link_program (cl_program program, cl_uint device_i,
                          cl_uint num_input_programs,
                          const cl_program *input_programs, int create_library)
{
  assert (program->devices[device_i]->linker_available == CL_TRUE);

#ifdef ENABLE_LLVM
  cl_device_id device = program->devices[device_i];
  /* just link binaries. */
  unsigned char **cur_device_binaries = (unsigned char **)alloca (
      num_input_programs * sizeof (unsigned char *));
  size_t *cur_device_binary_sizes
      = (size_t *)alloca (num_input_programs * sizeof (size_t));
  void **cur_device_llvm_irs
      = (void **)alloca (num_input_programs * sizeof (void *));

  cl_uint i;
  for (i = 0; i < num_input_programs; i++)
    {
      assert (device == input_programs[i]->devices[device_i]);
      POCL_LOCK_OBJ (input_programs[i]);

      cur_device_binaries[i] = input_programs[i]->binaries[device_i];
      assert (cur_device_binaries[i]);
      cur_device_binary_sizes[i] = input_programs[i]->binary_sizes[device_i];
      assert (cur_device_binary_sizes[i] > 0);

      pocl_llvm_read_program_llvm_irs (input_programs[i], device_i, NULL);

      cur_device_llvm_irs[i] = input_programs[i]->data[device_i];
      assert (cur_device_llvm_irs[i]);
      POCL_UNLOCK_OBJ (input_programs[i]);
    }

  int err = pocl_llvm_link_program (
      program, device_i, num_input_programs, cur_device_binaries,
      cur_device_binary_sizes, cur_device_llvm_irs, !create_library, 0);

  POCL_RETURN_ERROR_ON ((err != CL_SUCCESS), CL_LINK_PROGRAM_FAILURE,
                        "This device requires LLVM to link binaries\n");
  return CL_SUCCESS;
#else
  POCL_RETURN_ERROR_ON (1, CL_BUILD_PROGRAM_FAILURE,
                        "This device cannot link anything\n");

#endif
}

int
pocl_driver_free_program (cl_device_id device, cl_program program,
                          unsigned program_device_i)
{
#ifdef ENABLE_LLVM
  pocl_llvm_free_llvm_irs (program, program_device_i);
#endif
  return 0;
}

int
pocl_driver_setup_metadata (cl_device_id device, cl_program program,
                            unsigned program_device_i)
{
  if (program->num_builtin_kernels > 0)
    return pocl_setup_builtin_metadata (device, program, program_device_i);

#ifdef ENABLE_LLVM
  unsigned num_kernels
      = pocl_llvm_get_kernel_count (program, program_device_i);

  /* TODO zero kernels in program case */
  if (num_kernels)
    {
      program->num_kernels = num_kernels;
      program->kernel_meta
          = calloc (program->num_kernels, sizeof (pocl_kernel_metadata_t));
      pocl_llvm_get_kernels_metadata (program, program_device_i);
    }
  return 1;
#else
  return 0;
#endif
}

int
pocl_driver_supports_binary (cl_device_id device, size_t length,
                             const char *binary)
{
#ifdef ENABLE_LLVM

  /* SPIR binary is supported */
  if (bitcode_is_triple (binary, length, "spir"))
    {
      POCL_RETURN_ERROR_ON (
          (strstr (device->extensions, "cl_khr_spir") == NULL),
          CL_BUILD_PROGRAM_FAILURE,
          "SPIR binary provided, but device has no SPIR support");
      return 1;
    }

  /* LLVM IR can be supported by the driver, if the triple matches */
  if (device->llvm_target_triplet
      && bitcode_is_triple (binary, length, device->llvm_target_triplet))
    return 1;

  return 0;
#else
  POCL_MSG_ERR (
      "This driver was not build with LLVM support, so "
      "don't support loading SPIR or LLVM IR binaries, only poclbinaries.\n");
  return 0;
#endif
}

/* Build the dynamic WG sized parallel.bc and device specific code,
   for each kernel. This must be called *after* metadata has been setup  */
int
pocl_driver_build_poclbinary (cl_program program, cl_uint device_i)
{
  unsigned i;
  _cl_command_node cmd;
  cl_device_id device = program->devices[device_i];

  assert (program->build_status == CL_BUILD_SUCCESS);
  if (program->num_kernels == 0)
    return CL_SUCCESS;

  /* For binaries of other than Executable type (libraries, compiled but
   * not linked programs, etc), do not attempt to compile the kernels. */
  if (program->binary_type != CL_PROGRAM_BINARY_TYPE_EXECUTABLE)
    return CL_SUCCESS;

  memset (&cmd, 0, sizeof (_cl_command_node));
  cmd.type = CL_COMMAND_NDRANGE_KERNEL;

  POCL_LOCK_OBJ (program);

  assert (program->binaries[device_i]);

  cmd.device = device;
  cmd.program_device_i = device_i;

  struct _cl_kernel fake_k;
  memset (&fake_k, 0, sizeof (fake_k));
  fake_k.context = program->context;
  fake_k.program = program;
  fake_k.next = NULL;
  cl_kernel kernel = &fake_k;

  for (i = 0; i < program->num_kernels; i++)
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

      cmd.command.run.pc.local_size[0] = local_x;
      cmd.command.run.pc.local_size[1] = local_y;
      cmd.command.run.pc.local_size[2] = local_z;

      cmd.command.run.kernel = kernel;

      cmd.command.run.pc.global_offset[0] = cmd.command.run.pc.global_offset[1]
          = cmd.command.run.pc.global_offset[2] = 0;

      /* Force generate a generic WG function to ensure all local sizes
         can be executed using the binary. */
      device->ops->compile_kernel (&cmd, kernel, device, 0);
      /* Then generate specialized ones as requested via the
         POCL_BINARY_SPECIALIZE_WG configuration option. */
      char *temp
          = strdup (pocl_get_string_option ("POCL_BINARY_SPECIALIZE_WG", ""));
      char *token;
      char *rest = temp;

      while ((token = strtok_r (rest, ",", &rest)))
        {
          /* By default don't specialize for the origo global offset. */
          cmd.command.run.pc.global_offset[0]
              = cmd.command.run.pc.global_offset[1]
              = cmd.command.run.pc.global_offset[2] = 1;

          /* By default don't specialize for the local size. */
          cmd.command.run.pc.local_size[0] = cmd.command.run.pc.local_size[1]
              = cmd.command.run.pc.local_size[2] = 0;

          /* By default don't specialize for a small grid size. */
          cmd.command.run.force_large_grid_wg_func = 1;

          /* The format of the specialization follows the format of the
             cache directory. E.g. 128-1-1-goffs0, 13-1-1-goffs0-smallgrid
             or 0-0-0-goffs0. We thus assume the local size is always given
             first. */

          char *param1 = NULL, *param2 = NULL;
          int params_found
              = sscanf (token, "%lu-%lu-%lu-%m[^-]-%m[^-]",
                        &cmd.command.run.pc.local_size[0],
                        &cmd.command.run.pc.local_size[1],
                        &cmd.command.run.pc.local_size[2], &param1, &param2);
          if (param1 != NULL)
            {
              if (strncmp (param1, "goffs0", 6) == 0)
                {
                  cmd.command.run.pc.global_offset[0]
                      = cmd.command.run.pc.global_offset[1]
                      = cmd.command.run.pc.global_offset[2] = 0;

                  if (param2 != NULL && strncmp (param2, "smallgrid", 8) == 0)
                    {
                      cmd.command.run.force_large_grid_wg_func = 0;
                    }
                }
              else if (strncmp (param1, "smallgrid", 8) == 0)
                {
                  cmd.command.run.force_large_grid_wg_func = 0;
                }
            }
          free (param1);
          free (param2);

          device->ops->compile_kernel (&cmd, kernel, device, 1);
        }
      free (temp);
    }

  POCL_UNLOCK_OBJ (program);

  return CL_SUCCESS;
}
