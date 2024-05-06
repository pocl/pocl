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
#include "pocl_timing.h"
#include "utlist.h"

// for pocl_aligned_malloc
#include "pocl_util.h"
#include "pocl_file_util.h"

// for SPIR-V handling
#include "pocl_cache.h"
#include "pocl_file_util.h"

// sanitize kernel name
#include "builtin_kernels.hh"
#include "pocl_workgroup_func.h"

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

/* required for PoCL's command buffer extensions */
void
pocl_driver_svm_copy_rect (cl_device_id dev,
                           void *__restrict__ dst_ptr,
                           const void *__restrict__ src_ptr,
                           const size_t *__restrict__ const dst_origin,
                           const size_t *__restrict__ const src_origin,
                           const size_t *__restrict__ const region,
                           size_t dst_row_pitch,
                           size_t dst_slice_pitch,
                           size_t src_row_pitch,
                           size_t src_slice_pitch)
{
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
pocl_driver_copy_rect (void *data,
                       pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf,
                       pocl_mem_identifier *src_mem_id,
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

  pocl_driver_svm_copy_rect (NULL, dst_ptr, src_ptr, dst_origin, src_origin,
                             region, dst_row_pitch, dst_slice_pitch,
                             src_row_pitch, src_slice_pitch);
}

void
pocl_driver_svm_fill_rect (cl_device_id dev,
                           void *__restrict__ svm_ptr,
                           const size_t *origin,
                           const size_t *region,
                           size_t row_pitch,
                           size_t slice_pitch,
                           void *__restrict__ pattern,
                           size_t pattern_size)
{
  char *__restrict__ adjusted_ptr = (char *)svm_ptr + origin[0]
                                    + row_pitch * origin[1]
                                    + slice_pitch * origin[2];

  POCL_MSG_PRINT_MEMORY ("FILL RECT \n"
                         "PTR %p \n"
                         "origin %u %u %u | region %u %u %u\n"
                         "row_pitch %lu slice_pitch %lu\n",
                         adjusted_ptr, (unsigned)origin[0],
                         (unsigned)origin[1], (unsigned)origin[2],
                         (unsigned)region[0], (unsigned)region[1],
                         (unsigned)region[2], (unsigned long)row_pitch,
                         (unsigned long)slice_pitch);

  size_t j, k;

  /* TODO: handle overlaping regions */
  if ((row_pitch == region[0]) && (slice_pitch == (region[1] * region[0])))
    {
      size_t size = region[0] * region[1] * region[2];
      pocl_fill_aligned_buf_with_pattern (adjusted_ptr, 0, size, pattern,
                                          pattern_size);
    }
  else
    {
      for (k = 0; k < region[2]; ++k)
        for (j = 0; j < region[1]; ++j)
          {
            size_t offset = row_pitch * j + slice_pitch * k;
            size_t size = region[0];
            pocl_fill_aligned_buf_with_pattern (adjusted_ptr, offset, size,
                                                pattern, pattern_size);
          }
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

  /* e.g. remote never has a mem_host_ptr but can have a map host_ptr */
  if (((mem->mem_host_ptr != NULL)
       && map->host_ptr != (mem->mem_host_ptr + map->offset))
      || (mem->mem_host_ptr == NULL && map->host_ptr != NULL))
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
  p->device_addr = p->mem_ptr;

  /* If requesting memory with a fixed device address, make them pinned
     so it won't get migrated away before being freed. */
  if (mem->has_device_address)
    p->is_pinned = 1;

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

void
pocl_driver_svm_copy (cl_device_id dev,
                      void *__restrict__ dst,
                      const void *__restrict__ src,
                      size_t size)
{
  memcpy (dst, src, size);
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

#define MAX_SPEC_CONST_CMDLINE_LEN 8192
#define MAX_SPEC_CONST_OPT_LEN 256

/* load LLVM IR binary from disk, deletes existing in-memory IR */
static int
pocl_reload_program_bc (char *program_bc_path, cl_program program,
                        cl_uint device_i)
{
  char *temp_binary = NULL;
  uint64_t temp_size = 0;
  int errcode = pocl_read_file (program_bc_path, &temp_binary, &temp_size);
  if (errcode != 0 || temp_size == 0)
    return -1;
  if (program->binaries[device_i])
    POCL_MEM_FREE (program->binaries[device_i]);
  program->binaries[device_i] = (unsigned char*)temp_binary;
  program->binary_sizes[device_i] = temp_size;
  return 0;
}

/* if some SPIR-V spec constants were changed, use llvm-spirv --spec-const=...
 * to generate new LLVM bitcode from SPIR-V */
static int
pocl_regen_spirv_binary (cl_program program, cl_uint device_i)
{
#ifdef LLVM_SPIRV
  int errcode = CL_SUCCESS;
  cl_device_id device = program->devices[device_i];
  int spec_constants_changed = 0;
  char concated_spec_const_option[MAX_SPEC_CONST_CMDLINE_LEN];
  concated_spec_const_option[0] = 0;
  char program_bc_spirv[POCL_MAX_PATHNAME_LENGTH];
  char unlinked_program_bc_temp[POCL_MAX_PATHNAME_LENGTH];
  program_bc_spirv[0] = 0;
  unlinked_program_bc_temp[0] = 0;

  /* using --spirv-target-env=CL2.0 here enables llvm-spirv to produce proper
   * OpenCL 2.0 atomics, unfortunately it also enables generic ptrs, which not
   * all PoCL devices support, hence check the device */
  char* spirv_target_env = (device->generic_as_support != CL_FALSE) ?
                        "--spirv-target-env=CL2.0" :  "--spirv-target-env=CL1.2";
  char *args[8] = { LLVM_SPIRV,
                    concated_spec_const_option,
                    spirv_target_env,
                    "-r", "-o",
                    unlinked_program_bc_temp,
                    program_bc_spirv,
                    NULL };
  char **final_args = args;

  errcode = pocl_cache_tempname(unlinked_program_bc_temp, ".bc", NULL);
  POCL_RETURN_ERROR_ON ((errcode != 0), CL_BUILD_PROGRAM_FAILURE,
                        "failed to create tmpfile in pocl cache\n");

  errcode = pocl_cache_write_spirv (program_bc_spirv,
                                    (const char *)program->program_il,
                                    (uint64_t)program->program_il_size);
  POCL_RETURN_ERROR_ON ((errcode != 0), CL_BUILD_PROGRAM_FAILURE,
                        "failed to write into pocl cache\n");

  for (unsigned i = 0; i < program->num_spec_consts; ++i)
    spec_constants_changed += program->spec_const_is_set[i];

  if (spec_constants_changed)
    {
      strcpy (concated_spec_const_option, "--spec-const=");
      for (unsigned i = 0; i < program->num_spec_consts; ++i)
        {
          if (program->spec_const_is_set[i])
            {
              char opt[MAX_SPEC_CONST_OPT_LEN];
              snprintf (opt, MAX_SPEC_CONST_OPT_LEN, "%u:i%u:%zu ",
                        program->spec_const_ids[i],
                        program->spec_const_sizes[i] * 8,
                        program->spec_const_values[i]);
              strcat (concated_spec_const_option, opt);
            }
        }
    }
  else
    {
      /* skip concated_spec_const_option */
      args[0] = NULL;
      args[1] = LLVM_SPIRV;
      final_args = args + 1;
    }

  errcode = pocl_run_command (final_args);
  POCL_GOTO_ERROR_ON ((errcode != 0), CL_INVALID_VALUE,
                      "External command (llvm-spirv translator) failed!\n");

  POCL_GOTO_ERROR_ON (
      (pocl_reload_program_bc (unlinked_program_bc_temp, program, device_i)),
      CL_INVALID_VALUE, "Can't read llvm-spirv converted bitcode file\n");

  errcode = CL_SUCCESS;

ERROR:
  if (pocl_get_bool_option ("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES", 0) == 0)
    {
      if (unlinked_program_bc_temp[0])
        pocl_remove (unlinked_program_bc_temp);
      if (program_bc_spirv[0])
        pocl_remove (program_bc_spirv);
    }
  return errcode;
#else
  return -1;
#endif
}

/* Converts SPIR-V / SPIR to LLVM IR, and links it to pocl's kernel library */
static int
pocl_llvm_convert_and_link_ir (cl_program program, cl_uint device_i,
                               int link_builtin_lib, int spir_build)
{
  cl_device_id device = program->devices[device_i];
  int errcode;
  char program_bc_path[POCL_MAX_PATHNAME_LENGTH];

  if (program->binaries[device_i])
    {
      int spir_binary
          = pocl_bitcode_is_triple ((char *)program->binaries[device_i],
                               program->binary_sizes[device_i], "spir");
      if (spir_binary)
        {
          POCL_MSG_PRINT_LLVM ("LLVM-SPIR binary detected\n");
          if (!spir_build)
            POCL_MSG_WARN (
                "SPIR binary provided, but no spir in build options\n");
        }
      else
        POCL_MSG_PRINT_LLVM ("building from a BC binary for device %d\n",
                             device_i);
    }

  // SPIR-V requires special handling because of spec constants
  if (program->program_il && program->program_il_size > 0)
    {
#ifdef ENABLE_SPIRV
      if (!strstr (device->extensions, "cl_khr_il_program"))
        {
          APPEND_TO_BUILD_LOG_RET (CL_LINK_PROGRAM_FAILURE,
                                   "SPIR support is not available"
                                   "for device %s\n",
                                   device->short_name);
        }

      /* the hash created here should now reflect the source (SPIR-V),
       * PoCL build, LLVM version, the compiler options, the device's
       * LLVM triple, and the Spec Constants */
      errcode = pocl_cache_create_program_cachedir (
          program, device_i, (char *)program->program_il,
          program->program_il_size, program_bc_path);
      POCL_RETURN_ERROR_ON (errcode, CL_LINK_PROGRAM_FAILURE,
                            "Failed to create cachedir for program.bc\n");

      if (pocl_exists (program_bc_path))
        {
          POCL_MSG_PRINT_LLVM ("Found cached compiled SPIRV binary at %s, "
                               "skipping compilation\n",
                               program_bc_path);
          POCL_RETURN_ERROR_ON (
              (pocl_reload_program_bc (program_bc_path, program, device_i)),
              CL_LINK_PROGRAM_FAILURE,
              "Can't read llvm-spirv converted bitcode file\n");

          pocl_llvm_free_llvm_irs (program, device_i);

          return CL_SUCCESS;
        }
      else
        {
          POCL_MSG_PRINT_LLVM ("Cached compiled SPIRV binary not found, "
                               "generating SPIR IR to %s\n",
                               program_bc_path);

          /* SPIR IR binaries need to be regenerated from SPIR-V
           * if specialization constants change. */
          errcode
              = pocl_regen_spirv_binary (program, device_i);
          POCL_RETURN_ERROR_ON ((errcode != CL_SUCCESS),
                                CL_LINK_PROGRAM_FAILURE,
                                "Failed to generate SPIR from SPIR-V "
                                "with specialization constants\n");

          pocl_llvm_free_llvm_irs (program, device_i);

          // can't return here yet, we need to also link the builtin library
        }

#else
      APPEND_TO_BUILD_LOG_RET (CL_LINK_PROGRAM_FAILURE,
                               "SPIR-V support is not available"
                               "for device %s\n",
                               device->short_name);
#endif
      // target-specific IR binaries & SPIR (not SPIRV) binaries handled here
    }
  else
    {
      /* the hash created here should now reflect the source (LLVM IR),
       * PoCL build, LLVM version, the compiler options, the device's
       * LLVM triple */
      errcode = pocl_cache_create_program_cachedir (
          program, device_i, (char *)program->binaries[device_i],
          program->binary_sizes[device_i], program_bc_path);
      POCL_RETURN_ERROR_ON (errcode, CL_LINK_PROGRAM_FAILURE,
                            "Failed to create cachedir for program.bc\n");

      if (pocl_exists (program_bc_path))
        {
          POCL_MSG_PRINT_LLVM (
              "Found cached binary at %s, skipping compilation\n",
              program_bc_path);

          POCL_RETURN_ERROR_ON (
              (pocl_reload_program_bc (program_bc_path, program, device_i)),
              CL_LINK_PROGRAM_FAILURE,
              "Can't read llvm-spirv converted bitcode file\n");

          pocl_llvm_free_llvm_irs (program, device_i);

          return CL_SUCCESS;
        }
    }

  /* convert module from SPIR to Target triple, and if requested
   * link the resulting binary to the builtin library */
  errcode = pocl_llvm_link_program (
      program, device_i, 1, &program->binaries[device_i],
      &program->binary_sizes[device_i], NULL, link_builtin_lib, CL_FALSE);
  POCL_RETURN_ERROR_ON (errcode, CL_LINK_PROGRAM_FAILURE,
                        "Failed to link program.bc\n");
  return CL_SUCCESS;
}


#endif

int
pocl_driver_build_source (cl_program program, cl_uint device_i,
                          cl_uint num_input_headers,
                          const cl_program *input_headers,
                          const char **header_include_names,
                          int link_builtin_lib)
{
  assert (program->devices[device_i]->compiler_available == CL_TRUE);
  assert (program->devices[device_i]->linker_available == CL_TRUE);

#ifdef ENABLE_LLVM

  POCL_MSG_PRINT_LLVM ("building from sources for device %d\n", device_i);

  return pocl_llvm_build_program (program, device_i, num_input_headers,
                                  input_headers, header_include_names,
                                  link_builtin_lib);

#else
  POCL_RETURN_ERROR_ON (1, CL_BUILD_PROGRAM_FAILURE,
                        "This device requires LLVM to build from sources\n");
#endif
}

int
pocl_driver_build_binary (cl_program program, cl_uint device_i,
                          int link_builtin_lib, int spir_build)
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
  else /* has program->binaries or SPIR-V, but not poclbinary */
    {
      assert (program->binaries[device_i] || program->program_il);
      int err = pocl_llvm_convert_and_link_ir (program, device_i,
                                               link_builtin_lib, spir_build);
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

      cur_device_llvm_irs[i] = input_programs[i]->llvm_irs[device_i];
      assert (cur_device_llvm_irs[i]);
      POCL_UNLOCK_OBJ (input_programs[i]);
    }

  int err = pocl_llvm_link_program (
      program, device_i, num_input_programs, cur_device_binaries,
      cur_device_binary_sizes, cur_device_llvm_irs, !create_library, CL_TRUE);

  POCL_RETURN_ERROR_ON ((err != CL_SUCCESS), CL_LINK_PROGRAM_FAILURE,
                        "Linking of program failed\n");
  return CL_SUCCESS;
#else
  POCL_RETURN_ERROR_ON (1, CL_BUILD_PROGRAM_FAILURE,
                        "This device requires LLVM to link binaries\n");

#endif
}

int
pocl_driver_free_program (cl_device_id device, cl_program program,
                          unsigned dev_i)
{
#ifdef ENABLE_LLVM
  pocl_llvm_free_llvm_irs (program, dev_i);
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

  /* SPIR-V binaries are supported if we have llvm-spirv */
#ifdef ENABLE_SPIRV
  if (pocl_bitcode_is_spirv_execmodel_kernel (binary, length))
    return 1;
#endif

  /* LLVM IR can be supported by the driver, if the triple matches */
  if (device->llvm_target_triplet
      && pocl_bitcode_is_triple (binary, length, device->llvm_target_triplet))
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

  pocl_driver_build_gvar_init_kernel (program, device_i, device, NULL);

  POCL_UNLOCK_OBJ (program);

  return CL_SUCCESS;
}


int
pocl_driver_build_opencl_builtins (cl_program program, cl_uint device_i)
{
  int err;

  cl_device_id dev = program->devices[device_i];

  if (dev->compiler_available == CL_FALSE || dev->llvm_cpu == NULL)
    return 0;

// TODO this should probably be outside
#ifdef ENABLE_LLVM
  POCL_MSG_PRINT_LLVM ("building builtin kernels for %s\n", dev->short_name);

  assert (program->build_status == CL_BUILD_NONE);

  uint64_t builtins_file_len = 0;
  char builtin_path[POCL_MAX_PATHNAME_LENGTH];
  char *builtins_file = NULL;

  uint64_t common_builtins_file_len = 0;
  char common_builtin_path[POCL_MAX_PATHNAME_LENGTH];
  char *common_builtins_file = NULL;

  char filename[64];
  filename[0] = 0;
  if (dev->builtins_sources_path)
    {
      filename[0] = '/';
      pocl_str_tolower (filename + 1, dev->ops->device_name);
      strcat (filename, "/");
      strcat (filename, dev->builtins_sources_path);
    }

  /* filename is now e.g. "/cuda/builtins.cl";
   * loads either
   * <srcdir>/lib/CL/devices/cuda/builtins.cl
   * or
   * <private_datadir>/cuda/builtins.cl
   */
  pocl_get_srcdir_or_datadir (builtin_path, "/lib/CL/devices", "", filename);
  pocl_read_file (builtin_path, &builtins_file, &builtins_file_len);

  pocl_get_srcdir_or_datadir (common_builtin_path, "/lib/CL/devices", "",
                              "/common_builtin_kernels.cl");
  pocl_read_file (common_builtin_path, &common_builtins_file,
                  &common_builtins_file_len);

  POCL_RETURN_ERROR_ON (
      (builtins_file == NULL && common_builtins_file == NULL),
      CL_BUILD_PROGRAM_FAILURE,
      "failed to open either of the sources for builtin kernels: \n%s\n%s\n",
      common_builtin_path, builtin_path);

  if (builtins_file != NULL)
    program->source = builtins_file;
  if (common_builtins_file != NULL)
    program->source = common_builtins_file;

  if (builtins_file != NULL && common_builtins_file != NULL)
    {
      program->source
          = malloc (builtins_file_len + common_builtins_file_len + 1);
      memcpy (program->source, common_builtins_file, common_builtins_file_len);
      memcpy (program->source + common_builtins_file_len, builtins_file,
              builtins_file_len);
      program->source[common_builtins_file_len + builtins_file_len] = 0;
      POCL_MEM_FREE (builtins_file);
      POCL_MEM_FREE (common_builtins_file);
    }

  err = pocl_driver_build_source (program, device_i, 0, NULL, NULL, 1);
  POCL_RETURN_ERROR_ON ((err != CL_SUCCESS), CL_BUILD_PROGRAM_FAILURE,
                        "failed to build OpenCL builtins for %s\n",
                        dev->short_name);
  return 0;
#else
  return -1;
#endif
}

/**
* \brief  Since the ProgramScopeVariables pass creates a hidden kernel
* for variable initialization, we need to compile (and run) that kernel
* somehow. This creates a temporary _cl_command_node and _cl_kernel structs
* on stack, fills them with data, and calls device->ops->compile_kernel().
* Then, optionally, runs the provided callback with these on-stack structs
* as arguments.
*
* Can be used either to compile the hidden initialization kernel for
* a poclbinary, or to compile & run program-scope initialization before
* kernel execution. Should work with all devices that use PoCL's LLVM
* compilation.
*
* \param [in] program the cl_program used
* \param [in] device the device used
* \param [in] dev_i index into program->devices[] corresponding to device
* \param [in] callback either NULL or a callback
*/
void
pocl_driver_build_gvar_init_kernel (cl_program program, cl_uint dev_i,
                                    cl_device_id device, gvar_init_callback_t callback)
{
#ifdef ENABLE_HOST_CPU_DEVICES
  if (device->run_program_scope_variables_pass == CL_FALSE)
    return;

  if (program->global_var_total_size[dev_i] > 0 && program->gvar_storage[dev_i] == NULL)
    {
      program->gvar_storage[dev_i] = pocl_aligned_malloc (
          MAX_EXTENDED_ALIGNMENT, program->global_var_total_size[dev_i]);

      pocl_kernel_metadata_t fake_meta;
      memset (&fake_meta, 0, sizeof (fake_meta));
      pocl_kernel_hash_t fake_build_hash;

      SHA1_CTX hash_ctx;
      pocl_SHA1_Init (&hash_ctx);
      pocl_SHA1_Update (&hash_ctx, (uint8_t *)program->build_hash[dev_i],
                        sizeof (SHA1_digest_t));
      pocl_SHA1_Update (&hash_ctx, (uint8_t *)POCL_GVAR_INIT_KERNEL_NAME,
                        strlen (POCL_GVAR_INIT_KERNEL_NAME));
      pocl_SHA1_Final (&hash_ctx, fake_build_hash);

      fake_meta.build_hash = &fake_build_hash;

      struct _cl_kernel fake_kernel;
      memset (&fake_kernel, 0, sizeof (fake_kernel));
      fake_kernel.meta = &fake_meta;
      fake_kernel.name = POCL_GVAR_INIT_KERNEL_NAME;
      fake_kernel.context = program->context;
      fake_kernel.program = program;

      _cl_command_node fake_cmd;
      memset (&fake_cmd, 0, sizeof (_cl_command_node));
      fake_cmd.program_device_i = dev_i;
      fake_cmd.device = device;
      fake_cmd.type = CL_COMMAND_NDRANGE_KERNEL;
      fake_cmd.command.run.kernel = &fake_kernel;
      fake_cmd.command.run.hash = fake_meta.build_hash;
      fake_cmd.command.run.pc.local_size[0] = 1;
      fake_cmd.command.run.pc.local_size[1] = 1;
      fake_cmd.command.run.pc.local_size[2] = 1;
      fake_cmd.command.run.pc.work_dim = 3;
      fake_cmd.command.run.pc.num_groups[0] = 1;
      fake_cmd.command.run.pc.num_groups[1] = 1;
      fake_cmd.command.run.pc.num_groups[2] = 1;
      fake_cmd.command.run.pc.global_offset[0] = 0;
      fake_cmd.command.run.pc.global_offset[1] = 0;
      fake_cmd.command.run.pc.global_offset[2] = 0;
      fake_cmd.command.run.pc.global_var_buffer = program->gvar_storage[dev_i];

      device->ops->compile_kernel (&fake_cmd, &fake_kernel, device, 0);

      if (callback) {
        callback (program, dev_i, &fake_cmd);
      }
    }
#endif
}


/**
* \brief  Callback (for CPU devices only), to be used with
* pocl_driver_build_gvar_init_kernel, to actually run the program-scope
* initialization kernel.
*
* \param [in] Program the cl_program used
* \param [in] dev_i index into program->devices[]
* \param [in] fake_cmd temporary _cl_command_node for the
*/
void
pocl_cpu_gvar_init_callback(cl_program program, cl_uint dev_i,
                            _cl_command_node *fake_cmd)
{
#ifdef ENABLE_HOST_CPU_DEVICES
  uint8_t arguments[1];
  pocl_workgroup_func gvar_init_wg
      = (pocl_workgroup_func)fake_cmd->command.run.wg;
  gvar_init_wg ((uint8_t *)arguments, (uint8_t *)&fake_cmd->command.run.pc,
                0, 0, 0);
#endif
}

cl_int pocl_driver_get_synchronized_timestamps (cl_device_id dev,
                                                cl_ulong *dev_timestamp,
                                                cl_ulong *host_timestamp)
{
  uint64_t timestamp = pocl_gettimemono_ns();
  if (dev_timestamp) {
    *dev_timestamp = timestamp;
  }
  if (host_timestamp) {
    *host_timestamp = timestamp;
  }
  return CL_SUCCESS;
}
