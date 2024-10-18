/* OpenCL runtime library: caching functions

   Copyright (c) 2015-2023 pocl developers

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

#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "config.h"
#include "common.h"
#include "pocl_build_timestamp.h"
#include "pocl_version.h"

#ifdef ENABLE_LLVM
#include "kernellib_hash.h"
#endif

#include "pocl_hash.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"

#include "pocl_cl.h"
#include "pocl_runtime_config.h"

#define POCL_LAST_ACCESSED_FILENAME "/last_accessed"
/* The filename in which the program's build log is stored */
#define POCL_BUILDLOG_FILENAME      "/build.log"
/* The filename in which the program source is stored in the program's temp
 * dir. */
#define POCL_PROGRAM_CL_FILENAME "/program.cl"
/* The filename in which the program LLVM bc is stored in the program's temp
 * dir. */
#define POCL_PROGRAM_BC_FILENAME "/program.bc"
#define POCL_PROGRAM_SPV_FILENAME "/program.spv"

static char cache_topdir[POCL_MAX_PATHNAME_LENGTH];
static char tempfile_pattern[POCL_MAX_PATHNAME_LENGTH];
static char tempdir_pattern[POCL_MAX_PATHNAME_LENGTH];
static int cache_topdir_initialized = 0;
static int use_kernel_cache = 0;

/* sanity check on SHA1 digest emptiness */
unsigned pocl_cache_buildhash_is_valid(cl_program program, unsigned device_i)
{
  unsigned i, sum = 0;
  for(i=0; i<sizeof(SHA1_digest_t); i++)
    sum += program->build_hash[device_i][i];
  return sum;
}

static void program_device_dir(char *path,
                               cl_program program,
                               unsigned device_i,
                               const char* append_path)
{
    assert(path);
    assert(program);
    assert(device_i < program->num_devices);
    assert(pocl_cache_buildhash_is_valid(program, device_i));

    int bytes_written
        = snprintf (path, POCL_MAX_PATHNAME_LENGTH, "%s/%s%s", cache_topdir,
                    program->build_hash[device_i], append_path);
    assert (bytes_written > 0 && bytes_written < POCL_MAX_PATHNAME_LENGTH);
}

void pocl_cache_program_path(char*        path,
                             cl_program   program,
                             unsigned     device_i)
{
  program_device_dir (path, program, device_i, "");
}

void pocl_cache_program_bc_path(char*        program_bc_path,
                                cl_program   program,
                                unsigned     device_i) {
  program_device_dir (program_bc_path, program,
                      device_i, POCL_PROGRAM_BC_FILENAME);
}

void
pocl_cache_program_spv_path (char *program_bc_path, cl_program program,
                             unsigned device_i)
{
  program_device_dir (program_bc_path, program, device_i,
                      POCL_PROGRAM_SPV_FILENAME);
}

/**
 * \brief Return a string clipped to a given length with a hash appended.
 *
 * Returns the string as is if it is not too long. Clips the string and appends
 * a hash of the original string to make it uniqueish, if it's too long.
 *
 * \param str [in] the original string
 * \param max_length [in] maximum size of the generated string in chars
 * \param new_str [out] storage where the generated zero terminated
 * string will be written (must have at least max_length + 1 of storage).
 */
static void
pocl_hash_clipped_name (const char *str, size_t max_length, char *new_str)
{
  if (strlen (str) > max_length)
    {
      SHA1_CTX hash_ctx;
      uint8_t digest[SHA1_DIGEST_SIZE];
      int i = 0;
      char *new_str_pos = new_str;
      pocl_SHA1_Init (&hash_ctx);
      pocl_SHA1_Update (&hash_ctx, (uint8_t *)str, strlen (str));
      pocl_SHA1_Final (&hash_ctx, digest);

      strncpy (new_str, str, max_length - SHA1_DIGEST_SIZE * 2 - 1);
      new_str_pos += max_length - SHA1_DIGEST_SIZE * 2 - 1;
#ifndef _WIN32
      *new_str_pos++ = '.';
#else
      *new_str_pos++ = ' ';
#endif

      /* Convert the digest to an alphabetic string. */
      for (i = 0; i < SHA1_DIGEST_SIZE; i++)
        {
          *new_str_pos++ = (digest[i] & 0x0F) + 65;
          *new_str_pos++ = ((digest[i] & 0xF0) >> 4) + 65;
        }
      *new_str_pos = 0;
      POCL_MSG_PRINT_GENERAL ("Generated a shortened name '%s'\n", new_str);
    }
  else
    {
      strncpy (new_str, str, strlen (str) + 1);
    }
}

/* Return the cache directory for the given work-group function.
   If specialized = 1, specialization parameters are derived from run_cmd,
   otherwise a generic directory name is returned.

   The current specialization parameters are:
   - local size
   - if the global offset is zero (in all dimensions) or not
   - if the grid size in any dimension is smaller than a device
   specified limit ("smallgrid" specialization)
*/
void
pocl_cache_kernel_cachedir_path (char *kernel_cachedir_path,
                                 cl_program program, unsigned program_device_i,
                                 cl_kernel kernel, const char *append_str,
                                 _cl_command_node *command, int specialized)
{
  int bytes_written;
  _cl_command_run *run_cmd = &command->command.run;
  char tempstring[POCL_MAX_PATHNAME_LENGTH];
  cl_device_id dev = command->device;
  size_t max_grid_width = pocl_cmd_max_grid_dim_width (run_cmd);

  char kernel_dir_name[POCL_MAX_DIRNAME_LENGTH + 1];
  pocl_hash_clipped_name (kernel->name, POCL_MAX_DIRNAME_LENGTH,
                          &kernel_dir_name[0]);

  bytes_written = snprintf (
      tempstring, POCL_MAX_PATHNAME_LENGTH, "/%s/%zu-%zu-%zu%s%s%s",
      kernel_dir_name, !specialized ? 0 : run_cmd->pc.local_size[0],
      !specialized ? 0 : run_cmd->pc.local_size[1],
      !specialized ? 0 : run_cmd->pc.local_size[2],
      (specialized && run_cmd->pc.global_offset[0] == 0
       && run_cmd->pc.global_offset[1] == 0
       && run_cmd->pc.global_offset[2] == 0)
          ? "-goffs0"
          : "",
      specialized && !run_cmd->force_large_grid_wg_func
              && max_grid_width < dev->grid_width_specialization_limit
          ? "-smallgrid"
          : "",
      append_str);
  assert (bytes_written > 0 && bytes_written < POCL_MAX_PATHNAME_LENGTH);

  program_device_dir (kernel_cachedir_path, program, program_device_i,
                      tempstring);
}

void
pocl_cache_kernel_cachedir (char *kernel_cachedir_path, cl_program program,
                            unsigned device_i, const char *kernel_name)
{
  int bytes_written;
  char tempstring[POCL_MAX_PATHNAME_LENGTH];
  char file_name[POCL_MAX_FILENAME_LENGTH + 1];

  pocl_hash_clipped_name (kernel_name, POCL_MAX_FILENAME_LENGTH, &file_name[0]);

  bytes_written
      = snprintf (tempstring, POCL_MAX_PATHNAME_LENGTH, "/%s", file_name);
  assert (bytes_written > 0 && bytes_written < POCL_MAX_PATHNAME_LENGTH);
  program_device_dir (kernel_cachedir_path, program, device_i, tempstring);
}

// required in llvm API
void
pocl_cache_work_group_function_path (char *parallel_bc_path,
                                     cl_program program, unsigned device_i,
                                     cl_kernel kernel,
                                     _cl_command_node *command, int specialize)
{
  assert (kernel->name);

  pocl_cache_kernel_cachedir_path (parallel_bc_path, program, device_i, kernel,
                                   POCL_PARALLEL_BC_FILENAME, command,
                                   specialize);
}

/* Return the final binary path for the given work-group function.
   If specialized is 1, find the WG function specialized for the
   given command's properties, if 0, return the path to a generic version. */
void
pocl_cache_final_binary_path (char *final_binary_path, cl_program program,
                              unsigned device_i, cl_kernel kernel,
                              _cl_command_node *command, int specialized)
{
  assert (kernel->name);

  /* TODO: This should be probably refactored to either get the binary name
     from the device itself, or let the device ops call
     pocl_llvm_generate_workgroup_function() on their own */

  int bytes_written;
  char final_binary_name[POCL_MAX_PATHNAME_LENGTH];

  /* FIXME: Why different naming for SPMD and why the .brig suffix? */
  if (kernel->program->devices[device_i]->spmd)
    bytes_written = snprintf (final_binary_name, POCL_MAX_PATHNAME_LENGTH,
                              "%s.brig", POCL_PARALLEL_BC_FILENAME);
  else
    {
      char file_name[POCL_MAX_FILENAME_LENGTH + 1];
      /* -5: Leave space for .so and for additional .o if temp file debugging
         is enabled. */
      pocl_hash_clipped_name (kernel->name, POCL_MAX_FILENAME_LENGTH - 5,
                              &file_name[0]);
      bytes_written = snprintf (final_binary_name, POCL_MAX_PATHNAME_LENGTH,
                                "/%s.so", file_name);
    }

  assert (bytes_written > 0 && bytes_written < POCL_MAX_PATHNAME_LENGTH);

  pocl_cache_kernel_cachedir_path (final_binary_path, program, device_i,
                                   kernel, final_binary_name, command,
                                   specialized);
}

/******************************************************************************/
/******************************************************************************/

int
pocl_cache_create_tempdir (char *path)
{
  return pocl_mk_tempdir (path, tempdir_pattern);
}

int
pocl_cache_tempname (char *path, const char *suffix, int *fd)
{
  assert (cache_topdir_initialized);
  assert (path);

  return pocl_mk_tempname (path, tempfile_pattern, suffix, fd);
}

int
pocl_cache_write_program_source (char *program_cl_path, cl_program program)
{
  return pocl_write_tempfile (program_cl_path, tempfile_pattern, ".cl",
                              program->source, strlen (program->source));
}

int
pocl_cache_write_kernel_objfile (char *objfile_path,
                                 const char *objfile_content,
                                 uint64_t objfile_size)
{
  return pocl_write_tempfile (objfile_path, tempfile_pattern,
                              OBJ_EXT,
                              objfile_content, objfile_size);
}

int
pocl_cache_write_spirv (char *spirv_path,
                        const char *spirv_content,
                        uint64_t file_size)
{
  return pocl_write_tempfile (spirv_path, tempfile_pattern, ".spirv",
                              spirv_content, file_size);
}

int
pocl_cache_write_kernel_asmfile (char *asmfile_path,
                                 const char *asmfile_content,
                                 uint64_t asmfile_size)
{
  return pocl_write_tempfile (asmfile_path, tempfile_pattern, ASM_EXT,
                              asmfile_content, asmfile_size);
}

int
pocl_cache_write_generic_objfile (char *objfile_path,
                                  const char *objfile_content,
                                  uint64_t objfile_size)
{
  return pocl_write_tempfile (objfile_path, tempfile_pattern, ".binary",
                              objfile_content, objfile_size);
}

/******************************************************************************/

int pocl_cache_update_program_last_access(cl_program program,
                                          unsigned device_i) {
  if (!use_kernel_cache)
    return 0;

  char last_accessed_path[POCL_MAX_PATHNAME_LENGTH];
  program_device_dir (last_accessed_path, program, device_i,
                      POCL_LAST_ACCESSED_FILENAME);

  return pocl_touch_file (last_accessed_path);
}

/******************************************************************************/

int pocl_cache_device_cachedir_exists(cl_program   program,
                                      unsigned device_i) {
  char device_cachedir_path[POCL_MAX_PATHNAME_LENGTH];
  program_device_dir (device_cachedir_path, program, device_i, "");

  return pocl_exists (device_cachedir_path);
}

/******************************************************************************/

int
pocl_cache_write_descriptor (_cl_command_node *command, cl_kernel kernel,
                             int specialize, const char *content, size_t size)
{
  char dirr[POCL_MAX_PATHNAME_LENGTH];

  pocl_cache_kernel_cachedir_path (dirr, kernel->program, command->program_device_i,
                                   kernel, "", command, specialize);

  char descriptor[POCL_MAX_PATHNAME_LENGTH];

  pocl_cache_kernel_cachedir_path (
      descriptor, kernel->program, command->program_device_i, kernel,
      "/../descriptor.so.kernel_obj.c", command, specialize);

  if (pocl_exists (descriptor))
    return 0;

  if (pocl_mkdir_p (dirr))
    return -1;

  return pocl_write_file (descriptor, content, size, 0);
}

/******************************************************************************/

char* pocl_cache_read_buildlog(cl_program program,
                               unsigned device_i) {
  char buildlog_path[POCL_MAX_PATHNAME_LENGTH];
  if (program->build_hash[device_i][0] == 0)
    return NULL;
  program_device_dir (buildlog_path, program, device_i,
                      POCL_BUILDLOG_FILENAME);

  if (!pocl_exists (buildlog_path))
    return NULL;

  char *res = NULL;
  uint64_t filesize;
  if (pocl_read_file (buildlog_path, &res, &filesize))
    return NULL;
  return res;
}

int pocl_cache_append_to_buildlog(cl_program  program,
                                  unsigned    device_i,
                                  const char *content,
                                  size_t      size) {
    if (!pocl_cache_buildhash_is_valid (program, device_i))
      return -1;

    char buildlog_path[POCL_MAX_PATHNAME_LENGTH];
    program_device_dir(buildlog_path, program,
                       device_i, POCL_BUILDLOG_FILENAME);

    return pocl_write_file(buildlog_path, content, size, 1);
}

/******************************************************************************/

#ifdef ENABLE_LLVM
int
pocl_cache_write_kernel_parallel_bc (void *bc, cl_program program,
                                     int device_i, cl_kernel kernel,
                                     _cl_command_node *command, int specialize)
{
  assert (bc);
  char kernel_parallel_path[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_kernel_cachedir_path (kernel_parallel_path, program, device_i,
                                   kernel, "", command, specialize);
  int err = pocl_mkdir_p (kernel_parallel_path);
  if (err)
    {
      POCL_MSG_PRINT_GENERAL ("Unable to create directory %s.\n",
                              kernel_parallel_path);
      return err;
    }

  assert (strlen (kernel_parallel_path)
          < (POCL_MAX_PATHNAME_LENGTH - strlen (POCL_PARALLEL_BC_FILENAME)));
  strcat (kernel_parallel_path, POCL_PARALLEL_BC_FILENAME);
  return pocl_write_module (bc, kernel_parallel_path);
}
#endif

/******************************************************************************/

static inline void
build_program_compute_hash (cl_program program, unsigned device_i,
                            const char *hash_source, size_t source_len)
{
    SHA1_CTX hash_ctx;
    unsigned i;
    cl_device_id device = program->devices[device_i];

    static const char *builtin_seed = POCL_VERSION_BASE POCL_BUILD_TIMESTAMP
#ifdef ENABLE_LLVM
        LLVM_VERSION POCL_KERNELLIB_SHA1
#endif
        ;

    pocl_SHA1_Init(&hash_ctx);
    pocl_SHA1_Update (&hash_ctx, (uint8_t *)builtin_seed,
                      strlen (builtin_seed));

    assert (hash_source);
    assert (source_len > 0);
    pocl_SHA1_Update (&hash_ctx, (uint8_t *)hash_source, source_len);

    if (program->compiler_options)
        pocl_SHA1_Update(&hash_ctx, (uint8_t*) program->compiler_options,
                         strlen(program->compiler_options));

    pocl_SHA1_Update (&hash_ctx,
                      (uint8_t *)&program->binary_type,
                      sizeof(cl_program_binary_type));

#ifdef ENABLE_LLVM
    /* The kernel compiler work-group function method affects the
       produced binary heavily. */
    if (device->llvm_target_triplet)
      {
        const char *wg_method
            = pocl_get_string_option ("POCL_WORK_GROUP_METHOD", NULL);
        if (wg_method)
          pocl_SHA1_Update (&hash_ctx, (uint8_t *)wg_method,
                            strlen (wg_method));
      }
#endif

#ifdef ENABLE_SPIRV
    for (size_t i = 0; i < program->num_spec_consts; ++i)
      {
        if (program->spec_const_is_set[i])
          {
            pocl_SHA1_Update (&hash_ctx,
                              (uint8_t *)&program->spec_const_ids[i],
                              sizeof (cl_uint));
            pocl_SHA1_Update (&hash_ctx,
                              (uint8_t *)&program->spec_const_values[i],
                              program->spec_const_sizes[i]);
          }
      }
#endif

    /*devices may include their own information to hash */
    if (device->ops->build_hash)
      {
        char *dev_hash = device->ops->build_hash(device);
        pocl_SHA1_Update(&hash_ctx, (const uint8_t *)dev_hash, strlen(dev_hash));
        free(dev_hash);
      }

    uint8_t digest[SHA1_DIGEST_SIZE];
    pocl_SHA1_Final(&hash_ctx, digest);

    unsigned char* hashstr = program->build_hash[device_i];
    for (i=0; i < SHA1_DIGEST_SIZE; i++)
        {
            *hashstr++ = (digest[i] & 0x0F) + 65;
            *hashstr++ = ((digest[i] & 0xF0) >> 4) + 65;
        }
    *hashstr = 0;

    program->build_hash[device_i][2] = '/';
}


/******************************************************************************/

int
pocl_cache_init_topdir ()
{
  if (cache_topdir_initialized)
    return 0;

  use_kernel_cache
      = pocl_get_bool_option ("POCL_KERNEL_CACHE", POCL_KERNEL_CACHE_DEFAULT);

  const char *tmp_path = pocl_get_string_option ("POCL_CACHE_DIR", NULL);
  int needed;

  if (tmp_path)
    {
      needed = snprintf (cache_topdir, POCL_MAX_PATHNAME_LENGTH, "%s", tmp_path);
    } else {
#ifdef __ANDROID__
      POCL_MSG_ERR ("Please set the POCL_CACHE_DIR env var to your app's "
                    "cache directory (Context.getCacheDir())\n");
      return CL_FAILED;

#elif defined(_WIN32)
        tmp_path = getenv("LOCALAPPDATA");
        if (!tmp_path)
          tmp_path = getenv ("TEMP");
        if (tmp_path == NULL)
          return CL_FAILED;
        needed = snprintf (cache_topdir, POCL_MAX_PATHNAME_LENGTH, "%s\\pocl",
                           tmp_path);
#else
        // "If $XDG_CACHE_HOME is either not set or empty, a default equal to
        // $HOME/.cache should be used."
        // https://standards.freedesktop.org/basedir-spec/latest/
        tmp_path = getenv("XDG_CACHE_HOME");
        const char *p;
        if (use_kernel_cache)
          p = "pocl/kcache";
        else
          p = "pocl/uncached";

        if (tmp_path && tmp_path[0] != '\0') {
            needed = snprintf (cache_topdir, POCL_MAX_PATHNAME_LENGTH, "%s/%s",
                               tmp_path, p);
        }
        else if ((tmp_path = getenv("HOME")) != NULL) {
          needed = snprintf (cache_topdir, POCL_MAX_PATHNAME_LENGTH,
                             "%s/.cache/%s", tmp_path, p);
        }
        else {
          needed = snprintf (cache_topdir, POCL_MAX_PATHNAME_LENGTH, "/tmp/%s",
                             p);
        }
#endif
    }

  if (needed >= POCL_MAX_PATHNAME_LENGTH)
    {
      POCL_MSG_ERR ("pocl: cache path longer than maximum filename length\n");
      return CL_FAILED;
    }

    assert(strlen(cache_topdir) > 0);

    if (pocl_mkdir_p(cache_topdir))
      {
        POCL_MSG_ERR (
            "Could not create top directory (%s) for cache. \n\nNote: "
            "if you have proper rights to create that directory, and still "
            "get the error, then most likely pocl and the program you're "
            "trying to run are linked to different versions of libstdc++ "
            "library. \nThis is not a bug in pocl and there's nothing we "
            "can do to fix it - you need both pocl and your program to be"
            " compiled for your system. This is known to happen with "
            "Luxmark benchmark binaries downloaded from website; Luxmark "
            "installed from your linux distribution's packages should "
            "work.\n",
            cache_topdir);
        return CL_FAILED;
      }

    strncpy (tempfile_pattern, cache_topdir, POCL_MAX_PATHNAME_LENGTH);
    size_t len = strlen (tempfile_pattern);
    strncpy (tempfile_pattern + len, "/tempfile",
             (POCL_MAX_PATHNAME_LENGTH - len));
    tempfile_pattern[POCL_MAX_PATHNAME_LENGTH - 1] = 0;
    assert (strlen (tempfile_pattern) < POCL_MAX_PATHNAME_LENGTH);

    int bytes_written;
    if (use_kernel_cache)
      {
        bytes_written = snprintf (tempdir_pattern, POCL_MAX_PATHNAME_LENGTH,
                                  "%s/tempdir", cache_topdir);
        assert (bytes_written > 0 && bytes_written < POCL_MAX_PATHNAME_LENGTH);
      }
    else
      {
        bytes_written = snprintf (tempdir_pattern, POCL_MAX_PATHNAME_LENGTH,
                                  "%s/_UNCACHED", cache_topdir);
        assert (bytes_written > 0 && bytes_written < POCL_MAX_PATHNAME_LENGTH);
      }

    cache_topdir_initialized = 1;

    return CL_SUCCESS;
}

/* Create the new program cachedir, invalidating the old program
 * binaries and IRs if the new computed hash is different from the old
 * one. The source hash is computed from the preprocessed source
 * if present, from the original source otherwise: this is to ensure
 * that cache-related functions (which include log retrieval) still
 * work correctly even if preprocessing fails
 */

int
pocl_cache_create_program_cachedir (cl_program program, unsigned device_i,
                                    const char *hash_source,
                                    size_t hash_source_len,
                                    char *program_bc_path)
{
    assert(cache_topdir_initialized);
    assert (program_bc_path);

    /* NULL is used only in one place, clCreateWithBinary,
     * and we want to keep the original hash value in that case */
    if (hash_source == NULL)
      {
        assert (pocl_cache_buildhash_is_valid (program, device_i));
        program_device_dir (program_bc_path, program, device_i, "");

        if (pocl_mkdir_p (program_bc_path))
          return 1;
      }
    else if (use_kernel_cache)
      {
        build_program_compute_hash (program, device_i, hash_source,
                                    hash_source_len);
        assert (pocl_cache_buildhash_is_valid (program, device_i));

        program_device_dir (program_bc_path, program, device_i, "");

        if (pocl_mkdir_p (program_bc_path))
          return 1;
      }
    else
      {
        /* if kernel cache is disabled, use a random dir. */
        char random_dir[POCL_MAX_PATHNAME_LENGTH];
        if (pocl_cache_create_tempdir (random_dir))
          return 1;
        size_t s = strlen (cache_topdir) + 1;
        assert (strlen (random_dir) == (s + 16));
        memcpy (program->build_hash[device_i], random_dir + s, 16);
      }

    pocl_cache_program_bc_path (program_bc_path, program, device_i);

    return 0;
}

void pocl_cache_cleanup_cachedir(cl_program program) {

  /* only rm -rf if kernel cache is disabled */
  if (use_kernel_cache)
    return;

  unsigned i;

  for (i = 0; i < program->num_devices; i++)
    {
      if (!pocl_cache_buildhash_is_valid (program, i))
        continue;

      char cachedir[POCL_MAX_PATHNAME_LENGTH];
      program_device_dir (cachedir, program, i, "");
      pocl_rm_rf (cachedir);
    }
}

/******************************************************************************/
