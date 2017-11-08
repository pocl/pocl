/* OpenCL runtime library: caching functions

   Copyright (c) 2015 pocl developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
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
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "config.h"
#include "pocl_build_timestamp.h"

#ifdef OCS_AVAILABLE
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
/* The filename in which the program source is stored in the program's temp dir. */
#define POCL_PROGRAM_CL_FILENAME "/program.cl"
/* The filename in which the program LLVM bc is stored in the program's temp dir. */
#define POCL_PROGRAM_BC_FILENAME "/program.bc"

static char cache_topdir[POCL_FILENAME_LENGTH];
static int cache_topdir_initialized = 0;

/* sanity check on SHA1 digest emptiness */
static unsigned buildhash_is_valid(cl_program   program, unsigned     device_i)
{
  unsigned i, sum = 0;
  for(i=0; i<sizeof(SHA1_digest_t); i++)
    sum += program->build_hash[device_i][i];
  return sum;
}

int pocl_cl_device_to_index(cl_program   program,
                            cl_device_id device) {
    unsigned i;
    assert(program);
    for (i = 0; i < program->num_devices; i++)
        if (program->devices[i] == device ||
            program->devices[i] == device->parent_device)
            return i;
    return -1;
}

static void program_device_dir(char *path,
                               cl_program program,
                               unsigned device_i,
                               const char* append_path)
{
    assert(path);
    assert(program);
    assert(device_i < program->num_devices);
    assert(buildhash_is_valid(program, device_i));

    int bytes_written = snprintf(path, POCL_FILENAME_LENGTH,
                                 "%s/%s%s", cache_topdir,
                                 program->build_hash[device_i],
                                 append_path);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);
}

void pocl_cache_program_path(char*        path,
                             cl_program   program,
                             unsigned     device_i)
{
  program_device_dir (path, program, device_i, "");
}

// required in llvm API
void pocl_cache_program_bc_path(char*        program_bc_path,
                                cl_program   program,
                                unsigned     device_i) {
    program_device_dir(program_bc_path, program,
                       device_i, POCL_PROGRAM_BC_FILENAME);
}

void pocl_cache_kernel_cachedir_path (char* kernel_cachedir_path,
                                             cl_program program,
                                             unsigned device_i,
                                             cl_kernel kernel,
                                             char* append_str,
                                             size_t local_x,
                                             size_t local_y,
                                             size_t local_z)
{
  int bytes_written;
  char tempstring[POCL_FILENAME_LENGTH];

  if (program->devices[device_i]->spmd)
    {
      bytes_written = snprintf(tempstring, POCL_FILENAME_LENGTH,
                               "/%s/SPMD%s", kernel->name, append_str);
    }
  else
    {
      bytes_written = snprintf(tempstring, POCL_FILENAME_LENGTH,
                               "/%s/%zu-%zu-%zu%s", kernel->name,
                               local_x, local_y, local_z, append_str);
    }

  assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);

  program_device_dir(kernel_cachedir_path, program, device_i, tempstring);

}

void pocl_cache_kernel_cachedir(char* kernel_cachedir_path, cl_program   program,
                                unsigned device_i, cl_kernel kernel)
{
  int bytes_written;
  char tempstring[POCL_FILENAME_LENGTH];
  bytes_written = snprintf(tempstring, POCL_FILENAME_LENGTH,
                           "/%s", kernel->name);
  assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);
  program_device_dir(kernel_cachedir_path, program, device_i, tempstring);
}


// required in llvm API
void pocl_cache_work_group_function_path(char* parallel_bc_path, cl_program program,
                               unsigned device_i, cl_kernel kernel,
                               size_t local_x, size_t local_y,
                               size_t local_z) {
    assert(kernel->name);

    pocl_cache_kernel_cachedir_path(parallel_bc_path, program,
                         device_i, kernel, POCL_PARALLEL_BC_FILENAME,
                         local_x, local_y, local_z);
}

void pocl_cache_final_binary_path(char* final_binary_path, cl_program program,
                               unsigned device_i, cl_kernel kernel,
                               size_t local_x, size_t local_y,
                               size_t local_z) {
    assert(kernel->name);


    /* TODO this should be probably refactored to either
     * get the binary name from the device itself, or
     * let the device ops call pocl_llvm_generate_workgroup_function() on their own */

    int bytes_written;
    char final_binary_name[POCL_FILENAME_LENGTH];

    if (program->devices[device_i]->spmd)
        bytes_written = snprintf(final_binary_name, POCL_FILENAME_LENGTH,
                                 "%s.brig", POCL_PARALLEL_BC_FILENAME);
    else
        bytes_written = snprintf(final_binary_name, POCL_FILENAME_LENGTH,
                                 "/%s.so", kernel->name);

    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);

    pocl_cache_kernel_cachedir_path(final_binary_path, program,
                         device_i, kernel, final_binary_name,
                         local_x, local_y, local_z);
}

/******************************************************************************/

static void* acquire_program_lock(cl_program program,
                                  unsigned device_i,
                                  const char* lock_type,
                                  int shared) {
    char lock_path[POCL_FILENAME_LENGTH];
    program_device_dir(lock_path, program, device_i, lock_type);

    return acquire_lock(lock_path, shared);
}

// EXCLUSIVE writer lock
void* pocl_cache_acquire_writer_lock_i(cl_program program,
                                       unsigned device_i) {
    return acquire_program_lock(program, device_i, "_write", 0);
}

// SHARED reader lock (on clReleaseProgram, request EXCLUSIVE reader..)
void* pocl_cache_acquire_reader_lock_i(cl_program program,
                                       unsigned device_i) {
    return acquire_program_lock(program, device_i, "_read", 1);
}

void pocl_cache_release_lock(void* lock) {
    return release_lock(lock);
}

void* pocl_cache_acquire_writer_lock(cl_program program,
                                     cl_device_id device) {
    int index = pocl_cl_device_to_index(program, device);
    assert(index >= 0);
    return pocl_cache_acquire_writer_lock_i(program, (unsigned)index);
}

void* pocl_cache_acquire_reader_lock(cl_program program,
                                     cl_device_id device) {
    int index = pocl_cl_device_to_index(program, device);
    assert(index >= 0);
    return pocl_cache_acquire_reader_lock_i(program, (unsigned)index);
}

/******************************************************************************/

static void
pocl_cache_mk_temp_name (char *path_template, unsigned suffix_len, int *ret_fd)
{
  assert (cache_topdir_initialized);
#if defined(_MSC_VER) || defined(__MINGW32__)
    char* tmp = _tempnam(cache_topdir, "pocl_");
    assert(tmp);
    int bytes_written
        = snprintf (path_template, POCL_FILENAME_LENGTH, "%s", tmp);
    free(tmp);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);
#else
    /* using mkstemp() instead of tmpnam() has no real benefit
     * here, as we have to pass the filename to llvm,
     * but tmpnam() generates an annoying warning... */
    int fd;

    if (suffix_len)
      fd = mkstemps (path_template, suffix_len);
    else
      fd = mkstemp (path_template);

    if (fd < 0)
      {
        char buf[512];
        strerror_r (errno, buf, 512);
        POCL_ABORT ("mkstemp failed: %s\n", buf);
      }

    if (ret_fd)
      *ret_fd = fd;
    else
      close (fd);

    return;
#endif
}

int
pocl_cache_create_tempdir (char *path)
{
  assert (cache_topdir_initialized);
#if defined(_MSC_VER) || defined(__MINGW32__)
  char *tmp = _tempnam (cache_topdir, "pocl_");
  assert (tmp);
  int bytes_written = snprintf (path, POCL_FILENAME_LENGTH, "%s", tmp);
  free (tmp);
  assert (bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);
  return 0;
#else
  int bytes_written = snprintf (path, POCL_FILENAME_LENGTH,
                                "%s/tempdir_XXXXXX", cache_topdir);
  assert (bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);
  /* TODO mkdtemp() might not be portable */
  return (mkdtemp (path) == NULL);
#endif
}

void
pocl_cache_tempname (char *path_template, const char *suffix, int *fd)
{
  assert (cache_topdir_initialized);
  assert (path_template);
  strcpy (path_template, cache_topdir);
  size_t suffixlen = (suffix ? strlen (suffix) : 0);
  size_t max = POCL_FILENAME_LENGTH - 16 - suffixlen;
  assert (strlen (path_template) < max);
  strcat (path_template, "/tempfile_XXXXXX");
  if (suffix)
    strcat (path_template, suffix);

  pocl_cache_mk_temp_name (path_template, suffixlen, fd);
}

int
pocl_cache_write_program_source (char *program_cl_path, cl_program program)
{
  pocl_cache_tempname (program_cl_path, ".cl", NULL);
  return pocl_write_file (program_cl_path, program->source,
                          strlen (program->source), 0, 0);
}

/******************************************************************************/

int pocl_cache_update_program_last_access(cl_program program,
                                          unsigned device_i) {
    char last_accessed_path[POCL_FILENAME_LENGTH];
    program_device_dir(last_accessed_path, program,
                       device_i, POCL_LAST_ACCESSED_FILENAME);

    return pocl_touch_file(last_accessed_path);
}

/******************************************************************************/

int pocl_cache_device_cachedir_exists(cl_program   program,
                                      unsigned device_i) {
    char device_cachedir_path[POCL_FILENAME_LENGTH];
    program_device_dir(device_cachedir_path, program, device_i, "");

    return pocl_exists(device_cachedir_path);
}

/******************************************************************************/

int pocl_cache_write_descriptor(cl_program   program,
                                unsigned     device_i,
                                const char*  kernel_name,
                                const char*  content,
                                size_t       size) {
    char devdir[POCL_FILENAME_LENGTH];
    program_device_dir(devdir, program, device_i, "");

    char descriptor[POCL_FILENAME_LENGTH];
    int bytes_written = snprintf(descriptor, POCL_FILENAME_LENGTH,
                                 "%s/%s", devdir, kernel_name);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);
    if (pocl_mkdir_p(descriptor))
        return 1;

    bytes_written = snprintf(descriptor, POCL_FILENAME_LENGTH,
                                 "%s/%s/descriptor.so.kernel_obj.c",
                                 devdir, kernel_name);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);

    return pocl_write_file(descriptor, content, size, 0, 1);
}

/******************************************************************************/


char* pocl_cache_read_buildlog(cl_program program,
                               unsigned device_i) {
    char buildlog_path[POCL_FILENAME_LENGTH];
    if (program->build_hash[device_i][0] == 0)
        return NULL;
    program_device_dir(buildlog_path, program,
                       device_i, POCL_BUILDLOG_FILENAME);

    if (!pocl_exists(buildlog_path))
      return strdup("");

    char* res=NULL;
    uint64_t filesize;
    if (pocl_read_file(buildlog_path, &res, &filesize))
        return NULL;
    return res;
}


int pocl_cache_append_to_buildlog(cl_program  program,
                                  unsigned    device_i,
                                  const char *content,
                                  size_t      size) {
    char buildlog_path[POCL_FILENAME_LENGTH];
    program_device_dir(buildlog_path, program,
                       device_i, POCL_BUILDLOG_FILENAME);

    return pocl_write_file(buildlog_path, content, size, 1, 1);
}

/******************************************************************************/

#ifdef OCS_AVAILABLE
int pocl_cache_write_kernel_parallel_bc(void*        bc,
                                        cl_program   program,
                                        unsigned     device_i,
                                        cl_kernel    kernel,
                                        size_t       local_x,
                                        size_t       local_y,
                                        size_t       local_z) {
    assert(bc);

    char kernel_parallel_path[POCL_FILENAME_LENGTH];
    pocl_cache_kernel_cachedir_path(kernel_parallel_path, program, device_i,
                                    kernel, "", local_x, local_y, local_z);
    int err = pocl_mkdir_p(kernel_parallel_path);
    if (err)
      return err;

    assert( strlen(kernel_parallel_path) <
            (POCL_FILENAME_LENGTH - strlen(POCL_PARALLEL_BC_FILENAME)));
    strcat(kernel_parallel_path, POCL_PARALLEL_BC_FILENAME);
    return pocl_write_module(bc, kernel_parallel_path, 0);
}


/******************************************************************************/

static inline void
build_program_compute_hash(cl_program program,
                           unsigned   device_i,
                           const char*      preprocessed_source,
                           size_t     source_len)
{
    SHA1_CTX hash_ctx;
    unsigned i;
    cl_device_id device = program->devices[device_i];

    pocl_SHA1_Init(&hash_ctx);

    if (program->source) {
        assert(preprocessed_source);
        assert(source_len > 0);
        pocl_SHA1_Update(&hash_ctx, (uint8_t*)preprocessed_source,
                         source_len);
    } else if (program->pocl_binary_sizes[device_i] > 0) {
      /* Program was created with clCreateProgramWithBinary() with
	 a pocl binary */
      assert(program->pocl_binaries[device_i]);
      pocl_SHA1_Update(&hash_ctx,
		       (uint8_t*) program->pocl_binaries[device_i],
		       program->pocl_binary_sizes[device_i]);
      }
    else if (program->binary_sizes[device_i] > 0)
      {
        /* Program was created with clCreateProgramWithBinary() with an LLVM IR
         * binary */
        assert (program->binaries[device_i]);
        pocl_SHA1_Update (&hash_ctx, (uint8_t *)program->binaries[device_i],
                          program->binary_sizes[device_i]);
      }
    else
      {
        /* Program is linked from binaries, has no source or binary */
        // assert(program->binary_type == CL_PROGRAM_BIN)
        assert (preprocessed_source);
        assert (source_len > 0);
        pocl_SHA1_Update (&hash_ctx, (uint8_t *)preprocessed_source,
                          source_len);
      }

    if (program->compiler_options)
        pocl_SHA1_Update(&hash_ctx, (uint8_t*) program->compiler_options,
                         strlen(program->compiler_options));

    /* The kernel compiler work-group function method affects the
       produced binary heavily. */
    const char *wg_method=
        pocl_get_string_option("POCL_WORK_GROUP_METHOD", "");

    pocl_SHA1_Update(&hash_ctx, (uint8_t*) wg_method, strlen(wg_method));
    pocl_SHA1_Update(&hash_ctx, (uint8_t*) PACKAGE_VERSION,
                     strlen(PACKAGE_VERSION));
#ifdef POCL_KCACHE_SALT
    pocl_SHA1_Update(&hash_ctx, (uint8_t*) POCL_KCACHE_SALT,
                     strlen(POCL_KCACHE_SALT));
#endif
    pocl_SHA1_Update(&hash_ctx, (uint8_t*) LLVM_VERSION,
                     strlen(LLVM_VERSION));
    pocl_SHA1_Update(&hash_ctx, (uint8_t*) POCL_BUILD_TIMESTAMP,
                     strlen(POCL_BUILD_TIMESTAMP));
    pocl_SHA1_Update(&hash_ctx, (const uint8_t *)POCL_KERNELLIB_SHA1,
                     strlen(POCL_KERNELLIB_SHA1));
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
#endif


#ifdef ANDROID
static
char* pocl_get_process_name()
{
    char tmpStr[64], cmdline[512], *processName=NULL;
    FILE *statusFile;
    size_t len, i, begin;

    snprintf(tmpStr, 64, "/proc/%d/cmdline", getpid());
    statusFile=fopen(tmpStr, "r");
    if (statusFile == NULL)
        return NULL;

    if (fgets(cmdline, 511, statusFile) != NULL) {
        len=strlen(cmdline);
        begin=0;
        for (i=len-1; i >= 0; i--) { /* Extract program-name after last '/' */
            if (cmdline[i] == '/') {
                begin=i + 1;
                break;
            }
        }
        processName=strdup(cmdline + begin);
    }

    fclose(statusFile);
    return processName;
}
#endif


/******************************************************************************/

int
pocl_cache_init_topdir ()
{

  if (cache_topdir_initialized)
    return 0;

  const char *tmp_path = pocl_get_string_option ("POCL_CACHE_DIR", NULL);
  int needed;

  if (tmp_path)
    {
      needed = snprintf (cache_topdir, POCL_FILENAME_LENGTH, "%s", tmp_path);
    } else     {
#ifdef POCL_ANDROID
        char* process_name = pocl_get_process_name();
        needed = snprintf(cache_topdir, POCL_FILENAME_LENGTH,
                          "/data/data/%s/cache/", process_name);
        free(process_name);

        if (!pocl_exists(cache_topdir))
            needed = snprintf(cache_topdir,
                              POCL_FILENAME_LENGTH,
                              "/sdcard/pocl/kcache");
#elif defined(_MSC_VER) || defined(__MINGW32__)
        tmp_path = getenv("LOCALAPPDATA");
        if (!tmp_path)
            tmp_path = getenv("TEMP");
        assert(tmp_path);
        needed = snprintf(cache_topdir, POCL_FILENAME_LENGTH,
                          "%s\\pocl", tmp_path);
#else
        // "If $XDG_CACHE_HOME is either not set or empty, a default equal to
        // $HOME/.cache should be used."
        // http://standards.freedesktop.org/basedir-spec/latest/
        tmp_path = getenv("XDG_CACHE_HOME");

        if (tmp_path && tmp_path[0] != '\0') {
            needed = snprintf(cache_topdir, POCL_FILENAME_LENGTH,
                              "%s/pocl/kcache", tmp_path);
        }
        else if ((tmp_path = getenv("HOME")) != NULL) {
            needed = snprintf(cache_topdir, POCL_FILENAME_LENGTH,
                              "%s/.cache/pocl/kcache", tmp_path);
        }
        else {
            needed = snprintf(cache_topdir, POCL_FILENAME_LENGTH,
                              "/tmp/pocl/kcache");
        }
#endif
    }

  if (needed >= POCL_FILENAME_LENGTH)
    {
      POCL_MSG_ERR ("pocl: cache path longer than maximum filename length\n");
      return 1;
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
            "Luxmark benchmark binaries dowloaded from website; Luxmark "
            "installed from your linux distribution's packages should "
            "work.\n",
            cache_topdir);
        return 1;
      }

    cache_topdir_initialized = 1;
    return 0;
}

/* Create the new program cachedir, invalidating the old program
 * binaries and IRs if the new computed hash is different from the old
 * one. The source hash is computed from the preprocessed source
 * if present, from the original source otherwise: this is to ensure
 * that cache-related functions (which include log retrieval) still
 * work correctly even if preprocessing fails
 */

int
pocl_cache_create_program_cachedir(cl_program program,
                                   unsigned device_i,
                                   const char* preprocessed_source,
                                   size_t source_len,
                                   char* program_bc_path)
{
    assert(cache_topdir_initialized);

#ifdef OCS_AVAILABLE
    const char *hash_source = NULL;
    uint8_t old_build_hash[SHA1_DIGEST_SIZE] = {0};
    size_t hs_len = 0;

    if (program->source && preprocessed_source==NULL) {
        hash_source = program->source;
        hs_len = strlen(program->source);
    } else {
        hash_source = preprocessed_source;
        hs_len = source_len;
    }

    if (program->build_hash[device_i])
        memcpy(old_build_hash, program->build_hash[device_i], SHA1_DIGEST_SIZE);

    build_program_compute_hash(program, device_i, hash_source, hs_len);

    /* if the old hash is nonzero and different, we must free the built binaries
       before returning, so that they get loaded from the new location */
    if (old_build_hash[0] && memcmp(old_build_hash, program->build_hash[device_i],
            SHA1_DIGEST_SIZE))
    {
        if (program->binaries[device_i]) {
            POCL_MEM_FREE(program->binaries[device_i]);
            program->binary_sizes[device_i] = 0;
        }
        pocl_free_llvm_irs(program, device_i);
        pocl_cache_release_lock(program->read_locks[device_i]);
        program->read_locks[device_i] = NULL;
    }
#else
    assert(buildhash_is_valid(program, device_i));
    assert(program->read_locks[device_i] == NULL);
#endif

    program_device_dir(program_bc_path, program, device_i, "");

    if (pocl_mkdir_p(program_bc_path))
        return 1;

    pocl_cache_program_bc_path(program_bc_path, program, device_i);

    program->read_locks[device_i] = pocl_cache_acquire_reader_lock_i(program, device_i);
    assert(program->read_locks[device_i]);

    return 0;
}

void pocl_cache_cleanup_cachedir(cl_program program) {

    unsigned i;

    for (i = 0; i < program->num_devices; ++i)
      pocl_cache_release_lock(program->read_locks[i]);
    POCL_MEM_FREE(program->read_locks);

    if (!pocl_get_bool_option("POCL_KERNEL_CACHE", POCL_KERNEL_CACHE_DEFAULT)) {

        for (i=0; i< program->num_devices; i++) {
            if (program->build_hash[i][0] == 0)
                continue;

            void* lock = acquire_program_lock(program, i, "_read", 0);
            if (!lock)
              {
                POCL_MSG_PRINT (WARN, "", "Could not get an exclusive lock "
                                          "to remove program cachedir\n");
                continue;
              }
            char cachedir[POCL_FILENAME_LENGTH];
            program_device_dir(cachedir, program, i, "");
            pocl_rm_rf(cachedir);
            release_lock(lock);
        }
    }
}

/******************************************************************************/
