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

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "config.h"
#ifdef POCL_BUILT_WITH_CMAKE
#include "pocl_build_timestamp.h"
#endif
#include "kernellib_hash.h"

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

static void program_device_dir(char*        path,
                              cl_program   program,
                              unsigned     device_i,
                              char*        append_path) {
    assert(path);
    assert(program);
    assert(device_i < program->num_devices);
    /* sanity check on SHA1 digest emptiness */
    assert(program->build_hash[device_i][0] > 0);

    int bytes_written = snprintf(path, POCL_FILENAME_LENGTH,
                                 "%s/%s%s", cache_topdir,
                                 program->build_hash[device_i],
                                 append_path);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);
}


// required in llvm API
void pocl_cache_program_bc_path(char*        program_bc_path,
                                cl_program   program,
                                unsigned     device_i) {
    program_device_dir(program_bc_path, program,
                       device_i, POCL_PROGRAM_BC_FILENAME);
}

// required in llvm API
void pocl_cache_work_group_function_so_path(char* kernel_so_path, cl_program program,
                               unsigned device_i, cl_kernel kernel,
                               size_t local_x, size_t local_y,
                               size_t local_z) {
    assert(kernel->name);

    char tempstring[POCL_FILENAME_LENGTH];
    int bytes_written = snprintf (tempstring, POCL_FILENAME_LENGTH,
                                  "/%s/%zu-%zu-%zu/%s.so", kernel->name,
                                  local_x, local_y, local_z, kernel->name);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);
    program_device_dir(kernel_so_path, program, device_i, tempstring);
}

/******************************************************************************/

static void* acquire_program_lock(cl_program program,
                                  unsigned device_i,
                                  int shared) {
    char lock_path[POCL_FILENAME_LENGTH];
    program_device_dir(lock_path, program, device_i, "_rw");

    return acquire_lock(lock_path, shared);
}

void* pocl_cache_acquire_writer_lock_i(cl_program program,
                                       unsigned device_i) {
    return acquire_program_lock(program, device_i, 0);
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


/******************************************************************************/

void pocl_cache_mk_temp_name(char* path) {
    assert(cache_topdir_initialized);
#if defined(_MSC_VER) || defined(__MINGW32__)
    char* tmp = _tempnam(cache_topdir, "pocl_");
    assert(tmp);
    int bytes_written = snprintf(path, POCL_FILENAME_LENGTH, "%s", tmp);
    free(tmp);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);
#else
    int bytes_written = snprintf(path, POCL_FILENAME_LENGTH,
             "%s/temp_XXXXXX.cl", cache_topdir);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);
    /* using mkstemp() instead of tmpnam() has no real benefit
     * here, as we have to pass the filename to llvm,
     * but tmpnam() generates an annoying warning... */
    int fd = mkstemps(path, 3);
    assert(fd >= 0);
    close(fd);
#endif
}

int pocl_cache_write_program_source(char *program_cl_path,
                                    cl_program program) {
    pocl_cache_mk_temp_name(program_cl_path);
    return pocl_write_file(program_cl_path, program->source,
                           strlen(program->source), 0, 0);
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

static int make_kernel_cachedir_path(char*        kernel_cachedir_path,
                                     cl_program   program,
                                     unsigned     device_i,
                                     cl_kernel    kernel,
                                     unsigned     spmd,
                                     size_t       local_x,
                                     size_t       local_y,
                                     size_t       local_z) {
    assert(kernel->name);
    char tempstring[POCL_FILENAME_LENGTH];
    int bytes_written;

    if (spmd)
      {
        bytes_written = snprintf(tempstring, POCL_FILENAME_LENGTH,
                                 "/%s/SPMD", kernel->name);
      }
    else
      {
        bytes_written = snprintf(tempstring, POCL_FILENAME_LENGTH,
                                 "/%s/%zu-%zu-%zu", kernel->name,
                                 local_x, local_y, local_z);
      }

    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);

    program_device_dir(kernel_cachedir_path, program, device_i, tempstring);

    return pocl_mkdir_p(kernel_cachedir_path);
}


int pocl_cache_write_kernel_parallel_bc(void*        bc,
                                        cl_program   program,
                                        unsigned     device_i,
                                        cl_kernel    kernel,
                                        size_t       local_x,
                                        size_t       local_y,
                                        size_t       local_z) {
    assert(bc);

    char kernel_parallel_path[POCL_FILENAME_LENGTH];
    make_kernel_cachedir_path(kernel_parallel_path, program, device_i,
                              kernel, program->devices[device_i]->spmd,
                              local_x, local_y, local_z);

    assert( strlen(kernel_parallel_path) <
            (POCL_FILENAME_LENGTH - strlen(POCL_PARALLEL_BC_FILENAME)));
    strcat(kernel_parallel_path, POCL_PARALLEL_BC_FILENAME);
    return pocl_write_module(bc, kernel_parallel_path, 0);
}

int pocl_cache_make_kernel_cachedir_path(char*        kernel_cachedir_path,
                                         cl_program   program,
                                         cl_device_id device,
                                         cl_kernel    kernel,
                                         size_t       local_x,
                                         size_t       local_y,
                                         size_t       local_z) {
    int index = pocl_cl_device_to_index(program, device);
    assert(index >= 0);
    return make_kernel_cachedir_path(kernel_cachedir_path, program, index,
                                     kernel, device->spmd, local_x, local_y, local_z);
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
    } else     { /* Program was created with clCreateProgramWithBinary() */
        assert(program->binaries[device_i]);
        pocl_SHA1_Update(&hash_ctx,
                         (uint8_t*) program->binaries[device_i],
                         program->binary_sizes[device_i]);
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
    pocl_SHA1_Update(&hash_ctx, (uint8_t*) LLVM_VERSION,
                     strlen(LLVM_VERSION));
    pocl_SHA1_Update(&hash_ctx, (uint8_t*) POCL_BUILD_TIMESTAMP,
                     strlen(POCL_BUILD_TIMESTAMP));
    pocl_SHA1_Update(&hash_ctx, (const uint8_t *)POCL_KERNELLIB_SHA1,
                     strlen(POCL_KERNELLIB_SHA1));
    /*devices may include their own information to hash */
    if (device->ops->build_hash)
        device->ops->build_hash(device->data, &hash_ctx);


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

void pocl_cache_init_topdir() {

    if (cache_topdir_initialized)
        return;

    const char *tmp_path = pocl_get_string_option("POCL_CACHE_DIR", NULL);
    int needed;

    if (tmp_path && (pocl_exists(tmp_path))) {
        needed = snprintf(cache_topdir, POCL_FILENAME_LENGTH, "%s", tmp_path);
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

    if (needed >= POCL_FILENAME_LENGTH) {
        POCL_ABORT("pocl: cache path longer than maximum filename length");
    }

    assert(strlen(cache_topdir) > 0);
    if (pocl_mkdir_p(cache_topdir))
        POCL_ABORT("Could not create topdir for cache");
    cache_topdir_initialized = 1;

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
                                   char* program_bc_path,
                                   void** cache_lock)
{
    const char *hash_source = NULL;
    uint8_t old_build_hash[SHA1_DIGEST_SIZE] = {0};
    size_t hs_len = 0;

    assert(cache_topdir_initialized);

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
    }

    program_device_dir(program_bc_path, program, device_i, "");

    if (pocl_mkdir_p(program_bc_path))
        return 1;

    pocl_cache_program_bc_path(program_bc_path, program, device_i);

    *cache_lock = pocl_cache_acquire_writer_lock_i(program, device_i);

    return 0;
}

void pocl_cache_cleanup_cachedir(cl_program program) {

    unsigned i;

    if (!pocl_get_bool_option("POCL_KERNEL_CACHE", POCL_BUILD_KERNEL_CACHE)) {

        for (i=0; i< program->num_devices; i++) {
            if (program->build_hash[i][0] == 0)
                continue;

            void* lock = acquire_program_lock(program, i, 0);
            if (!lock)
                return;
            char cachedir[POCL_FILENAME_LENGTH];
            program_device_dir(cachedir, program, i, "");
            pocl_rm_rf(cachedir);
            release_lock(lock);
        }
    }
}

/******************************************************************************/
