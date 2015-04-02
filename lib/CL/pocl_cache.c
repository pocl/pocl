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

#include "pocl_hash.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"

#include "pocl_cl.h"
#include "pocl_runtime_config.h"

#define POCL_LAST_ACCESSED_FILENAME "last_accessed"
/* The filename in which the program's build log is stored */
#define POCL_BUILDLOG_FILENAME      "build.log"
/* The filename in which the program source is stored in the program's temp dir. */
#define POCL_PROGRAM_CL_FILENAME "program.cl"
/* The filename in which the program LLVM bc is stored in the program's temp dir. */
#define POCL_PROGRAM_BC_FILENAME "program.bc"


// required in llvm API
void pocl_cache_program_bc_path(char*        program_bc_path,
                                cl_program   program,
                                cl_device_id device) {
    int bytes_written=snprintf(program_bc_path, POCL_FILENAME_LENGTH,
                               "%s/%s/%s", program->cache_dir,
                               device->cache_dir_name, POCL_PROGRAM_BC_FILENAME);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);
}

// required in llvm API
void pocl_cache_kernel_so_path(char* kernel_so_path, cl_program program,
                               cl_device_id device, cl_kernel kernel,
                               size_t local_x, size_t local_y,
                               size_t local_z) {
    int bytes_written=snprintf(kernel_so_path, POCL_FILENAME_LENGTH,
                               "%s/%s/%s/%zu-%zu-%zu/%s.so", program->cache_dir,
                               device->cache_dir_name, kernel->name, local_x,
                               local_y, local_z, kernel->name);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);
}

/******************************************************************************/

int pocl_cache_write_program_source(char *     program_cl_path,
                                    cl_program program) {
    int bytes_written=snprintf(program_cl_path, POCL_FILENAME_LENGTH, "%s/%s",
                               program->cache_dir, POCL_PROGRAM_CL_FILENAME);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);

    return pocl_write_file(program_cl_path, program->source,
                           strlen(program->source), 0, 0);
}

/******************************************************************************/

/* If program contains "#include", disable caching
 * Included headers might get modified, force recompilation in all the cases
 * Yes, this is a very dirty way to find "# include"
 * but we can live with this for now
 */
int pocl_cache_requires_refresh(cl_program program)
{
    char *s_ptr=NULL, *ss_ptr=NULL;
    if (!pocl_get_bool_option("POCL_KERNEL_CACHE_IGNORE_INCLUDES", 0) &&
        program->source) {
        for (s_ptr=program->source; (*s_ptr); s_ptr++) {
            if ((*s_ptr) == '#') {
                /* Skip all the white-spaces between # & include */
                for (ss_ptr=s_ptr+1; *ss_ptr == ' '; ss_ptr++) ;

                if (strncmp(ss_ptr, "include", 7) == 0)
                    return 1;
            }
        }
    }
    return 0;
}

/******************************************************************************/

int pocl_cache_update_program_last_access(cl_program program) {
    char last_accessed_path[POCL_FILENAME_LENGTH];

    int bytes_written=snprintf(last_accessed_path, POCL_FILENAME_LENGTH, "%s/%s",
                               program->cache_dir, POCL_LAST_ACCESSED_FILENAME);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);

    return pocl_touch_file(last_accessed_path);
}

/******************************************************************************/

static void pocl_cache_device_cachedir_path(char*        device_cachedir,
                                            cl_program   program,
                                            cl_device_id device) {
    int bytes_written=snprintf(device_cachedir, POCL_FILENAME_LENGTH, "%s/%s",
                               program->cache_dir, device->cache_dir_name);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);
}


int pocl_cache_make_device_cachedir(cl_program   program,
                                    cl_device_id device) {
    char device_cachedir_path[POCL_FILENAME_LENGTH];
    pocl_cache_device_cachedir_path(device_cachedir_path, program, device);

    return pocl_mkdir_p(device_cachedir_path);
}

int pocl_cache_device_cachedir_exists(cl_program   program,
                                      cl_device_id device) {
    char device_cachedir_path[POCL_FILENAME_LENGTH];
    pocl_cache_device_cachedir_path(device_cachedir_path, program, device);

    return pocl_exists(device_cachedir_path);
}

char* pocl_cache_device_switches(cl_program program, cl_device_id device) {
    char device_tmpdir[POCL_FILENAME_LENGTH];
    pocl_cache_device_cachedir_path(device_tmpdir, program, device);

    if (device->ops->init_build != NULL) {
        return device->ops->init_build(device->data, device_tmpdir);
    } else
        return NULL;
}

/******************************************************************************/

int pocl_cache_write_descriptor(cl_program   program,
                                cl_device_id device,
                                const char*  kernel_name,
                                const char*  content,
                                size_t       size) {
    char devdir[POCL_FILENAME_LENGTH];
    pocl_cache_device_cachedir_path(devdir, program, device);

    char descriptor[POCL_FILENAME_LENGTH];
    int bytes_written = snprintf(descriptor, POCL_FILENAME_LENGTH,
                               "%s/%s/descriptor.so.kernel_obj.c",
                               devdir, kernel_name);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);

    return pocl_write_file(descriptor, content, size, 0, 1);
}

/******************************************************************************/

static void pocl_cache_buildlog_path(char* buildlog_path, cl_program program) {
    int bytes_written=snprintf(buildlog_path, POCL_FILENAME_LENGTH, "%s/%s",
                               program->cache_dir, POCL_BUILDLOG_FILENAME);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);
}


char* pocl_cache_read_buildlog(cl_program program) {
    char buildlog_path[POCL_FILENAME_LENGTH];
    pocl_cache_buildlog_path(buildlog_path, program);

    char* res=NULL;
    uint64_t filesize;
    if (pocl_read_file(buildlog_path, &res, &filesize))
        return NULL;

    res[filesize]='0';
    return res;
}


int pocl_cache_append_to_buildlog(cl_program  program,
                                  const char *content,
                                  size_t      size) {
    char buildlog_path[POCL_FILENAME_LENGTH];
    pocl_cache_buildlog_path(buildlog_path, program);

    return pocl_write_file(buildlog_path, content, size, 1, 1);
}

/******************************************************************************/

int pocl_cache_write_kernel_parallel_bc(void*        bc,
                                        cl_program   program,
                                        cl_device_id device,
                                        cl_kernel    kernel,
                                        size_t       local_x,
                                        size_t       local_y,
                                        size_t       local_z) {
    char kernel_parallel_path[POCL_FILENAME_LENGTH];

    int bytes_written = snprintf(kernel_parallel_path, POCL_FILENAME_LENGTH,
                               "%s/%s/%s/%zu-%zu-%zu/%s", program->cache_dir,
                               device->cache_dir_name, kernel->name, local_x,
                               local_y, local_z, POCL_PARALLEL_BC_FILENAME);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);

    return pocl_write_module(bc, kernel_parallel_path, 0);
}


int pocl_cache_make_kernel_cachedir_path(char*        kernel_cachedir_path,
                                         cl_program   program,
                                         cl_device_id device,
                                         cl_kernel    kernel,
                                         size_t       local_x,
                                         size_t       local_y,
                                         size_t       local_z) {
    int bytes_written = snprintf(kernel_cachedir_path, POCL_FILENAME_LENGTH,
                               "%s/%s/%s/%zu-%zu-%zu", program->cache_dir,
                               device->cache_dir_name, kernel->name,
                               local_x, local_y, local_z);
    assert(bytes_written > 0 && bytes_written < POCL_FILENAME_LENGTH);

    return pocl_mkdir_p(kernel_cachedir_path);
}


/******************************************************************************/







static inline void
build_program_compute_hash(cl_program program)
{
    SHA1_CTX hash_ctx;
    unsigned i;

    pocl_SHA1_Init(&hash_ctx);

    if (program->source) {
        pocl_SHA1_Update(&hash_ctx, (uint8_t*) program->source,
                         strlen(program->source));
    } else     { /* Program was created with clCreateProgramWithBinary() */
        for (i=0; i < program->num_devices; ++i)
            pocl_SHA1_Update(&hash_ctx,
                            (uint8_t*) program->binaries[i],
                            program->binary_sizes[i]);

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
    /*devices may include their own information to hash */
    for (i=0; i < program->num_devices; ++i) {
        if (program->devices[i]->ops->build_hash)
            program->devices[i]->ops->build_hash(program->devices[i]->data,
                                                 &hash_ctx);
    }

    pocl_SHA1_Final(&hash_ctx, program->build_hash);
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

// 0 on success
int
pocl_cache_create_program_cachedir(cl_program program)
{
    char *tmp_path=NULL, *cache_path=NULL;
    char hash_str[SHA1_DIGEST_SIZE * 2 + 1];
    int i;

    build_program_compute_hash(program);

    for (i=0; i < SHA1_DIGEST_SIZE; i++)
        sprintf(&hash_str[i*2], "%02x", (unsigned int) program->build_hash[i]);

    cache_path=(char*)malloc(POCL_FILENAME_LENGTH);

    tmp_path=getenv("POCL_CACHE_DIR");
    if (tmp_path && (pocl_exists(tmp_path))) {
        snprintf(cache_path, POCL_FILENAME_LENGTH, "%s/%s", tmp_path, hash_str);
    } else     {
#ifdef POCL_ANDROID
        char* process_name=pocl_get_process_name();
        snprintf(cache_path, POCL_FILENAME_LENGTH,
                 "/data/data/%s/cache/", process_name);
        free(process_name);

        if (pocl_exists(cache_path))
            strcat(cache_path, hash_str);
        else
            snprintf(cache_path,
                     POCL_FILENAME_LENGTH,
                     "/sdcard/pocl/kcache/%s",
                     hash_str);
#elif defined(_MSC_VER) || defined(__MINGW32__)
#else
        tmp_path=getenv("HOME");

        if (tmp_path)
            snprintf(cache_path,
                     POCL_FILENAME_LENGTH,
                     "%s/.pocl/kcache/%s",
                     tmp_path,
                     hash_str);
        else
            snprintf(cache_path,
                     POCL_FILENAME_LENGTH,
                     "/tmp/pocl/kcache/%s",
                     hash_str);
#endif
    }

    pocl_mkdir_p(cache_path);

    program->cache_dir=cache_path;

    program->cachedir_lock=acquire_lock_immediate(cache_path);

    return 0;
}

int pocl_cache_cleanup_cachedir(cl_program program) {
    /* we only rm -rf if we actually own the program's cache directory. */
    if (program->cachedir_lock) {
        if ((!pocl_get_bool_option("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES",
                                   0)) &&
            program->cache_dir) {
            pocl_rm_rf(program->cache_dir);
            POCL_MEM_FREE(program->cache_dir);
        }
        release_lock(program->cachedir_lock, 0);
        program->cachedir_lock=NULL;
    }
    return 0;
}

/******************************************************************************/
