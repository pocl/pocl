/* pocl_file_util.h: global declarations of portable file utility functions
   defined in lib/llvmopencl, due to using llvm::sys::fs & other llvm APIs

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


#ifndef POCL_FILE_UTIL_H
#define POCL_FILE_UTIL_H


#ifdef __cplusplus
extern "C" {
#endif

#pragma GCC visibility push(hidden)

#define LOCK_ACQUIRE_FAIL 3210

void* acquire_lock_check_file_exists(const char* path, int* file_exists);

void* acquire_lock(const char* path);

void* acquire_lock_immediate(const char* path);

void release_lock(void* lock, int mark_as_done);



/* Remove a directory, recursively */
int pocl_rm_rf(const char* path);

/* Make a directory, including all directories along path */
int pocl_mkdir_p(const char* path);

/* Remove a file or empty directory */
int pocl_remove(const char* path);

int pocl_exists(const char* path);

int pocl_filesize(const char* path, uint64_t* res);

/* Writes or appends data to a file. Locks the file (atomic operation) */
int pocl_write_file(const char* path, const char* content,
                    uint64_t count, int append, int dont_rewrite);

/* Allocates memory and places file contents in it. Returns number of chars read.
 * Locks the file (atomic operation) */
int pocl_read_file(const char* path, char** content, uint64_t *filesize);

/* Touch file to change last modified time. For portability, this
 * removes & creates the file. It uses a lock, so its atomic. */
int pocl_touch_file(const char* path);

int pocl_write_module(void *module, const char* path, int dont_rewrite);

int pocl_remove_locked(const char* path);

#pragma GCC visibility pop


#ifdef __cplusplus
}
#endif


#endif
