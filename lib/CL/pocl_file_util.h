/* pocl_file_util.h: global declarations of portable file utility functions

   Copyright (c) 2015-2024 pocl developers

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

#ifndef POCL_FILE_UTIL_H
#define POCL_FILE_UTIL_H

#include "pocl_export.h"

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

typedef enum pocl_file_type_e {
  POCL_FS_STATUS_ERROR = 0,
  POCL_FS_NOT_FOUND,
  POCL_FS_REGULAR,
  POCL_FS_DIRECTORY,
  /* TODO: Add more when needed. */
  POCL_FS_UNKNOWN,
} pocl_file_type;

typedef struct pocl_dir_iter_s {
  void *handle;
} pocl_dir_iter;

/** Remove a directory, recursively */
int pocl_rm_rf(const char* path);

/** Make a directory, including all directories along path */
POCL_EXPORT
int pocl_mkdir_p(const char* path);

/** Remove a file or empty directory */
POCL_EXPORT
int pocl_remove(const char* path);

POCL_EXPORT
int pocl_rename(const char *oldpath, const char *newpath);

POCL_EXPORT
int pocl_exists(const char* path);

/** Touch file to change last modified time. For portability, this
 * removes & creates the file. */
int pocl_touch_file(const char* path);

/** Writes or appends data to a file.  */
POCL_EXPORT
int pocl_write_file(const char* path, const char* content,
                    uint64_t count, int append);

int pocl_write_tempfile (char *output_path,
                         const char *prefix,
                         const char *suffix,
                         const char *content,
                         uint64_t count);

/** Allocates memory and places file contents in it.
 * Returns negative errno on error, zero otherwise. */
POCL_EXPORT
int pocl_read_file(const char* path, char** content, uint64_t *filesize);

int pocl_write_module(void *module, const char* path);

int pocl_mk_tempdir (char *output, const char *prefix);

POCL_EXPORT
int pocl_mk_tempname (char *output, const char *prefix, const char *suffix,
                      int *ret_fd);

/** Gives parent directory.
 *
 * This method follows the behavior of C++
 * std::filesystem::path::parent_path().
 *
 * The given path is modified inplace as well as returned.
 */
char *pocl_parent_path (char *path);

/** Returns the type of the given file.
 *
 * On an error returns POCL_FS_STATUS_ERROR.  */
POCL_EXPORT
pocl_file_type pocl_get_file_type (const char *path);

/** Create a directory iterator over the contents of the given directory 'path'.
 *
 * The creted iterator won't recurse into the directories found in the
 * given 'path'.
 *
 * Returns non-zero on an error. Otherwise, returns 0 and the object
 * is returned via 'iter'.
 */
POCL_EXPORT
int pocl_dir_iterator (const char *path, pocl_dir_iter *iter);

/** Gets the next directory entry (or the first one on the newly created
 * iterator). Returns zero if there are no entries left and,
 * otherwise, non-zero.  */
POCL_EXPORT
int pocl_dir_next_entry (pocl_dir_iter iter);

/** Returns the path of the currently pointed directory entry.
 *
 * The returned path includes the path given in pocl_dir_iterator().
 *
 * pocl_dir_next_entry() must be called successfully prior calling
 * this function.  The returned pointer is invalidated on the next
 * pocl_dir_next_entry() call.  */
POCL_EXPORT
const char *pocl_dir_iter_get_path (pocl_dir_iter iter);

/** Releases resources allocated for the iterator.
 *
 * The iterator must have been created with pocl_dir_iterator() successfully. */
POCL_EXPORT
void pocl_release_dir_iterator (pocl_dir_iter *iter);

#ifdef __cplusplus
}
#endif


#endif
