/*
    Copyright (c) 2016-2017 Tampere University of Technology.

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
 */
/**
 * @file LLVMFileUtils.cc
 *
 * File utility functions based on LLVM APIs.
 *
 * @author Michal Babej 2016. 2017
 */

#include "config.h"

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifdef HAVE_UTIME
#include <utime.h>
#endif
#else
#include <io.h>
#endif

#include <assert.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <cstdio>

#include "pocl.h"
#include "pocl_file_util.h"
#include "pocl_timing.h"

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"

#include <llvm/Support/Errc.h>

POP_COMPILER_DIAGS

#define RETURN_IF_EC if (ec) return ec.default_error_condition().value()
#define OPEN_FOR_READ ec = sys::fs::openFileForRead(p, fd)

template <typename SmallStringOrVectorT>
std::error_code CreateUniqueFileImpl(llvm::Twine p, llvm::Twine S, int fd,
                                     SmallStringOrVectorT TmpPath) {
#if LLVM_OLDER_THAN_13_0
  return llvm::sys::fs::createUniqueFile((p + S), fd, TmpPath,
                                         llvm::sys::fs::perms::owner_read |
                                             llvm::sys::fs::perms::owner_write);
#else
  return llvm::sys::fs::createUniqueFile(
      (p + S), fd, TmpPath, llvm::sys::fs::OpenFlags::OF_None,
      llvm::sys::fs::perms::owner_read | llvm::sys::fs::perms::owner_write);
#endif
}

#define CREATE_UNIQUE_FILE(S) ec = CreateUniqueFileImpl(p, S, fd, TmpPath);

#ifdef LLVM_OLDER_THAN_7_0
#define OPEN_CREATE ec = sys::fs::openFileForWrite(p, fd, sys::fs::F_RW)
#define OPEN_FOR_APPEND                                                        \
  ec = sys::fs::openFileForWrite(p, fd, sys::fs::F_RW | sys::fs::F_Append)
#else
#define OPEN_CREATE                                                            \
  ec = sys::fs::openFileForWrite(p, fd, sys::fs::CD_CreateAlways,              \
                                 sys::fs::OF_None)
#define OPEN_FOR_APPEND                                                        \
  ec = sys::fs::openFileForWrite(p, fd, sys::fs::CD_CreateAlways,              \
                                 sys::fs::OF_Append)
#endif

using namespace llvm;

static const Twine random_pattern("-%%-%%-%%-%%-%%");

/*****************************************************************************/

int
pocl_rm_rf(const char* path) {
    std::error_code ec;
    SmallString<128> DirNative;

    sys::path::native(Twine(path), DirNative);

    std::vector<std::string> FileSet, DirSet;

    for (sys::fs::recursive_directory_iterator Dir(DirNative.str(), ec), DirEnd;
         Dir != DirEnd && !ec; Dir.increment(ec)) {
        Twine p = Dir->path();
        std::string s = p.str();
        if (sys::fs::is_directory(p)) {
            DirSet.push_back(s);
        } else
            FileSet.push_back(s);
    }
    RETURN_IF_EC;

    std::vector<std::string>::iterator it;
    for (it = FileSet.begin(); it != FileSet.end(); ++it) {
        ec = sys::fs::remove(*it, true);
        RETURN_IF_EC;
    }

    std::sort(DirSet.begin(), DirSet.end());
    std::vector<std::string>::reverse_iterator it2;
    for (it2 = DirSet.rbegin(); it2 != DirSet.rend(); ++it2) {
        ec = sys::fs::remove(*it2, true);
        RETURN_IF_EC;
    }

    sys::fs::remove(Twine(path));
    return 0;
}


int
pocl_mkdir_p(const char* path) {
    Twine p(path);
    std::error_code ec = sys::fs::create_directories(p, true);
    return ec.default_error_condition().value();
}

int
pocl_remove(const char* path) {
    Twine p(path);
    std::error_code ec = sys::fs::remove(p, true);
    return ec.default_error_condition().value();
}

int pocl_remove2(Twine &p) {
  std::error_code ec = sys::fs::remove(p, true);
  return ec.default_error_condition().value();
}

int pocl_exists2(Twine &p) {
  return (sys::fs::exists(p) ? 1 : 0);
}

int
pocl_exists(const char* path) {
    Twine p(path);
    return pocl_exists2(p);
}


int pocl_touch_file(const char* path) {
#ifdef HAVE_UTIME
    int res = utime(path, NULL);
    return (res ? errno : 0);
#elif defined(HAVE_FUTIMENS)
    Twine p(path);
    int fd;
    std::error_code ec;

    OPEN_CREATE;
    RETURN_IF_EC;
    futimens(fd, NULL);

    return (close(fd) ? errno : 0);
#else
#warning No utime or futimens found, pocl will not update cache timestamps
    return 0;
#endif
}

int pocl_rename(const char *oldpath, const char *newpath) {
    Twine op(oldpath);
    Twine np(newpath);
    std::error_code ec = sys::fs::rename(op, np);
    return ec.default_error_condition().value();
}

int pocl_rename2(Twine &op, Twine &np) {
  std::error_code ec = sys::fs::rename(op, np);
  return ec.default_error_condition().value();
}

int pocl_mk_tempdir(char *output, const char *prefix) {
  Twine p(prefix);
  SmallString<512> TmpPath;

  std::error_code ec = sys::fs::createUniqueDirectory(p, TmpPath);
  RETURN_IF_EC;

  strncpy(output, TmpPath.c_str(), POCL_FILENAME_LENGTH);
  return 0;
}

int pocl_mk_tempname(char *output, const char *prefix, const char *suffix,
                     int *ret_fd) {
  Twine p(prefix);
  if (suffix == NULL)
    suffix = "";
  Twine suf(suffix);

  SmallString<512> TmpPath;
  int fd;
  std::error_code ec;

  CREATE_UNIQUE_FILE(random_pattern + suf);

  RETURN_IF_EC;

  if (ret_fd)
    *ret_fd = fd;
  else {
    if (close(fd))
      return errno ? -errno : -1;
  }

  strncpy(output, TmpPath.c_str(), POCL_FILENAME_LENGTH);
  return 0;
}

/****************************************************************************/

#define CHUNK_SIZE (2 * 1024 * 1024)

int
pocl_read_file(const char* path, char** content, uint64_t *filesize) {
    assert(content);
    assert(path);
    assert(filesize);
    *content = nullptr;
    *filesize = 0;

    int fd;
    std::error_code ec;
    Twine p(path);

    /* files in /proc return zero size, while
       files in /sys return size larger than actual content size;
       this reads the content sequentially.  */
    ssize_t total_size = 0;
    ssize_t actually_read = 0;
    char *ptr = (char *)malloc(CHUNK_SIZE + 1);
    if (ptr == nullptr)
      return -1;

    ec = sys::fs::openFileForRead(p, fd);
    if (ec)
      goto ERROR;

    do {
      char *reallocated = (char *)realloc(ptr, (total_size + CHUNK_SIZE + 1));
      if (reallocated == NULL)
        goto ERROR;
      ptr = reallocated;

      actually_read = read(fd, ptr + total_size, CHUNK_SIZE);
      if (actually_read > 0)
        total_size += actually_read;

    } while (actually_read > 0);

    if (actually_read < 0)
      goto ERROR;

    if (close(fd))
      goto ERROR;

  /* add an extra NULL character for strings */
    ptr[total_size] = 0;
    *content = ptr;
    *filesize = (uint64_t)total_size;
    return 0;

ERROR:
    free (ptr);
    return -1;
}

/* Atomic write - with rename() */
int pocl_write_file(const char *path, const char *content, uint64_t count,
                    int append, int dont_rewrite) {
    int fd;
    std::error_code ec;

    assert(path);
    assert(content);
    Twine p(path);
    SmallVector<char, 128> TmpPath;

    if (pocl_exists2(p)) {
      if (dont_rewrite) {
        if (!append)
          return 0;
      } else {
        int res = pocl_remove2(p);
        if (res)
          return res;
      }
    }

    if (append) {
        OPEN_FOR_APPEND;
        assert(fd >= 0);
    } else {
      CREATE_UNIQUE_FILE(random_pattern);
      assert(fd >= 0);
    }

    RETURN_IF_EC;

    if (write(fd, content, (ssize_t)count) < (ssize_t)count)
      return errno ? -errno : -1;

#ifdef HAVE_FDATASYNC
    if (fdatasync(fd))
      return errno ? -errno : -1;
#elif defined(HAVE_FSYNC)
    if (fsync(fd))
      return errno ? -errno : -1;
#endif

    if (close(fd))
      return -errno;

    if (append)
      return 0;
    else {
      Twine t(TmpPath);
      return pocl_rename2(t, p);
    }
}

/* write content[count] into a temporary file, and return the tempfile name in
 * output_path */
int pocl_write_tempfile(char *output_path, const char *prefix,
                        const char *suffix, const char *content,
                        uint64_t count, int *ret_fd) {
  int fd;
  std::error_code ec;

  assert(output_path);
  assert(content);
  Twine p(prefix);
  if (suffix == NULL)
    suffix = "";
  Twine suf(suffix);
  SmallString<512> TmpPath;

  CREATE_UNIQUE_FILE(random_pattern + suf);
  RETURN_IF_EC;
  assert(fd >= 0);

  ssize_t writeb = write(fd, content, (ssize_t)count);
  if (writeb < (ssize_t)count) {
    return errno ? -errno : -1;
  }

#ifdef HAVE_FDATASYNC
  if (fdatasync(fd)) {
    return errno ? -errno : -1;
  }
#elif defined(HAVE_FSYNC)
  if (fsync(fd))
    return errno ? -errno : -1;
#endif

  if (ret_fd)
    *ret_fd = fd;
  else
    close(fd);

  strncpy(output_path, TmpPath.c_str(), POCL_FILENAME_LENGTH);
  return 0;
}

/* Atomic write of IR - with rename() */
int pocl_write_module(void *module, const char* path, int dont_rewrite) {

    assert(module);
    assert(path);
    int fd;
    Twine p(path);
    std::error_code ec;

    if (pocl_exists2(p)) {
      if (dont_rewrite)
        return 0;
      else {
        int res = pocl_remove2(p);
        if (res)
          return res;
      }
    }

    /* To avoid corrupted .bc files, create a tmp file first and
       then rename it */
    SmallVector<char, 128> TmpPath;
    CREATE_UNIQUE_FILE(random_pattern);
    assert(fd >= 0);

#ifdef LLVM_OLDER_THAN_7_0
    raw_fd_ostream os(fd, 1, sys::fs::F_RW | sys::fs::F_Excl);
    RETURN_IF_EC;
    WriteBitcodeToFile((llvm::Module*)module, os);
#else
    raw_fd_ostream os(fd, true);
    RETURN_IF_EC;
    WriteBitcodeToFile(*(llvm::Module*)module, os);
#endif

    os.flush();
#ifdef HAVE_FDATASYNC
    if (fdatasync(fd))
      return errno ? -errno : -1;
#elif defined(HAVE_FSYNC)
    if (fsync(fd))
      return errno ? -errno : -1;
#endif

    os.close();
    if (os.has_error())
      return 1;

    Twine t(TmpPath);
    return pocl_rename2(t, p);
}

/****************************************************************************/
