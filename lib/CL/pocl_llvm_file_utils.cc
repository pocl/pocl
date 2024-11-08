/*
    Copyright (c) 2016-2017 Tampere University of Technology.

    Copyright (c) 2024 Michal Babej / Intel Finland Oy

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
 * @author Michal Babej 2016. 2017, 2024
 */

#include "config.h"

#if !defined(_WIN32) && (defined(HAVE_FDATASYNC) || defined(HAVE_FSYNC))
#define _GNU_SOURCE
#define _DEFAULT_SOURCE
#define _BSD_SOURCE
#include <unistd.h>
#endif

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <fileapi.h>
#include <windows.h>
#endif

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "pocl.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_os_ostream.h>

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>

#include <llvm/Support/Errc.h>

POP_COMPILER_DIAGS

using namespace llvm::sys;

static const llvm::Twine random_pattern("-%%-%%-%%-%%-%%");

/*****************************************************************************/

int pocl_rm_rf(const char *path) {
  std::error_code ec;
  llvm::SmallString<128> DirNative;

  path::native(llvm::Twine(path), DirNative);

  std::vector<std::string> FileSet, DirSet;

  for (fs::recursive_directory_iterator Dir(DirNative.str(), ec), DirEnd;
       Dir != DirEnd && !ec; Dir.increment(ec)) {
    llvm::Twine p = Dir->path();
    std::string s = p.str();
    if (fs::is_directory(p)) {
      DirSet.push_back(s);
    } else
      FileSet.push_back(s);
  }
  if (ec)
    return -1;

  std::vector<std::string>::iterator it;
  for (it = FileSet.begin(); it != FileSet.end(); ++it) {
    ec = fs::remove(*it, true);
    if (ec)
      return -1;
  }

  std::sort(DirSet.begin(), DirSet.end());
  std::vector<std::string>::reverse_iterator it2;
  for (it2 = DirSet.rbegin(); it2 != DirSet.rend(); ++it2) {
    ec = fs::remove(*it2, true);
    if (ec)
      return -1;
  }

  fs::remove(llvm::Twine(path));
  return 0;
}

int pocl_mkdir_p(const char *path) {
  llvm::Twine p(path);
  std::error_code ec = fs::create_directories(p, true);
  return ec.default_error_condition().value();
}

int pocl_remove(const char *path) {
  llvm::Twine p(path);
  std::error_code ec = fs::remove(p, true);
  return ec.default_error_condition().value();
}

int pocl_remove2(llvm::Twine &p) {
  std::error_code ec = fs::remove(p, true);
  return ec.default_error_condition().value();
}

int pocl_exists2(llvm::Twine &p) { return (fs::exists(p) ? 1 : 0); }

int pocl_exists(const char *path) {
  llvm::Twine p(path);
  return pocl_exists2(p);
}

int pocl_touch_file(const char *path) {
  /*llvm::fs:: TODO */
  return 0;
}

int pocl_rename(const char *oldpath, const char *newpath) {
  llvm::Twine op(oldpath);
  llvm::Twine np(newpath);
  std::error_code ec = fs::rename(op, np);
  return ec.default_error_condition().value();
}

int pocl_rename2(llvm::Twine &op, llvm::Twine &np) {
  std::error_code ec = fs::rename(op, np);
  return ec.default_error_condition().value();
}

int pocl_mk_tempdir(char *output, const char *prefix) {
  llvm::Twine p(prefix);
  llvm::SmallString<512> TmpPath;

  std::error_code ec = fs::createUniqueDirectory(p, TmpPath);
  if (ec)
    return -1;

  strncpy(output, TmpPath.c_str(), POCL_MAX_PATHNAME_LENGTH);
  return 0;
}

int pocl_mk_tempname(char *output, const char *prefix, const char *suffix,
                     int *ret_fd) {
  llvm::Twine p(prefix);
  if (suffix == NULL)
    suffix = "";
  llvm::Twine suf(suffix);

  llvm::SmallString<512> TmpPath;
  std::error_code ec;

  if (ret_fd) {
    int fd = -1;
    ec = fs::createUniqueFile(p + random_pattern + suf, fd, TmpPath,
                              fs::OpenFlags::OF_None,
                              fs::perms::owner_read | fs::perms::owner_write);
    *ret_fd = fd;
  } else {
    ec = fs::createUniqueFile(p + random_pattern + suf, TmpPath,
                              fs::perms::owner_read | fs::perms::owner_write);
  }
  if (ec)
    return -1;
  strncpy(output, TmpPath.c_str(), POCL_MAX_PATHNAME_LENGTH);
  return 0;
}

/****************************************************************************/

#define CHUNK_SIZE (2 * 1024 * 1024)

int pocl_read_file(const char *path, char **content, uint64_t *filesize) {
  assert(content);
  assert(path);
  assert(filesize);
  *content = nullptr;
  *filesize = 0;

  fs::file_t fh;
  int fd;
  std::error_code ec;
  llvm::Twine p(path);

  /* files in /proc return zero size, while
     files in /sys return size larger than actual content size;
     this reads the content sequentially.  */
  ssize_t total_size = 0;
  ssize_t actually_read = 0;
  char *ptr = (char *)malloc(CHUNK_SIZE + 1);
  if (ptr == nullptr) {
    POCL_MSG_ERR("failed to malloc mem for reading %s\n", path);
    return -1;
  }

  ec = fs::openFileForRead(p, fd);
  if (ec) {
    POCL_MSG_ERR("failed to open file %s\n", path);
    goto ERROR;
  }
  fh = fs::convertFDToNativeFile(fd);

  do {
    char *reallocated = (char *)realloc(ptr, (total_size + CHUNK_SIZE + 1));
    if (reallocated == nullptr) {
      POCL_MSG_ERR("failed to realloc mem for reading %s\n", path);
      goto ERROR;
    }
    ptr = reallocated;

    llvm::MutableArrayRef<char> Buf{ptr + total_size, CHUNK_SIZE};
    auto Res = fs::readNativeFile(fh, Buf);
    if (!Res) {
      auto E = Res.takeError();
      POCL_MSG_ERR("failed to read file %s\n", path);
      goto ERROR;
    }
    actually_read = Res.get();
    if (actually_read > 0)
      total_size += actually_read;

  } while (actually_read > 0);

  ec = fs::closeFile(fh);
  if (ec) {
    POCL_MSG_ERR("failed to close file %s\n", path);
    goto ERROR;
  }

  /* add an extra NULL character for strings */
  ptr[total_size] = 0;
  *content = ptr;
  *filesize = (uint64_t)total_size;
  return 0;

ERROR:
  free(ptr);
  return -1;
}

/* Atomic write - with rename() */
static int pocl_write_file2(
    const char *path, // final path
    const char *content, uint64_t count, llvm::Module *Mod,
    bool append,      // append to file (dont truncate)
    bool dont_rename, // don't rename to final path (output to TmpPath)
    llvm::SmallVector<char, 512> &TmpPath, llvm::Twine TmpSuffix) {
  fs::file_t fh;
  int fd;
  std::error_code ec;

  assert(content != nullptr || Mod != nullptr);
  assert(path);
  llvm::Twine FinalPath(path);

  if (append) {
    ec = fs::openFileForWrite(FinalPath, fd, fs::CD_OpenAlways, fs::OF_Append);
    if (ec) {
      POCL_MSG_ERR("failed to open file WR 1 %s\n", path);
      return -1;
    }
  } else {
    ec = fs::createUniqueFile(FinalPath + random_pattern + TmpSuffix, fd,
                              TmpPath, fs::OpenFlags::OF_None,
                              fs::perms::owner_read | fs::perms::owner_write);
    if (ec) {
      POCL_MSG_ERR("failed to open file WR 2 %s\n", path);
      return -1;
    }
  }
  fh = fs::convertFDToNativeFile(fd);

  llvm::raw_fd_ostream FDO{fd, false};
  if (Mod) {
    WriteBitcodeToFile(*Mod, FDO);
  } else {
    FDO.write(content, count);
  }
  FDO.flush();
  if (FDO.has_error()) {
    POCL_MSG_ERR("raw_fd_stream write failed\n");
    return -1;
  }

#if defined(HAVE_FDATASYNC)
  if (fdatasync(fd)) {
    POCL_MSG_ERR("fdatasync() failed\n");
    return -1;
  }
#elif defined(HAVE_FSYNC)
  if (fsync(fd)) {
    POCL_MSG_ERR("fsync() failed\n");
    return -1;
  }
#elif defined(_WIN32)
  FlushFileBuffers(fh);
#endif

  if (fs::closeFile(fh)) {
    POCL_MSG_ERR("failed to close file WR %s\n", path);
    return -2;
  }

  if (append || dont_rename)
    return 0;
  else {
    llvm::Twine TmpP(TmpPath);
    return pocl_rename2(TmpP, FinalPath);
  }
}

int pocl_write_file(const char *path, const char *content, uint64_t count,
                    int append) {
  llvm::SmallVector<char, 512> TmpPath;
  return pocl_write_file2(path, content, count, nullptr, append, false, TmpPath,
                          ".temp");
}

/* write content[count] into a temporary file, and return the tempfile name in
 * output_path */
int pocl_write_tempfile(char *output_path, const char *prefix,
                        const char *suffix, const char *content,
                        uint64_t count) {
  llvm::SmallVector<char, 512> TmpPath;
  int err = pocl_write_file2(prefix, content, count, nullptr, false, true,
                             TmpPath, suffix);
  if (err)
    return err;
  if (TmpPath.size() >= POCL_MAX_PATHNAME_LENGTH) {
    POCL_MSG_ERR("Path name too long \n");
    return -1;
  }
  memcpy(output_path, TmpPath.data(), TmpPath.size());
  output_path[TmpPath.size()] = 0;
  return 0;
}

/* Atomic write of IR - with rename() */
#if 0
int pocl_write_module(void *module, const char* path) {
  llvm::SmallVector<char, 512> TmpPath;
  return pocl_write_file2(path, nullptr, 0, (llvm::Module *)module,
                          false, false,
                          TmpPath, ".temp.bc");
}
#endif

char *pocl_parent_path(char *Path) {
  llvm::StringRef OrigPath(Path);
  llvm::StringRef Result = path::parent_path(OrigPath);
  // The result is just a substring of the Path.
  assert(OrigPath.starts_with(Result));
  Path[Result.size()] = '\0';
  return Path;
}

pocl_file_type pocl_get_file_type(const char *path) {

  switch (fs::get_file_type(path, false)) {
  default:
    return POCL_FS_STATUS_ERROR;
  case fs::file_type::regular_file:
    return POCL_FS_REGULAR;
  case fs::file_type::directory_file:
    return POCL_FS_DIRECTORY;
  case fs::file_type::file_not_found:
    return POCL_FS_NOT_FOUND;
  }
  assert(!"UNREACHABLE!");
  return POCL_FS_STATUS_ERROR;
}

static const fs::directory_iterator DirIteratorEnd;

struct DirIteratorHandle {
  fs::directory_iterator It;
  bool First = true;
};

int pocl_dir_iterator(const char *path, pocl_dir_iter *iter) {
  assert(iter && "iter must not be nullptr!");
  std::error_code EC;
  auto It = fs::directory_iterator(path, EC);
  if (EC)
    return -1;

  auto *DIH = new DirIteratorHandle;
  DIH->It = It;
  iter->handle = static_cast<void *>(DIH);
  return 0;
}

int pocl_dir_next_entry(pocl_dir_iter iter) {
  assert(iter.handle && "Must call pocl_dir_iterator() first!");
  auto *DIH = static_cast<DirIteratorHandle *>(iter.handle);
  if (DIH->It == DirIteratorEnd)
    return 0; // Empty directory or ran out of entries.

  if (DIH->First) {
    DIH->First = false;
    return 1;
  }

  std::error_code EC;
  return DIH->It.increment(EC) != DirIteratorEnd && !EC;
}

const char *pocl_dir_iter_get_path(pocl_dir_iter iter) {
  assert(iter.handle && "Must call pocl_dir_iterator() first!");
  auto *DIH = static_cast<DirIteratorHandle *>(iter.handle);
  assert(!DIH->First && "Must call pocl_dir_next_entry() first!");
  assert(DIH->It != DirIteratorEnd && "Invalid directory iterator.");
  return DIH->It->path().c_str();
}

void pocl_release_dir_iterator(pocl_dir_iter *iter) {
  assert(iter && "Invalid pocl_dir_iter handle!");
  if (!iter->handle)
    return;
  delete static_cast<DirIteratorHandle *>(iter->handle);
  iter->handle = nullptr;
}

/****************************************************************************/
