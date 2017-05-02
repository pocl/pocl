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
 * @author Michal Babej 2016
 */

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include <llvm/Support/LockFileManager.h>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"

#ifdef LLVM_OLDER_THAN_4_0
#include "llvm/Bitcode/ReaderWriter.h"
#else
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#endif

#include <llvm/Support/Errc.h>

POP_COMPILER_DIAGS

#define RETURN_IF_EC if (ec) return ec.default_error_condition().value()
#define OPEN_FOR_READ ec = sys::fs::openFileForRead(p, fd)
#define OPEN_CREATE ec = sys::fs::openFileForWrite(p, fd, sys::fs::F_RW | sys::fs::F_Excl)
#define OPEN_FOR_APPEND ec = sys::fs::openFileForWrite(p, fd, sys::fs::F_RW | sys::fs::F_Append)

/* #define to disable locking completely */
#undef DISABLE_LOCKMANAGER

using namespace llvm;

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

int
pocl_exists(const char* path) {
    Twine p(path);
    if (sys::fs::exists(p))
        return 1;
    else
        return 0;
}

int
pocl_filesize(const char* path, uint64_t* res) {
    Twine p(path);
    std::error_code ec = sys::fs::file_size(p, *res);
    return ec.default_error_condition().value();
}

int pocl_touch_file(const char* path) {
    Twine p(path);
    std::error_code ec = sys::fs::remove(p, true);
    RETURN_IF_EC;

    int fd;
    OPEN_CREATE;
    RETURN_IF_EC;

    return (close(fd) ? (-errno) : 0);

}

int pocl_rename(const char *oldpath, const char *newpath) {

    Twine op(oldpath);
    Twine np(newpath);
    std::error_code ec = sys::fs::rename(op, np);
    return ec.default_error_condition().value();
}

/****************************************************************************/

int
pocl_read_file(const char* path, char** content, uint64_t *filesize) {
    assert(content);
    assert(path);
    assert(filesize);

    *content = NULL;

    int errcode = pocl_filesize(path, filesize);
    ssize_t fsize = (ssize_t)(*filesize);
    if (!errcode) {
        int fd;
        std::error_code ec;
        Twine p(path);

        OPEN_FOR_READ;
        RETURN_IF_EC;

        // +1 so we can later simply turn it into a C string, if needed
        *content = (char*)malloc(fsize+1);

        size_t rsize = read(fd, *content, fsize);
        (*content)[rsize] = '\0';
        if (rsize < (size_t)fsize) {
            errcode = errno ? -errno : -1;
            close(fd);
        } else {
            if (close(fd))
                errcode = errno ? -errno : -1;
        }
    }
    return errcode;
}


/* Atomic write - with rename()
 * NOTE: still requires a pocl lock before attempting it - because the tempfile
 * name is not random */
int pocl_write_file(const char *path, const char *content, uint64_t count,
                    int append, int dont_rewrite) {
    int fd;
    std::error_code ec;
    std::string TmpPath(path);
    TmpPath.append(".tmp");

    assert(path);
    assert(content);

    if (pocl_exists(path)) {
        if (dont_rewrite) {
            if (!append)
                return 0;
        } else {
            int res = pocl_remove(path);
            if (res)
                return res;
        }
    }

    if (append) {
        Twine p(path);
        OPEN_FOR_APPEND;
    } else {
        Twine p(TmpPath);
        OPEN_CREATE;
    }

    RETURN_IF_EC;

    if (write(fd, content, (ssize_t)count) < (ssize_t)count)
        return errno ? -errno : -1;

    if (close(fd))
      return -errno;

    if (append)
      return 0;
    else
      return pocl_rename(TmpPath.c_str(), path);
}




/* Atomic write of IR - with rename()
 * NOTE: still requires a pocl lock before attempting it - because the tempfile
 * name is not random */
int pocl_write_module(void *module, const char* path, int dont_rewrite) {

    assert(module);
    assert(path);

    Twine p(path);
    std::error_code ec;

    if (pocl_exists(path)) {
        if (dont_rewrite)
            return 0;
        else {
            int res = pocl_remove(path);
            if (res)
                return res;
        }
    }
    /* To avoid corrupted .bc files, create a tmp file first and
       then rename it */
    std::string TmpPath(path);
    TmpPath.append(".tmp");

    raw_fd_ostream os(TmpPath, ec, sys::fs::F_RW | sys::fs::F_Excl);
    RETURN_IF_EC;

    WriteBitcodeToFile((llvm::Module*)module, os);
    os.close();
    if (os.has_error())
      return 1;

    return pocl_rename(TmpPath.c_str(), path);
}



/****************************************************************************/

static void* acquire_lock_internal(const char* path, int shared) {
#ifdef DISABLE_LOCKMANAGER
    /* Can't return value that compares equal to NULL */
    return (void*)4096;
#else
    assert(path);
    LockFileManager *lfm = new LockFileManager(path);

    switch (lfm->getState()) {
    case LockFileManager::LockFileState::LFS_Owned:
        return (void*)lfm;

    case LockFileManager::LockFileState::LFS_Shared:
        if (shared)
            return (void*)lfm;
        lfm->waitForUnlock();
        delete lfm;
        lfm = new LockFileManager(path);
        if (lfm->getState() == LockFileManager::LockFileState::LFS_Owned)
            return (void*)lfm;
        else
          {
            delete lfm;
            return NULL;
          }

    case LockFileManager::LockFileState::LFS_Error:
        return NULL;
    }
    return NULL;
#endif
}


void* acquire_lock(const char *path, int shared) {
    return acquire_lock_internal(path, shared);
}


void release_lock(void* lock) {
#ifdef DISABLE_LOCKMANAGER
    return;
#else
    if (!lock)
        return;
    LockFileManager *l = (LockFileManager*)lock;
    delete l;
#endif
}
