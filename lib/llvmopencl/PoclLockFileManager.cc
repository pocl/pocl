/* PoclLockFileManager.cc: a portable lock-file class using llvm::LockFileManager
   with a few tweaks. Mostly used by pocl_cache.h and pocl_file_util.h functions

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

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#else
#include <io.h>
#endif

#include "pocl.h"
#include "pocl_file_util.h"
#include "pocl_cache.h"

#ifdef LLVM_3_2
#include "llvm/Module.h"
#else
#include "llvm/IR/Module.h"
#endif

#include <llvm/Support/LockFileManager.h>
#include <llvm/Support/Errc.h>
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"

#include "llvm/Bitcode/ReaderWriter.h"

#include "PoclLockFileManager.h"

#define RETURN_IF_EC if (ec) return ec.default_error_condition().value()

using namespace llvm;

PoclLockFileManager::PoclLockFileManager(StringRef FileName,
                                         int       immediate) : filename(
        FileName) {

    lfm = new LockFileManager(filename);
    done_filename = filename + ".done";
    // only 2 tries to lock - if the first time it's locked, waitForUnlock()
    switch (lfm->getState()) {
    case LockFileManager::LockFileState::LFS_Owned:
        is_owned = true;
        break;

    case LockFileManager::LockFileState::LFS_Shared:
        if (!immediate) {
            lfm->waitForUnlock();
            delete lfm;
            lfm = new LockFileManager(FileName);
            is_owned =
                (lfm->getState() == LockFileManager::LockFileState::LFS_Owned);
            break;
        }

    case LockFileManager::LockFileState::LFS_Error:
        is_owned = false;
        break;
    }
}

PoclLockFileManager::~PoclLockFileManager() {
    if (lfm)
        delete lfm;
}


bool PoclLockFileManager::file_exists() {
    bool file = sys::fs::exists(this->filename);
    bool donefile = sys::fs::exists(this->done_filename);

    if (file && donefile)
        return true;
    if (!file && !donefile)
        return false;

    remove_file();
    return false;
}

int PoclLockFileManager::done() {
    if (!is_owned)
        return LOCK_ACQUIRE_FAIL;

    if (pocl_exists(this->done_filename.c_str()))
        return 0;

    return touch(this->done_filename);
}

int PoclLockFileManager::read_file(char* content, uint64_t read_bytes) {

    if (!is_owned || !file_exists())
        return LOCK_ACQUIRE_FAIL;

    int fd;
    std::error_code ec;
    Twine p(this->filename);

    ec = sys::fs::openFileForRead(p, fd);
    RETURN_IF_EC;

    if (read(fd, content, read_bytes) < (ssize_t)read_bytes)
        return (-errno || -1);

    return (close(fd) ? (-errno) : 0);
}

int PoclLockFileManager::write_file(const std::string &content,
                                    int                append,
                                    int                dont_rewrite) {
    return write_file(content.c_str(), content.size(), append, dont_rewrite);
}

int PoclLockFileManager::write_file(const char* content,
                                    uint64_t    count,
                                    int         append,
                                    int         dont_rewrite) {
    if (!is_owned)
        return LOCK_ACQUIRE_FAIL;

    int fd;
    std::error_code ec;
    Twine p(this->filename);

    if (file_exists()) {
        if (dont_rewrite) {
            if (!append)
                return 0;
        } else {
            int res = remove_file();
            if (res)
                return res;
        }
    }

    if (append)
        ec = sys::fs::openFileForWrite(p, fd, sys::fs::F_RW | sys::fs::F_Append);
    else
        ec = sys::fs::openFileForWrite(p, fd, sys::fs::F_RW | sys::fs::F_Excl);

    RETURN_IF_EC;

    if (write(fd, content, count) < (ssize_t)count)
        return (-errno || -1);

    if (close(fd))
        return (-errno);

    return done();
}

int PoclLockFileManager::remove_file() {
    if (!is_owned)
        return LOCK_ACQUIRE_FAIL;

    Twine d(this->done_filename);
    Twine p(this->filename);

    std::error_code ec;
    ec = sys::fs::remove(d, true);
    if (!ec)
        ec = sys::fs::remove(p, true);
    RETURN_IF_EC;

    return 0;
}

int PoclLockFileManager::touch_file() {
    if (!is_owned)
        return LOCK_ACQUIRE_FAIL;

    return touch(this->filename);
}

int PoclLockFileManager::write_module(llvm::Module* mod, int dont_rewrite) {
    if (!is_owned)
        return LOCK_ACQUIRE_FAIL;

    Twine p(this->filename);
    std::error_code ec;

    if (file_exists()) {
        if (dont_rewrite)
            return 0;
        else {
            int res = remove_file();
            if (res)
                return res;
        }
    }

    raw_fd_ostream os(this->filename, ec, sys::fs::F_RW | sys::fs::F_Excl);
    RETURN_IF_EC;

    WriteBitcodeToFile(mod, os);
    os.close();
    if (os.has_error())
        return 1;

    return done();
}

int PoclLockFileManager::touch(std::string &s) {
    Twine p(s);
    std::error_code ec = sys::fs::remove(p, true);

    RETURN_IF_EC;

    int fd;
    ec = sys::fs::openFileForWrite(p, fd, sys::fs::F_RW | sys::fs::F_Excl);
    RETURN_IF_EC;

    return (close(fd) ? (-errno) : 0);
}
