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

#include <llvm/Support/LockFileManager.h>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"

#include "llvm/Bitcode/ReaderWriter.h"

/* namespace of OpenFlags enum (F_Binary, F_Excl etc) */
#if defined(LLVM_3_2) || defined(LLVM_3_3)
#define OPEN_FLAGS_ENUM raw_fd_ostream
#else
#define OPEN_FLAGS_ENUM sys::fs
#endif


/* Older llvms:
 * llvm::error_code instead of std::error_code
 * "file existed" is an output argument, instead of "ignore if file existed"
 * different OpenFlags (no F_Binary)
 * File locking disabled: llvm::LockManager class doesn't exist in LLVM < 3.5
 */
#if LLVM_OLDER_THAN_3_5

#define DISABLE_LOCKMANAGER

#include <llvm/Support/system_error.h>
#define STD_ERROR_CODE llvm::error_code
static bool existed;
#define EXIST_ARG existed
#define DEFAULT_OPEN_FLAGS OPEN_FLAGS_ENUM::F_Binary

#else

#include <llvm/Support/Errc.h>
#define STD_ERROR_CODE std::error_code
#define EXIST_ARG true
#define DEFAULT_OPEN_FLAGS OPEN_FLAGS_ENUM::F_RW

#endif

#define RETURN_IF_EC if (ec) return ec.default_error_condition().value()

/* no openFile* functions in sys::fs before llvm 3.4, so fallback to open */
#if defined(LLVM_3_2) || defined(LLVM_3_3)
    #define OPEN_FOR_READ fd = open(path, O_RDONLY)
    #define OPEN_CREATE fd = open(path, O_CREAT | O_EXCL, S_IRUSR | S_IWUSR)
    #define OPEN_FOR_APPEND fd = open(path, O_WRONLY | O_APPEND)
    #define RETURN_IF_ERRNO if (fd < 0) return errno;
#else
    #define OPEN_FOR_READ ec = sys::fs::openFileForRead(p, fd)
    #define OPEN_CREATE ec = sys::fs::openFileForWrite(p, fd, DEFAULT_OPEN_FLAGS | sys::fs::F_Excl)
    #define OPEN_FOR_APPEND ec = sys::fs::openFileForWrite(p, fd, DEFAULT_OPEN_FLAGS | sys::fs::F_Append)
    #define RETURN_IF_ERRNO RETURN_IF_EC
#endif



using namespace llvm;

/*****************************************************************************/

int
pocl_rm_rf(const char* path) {
    STD_ERROR_CODE ec;
    SmallString<128> DirNative;

    sys::path::native(Twine(path), DirNative);

    std::vector<std::string> FileSet, DirSet;

    for (sys::fs::recursive_directory_iterator Dir(DirNative.str(), ec), DirEnd;
         Dir != DirEnd && !ec; Dir.increment(ec)) {
        Twine p = Dir->path();
        std::string s = p.str();
#if defined(LLVM_3_2) || defined(LLVM_3_3)
        sys::fs::file_status result;
        sys::fs::status(p, result);
        if (sys::fs::is_directory(result)) {
#else
        if (sys::fs::is_directory(p)) {
#endif
            DirSet.push_back(s);
        } else
            FileSet.push_back(s);
    }
    RETURN_IF_EC;

    std::vector<std::string>::iterator it;
    for (it = FileSet.begin(); it != FileSet.end(); ++it) {
        ec = sys::fs::remove(*it, EXIST_ARG);
        RETURN_IF_EC;
    }

    std::sort(DirSet.begin(), DirSet.end());
    std::vector<std::string>::reverse_iterator it2;
    for (it2 = DirSet.rbegin(); it2 != DirSet.rend(); ++it2) {
        ec = sys::fs::remove(*it2, EXIST_ARG);
        RETURN_IF_EC;
    }

    return 0;
}


int
pocl_mkdir_p(const char* path) {
    Twine p(path);
    STD_ERROR_CODE ec = sys::fs::create_directories(p, EXIST_ARG);
    return ec.default_error_condition().value();
}

int
pocl_remove(const char* path) {
    Twine p(path);
    STD_ERROR_CODE ec = sys::fs::remove(p, EXIST_ARG);
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
    STD_ERROR_CODE ec = sys::fs::file_size(p, *res);
    return ec.default_error_condition().value();
}

int pocl_touch_file(const char* path) {
    Twine p(path);
    STD_ERROR_CODE ec = sys::fs::remove(p, EXIST_ARG);
    RETURN_IF_EC;

    int fd;
    OPEN_CREATE;
    RETURN_IF_ERRNO;

    return (close(fd) ? (-errno) : 0);

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
        // +1 so we can later simply turn it into a C string, if needed
        *content = (char*)malloc(fsize+1);

        int fd;
        STD_ERROR_CODE ec;
        Twine p(path);

        OPEN_FOR_READ;
        RETURN_IF_ERRNO;

        if (read(fd, *content, fsize) < fsize)
            return (-errno || -1);

        return (close(fd) ? (-errno) : 0);

    }
    return errcode;
}



int pocl_write_file(const char *path, const char* content,
                                    uint64_t    count,
                                    int         append,
                                    int         dont_rewrite) {
    int fd;
    STD_ERROR_CODE ec;
    Twine p(path);

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

    if (append)
        OPEN_FOR_APPEND;
    else
        OPEN_CREATE;

    RETURN_IF_ERRNO;

    if (write(fd, content, (ssize_t)count) < (ssize_t)count)
        return (-errno || -1);

    return (close(fd) ? (-errno) : 0);
}





int pocl_write_module(void *module, const char* path, int dont_rewrite) {

    assert(module);
    assert(path);

    Twine p(path);
#ifdef LLVM_OLDER_THAN_3_6
    std::string ec;
#else
    STD_ERROR_CODE ec;
#endif

    if (pocl_exists(path)) {
        if (dont_rewrite)
            return 0;
        else {
            int res = pocl_remove(path);
            if (res)
                return res;
        }
    }

    raw_fd_ostream os(path, ec, DEFAULT_OPEN_FLAGS | OPEN_FLAGS_ENUM::F_Excl);
#ifdef LLVM_OLDER_THAN_3_6
    if (!ec.empty())
        return 2;
#else
    RETURN_IF_EC;
#endif

    WriteBitcodeToFile((llvm::Module*)module, os);
    os.close();
    return (os.has_error() ? 1 : 0);

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
