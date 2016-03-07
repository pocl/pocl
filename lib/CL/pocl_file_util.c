#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#else
#include <io.h>
#endif

#include <assert.h>
#include <stdio.h>
#include <errno.h>

#include "pocl.h"
#include "pocl_file_util.h"

/* #define to disable locking completely */
#undef DISABLE_LOCKMANAGER

// using namespace llvm;

/*****************************************************************************/

int
pocl_rm_rf(const char* path) {
    return -1;

    /* std::error_code ec; */
    /* SmallString<128> DirNative; */

    /* sys::path::native(Twine(path), DirNative); */

    /* std::vector<std::string> FileSet, DirSet; */

    /* for (sys::fs::recursive_directory_iterator Dir(DirNative.str(), ec), DirEnd; */
    /*      Dir != DirEnd && !ec; Dir.increment(ec)) { */
    /*     Twine p = Dir->path(); */
    /*     std::string s = p.str(); */
    /*     if (sys::fs::is_directory(p)) { */
    /*         DirSet.push_back(s); */
    /*     } else */
    /*         FileSet.push_back(s); */
    /* } */
    /* RETURN_IF_EC; */

    /* std::vector<std::string>::iterator it; */
    /* for (it = FileSet.begin(); it != FileSet.end(); ++it) { */
    /*     ec = sys::fs::remove(*it, true); */
    /*     RETURN_IF_EC; */
    /* } */

    /* std::sort(DirSet.begin(), DirSet.end()); */
    /* std::vector<std::string>::reverse_iterator it2; */
    /* for (it2 = DirSet.rbegin(); it2 != DirSet.rend(); ++it2) { */
    /*     ec = sys::fs::remove(*it2, true); */
    /*     RETURN_IF_EC; */
    /* } */

    /* return 0; */
}


int
pocl_mkdir_p(const char* path) {
    int error = mkdir(path,0);
    if (errno == EEXIST)
        return 0;
    return error;
}

int
pocl_remove(const char* path) {
    return remove(path);
}

int
pocl_exists(const char* path) {
    FILE *f;
    if (f = fopen(path, "r")){
        fclose(f);
        return 1;
    }
    return 0;
}

int
pocl_filesize(const char* path, uint64_t* res) {
    FILE *f = fopen(path, "r");
    if (f == NULL)
        return -1;

    fseek(f, 0, SEEK_END);
    *res = ftell(f);
    
    fclose(f);
    return 0;
}

int pocl_touch_file(const char* path) {
    FILE *f;
    if (f = fopen(path, "w")){
        fclose(f);
        return 0;
    }
    return -1;
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
        FILE *f = fopen(path, "r");

        *content = (char*)malloc(fsize+1);

        size_t rsize = fread(*content, 1, fsize, f);
        (*content)[rsize] = '\0';
        if (rsize < (size_t)fsize) {
            errcode = -1;
            fclose(f);
        } else {
            if (fclose(f))
                errcode = -1;
        }
    }
    return errcode;
}



int pocl_write_file(const char *path, const char* content,
                                    uint64_t    count,
                                    int         append,
                                    int         dont_rewrite) {
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

    FILE *f;
    if (append)
        f = fopen(path, "a");
    else
        f = fopen(path, "w");

    if (f == NULL)
        return -1;

    if (fwrite(content, 1, (ssize_t)count, f) < (ssize_t)count)
        return -1;

    
    return fclose(f);
}


/****************************************************************************/

static void* acquire_lock_internal(const char* path, int shared) {
    /* Can't return value that compares equal to NULL */
    return (void*)4096;
}


void* acquire_lock(const char *path, int shared) {
    return acquire_lock_internal(path, shared);
}


void release_lock(void* lock) {
    return;
}
