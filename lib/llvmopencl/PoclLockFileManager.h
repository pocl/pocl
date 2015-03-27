/* PoclLockFileManager.h: a portable lock-file class using llvm::LockFileManager
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



#ifndef _POCL_KERNEL_CACHE_H
#define _POCL_KERNEL_CACHE_H

#pragma GCC visibility push(hidden)

class PoclLockFileManager {
private:
    llvm::LockFileManager *lfm;
    bool is_owned;
    std::string filename, done_filename;
    int touch(std::string &s);

public:

    PoclLockFileManager(llvm::StringRef FileName, int immediate=0);
    ~PoclLockFileManager();

    operator bool() const { return is_owned; };

    bool file_exists();

    int read_file(char* content, uint64_t read_bytes);
    int write_file(const std::string &content, int append, int dont_rewrite);
    int write_file(const char* content,
                   uint64_t    count,
                   int         append,
                   int         dont_rewrite);
    int write_module(llvm::Module* mod, int dont_rewrite);
    int remove_file();
    int touch_file();
    int done();
};


#pragma GCC visibility pop

#endif
