// Check PoCL filesystem utilities.
//
// Copyright (c) 2024 Henry Linjamäki / Intel Finland Oy
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "config.h"
#include "pocl_file_util.h"

#include <cstdint>
#include <iostream>
#include <string>

#define TEST_ASSERT(expr)                                                      \
  if (!(expr)) {                                                               \
    std::cout << __FILE__ << ":" << __LINE__ << ": "                           \
              << "Assertion failure: '" << #expr << std::endl;                 \
    std::exit(1);                                                              \
  }

static bool endsWith(const std::string &Str, const std::string &End) {
  return Str.rfind(End) <= Str.size();
}

void TestDirIterators() {
  const char *TestDir = SRCDIR "/tests/unit/input";

  pocl_dir_iter DirIter;
  TEST_ASSERT(!pocl_dir_iterator(TestDir, &DirIter));

  bool SawSomeFile = false;
  bool SawSomeDir = false;
  while (pocl_dir_next_entry(DirIter)) {
    std::string PathStr = pocl_dir_iter_get_path(DirIter);
    TEST_ASSERT(PathStr != ".." && PathStr != "." &&
                "'.'. and '..' should not appear");

    auto FileType = pocl_get_file_type(PathStr.c_str());
    TEST_ASSERT(FileType != POCL_FS_STATUS_ERROR &&
                "Unexpected file status error!");

    if (endsWith(PathStr, "somefile.txt")) {
      TEST_ASSERT(FileType == POCL_FS_REGULAR &&
                  "Answer should be POCL_FS_REGULAR for 'somefile.txt'");
      SawSomeFile = true;
    }

    if (endsWith(PathStr, "somedir")) {
      TEST_ASSERT(FileType == POCL_FS_DIRECTORY &&
                  "Answer should be POCL_FS_DIRECTORY for 'somedir'");
      SawSomeDir = true;
    }

    // Check we aren't recursing. The 'oops.txt' is under 'somedir'.
    TEST_ASSERT(!endsWith(PathStr, "oops.txt"))
  }

  TEST_ASSERT(SawSomeFile && "Missed somefile.txt!");
  TEST_ASSERT(SawSomeDir && "Missed somedir!");

  pocl_release_dir_iterator(&DirIter);
}

int main() {
  TestDirIterators();
  return 0;
}
