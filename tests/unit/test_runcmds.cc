// Check PoCL pocl_run_command_capture_output.
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
#include "pocl_util.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#define TEST_ASSERT(expr)                                                      \
  if (!(expr)) {                                                               \
    std::cout << __FILE__ << ":" << __LINE__ << ": "                           \
              << "Assertion failure: '" << #expr << std::endl;                 \
    std::exit(1);                                                              \
  }

int main() {
  std::vector<char> CaptureBuffer(20, 'A');
  const char *ArgList[] = {CMAKE_COMMAND, "-E", "echo", "Hello, World!",
                           nullptr};

  size_t CaptureSize = CaptureBuffer.size();
  int ExitCode = pocl_run_command_capture_output(CaptureBuffer.data(),
                                                 &CaptureSize, ArgList);

  // CMake commands return 0 on success, not EXIT_SUCCESS.
  TEST_ASSERT(ExitCode == 0);
  // The size of the message + newline from echo.
  TEST_ASSERT(CaptureSize == 14);

  std::string ResultStr(CaptureBuffer.data(), CaptureBuffer.size());
  TEST_ASSERT(ResultStr.find("Hello, World!\n") == 0);

  return 0;
}
