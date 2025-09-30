/* spirv_fix_atomic_compare_exchange: remove an unnecessary bitcast in the input
   SPIR-V that causes SPIR-V validation to fail with an error.

   Copyright (c) 2024 Michal Babej / Intel Finland Oy

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

#include "poclu.h"

#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <cassert>
#include <iostream>

#include "spirv_parser.hh"

using namespace std;

int main(int Argc, char *Argv[]) {
  if (Argc != 3) {
    std::cerr << "USAGE: $0 input output\n";
    return EXIT_FAILURE;
  }

  std::string InputName(Argv[1]);
  std::string OutputName(Argv[2]);
  size_t InputSize, OutputSize;
  char *InputChars = poclu_read_binfile(InputName.c_str(), &InputSize);
  if (InputChars == nullptr || InputSize == 0) {
    std::cerr << "Can't read input file: " << InputName << "\n";
    return EXIT_FAILURE;
  }
  std::vector<int32_t> InStream(InputSize / 4 + 1);
  memcpy(InStream.data(), InputChars, InputSize);
  free(InputChars);
  std::vector<uint8_t> OutStream;
  if (!SPIRVParser::applyAtomicCmpXchgWorkaround(InStream.data(), InputSize / 4,
                                                 OutStream))
    return EXIT_FAILURE;

  int Ret = poclu_write_binfile(OutputName.c_str(), (char *)OutStream.data(),
                                OutStream.size());

  return Ret ? EXIT_FAILURE : EXIT_SUCCESS;
}
