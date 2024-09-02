/* spirv_parser.hh - interface to a light parser for SPIR-V binaries. Only
 * parses enough to get kernel function signatures and their argument
 * metadata (types, sizes, AS..)
 *
 * Copyright (c) 2021-22 CHIP-SPV developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef HIP_COMMON_HH
#define HIP_COMMON_HH

#include "pocl_export.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace SPIRVParser {

#ifndef HAVE_SIZE_T_3
#define HAVE_SIZE_T_3
typedef struct
{
  size_t size[3];
} size_t_3;
#endif

enum class OCLType : unsigned {
  POD,
  Pointer,
  Image,
  Sampler,
  Opaque,
};

enum class OCLSpace : unsigned {
  Private = 0,
  Global = 1,
  Constant = 2,
  Local = 3,
  Unknown = 1000
};

typedef struct {
  int CPacked : 1;
  int Restrict : 1;
  int Volatile : 1;
  int Constant : 1;
  int ReadableImg : 1;
  int WriteableImg : 1;
} ArgAttrs;

struct OCLArgTypeInfo {
  OCLType Type;
  OCLSpace Space;
  size_t Size;
  std::string Name;
  ArgAttrs Attrs;
  uint32_t Alignment;
  // this SPIRV type ID; for parsing only,
  // but it has to be stored
  int32_t TypeID;
};

struct OCLFuncInfo {
  std::vector<OCLArgTypeInfo> ArgTypeInfo;
  OCLArgTypeInfo RetTypeInfo;
  size_t_3 ReqLocalSize;
  size_t_3 LocalSizeHint;
  size_t_3 VecTypeHint;
};

typedef std::map<std::string, std::shared_ptr<OCLFuncInfo>>
    OpenCLFunctionInfoMap;

POCL_EXPORT
bool parseSPIRV(const int32_t *Stream, size_t NumWords,
                OpenCLFunctionInfoMap &FuncInfoMap);

POCL_EXPORT
void applyAtomicCmpXchgWorkaround(const int32_t *InStream, size_t NumWords,
                                  std::vector<uint8_t> &OutStream);

} // namespace SPIRVParser

#endif
