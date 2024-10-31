/* pocl_spirv_utils.hh - a collection of functions useful when using SPIR-V
 * internally in PoCL.
 *
 * Copyright (c) 2022-2024 Michal Babej / Intel Finland Oy
 * Copyright (c) 2024 Robin Bijl / Tampere university
 *
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

#ifndef _POCL_SPIRV_UTILS_H_
#define _POCL_SPIRV_UTILS_H_

#include "spirv_parser.hh"
#include "pocl_cl.h"

/// Maps OCLFuncInfo objects to PoCL's kernel metadata struct.
///
/// \note Not all metadata will be populated, only that which is present in the
/// funcInfo argument.
/// \param funcInfo [in] can be the result of parsing SPIR-V with parseSPIRV.
/// \param kernelName [in] can be the result of parsing SPIR-V with parseSPIRV.
/// \param numDevices [in] used to allocate enough memory.
/// \param kernelMetadata [out] stores resulting metadata.
POCL_EXPORT
void mapToPoCLMetadata(SPIRVParser::OCLFuncInfo *funcInfo, const std::string& kernelName,
                       size_t numDevices,
                       pocl_kernel_metadata_t *kernelMetadata);

/// Overloaded version of mapToPoCLMetdata what is intended for usage when
/// iterating over an OpenCLFunctionInfoMap object.
POCL_EXPORT
void mapToPoCLMetadata(
    std::pair<const std::string, std::shared_ptr<SPIRVParser::OCLFuncInfo>> &pair,
size_t numDevices, pocl_kernel_metadata_t *kernelMetadata);

#endif //_POCL_SPIRV_UTILS_H_
