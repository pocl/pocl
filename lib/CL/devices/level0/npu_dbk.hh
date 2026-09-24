/// Copyright (c) 2025 Michal Babej, Henry Linjamäki / Intel Finland Oy
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.

#ifndef NPU_DBK_HH
#define NPU_DBK_HH

#include "pocl_cl.h"

#include <map>
#include <string>
#include <vector>

typedef std::map<const char *, std::string> ReplaceMapT;

namespace pocl {
std::string toOpenvinoOpType(cl_dbk_id_exp);
}

void replaceAllStringsInMap(std::string &Buffer, ReplaceMapT RepMap);

const char *dtype2precision(cl_tensor_datatype_exp dtype);

const char *dtype2elemtype(cl_tensor_datatype_exp dtype);

const char *layout2str(const cl_tensor_desc_exp &tensor);

bool instantiateTemplateMATMUL(cl_dbk_id_exp DbkId, const void *KernelAttrs,
                               std::string &ModelXMLInstance,
                               std::vector<uint8_t> &ModelBinary,
                               std::string &BuildFlagsInstance);

bool instantiateTemplateGEMM(cl_dbk_id_exp DbkId, const void *KernelAttrs,
                             std::string &ModelXMLInstance,
                             std::vector<uint8_t> &ModelBinary,
                             std::string &BuildFlagsInstance);

bool instantiateTemplateCONVERT(cl_dbk_id_exp DbkId, const void *KernelAttrs,
                                std::string &ModelXMLInstance,
                                std::vector<uint8_t> &ModelBinary,
                                std::string &BuildFlagsInstance);

bool instantiateTemplateSET_ROWS(cl_dbk_id_exp DbkId, const void *KernelAttrs,
                                 std::string &ModelXMLInstance,
                                 std::vector<uint8_t> &ModelBinary,
                                 std::string &BuildFlagsInstance);

bool instantiateTemplateBINOP(cl_dbk_id_exp DbkId, const void *KernelAttrs,
                              std::string &ModelXMLInstance,
                              std::vector<uint8_t> &ModelBinary,
                              std::string &BuildFlagsInstance);

bool instantiateTemplateRMSNORM(cl_dbk_id_exp DbkId, const void *KernelAttrs,
                                std::string &ModelXMLInstance,
                                std::vector<uint8_t> &ModelBinary,
                                std::string &BuildFlagsInstance);

#endif // NPU_DBK_HH
