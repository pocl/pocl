/// Level0/npu graph templates for elementwise operations.
///
/// Copyright (c) 2025 Henry Linjamäki / Intel Finland Oy
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

#include <cassert>
#include <map>
#include <string>

#include "dbk/pocl_dbk_util.h"
#include "npu_dbk.hh"
#include "pocl_tensor_util.h"

static const char *GraphTemplate = R"(
<?xml version="1.0"?>
<net name="pocl_etlwise_op" version="11">
  <layers>
    <layer id="0" name="src0" type="Parameter" version="opset1">
      <data shape="SRC0_DIM0,SRC0_DIM1,SRC0_DIM2," element_type="ELEM_TYPE" />
      <output>
        <port id="0" precision="PREC">
          <dim>SRC0_DIM0</dim>
          <dim>SRC0_DIM1</dim>
          <dim>SRC0_DIM2</dim>
        </port>
      </output>
    </layer>

    <layer id="1" name="src1" type="Parameter" version="opset1">
      <data shape="SRC1_DIM0,SRC1_DIM1,SRC1_DIM2," element_type="ELEM_TYPE" />
      <output>
        <port id="0" precision="PREC">
          <dim>SRC1_DIM0</dim>
          <dim>SRC1_DIM1</dim>
          <dim>SRC1_DIM2</dim>
        </port>
      </output>
    </layer>

    <layer id="2" name="eltwise_op" type="BIN_OP" version="opset1">
      <data destination_type="ELEM_TYPE" />
      <input>
        <port id="0" precision="PREC">
          <dim>SRC0_DIM0</dim>
          <dim>SRC0_DIM1</dim>
          <dim>SRC0_DIM2</dim>
        </port>
        <port id="1" precision="PREC">
          <dim>SRC1_DIM0</dim>
          <dim>SRC1_DIM1</dim>
          <dim>SRC1_DIM2</dim>
        </port>
      </input>
      <output>
        <port id="1" precision="PREC">
          <dim>DST_DIM0</dim>
          <dim>DST_DIM1</dim>
          <dim>DST_DIM2</dim>
        </port>
      </output>
    </layer>

    <layer id="3" name="dst" type="Result" version="opset1">
      <input>
        <port id="0" precision="PREC">
          <dim>DST_DIM0</dim>
          <dim>DST_DIM1</dim>
          <dim>DST_DIM2</dim>
        </port>
      </input>
    </layer>
  </layers>

  <edges>
    <edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
    <edge from-layer="1" from-port="0" to-layer="2" to-port="1" />
    <edge from-layer="2" from-port="1" to-layer="3" to-port="0" />
  </edges>

  <rt_info>
    <Runtime_version value="2023.3.0-13775-ceeafaf64f3-releases/2023/3" />
    <conversion_parameters>
      <is_python_object value="False" />
    </conversion_parameters>
  </rt_info>
</net>
)";

static const char *BuildFlagsTemplate =
    R"RAW(--inputs_precisions="src0:PREC src1:PREC" --inputs_layouts="src0:LAYOUT src1:LAYOUT" --outputs_precisions="dst:PREC" --outputs_layouts="dst:LAYOUT" --config   NPU_PLATFORM="3720" PERFORMANCE_HINT="LATENCY")RAW";

bool instantiateTemplateBINOP(cl_dbk_id_exp DbkId, const void *KernelAttrs,
                              std::string &ModelXMLInstance,
                              std::vector<uint8_t> &ModelBinary,
                              std::string &BuildFlagsInstance) {

  const cl_tensor_desc_exp *Operands[3];
  pocl_dbk_unpack_bin_operands(DbkId, KernelAttrs, Operands);

  for (const auto *Op : Operands)
    assert(pocl_tensor_data_is_contiguous(Op));

  ModelXMLInstance = GraphTemplate;
  BuildFlagsInstance = BuildFlagsTemplate;
  ReplaceMapT ReplaceMap;

  ReplaceMap["BIN_OP"] = pocl::toOpenvinoOpType(DbkId);

  ReplaceMap["SRC0_DIM0"] =
      std::to_string(pocl_tensor_dim_size_or(Operands[0], -3, 1));
  ReplaceMap["SRC0_DIM1"] =
      std::to_string(pocl_tensor_dim_size_or(Operands[0], -2, 1));
  ReplaceMap["SRC0_DIM2"] =
      std::to_string(pocl_tensor_dim_size(Operands[0], -1));

  ReplaceMap["SRC1_DIM0"] =
      std::to_string(pocl_tensor_dim_size_or(Operands[1], -3, 1));
  ReplaceMap["SRC1_DIM1"] =
      std::to_string(pocl_tensor_dim_size_or(Operands[1], -2, 1));
  ReplaceMap["SRC1_DIM2"] =
      std::to_string(pocl_tensor_dim_size(Operands[1], -1));

  ReplaceMap["DST_DIM0"] =
      std::to_string(pocl_tensor_dim_size_or(Operands[2], -3, 1));
  ReplaceMap["DST_DIM1"] =
      std::to_string(pocl_tensor_dim_size_or(Operands[2], -2, 1));
  ReplaceMap["DST_DIM2"] =
      std::to_string(pocl_tensor_dim_size(Operands[2], -1));

  ReplaceMap["PREC"] = dtype2precision(Operands[0]->dtype);
  ReplaceMap["ELEM_TYPE"] = dtype2elemtype(Operands[0]->dtype);
  ReplaceMap["LAYOUT"] = "C";

  replaceAllStringsInMap(ModelXMLInstance, ReplaceMap);
  replaceAllStringsInMap(BuildFlagsInstance, ReplaceMap);
  return true;
}
