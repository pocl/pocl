/// Level0/npu graph template for convert_exp
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

#include "npu_dbk.hh"
#include "pocl_tensor_util.h"

static const char *GraphTemplate = R"(
<?xml version="1.0"?>
<net name="Model0" version="11">
  <layers>
    <layer id="0" name="Parameter_3" type="Parameter" version="opset1">
      <data shape="DATA_DIM0,DATA_DIM1" element_type="DATA_ELEM_TYPE" />
      <output>
        <port id="0" precision="DATA_PREC" names="Parameter_3">
          <dim>DATA_DIM0</dim>
          <dim>DATA_DIM1</dim>
        </port>
      </output>
    </layer>
    <layer id="2" name="Parameter_1" type="Parameter" version="opset1">
      <data shape="ROWS_DIM0,ROWS_DIM1" element_type="f32" />
      <output>
        <port id="0" precision="FP32" names="Parameter_1">
          <dim>ROWS_DIM0</dim>
          <dim>ROWS_DIM1</dim>
        </port>
      </output>
    </layer>
    <layer id="1" name="Parameter_2" type="Parameter" version="opset1">
      <data shape="INDICES_DIM" element_type="i64" />
      <output>
        <port id="0" precision="I64" names="Parameter_2">
          <dim>INDICES_DIM</dim>
        </port>
      </output>
    </layer>
    <layer id="3" name="ConvertLike_6" type="ConvertLike" version="opset1">
      <input>
        <port id="0" precision="FP32">
          <dim>ROWS_DIM0</dim>
          <dim>ROWS_DIM1</dim>
        </port>
        <port id="1" precision="DATA_PREC">
          <dim>DATA_DIM0</dim>
          <dim>DATA_DIM1</dim>
        </port>
      </input>
      <output>
        <port id="2" precision="DATA_PREC">
          <dim>ROWS_DIM0</dim>
          <dim>ROWS_DIM1</dim>
        </port>
      </output>
    </layer>
    <layer id="4" name="Constant_5" type="Const" version="opset1">
      <data element_type="i64" shape="1" offset="0" size="8" />
      <output>
        <port id="0" precision="I64">
          <dim>1</dim>
        </port>
      </output>
    </layer>
    <layer id="5" name="ScatterUpdate_7" type="ScatterUpdate" version="opset3">
      <input>
        <port id="0" precision="DATA_PREC">
          <dim>DATA_DIM0</dim>
          <dim>DATA_DIM1</dim>
        </port>
        <port id="1" precision="I64">
          <dim>INDICES_DIM</dim>
        </port>
        <port id="2" precision="DATA_PREC">
          <dim>ROWS_DIM0</dim>
          <dim>ROWS_DIM1</dim>
        </port>
        <port id="3" precision="I64">
          <dim>1</dim>
        </port>
      </input>
      <output>
        <port id="4" precision="DATA_PREC" names="Result_8">
          <dim>DATA_DIM0</dim>
          <dim>DATA_DIM1</dim>
        </port>
      </output>
    </layer>
    <layer id="6" name="Result_8" type="Result" version="opset1" output_names="Result_8">
      <input>
        <port id="0" precision="DATA_PREC">
          <dim>DATA_DIM0</dim>
          <dim>DATA_DIM1</dim>
        </port>
      </input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="3" to-port="1" />
    <edge from-layer="0" from-port="0" to-layer="5" to-port="0" />
    <edge from-layer="1" from-port="0" to-layer="5" to-port="1" />
    <edge from-layer="2" from-port="0" to-layer="3" to-port="0" />
    <edge from-layer="3" from-port="2" to-layer="5" to-port="2" />
    <edge from-layer="4" from-port="0" to-layer="5" to-port="3" />
    <edge from-layer="5" from-port="4" to-layer="6" to-port="0" />
  </edges>
  <rt_info />
</net>
)";

static const char *BuildFlagsTemplate =
    R"RAW(--inputs_precisions="0:DATA_PREC 1:ROWS_PREC 2:I64" --inputs_layouts="0:DATA_LAYOUT 1:ROWS_LAYOUT 2:INDICES_LAYOUT" --outputs_precisions="0:DATA_PREC" --outputs_layouts="0:DATA_LAYOUT" --config NPU_BATCH_MODE="AUTO" NPU_PLATFORM="3720")RAW";

bool instantiateTemplateSET_ROWS(const void *KernelAttrs,
                                 std::string &ModelXMLInstance,
                                 std::vector<uint8_t> &ModelBinary,
                                 std::string &BuildFlagsInstance) {

  auto *Attrs =
      static_cast<const cl_dbk_attributes_set_rows_exp *>(KernelAttrs);
  ModelXMLInstance = GraphTemplate;
  BuildFlagsInstance = BuildFlagsTemplate;
  ReplaceMapT ReplaceMap;

  assert(pocl_tensor_data_is_contiguous(&Attrs->data_in));
  assert(pocl_tensor_data_is_contiguous(&Attrs->rows));
  assert(pocl_tensor_data_is_contiguous(&Attrs->indices));
  assert(pocl_tensor_data_is_contiguous(&Attrs->data_out));

  ReplaceMap["ROWS_DIM0"] =
      std::to_string(pocl_tensor_dim_size(&Attrs->rows, -2));
  ReplaceMap["ROWS_DIM1"] =
      std::to_string(pocl_tensor_dim_size(&Attrs->rows, -1));

  ReplaceMap["INDICES_DIM"] =
      std::to_string(pocl_tensor_dim_size(&Attrs->indices, -1));

  ReplaceMap["DATA_DIM0"] =
      std::to_string(pocl_tensor_dim_size(&Attrs->data_in, -2));
  ReplaceMap["DATA_DIM1"] =
      std::to_string(pocl_tensor_dim_size(&Attrs->data_in, -1));

  ReplaceMap["ROWS_PREC"] = dtype2precision(Attrs->rows.dtype);
  ReplaceMap["DATA_PREC"] = dtype2precision(Attrs->data_in.dtype);
  ReplaceMap["DATA_ELEM_TYPE"] = dtype2elemtype(Attrs->data_in.dtype);

  ReplaceMap["ROWS_LAYOUT"] = "NC";
  ReplaceMap["INDICES_LAYOUT"] = "C";
  ReplaceMap["DATA_LAYOUT"] = "NC";

  replaceAllStringsInMap(ModelXMLInstance, ReplaceMap);
  replaceAllStringsInMap(BuildFlagsInstance, ReplaceMap);

  ModelBinary.resize(8, 0); // Set value (= 0) for 'Constant_5' layer.
  return true;
}
