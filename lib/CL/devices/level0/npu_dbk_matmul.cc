/// Level0/npu graph template for matmul_exp
///
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

#include <cassert>
#include <map>
#include <string>

#include "CL/opencl.h"

#include "npu_dbk.hh"

// Template for matmul(T, T) -> T.
const char *MATMUL_T_XML_Template = R"(
<?xml version="1.0"?>
<net name="TensorFlow_Frontend_IR" version="11">
  <layers>
    <layer id="1" name="x1" type="Parameter" version="opset1">
      <data shape="SHAPE_A_ROWS,SHAPE_A_COLS" element_type="OUTPUT_ELEM_TYPE" />
      <output>
        <port id="0" precision="OUTPUT_PREC" names="x1">
          <dim>SHAPE_A_ROWS</dim>
          <dim>SHAPE_A_COLS</dim>
        </port>
      </output>
    </layer>

    <layer id="0" name="x2" type="Parameter" version="opset1">
      <data shape="SHAPE_B_ROWS,SHAPE_B_COLS" element_type="OUTPUT_ELEM_TYPE" />
      <output>
        <port id="0" precision="OUTPUT_PREC" names="x2">
          <dim>SHAPE_B_ROWS</dim>
          <dim>SHAPE_B_COLS</dim>
        </port>
      </output>
    </layer>

    <layer id="2" name="model/dot/MatMul" type="MatMul" version="opset1">
      <data transpose_a="TRANSPOSE_A" transpose_b="TRANSPOSE_B" />
      <input>
        <port id="0" precision="OUTPUT_PREC">
          <dim>SHAPE_A_ROWS</dim>
          <dim>SHAPE_A_COLS</dim>
        </port>
        <port id="1" precision="OUTPUT_PREC">
          <dim>SHAPE_B_ROWS</dim>
          <dim>SHAPE_B_COLS</dim>
        </port>
      </input>
      <output>
        <port id="2" precision="OUTPUT_PREC" names="dot">
          <dim>SHAPE_C_ROWS</dim>
          <dim>SHAPE_C_COLS</dim>
        </port>
      </output>
    </layer>

    <layer id="3" name="dot" type="Result" version="opset1">
      <input>
        <port id="0" precision="OUTPUT_PREC">
          <dim>SHAPE_C_ROWS</dim>
          <dim>SHAPE_C_COLS</dim>
        </port>
      </input>
    </layer>
  </layers>

  <edges>
    <edge from-layer="0" from-port="0" to-layer="2" to-port="1" />
    <edge from-layer="1" from-port="0" to-layer="2" to-port="0" />
    <edge from-layer="2" from-port="2" to-layer="3" to-port="0" />
  </edges>

  <rt_info>
    <Runtime_version value="2023.3.0-13775-ceeafaf64f3-releases/2023/3" />
    <conversion_parameters>
      <is_python_object value="False" />
    </conversion_parameters>
  </rt_info>
</net>
)";

// Template for matmul(T, U) -> V.
const char *MATMUL_T_U_V_XML_Template = R"(
<?xml version="1.0"?>
<net name="TensorFlow_Frontend_IR" version="11">
  <layers>
    <layer id="1" name="x1" type="Parameter" version="opset1">
      <data shape="SHAPE_A_ROWS,SHAPE_A_COLS" element_type="INPUT_A_ELEM_TYPE" />
      <output>
        <port id="0" precision="INPUT_A_PREC" names="x1">
          <dim>SHAPE_A_ROWS</dim>
          <dim>SHAPE_A_COLS</dim>
        </port>
      </output>
    </layer>

    <layer id="0" name="x2" type="Parameter" version="opset1">
      <data shape="SHAPE_B_ROWS,SHAPE_B_COLS" element_type="INPUT_B_ELEM_TYPE" />
      <output>
        <port id="0" precision="INPUT_B_PREC" names="x2">
          <dim>SHAPE_B_ROWS</dim>
          <dim>SHAPE_B_COLS</dim>
        </port>
      </output>
    </layer>

    <layer id="2" name="convert0" type="Convert" version="opset1">
      <data destination_type="OUTPUT_ELEM_TYPE" />
      <input>
        <port id="0" precision="INPUT_PREC">
          <dim>SHAPE_A_ROWS</dim>
          <dim>SHAPE_A_COLS</dim>
        </port>
      </input>
      <output>
        <port id="1" precision="OUTPUT_PREC">
          <dim>SHAPE_A_ROWS</dim>
          <dim>SHAPE_A_COLS</dim>
        </port>
      </output>
    </layer>

    <layer id="3" name="convert1" type="Convert" version="opset1">
      <data destination_type="OUTPUT_ELEM_TYPE" />
      <input>
        <port id="0" precision="INPUT_PREC">
          <dim>SHAPE_B_ROWS</dim>
          <dim>SHAPE_B_COLS</dim>
        </port>
      </input>
      <output>
        <port id="1" precision="OUTPUT_PREC">
          <dim>SHAPE_B_ROWS</dim>
          <dim>SHAPE_B_COLS</dim>
        </port>
      </output>
    </layer>

    <layer id="4" name="model/dot/MatMul" type="MatMul" version="opset1">
      <data transpose_a="TRANSPOSE_A" transpose_b="TRANSPOSE_B" />
      <input>
        <port id="0" precision="OUTPUT_PREC">
          <dim>SHAPE_A_ROWS</dim>
          <dim>SHAPE_A_COLS</dim>
        </port>
        <port id="1" precision="OUTPUT_PREC">
          <dim>SHAPE_B_ROWS</dim>
          <dim>SHAPE_B_COLS</dim>
        </port>
      </input>
      <output>
        <port id="2" precision="OUTPUT_PREC" names="dot">
          <dim>SHAPE_C_ROWS</dim>
          <dim>SHAPE_C_COLS</dim>
        </port>
      </output>
    </layer>

    <layer id="5" name="dot" type="Result" version="opset1">
      <input>
        <port id="0" precision="OUTPUT_PREC">
          <dim>SHAPE_C_ROWS</dim>
          <dim>SHAPE_C_COLS</dim>
        </port>
      </input>
    </layer>
  </layers>

  <edges>
    <edge from-layer="0" from-port="0" to-layer="3" to-port="0" />
    <edge from-layer="1" from-port="0" to-layer="2" to-port="0" />
    <edge from-layer="2" from-port="1" to-layer="4" to-port="0" />
    <edge from-layer="3" from-port="1" to-layer="4" to-port="1" />
    <edge from-layer="4" from-port="2" to-layer="5" to-port="0" />
  </edges>

  <rt_info>
    <Runtime_version value="2023.3.0-13775-ceeafaf64f3-releases/2023/3" />
    <conversion_parameters>
      <is_python_object value="False" />
    </conversion_parameters>
  </rt_info>
</net>
)";

const char *MATMUL_Flags_Template =
    R"RAW(--inputs_precisions="x1:INPUT_A_PREC x2:INPUT_B_PREC" --inputs_layouts="x1:INPUT_LAYOUT x2:INPUT_LAYOUT" --outputs_precisions="model/dot/MatMul:OUTPUT_PREC" --outputs_layouts="model/dot/MatMul:OUTPUT_LAYOUT" --config   NPU_PLATFORM="3720" PERFORMANCE_HINT="LATENCY")RAW";

bool instantiateTemplateMATMUL(cl_dbk_id_exp DbkId, const void *KernelAttrs,
                               std::string &ModelXMLInstance,
                               std::vector<uint8_t> &ModelBinary,
                               std::string &BuildFlagsInstance) {

  auto *Attrs = (const cl_dbk_attributes_matmul_exp *)KernelAttrs;
  bool OperandsHaveSameType =
      Attrs->a.dtype == Attrs->b.dtype && Attrs->a.dtype == Attrs->c.dtype;
  ModelXMLInstance =
      OperandsHaveSameType ? MATMUL_T_XML_Template : MATMUL_T_U_V_XML_Template;

  BuildFlagsInstance = MATMUL_Flags_Template;
  ReplaceMapT ReplaceMap;
  cl_tensor_layout_ml_exp *L = nullptr;

  // Batch dimension is not supported yet.
  assert(Attrs->a.rank == 2 || (Attrs->a.rank == 3 && Attrs->a.shape[0] == 1));

  size_t MatDimOffset = Attrs->a.rank - 2;

  ReplaceMap["SHAPE_A_ROWS"] = std::to_string(Attrs->a.shape[MatDimOffset + 0]);
  ReplaceMap["SHAPE_A_COLS"] = std::to_string(Attrs->a.shape[MatDimOffset + 1]);

  ReplaceMap["SHAPE_B_ROWS"] = std::to_string(Attrs->b.shape[MatDimOffset + 0]);
  ReplaceMap["SHAPE_B_COLS"] = std::to_string(Attrs->b.shape[MatDimOffset + 1]);

  ReplaceMap["SHAPE_C_ROWS"] = std::to_string(Attrs->c.shape[MatDimOffset + 0]);
  ReplaceMap["SHAPE_C_COLS"] = std::to_string(Attrs->c.shape[MatDimOffset + 1]);

  // TODO: Fix this assert to take account transpose settings.
  // assert(Attrs->a.shape[1] == Attrs->b.shape[0]);

  assert(Attrs->a.layout_type == CL_TENSOR_LAYOUT_ML_EXP ||
         Attrs->a.layout_type == CL_TENSOR_LAYOUT_BLAS_EXP);

  ReplaceMap["INPUT_A_PREC"] = dtype2precision(Attrs->a.dtype);
  ReplaceMap["INPUT_B_PREC"] = dtype2precision(Attrs->b.dtype);
  ReplaceMap["OUTPUT_PREC"] = dtype2precision(Attrs->c.dtype);
  ReplaceMap["INPUT_A_ELEM_TYPE"] = dtype2elemtype(Attrs->a.dtype);
  ReplaceMap["INPUT_B_ELEM_TYPE"] = dtype2elemtype(Attrs->b.dtype);
  ReplaceMap["OUTPUT_ELEM_TYPE"] = dtype2elemtype(Attrs->c.dtype);

  ReplaceMap["INPUT_LAYOUT"] = layout2str(Attrs->a);
  ReplaceMap["INPUT_LAYOUT"] = layout2str(Attrs->b);
  ReplaceMap["OUTPUT_LAYOUT"] = layout2str(Attrs->c);

  ReplaceMap["TRANSPOSE_A"] = Attrs->trans_a ? "true" : "false";
  ReplaceMap["TRANSPOSE_B"] = Attrs->trans_b ? "true" : "false";

  replaceAllStringsInMap(ModelXMLInstance, ReplaceMap);
  replaceAllStringsInMap(BuildFlagsInstance, ReplaceMap);
  return true;
}
