/// Level0/npu graph template for rms_norm_exp DBK.
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

#include "npu_dbk.hh"

#include "pocl_tensor_util.h"
#include "pocl_util.h"

#include <cassert>
#include <cstring>
#include <map>
#include <numeric>
#include <string>

static const char *GraphTemplate = R"(
<?xml version="1.0"?>
<net name="Model0" version="11">
  <layers>
    <layer id="0" name="Parameter_1" type="Parameter" version="opset1">
      <data shape="DATA_DIM0,DATA_DIM1,DATA_DIM2,DATA_DIM3" element_type="ELEM_TYPE" />
      <output>
        <port id="0" precision="PREC" names="Parameter_1">
          <dim>DATA_DIM0</dim>
          <dim>DATA_DIM1</dim>
          <dim>DATA_DIM2</dim>
          <dim>DATA_DIM3</dim>
        </port>
      </output>
    </layer>

    <layer id="1" name="Multiply_4" type="Multiply" version="opset1">
      <data auto_broadcast="numpy" />
      <input>
        <port id="0" precision="PREC">
          <dim>DATA_DIM0</dim>
          <dim>DATA_DIM1</dim>
          <dim>DATA_DIM2</dim>
          <dim>DATA_DIM3</dim>
        </port>
        <port id="1" precision="PREC">
          <dim>DATA_DIM0</dim>
          <dim>DATA_DIM1</dim>
          <dim>DATA_DIM2</dim>
          <dim>DATA_DIM3</dim>
        </port>
      </input>
      <output>
        <port id="2" precision="PREC">
          <dim>DATA_DIM0</dim>
          <dim>DATA_DIM1</dim>
          <dim>DATA_DIM2</dim>
          <dim>DATA_DIM3</dim>
        </port>
      </output>
    </layer>

    <layer id="2" name="ReduceAxes" type="Const" version="opset1">
      <data element_type="i32" shape="REDUCE_AXES_DIM" offset="CONST_REDUCE_AXES_OFFSET" size="CONST_REDUCE_AXES_BYTESIZE" />
      <output>
        <port id="0" precision="I32">
          <dim>REDUCE_AXES_DIM</dim>
        </port>
      </output>
    </layer>

    <layer id="3" name="ReduceMean_5" type="ReduceMean" version="opset1">
      <data keep_dims="true" />
      <input>
        <port id="0" precision="PREC">
          <dim>DATA_DIM0</dim>
          <dim>DATA_DIM1</dim>
          <dim>DATA_DIM2</dim>
          <dim>DATA_DIM3</dim>
        </port>
        <port id="1" precision="I32">
          <dim>REDUCE_AXES_DIM</dim>
        </port>
      </input>
      <output>
        <port id="2" precision="PREC">
          <dim>REDUCE_DIM0</dim>
          <dim>REDUCE_DIM1</dim>
          <dim>REDUCE_DIM2</dim>
          <dim>REDUCE_DIM3</dim>
        </port>
      </output>
    </layer>

    <layer id="4" name="Epsilon" type="Const" version="opset1">
      <data element_type="ELEM_TYPE" shape="1" offset="CONST_EPSILON_OFFSET" size="CONST_EPSILON_BYTESIZE" />
      <output>
        <port id="0" precision="PREC">
          <dim>1</dim>
        </port>
      </output>
    </layer>

    <layer id="5" name="Add_6" type="Add" version="opset1">
      <data auto_broadcast="numpy" />
      <input>
        <port id="0" precision="PREC">
          <dim>REDUCE_DIM0</dim>
          <dim>REDUCE_DIM1</dim>
          <dim>REDUCE_DIM2</dim>
          <dim>REDUCE_DIM3</dim>
        </port>
        <port id="1" precision="PREC">
          <dim>1</dim>
        </port>
      </input>
      <output>
        <port id="2" precision="PREC">
          <dim>REDUCE_DIM0</dim>
          <dim>REDUCE_DIM1</dim>
          <dim>REDUCE_DIM2</dim>
          <dim>REDUCE_DIM3</dim>
        </port>
      </output>
    </layer>

    <layer id="6" name="Sqrt_7" type="Sqrt" version="opset1">
      <input>
        <port id="0" precision="PREC">
          <dim>REDUCE_DIM0</dim>
          <dim>REDUCE_DIM1</dim>
          <dim>REDUCE_DIM2</dim>
          <dim>REDUCE_DIM3</dim>
        </port>
      </input>
      <output>
        <port id="1" precision="PREC">
          <dim>REDUCE_DIM0</dim>
          <dim>REDUCE_DIM1</dim>
          <dim>REDUCE_DIM2</dim>
          <dim>REDUCE_DIM3</dim>
        </port>
      </output>
    </layer>

    <layer id="7" name="Divide_8" type="Divide" version="opset1">
      <data auto_broadcast="numpy" m_pythondiv="true" />
      <input>
        <port id="0" precision="PREC">
          <dim>DATA_DIM0</dim>
          <dim>DATA_DIM1</dim>
          <dim>DATA_DIM2</dim>
          <dim>DATA_DIM3</dim>
        </port>
        <port id="1" precision="PREC">
          <dim>REDUCE_DIM0</dim>
          <dim>REDUCE_DIM1</dim>
          <dim>REDUCE_DIM2</dim>
          <dim>REDUCE_DIM3</dim>
        </port>
      </input>
      <output>
        <port id="2" precision="PREC" names="Result_9">
          <dim>DATA_DIM0</dim>
          <dim>DATA_DIM1</dim>
          <dim>DATA_DIM2</dim>
          <dim>DATA_DIM3</dim>
        </port>
      </output>
    </layer>

    <layer id="8" name="Result_9" type="Result" version="opset1" output_names="Result_9">
      <input>
        <port id="0" precision="PREC">
          <dim>DATA_DIM0</dim>
          <dim>DATA_DIM1</dim>
          <dim>DATA_DIM2</dim>
          <dim>DATA_DIM3</dim>
        </port>
      </input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
    <edge from-layer="0" from-port="0" to-layer="1" to-port="1" />
    <edge from-layer="0" from-port="0" to-layer="7" to-port="0" />
    <edge from-layer="1" from-port="2" to-layer="3" to-port="0" />
    <edge from-layer="2" from-port="0" to-layer="3" to-port="1" />
    <edge from-layer="3" from-port="2" to-layer="5" to-port="0" />
    <edge from-layer="4" from-port="0" to-layer="5" to-port="1" />
    <edge from-layer="5" from-port="2" to-layer="6" to-port="0" />
    <edge from-layer="6" from-port="1" to-layer="7" to-port="1" />
    <edge from-layer="7" from-port="2" to-layer="8" to-port="0" />
  </edges>
  <rt_info />
</net>
)";

static const char *BuildFlagsTemplate =
    R"RAW(--inputs_precisions="src:PREC")RAW"
    R"RAW( --inputs_layouts="src:INPUT_LAYOUT")RAW"
    R"RAW( --outputs_precisions="Result_12:PREC")RAW"
    R"RAW( --outputs_layouts="Result_12:OUTPUT_LAYOUT")RAW"
    R"RAW( --config NPU_PLATFORM="3720" PERFORMANCE_HINT="LATENCY")RAW";

bool instantiateTemplateRMSNORM(cl_dbk_id_exp DbkId, const void *KernelAttrs,
                                std::string &ModelXMLInstance,
                                std::vector<uint8_t> &ModelBinary,
                                std::string &BuildFlagsInstance) {

  auto *Attrs =
      static_cast<const cl_dbk_attributes_rms_norm_exp *>(KernelAttrs);
  ModelXMLInstance = GraphTemplate;
  BuildFlagsInstance = BuildFlagsTemplate;
  ReplaceMapT ReplaceMap;

  assert(pocl_tensor_data_is_contiguous(&Attrs->src));
  assert(pocl_tensor_data_is_contiguous(&Attrs->dst));

  ReplaceMap["DATA_DIM0"] =
      std::to_string(pocl_tensor_dim_size_or(&Attrs->dst, -4, 1));
  ReplaceMap["DATA_DIM1"] =
      std::to_string(pocl_tensor_dim_size_or(&Attrs->dst, -3, 1));
  ReplaceMap["DATA_DIM2"] =
      std::to_string(pocl_tensor_dim_size_or(&Attrs->dst, -2, 1));
  ReplaceMap["DATA_DIM3"] =
      std::to_string(pocl_tensor_dim_size(&Attrs->dst, -1));

  // Compute tensor shape for the output of the ReduceMean layer.
  const unsigned ReduceRank = 4;
  assert(Attrs->dst.rank <= ReduceRank);
  size_t ReduceShape[ReduceRank] = {
      pocl_tensor_dim_size_or(&Attrs->dst, -4, 1),
      pocl_tensor_dim_size_or(&Attrs->dst, -3, 1),
      pocl_tensor_dim_size_or(&Attrs->dst, -2, 1),
      pocl_tensor_dim_size(&Attrs->dst, -1),
  };
  unsigned CorrectedStartDim = ReduceRank - Attrs->dst.rank + Attrs->start_dim;
  for (unsigned I = CorrectedStartDim; I < ReduceRank; I++)
    ReduceShape[I] = 1;

  ReplaceMap["REDUCE_DIM0"] = std::to_string(ReduceShape[0]);
  ReplaceMap["REDUCE_DIM1"] = std::to_string(ReduceShape[1]);
  ReplaceMap["REDUCE_DIM2"] = std::to_string(ReduceShape[2]);
  ReplaceMap["REDUCE_DIM3"] = std::to_string(ReduceShape[3]);

  // Materialize values specifying reduction dimensions for the ReduceMean
  // layer.
  std::vector<uint32_t> ReductionDims(Attrs->dst.rank - Attrs->start_dim);
  std::iota(ReductionDims.begin(), ReductionDims.end(), CorrectedStartDim);

  ReplaceMap["REDUCE_AXES_DIM"] = std::to_string(ReductionDims.size());

  size_t EpsilonTypeSize = pocl_tensor_type_size(Attrs->dst.dtype);
  ReplaceMap["CONST_EPSILON_OFFSET"] = "0";
  ReplaceMap["CONST_EPSILON_BYTESIZE"] = std::to_string(EpsilonTypeSize);

  size_t ReductionDimsPos = pocl_align_value(EpsilonTypeSize, sizeof(uint32_t));
  ReplaceMap["CONST_REDUCE_AXES_OFFSET"] = std::to_string(ReductionDimsPos);
  size_t ReductionDimsSize = ReductionDims.size() * sizeof(uint32_t);
  ReplaceMap["CONST_REDUCE_AXES_BYTESIZE"] = std::to_string(ReductionDimsSize);

  ReplaceMap["ELEM_TYPE"] = dtype2elemtype(Attrs->dst.dtype);
  ReplaceMap["PREC"] = dtype2precision(Attrs->dst.dtype);

  ReplaceMap["INPUT_LAYOUT"] = "NCHW";
  ReplaceMap["OUTPUT_LAYOUT"] = "NCHW";

  replaceAllStringsInMap(ModelXMLInstance, ReplaceMap);
  replaceAllStringsInMap(BuildFlagsInstance, ReplaceMap);

  ModelBinary.resize(ReductionDimsPos + ReductionDimsSize, 0);
  std::memcpy(ModelBinary.data(), &Attrs->epsilon, EpsilonTypeSize);
  std::memcpy(ModelBinary.data() + ReductionDimsPos, ReductionDims.data(),
              ReductionDimsSize);

  return true;
}
