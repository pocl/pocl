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
<net name="pocl_convert_exp" version="11">
  <layers>
    <layer id="0" name="src" type="Parameter" version="opset1">
      <data shape="DIM" element_type="INPUT_ELEM_TYPE" />
      <output>
        <port id="0" precision="INPUT_PREC">
          <dim>DIM</dim>
        </port>
      </output>
    </layer>

    <layer id="1" name="convert_op" type="Convert" version="opset1">
      <data destination_type="OUTPUT_ELEM_TYPE" />
      <input>
        <port id="0" precision="INPUT_PREC">
          <dim>DIM</dim>
        </port>
      </input>
      <output>
        <port id="1" precision="OUTPUT_PREC">
          <dim>DIM</dim>
        </port>
      </output>
    </layer>

    <layer id="2" name="dst" type="Result" version="opset1">
      <input>
        <port id="0" precision="OUTPUT_PREC">
          <dim>DIM</dim>
        </port>
      </input>
    </layer>
  </layers>

  <edges>
    <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
    <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
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
    R"RAW(--inputs_precisions="src:INPUT_PREC")RAW"
    R"RAW( --inputs_layouts="src:INPUT_LAYOUT")RAW"
    R"RAW( --outputs_precisions="convert:OUTPUT_PREC")RAW"
    R"RAW( --outputs_layouts="convert:OUTPUT_LAYOUT")RAW"
    R"RAW( --config NPU_PLATFORM="3720" PERFORMANCE_HINT="LATENCY")RAW";

bool instantiateTemplateCONVERT(const void *KernelAttrs,
                                std::string &ModelXMLInstance,
                                std::vector<uint8_t> &ModelBinary,
                                std::string &BuildFlagsInstance) {

  auto *Attrs = (const cl_dbk_attributes_convert_exp *)KernelAttrs;
  ModelXMLInstance = GraphTemplate;
  BuildFlagsInstance = BuildFlagsTemplate;
  ReplaceMapT ReplaceMap;

  assert(pocl_tensor_data_is_contiguous(&Attrs->src));
  assert(pocl_tensor_data_is_contiguous(&Attrs->dst));

  ReplaceMap["DIM"] = std::to_string(pocl_tensor_element_count(&Attrs->src));

  ReplaceMap["INPUT_PREC"] = dtype2precision(Attrs->src.dtype);
  ReplaceMap["OUTPUT_PREC"] = dtype2precision(Attrs->dst.dtype);
  ReplaceMap["INPUT_ELEM_TYPE"] = dtype2elemtype(Attrs->src.dtype);
  ReplaceMap["OUTPUT_ELEM_TYPE"] = dtype2elemtype(Attrs->dst.dtype);

  ReplaceMap["INPUT_LAYOUT"] = layout2str(Attrs->src);
  ReplaceMap["OUTPUT_LAYOUT"] = layout2str(Attrs->dst);

  replaceAllStringsInMap(ModelXMLInstance, ReplaceMap);
  replaceAllStringsInMap(BuildFlagsInstance, ReplaceMap);
  return true;
}
