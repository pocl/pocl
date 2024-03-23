
#include <string>
#include <map>
#include <cassert>

#include "CL/opencl.h"

#include "npu_dbk.h"

#ifndef NPU_GEMM_H
#define NPU_GEMM_H


const char *GEMM_XML_Template = R"(
<?xml version="1.0"?>
<net name="TensorFlow_Frontend_IR" version="11">
  <layers>
    <layer id="1" name="x1" type="Parameter" version="opset1">
      <data shape="SHAPE_M,SHAPE_K" element_type="INPUT_ELEM_TYPE" />
      <output>
        <port id="0" precision="INPUT_PREC" names="x1">
          <dim>SHAPE_M</dim>
          <dim>SHAPE_K</dim>
        </port>
      </output>
    </layer>
    <layer id="0" name="x2" type="Parameter" version="opset1">
      <data shape="SHAPE_K,SHAPE_N" element_type="INPUT_ELEM_TYPE" />
      <output>
        <port id="0" precision="INPUT_PREC" names="x2">
          <dim>SHAPE_K</dim>
          <dim>SHAPE_N</dim>
        </port>
      </output>
    </layer>
    <layer id="2" name="model/dot/MatMul" type="MatMul" version="opset1">
      <data transpose_a="TRANSPOSE_A" transpose_b="TRANSPOSE_B" />
      <input>
        <port id="0" precision="INPUT_PREC">
          <dim>SHAPE_M</dim>
          <dim>SHAPE_K</dim>
        </port>
        <port id="1" precision="INPUT_PREC">
          <dim>SHAPE_K</dim>
          <dim>SHAPE_N</dim>
        </port>
      </input>
      <output>
        <port id="2" precision="OUTPUT_PREC" names="dot">
          <dim>SHAPE_M</dim>
          <dim>SHAPE_N</dim>
        </port>
      </output>
    </layer>
    <layer id="3" name="dot" type="Result" version="opset1">
      <input>
        <port id="0" precision="OUTPUT_PREC">
          <dim>SHAPE_M</dim>
          <dim>SHAPE_N</dim>
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

const char *GEMM_Flags_Template = R"RAW(--inputs_precisions="x1:INPUT_PREC x2:INPUT_PREC" --inputs_layouts="x1:INPUT_LAYOUT x2:INPUT_LAYOUT" --outputs_precisions="model/dot/MatMul:OUTPUT_PREC" --outputs_layouts="model/dot/MatMul:OUTPUT_LAYOUT" --config   NPU_PLATFORM="3720" PERFORMANCE_HINT="LATENCY")RAW";

bool instantiateTemplateGEMM(const void* KernelAttrs,
                             std::string &ModelXMLInstance,
                             std::string &BuildFlagsInstance) {
  ModelXMLInstance = GEMM_XML_Template;
  BuildFlagsInstance = GEMM_Flags_Template;
  ReplaceMapT ReplaceMap;
  cl_tensor_layout_ml *L = nullptr;

  // TODO alpha, beta

  const cl_dbk_attributes_exp_gemm *Attrs
    = (const cl_dbk_attributes_exp_gemm *)KernelAttrs;
  ReplaceMap["SHAPE_M"] = std::to_string(Attrs->a.shape[0]);
  ReplaceMap["SHAPE_K"] = std::to_string(Attrs->a.shape[1]);
  assert(Attrs->a.shape[1] == Attrs->b.shape[0]);
  //ReplaceMap["SHAPE_K"] = std::to_string(Attrs->b.shape[0]);
  ReplaceMap["SHAPE_N"] = std::to_string(Attrs->b.shape[1]);

  assert(Attrs->a.layout_type == CL_TENSOR_LAYOUT_ML);

  ReplaceMap["INPUT_PREC"] = dtype2precision(Attrs->a.dtype);
  ReplaceMap["OUTPUT_PREC"] = dtype2precision(Attrs->c_out.dtype);
  ReplaceMap["INPUT_ELEM_TYPE"] = dtype2elemtype(Attrs->a.dtype);
  ReplaceMap["INPUT_ELEM_TYPE"] = dtype2elemtype(Attrs->c_out.dtype);

  assert(Attrs->a.layout_type == CL_TENSOR_LAYOUT_ML);
  L = (cl_tensor_layout_ml *)Attrs->a.layout;
  ReplaceMap["INPUT_LAYOUT"] = layout2str(L->ml_type);
  L = (cl_tensor_layout_ml *)Attrs->b.layout;
  ReplaceMap["INPUT_LAYOUT"] = layout2str(L->ml_type);

  assert(Attrs->c_out.layout_type == CL_TENSOR_LAYOUT_ML);
  L = (cl_tensor_layout_ml *)Attrs->c_out.layout;
  ReplaceMap["OUTPUT_LAYOUT"] = layout2str(L->ml_type);

  ReplaceMap["TRANSPOSE_A"] = Attrs->trans_a ? "true" : "false";
  ReplaceMap["TRANSPOSE_B"] = Attrs->trans_b ? "true" : "false";

  replaceAllStringsInMap(ModelXMLInstance, ReplaceMap);
  replaceAllStringsInMap(BuildFlagsInstance, ReplaceMap);
  return true;
}


#endif // NPU_GEMM_H
