
#ifndef POCL_OPS
#define POCL_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def PoclDialect : Dialect {
  let name = "pocl";
  let cppNamespace = "::mlir::pocl";
}

class PoclDialect_Op<string mnemonic, list<Trait> traits = []> :
  Op<PoclDialect, mnemonic, traits>;

def WorkDimOp
    : PoclDialect_Op<"work_dim", []> {

  let results = (outs I32:$output);
  let summary = "Return value of OpenCL get_work_dim()";
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

#endif
