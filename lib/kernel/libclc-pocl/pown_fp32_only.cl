/* OpenCL built-in library: pown() for float only, from libclc.

   Used with ENABLE_HOST_CPU_VECTORIZE_BUILTINS, together with
   pown_fp64_builtin.cl for double. libclc's float pown meets the 16 ULP
   bound where llvm.powi does not; libclc's double pown goes through
   pow_base_fp64, whose vector frexp LLVM miscompiles at baseline x86-64
   targets (llvm/llvm-project#224127), so double stays on the builtin.

   SPDX-License-Identifier: MIT
*/

#define POCL_POWN_FP32_ONLY
#include "pown.cl"
