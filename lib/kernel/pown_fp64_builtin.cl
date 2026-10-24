/* OpenCL built-in library: pown() for double only, via __builtin_powi.

   Used with ENABLE_HOST_CPU_VECTORIZE_BUILTINS, together with
   libclc-pocl/pown_fp32_only.cl for float. This is what pown.cl provides
   for double; float is left to libclc for accuracy.

   SPDX-License-Identifier: MIT
*/

#include "templates.h"

__IF_FP64(
double __attribute__ ((overloadable))
pown(double a, int b)
{
  return __builtin_powi(a, b);
}
IMPLEMENT_BUILTIN_V_VJ(pown, double2 , int2 , lo, hi)
IMPLEMENT_BUILTIN_V_VJ(pown, double3 , int3 , lo, s2)
IMPLEMENT_BUILTIN_V_VJ(pown, double4 , int4 , lo, hi)
IMPLEMENT_BUILTIN_V_VJ(pown, double8 , int8 , lo, hi)
IMPLEMENT_BUILTIN_V_VJ(pown, double16, int16, lo, hi))
