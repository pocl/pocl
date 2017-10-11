/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See ROCM_LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define DOUBLE_SPECIALIZATION
#include "ocml_helpers.h"
#include "ocml_helpers_impl.cl"
#undef DOUBLE_SPECIALIZATION

OCML_ATTR _CL_OVERLOADABLE vtype
fnma(vtype a, vtype b, vtype c)
{
    return BUILTIN_FMA_F64(-a, b, c);
}
