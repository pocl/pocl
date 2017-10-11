/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See ROCM_LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

 OCML_ATTR v2type con(vtype a, vtype b);

 OCML_ATTR v2type csgn(v2type a, vtype b);

 OCML_ATTR v2type csgn(v2type a, v2type b);

 OCML_ATTR v2type fadd(vtype a, vtype b);

 OCML_ATTR v2type nrm(v2type a);

 OCML_ATTR v2type onrm(v2type a);

 OCML_ATTR v2type fsub(vtype a, vtype b);

 OCML_ATTR v2type add(vtype a, vtype b);

 OCML_ATTR v2type sub(vtype a, vtype b);

 OCML_ATTR v2type mul(vtype a, vtype b);

 OCML_ATTR v2type sqr(vtype a);

 OCML_ATTR v2type add(v2type a, vtype b);

 OCML_ATTR v2type fadd(v2type a, vtype b);

 OCML_ATTR v2type add(vtype a, v2type b);

 OCML_ATTR v2type fadd(vtype a, v2type b);

 OCML_ATTR v2type add(v2type a, v2type b);

 OCML_ATTR v2type fadd(v2type a, v2type b);

 OCML_ATTR v2type sub(v2type a, vtype b);

 OCML_ATTR v2type fsub(v2type a, vtype b);

 OCML_ATTR v2type sub(vtype a, v2type b);

 OCML_ATTR v2type fsub(vtype a, v2type b);

 OCML_ATTR v2type sub(v2type a, v2type b);

 OCML_ATTR v2type fsub(v2type a, v2type b);

 OCML_ATTR v2type ldx(v2type a, int e);

 OCML_ATTR v2type mul(v2type a, vtype b);

 OCML_ATTR v2type omul(v2type a, vtype b);

 OCML_ATTR v2type mul(vtype a, v2type b);

 OCML_ATTR v2type omul(vtype a, v2type b);

 OCML_ATTR v2type mul(v2type a, v2type b);

 OCML_ATTR v2type omul(v2type a, v2type b);

 OCML_ATTR v2type div(vtype a, vtype b);

 OCML_ATTR v2type div(v2type a, vtype b);

 OCML_ATTR v2type div(vtype a, v2type b);

 OCML_ATTR v2type fdiv(v2type a, v2type b);

 OCML_ATTR v2type div(v2type a, v2type b);

 OCML_ATTR v2type rcp(vtype b);

 OCML_ATTR v2type frcp(v2type b);

 OCML_ATTR v2type rcp(v2type b);

 OCML_ATTR v2type sqr(v2type a);

 OCML_ATTR v2type root2(vtype a);

 OCML_ATTR v2type root2(v2type a);

 OCML_ATTR vtype fnma(vtype a, vtype b, vtype c);

_CL_OVERLOADABLE vtype _cl_frfrexp(vtype a);

_CL_OVERLOADABLE itype _cl_expfrexp(vtype a);
