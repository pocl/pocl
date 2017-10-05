/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
 *
 * Copyright (c) 2017 Michal Babej / Tampere University of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef __VTABLES_H__
#define __VTABLES_H__

#define VTABLE_SPACE static constant

#define VTABLE_MANGLE(NAME) __pocl_v_##NAME

#define DECLARE_VTABLE(TYPE,NAME,LENGTH) \
    VTABLE_SPACE TYPE NAME [ LENGTH ]

#define SS(x) x
#define SGFY(x, y) x ## y

#define VTABLE_FUNCTION_DECL(TYPE, NAME) \
    _CL_OVERLOADABLE TYPE VTABLE_MANGLE(NAME)(uint idx); \
    _CL_OVERLOADABLE SGFY(TYPE, 2) VTABLE_MANGLE(NAME)(uint2 idx); \
    _CL_OVERLOADABLE SGFY(TYPE, 3) VTABLE_MANGLE(NAME)(uint3 idx); \
    _CL_OVERLOADABLE SGFY(TYPE, 4) VTABLE_MANGLE(NAME)(uint4 idx); \
    _CL_OVERLOADABLE SGFY(TYPE, 8) VTABLE_MANGLE(NAME)(uint8 idx); \
    _CL_OVERLOADABLE SGFY(TYPE, 16) VTABLE_MANGLE(NAME)(uint16 idx);

#define USE_VTABLE(NAME, IDX) \
    VTABLE_MANGLE(NAME)(IDX)






VTABLE_FUNCTION_DECL(v4uint, pibits_tbl);

VTABLE_FUNCTION_DECL(float, log_inv_tbl);
VTABLE_FUNCTION_DECL(float, exp_tbl);

VTABLE_FUNCTION_DECL(v2float, loge_tbl);
VTABLE_FUNCTION_DECL(v2float, log2_tbl);
VTABLE_FUNCTION_DECL(v2float, sinhcosh_tbl);
VTABLE_FUNCTION_DECL(v2float, cbrt_tbl);
VTABLE_FUNCTION_DECL(v2float, exp_tbl_ep);




#ifdef cl_khr_fp64

VTABLE_FUNCTION_DECL(double, cbrt_inv_tbl);

VTABLE_FUNCTION_DECL(v2double, ln_tbl);
VTABLE_FUNCTION_DECL(v2double, atan_jby256_tbl);
VTABLE_FUNCTION_DECL(v2double, two_to_jby64_ep_tbl);
VTABLE_FUNCTION_DECL(v2double, sinh_tbl);
VTABLE_FUNCTION_DECL(v2double, cosh_tbl);
VTABLE_FUNCTION_DECL(v2double, cbrt_dbl_tbl);
VTABLE_FUNCTION_DECL(v2double, cbrt_rem_tbl);

#endif




#endif // __VTABLES_H__
