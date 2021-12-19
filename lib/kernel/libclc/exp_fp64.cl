
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


#include "exp_helper.h"

_CL_OVERLOADABLE vtype exp(vtype x)
{
    const vtype X_MIN = -(vtype)0x1.74910d52d3051p+9; // -1075*ln(2)
    const vtype X_MAX = (vtype)0x1.62e42fefa39efp+9; // 1024*ln(2)
    const vtype R_64_BY_LOG2 = (vtype)0x1.71547652b82fep+6; // 64/ln(2)
    const vtype R_LOG2_BY_64_LD = (vtype)-0x1.62e42fefa0000p-7; // head ln(2)/64
    const vtype R_LOG2_BY_64_TL = (vtype)-0x1.cf79abc9e3b39p-46; // tail ln(2)/64

    inttype n = convert_inttype(x * R_64_BY_LOG2);
    vtype r = pocl_fma(R_LOG2_BY_64_TL, (vtype)n, pocl_fma(R_LOG2_BY_64_LD, (vtype)n, x));
    return __clc_exp_helper(x, X_MIN, X_MAX, r, n);
}
