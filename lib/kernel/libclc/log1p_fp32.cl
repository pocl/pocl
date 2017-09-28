/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
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

_CL_OVERLOADABLE float log1p(float x)
{
    float w = x;
    uint ux = as_uint(x);
    uint ax = ux & EXSIGNBIT_SP32;

    // |x| < 2^-4
    float u2 = MATH_DIVIDE(x, 2.0f + x);
    float u = u2 + u2;
    float v = u * u;
    // 2/(5 * 2^5), 2/(3 * 2^3)
    float zsmall = pocl_fma(-u2, x, pocl_fma(v, 0x1.99999ap-7f, 0x1.555556p-4f) * v * u) + x;

    // |x| >= 2^-4
    ux = as_uint(x + 1.0f);

    int m = (int)((ux >> EXPSHIFTBITS_SP32) & 0xff) - EXPBIAS_SP32;
    float mf = (float)m;
    uint indx = (ux & 0x007f0000) + ((ux & 0x00008000) << 1);
    float F = as_float(indx | 0x3f000000);

    // x > 2^24
    float fg24 = F - as_float(0x3f000000 | (ux & MANTBITS_SP32));

    // x <= 2^24
    uint xhi = ux & 0xffff8000;
    float xh = as_float(xhi);
    float xt = (1.0f - xh) + w;
    uint xnm = ((~(xhi & 0x7f800000)) - 0x00800000) & 0x7f800000;
    xt = xt * as_float(xnm) * 0.5f;
    float fl24 = F - as_float(0x3f000000 | (xhi & MANTBITS_SP32)) - xt;

    float f = mf > 24.0f ? fg24 : fl24;

    indx = indx >> 16;
    float r = f * USE_TABLE(log_inv_tbl, indx);

    // 1/3, 1/2
    float poly = pocl_fma(pocl_fma(r, 0x1.555556p-2f, 0x1.0p-1f), r*r, r);

    const float LOG2_HEAD = 0x1.62e000p-1f;   // 0.693115234
    const float LOG2_TAIL = 0x1.0bfbe8p-15f;  // 0.0000319461833

    float2 tv = USE_TABLE(loge_tbl, indx);
    float z1 = pocl_fma(mf, LOG2_HEAD, tv.s0);
    float z2 = pocl_fma(mf, LOG2_TAIL, -poly) + tv.s1;
    float z = z1 + z2;

    z = ax < 0x3d800000U ? zsmall : z;



    // Edge cases
    z = ax >= PINFBITPATT_SP32 ? w : z;
    z = w  < -1.0f ? as_float(QNANBITPATT_SP32) : z;
    z = w == -1.0f ? as_float(NINFBITPATT_SP32) : z;
        //fix subnormals
        z = ax  < 0x33800000 ? x : z;

    return z;
}