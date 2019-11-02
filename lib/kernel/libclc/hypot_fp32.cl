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



// Returns sqrt(x*x + y*y) with no overflow or underflow unless the result warrants it

_CL_OVERLOADABLE vtype hypot(vtype x, vtype y)
{
    utype ux = as_utype(x);
    utype aux = ux & (utype)EXSIGNBIT_SP32;
    utype uy = as_utype(y);
    utype auy = uy & (utype)EXSIGNBIT_SP32;
    vtype retval;
    itype c = aux > auy;
    ux = c ? aux : auy;
    uy = c ? auy : aux;

    itype xexp = clamp(as_itype(ux >> (utype)EXPSHIFTBITS_SP32) - (itype)EXPBIAS_SP32, -126, 126);
    vtype fx_exp = as_vtype((xexp + (itype)EXPBIAS_SP32) << (itype)EXPSHIFTBITS_SP32);
    vtype fi_exp = as_vtype((-xexp + (itype)EXPBIAS_SP32) << (itype)EXPSHIFTBITS_SP32);
    vtype fx = as_vtype(ux) * fi_exp;
    vtype fy = as_vtype(uy) * fi_exp;
    retval = sqrt(mad(fx, fx, fy*fy)) * fx_exp;

    retval = ((ux > (utype)PINFBITPATT_SP32) | (uy == (utype)0)) ? as_vtype(ux) : retval;
    retval = ((ux == (utype)PINFBITPATT_SP32) | (uy == (utype)PINFBITPATT_SP32)) ? as_vtype((itype)PINFBITPATT_SP32) : retval;
    return retval;
}
