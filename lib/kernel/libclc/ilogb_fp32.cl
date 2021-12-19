/*
 * Copyright (c) 2015 Advanced Micro Devices, Inc.
 * Copyright (c) 2016 Aaron Watry
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



_CL_OVERLOADABLE itype ilogb(vtype x) {
    utype ux = as_utype(x);
    utype ax = ux & (utype)EXSIGNBIT_SP32;

    itype rs = (itype)-118 - convert_itype(clz(ux & (utype)MANTBITS_SP32));
    itype r = as_itype(ax >> EXPSHIFTBITS_SP32) - (itype)EXPBIAS_SP32;
    r = (ax < (utype)0x00800000U) ? rs : r;
    r = ((ax > (utype)EXPBITS_SP32) | (ax == (utype)0)) ? (itype)0x80000000 : r;
    r = (ax == (utype)EXPBITS_SP32) ? (itype)0x7fffffff : r;
    return r;
}
