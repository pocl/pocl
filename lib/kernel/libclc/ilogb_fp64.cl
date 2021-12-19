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



_CL_OVERLOADABLE inttype ilogb(vtype x) {
    utype ux = as_utype(x);
    utype ax = ux & (utype)(~SIGNBIT_DP64);

    inttype r = convert_inttype(ax >> EXPSHIFTBITS_DP64) - (inttype)EXPBIAS_DP64;
    inttype rs = (inttype)-1011 - convert_inttype(clz(ax & (utype)MANTBITS_DP64));
    r = (ax < (utype)0x0010000000000000UL) ? rs : r;
    r = (ax > (utype)0x7ff0000000000000UL) | (ax == (utype)0UL) ? (utype)0x80000000 : r;
    r = (ax == (utype)0x7ff0000000000000UL) ? (utype)0x7fffffff : r;
    return r;
}
