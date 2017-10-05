/* OpenCL built-in library: vtables_macros.h

   Copyright (c) 2017 Michal Babej / Tampere University of Technology

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/


#define VTABLE_FUNCTION(TYPE,VTABLE,NAME)     \
     _CL_OVERLOADABLE TYPE VTABLE_MANGLE(NAME)(uint idx) {   \
        TYPE retval;                        \
        retval = VTABLE[ idx ];          \
        return retval;                      \
    }\
     _CL_OVERLOADABLE TYPE ## 2 VTABLE_MANGLE(NAME)(uint2 idx) {   \
        TYPE ## 2 retval;                        \
        retval.s0 = VTABLE [idx.s0];       \
        retval.s1 = VTABLE [idx.s1];       \
        return retval;                      \
    }\
     _CL_OVERLOADABLE TYPE ## 3 VTABLE_MANGLE(NAME)(uint3 idx) {   \
        TYPE ## 3 retval;                        \
        retval.s0 = VTABLE [idx.s0];       \
        retval.s1 = VTABLE [idx.s1];       \
        retval.s2 = VTABLE [idx.s2];       \
        return retval;                      \
    }\
     _CL_OVERLOADABLE TYPE ## 4 VTABLE_MANGLE(NAME)(uint4 idx) {   \
        TYPE ## 4 retval;                        \
        retval.s0 = VTABLE [idx.s0];       \
        retval.s1 = VTABLE [idx.s1];       \
        retval.s2 = VTABLE [idx.s2];       \
        retval.s3 = VTABLE [idx.s3];       \
        return retval;                      \
    }\
     _CL_OVERLOADABLE TYPE ## 8 VTABLE_MANGLE(NAME)(uint8 idx) {   \
        TYPE ## 8 retval;                        \
        retval.s0 = VTABLE [idx.s0];       \
        retval.s1 = VTABLE [idx.s1];       \
        retval.s2 = VTABLE [idx.s2];       \
        retval.s3 = VTABLE [idx.s3];       \
        retval.s4 = VTABLE [idx.s4];       \
        retval.s5 = VTABLE [idx.s5];       \
        retval.s6 = VTABLE [idx.s6];       \
        retval.s7 = VTABLE [idx.s7];       \
        return retval;                      \
    }\
     _CL_OVERLOADABLE TYPE ## 16 VTABLE_MANGLE(NAME)(uint16 idx) {   \
        TYPE ## 16 retval;                        \
        retval.s0 = VTABLE [idx.s0];       \
        retval.s1 = VTABLE [idx.s1];       \
        retval.s2 = VTABLE [idx.s2];       \
        retval.s3 = VTABLE [idx.s3];       \
        retval.s4 = VTABLE [idx.s4];       \
        retval.s5 = VTABLE [idx.s5];       \
        retval.s6 = VTABLE [idx.s6];       \
        retval.s7 = VTABLE [idx.s7];       \
        retval.s8 = VTABLE [idx.s8];       \
        retval.s9 = VTABLE [idx.s9];       \
        retval.sA = VTABLE [idx.sA];       \
        retval.sB = VTABLE [idx.sB];       \
        retval.sC = VTABLE [idx.sC];       \
        retval.sD = VTABLE [idx.sD];       \
        retval.sE = VTABLE [idx.sE];       \
        retval.sF = VTABLE [idx.sF];       \
        return retval;                      \
    }





#define VTABLE_FUNCTION2(TYPE,VTABLE,NAME)     \
     _CL_OVERLOADABLE TYPE VTABLE_MANGLE(NAME)(uint idx) {   \
        TYPE retval;                        \
        retval.lo = VTABLE[ idx ].lo;             \
        retval.hi = VTABLE[ idx ].hi;             \
        return retval;                      \
    }\
     _CL_OVERLOADABLE TYPE ## 2 VTABLE_MANGLE(NAME)(uint2 idx) {   \
        TYPE ## 2 retval;                        \
        retval.lo.s0 = VTABLE [idx.s0].lo; retval.hi.s0 = VTABLE [idx.s0].hi;       \
        retval.lo.s1 = VTABLE [idx.s1].lo; retval.hi.s1 = VTABLE [idx.s1].hi;       \
        return retval;                      \
    }\
     _CL_OVERLOADABLE TYPE ## 3 VTABLE_MANGLE(NAME)(uint3 idx) {   \
        TYPE ## 3 retval;                        \
        retval.lo.s0 = VTABLE [idx.s0].lo; retval.hi.s0 = VTABLE [idx.s0].hi;       \
        retval.lo.s1 = VTABLE [idx.s1].lo; retval.hi.s1 = VTABLE [idx.s1].hi;       \
        retval.lo.s2 = VTABLE [idx.s2].lo; retval.hi.s2 = VTABLE [idx.s2].hi;       \
        return retval;                      \
    }\
     _CL_OVERLOADABLE TYPE ## 4 VTABLE_MANGLE(NAME)(uint4 idx) {   \
        TYPE ## 4 retval;                        \
        retval.lo.s0 = VTABLE [idx.s0].lo; retval.hi.s0 = VTABLE [idx.s0].hi;       \
        retval.lo.s1 = VTABLE [idx.s1].lo; retval.hi.s1 = VTABLE [idx.s1].hi;       \
        retval.lo.s2 = VTABLE [idx.s2].lo; retval.hi.s2 = VTABLE [idx.s2].hi;       \
        retval.lo.s3 = VTABLE [idx.s3].lo; retval.hi.s3 = VTABLE [idx.s3].hi;       \
        return retval;                      \
    }\
     _CL_OVERLOADABLE TYPE ## 8 VTABLE_MANGLE(NAME)(uint8 idx) {   \
        TYPE ## 8 retval;                        \
        retval.lo.s0 = VTABLE [idx.s0].lo; retval.hi.s0 = VTABLE [idx.s0].hi;       \
        retval.lo.s1 = VTABLE [idx.s1].lo; retval.hi.s1 = VTABLE [idx.s1].hi;       \
        retval.lo.s2 = VTABLE [idx.s2].lo; retval.hi.s2 = VTABLE [idx.s2].hi;       \
        retval.lo.s3 = VTABLE [idx.s3].lo; retval.hi.s3 = VTABLE [idx.s3].hi;       \
        retval.lo.s4 = VTABLE [idx.s4].lo; retval.hi.s4 = VTABLE [idx.s4].hi;       \
        retval.lo.s5 = VTABLE [idx.s5].lo; retval.hi.s5 = VTABLE [idx.s5].hi;       \
        retval.lo.s6 = VTABLE [idx.s6].lo; retval.hi.s6 = VTABLE [idx.s6].hi;       \
        retval.lo.s7 = VTABLE [idx.s7].lo; retval.hi.s7 = VTABLE [idx.s7].hi;       \
        return retval;                      \
    }\
     _CL_OVERLOADABLE TYPE ## 16 VTABLE_MANGLE(NAME)(uint16 idx) {   \
        TYPE ## 16 retval;                        \
        retval.lo.s0 = VTABLE [idx.s0].lo; retval.hi.s0 = VTABLE [idx.s0].hi;       \
        retval.lo.s1 = VTABLE [idx.s1].lo; retval.hi.s1 = VTABLE [idx.s1].hi;       \
        retval.lo.s2 = VTABLE [idx.s2].lo; retval.hi.s2 = VTABLE [idx.s2].hi;       \
        retval.lo.s3 = VTABLE [idx.s3].lo; retval.hi.s3 = VTABLE [idx.s3].hi;       \
        retval.lo.s4 = VTABLE [idx.s4].lo; retval.hi.s4 = VTABLE [idx.s4].hi;       \
        retval.lo.s5 = VTABLE [idx.s5].lo; retval.hi.s5 = VTABLE [idx.s5].hi;       \
        retval.lo.s6 = VTABLE [idx.s6].lo; retval.hi.s6 = VTABLE [idx.s6].hi;       \
        retval.lo.s7 = VTABLE [idx.s7].lo; retval.hi.s7 = VTABLE [idx.s7].hi;       \
        retval.lo.s8 = VTABLE [idx.s8].lo; retval.hi.s8 = VTABLE [idx.s8].hi;       \
        retval.lo.s9 = VTABLE [idx.s9].lo; retval.hi.s9 = VTABLE [idx.s9].hi;       \
        retval.lo.sA = VTABLE [idx.sA].lo; retval.hi.sA = VTABLE [idx.sA].hi;       \
        retval.lo.sB = VTABLE [idx.sB].lo; retval.hi.sB = VTABLE [idx.sB].hi;       \
        retval.lo.sC = VTABLE [idx.sC].lo; retval.hi.sC = VTABLE [idx.sC].hi;       \
        retval.lo.sD = VTABLE [idx.sD].lo; retval.hi.sD = VTABLE [idx.sD].hi;       \
        retval.lo.sE = VTABLE [idx.sE].lo; retval.hi.sE = VTABLE [idx.sE].hi;       \
        retval.lo.sF = VTABLE [idx.sF].lo; retval.hi.sF = VTABLE [idx.sF].hi;       \
        return retval;                      \
    }




#define VTABLE_FUNCTION4(VTABLE,NAME)     \
     _CL_OVERLOADABLE v4uint VTABLE_MANGLE(NAME)(uint idx) {   \
        v4uint retval;                      \
        retval = *(__constant v4uint *)(VTABLE + idx);             \
        return retval;                      \
    }\
     _CL_OVERLOADABLE v4uint2 VTABLE_MANGLE(NAME)(uint2 idx) {   \
        v4uint2 retval; uint4 tmp;                       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s0); retval.s0.s0 = tmp.s0; retval.s1.s0 = tmp.s1; retval.s2.s0 = tmp.s2; retval.s3.s0 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s1); retval.s0.s1 = tmp.s0; retval.s1.s1 = tmp.s1; retval.s2.s1 = tmp.s2; retval.s3.s1 = tmp.s3;       \
        return retval;                      \
    }\
     _CL_OVERLOADABLE v4uint3 VTABLE_MANGLE(NAME)(uint3 idx) {   \
        v4uint3 retval; uint4 tmp;                       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s0); retval.s0.s0 = tmp.s0; retval.s1.s0 = tmp.s1; retval.s2.s0 = tmp.s2; retval.s3.s0 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s1); retval.s0.s1 = tmp.s0; retval.s1.s1 = tmp.s1; retval.s2.s1 = tmp.s2; retval.s3.s1 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s2); retval.s0.s2 = tmp.s0; retval.s1.s2 = tmp.s1; retval.s2.s2 = tmp.s2; retval.s3.s2 = tmp.s3;       \
        return retval;                      \
    }\
     _CL_OVERLOADABLE v4uint4 VTABLE_MANGLE(NAME)(uint4 idx) {   \
        v4uint4 retval; uint4 tmp;                       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s0); retval.s0.s0 = tmp.s0; retval.s1.s0 = tmp.s1; retval.s2.s0 = tmp.s2; retval.s3.s0 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s1); retval.s0.s1 = tmp.s0; retval.s1.s1 = tmp.s1; retval.s2.s1 = tmp.s2; retval.s3.s1 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s2); retval.s0.s2 = tmp.s0; retval.s1.s2 = tmp.s1; retval.s2.s2 = tmp.s2; retval.s3.s2 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s3); retval.s0.s3 = tmp.s0; retval.s1.s3 = tmp.s1; retval.s2.s3 = tmp.s2; retval.s3.s3 = tmp.s3;       \
        return retval;                      \
    }\
     _CL_OVERLOADABLE v4uint8 VTABLE_MANGLE(NAME)(uint8 idx) {   \
        v4uint8 retval; uint4 tmp;                        \
        tmp = *(__constant uint4 *)(VTABLE + idx.s0); retval.s0.s0 = tmp.s0; retval.s1.s0 = tmp.s1; retval.s2.s0 = tmp.s2; retval.s3.s0 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s1); retval.s0.s1 = tmp.s0; retval.s1.s1 = tmp.s1; retval.s2.s1 = tmp.s2; retval.s3.s1 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s2); retval.s0.s2 = tmp.s0; retval.s1.s2 = tmp.s1; retval.s2.s2 = tmp.s2; retval.s3.s2 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s3); retval.s0.s3 = tmp.s0; retval.s1.s3 = tmp.s1; retval.s2.s3 = tmp.s2; retval.s3.s3 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s4); retval.s0.s4 = tmp.s0; retval.s1.s4 = tmp.s1; retval.s2.s4 = tmp.s2; retval.s3.s4 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s5); retval.s0.s5 = tmp.s0; retval.s1.s5 = tmp.s1; retval.s2.s5 = tmp.s2; retval.s3.s5 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s6); retval.s0.s6 = tmp.s0; retval.s1.s6 = tmp.s1; retval.s2.s6 = tmp.s2; retval.s3.s6 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s7); retval.s0.s7 = tmp.s0; retval.s1.s7 = tmp.s1; retval.s2.s7 = tmp.s2; retval.s3.s7 = tmp.s3;       \
        return retval;                      \
    }\
     _CL_OVERLOADABLE v4uint16 VTABLE_MANGLE(NAME)(uint16 idx) {   \
        v4uint16 retval; uint4 tmp;                        \
        tmp = *(__constant uint4 *)(VTABLE + idx.s0); retval.s0.s0 = tmp.s0; retval.s1.s0 = tmp.s1; retval.s2.s0 = tmp.s2; retval.s3.s0 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s1); retval.s0.s1 = tmp.s0; retval.s1.s1 = tmp.s1; retval.s2.s1 = tmp.s2; retval.s3.s1 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s2); retval.s0.s2 = tmp.s0; retval.s1.s2 = tmp.s1; retval.s2.s2 = tmp.s2; retval.s3.s2 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s3); retval.s0.s3 = tmp.s0; retval.s1.s3 = tmp.s1; retval.s2.s3 = tmp.s2; retval.s3.s3 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s4); retval.s0.s4 = tmp.s0; retval.s1.s4 = tmp.s1; retval.s2.s4 = tmp.s2; retval.s3.s4 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s5); retval.s0.s5 = tmp.s0; retval.s1.s5 = tmp.s1; retval.s2.s5 = tmp.s2; retval.s3.s5 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s6); retval.s0.s6 = tmp.s0; retval.s1.s6 = tmp.s1; retval.s2.s6 = tmp.s2; retval.s3.s6 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s7); retval.s0.s7 = tmp.s0; retval.s1.s7 = tmp.s1; retval.s2.s7 = tmp.s2; retval.s3.s7 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s8); retval.s0.s8 = tmp.s0; retval.s1.s8 = tmp.s1; retval.s2.s8 = tmp.s2; retval.s3.s8 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.s9); retval.s0.s9 = tmp.s0; retval.s1.s9 = tmp.s1; retval.s2.s9 = tmp.s2; retval.s3.s9 = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.sA); retval.s0.sA = tmp.s0; retval.s1.sA = tmp.s1; retval.s2.sA = tmp.s2; retval.s3.sA = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.sB); retval.s0.sB = tmp.s0; retval.s1.sB = tmp.s1; retval.s2.sB = tmp.s2; retval.s3.sB = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.sC); retval.s0.sC = tmp.s0; retval.s1.sC = tmp.s1; retval.s2.sC = tmp.s2; retval.s3.sC = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.sD); retval.s0.sD = tmp.s0; retval.s1.sD = tmp.s1; retval.s2.sD = tmp.s2; retval.s3.sD = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.sE); retval.s0.sE = tmp.s0; retval.s1.sE = tmp.s1; retval.s2.sE = tmp.s2; retval.s3.sE = tmp.s3;       \
        tmp = *(__constant uint4 *)(VTABLE + idx.sF); retval.s0.sF = tmp.s0; retval.s1.sF = tmp.s1; retval.s2.sF = tmp.s2; retval.s3.sF = tmp.s3;       \
        return retval;                      \
    }
