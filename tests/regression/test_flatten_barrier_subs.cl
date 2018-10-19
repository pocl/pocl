/*
  Original code from

    https://github.com/GPUOpen-LibrariesAndSDKs/RadeonRays_SDK/blob/master/CLW/CL/CLW.cl

*/

/*
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#define DEFINE_SAFE_LOAD_4(type)\
    type##4 safe_load_##type##4(__global type##4 const* source, uint idx, uint sizeInTypeUnits)\
{\
    type##4 res = (type##4)(0, 0, 0, 0);\
    if (((idx + 1) << 2)  <= sizeInTypeUnits)\
    res = source[idx];\
    else\
    {\
    if ((idx << 2) < sizeInTypeUnits) res.x = source[idx].x;\
    if ((idx << 2) + 1 < sizeInTypeUnits) res.y = source[idx].y;\
    if ((idx << 2) + 2 < sizeInTypeUnits) res.z = source[idx].z;\
    }\
    return res;\
}

DEFINE_SAFE_LOAD_4(int)

#define DEFINE_SAFE_STORE_4(type)\
    void safe_store_##type##4(type##4 val, __global type##4* dest, uint idx, uint sizeInTypeUnits)\
{\
    if ((idx + 1) * 4  <= sizeInTypeUnits)\
    dest[idx] = val;\
    else\
    {\
    if (idx*4 < sizeInTypeUnits) dest[idx].x = val.x;\
    if (idx*4 + 1 < sizeInTypeUnits) dest[idx].y = val.y;\
    if (idx*4 + 2 < sizeInTypeUnits) dest[idx].z = val.z;\
    }\
}

DEFINE_SAFE_STORE_4(int)

#define DEFINE_GROUP_SCAN_EXCLUSIVE_PART(type)\
    type group_scan_exclusive_part_##type( int localId, int groupSize, __local type* shmem)\
{\
    type sum = 0;\
    for (int stride = 1; stride <= (groupSize >> 1); stride <<= 1)\
    {\
    if (localId < groupSize/(2*stride))\
        {\
        shmem[2*(localId + 1)*stride-1] = shmem[2*(localId + 1)*stride-1] + shmem[(2*localId + 1)*stride-1];\
        }\
        barrier(CLK_LOCAL_MEM_FENCE);\
    }\
    if (localId == 0)\
    {\
    sum = shmem[groupSize - 1];\
    shmem[groupSize - 1] = 0;\
    }\
    barrier(CLK_LOCAL_MEM_FENCE);\
    for (int stride = (groupSize >> 1); stride > 0; stride >>= 1)\
    {\
    if (localId < groupSize/(2*stride))\
        {\
        type temp = shmem[(2*localId + 1)*stride-1];\
        shmem[(2*localId + 1)*stride-1] = shmem[2*(localId + 1)*stride-1];\
        shmem[2*(localId + 1)*stride-1] = shmem[2*(localId + 1)*stride-1] + temp;\
        }\
        barrier(CLK_LOCAL_MEM_FENCE);\
    }\
    return sum;\
}

DEFINE_GROUP_SCAN_EXCLUSIVE_PART(int)

#define DEFINE_SCAN_EXCLUSIVE_PART_4(type)\
    __attribute__((reqd_work_group_size(64, 1, 1)))\
    __kernel void scan_exclusive_part_##type##4(__global type##4 const* in_array, __global type##4* out_array, uint numElems, __global type* out_sums, __local type* shmem)\
{\
    int globalId  = get_global_id(0);\
    int localId   = get_local_id(0);\
    int groupId   = get_group_id(0);\
    int groupSize = get_local_size(0);\
    type##4 v1 = safe_load_##type##4(in_array, 2*globalId, numElems);\
    type##4 v2 = safe_load_##type##4(in_array, 2*globalId + 1, numElems);\
    v1.y += v1.x; v1.w += v1.z; v1.w += v1.y;\
    v2.y += v2.x; v2.w += v2.z; v2.w += v2.y;\
    v2.w += v1.w;\
    shmem[localId] = v2.w;\
    barrier(CLK_LOCAL_MEM_FENCE);\
    type sum = group_scan_exclusive_part_##type(localId, groupSize, shmem);\
    if (localId == 0) out_sums[groupId] = sum;\
    v2.w = shmem[localId];\
    type t = v1.w; v1.w = v2.w; v2.w += t;\
    t = v1.y; v1.y = v1.w; v1.w += t;\
    t = v2.y; v2.y = v2.w; v2.w += t;\
    t = v1.x; v1.x = v1.y; v1.y += t;\
    t = v2.x; v2.x = v2.y; v2.y += t;\
    t = v1.z; v1.z = v1.w; v1.w += t;\
    t = v2.z; v2.z = v2.w; v2.w += t;\
    safe_store_##type##4(v2, out_array, 2 * globalId + 1, numElems);\
    safe_store_##type##4(v1, out_array, 2 * globalId, numElems);\
}

DEFINE_SCAN_EXCLUSIVE_PART_4(int)

#define DEFINE_GROUP_SCAN_EXCLUSIVE(type)\
    void group_scan_exclusive_##type(int localId, int groupSize, __local type* shmem)\
{\
    for (int stride = 1; stride <= (groupSize >> 1); stride <<= 1)\
    {\
    if (localId < groupSize/(2*stride))\
        {\
        shmem[2*(localId + 1)*stride-1] = shmem[2*(localId + 1)*stride-1] + shmem[(2*localId + 1)*stride-1];\
        }\
        barrier(CLK_LOCAL_MEM_FENCE);\
    }\
    if (localId == 0)\
    shmem[groupSize - 1] = 0;\
    barrier(CLK_LOCAL_MEM_FENCE);\
    for (int stride = (groupSize >> 1); stride > 0; stride >>= 1)\
    {\
    if (localId < groupSize/(2*stride))\
        {\
        type temp = shmem[(2*localId + 1)*stride-1];\
        shmem[(2*localId + 1)*stride-1] = shmem[2*(localId + 1)*stride-1];\
        shmem[2*(localId + 1)*stride-1] = shmem[2*(localId + 1)*stride-1] + temp;\
        }\
        barrier(CLK_LOCAL_MEM_FENCE);\
    }\
}

DEFINE_GROUP_SCAN_EXCLUSIVE(int)

#define DEFINE_SCAN_EXCLUSIVE_4(type)\
    __attribute__((reqd_work_group_size(64, 1, 1)))\
    __kernel void scan_exclusive_##type##4(__global type##4 const* in_array, __global type##4* out_array, uint numElems, __local type* shmem)\
{\
    int globalId  = get_global_id(0);\
    int localId   = get_local_id(0);\
    int groupSize = get_local_size(0);\
    type##4 v1 = safe_load_##type##4(in_array, 2*globalId, numElems);\
    type##4 v2 = safe_load_##type##4(in_array, 2*globalId + 1, numElems);\
    v1.y += v1.x; v1.w += v1.z; v1.w += v1.y;\
    v2.y += v2.x; v2.w += v2.z; v2.w += v2.y;\
    v2.w += v1.w;\
    shmem[localId] = v2.w;\
    barrier(CLK_LOCAL_MEM_FENCE);\
    group_scan_exclusive_##type(localId, groupSize, shmem);\
    v2.w = shmem[localId];\
    type t = v1.w; v1.w = v2.w; v2.w += t;\
    t = v1.y; v1.y = v1.w; v1.w += t;\
    t = v2.y; v2.y = v2.w; v2.w += t;\
    t = v1.x; v1.x = v1.y; v1.y += t;\
    t = v2.x; v2.x = v2.y; v2.y += t;\
    t = v1.z; v1.z = v1.w; v1.w += t;\
    t = v2.z; v2.z = v2.w; v2.w += t;\
    safe_store_##type##4(v2, out_array, 2 * globalId + 1, numElems);\
    safe_store_##type##4(v1, out_array, 2 * globalId, numElems);\
}

DEFINE_SCAN_EXCLUSIVE_4(int)

#define DEFINE_DISTRIBUTE_PART_SUM_4(type)\
    __kernel void distribute_part_sum_##type##4( __global type* in_sums, __global type##4* inout_array, uint numElems)\
{\
    int globalId  = get_global_id(0);\
    int groupId   = get_group_id(0);\
    type##4 v1 = safe_load_##type##4(inout_array, globalId, numElems);\
    type    sum = in_sums[groupId >> 1];\
    v1.xyzw += sum;\
    safe_store_##type##4(v1, inout_array, globalId, numElems);\
}

DEFINE_DISTRIBUTE_PART_SUM_4(int)

void kernel scan_exclusive(global const int* input, global int* output)
{
    int i = get_global_id(0);
    output[i] = input[i]+1;
}
