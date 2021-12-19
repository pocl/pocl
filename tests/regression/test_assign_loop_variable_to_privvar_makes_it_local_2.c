/* Tests a case where assigning a private for iterator to a private variable
   making the private variable uniform.

   Copyright (c) 2014 Matias Koskela and Pekka Jääskeläinen of
                      Tampere University of Technology
   
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

#include "pocl_opencl.h"
#include <string.h>

const char* kernel_src = 
"__kernel void draw(const __global int *limit) {\n"
"  int hitIndex = -1;\n"
"\n"
"  for(int i = 0; i < *limit; ++i){\n"
"    if(i == 3 && get_global_id(0) == 6){\n"
"      hitIndex = i;\n"
"      printf(\"changing the value at global_id: %d, local_id %d, group_id %d, to: %i\\n\", get_global_id(0), get_local_id(0), get_group_id(0), hitIndex);\n"
"    }\n"
"  }\n"
"\n"
"  if(hitIndex != -1){\n"
"    // (This should print if first print is printed with the same id)\n"
"    printf(\"value is changed at global_id: %d, local_id %d, group_id %d, to: %i\\n\", get_global_id(0), get_local_id(0), get_group_id(0), hitIndex);\n"
"  }\n"
"}\n";

int main() {
    int ret = 0;
    cl_context context;
    cl_device_id device;
    cl_command_queue command_queue;
    poclu_get_any_device(&context, &device, &command_queue);

    cl_mem faceCount_mem_obj = 
        clCreateBuffer(context, CL_MEM_READ_ONLY, 
                       sizeof(int), NULL, &ret);
    int faceCount = 4;
    ret |= clEnqueueWriteBuffer(command_queue, faceCount_mem_obj, CL_TRUE, 0, 
                               sizeof(int), &faceCount, 0, NULL, NULL);

    cl_int err;
    size_t length = strlen(kernel_src);
    cl_program program = 
        clCreateProgramWithSource(context, 1, &kernel_src, &length, &err);

    ret |= err;

    clBuildProgram(program, 1, &device, "", NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "draw", &ret);

    ret |= clSetKernelArg(kernel, 0, sizeof(cl_mem),(void*)&faceCount_mem_obj);

    size_t global_item_size = 8;
    size_t workGroupSize = 4;
    ret |= clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
                                 &global_item_size, &workGroupSize, 0, NULL, 
                                 NULL);
    
    clFinish(command_queue);
    ret |= clReleaseKernel(kernel);
    return ret;
}
