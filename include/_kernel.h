/* pocl/_kernel.h - OpenCL types and runtime library
   functions declarations.

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
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

/* Enable double precision. This should really only be done when
   building the run-time library; when building application code, we
   should instead check a macro to see whether the application has
   enabled this. At the moment, always enable this seems fine, since
   all our target devices will support double precision anyway.
*/
#pragma OPENCL EXTENSION cl_khr_fp64: enable

//#define __kernel __attribute__ ((noinline))
#define __global __attribute__ ((address_space(3)))
#define __local __attribute__ ((address_space(4)))
#define __constant __attribute__ ((address_space(5)))

#define global __attribute__ ((address_space(3)))
#define local __attribute__ ((address_space(4)))
#define constant __attribute__ ((address_space(5)))

typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef float float16 __attribute__((ext_vector_type(16)));

typedef double double2 __attribute__((ext_vector_type(2)));
typedef double double3 __attribute__((ext_vector_type(3)));
typedef double double4 __attribute__((ext_vector_type(4)));
typedef double double8 __attribute__((ext_vector_type(8)));
typedef double double16 __attribute__((ext_vector_type(16)));

typedef unsigned uint;
typedef enum {
  CLK_LOCAL_MEM_FENCE = 0x1,
  CLK_GLOBAL_MEM_FENCE = 0x2
} cl_mem_fence_flags;

uint get_global_id(uint);
uint get_group_id(uint);
uint get_local_id(uint);

int mad24(int x, int y, int z);

#define dot cl_dot
float __attribute__ ((overloadable)) cl_dot(float a, float b);
float __attribute__ ((overloadable)) cl_dot(float2 a, float2 b);
float __attribute__ ((overloadable)) cl_dot(float3 a, float3 b);
float __attribute__ ((overloadable)) cl_dot(float4 a, float4 b);
float __attribute__ ((overloadable)) cl_dot(float8 a, float8 b);
float __attribute__ ((overloadable)) cl_dot(float16 a, float16 b);

double __attribute__ ((overloadable)) cl_dot(double a, double b);
double __attribute__ ((overloadable)) cl_dot(double2 a, double2 b);
double __attribute__ ((overloadable)) cl_dot(double3 a, double3 b);
double __attribute__ ((overloadable)) cl_dot(double4 a, double4 b);
double __attribute__ ((overloadable)) cl_dot(double8 a, double8 b);
double __attribute__ ((overloadable)) cl_dot(double16 a, double16 b);

#define sin cl_sin
float __attribute__ ((overloadable)) sin(float b);
double __attribute__ ((overloadable)) sin(double b);

__attribute__ ((noinline)) void barrier (cl_mem_fence_flags flags);
