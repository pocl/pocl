/* pocl/_kernel_c.h - C compatible OpenCL types and runtime library
   functions declarations.

   Copyright (c) 2011 Universidad Rey Juan Carlos
   Copyright (c) 2011-2013 Pekka Jääskeläinen / TUT
   Copyright (c) 2011-2013 Erik Schnetter <eschnetter@perimeterinstitute.ca>
                           Perimeter Institute for Theoretical Physics
   
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
/**
 * Header that can be implemented in C compiled implementations of
 * built-in functions to introduce the OpenCL C compatible types etc.
 */
#ifndef _KERNEL_C_H
#define _KERNEL_C_H

#include "pocl_types.h"

#include "_kernel_constants.h"

/* Function/type attributes supported by Clang/SPIR */
#if __has_attribute(__always_inline__)
#  define _CL_ALWAYSINLINE __attribute__((__always_inline__))
#else
#  define _CL_ALWAYSINLINE
#endif
#if __has_attribute(__noinline__)
#  define _CL_NOINLINE __attribute__((__noinline__))
#else
#  define _CL_NOINLINE
#endif
#if __has_attribute(__overloadable__)
#  define _CL_OVERLOADABLE __attribute__((__overloadable__))
#else
#  define _CL_OVERLOADABLE
#endif
#if (__clang_major__ == 3) && (__clang_minor__ >= 2)
/* This causes an error with Clang 3.1: */
/* #if __has_attribute(__const__) */
#  define _CL_READNONE __attribute__((__const__))
#else
#  define _CL_READNONE
#endif
#if __has_attribute(__pure__)
#  define _CL_READONLY __attribute__((__pure__))
#else
#  define _CL_READONLY
#endif
#if __has_attribute(__unavailable__)
#  define _CL_UNAVAILABLE __attribute__((__unavailable__))
#else
#  define _CL_UNAVAILABLE
#endif

typedef char char2  __attribute__((__ext_vector_type__(2)));
typedef char char3  __attribute__((__ext_vector_type__(3)));
typedef char char4  __attribute__((__ext_vector_type__(4)));
typedef char char8  __attribute__((__ext_vector_type__(8)));
typedef char char16 __attribute__((__ext_vector_type__(16)));

typedef uchar uchar2  __attribute__((__ext_vector_type__(2)));
typedef uchar uchar3  __attribute__((__ext_vector_type__(3)));
typedef uchar uchar4  __attribute__((__ext_vector_type__(4)));
typedef uchar uchar8  __attribute__((__ext_vector_type__(8)));
typedef uchar uchar16 __attribute__((__ext_vector_type__(16)));

typedef short short2  __attribute__((__ext_vector_type__(2)));
typedef short short3  __attribute__((__ext_vector_type__(3)));
typedef short short4  __attribute__((__ext_vector_type__(4)));
typedef short short8  __attribute__((__ext_vector_type__(8)));
typedef short short16 __attribute__((__ext_vector_type__(16)));

typedef ushort ushort2  __attribute__((__ext_vector_type__(2)));
typedef ushort ushort3  __attribute__((__ext_vector_type__(3)));
typedef ushort ushort4  __attribute__((__ext_vector_type__(4)));
typedef ushort ushort8  __attribute__((__ext_vector_type__(8)));
typedef ushort ushort16 __attribute__((__ext_vector_type__(16)));

typedef int int2  __attribute__((__ext_vector_type__(2)));
typedef int int3  __attribute__((__ext_vector_type__(3)));
typedef int int4  __attribute__((__ext_vector_type__(4)));
typedef int int8  __attribute__((__ext_vector_type__(8)));
typedef int int16 __attribute__((__ext_vector_type__(16)));

typedef uint uint2  __attribute__((__ext_vector_type__(2)));
typedef uint uint3  __attribute__((__ext_vector_type__(3)));
typedef uint uint4  __attribute__((__ext_vector_type__(4)));
typedef uint uint8  __attribute__((__ext_vector_type__(8)));
typedef uint uint16 __attribute__((__ext_vector_type__(16)));

#if defined(__CBUILD__) && defined(cl_khr_fp16)
/* NOTE: the Clang's __fp16 does not work robustly in C mode, 
   it might produce invalid code at least with half vectors.
   Using the native 'half' type in OpenCL C mode works better. */
typedef __fp16 half;
#endif

#ifdef cl_khr_fp16
typedef half half2  __attribute__((__ext_vector_type__(2)));
typedef half half3  __attribute__((__ext_vector_type__(3)));
typedef half half4  __attribute__((__ext_vector_type__(4)));
typedef half half8  __attribute__((__ext_vector_type__(8)));
typedef half half16 __attribute__((__ext_vector_type__(16)));
#endif

typedef float float2  __attribute__((__ext_vector_type__(2)));
typedef float float3  __attribute__((__ext_vector_type__(3)));
typedef float float4  __attribute__((__ext_vector_type__(4)));
typedef float float8  __attribute__((__ext_vector_type__(8)));
typedef float float16 __attribute__((__ext_vector_type__(16)));

#ifdef cl_khr_fp64
#  ifndef __CBUILD__
#    pragma OPENCL EXTENSION cl_khr_fp64 : enable
#  endif
typedef double double2  __attribute__((__ext_vector_type__(2)));
typedef double double3  __attribute__((__ext_vector_type__(3)));
typedef double double4  __attribute__((__ext_vector_type__(4)));
typedef double double8  __attribute__((__ext_vector_type__(8)));
typedef double double16 __attribute__((__ext_vector_type__(16)));
#endif

#ifdef cl_khr_int64
typedef long long2  __attribute__((__ext_vector_type__(2)));
typedef long long3  __attribute__((__ext_vector_type__(3)));
typedef long long4  __attribute__((__ext_vector_type__(4)));
typedef long long8  __attribute__((__ext_vector_type__(8)));
typedef long long16 __attribute__((__ext_vector_type__(16)));

typedef ulong ulong2  __attribute__((__ext_vector_type__(2)));
typedef ulong ulong3  __attribute__((__ext_vector_type__(3)));
typedef ulong ulong4  __attribute__((__ext_vector_type__(4)));
typedef ulong ulong8  __attribute__((__ext_vector_type__(8)));
typedef ulong ulong16 __attribute__((__ext_vector_type__(16)));
#endif

/* Image support */

/* Starting from Clang 3.3 the image and sampler are detected
   as opaque types by the frontend. In order to define
   the default builtins we use C functions which require
   the typedefs to the actual underlying types.
*/
#if defined(__CBUILD__)
typedef int sampler_t;

/* Since some built-ins have different return types
 * (e.g. get_image_dim returns an int2 for 2D images and arrays,
 *  but an int4 for 3D images) we want each image type to
 * point to a different type which is actually always the same.
 * We do this by making it pointer to structs whose only element is a
 * dev_image_t. The structs are not anonymous to allow identification
 * by name.
 */
typedef struct _pocl_image2d_t { dev_image_t base; }* image2d_t;
typedef struct _pocl_image3d_t { dev_image_t base; }* image3d_t;
typedef struct _pocl_image1d_t { dev_image_t base; }* image1d_t;
typedef struct _pocl_image1d_buffer_t { dev_image_t base; }* image1d_buffer_t;
typedef struct _pocl_image2d_array_t { dev_image_t base; }* image2d_array_t;
typedef struct _pocl_image1d_array_t { dev_image_t base; }* image1d_array_t;
#endif


float4 _CL_OVERLOADABLE read_imagef (image2d_t image, sampler_t sampler,
                                     int2 coord);

float4 _CL_OVERLOADABLE read_imagef (image2d_t image, sampler_t sampler,
                                     float2 coord);

uint4 _CL_OVERLOADABLE read_imageui (image2d_t image, sampler_t sampler, 
                                     int2 coord);

uint4 _CL_OVERLOADABLE read_imageui (image2d_t image, sampler_t sampler, 
                                     int4 coord);

uint4 _CL_OVERLOADABLE read_imageui (image3d_t image, sampler_t sampler, 
                                     int4 coord);

int4 _CL_OVERLOADABLE read_imagei (image2d_t image, sampler_t sampler, 
                                   int2 coord);


void _CL_OVERLOADABLE write_imagei (image2d_t image, int2 coord, int4 color);

void _CL_OVERLOADABLE write_imageui (image2d_t image, int2 coord, uint4 color);



void _CL_OVERLOADABLE write_imagef (image2d_t image, int2 coord,
                                    float4 color);
/* not implemented 
void _CL_OVERLOADABLE write_imagef (image2d_array_t image, int4 coord,
                                    float4 color);

void _CL_OVERLOADABLE write_imagei (image2d_array_t image, int4 coord,
                                    int4 color);

void _CL_OVERLOADABLE write_imageui (image2d_array_t image, int4 coord,
                                     uint4 color);

void _CL_OVERLOADABLE write_imagef (image1d_t image, int coord,
                                    float4 color);

void _CL_OVERLOADABLE write_imagei (image1d_t image, int coord,
                                    int4 color);

void _CL_OVERLOADABLE write_imageui (image1d_t image, int coord, 
                                     uint4 color);

void _CL_OVERLOADABLE write_imagef (image1d_buffer_t image, int coord, 
                                    float4 color);

void _CL_OVERLOADABLE write_imagei (image1d_buffer_t image, int coord,
                                     int4 color);

void _CL_OVERLOADABLE write_imageui (image1d_buffer_t image, int coord,
                                     uint4 color);

void _CL_OVERLOADABLE write_imagef (image1d_array_t image, int2 coord,
                                    float4 color);

void _CL_OVERLOADABLE write_imagei (image1d_array_t image, int2 coord,
                                    int4 color);

void _CL_OVERLOADABLE write_imageui (image1d_array_t image, int2 coord,
                                     uint4 color);

void _CL_OVERLOADABLE write_imageui (image3d_t image, int4 coord,
                                     uint4 color);
*/
int _CL_OVERLOADABLE get_image_width (image1d_t image);
int _CL_OVERLOADABLE get_image_width (image2d_t image);
int _CL_OVERLOADABLE get_image_width (image3d_t image);

int _CL_OVERLOADABLE get_image_height (image1d_t image);
int _CL_OVERLOADABLE get_image_height (image2d_t image);
int _CL_OVERLOADABLE get_image_height (image3d_t image);

int _CL_OVERLOADABLE get_image_depth (image1d_t image);
int _CL_OVERLOADABLE get_image_depth (image2d_t image);
int _CL_OVERLOADABLE get_image_depth (image3d_t image);

int2 _CL_OVERLOADABLE get_image_dim (image2d_t image);
int2 _CL_OVERLOADABLE get_image_dim (image2d_array_t image);
int4 _CL_OVERLOADABLE get_image_dim (image3d_t image);

#endif
