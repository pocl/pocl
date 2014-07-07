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
   the typedefs to the actual underlying types. Clang 3.2
   the typedefs throughout as the types are not detected
   by the frontend. */
#if !defined(_CL_HAS_IMAGE_ACCESS)
typedef int sampler_t;
typedef struct dev_image_t* image2d_t;
typedef struct dev_image_t* image3d_t;
typedef struct dev_image_t* image1d_t;
typedef struct dev_image_t* image1d_buffer_t;
typedef struct dev_image_t* image2d_array_t;
typedef struct dev_image_t* image1d_array_t;
#endif


/* cl_channel_order */
#define CL_R                                        0x10B0
#define CL_A                                        0x10B1
#define CL_RG                                       0x10B2
#define CL_RA                                       0x10B3
#define CL_RGB                                      0x10B4
#define CL_RGBA                                     0x10B5
#define CL_BGRA                                     0x10B6
#define CL_ARGB                                     0x10B7
#define CL_INTENSITY                                0x10B8
#define CL_LUMINANCE                                0x10B9
#define CL_Rx                                       0x10BA
#define CL_RGx                                      0x10BB
#define CL_RGBx                                     0x10BC
#define CL_DEPTH                                    0x10BD
#define CL_DEPTH_STENCIL                            0x10BE

/* cl_channel_type */
#define CL_SNORM_INT8                               0x10D0
#define CL_SNORM_INT16                              0x10D1
#define CL_UNORM_INT8                               0x10D2
#define CL_UNORM_INT16                              0x10D3
#define CL_UNORM_SHORT_565                          0x10D4
#define CL_UNORM_SHORT_555                          0x10D5
#define CL_UNORM_INT_101010                         0x10D6
#define CL_SIGNED_INT8                              0x10D7
#define CL_SIGNED_INT16                             0x10D8
#define CL_SIGNED_INT32                             0x10D9
#define CL_UNSIGNED_INT8                            0x10DA
#define CL_UNSIGNED_INT16                           0x10DB
#define CL_UNSIGNED_INT32                           0x10DC
#define CL_HALF_FLOAT                               0x10DD
#define CL_FLOAT                                    0x10DE
#define CL_UNORM_INT24                              0x10DF

/* cl_addressing _mode */
#define CLK_ADDRESS_NONE                            0x00
#define CLK_ADDRESS_MIRRORED_REPEAT                 0x01
#define CLK_ADDRESS_REPEAT                          0x02
#define CLK_ADDRESS_CLAMP_TO_EDGE                   0x03
#define CLK_ADDRESS_CLAMP                           0x04

/* cl_sampler_info */
#define CLK_NORMALIZED_COORDS_FALSE                 0x00
#define CLK_NORMALIZED_COORDS_TRUE                  0x08

/* filter_mode */
#define CLK_FILTER_NEAREST                          0x00
#define CLK_FILTER_LINEAR                           0x10

//#ifdef _CL_HAS_IMAGE_ACCESS

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
#endif
