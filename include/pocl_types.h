// Scalar type definitions

#if defined cl_khr_fp64 && !defined cl_khr_int64
#  error "cl_khr_fp64 requires cl_khr_int64"
#endif



/* Disable undefined datatypes */

/* The definitions below intentionally lead to errors if these types
   are used when they are not available in the language. This prevents
   accidentally using them if the compiler does not disable these
   types, but only e.g. defines them with an incorrect size.*/

#ifndef cl_khr_int64
typedef struct error_undefined_type_long error_undefined_type_long;
#  define long error_undefined_type_long
typedef struct error_undefined_type_ulong error_undefined_type_ulong;
#  define ulong error_undefined_type_ulong
#endif

#ifndef cl_khr_fp16
typedef struct error_undefined_type_half error_undefined_type_half;
#  define half error_undefined_type_half
#endif

#ifndef cl_khr_fp64
typedef struct error_undefined_type_double error_undefined_type_double;
#  define double error_undefined_type_double
#endif



/* Define unsigned datatypes */

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
#ifdef cl_khr_int64
typedef unsigned long ulong;
#endif

/* Define pointer helper types */

typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef ptrdiff_t intptr_t;
typedef size_t uintptr_t;


/* Image types.
 * Note: there is a duplicate definition in
 * lib/CL/devices/dev_image.h - keep in sync?
 */
typedef int dev_sampler_t;

typedef struct dev_image_t {
  void* _data;
  int _width;
  int _height;
  int _depth;
  int _image_array_size;
  int _row_pitch;
  int _slice_pitch;
  int _num_mip_levels; /* maybe not needed */
  int _num_samples; /* maybe not needed */
  int _order;
  int _data_type;
  int _num_channels;
  int _elem_size;
} dev_image_t;

