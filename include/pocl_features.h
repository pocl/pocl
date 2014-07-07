// Supported datataypes

// All supported Clang versions support __fp16 to some extent,
// however, the support for 'half' of OpenCL C properly added
// only in 3.3, and even that does not handle half vectors well
// for targets without native support. 

#if (__clang_major__ == 3) && (__clang_minor__ > 2) && !defined(_CL_DISABLE_HALF)
#  define cl_khr_fp16
#else
#  undef cl_khr_fp16
#endif

// Is long supported in OpenCL C?
// This is checked at configure-time
#ifndef _CL_DISABLE_LONG
#  define cl_khr_int64
#else
#  undef cl_khr_int64
#endif

// Is double supported?
#if defined cl_khr_int64 && __SIZEOF_DOUBLE__ == 8
#  define cl_khr_fp64
#else
#  undef cl_khr_fp64
#endif

// Architecture-specific overrides
#ifdef __TCE__
#  define __EMBEDDED_PROFILE__ 1
// TODO: Are these necessary?
#  undef cl_khr_int64
#  undef cl_khr_fp64
#endif
