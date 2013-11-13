// Supported datataypes

// LLVM always supports fp16 (aka half)
#define cl_khr_fp16

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
