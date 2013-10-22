// Supported datataypes

// LLVM always supports fp16 (aka half)
#define cl_khr_fp16

// Is long supported?
// Note: The definitions of "long" below differs between languages. We
// therefore need to check "long long" instead as well.
#if __SIZEOF_LONG__ == 8 || __SIZEOF_LONG_LONG__ == 8
#  define cles_khr_int64
#else
#  undef cles_khr_int64
#endif

// Is double supported?
#if defined cles_khr_int64 && __SIZEOF_DOUBLE__ == 8
#  define cl_khr_fp64
#else
#  undef cl_khr_fp64
#endif
