// Supported datataypes

// Note: The definitions of "long" below differs between languages. We
// therefore check "long long" instead.



// LLVM always supports fp16 (aka half)
#define cl_khr_fp16

// Is long supported?
#if __SIZEOF_LONG_LONG__ == 8
#  define cles_khr_int64
#else
#  undef cles_khr_int64
#  warning "int64 not supported"
#endif

// Is double supported?
#if defined cles_khr_int64 && __SIZEOF_DOUBLE__ == 8
#  define cl_khr_fp64
#else
#  undef cl_khr_fp64
#  warning "fp64 not supported"
#endif
