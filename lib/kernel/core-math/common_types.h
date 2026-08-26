
#include "../templates.h"

typedef union {half f; ushort u;} b16u16_u;

typedef union {float f; uint u;} b32u32_u;

#if defined(cl_khr_fp64) && defined(__opencl_c_fp64)
typedef union {double f; ulong u;} b64u64_u;
#endif
