#include "common_types.h"

#undef isfinite
#undef isnormal
#undef isinf
#undef isnan

DEFINE_FP16_BUILTIN_FPCLASS (isfinite, 504)

DEFINE_FP16_BUILTIN_FPCLASS (isnormal, 264)

DEFINE_FP16_BUILTIN_FPCLASS (isinf, 516)

DEFINE_FP16_BUILTIN_FPCLASS (isnan, 3)
