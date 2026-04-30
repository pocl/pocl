#include "templates.h"

#if !defined(__has_builtin) || !__has_builtin(__builtin_readcyclecounter)
#error "__builtin_readcyclecounter is not available on this target"
#endif

ulong _CL_OVERLOADABLE _CL_CONV _cl_clock_read_device(void) {
  return __builtin_readcyclecounter();
}

ulong _CL_OVERLOADABLE _CL_CONV _cl_clock_read_work_group(void) {
  return __builtin_readcyclecounter();
}

ulong _CL_OVERLOADABLE _CL_CONV _cl_clock_read_sub_group(void) {
  return __builtin_readcyclecounter();
}

uint2 _CL_OVERLOADABLE _CL_CONV _cl_clock_read_hilo_device(void) {
  ulong clk = __builtin_readcyclecounter();
  return (uint2)((uint)(clk & 0xFFFFFFFF), (uint)(clk >> 32));
}

uint2 _CL_OVERLOADABLE _CL_CONV _cl_clock_read_hilo_work_group(void) {
  ulong clk = __builtin_readcyclecounter();
  return (uint2)((uint)(clk & 0xFFFFFFFF), (uint)(clk >> 32));
}

uint2 _CL_OVERLOADABLE _CL_CONV _cl_clock_read_hilo_sub_group(void) {
  ulong clk = __builtin_readcyclecounter();
  return (uint2)((uint)(clk & 0xFFFFFFFF), (uint)(clk >> 32));
}
