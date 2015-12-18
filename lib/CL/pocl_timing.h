


#include "config.h"

#ifndef _MSC_VER
#  ifndef __STDC_FORMAT_MACROS
#    define __STDC_FORMAT_MACROS
#  endif
#  include <inttypes.h>
#  ifdef HAVE_CLOCK_GETTIME
#    include <time.h>
#  else
#    include <sys/time.h>
#  endif
#  ifdef __MACH__
#    include <mach/clock.h>
#    include <mach/mach.h>
#  endif
#  include <sys/resource.h>
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#  include <stdint.h>
#  include <stddef.h> // size_t
#endif


extern const size_t pocl_timer_resolution;

uint64_t pocl_gettime_ns();
