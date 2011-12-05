#ifdef __TCE_DEVICE__

#include "tce/types.h"

#else

#include "clconfig.h"

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

#if SIZEOF_LONG == 8
#  define cles_khr_int64
#  if SIZEOF_DOUBLE == 8
#    define cl_khr_fp64
#  else
#    undef cl_khr_fp64
#  endif
#else /* SIZEOF_LONG != 8 */
#  define __EMBEDDED_PROFILE__ 1
#  undef cles_khr_int64
#  undef cl_khr_fp64
#endif /* SIZEOF_LONG != 8 */

#if SIZEOF_VOID_P == 8
typedef ulong size_t;
typedef long ptrdiff_t;
typedef long intptr_t;
typedef ulong uintptr_t;
#else
typedef uint size_t;
typedef int ptrdiff_t;
typedef int intptr_t;
typedef uint uintptr_t;
#endif

#endif
