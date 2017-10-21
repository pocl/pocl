#include "misc.h"




#ifdef HAVE_FMA32_32
#define HAVE_FMA32 1
#else
#define HAVE_FMA32 0
#endif
#define SINGLEVEC
#define vtype float
#define v2type v2float
#define itype int
#define utype uint
#define inttype int
#define as_vtype as_float
#define as_itype as_int
#define as_utype as_uint
#define convert_vtype convert_float
#define convert_itype convert_int
#define convert_inttype convert_int
#define convert_uinttype convert_uint
#define convert_utype convert_uint

#include "vtables.h"

#include "singlevec.h"


#include "sincos_helpers_fp32.h"
#include "cospi_fp32.cl"

#undef v2type
#undef itype4
#undef vtype
#undef itype
#undef inttype
#undef utype
#undef as_vtype
#undef as_itype
#undef as_utype
#undef convert_vtype
#undef convert_itype
#undef convert_inttype
#undef convert_uinttype
#undef convert_utype
#undef HAVE_FMA32
#undef SINGLEVEC



#ifdef HAVE_FMA32_64
#define HAVE_FMA32 1
#else
#define HAVE_FMA32 0
#endif
#define vtype float2
#define v2type v2float2
#define itype int2
#define utype uint2
#define inttype int2
#define as_vtype as_float2
#define as_itype as_int2
#define as_utype as_uint2
#define convert_vtype convert_float2
#define convert_itype convert_int2
#define convert_inttype convert_int2
#define convert_uinttype convert_uint2
#define convert_utype convert_uint2

#include "vtables.h"

#include "singlevec.h"


#include "sincos_helpers_fp32.h"
#include "cospi_fp32.cl"

#undef v2type
#undef itype4
#undef vtype
#undef itype
#undef inttype
#undef utype
#undef as_vtype
#undef as_itype
#undef as_utype
#undef convert_vtype
#undef convert_itype
#undef convert_inttype
#undef convert_uinttype
#undef convert_utype
#undef HAVE_FMA32



#ifdef HAVE_FMA32_96
#define HAVE_FMA32 1
#else
#define HAVE_FMA32 0
#endif
#define vtype float3
#define v2type v2float3
#define itype int3
#define utype uint3
#define inttype int3
#define as_vtype as_float3
#define as_itype as_int3
#define as_utype as_uint3
#define convert_vtype convert_float3
#define convert_itype convert_int3
#define convert_inttype convert_int3
#define convert_uinttype convert_uint3
#define convert_utype convert_uint3

#include "vtables.h"

#include "singlevec.h"


#include "sincos_helpers_fp32.h"
#include "cospi_fp32.cl"

#undef v2type
#undef itype4
#undef vtype
#undef itype
#undef inttype
#undef utype
#undef as_vtype
#undef as_itype
#undef as_utype
#undef convert_vtype
#undef convert_itype
#undef convert_inttype
#undef convert_uinttype
#undef convert_utype
#undef HAVE_FMA32



#ifdef HAVE_FMA32_128
#define HAVE_FMA32 1
#else
#define HAVE_FMA32 0
#endif
#define vtype float4
#define v2type v2float4
#define itype int4
#define utype uint4
#define inttype int4
#define as_vtype as_float4
#define as_itype as_int4
#define as_utype as_uint4
#define convert_vtype convert_float4
#define convert_itype convert_int4
#define convert_inttype convert_int4
#define convert_uinttype convert_uint4
#define convert_utype convert_uint4

#include "vtables.h"

#include "singlevec.h"


#include "sincos_helpers_fp32.h"
#include "cospi_fp32.cl"

#undef v2type
#undef itype4
#undef vtype
#undef itype
#undef inttype
#undef utype
#undef as_vtype
#undef as_itype
#undef as_utype
#undef convert_vtype
#undef convert_itype
#undef convert_inttype
#undef convert_uinttype
#undef convert_utype
#undef HAVE_FMA32



#ifdef HAVE_FMA32_256
#define HAVE_FMA32 1
#else
#define HAVE_FMA32 0
#endif
#define vtype float8
#define v2type v2float8
#define itype int8
#define utype uint8
#define inttype int8
#define as_vtype as_float8
#define as_itype as_int8
#define as_utype as_uint8
#define convert_vtype convert_float8
#define convert_itype convert_int8
#define convert_inttype convert_int8
#define convert_uinttype convert_uint8
#define convert_utype convert_uint8

#include "vtables.h"

#include "singlevec.h"


#include "sincos_helpers_fp32.h"
#include "cospi_fp32.cl"

#undef v2type
#undef itype4
#undef vtype
#undef itype
#undef inttype
#undef utype
#undef as_vtype
#undef as_itype
#undef as_utype
#undef convert_vtype
#undef convert_itype
#undef convert_inttype
#undef convert_uinttype
#undef convert_utype
#undef HAVE_FMA32



#ifdef HAVE_FMA32_512
#define HAVE_FMA32 1
#else
#define HAVE_FMA32 0
#endif
#define vtype float16
#define v2type v2float16
#define itype int16
#define utype uint16
#define inttype int16
#define as_vtype as_float16
#define as_itype as_int16
#define as_utype as_uint16
#define convert_vtype convert_float16
#define convert_itype convert_int16
#define convert_inttype convert_int16
#define convert_uinttype convert_uint16
#define convert_utype convert_uint16

#include "vtables.h"

#include "singlevec.h"


#include "sincos_helpers_fp32.h"
#include "cospi_fp32.cl"

#undef v2type
#undef itype4
#undef vtype
#undef itype
#undef inttype
#undef utype
#undef as_vtype
#undef as_itype
#undef as_utype
#undef convert_vtype
#undef convert_itype
#undef convert_inttype
#undef convert_uinttype
#undef convert_utype
#undef HAVE_FMA32


#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable



#ifdef HAVE_FMA64_64
#define HAVE_FMA64 1
#else
#define HAVE_FMA64 0
#endif
#define SINGLEVEC
#define vtype double
#define v2type v2double
#define itype long
#define utype ulong
#define uinttype uint
#define inttype int
#define utype4 v4uint
#define itype4 v4int
#define as_vtype as_double
#define as_itype as_long
#define as_utype as_ulong
#define convert_vtype convert_double
#define convert_itype convert_long
#define convert_inttype convert_int
#define convert_uinttype convert_uint
#define convert_utype convert_ulong

#include "vtables.h"

#include "singlevec.h"


#include "sincos_helpers_fp64.h"
#include "ep_log.h"
#include "cospi_fp64.cl"

#undef v2type
#undef itype4
#undef utype4
#undef uinttype
#undef inttype
#undef vtype
#undef itype
#undef utype
#undef as_vtype
#undef as_itype
#undef as_utype
#undef convert_vtype
#undef convert_itype
#undef convert_inttype
#undef convert_uinttype
#undef convert_utype
#undef HAVE_FMA64
#undef SINGLEVEC



#ifdef HAVE_FMA64_128
#define HAVE_FMA64 1
#else
#define HAVE_FMA64 0
#endif
#define vtype double2
#define v2type v2double2
#define itype long2
#define utype ulong2
#define uinttype uint2
#define inttype int2
#define utype4 v4uint2
#define itype4 v4int2
#define as_vtype as_double2
#define as_itype as_long2
#define as_utype as_ulong2
#define convert_vtype convert_double2
#define convert_itype convert_long2
#define convert_inttype convert_int2
#define convert_uinttype convert_uint2
#define convert_utype convert_ulong2

#include "vtables.h"

#include "singlevec.h"


#include "sincos_helpers_fp64.h"
#include "ep_log.h"
#include "cospi_fp64.cl"

#undef v2type
#undef itype4
#undef utype4
#undef uinttype
#undef inttype
#undef vtype
#undef itype
#undef utype
#undef as_vtype
#undef as_itype
#undef as_utype
#undef convert_vtype
#undef convert_itype
#undef convert_inttype
#undef convert_uinttype
#undef convert_utype
#undef HAVE_FMA64



#ifdef HAVE_FMA64_192
#define HAVE_FMA64 1
#else
#define HAVE_FMA64 0
#endif
#define vtype double3
#define v2type v2double3
#define itype long3
#define utype ulong3
#define uinttype uint3
#define inttype int3
#define utype4 v4uint3
#define itype4 v4int3
#define as_vtype as_double3
#define as_itype as_long3
#define as_utype as_ulong3
#define convert_vtype convert_double3
#define convert_itype convert_long3
#define convert_inttype convert_int3
#define convert_uinttype convert_uint3
#define convert_utype convert_ulong3

#include "vtables.h"

#include "singlevec.h"


#include "sincos_helpers_fp64.h"
#include "ep_log.h"
#include "cospi_fp64.cl"

#undef v2type
#undef itype4
#undef utype4
#undef uinttype
#undef inttype
#undef vtype
#undef itype
#undef utype
#undef as_vtype
#undef as_itype
#undef as_utype
#undef convert_vtype
#undef convert_itype
#undef convert_inttype
#undef convert_uinttype
#undef convert_utype
#undef HAVE_FMA64



#ifdef HAVE_FMA64_256
#define HAVE_FMA64 1
#else
#define HAVE_FMA64 0
#endif
#define vtype double4
#define v2type v2double4
#define itype long4
#define utype ulong4
#define uinttype uint4
#define inttype int4
#define utype4 v4uint4
#define itype4 v4int4
#define as_vtype as_double4
#define as_itype as_long4
#define as_utype as_ulong4
#define convert_vtype convert_double4
#define convert_itype convert_long4
#define convert_inttype convert_int4
#define convert_uinttype convert_uint4
#define convert_utype convert_ulong4

#include "vtables.h"

#include "singlevec.h"


#include "sincos_helpers_fp64.h"
#include "ep_log.h"
#include "cospi_fp64.cl"

#undef v2type
#undef itype4
#undef utype4
#undef uinttype
#undef inttype
#undef vtype
#undef itype
#undef utype
#undef as_vtype
#undef as_itype
#undef as_utype
#undef convert_vtype
#undef convert_itype
#undef convert_inttype
#undef convert_uinttype
#undef convert_utype
#undef HAVE_FMA64



#ifdef HAVE_FMA64_512
#define HAVE_FMA64 1
#else
#define HAVE_FMA64 0
#endif
#define vtype double8
#define v2type v2double8
#define itype long8
#define utype ulong8
#define uinttype uint8
#define inttype int8
#define utype4 v4uint8
#define itype4 v4int8
#define as_vtype as_double8
#define as_itype as_long8
#define as_utype as_ulong8
#define convert_vtype convert_double8
#define convert_itype convert_long8
#define convert_inttype convert_int8
#define convert_uinttype convert_uint8
#define convert_utype convert_ulong8

#include "vtables.h"

#include "singlevec.h"


#include "sincos_helpers_fp64.h"
#include "ep_log.h"
#include "cospi_fp64.cl"

#undef v2type
#undef itype4
#undef utype4
#undef uinttype
#undef inttype
#undef vtype
#undef itype
#undef utype
#undef as_vtype
#undef as_itype
#undef as_utype
#undef convert_vtype
#undef convert_itype
#undef convert_inttype
#undef convert_uinttype
#undef convert_utype
#undef HAVE_FMA64



#ifdef HAVE_FMA64_1024
#define HAVE_FMA64 1
#else
#define HAVE_FMA64 0
#endif
#define vtype double16
#define v2type v2double16
#define itype long16
#define utype ulong16
#define uinttype uint16
#define inttype int16
#define utype4 v4uint16
#define itype4 v4int16
#define as_vtype as_double16
#define as_itype as_long16
#define as_utype as_ulong16
#define convert_vtype convert_double16
#define convert_itype convert_long16
#define convert_inttype convert_int16
#define convert_uinttype convert_uint16
#define convert_utype convert_ulong16

#include "vtables.h"

#include "singlevec.h"


#include "sincos_helpers_fp64.h"
#include "ep_log.h"
#include "cospi_fp64.cl"

#undef v2type
#undef itype4
#undef utype4
#undef uinttype
#undef inttype
#undef vtype
#undef itype
#undef utype
#undef as_vtype
#undef as_itype
#undef as_utype
#undef convert_vtype
#undef convert_itype
#undef convert_inttype
#undef convert_uinttype
#undef convert_utype
#undef HAVE_FMA64
#endif
