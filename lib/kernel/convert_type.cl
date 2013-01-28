
#include "templates.h"

void cl_set_rounding_mode(int mode);
void cl_set_default_rounding_mode();


  char _cl_overloadable convert_char(char a)
  {
    return (char)a;
  }


  char2 _cl_overloadable convert_char2(char2 a)
  {
    return (char2)(convert_char(a.lo), convert_char(a.hi));
  }


  char4 _cl_overloadable convert_char4(char4 a)
  {
    return (char4)(convert_char2(a.lo), convert_char2(a.hi));
  }


  char8 _cl_overloadable convert_char8(char8 a)
  {
    return (char8)(convert_char4(a.lo), convert_char4(a.hi));
  }


  char16 _cl_overloadable convert_char16(char16 a)
  {
    return (char16)(convert_char8(a.lo), convert_char8(a.hi));
  }


  char3 _cl_overloadable convert_char3(char3 a)
  {
    return (char3)(convert_char2(a.s01), convert_char(a.s2));
  }


  uchar _cl_overloadable convert_uchar(char a)
  {
    return (uchar)a;
  }


  uchar2 _cl_overloadable convert_uchar2(char2 a)
  {
    return (uchar2)(convert_uchar(a.lo), convert_uchar(a.hi));
  }


  uchar4 _cl_overloadable convert_uchar4(char4 a)
  {
    return (uchar4)(convert_uchar2(a.lo), convert_uchar2(a.hi));
  }


  uchar8 _cl_overloadable convert_uchar8(char8 a)
  {
    return (uchar8)(convert_uchar4(a.lo), convert_uchar4(a.hi));
  }


  uchar16 _cl_overloadable convert_uchar16(char16 a)
  {
    return (uchar16)(convert_uchar8(a.lo), convert_uchar8(a.hi));
  }


  uchar3 _cl_overloadable convert_uchar3(char3 a)
  {
    return (uchar3)(convert_uchar2(a.s01), convert_uchar(a.s2));
  }


  short _cl_overloadable convert_short(char a)
  {
    return (short)a;
  }


  short2 _cl_overloadable convert_short2(char2 a)
  {
    return (short2)(convert_short(a.lo), convert_short(a.hi));
  }


  short4 _cl_overloadable convert_short4(char4 a)
  {
    return (short4)(convert_short2(a.lo), convert_short2(a.hi));
  }


  short8 _cl_overloadable convert_short8(char8 a)
  {
    return (short8)(convert_short4(a.lo), convert_short4(a.hi));
  }


  short16 _cl_overloadable convert_short16(char16 a)
  {
    return (short16)(convert_short8(a.lo), convert_short8(a.hi));
  }


  short3 _cl_overloadable convert_short3(char3 a)
  {
    return (short3)(convert_short2(a.s01), convert_short(a.s2));
  }


  ushort _cl_overloadable convert_ushort(char a)
  {
    return (ushort)a;
  }


  ushort2 _cl_overloadable convert_ushort2(char2 a)
  {
    return (ushort2)(convert_ushort(a.lo), convert_ushort(a.hi));
  }


  ushort4 _cl_overloadable convert_ushort4(char4 a)
  {
    return (ushort4)(convert_ushort2(a.lo), convert_ushort2(a.hi));
  }


  ushort8 _cl_overloadable convert_ushort8(char8 a)
  {
    return (ushort8)(convert_ushort4(a.lo), convert_ushort4(a.hi));
  }


  ushort16 _cl_overloadable convert_ushort16(char16 a)
  {
    return (ushort16)(convert_ushort8(a.lo), convert_ushort8(a.hi));
  }


  ushort3 _cl_overloadable convert_ushort3(char3 a)
  {
    return (ushort3)(convert_ushort2(a.s01), convert_ushort(a.s2));
  }


  int _cl_overloadable convert_int(char a)
  {
    return (int)a;
  }


  int2 _cl_overloadable convert_int2(char2 a)
  {
    return (int2)(convert_int(a.lo), convert_int(a.hi));
  }


  int4 _cl_overloadable convert_int4(char4 a)
  {
    return (int4)(convert_int2(a.lo), convert_int2(a.hi));
  }


  int8 _cl_overloadable convert_int8(char8 a)
  {
    return (int8)(convert_int4(a.lo), convert_int4(a.hi));
  }


  int16 _cl_overloadable convert_int16(char16 a)
  {
    return (int16)(convert_int8(a.lo), convert_int8(a.hi));
  }


  int3 _cl_overloadable convert_int3(char3 a)
  {
    return (int3)(convert_int2(a.s01), convert_int(a.s2));
  }


  uint _cl_overloadable convert_uint(char a)
  {
    return (uint)a;
  }


  uint2 _cl_overloadable convert_uint2(char2 a)
  {
    return (uint2)(convert_uint(a.lo), convert_uint(a.hi));
  }


  uint4 _cl_overloadable convert_uint4(char4 a)
  {
    return (uint4)(convert_uint2(a.lo), convert_uint2(a.hi));
  }


  uint8 _cl_overloadable convert_uint8(char8 a)
  {
    return (uint8)(convert_uint4(a.lo), convert_uint4(a.hi));
  }


  uint16 _cl_overloadable convert_uint16(char16 a)
  {
    return (uint16)(convert_uint8(a.lo), convert_uint8(a.hi));
  }


  uint3 _cl_overloadable convert_uint3(char3 a)
  {
    return (uint3)(convert_uint2(a.s01), convert_uint(a.s2));
  }

__IF_INT64(

  long _cl_overloadable convert_long(char a)
  {
    return (long)a;
  }


  long2 _cl_overloadable convert_long2(char2 a)
  {
    return (long2)(convert_long(a.lo), convert_long(a.hi));
  }


  long4 _cl_overloadable convert_long4(char4 a)
  {
    return (long4)(convert_long2(a.lo), convert_long2(a.hi));
  }


  long8 _cl_overloadable convert_long8(char8 a)
  {
    return (long8)(convert_long4(a.lo), convert_long4(a.hi));
  }


  long16 _cl_overloadable convert_long16(char16 a)
  {
    return (long16)(convert_long8(a.lo), convert_long8(a.hi));
  }


  long3 _cl_overloadable convert_long3(char3 a)
  {
    return (long3)(convert_long2(a.s01), convert_long(a.s2));
  }

)
__IF_INT64(

  ulong _cl_overloadable convert_ulong(char a)
  {
    return (ulong)a;
  }


  ulong2 _cl_overloadable convert_ulong2(char2 a)
  {
    return (ulong2)(convert_ulong(a.lo), convert_ulong(a.hi));
  }


  ulong4 _cl_overloadable convert_ulong4(char4 a)
  {
    return (ulong4)(convert_ulong2(a.lo), convert_ulong2(a.hi));
  }


  ulong8 _cl_overloadable convert_ulong8(char8 a)
  {
    return (ulong8)(convert_ulong4(a.lo), convert_ulong4(a.hi));
  }


  ulong16 _cl_overloadable convert_ulong16(char16 a)
  {
    return (ulong16)(convert_ulong8(a.lo), convert_ulong8(a.hi));
  }


  ulong3 _cl_overloadable convert_ulong3(char3 a)
  {
    return (ulong3)(convert_ulong2(a.s01), convert_ulong(a.s2));
  }

)

  float _cl_overloadable convert_float(char a)
  {
    return (float)a;
  }


  float2 _cl_overloadable convert_float2(char2 a)
  {
    return (float2)(convert_float(a.lo), convert_float(a.hi));
  }


  float4 _cl_overloadable convert_float4(char4 a)
  {
    return (float4)(convert_float2(a.lo), convert_float2(a.hi));
  }


  float8 _cl_overloadable convert_float8(char8 a)
  {
    return (float8)(convert_float4(a.lo), convert_float4(a.hi));
  }


  float16 _cl_overloadable convert_float16(char16 a)
  {
    return (float16)(convert_float8(a.lo), convert_float8(a.hi));
  }


  float3 _cl_overloadable convert_float3(char3 a)
  {
    return (float3)(convert_float2(a.s01), convert_float(a.s2));
  }

__IF_FP64(

  double _cl_overloadable convert_double(char a)
  {
    return (double)a;
  }


  double2 _cl_overloadable convert_double2(char2 a)
  {
    return (double2)(convert_double(a.lo), convert_double(a.hi));
  }


  double4 _cl_overloadable convert_double4(char4 a)
  {
    return (double4)(convert_double2(a.lo), convert_double2(a.hi));
  }


  double8 _cl_overloadable convert_double8(char8 a)
  {
    return (double8)(convert_double4(a.lo), convert_double4(a.hi));
  }


  double16 _cl_overloadable convert_double16(char16 a)
  {
    return (double16)(convert_double8(a.lo), convert_double8(a.hi));
  }


  double3 _cl_overloadable convert_double3(char3 a)
  {
    return (double3)(convert_double2(a.s01), convert_double(a.s2));
  }

)

  char _cl_overloadable convert_char(uchar a)
  {
    return (char)a;
  }


  char2 _cl_overloadable convert_char2(uchar2 a)
  {
    return (char2)(convert_char(a.lo), convert_char(a.hi));
  }


  char4 _cl_overloadable convert_char4(uchar4 a)
  {
    return (char4)(convert_char2(a.lo), convert_char2(a.hi));
  }


  char8 _cl_overloadable convert_char8(uchar8 a)
  {
    return (char8)(convert_char4(a.lo), convert_char4(a.hi));
  }


  char16 _cl_overloadable convert_char16(uchar16 a)
  {
    return (char16)(convert_char8(a.lo), convert_char8(a.hi));
  }


  char3 _cl_overloadable convert_char3(uchar3 a)
  {
    return (char3)(convert_char2(a.s01), convert_char(a.s2));
  }


  uchar _cl_overloadable convert_uchar(uchar a)
  {
    return (uchar)a;
  }


  uchar2 _cl_overloadable convert_uchar2(uchar2 a)
  {
    return (uchar2)(convert_uchar(a.lo), convert_uchar(a.hi));
  }


  uchar4 _cl_overloadable convert_uchar4(uchar4 a)
  {
    return (uchar4)(convert_uchar2(a.lo), convert_uchar2(a.hi));
  }


  uchar8 _cl_overloadable convert_uchar8(uchar8 a)
  {
    return (uchar8)(convert_uchar4(a.lo), convert_uchar4(a.hi));
  }


  uchar16 _cl_overloadable convert_uchar16(uchar16 a)
  {
    return (uchar16)(convert_uchar8(a.lo), convert_uchar8(a.hi));
  }


  uchar3 _cl_overloadable convert_uchar3(uchar3 a)
  {
    return (uchar3)(convert_uchar2(a.s01), convert_uchar(a.s2));
  }


  short _cl_overloadable convert_short(uchar a)
  {
    return (short)a;
  }


  short2 _cl_overloadable convert_short2(uchar2 a)
  {
    return (short2)(convert_short(a.lo), convert_short(a.hi));
  }


  short4 _cl_overloadable convert_short4(uchar4 a)
  {
    return (short4)(convert_short2(a.lo), convert_short2(a.hi));
  }


  short8 _cl_overloadable convert_short8(uchar8 a)
  {
    return (short8)(convert_short4(a.lo), convert_short4(a.hi));
  }


  short16 _cl_overloadable convert_short16(uchar16 a)
  {
    return (short16)(convert_short8(a.lo), convert_short8(a.hi));
  }


  short3 _cl_overloadable convert_short3(uchar3 a)
  {
    return (short3)(convert_short2(a.s01), convert_short(a.s2));
  }


  ushort _cl_overloadable convert_ushort(uchar a)
  {
    return (ushort)a;
  }


  ushort2 _cl_overloadable convert_ushort2(uchar2 a)
  {
    return (ushort2)(convert_ushort(a.lo), convert_ushort(a.hi));
  }


  ushort4 _cl_overloadable convert_ushort4(uchar4 a)
  {
    return (ushort4)(convert_ushort2(a.lo), convert_ushort2(a.hi));
  }


  ushort8 _cl_overloadable convert_ushort8(uchar8 a)
  {
    return (ushort8)(convert_ushort4(a.lo), convert_ushort4(a.hi));
  }


  ushort16 _cl_overloadable convert_ushort16(uchar16 a)
  {
    return (ushort16)(convert_ushort8(a.lo), convert_ushort8(a.hi));
  }


  ushort3 _cl_overloadable convert_ushort3(uchar3 a)
  {
    return (ushort3)(convert_ushort2(a.s01), convert_ushort(a.s2));
  }


  int _cl_overloadable convert_int(uchar a)
  {
    return (int)a;
  }


  int2 _cl_overloadable convert_int2(uchar2 a)
  {
    return (int2)(convert_int(a.lo), convert_int(a.hi));
  }


  int4 _cl_overloadable convert_int4(uchar4 a)
  {
    return (int4)(convert_int2(a.lo), convert_int2(a.hi));
  }


  int8 _cl_overloadable convert_int8(uchar8 a)
  {
    return (int8)(convert_int4(a.lo), convert_int4(a.hi));
  }


  int16 _cl_overloadable convert_int16(uchar16 a)
  {
    return (int16)(convert_int8(a.lo), convert_int8(a.hi));
  }


  int3 _cl_overloadable convert_int3(uchar3 a)
  {
    return (int3)(convert_int2(a.s01), convert_int(a.s2));
  }


  uint _cl_overloadable convert_uint(uchar a)
  {
    return (uint)a;
  }


  uint2 _cl_overloadable convert_uint2(uchar2 a)
  {
    return (uint2)(convert_uint(a.lo), convert_uint(a.hi));
  }


  uint4 _cl_overloadable convert_uint4(uchar4 a)
  {
    return (uint4)(convert_uint2(a.lo), convert_uint2(a.hi));
  }


  uint8 _cl_overloadable convert_uint8(uchar8 a)
  {
    return (uint8)(convert_uint4(a.lo), convert_uint4(a.hi));
  }


  uint16 _cl_overloadable convert_uint16(uchar16 a)
  {
    return (uint16)(convert_uint8(a.lo), convert_uint8(a.hi));
  }


  uint3 _cl_overloadable convert_uint3(uchar3 a)
  {
    return (uint3)(convert_uint2(a.s01), convert_uint(a.s2));
  }

__IF_INT64(

  long _cl_overloadable convert_long(uchar a)
  {
    return (long)a;
  }


  long2 _cl_overloadable convert_long2(uchar2 a)
  {
    return (long2)(convert_long(a.lo), convert_long(a.hi));
  }


  long4 _cl_overloadable convert_long4(uchar4 a)
  {
    return (long4)(convert_long2(a.lo), convert_long2(a.hi));
  }


  long8 _cl_overloadable convert_long8(uchar8 a)
  {
    return (long8)(convert_long4(a.lo), convert_long4(a.hi));
  }


  long16 _cl_overloadable convert_long16(uchar16 a)
  {
    return (long16)(convert_long8(a.lo), convert_long8(a.hi));
  }


  long3 _cl_overloadable convert_long3(uchar3 a)
  {
    return (long3)(convert_long2(a.s01), convert_long(a.s2));
  }

)
__IF_INT64(

  ulong _cl_overloadable convert_ulong(uchar a)
  {
    return (ulong)a;
  }


  ulong2 _cl_overloadable convert_ulong2(uchar2 a)
  {
    return (ulong2)(convert_ulong(a.lo), convert_ulong(a.hi));
  }


  ulong4 _cl_overloadable convert_ulong4(uchar4 a)
  {
    return (ulong4)(convert_ulong2(a.lo), convert_ulong2(a.hi));
  }


  ulong8 _cl_overloadable convert_ulong8(uchar8 a)
  {
    return (ulong8)(convert_ulong4(a.lo), convert_ulong4(a.hi));
  }


  ulong16 _cl_overloadable convert_ulong16(uchar16 a)
  {
    return (ulong16)(convert_ulong8(a.lo), convert_ulong8(a.hi));
  }


  ulong3 _cl_overloadable convert_ulong3(uchar3 a)
  {
    return (ulong3)(convert_ulong2(a.s01), convert_ulong(a.s2));
  }

)

  float _cl_overloadable convert_float(uchar a)
  {
    return (float)a;
  }


  float2 _cl_overloadable convert_float2(uchar2 a)
  {
    return (float2)(convert_float(a.lo), convert_float(a.hi));
  }


  float4 _cl_overloadable convert_float4(uchar4 a)
  {
    return (float4)(convert_float2(a.lo), convert_float2(a.hi));
  }


  float8 _cl_overloadable convert_float8(uchar8 a)
  {
    return (float8)(convert_float4(a.lo), convert_float4(a.hi));
  }


  float16 _cl_overloadable convert_float16(uchar16 a)
  {
    return (float16)(convert_float8(a.lo), convert_float8(a.hi));
  }


  float3 _cl_overloadable convert_float3(uchar3 a)
  {
    return (float3)(convert_float2(a.s01), convert_float(a.s2));
  }

__IF_FP64(

  double _cl_overloadable convert_double(uchar a)
  {
    return (double)a;
  }


  double2 _cl_overloadable convert_double2(uchar2 a)
  {
    return (double2)(convert_double(a.lo), convert_double(a.hi));
  }


  double4 _cl_overloadable convert_double4(uchar4 a)
  {
    return (double4)(convert_double2(a.lo), convert_double2(a.hi));
  }


  double8 _cl_overloadable convert_double8(uchar8 a)
  {
    return (double8)(convert_double4(a.lo), convert_double4(a.hi));
  }


  double16 _cl_overloadable convert_double16(uchar16 a)
  {
    return (double16)(convert_double8(a.lo), convert_double8(a.hi));
  }


  double3 _cl_overloadable convert_double3(uchar3 a)
  {
    return (double3)(convert_double2(a.s01), convert_double(a.s2));
  }

)

  char _cl_overloadable convert_char(short a)
  {
    return (char)a;
  }


  char2 _cl_overloadable convert_char2(short2 a)
  {
    return (char2)(convert_char(a.lo), convert_char(a.hi));
  }


  char4 _cl_overloadable convert_char4(short4 a)
  {
    return (char4)(convert_char2(a.lo), convert_char2(a.hi));
  }


  char8 _cl_overloadable convert_char8(short8 a)
  {
    return (char8)(convert_char4(a.lo), convert_char4(a.hi));
  }


  char16 _cl_overloadable convert_char16(short16 a)
  {
    return (char16)(convert_char8(a.lo), convert_char8(a.hi));
  }


  char3 _cl_overloadable convert_char3(short3 a)
  {
    return (char3)(convert_char2(a.s01), convert_char(a.s2));
  }


  uchar _cl_overloadable convert_uchar(short a)
  {
    return (uchar)a;
  }


  uchar2 _cl_overloadable convert_uchar2(short2 a)
  {
    return (uchar2)(convert_uchar(a.lo), convert_uchar(a.hi));
  }


  uchar4 _cl_overloadable convert_uchar4(short4 a)
  {
    return (uchar4)(convert_uchar2(a.lo), convert_uchar2(a.hi));
  }


  uchar8 _cl_overloadable convert_uchar8(short8 a)
  {
    return (uchar8)(convert_uchar4(a.lo), convert_uchar4(a.hi));
  }


  uchar16 _cl_overloadable convert_uchar16(short16 a)
  {
    return (uchar16)(convert_uchar8(a.lo), convert_uchar8(a.hi));
  }


  uchar3 _cl_overloadable convert_uchar3(short3 a)
  {
    return (uchar3)(convert_uchar2(a.s01), convert_uchar(a.s2));
  }


  short _cl_overloadable convert_short(short a)
  {
    return (short)a;
  }


  short2 _cl_overloadable convert_short2(short2 a)
  {
    return (short2)(convert_short(a.lo), convert_short(a.hi));
  }


  short4 _cl_overloadable convert_short4(short4 a)
  {
    return (short4)(convert_short2(a.lo), convert_short2(a.hi));
  }


  short8 _cl_overloadable convert_short8(short8 a)
  {
    return (short8)(convert_short4(a.lo), convert_short4(a.hi));
  }


  short16 _cl_overloadable convert_short16(short16 a)
  {
    return (short16)(convert_short8(a.lo), convert_short8(a.hi));
  }


  short3 _cl_overloadable convert_short3(short3 a)
  {
    return (short3)(convert_short2(a.s01), convert_short(a.s2));
  }


  ushort _cl_overloadable convert_ushort(short a)
  {
    return (ushort)a;
  }


  ushort2 _cl_overloadable convert_ushort2(short2 a)
  {
    return (ushort2)(convert_ushort(a.lo), convert_ushort(a.hi));
  }


  ushort4 _cl_overloadable convert_ushort4(short4 a)
  {
    return (ushort4)(convert_ushort2(a.lo), convert_ushort2(a.hi));
  }


  ushort8 _cl_overloadable convert_ushort8(short8 a)
  {
    return (ushort8)(convert_ushort4(a.lo), convert_ushort4(a.hi));
  }


  ushort16 _cl_overloadable convert_ushort16(short16 a)
  {
    return (ushort16)(convert_ushort8(a.lo), convert_ushort8(a.hi));
  }


  ushort3 _cl_overloadable convert_ushort3(short3 a)
  {
    return (ushort3)(convert_ushort2(a.s01), convert_ushort(a.s2));
  }


  int _cl_overloadable convert_int(short a)
  {
    return (int)a;
  }


  int2 _cl_overloadable convert_int2(short2 a)
  {
    return (int2)(convert_int(a.lo), convert_int(a.hi));
  }


  int4 _cl_overloadable convert_int4(short4 a)
  {
    return (int4)(convert_int2(a.lo), convert_int2(a.hi));
  }


  int8 _cl_overloadable convert_int8(short8 a)
  {
    return (int8)(convert_int4(a.lo), convert_int4(a.hi));
  }


  int16 _cl_overloadable convert_int16(short16 a)
  {
    return (int16)(convert_int8(a.lo), convert_int8(a.hi));
  }


  int3 _cl_overloadable convert_int3(short3 a)
  {
    return (int3)(convert_int2(a.s01), convert_int(a.s2));
  }


  uint _cl_overloadable convert_uint(short a)
  {
    return (uint)a;
  }


  uint2 _cl_overloadable convert_uint2(short2 a)
  {
    return (uint2)(convert_uint(a.lo), convert_uint(a.hi));
  }


  uint4 _cl_overloadable convert_uint4(short4 a)
  {
    return (uint4)(convert_uint2(a.lo), convert_uint2(a.hi));
  }


  uint8 _cl_overloadable convert_uint8(short8 a)
  {
    return (uint8)(convert_uint4(a.lo), convert_uint4(a.hi));
  }


  uint16 _cl_overloadable convert_uint16(short16 a)
  {
    return (uint16)(convert_uint8(a.lo), convert_uint8(a.hi));
  }


  uint3 _cl_overloadable convert_uint3(short3 a)
  {
    return (uint3)(convert_uint2(a.s01), convert_uint(a.s2));
  }

__IF_INT64(

  long _cl_overloadable convert_long(short a)
  {
    return (long)a;
  }


  long2 _cl_overloadable convert_long2(short2 a)
  {
    return (long2)(convert_long(a.lo), convert_long(a.hi));
  }


  long4 _cl_overloadable convert_long4(short4 a)
  {
    return (long4)(convert_long2(a.lo), convert_long2(a.hi));
  }


  long8 _cl_overloadable convert_long8(short8 a)
  {
    return (long8)(convert_long4(a.lo), convert_long4(a.hi));
  }


  long16 _cl_overloadable convert_long16(short16 a)
  {
    return (long16)(convert_long8(a.lo), convert_long8(a.hi));
  }


  long3 _cl_overloadable convert_long3(short3 a)
  {
    return (long3)(convert_long2(a.s01), convert_long(a.s2));
  }

)
__IF_INT64(

  ulong _cl_overloadable convert_ulong(short a)
  {
    return (ulong)a;
  }


  ulong2 _cl_overloadable convert_ulong2(short2 a)
  {
    return (ulong2)(convert_ulong(a.lo), convert_ulong(a.hi));
  }


  ulong4 _cl_overloadable convert_ulong4(short4 a)
  {
    return (ulong4)(convert_ulong2(a.lo), convert_ulong2(a.hi));
  }


  ulong8 _cl_overloadable convert_ulong8(short8 a)
  {
    return (ulong8)(convert_ulong4(a.lo), convert_ulong4(a.hi));
  }


  ulong16 _cl_overloadable convert_ulong16(short16 a)
  {
    return (ulong16)(convert_ulong8(a.lo), convert_ulong8(a.hi));
  }


  ulong3 _cl_overloadable convert_ulong3(short3 a)
  {
    return (ulong3)(convert_ulong2(a.s01), convert_ulong(a.s2));
  }

)

  float _cl_overloadable convert_float(short a)
  {
    return (float)a;
  }


  float2 _cl_overloadable convert_float2(short2 a)
  {
    return (float2)(convert_float(a.lo), convert_float(a.hi));
  }


  float4 _cl_overloadable convert_float4(short4 a)
  {
    return (float4)(convert_float2(a.lo), convert_float2(a.hi));
  }


  float8 _cl_overloadable convert_float8(short8 a)
  {
    return (float8)(convert_float4(a.lo), convert_float4(a.hi));
  }


  float16 _cl_overloadable convert_float16(short16 a)
  {
    return (float16)(convert_float8(a.lo), convert_float8(a.hi));
  }


  float3 _cl_overloadable convert_float3(short3 a)
  {
    return (float3)(convert_float2(a.s01), convert_float(a.s2));
  }

__IF_FP64(

  double _cl_overloadable convert_double(short a)
  {
    return (double)a;
  }


  double2 _cl_overloadable convert_double2(short2 a)
  {
    return (double2)(convert_double(a.lo), convert_double(a.hi));
  }


  double4 _cl_overloadable convert_double4(short4 a)
  {
    return (double4)(convert_double2(a.lo), convert_double2(a.hi));
  }


  double8 _cl_overloadable convert_double8(short8 a)
  {
    return (double8)(convert_double4(a.lo), convert_double4(a.hi));
  }


  double16 _cl_overloadable convert_double16(short16 a)
  {
    return (double16)(convert_double8(a.lo), convert_double8(a.hi));
  }


  double3 _cl_overloadable convert_double3(short3 a)
  {
    return (double3)(convert_double2(a.s01), convert_double(a.s2));
  }

)

  char _cl_overloadable convert_char(ushort a)
  {
    return (char)a;
  }


  char2 _cl_overloadable convert_char2(ushort2 a)
  {
    return (char2)(convert_char(a.lo), convert_char(a.hi));
  }


  char4 _cl_overloadable convert_char4(ushort4 a)
  {
    return (char4)(convert_char2(a.lo), convert_char2(a.hi));
  }


  char8 _cl_overloadable convert_char8(ushort8 a)
  {
    return (char8)(convert_char4(a.lo), convert_char4(a.hi));
  }


  char16 _cl_overloadable convert_char16(ushort16 a)
  {
    return (char16)(convert_char8(a.lo), convert_char8(a.hi));
  }


  char3 _cl_overloadable convert_char3(ushort3 a)
  {
    return (char3)(convert_char2(a.s01), convert_char(a.s2));
  }


  uchar _cl_overloadable convert_uchar(ushort a)
  {
    return (uchar)a;
  }


  uchar2 _cl_overloadable convert_uchar2(ushort2 a)
  {
    return (uchar2)(convert_uchar(a.lo), convert_uchar(a.hi));
  }


  uchar4 _cl_overloadable convert_uchar4(ushort4 a)
  {
    return (uchar4)(convert_uchar2(a.lo), convert_uchar2(a.hi));
  }


  uchar8 _cl_overloadable convert_uchar8(ushort8 a)
  {
    return (uchar8)(convert_uchar4(a.lo), convert_uchar4(a.hi));
  }


  uchar16 _cl_overloadable convert_uchar16(ushort16 a)
  {
    return (uchar16)(convert_uchar8(a.lo), convert_uchar8(a.hi));
  }


  uchar3 _cl_overloadable convert_uchar3(ushort3 a)
  {
    return (uchar3)(convert_uchar2(a.s01), convert_uchar(a.s2));
  }


  short _cl_overloadable convert_short(ushort a)
  {
    return (short)a;
  }


  short2 _cl_overloadable convert_short2(ushort2 a)
  {
    return (short2)(convert_short(a.lo), convert_short(a.hi));
  }


  short4 _cl_overloadable convert_short4(ushort4 a)
  {
    return (short4)(convert_short2(a.lo), convert_short2(a.hi));
  }


  short8 _cl_overloadable convert_short8(ushort8 a)
  {
    return (short8)(convert_short4(a.lo), convert_short4(a.hi));
  }


  short16 _cl_overloadable convert_short16(ushort16 a)
  {
    return (short16)(convert_short8(a.lo), convert_short8(a.hi));
  }


  short3 _cl_overloadable convert_short3(ushort3 a)
  {
    return (short3)(convert_short2(a.s01), convert_short(a.s2));
  }


  ushort _cl_overloadable convert_ushort(ushort a)
  {
    return (ushort)a;
  }


  ushort2 _cl_overloadable convert_ushort2(ushort2 a)
  {
    return (ushort2)(convert_ushort(a.lo), convert_ushort(a.hi));
  }


  ushort4 _cl_overloadable convert_ushort4(ushort4 a)
  {
    return (ushort4)(convert_ushort2(a.lo), convert_ushort2(a.hi));
  }


  ushort8 _cl_overloadable convert_ushort8(ushort8 a)
  {
    return (ushort8)(convert_ushort4(a.lo), convert_ushort4(a.hi));
  }


  ushort16 _cl_overloadable convert_ushort16(ushort16 a)
  {
    return (ushort16)(convert_ushort8(a.lo), convert_ushort8(a.hi));
  }


  ushort3 _cl_overloadable convert_ushort3(ushort3 a)
  {
    return (ushort3)(convert_ushort2(a.s01), convert_ushort(a.s2));
  }


  int _cl_overloadable convert_int(ushort a)
  {
    return (int)a;
  }


  int2 _cl_overloadable convert_int2(ushort2 a)
  {
    return (int2)(convert_int(a.lo), convert_int(a.hi));
  }


  int4 _cl_overloadable convert_int4(ushort4 a)
  {
    return (int4)(convert_int2(a.lo), convert_int2(a.hi));
  }


  int8 _cl_overloadable convert_int8(ushort8 a)
  {
    return (int8)(convert_int4(a.lo), convert_int4(a.hi));
  }


  int16 _cl_overloadable convert_int16(ushort16 a)
  {
    return (int16)(convert_int8(a.lo), convert_int8(a.hi));
  }


  int3 _cl_overloadable convert_int3(ushort3 a)
  {
    return (int3)(convert_int2(a.s01), convert_int(a.s2));
  }


  uint _cl_overloadable convert_uint(ushort a)
  {
    return (uint)a;
  }


  uint2 _cl_overloadable convert_uint2(ushort2 a)
  {
    return (uint2)(convert_uint(a.lo), convert_uint(a.hi));
  }


  uint4 _cl_overloadable convert_uint4(ushort4 a)
  {
    return (uint4)(convert_uint2(a.lo), convert_uint2(a.hi));
  }


  uint8 _cl_overloadable convert_uint8(ushort8 a)
  {
    return (uint8)(convert_uint4(a.lo), convert_uint4(a.hi));
  }


  uint16 _cl_overloadable convert_uint16(ushort16 a)
  {
    return (uint16)(convert_uint8(a.lo), convert_uint8(a.hi));
  }


  uint3 _cl_overloadable convert_uint3(ushort3 a)
  {
    return (uint3)(convert_uint2(a.s01), convert_uint(a.s2));
  }

__IF_INT64(

  long _cl_overloadable convert_long(ushort a)
  {
    return (long)a;
  }


  long2 _cl_overloadable convert_long2(ushort2 a)
  {
    return (long2)(convert_long(a.lo), convert_long(a.hi));
  }


  long4 _cl_overloadable convert_long4(ushort4 a)
  {
    return (long4)(convert_long2(a.lo), convert_long2(a.hi));
  }


  long8 _cl_overloadable convert_long8(ushort8 a)
  {
    return (long8)(convert_long4(a.lo), convert_long4(a.hi));
  }


  long16 _cl_overloadable convert_long16(ushort16 a)
  {
    return (long16)(convert_long8(a.lo), convert_long8(a.hi));
  }


  long3 _cl_overloadable convert_long3(ushort3 a)
  {
    return (long3)(convert_long2(a.s01), convert_long(a.s2));
  }

)
__IF_INT64(

  ulong _cl_overloadable convert_ulong(ushort a)
  {
    return (ulong)a;
  }


  ulong2 _cl_overloadable convert_ulong2(ushort2 a)
  {
    return (ulong2)(convert_ulong(a.lo), convert_ulong(a.hi));
  }


  ulong4 _cl_overloadable convert_ulong4(ushort4 a)
  {
    return (ulong4)(convert_ulong2(a.lo), convert_ulong2(a.hi));
  }


  ulong8 _cl_overloadable convert_ulong8(ushort8 a)
  {
    return (ulong8)(convert_ulong4(a.lo), convert_ulong4(a.hi));
  }


  ulong16 _cl_overloadable convert_ulong16(ushort16 a)
  {
    return (ulong16)(convert_ulong8(a.lo), convert_ulong8(a.hi));
  }


  ulong3 _cl_overloadable convert_ulong3(ushort3 a)
  {
    return (ulong3)(convert_ulong2(a.s01), convert_ulong(a.s2));
  }

)

  float _cl_overloadable convert_float(ushort a)
  {
    return (float)a;
  }


  float2 _cl_overloadable convert_float2(ushort2 a)
  {
    return (float2)(convert_float(a.lo), convert_float(a.hi));
  }


  float4 _cl_overloadable convert_float4(ushort4 a)
  {
    return (float4)(convert_float2(a.lo), convert_float2(a.hi));
  }


  float8 _cl_overloadable convert_float8(ushort8 a)
  {
    return (float8)(convert_float4(a.lo), convert_float4(a.hi));
  }


  float16 _cl_overloadable convert_float16(ushort16 a)
  {
    return (float16)(convert_float8(a.lo), convert_float8(a.hi));
  }


  float3 _cl_overloadable convert_float3(ushort3 a)
  {
    return (float3)(convert_float2(a.s01), convert_float(a.s2));
  }

__IF_FP64(

  double _cl_overloadable convert_double(ushort a)
  {
    return (double)a;
  }


  double2 _cl_overloadable convert_double2(ushort2 a)
  {
    return (double2)(convert_double(a.lo), convert_double(a.hi));
  }


  double4 _cl_overloadable convert_double4(ushort4 a)
  {
    return (double4)(convert_double2(a.lo), convert_double2(a.hi));
  }


  double8 _cl_overloadable convert_double8(ushort8 a)
  {
    return (double8)(convert_double4(a.lo), convert_double4(a.hi));
  }


  double16 _cl_overloadable convert_double16(ushort16 a)
  {
    return (double16)(convert_double8(a.lo), convert_double8(a.hi));
  }


  double3 _cl_overloadable convert_double3(ushort3 a)
  {
    return (double3)(convert_double2(a.s01), convert_double(a.s2));
  }

)

  char _cl_overloadable convert_char(int a)
  {
    return (char)a;
  }


  char2 _cl_overloadable convert_char2(int2 a)
  {
    return (char2)(convert_char(a.lo), convert_char(a.hi));
  }


  char4 _cl_overloadable convert_char4(int4 a)
  {
    return (char4)(convert_char2(a.lo), convert_char2(a.hi));
  }


  char8 _cl_overloadable convert_char8(int8 a)
  {
    return (char8)(convert_char4(a.lo), convert_char4(a.hi));
  }


  char16 _cl_overloadable convert_char16(int16 a)
  {
    return (char16)(convert_char8(a.lo), convert_char8(a.hi));
  }


  char3 _cl_overloadable convert_char3(int3 a)
  {
    return (char3)(convert_char2(a.s01), convert_char(a.s2));
  }


  uchar _cl_overloadable convert_uchar(int a)
  {
    return (uchar)a;
  }


  uchar2 _cl_overloadable convert_uchar2(int2 a)
  {
    return (uchar2)(convert_uchar(a.lo), convert_uchar(a.hi));
  }


  uchar4 _cl_overloadable convert_uchar4(int4 a)
  {
    return (uchar4)(convert_uchar2(a.lo), convert_uchar2(a.hi));
  }


  uchar8 _cl_overloadable convert_uchar8(int8 a)
  {
    return (uchar8)(convert_uchar4(a.lo), convert_uchar4(a.hi));
  }


  uchar16 _cl_overloadable convert_uchar16(int16 a)
  {
    return (uchar16)(convert_uchar8(a.lo), convert_uchar8(a.hi));
  }


  uchar3 _cl_overloadable convert_uchar3(int3 a)
  {
    return (uchar3)(convert_uchar2(a.s01), convert_uchar(a.s2));
  }


  short _cl_overloadable convert_short(int a)
  {
    return (short)a;
  }


  short2 _cl_overloadable convert_short2(int2 a)
  {
    return (short2)(convert_short(a.lo), convert_short(a.hi));
  }


  short4 _cl_overloadable convert_short4(int4 a)
  {
    return (short4)(convert_short2(a.lo), convert_short2(a.hi));
  }


  short8 _cl_overloadable convert_short8(int8 a)
  {
    return (short8)(convert_short4(a.lo), convert_short4(a.hi));
  }


  short16 _cl_overloadable convert_short16(int16 a)
  {
    return (short16)(convert_short8(a.lo), convert_short8(a.hi));
  }


  short3 _cl_overloadable convert_short3(int3 a)
  {
    return (short3)(convert_short2(a.s01), convert_short(a.s2));
  }


  ushort _cl_overloadable convert_ushort(int a)
  {
    return (ushort)a;
  }


  ushort2 _cl_overloadable convert_ushort2(int2 a)
  {
    return (ushort2)(convert_ushort(a.lo), convert_ushort(a.hi));
  }


  ushort4 _cl_overloadable convert_ushort4(int4 a)
  {
    return (ushort4)(convert_ushort2(a.lo), convert_ushort2(a.hi));
  }


  ushort8 _cl_overloadable convert_ushort8(int8 a)
  {
    return (ushort8)(convert_ushort4(a.lo), convert_ushort4(a.hi));
  }


  ushort16 _cl_overloadable convert_ushort16(int16 a)
  {
    return (ushort16)(convert_ushort8(a.lo), convert_ushort8(a.hi));
  }


  ushort3 _cl_overloadable convert_ushort3(int3 a)
  {
    return (ushort3)(convert_ushort2(a.s01), convert_ushort(a.s2));
  }


  int _cl_overloadable convert_int(int a)
  {
    return (int)a;
  }


  int2 _cl_overloadable convert_int2(int2 a)
  {
    return (int2)(convert_int(a.lo), convert_int(a.hi));
  }


  int4 _cl_overloadable convert_int4(int4 a)
  {
    return (int4)(convert_int2(a.lo), convert_int2(a.hi));
  }


  int8 _cl_overloadable convert_int8(int8 a)
  {
    return (int8)(convert_int4(a.lo), convert_int4(a.hi));
  }


  int16 _cl_overloadable convert_int16(int16 a)
  {
    return (int16)(convert_int8(a.lo), convert_int8(a.hi));
  }


  int3 _cl_overloadable convert_int3(int3 a)
  {
    return (int3)(convert_int2(a.s01), convert_int(a.s2));
  }


  uint _cl_overloadable convert_uint(int a)
  {
    return (uint)a;
  }


  uint2 _cl_overloadable convert_uint2(int2 a)
  {
    return (uint2)(convert_uint(a.lo), convert_uint(a.hi));
  }


  uint4 _cl_overloadable convert_uint4(int4 a)
  {
    return (uint4)(convert_uint2(a.lo), convert_uint2(a.hi));
  }


  uint8 _cl_overloadable convert_uint8(int8 a)
  {
    return (uint8)(convert_uint4(a.lo), convert_uint4(a.hi));
  }


  uint16 _cl_overloadable convert_uint16(int16 a)
  {
    return (uint16)(convert_uint8(a.lo), convert_uint8(a.hi));
  }


  uint3 _cl_overloadable convert_uint3(int3 a)
  {
    return (uint3)(convert_uint2(a.s01), convert_uint(a.s2));
  }

__IF_INT64(

  long _cl_overloadable convert_long(int a)
  {
    return (long)a;
  }


  long2 _cl_overloadable convert_long2(int2 a)
  {
    return (long2)(convert_long(a.lo), convert_long(a.hi));
  }


  long4 _cl_overloadable convert_long4(int4 a)
  {
    return (long4)(convert_long2(a.lo), convert_long2(a.hi));
  }


  long8 _cl_overloadable convert_long8(int8 a)
  {
    return (long8)(convert_long4(a.lo), convert_long4(a.hi));
  }


  long16 _cl_overloadable convert_long16(int16 a)
  {
    return (long16)(convert_long8(a.lo), convert_long8(a.hi));
  }


  long3 _cl_overloadable convert_long3(int3 a)
  {
    return (long3)(convert_long2(a.s01), convert_long(a.s2));
  }

)
__IF_INT64(

  ulong _cl_overloadable convert_ulong(int a)
  {
    return (ulong)a;
  }


  ulong2 _cl_overloadable convert_ulong2(int2 a)
  {
    return (ulong2)(convert_ulong(a.lo), convert_ulong(a.hi));
  }


  ulong4 _cl_overloadable convert_ulong4(int4 a)
  {
    return (ulong4)(convert_ulong2(a.lo), convert_ulong2(a.hi));
  }


  ulong8 _cl_overloadable convert_ulong8(int8 a)
  {
    return (ulong8)(convert_ulong4(a.lo), convert_ulong4(a.hi));
  }


  ulong16 _cl_overloadable convert_ulong16(int16 a)
  {
    return (ulong16)(convert_ulong8(a.lo), convert_ulong8(a.hi));
  }


  ulong3 _cl_overloadable convert_ulong3(int3 a)
  {
    return (ulong3)(convert_ulong2(a.s01), convert_ulong(a.s2));
  }

)

  float _cl_overloadable convert_float(int a)
  {
    return (float)a;
  }


  float2 _cl_overloadable convert_float2(int2 a)
  {
    return (float2)(convert_float(a.lo), convert_float(a.hi));
  }


  float4 _cl_overloadable convert_float4(int4 a)
  {
    return (float4)(convert_float2(a.lo), convert_float2(a.hi));
  }


  float8 _cl_overloadable convert_float8(int8 a)
  {
    return (float8)(convert_float4(a.lo), convert_float4(a.hi));
  }


  float16 _cl_overloadable convert_float16(int16 a)
  {
    return (float16)(convert_float8(a.lo), convert_float8(a.hi));
  }


  float3 _cl_overloadable convert_float3(int3 a)
  {
    return (float3)(convert_float2(a.s01), convert_float(a.s2));
  }

__IF_FP64(

  double _cl_overloadable convert_double(int a)
  {
    return (double)a;
  }


  double2 _cl_overloadable convert_double2(int2 a)
  {
    return (double2)(convert_double(a.lo), convert_double(a.hi));
  }


  double4 _cl_overloadable convert_double4(int4 a)
  {
    return (double4)(convert_double2(a.lo), convert_double2(a.hi));
  }


  double8 _cl_overloadable convert_double8(int8 a)
  {
    return (double8)(convert_double4(a.lo), convert_double4(a.hi));
  }


  double16 _cl_overloadable convert_double16(int16 a)
  {
    return (double16)(convert_double8(a.lo), convert_double8(a.hi));
  }


  double3 _cl_overloadable convert_double3(int3 a)
  {
    return (double3)(convert_double2(a.s01), convert_double(a.s2));
  }

)

  char _cl_overloadable convert_char(uint a)
  {
    return (char)a;
  }


  char2 _cl_overloadable convert_char2(uint2 a)
  {
    return (char2)(convert_char(a.lo), convert_char(a.hi));
  }


  char4 _cl_overloadable convert_char4(uint4 a)
  {
    return (char4)(convert_char2(a.lo), convert_char2(a.hi));
  }


  char8 _cl_overloadable convert_char8(uint8 a)
  {
    return (char8)(convert_char4(a.lo), convert_char4(a.hi));
  }


  char16 _cl_overloadable convert_char16(uint16 a)
  {
    return (char16)(convert_char8(a.lo), convert_char8(a.hi));
  }


  char3 _cl_overloadable convert_char3(uint3 a)
  {
    return (char3)(convert_char2(a.s01), convert_char(a.s2));
  }


  uchar _cl_overloadable convert_uchar(uint a)
  {
    return (uchar)a;
  }


  uchar2 _cl_overloadable convert_uchar2(uint2 a)
  {
    return (uchar2)(convert_uchar(a.lo), convert_uchar(a.hi));
  }


  uchar4 _cl_overloadable convert_uchar4(uint4 a)
  {
    return (uchar4)(convert_uchar2(a.lo), convert_uchar2(a.hi));
  }


  uchar8 _cl_overloadable convert_uchar8(uint8 a)
  {
    return (uchar8)(convert_uchar4(a.lo), convert_uchar4(a.hi));
  }


  uchar16 _cl_overloadable convert_uchar16(uint16 a)
  {
    return (uchar16)(convert_uchar8(a.lo), convert_uchar8(a.hi));
  }


  uchar3 _cl_overloadable convert_uchar3(uint3 a)
  {
    return (uchar3)(convert_uchar2(a.s01), convert_uchar(a.s2));
  }


  short _cl_overloadable convert_short(uint a)
  {
    return (short)a;
  }


  short2 _cl_overloadable convert_short2(uint2 a)
  {
    return (short2)(convert_short(a.lo), convert_short(a.hi));
  }


  short4 _cl_overloadable convert_short4(uint4 a)
  {
    return (short4)(convert_short2(a.lo), convert_short2(a.hi));
  }


  short8 _cl_overloadable convert_short8(uint8 a)
  {
    return (short8)(convert_short4(a.lo), convert_short4(a.hi));
  }


  short16 _cl_overloadable convert_short16(uint16 a)
  {
    return (short16)(convert_short8(a.lo), convert_short8(a.hi));
  }


  short3 _cl_overloadable convert_short3(uint3 a)
  {
    return (short3)(convert_short2(a.s01), convert_short(a.s2));
  }


  ushort _cl_overloadable convert_ushort(uint a)
  {
    return (ushort)a;
  }


  ushort2 _cl_overloadable convert_ushort2(uint2 a)
  {
    return (ushort2)(convert_ushort(a.lo), convert_ushort(a.hi));
  }


  ushort4 _cl_overloadable convert_ushort4(uint4 a)
  {
    return (ushort4)(convert_ushort2(a.lo), convert_ushort2(a.hi));
  }


  ushort8 _cl_overloadable convert_ushort8(uint8 a)
  {
    return (ushort8)(convert_ushort4(a.lo), convert_ushort4(a.hi));
  }


  ushort16 _cl_overloadable convert_ushort16(uint16 a)
  {
    return (ushort16)(convert_ushort8(a.lo), convert_ushort8(a.hi));
  }


  ushort3 _cl_overloadable convert_ushort3(uint3 a)
  {
    return (ushort3)(convert_ushort2(a.s01), convert_ushort(a.s2));
  }


  int _cl_overloadable convert_int(uint a)
  {
    return (int)a;
  }


  int2 _cl_overloadable convert_int2(uint2 a)
  {
    return (int2)(convert_int(a.lo), convert_int(a.hi));
  }


  int4 _cl_overloadable convert_int4(uint4 a)
  {
    return (int4)(convert_int2(a.lo), convert_int2(a.hi));
  }


  int8 _cl_overloadable convert_int8(uint8 a)
  {
    return (int8)(convert_int4(a.lo), convert_int4(a.hi));
  }


  int16 _cl_overloadable convert_int16(uint16 a)
  {
    return (int16)(convert_int8(a.lo), convert_int8(a.hi));
  }


  int3 _cl_overloadable convert_int3(uint3 a)
  {
    return (int3)(convert_int2(a.s01), convert_int(a.s2));
  }


  uint _cl_overloadable convert_uint(uint a)
  {
    return (uint)a;
  }


  uint2 _cl_overloadable convert_uint2(uint2 a)
  {
    return (uint2)(convert_uint(a.lo), convert_uint(a.hi));
  }


  uint4 _cl_overloadable convert_uint4(uint4 a)
  {
    return (uint4)(convert_uint2(a.lo), convert_uint2(a.hi));
  }


  uint8 _cl_overloadable convert_uint8(uint8 a)
  {
    return (uint8)(convert_uint4(a.lo), convert_uint4(a.hi));
  }


  uint16 _cl_overloadable convert_uint16(uint16 a)
  {
    return (uint16)(convert_uint8(a.lo), convert_uint8(a.hi));
  }


  uint3 _cl_overloadable convert_uint3(uint3 a)
  {
    return (uint3)(convert_uint2(a.s01), convert_uint(a.s2));
  }

__IF_INT64(

  long _cl_overloadable convert_long(uint a)
  {
    return (long)a;
  }


  long2 _cl_overloadable convert_long2(uint2 a)
  {
    return (long2)(convert_long(a.lo), convert_long(a.hi));
  }


  long4 _cl_overloadable convert_long4(uint4 a)
  {
    return (long4)(convert_long2(a.lo), convert_long2(a.hi));
  }


  long8 _cl_overloadable convert_long8(uint8 a)
  {
    return (long8)(convert_long4(a.lo), convert_long4(a.hi));
  }


  long16 _cl_overloadable convert_long16(uint16 a)
  {
    return (long16)(convert_long8(a.lo), convert_long8(a.hi));
  }


  long3 _cl_overloadable convert_long3(uint3 a)
  {
    return (long3)(convert_long2(a.s01), convert_long(a.s2));
  }

)
__IF_INT64(

  ulong _cl_overloadable convert_ulong(uint a)
  {
    return (ulong)a;
  }


  ulong2 _cl_overloadable convert_ulong2(uint2 a)
  {
    return (ulong2)(convert_ulong(a.lo), convert_ulong(a.hi));
  }


  ulong4 _cl_overloadable convert_ulong4(uint4 a)
  {
    return (ulong4)(convert_ulong2(a.lo), convert_ulong2(a.hi));
  }


  ulong8 _cl_overloadable convert_ulong8(uint8 a)
  {
    return (ulong8)(convert_ulong4(a.lo), convert_ulong4(a.hi));
  }


  ulong16 _cl_overloadable convert_ulong16(uint16 a)
  {
    return (ulong16)(convert_ulong8(a.lo), convert_ulong8(a.hi));
  }


  ulong3 _cl_overloadable convert_ulong3(uint3 a)
  {
    return (ulong3)(convert_ulong2(a.s01), convert_ulong(a.s2));
  }

)

  float _cl_overloadable convert_float(uint a)
  {
    return (float)a;
  }


  float2 _cl_overloadable convert_float2(uint2 a)
  {
    return (float2)(convert_float(a.lo), convert_float(a.hi));
  }


  float4 _cl_overloadable convert_float4(uint4 a)
  {
    return (float4)(convert_float2(a.lo), convert_float2(a.hi));
  }


  float8 _cl_overloadable convert_float8(uint8 a)
  {
    return (float8)(convert_float4(a.lo), convert_float4(a.hi));
  }


  float16 _cl_overloadable convert_float16(uint16 a)
  {
    return (float16)(convert_float8(a.lo), convert_float8(a.hi));
  }


  float3 _cl_overloadable convert_float3(uint3 a)
  {
    return (float3)(convert_float2(a.s01), convert_float(a.s2));
  }

__IF_FP64(

  double _cl_overloadable convert_double(uint a)
  {
    return (double)a;
  }


  double2 _cl_overloadable convert_double2(uint2 a)
  {
    return (double2)(convert_double(a.lo), convert_double(a.hi));
  }


  double4 _cl_overloadable convert_double4(uint4 a)
  {
    return (double4)(convert_double2(a.lo), convert_double2(a.hi));
  }


  double8 _cl_overloadable convert_double8(uint8 a)
  {
    return (double8)(convert_double4(a.lo), convert_double4(a.hi));
  }


  double16 _cl_overloadable convert_double16(uint16 a)
  {
    return (double16)(convert_double8(a.lo), convert_double8(a.hi));
  }


  double3 _cl_overloadable convert_double3(uint3 a)
  {
    return (double3)(convert_double2(a.s01), convert_double(a.s2));
  }

)
__IF_INT64(

  char _cl_overloadable convert_char(long a)
  {
    return (char)a;
  }


  char2 _cl_overloadable convert_char2(long2 a)
  {
    return (char2)(convert_char(a.lo), convert_char(a.hi));
  }


  char4 _cl_overloadable convert_char4(long4 a)
  {
    return (char4)(convert_char2(a.lo), convert_char2(a.hi));
  }


  char8 _cl_overloadable convert_char8(long8 a)
  {
    return (char8)(convert_char4(a.lo), convert_char4(a.hi));
  }


  char16 _cl_overloadable convert_char16(long16 a)
  {
    return (char16)(convert_char8(a.lo), convert_char8(a.hi));
  }


  char3 _cl_overloadable convert_char3(long3 a)
  {
    return (char3)(convert_char2(a.s01), convert_char(a.s2));
  }

)
__IF_INT64(

  uchar _cl_overloadable convert_uchar(long a)
  {
    return (uchar)a;
  }


  uchar2 _cl_overloadable convert_uchar2(long2 a)
  {
    return (uchar2)(convert_uchar(a.lo), convert_uchar(a.hi));
  }


  uchar4 _cl_overloadable convert_uchar4(long4 a)
  {
    return (uchar4)(convert_uchar2(a.lo), convert_uchar2(a.hi));
  }


  uchar8 _cl_overloadable convert_uchar8(long8 a)
  {
    return (uchar8)(convert_uchar4(a.lo), convert_uchar4(a.hi));
  }


  uchar16 _cl_overloadable convert_uchar16(long16 a)
  {
    return (uchar16)(convert_uchar8(a.lo), convert_uchar8(a.hi));
  }


  uchar3 _cl_overloadable convert_uchar3(long3 a)
  {
    return (uchar3)(convert_uchar2(a.s01), convert_uchar(a.s2));
  }

)
__IF_INT64(

  short _cl_overloadable convert_short(long a)
  {
    return (short)a;
  }


  short2 _cl_overloadable convert_short2(long2 a)
  {
    return (short2)(convert_short(a.lo), convert_short(a.hi));
  }


  short4 _cl_overloadable convert_short4(long4 a)
  {
    return (short4)(convert_short2(a.lo), convert_short2(a.hi));
  }


  short8 _cl_overloadable convert_short8(long8 a)
  {
    return (short8)(convert_short4(a.lo), convert_short4(a.hi));
  }


  short16 _cl_overloadable convert_short16(long16 a)
  {
    return (short16)(convert_short8(a.lo), convert_short8(a.hi));
  }


  short3 _cl_overloadable convert_short3(long3 a)
  {
    return (short3)(convert_short2(a.s01), convert_short(a.s2));
  }

)
__IF_INT64(

  ushort _cl_overloadable convert_ushort(long a)
  {
    return (ushort)a;
  }


  ushort2 _cl_overloadable convert_ushort2(long2 a)
  {
    return (ushort2)(convert_ushort(a.lo), convert_ushort(a.hi));
  }


  ushort4 _cl_overloadable convert_ushort4(long4 a)
  {
    return (ushort4)(convert_ushort2(a.lo), convert_ushort2(a.hi));
  }


  ushort8 _cl_overloadable convert_ushort8(long8 a)
  {
    return (ushort8)(convert_ushort4(a.lo), convert_ushort4(a.hi));
  }


  ushort16 _cl_overloadable convert_ushort16(long16 a)
  {
    return (ushort16)(convert_ushort8(a.lo), convert_ushort8(a.hi));
  }


  ushort3 _cl_overloadable convert_ushort3(long3 a)
  {
    return (ushort3)(convert_ushort2(a.s01), convert_ushort(a.s2));
  }

)
__IF_INT64(

  int _cl_overloadable convert_int(long a)
  {
    return (int)a;
  }


  int2 _cl_overloadable convert_int2(long2 a)
  {
    return (int2)(convert_int(a.lo), convert_int(a.hi));
  }


  int4 _cl_overloadable convert_int4(long4 a)
  {
    return (int4)(convert_int2(a.lo), convert_int2(a.hi));
  }


  int8 _cl_overloadable convert_int8(long8 a)
  {
    return (int8)(convert_int4(a.lo), convert_int4(a.hi));
  }


  int16 _cl_overloadable convert_int16(long16 a)
  {
    return (int16)(convert_int8(a.lo), convert_int8(a.hi));
  }


  int3 _cl_overloadable convert_int3(long3 a)
  {
    return (int3)(convert_int2(a.s01), convert_int(a.s2));
  }

)
__IF_INT64(

  uint _cl_overloadable convert_uint(long a)
  {
    return (uint)a;
  }


  uint2 _cl_overloadable convert_uint2(long2 a)
  {
    return (uint2)(convert_uint(a.lo), convert_uint(a.hi));
  }


  uint4 _cl_overloadable convert_uint4(long4 a)
  {
    return (uint4)(convert_uint2(a.lo), convert_uint2(a.hi));
  }


  uint8 _cl_overloadable convert_uint8(long8 a)
  {
    return (uint8)(convert_uint4(a.lo), convert_uint4(a.hi));
  }


  uint16 _cl_overloadable convert_uint16(long16 a)
  {
    return (uint16)(convert_uint8(a.lo), convert_uint8(a.hi));
  }


  uint3 _cl_overloadable convert_uint3(long3 a)
  {
    return (uint3)(convert_uint2(a.s01), convert_uint(a.s2));
  }

)
__IF_INT64(

  long _cl_overloadable convert_long(long a)
  {
    return (long)a;
  }


  long2 _cl_overloadable convert_long2(long2 a)
  {
    return (long2)(convert_long(a.lo), convert_long(a.hi));
  }


  long4 _cl_overloadable convert_long4(long4 a)
  {
    return (long4)(convert_long2(a.lo), convert_long2(a.hi));
  }


  long8 _cl_overloadable convert_long8(long8 a)
  {
    return (long8)(convert_long4(a.lo), convert_long4(a.hi));
  }


  long16 _cl_overloadable convert_long16(long16 a)
  {
    return (long16)(convert_long8(a.lo), convert_long8(a.hi));
  }


  long3 _cl_overloadable convert_long3(long3 a)
  {
    return (long3)(convert_long2(a.s01), convert_long(a.s2));
  }

)
__IF_INT64(

  ulong _cl_overloadable convert_ulong(long a)
  {
    return (ulong)a;
  }


  ulong2 _cl_overloadable convert_ulong2(long2 a)
  {
    return (ulong2)(convert_ulong(a.lo), convert_ulong(a.hi));
  }


  ulong4 _cl_overloadable convert_ulong4(long4 a)
  {
    return (ulong4)(convert_ulong2(a.lo), convert_ulong2(a.hi));
  }


  ulong8 _cl_overloadable convert_ulong8(long8 a)
  {
    return (ulong8)(convert_ulong4(a.lo), convert_ulong4(a.hi));
  }


  ulong16 _cl_overloadable convert_ulong16(long16 a)
  {
    return (ulong16)(convert_ulong8(a.lo), convert_ulong8(a.hi));
  }


  ulong3 _cl_overloadable convert_ulong3(long3 a)
  {
    return (ulong3)(convert_ulong2(a.s01), convert_ulong(a.s2));
  }

)
__IF_INT64(

  float _cl_overloadable convert_float(long a)
  {
    return (float)a;
  }


  float2 _cl_overloadable convert_float2(long2 a)
  {
    return (float2)(convert_float(a.lo), convert_float(a.hi));
  }


  float4 _cl_overloadable convert_float4(long4 a)
  {
    return (float4)(convert_float2(a.lo), convert_float2(a.hi));
  }


  float8 _cl_overloadable convert_float8(long8 a)
  {
    return (float8)(convert_float4(a.lo), convert_float4(a.hi));
  }


  float16 _cl_overloadable convert_float16(long16 a)
  {
    return (float16)(convert_float8(a.lo), convert_float8(a.hi));
  }


  float3 _cl_overloadable convert_float3(long3 a)
  {
    return (float3)(convert_float2(a.s01), convert_float(a.s2));
  }

)
__IF_INT64(

  double _cl_overloadable convert_double(long a)
  {
    return (double)a;
  }


  double2 _cl_overloadable convert_double2(long2 a)
  {
    return (double2)(convert_double(a.lo), convert_double(a.hi));
  }


  double4 _cl_overloadable convert_double4(long4 a)
  {
    return (double4)(convert_double2(a.lo), convert_double2(a.hi));
  }


  double8 _cl_overloadable convert_double8(long8 a)
  {
    return (double8)(convert_double4(a.lo), convert_double4(a.hi));
  }


  double16 _cl_overloadable convert_double16(long16 a)
  {
    return (double16)(convert_double8(a.lo), convert_double8(a.hi));
  }


  double3 _cl_overloadable convert_double3(long3 a)
  {
    return (double3)(convert_double2(a.s01), convert_double(a.s2));
  }

)
__IF_INT64(

  char _cl_overloadable convert_char(ulong a)
  {
    return (char)a;
  }


  char2 _cl_overloadable convert_char2(ulong2 a)
  {
    return (char2)(convert_char(a.lo), convert_char(a.hi));
  }


  char4 _cl_overloadable convert_char4(ulong4 a)
  {
    return (char4)(convert_char2(a.lo), convert_char2(a.hi));
  }


  char8 _cl_overloadable convert_char8(ulong8 a)
  {
    return (char8)(convert_char4(a.lo), convert_char4(a.hi));
  }


  char16 _cl_overloadable convert_char16(ulong16 a)
  {
    return (char16)(convert_char8(a.lo), convert_char8(a.hi));
  }


  char3 _cl_overloadable convert_char3(ulong3 a)
  {
    return (char3)(convert_char2(a.s01), convert_char(a.s2));
  }

)
__IF_INT64(

  uchar _cl_overloadable convert_uchar(ulong a)
  {
    return (uchar)a;
  }


  uchar2 _cl_overloadable convert_uchar2(ulong2 a)
  {
    return (uchar2)(convert_uchar(a.lo), convert_uchar(a.hi));
  }


  uchar4 _cl_overloadable convert_uchar4(ulong4 a)
  {
    return (uchar4)(convert_uchar2(a.lo), convert_uchar2(a.hi));
  }


  uchar8 _cl_overloadable convert_uchar8(ulong8 a)
  {
    return (uchar8)(convert_uchar4(a.lo), convert_uchar4(a.hi));
  }


  uchar16 _cl_overloadable convert_uchar16(ulong16 a)
  {
    return (uchar16)(convert_uchar8(a.lo), convert_uchar8(a.hi));
  }


  uchar3 _cl_overloadable convert_uchar3(ulong3 a)
  {
    return (uchar3)(convert_uchar2(a.s01), convert_uchar(a.s2));
  }

)
__IF_INT64(

  short _cl_overloadable convert_short(ulong a)
  {
    return (short)a;
  }


  short2 _cl_overloadable convert_short2(ulong2 a)
  {
    return (short2)(convert_short(a.lo), convert_short(a.hi));
  }


  short4 _cl_overloadable convert_short4(ulong4 a)
  {
    return (short4)(convert_short2(a.lo), convert_short2(a.hi));
  }


  short8 _cl_overloadable convert_short8(ulong8 a)
  {
    return (short8)(convert_short4(a.lo), convert_short4(a.hi));
  }


  short16 _cl_overloadable convert_short16(ulong16 a)
  {
    return (short16)(convert_short8(a.lo), convert_short8(a.hi));
  }


  short3 _cl_overloadable convert_short3(ulong3 a)
  {
    return (short3)(convert_short2(a.s01), convert_short(a.s2));
  }

)
__IF_INT64(

  ushort _cl_overloadable convert_ushort(ulong a)
  {
    return (ushort)a;
  }


  ushort2 _cl_overloadable convert_ushort2(ulong2 a)
  {
    return (ushort2)(convert_ushort(a.lo), convert_ushort(a.hi));
  }


  ushort4 _cl_overloadable convert_ushort4(ulong4 a)
  {
    return (ushort4)(convert_ushort2(a.lo), convert_ushort2(a.hi));
  }


  ushort8 _cl_overloadable convert_ushort8(ulong8 a)
  {
    return (ushort8)(convert_ushort4(a.lo), convert_ushort4(a.hi));
  }


  ushort16 _cl_overloadable convert_ushort16(ulong16 a)
  {
    return (ushort16)(convert_ushort8(a.lo), convert_ushort8(a.hi));
  }


  ushort3 _cl_overloadable convert_ushort3(ulong3 a)
  {
    return (ushort3)(convert_ushort2(a.s01), convert_ushort(a.s2));
  }

)
__IF_INT64(

  int _cl_overloadable convert_int(ulong a)
  {
    return (int)a;
  }


  int2 _cl_overloadable convert_int2(ulong2 a)
  {
    return (int2)(convert_int(a.lo), convert_int(a.hi));
  }


  int4 _cl_overloadable convert_int4(ulong4 a)
  {
    return (int4)(convert_int2(a.lo), convert_int2(a.hi));
  }


  int8 _cl_overloadable convert_int8(ulong8 a)
  {
    return (int8)(convert_int4(a.lo), convert_int4(a.hi));
  }


  int16 _cl_overloadable convert_int16(ulong16 a)
  {
    return (int16)(convert_int8(a.lo), convert_int8(a.hi));
  }


  int3 _cl_overloadable convert_int3(ulong3 a)
  {
    return (int3)(convert_int2(a.s01), convert_int(a.s2));
  }

)
__IF_INT64(

  uint _cl_overloadable convert_uint(ulong a)
  {
    return (uint)a;
  }


  uint2 _cl_overloadable convert_uint2(ulong2 a)
  {
    return (uint2)(convert_uint(a.lo), convert_uint(a.hi));
  }


  uint4 _cl_overloadable convert_uint4(ulong4 a)
  {
    return (uint4)(convert_uint2(a.lo), convert_uint2(a.hi));
  }


  uint8 _cl_overloadable convert_uint8(ulong8 a)
  {
    return (uint8)(convert_uint4(a.lo), convert_uint4(a.hi));
  }


  uint16 _cl_overloadable convert_uint16(ulong16 a)
  {
    return (uint16)(convert_uint8(a.lo), convert_uint8(a.hi));
  }


  uint3 _cl_overloadable convert_uint3(ulong3 a)
  {
    return (uint3)(convert_uint2(a.s01), convert_uint(a.s2));
  }

)
__IF_INT64(

  long _cl_overloadable convert_long(ulong a)
  {
    return (long)a;
  }


  long2 _cl_overloadable convert_long2(ulong2 a)
  {
    return (long2)(convert_long(a.lo), convert_long(a.hi));
  }


  long4 _cl_overloadable convert_long4(ulong4 a)
  {
    return (long4)(convert_long2(a.lo), convert_long2(a.hi));
  }


  long8 _cl_overloadable convert_long8(ulong8 a)
  {
    return (long8)(convert_long4(a.lo), convert_long4(a.hi));
  }


  long16 _cl_overloadable convert_long16(ulong16 a)
  {
    return (long16)(convert_long8(a.lo), convert_long8(a.hi));
  }


  long3 _cl_overloadable convert_long3(ulong3 a)
  {
    return (long3)(convert_long2(a.s01), convert_long(a.s2));
  }

)
__IF_INT64(

  ulong _cl_overloadable convert_ulong(ulong a)
  {
    return (ulong)a;
  }


  ulong2 _cl_overloadable convert_ulong2(ulong2 a)
  {
    return (ulong2)(convert_ulong(a.lo), convert_ulong(a.hi));
  }


  ulong4 _cl_overloadable convert_ulong4(ulong4 a)
  {
    return (ulong4)(convert_ulong2(a.lo), convert_ulong2(a.hi));
  }


  ulong8 _cl_overloadable convert_ulong8(ulong8 a)
  {
    return (ulong8)(convert_ulong4(a.lo), convert_ulong4(a.hi));
  }


  ulong16 _cl_overloadable convert_ulong16(ulong16 a)
  {
    return (ulong16)(convert_ulong8(a.lo), convert_ulong8(a.hi));
  }


  ulong3 _cl_overloadable convert_ulong3(ulong3 a)
  {
    return (ulong3)(convert_ulong2(a.s01), convert_ulong(a.s2));
  }

)
__IF_INT64(

  float _cl_overloadable convert_float(ulong a)
  {
    return (float)a;
  }


  float2 _cl_overloadable convert_float2(ulong2 a)
  {
    return (float2)(convert_float(a.lo), convert_float(a.hi));
  }


  float4 _cl_overloadable convert_float4(ulong4 a)
  {
    return (float4)(convert_float2(a.lo), convert_float2(a.hi));
  }


  float8 _cl_overloadable convert_float8(ulong8 a)
  {
    return (float8)(convert_float4(a.lo), convert_float4(a.hi));
  }


  float16 _cl_overloadable convert_float16(ulong16 a)
  {
    return (float16)(convert_float8(a.lo), convert_float8(a.hi));
  }


  float3 _cl_overloadable convert_float3(ulong3 a)
  {
    return (float3)(convert_float2(a.s01), convert_float(a.s2));
  }

)
__IF_INT64(

  double _cl_overloadable convert_double(ulong a)
  {
    return (double)a;
  }


  double2 _cl_overloadable convert_double2(ulong2 a)
  {
    return (double2)(convert_double(a.lo), convert_double(a.hi));
  }


  double4 _cl_overloadable convert_double4(ulong4 a)
  {
    return (double4)(convert_double2(a.lo), convert_double2(a.hi));
  }


  double8 _cl_overloadable convert_double8(ulong8 a)
  {
    return (double8)(convert_double4(a.lo), convert_double4(a.hi));
  }


  double16 _cl_overloadable convert_double16(ulong16 a)
  {
    return (double16)(convert_double8(a.lo), convert_double8(a.hi));
  }


  double3 _cl_overloadable convert_double3(ulong3 a)
  {
    return (double3)(convert_double2(a.s01), convert_double(a.s2));
  }

)

  char _cl_overloadable convert_char(float a)
  {
    return (char)a;
  }


  char2 _cl_overloadable convert_char2(float2 a)
  {
    return (char2)(convert_char(a.lo), convert_char(a.hi));
  }


  char4 _cl_overloadable convert_char4(float4 a)
  {
    return (char4)(convert_char2(a.lo), convert_char2(a.hi));
  }


  char8 _cl_overloadable convert_char8(float8 a)
  {
    return (char8)(convert_char4(a.lo), convert_char4(a.hi));
  }


  char16 _cl_overloadable convert_char16(float16 a)
  {
    return (char16)(convert_char8(a.lo), convert_char8(a.hi));
  }


  char3 _cl_overloadable convert_char3(float3 a)
  {
    return (char3)(convert_char2(a.s01), convert_char(a.s2));
  }


  uchar _cl_overloadable convert_uchar(float a)
  {
    return (uchar)a;
  }


  uchar2 _cl_overloadable convert_uchar2(float2 a)
  {
    return (uchar2)(convert_uchar(a.lo), convert_uchar(a.hi));
  }


  uchar4 _cl_overloadable convert_uchar4(float4 a)
  {
    return (uchar4)(convert_uchar2(a.lo), convert_uchar2(a.hi));
  }


  uchar8 _cl_overloadable convert_uchar8(float8 a)
  {
    return (uchar8)(convert_uchar4(a.lo), convert_uchar4(a.hi));
  }


  uchar16 _cl_overloadable convert_uchar16(float16 a)
  {
    return (uchar16)(convert_uchar8(a.lo), convert_uchar8(a.hi));
  }


  uchar3 _cl_overloadable convert_uchar3(float3 a)
  {
    return (uchar3)(convert_uchar2(a.s01), convert_uchar(a.s2));
  }


  short _cl_overloadable convert_short(float a)
  {
    return (short)a;
  }


  short2 _cl_overloadable convert_short2(float2 a)
  {
    return (short2)(convert_short(a.lo), convert_short(a.hi));
  }


  short4 _cl_overloadable convert_short4(float4 a)
  {
    return (short4)(convert_short2(a.lo), convert_short2(a.hi));
  }


  short8 _cl_overloadable convert_short8(float8 a)
  {
    return (short8)(convert_short4(a.lo), convert_short4(a.hi));
  }


  short16 _cl_overloadable convert_short16(float16 a)
  {
    return (short16)(convert_short8(a.lo), convert_short8(a.hi));
  }


  short3 _cl_overloadable convert_short3(float3 a)
  {
    return (short3)(convert_short2(a.s01), convert_short(a.s2));
  }


  ushort _cl_overloadable convert_ushort(float a)
  {
    return (ushort)a;
  }


  ushort2 _cl_overloadable convert_ushort2(float2 a)
  {
    return (ushort2)(convert_ushort(a.lo), convert_ushort(a.hi));
  }


  ushort4 _cl_overloadable convert_ushort4(float4 a)
  {
    return (ushort4)(convert_ushort2(a.lo), convert_ushort2(a.hi));
  }


  ushort8 _cl_overloadable convert_ushort8(float8 a)
  {
    return (ushort8)(convert_ushort4(a.lo), convert_ushort4(a.hi));
  }


  ushort16 _cl_overloadable convert_ushort16(float16 a)
  {
    return (ushort16)(convert_ushort8(a.lo), convert_ushort8(a.hi));
  }


  ushort3 _cl_overloadable convert_ushort3(float3 a)
  {
    return (ushort3)(convert_ushort2(a.s01), convert_ushort(a.s2));
  }


  int _cl_overloadable convert_int(float a)
  {
    return (int)a;
  }


  int2 _cl_overloadable convert_int2(float2 a)
  {
    return (int2)(convert_int(a.lo), convert_int(a.hi));
  }


  int4 _cl_overloadable convert_int4(float4 a)
  {
    return (int4)(convert_int2(a.lo), convert_int2(a.hi));
  }


  int8 _cl_overloadable convert_int8(float8 a)
  {
    return (int8)(convert_int4(a.lo), convert_int4(a.hi));
  }


  int16 _cl_overloadable convert_int16(float16 a)
  {
    return (int16)(convert_int8(a.lo), convert_int8(a.hi));
  }


  int3 _cl_overloadable convert_int3(float3 a)
  {
    return (int3)(convert_int2(a.s01), convert_int(a.s2));
  }


  uint _cl_overloadable convert_uint(float a)
  {
    return (uint)a;
  }


  uint2 _cl_overloadable convert_uint2(float2 a)
  {
    return (uint2)(convert_uint(a.lo), convert_uint(a.hi));
  }


  uint4 _cl_overloadable convert_uint4(float4 a)
  {
    return (uint4)(convert_uint2(a.lo), convert_uint2(a.hi));
  }


  uint8 _cl_overloadable convert_uint8(float8 a)
  {
    return (uint8)(convert_uint4(a.lo), convert_uint4(a.hi));
  }


  uint16 _cl_overloadable convert_uint16(float16 a)
  {
    return (uint16)(convert_uint8(a.lo), convert_uint8(a.hi));
  }


  uint3 _cl_overloadable convert_uint3(float3 a)
  {
    return (uint3)(convert_uint2(a.s01), convert_uint(a.s2));
  }

__IF_INT64(

  long _cl_overloadable convert_long(float a)
  {
    return (long)a;
  }


  long2 _cl_overloadable convert_long2(float2 a)
  {
    return (long2)(convert_long(a.lo), convert_long(a.hi));
  }


  long4 _cl_overloadable convert_long4(float4 a)
  {
    return (long4)(convert_long2(a.lo), convert_long2(a.hi));
  }


  long8 _cl_overloadable convert_long8(float8 a)
  {
    return (long8)(convert_long4(a.lo), convert_long4(a.hi));
  }


  long16 _cl_overloadable convert_long16(float16 a)
  {
    return (long16)(convert_long8(a.lo), convert_long8(a.hi));
  }


  long3 _cl_overloadable convert_long3(float3 a)
  {
    return (long3)(convert_long2(a.s01), convert_long(a.s2));
  }

)
__IF_INT64(

  ulong _cl_overloadable convert_ulong(float a)
  {
    return (ulong)a;
  }


  ulong2 _cl_overloadable convert_ulong2(float2 a)
  {
    return (ulong2)(convert_ulong(a.lo), convert_ulong(a.hi));
  }


  ulong4 _cl_overloadable convert_ulong4(float4 a)
  {
    return (ulong4)(convert_ulong2(a.lo), convert_ulong2(a.hi));
  }


  ulong8 _cl_overloadable convert_ulong8(float8 a)
  {
    return (ulong8)(convert_ulong4(a.lo), convert_ulong4(a.hi));
  }


  ulong16 _cl_overloadable convert_ulong16(float16 a)
  {
    return (ulong16)(convert_ulong8(a.lo), convert_ulong8(a.hi));
  }


  ulong3 _cl_overloadable convert_ulong3(float3 a)
  {
    return (ulong3)(convert_ulong2(a.s01), convert_ulong(a.s2));
  }

)

  float _cl_overloadable convert_float(float a)
  {
    return (float)a;
  }


  float2 _cl_overloadable convert_float2(float2 a)
  {
    return (float2)(convert_float(a.lo), convert_float(a.hi));
  }


  float4 _cl_overloadable convert_float4(float4 a)
  {
    return (float4)(convert_float2(a.lo), convert_float2(a.hi));
  }


  float8 _cl_overloadable convert_float8(float8 a)
  {
    return (float8)(convert_float4(a.lo), convert_float4(a.hi));
  }


  float16 _cl_overloadable convert_float16(float16 a)
  {
    return (float16)(convert_float8(a.lo), convert_float8(a.hi));
  }


  float3 _cl_overloadable convert_float3(float3 a)
  {
    return (float3)(convert_float2(a.s01), convert_float(a.s2));
  }

__IF_FP64(

  double _cl_overloadable convert_double(float a)
  {
    return (double)a;
  }


  double2 _cl_overloadable convert_double2(float2 a)
  {
    return (double2)(convert_double(a.lo), convert_double(a.hi));
  }


  double4 _cl_overloadable convert_double4(float4 a)
  {
    return (double4)(convert_double2(a.lo), convert_double2(a.hi));
  }


  double8 _cl_overloadable convert_double8(float8 a)
  {
    return (double8)(convert_double4(a.lo), convert_double4(a.hi));
  }


  double16 _cl_overloadable convert_double16(float16 a)
  {
    return (double16)(convert_double8(a.lo), convert_double8(a.hi));
  }


  double3 _cl_overloadable convert_double3(float3 a)
  {
    return (double3)(convert_double2(a.s01), convert_double(a.s2));
  }

)
__IF_FP64(

  char _cl_overloadable convert_char(double a)
  {
    return (char)a;
  }


  char2 _cl_overloadable convert_char2(double2 a)
  {
    return (char2)(convert_char(a.lo), convert_char(a.hi));
  }


  char4 _cl_overloadable convert_char4(double4 a)
  {
    return (char4)(convert_char2(a.lo), convert_char2(a.hi));
  }


  char8 _cl_overloadable convert_char8(double8 a)
  {
    return (char8)(convert_char4(a.lo), convert_char4(a.hi));
  }


  char16 _cl_overloadable convert_char16(double16 a)
  {
    return (char16)(convert_char8(a.lo), convert_char8(a.hi));
  }


  char3 _cl_overloadable convert_char3(double3 a)
  {
    return (char3)(convert_char2(a.s01), convert_char(a.s2));
  }

)
__IF_FP64(

  uchar _cl_overloadable convert_uchar(double a)
  {
    return (uchar)a;
  }


  uchar2 _cl_overloadable convert_uchar2(double2 a)
  {
    return (uchar2)(convert_uchar(a.lo), convert_uchar(a.hi));
  }


  uchar4 _cl_overloadable convert_uchar4(double4 a)
  {
    return (uchar4)(convert_uchar2(a.lo), convert_uchar2(a.hi));
  }


  uchar8 _cl_overloadable convert_uchar8(double8 a)
  {
    return (uchar8)(convert_uchar4(a.lo), convert_uchar4(a.hi));
  }


  uchar16 _cl_overloadable convert_uchar16(double16 a)
  {
    return (uchar16)(convert_uchar8(a.lo), convert_uchar8(a.hi));
  }


  uchar3 _cl_overloadable convert_uchar3(double3 a)
  {
    return (uchar3)(convert_uchar2(a.s01), convert_uchar(a.s2));
  }

)
__IF_FP64(

  short _cl_overloadable convert_short(double a)
  {
    return (short)a;
  }


  short2 _cl_overloadable convert_short2(double2 a)
  {
    return (short2)(convert_short(a.lo), convert_short(a.hi));
  }


  short4 _cl_overloadable convert_short4(double4 a)
  {
    return (short4)(convert_short2(a.lo), convert_short2(a.hi));
  }


  short8 _cl_overloadable convert_short8(double8 a)
  {
    return (short8)(convert_short4(a.lo), convert_short4(a.hi));
  }


  short16 _cl_overloadable convert_short16(double16 a)
  {
    return (short16)(convert_short8(a.lo), convert_short8(a.hi));
  }


  short3 _cl_overloadable convert_short3(double3 a)
  {
    return (short3)(convert_short2(a.s01), convert_short(a.s2));
  }

)
__IF_FP64(

  ushort _cl_overloadable convert_ushort(double a)
  {
    return (ushort)a;
  }


  ushort2 _cl_overloadable convert_ushort2(double2 a)
  {
    return (ushort2)(convert_ushort(a.lo), convert_ushort(a.hi));
  }


  ushort4 _cl_overloadable convert_ushort4(double4 a)
  {
    return (ushort4)(convert_ushort2(a.lo), convert_ushort2(a.hi));
  }


  ushort8 _cl_overloadable convert_ushort8(double8 a)
  {
    return (ushort8)(convert_ushort4(a.lo), convert_ushort4(a.hi));
  }


  ushort16 _cl_overloadable convert_ushort16(double16 a)
  {
    return (ushort16)(convert_ushort8(a.lo), convert_ushort8(a.hi));
  }


  ushort3 _cl_overloadable convert_ushort3(double3 a)
  {
    return (ushort3)(convert_ushort2(a.s01), convert_ushort(a.s2));
  }

)
__IF_FP64(

  int _cl_overloadable convert_int(double a)
  {
    return (int)a;
  }


  int2 _cl_overloadable convert_int2(double2 a)
  {
    return (int2)(convert_int(a.lo), convert_int(a.hi));
  }


  int4 _cl_overloadable convert_int4(double4 a)
  {
    return (int4)(convert_int2(a.lo), convert_int2(a.hi));
  }


  int8 _cl_overloadable convert_int8(double8 a)
  {
    return (int8)(convert_int4(a.lo), convert_int4(a.hi));
  }


  int16 _cl_overloadable convert_int16(double16 a)
  {
    return (int16)(convert_int8(a.lo), convert_int8(a.hi));
  }


  int3 _cl_overloadable convert_int3(double3 a)
  {
    return (int3)(convert_int2(a.s01), convert_int(a.s2));
  }

)
__IF_FP64(

  uint _cl_overloadable convert_uint(double a)
  {
    return (uint)a;
  }


  uint2 _cl_overloadable convert_uint2(double2 a)
  {
    return (uint2)(convert_uint(a.lo), convert_uint(a.hi));
  }


  uint4 _cl_overloadable convert_uint4(double4 a)
  {
    return (uint4)(convert_uint2(a.lo), convert_uint2(a.hi));
  }


  uint8 _cl_overloadable convert_uint8(double8 a)
  {
    return (uint8)(convert_uint4(a.lo), convert_uint4(a.hi));
  }


  uint16 _cl_overloadable convert_uint16(double16 a)
  {
    return (uint16)(convert_uint8(a.lo), convert_uint8(a.hi));
  }


  uint3 _cl_overloadable convert_uint3(double3 a)
  {
    return (uint3)(convert_uint2(a.s01), convert_uint(a.s2));
  }

)
__IF_INT64(

  long _cl_overloadable convert_long(double a)
  {
    return (long)a;
  }


  long2 _cl_overloadable convert_long2(double2 a)
  {
    return (long2)(convert_long(a.lo), convert_long(a.hi));
  }


  long4 _cl_overloadable convert_long4(double4 a)
  {
    return (long4)(convert_long2(a.lo), convert_long2(a.hi));
  }


  long8 _cl_overloadable convert_long8(double8 a)
  {
    return (long8)(convert_long4(a.lo), convert_long4(a.hi));
  }


  long16 _cl_overloadable convert_long16(double16 a)
  {
    return (long16)(convert_long8(a.lo), convert_long8(a.hi));
  }


  long3 _cl_overloadable convert_long3(double3 a)
  {
    return (long3)(convert_long2(a.s01), convert_long(a.s2));
  }

)
__IF_INT64(

  ulong _cl_overloadable convert_ulong(double a)
  {
    return (ulong)a;
  }


  ulong2 _cl_overloadable convert_ulong2(double2 a)
  {
    return (ulong2)(convert_ulong(a.lo), convert_ulong(a.hi));
  }


  ulong4 _cl_overloadable convert_ulong4(double4 a)
  {
    return (ulong4)(convert_ulong2(a.lo), convert_ulong2(a.hi));
  }


  ulong8 _cl_overloadable convert_ulong8(double8 a)
  {
    return (ulong8)(convert_ulong4(a.lo), convert_ulong4(a.hi));
  }


  ulong16 _cl_overloadable convert_ulong16(double16 a)
  {
    return (ulong16)(convert_ulong8(a.lo), convert_ulong8(a.hi));
  }


  ulong3 _cl_overloadable convert_ulong3(double3 a)
  {
    return (ulong3)(convert_ulong2(a.s01), convert_ulong(a.s2));
  }

)
__IF_FP64(

  float _cl_overloadable convert_float(double a)
  {
    return (float)a;
  }


  float2 _cl_overloadable convert_float2(double2 a)
  {
    return (float2)(convert_float(a.lo), convert_float(a.hi));
  }


  float4 _cl_overloadable convert_float4(double4 a)
  {
    return (float4)(convert_float2(a.lo), convert_float2(a.hi));
  }


  float8 _cl_overloadable convert_float8(double8 a)
  {
    return (float8)(convert_float4(a.lo), convert_float4(a.hi));
  }


  float16 _cl_overloadable convert_float16(double16 a)
  {
    return (float16)(convert_float8(a.lo), convert_float8(a.hi));
  }


  float3 _cl_overloadable convert_float3(double3 a)
  {
    return (float3)(convert_float2(a.s01), convert_float(a.s2));
  }

)
__IF_FP64(

  double _cl_overloadable convert_double(double a)
  {
    return (double)a;
  }


  double2 _cl_overloadable convert_double2(double2 a)
  {
    return (double2)(convert_double(a.lo), convert_double(a.hi));
  }


  double4 _cl_overloadable convert_double4(double4 a)
  {
    return (double4)(convert_double2(a.lo), convert_double2(a.hi));
  }


  double8 _cl_overloadable convert_double8(double8 a)
  {
    return (double8)(convert_double4(a.lo), convert_double4(a.hi));
  }


  double16 _cl_overloadable convert_double16(double16 a)
  {
    return (double16)(convert_double8(a.lo), convert_double8(a.hi));
  }


  double3 _cl_overloadable convert_double3(double3 a)
  {
    return (double3)(convert_double2(a.s01), convert_double(a.s2));
  }

)
  char _cl_overloadable
  convert_char_sat(char a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char(a < (char)DST_MIN) ? (char)DST_MIN :
            convert_char(a > (char)DST_MAX) ? (char)DST_MAX :
            convert_char(a));
  }
    
  char2 _cl_overloadable
  convert_char2_sat(char2 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char2(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char2(a < (char)DST_MIN) ? (char2)DST_MIN :
            convert_char2(a > (char)DST_MAX) ? (char2)DST_MAX :
            convert_char2(a));
  }
    
  char4 _cl_overloadable
  convert_char4_sat(char4 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char4(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char4(a < (char)DST_MIN) ? (char4)DST_MIN :
            convert_char4(a > (char)DST_MAX) ? (char4)DST_MAX :
            convert_char4(a));
  }
    
  char8 _cl_overloadable
  convert_char8_sat(char8 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char8(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char8(a < (char)DST_MIN) ? (char8)DST_MIN :
            convert_char8(a > (char)DST_MAX) ? (char8)DST_MAX :
            convert_char8(a));
  }
    
  char16 _cl_overloadable
  convert_char16_sat(char16 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char16(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char16(a < (char)DST_MIN) ? (char16)DST_MIN :
            convert_char16(a > (char)DST_MAX) ? (char16)DST_MAX :
            convert_char16(a));
  }
    
  uchar _cl_overloadable
  convert_uchar_sat(char a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar(a < (char)0) ? (uchar)0 :
              convert_uchar(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar(a < (char)0      ) ? (uchar)0 :
            convert_uchar(a > (char)DST_MAX) ? (uchar)DST_MAX :
            convert_uchar(a));
  }
    
  uchar2 _cl_overloadable
  convert_uchar2_sat(char2 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar2(a < (char)0) ? (uchar2)0 :
              convert_uchar2(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar2(a < (char)0      ) ? (uchar2)0 :
            convert_uchar2(a > (char)DST_MAX) ? (uchar2)DST_MAX :
            convert_uchar2(a));
  }
    
  uchar4 _cl_overloadable
  convert_uchar4_sat(char4 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar4(a < (char)0) ? (uchar4)0 :
              convert_uchar4(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar4(a < (char)0      ) ? (uchar4)0 :
            convert_uchar4(a > (char)DST_MAX) ? (uchar4)DST_MAX :
            convert_uchar4(a));
  }
    
  uchar8 _cl_overloadable
  convert_uchar8_sat(char8 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar8(a < (char)0) ? (uchar8)0 :
              convert_uchar8(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar8(a < (char)0      ) ? (uchar8)0 :
            convert_uchar8(a > (char)DST_MAX) ? (uchar8)DST_MAX :
            convert_uchar8(a));
  }
    
  uchar16 _cl_overloadable
  convert_uchar16_sat(char16 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar16(a < (char)0) ? (uchar16)0 :
              convert_uchar16(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar16(a < (char)0      ) ? (uchar16)0 :
            convert_uchar16(a > (char)DST_MAX) ? (uchar16)DST_MAX :
            convert_uchar16(a));
  }
    
  short _cl_overloadable
  convert_short_sat(char a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short(a < (char)DST_MIN) ? (short)DST_MIN :
            convert_short(a > (char)DST_MAX) ? (short)DST_MAX :
            convert_short(a));
  }
    
  short2 _cl_overloadable
  convert_short2_sat(char2 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short2(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short2(a < (char)DST_MIN) ? (short2)DST_MIN :
            convert_short2(a > (char)DST_MAX) ? (short2)DST_MAX :
            convert_short2(a));
  }
    
  short4 _cl_overloadable
  convert_short4_sat(char4 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short4(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short4(a < (char)DST_MIN) ? (short4)DST_MIN :
            convert_short4(a > (char)DST_MAX) ? (short4)DST_MAX :
            convert_short4(a));
  }
    
  short8 _cl_overloadable
  convert_short8_sat(char8 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short8(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short8(a < (char)DST_MIN) ? (short8)DST_MIN :
            convert_short8(a > (char)DST_MAX) ? (short8)DST_MAX :
            convert_short8(a));
  }
    
  short16 _cl_overloadable
  convert_short16_sat(char16 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short16(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short16(a < (char)DST_MIN) ? (short16)DST_MIN :
            convert_short16(a > (char)DST_MAX) ? (short16)DST_MAX :
            convert_short16(a));
  }
    
  ushort _cl_overloadable
  convert_ushort_sat(char a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort(a < (char)0) ? (ushort)0 :
              convert_ushort(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort(a < (char)0      ) ? (ushort)0 :
            convert_ushort(a > (char)DST_MAX) ? (ushort)DST_MAX :
            convert_ushort(a));
  }
    
  ushort2 _cl_overloadable
  convert_ushort2_sat(char2 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort2(a < (char)0) ? (ushort2)0 :
              convert_ushort2(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort2(a < (char)0      ) ? (ushort2)0 :
            convert_ushort2(a > (char)DST_MAX) ? (ushort2)DST_MAX :
            convert_ushort2(a));
  }
    
  ushort4 _cl_overloadable
  convert_ushort4_sat(char4 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort4(a < (char)0) ? (ushort4)0 :
              convert_ushort4(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort4(a < (char)0      ) ? (ushort4)0 :
            convert_ushort4(a > (char)DST_MAX) ? (ushort4)DST_MAX :
            convert_ushort4(a));
  }
    
  ushort8 _cl_overloadable
  convert_ushort8_sat(char8 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort8(a < (char)0) ? (ushort8)0 :
              convert_ushort8(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort8(a < (char)0      ) ? (ushort8)0 :
            convert_ushort8(a > (char)DST_MAX) ? (ushort8)DST_MAX :
            convert_ushort8(a));
  }
    
  ushort16 _cl_overloadable
  convert_ushort16_sat(char16 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort16(a < (char)0) ? (ushort16)0 :
              convert_ushort16(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort16(a < (char)0      ) ? (ushort16)0 :
            convert_ushort16(a > (char)DST_MAX) ? (ushort16)DST_MAX :
            convert_ushort16(a));
  }
    
  int _cl_overloadable
  convert_int_sat(char a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int(a < (char)DST_MIN) ? (int)DST_MIN :
            convert_int(a > (char)DST_MAX) ? (int)DST_MAX :
            convert_int(a));
  }
    
  int2 _cl_overloadable
  convert_int2_sat(char2 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int2(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int2(a < (char)DST_MIN) ? (int2)DST_MIN :
            convert_int2(a > (char)DST_MAX) ? (int2)DST_MAX :
            convert_int2(a));
  }
    
  int4 _cl_overloadable
  convert_int4_sat(char4 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int4(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int4(a < (char)DST_MIN) ? (int4)DST_MIN :
            convert_int4(a > (char)DST_MAX) ? (int4)DST_MAX :
            convert_int4(a));
  }
    
  int8 _cl_overloadable
  convert_int8_sat(char8 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int8(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int8(a < (char)DST_MIN) ? (int8)DST_MIN :
            convert_int8(a > (char)DST_MAX) ? (int8)DST_MAX :
            convert_int8(a));
  }
    
  int16 _cl_overloadable
  convert_int16_sat(char16 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int16(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int16(a < (char)DST_MIN) ? (int16)DST_MIN :
            convert_int16(a > (char)DST_MAX) ? (int16)DST_MAX :
            convert_int16(a));
  }
    
  uint _cl_overloadable
  convert_uint_sat(char a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint(a < (char)0) ? (uint)0 :
              convert_uint(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint(a < (char)0      ) ? (uint)0 :
            convert_uint(a > (char)DST_MAX) ? (uint)DST_MAX :
            convert_uint(a));
  }
    
  uint2 _cl_overloadable
  convert_uint2_sat(char2 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint2(a < (char)0) ? (uint2)0 :
              convert_uint2(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint2(a < (char)0      ) ? (uint2)0 :
            convert_uint2(a > (char)DST_MAX) ? (uint2)DST_MAX :
            convert_uint2(a));
  }
    
  uint4 _cl_overloadable
  convert_uint4_sat(char4 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint4(a < (char)0) ? (uint4)0 :
              convert_uint4(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint4(a < (char)0      ) ? (uint4)0 :
            convert_uint4(a > (char)DST_MAX) ? (uint4)DST_MAX :
            convert_uint4(a));
  }
    
  uint8 _cl_overloadable
  convert_uint8_sat(char8 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint8(a < (char)0) ? (uint8)0 :
              convert_uint8(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint8(a < (char)0      ) ? (uint8)0 :
            convert_uint8(a > (char)DST_MAX) ? (uint8)DST_MAX :
            convert_uint8(a));
  }
    
  uint16 _cl_overloadable
  convert_uint16_sat(char16 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint16(a < (char)0) ? (uint16)0 :
              convert_uint16(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint16(a < (char)0      ) ? (uint16)0 :
            convert_uint16(a > (char)DST_MAX) ? (uint16)DST_MAX :
            convert_uint16(a));
  }
    
__IF_INT64(
  long _cl_overloadable
  convert_long_sat(char a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long(a < (char)DST_MIN) ? (long)DST_MIN :
            convert_long(a > (char)DST_MAX) ? (long)DST_MAX :
            convert_long(a));
  }
    
  long2 _cl_overloadable
  convert_long2_sat(char2 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long2(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long2(a < (char)DST_MIN) ? (long2)DST_MIN :
            convert_long2(a > (char)DST_MAX) ? (long2)DST_MAX :
            convert_long2(a));
  }
    
  long4 _cl_overloadable
  convert_long4_sat(char4 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long4(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long4(a < (char)DST_MIN) ? (long4)DST_MIN :
            convert_long4(a > (char)DST_MAX) ? (long4)DST_MAX :
            convert_long4(a));
  }
    
  long8 _cl_overloadable
  convert_long8_sat(char8 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long8(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long8(a < (char)DST_MIN) ? (long8)DST_MIN :
            convert_long8(a > (char)DST_MAX) ? (long8)DST_MAX :
            convert_long8(a));
  }
    
  long16 _cl_overloadable
  convert_long16_sat(char16 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long16(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long16(a < (char)DST_MIN) ? (long16)DST_MIN :
            convert_long16(a > (char)DST_MAX) ? (long16)DST_MAX :
            convert_long16(a));
  }
    
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat(char a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong(a < (char)0) ? (ulong)0 :
              convert_ulong(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong(a < (char)0      ) ? (ulong)0 :
            convert_ulong(a > (char)DST_MAX) ? (ulong)DST_MAX :
            convert_ulong(a));
  }
    
  ulong2 _cl_overloadable
  convert_ulong2_sat(char2 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong2(a < (char)0) ? (ulong2)0 :
              convert_ulong2(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong2(a < (char)0      ) ? (ulong2)0 :
            convert_ulong2(a > (char)DST_MAX) ? (ulong2)DST_MAX :
            convert_ulong2(a));
  }
    
  ulong4 _cl_overloadable
  convert_ulong4_sat(char4 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong4(a < (char)0) ? (ulong4)0 :
              convert_ulong4(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong4(a < (char)0      ) ? (ulong4)0 :
            convert_ulong4(a > (char)DST_MAX) ? (ulong4)DST_MAX :
            convert_ulong4(a));
  }
    
  ulong8 _cl_overloadable
  convert_ulong8_sat(char8 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong8(a < (char)0) ? (ulong8)0 :
              convert_ulong8(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong8(a < (char)0      ) ? (ulong8)0 :
            convert_ulong8(a > (char)DST_MAX) ? (ulong8)DST_MAX :
            convert_ulong8(a));
  }
    
  ulong16 _cl_overloadable
  convert_ulong16_sat(char16 a)
  {
    int const src_size = sizeof(char);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong16(a < (char)0) ? (ulong16)0 :
              convert_ulong16(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong16(a < (char)0      ) ? (ulong16)0 :
            convert_ulong16(a > (char)DST_MAX) ? (ulong16)DST_MAX :
            convert_ulong16(a));
  }
    
)
  char _cl_overloadable
  convert_char_sat(uchar a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char(a > (uchar)DST_MAX) ? (char)DST_MAX :
            convert_char(a));
  }
    
  char2 _cl_overloadable
  convert_char2_sat(uchar2 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char2(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char2(a > (uchar)DST_MAX) ? (char2)DST_MAX :
            convert_char2(a));
  }
    
  char4 _cl_overloadable
  convert_char4_sat(uchar4 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char4(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char4(a > (uchar)DST_MAX) ? (char4)DST_MAX :
            convert_char4(a));
  }
    
  char8 _cl_overloadable
  convert_char8_sat(uchar8 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char8(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char8(a > (uchar)DST_MAX) ? (char8)DST_MAX :
            convert_char8(a));
  }
    
  char16 _cl_overloadable
  convert_char16_sat(uchar16 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char16(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char16(a > (uchar)DST_MAX) ? (char16)DST_MAX :
            convert_char16(a));
  }
    
  uchar _cl_overloadable
  convert_uchar_sat(uchar a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar(a > (uchar)DST_MAX) ? (uchar)DST_MAX :
            convert_uchar(a));
  }
    
  uchar2 _cl_overloadable
  convert_uchar2_sat(uchar2 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar2(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar2(a > (uchar)DST_MAX) ? (uchar2)DST_MAX :
            convert_uchar2(a));
  }
    
  uchar4 _cl_overloadable
  convert_uchar4_sat(uchar4 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar4(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar4(a > (uchar)DST_MAX) ? (uchar4)DST_MAX :
            convert_uchar4(a));
  }
    
  uchar8 _cl_overloadable
  convert_uchar8_sat(uchar8 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar8(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar8(a > (uchar)DST_MAX) ? (uchar8)DST_MAX :
            convert_uchar8(a));
  }
    
  uchar16 _cl_overloadable
  convert_uchar16_sat(uchar16 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar16(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar16(a > (uchar)DST_MAX) ? (uchar16)DST_MAX :
            convert_uchar16(a));
  }
    
  short _cl_overloadable
  convert_short_sat(uchar a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short(a > (uchar)DST_MAX) ? (short)DST_MAX :
            convert_short(a));
  }
    
  short2 _cl_overloadable
  convert_short2_sat(uchar2 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short2(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short2(a > (uchar)DST_MAX) ? (short2)DST_MAX :
            convert_short2(a));
  }
    
  short4 _cl_overloadable
  convert_short4_sat(uchar4 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short4(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short4(a > (uchar)DST_MAX) ? (short4)DST_MAX :
            convert_short4(a));
  }
    
  short8 _cl_overloadable
  convert_short8_sat(uchar8 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short8(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short8(a > (uchar)DST_MAX) ? (short8)DST_MAX :
            convert_short8(a));
  }
    
  short16 _cl_overloadable
  convert_short16_sat(uchar16 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short16(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short16(a > (uchar)DST_MAX) ? (short16)DST_MAX :
            convert_short16(a));
  }
    
  ushort _cl_overloadable
  convert_ushort_sat(uchar a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort(a > (uchar)DST_MAX) ? (ushort)DST_MAX :
            convert_ushort(a));
  }
    
  ushort2 _cl_overloadable
  convert_ushort2_sat(uchar2 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort2(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort2(a > (uchar)DST_MAX) ? (ushort2)DST_MAX :
            convert_ushort2(a));
  }
    
  ushort4 _cl_overloadable
  convert_ushort4_sat(uchar4 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort4(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort4(a > (uchar)DST_MAX) ? (ushort4)DST_MAX :
            convert_ushort4(a));
  }
    
  ushort8 _cl_overloadable
  convert_ushort8_sat(uchar8 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort8(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort8(a > (uchar)DST_MAX) ? (ushort8)DST_MAX :
            convert_ushort8(a));
  }
    
  ushort16 _cl_overloadable
  convert_ushort16_sat(uchar16 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort16(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort16(a > (uchar)DST_MAX) ? (ushort16)DST_MAX :
            convert_ushort16(a));
  }
    
  int _cl_overloadable
  convert_int_sat(uchar a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int(a > (uchar)DST_MAX) ? (int)DST_MAX :
            convert_int(a));
  }
    
  int2 _cl_overloadable
  convert_int2_sat(uchar2 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int2(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int2(a > (uchar)DST_MAX) ? (int2)DST_MAX :
            convert_int2(a));
  }
    
  int4 _cl_overloadable
  convert_int4_sat(uchar4 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int4(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int4(a > (uchar)DST_MAX) ? (int4)DST_MAX :
            convert_int4(a));
  }
    
  int8 _cl_overloadable
  convert_int8_sat(uchar8 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int8(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int8(a > (uchar)DST_MAX) ? (int8)DST_MAX :
            convert_int8(a));
  }
    
  int16 _cl_overloadable
  convert_int16_sat(uchar16 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int16(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int16(a > (uchar)DST_MAX) ? (int16)DST_MAX :
            convert_int16(a));
  }
    
  uint _cl_overloadable
  convert_uint_sat(uchar a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint(a > (uchar)DST_MAX) ? (uint)DST_MAX :
            convert_uint(a));
  }
    
  uint2 _cl_overloadable
  convert_uint2_sat(uchar2 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint2(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint2(a > (uchar)DST_MAX) ? (uint2)DST_MAX :
            convert_uint2(a));
  }
    
  uint4 _cl_overloadable
  convert_uint4_sat(uchar4 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint4(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint4(a > (uchar)DST_MAX) ? (uint4)DST_MAX :
            convert_uint4(a));
  }
    
  uint8 _cl_overloadable
  convert_uint8_sat(uchar8 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint8(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint8(a > (uchar)DST_MAX) ? (uint8)DST_MAX :
            convert_uint8(a));
  }
    
  uint16 _cl_overloadable
  convert_uint16_sat(uchar16 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint16(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint16(a > (uchar)DST_MAX) ? (uint16)DST_MAX :
            convert_uint16(a));
  }
    
__IF_INT64(
  long _cl_overloadable
  convert_long_sat(uchar a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long(a > (uchar)DST_MAX) ? (long)DST_MAX :
            convert_long(a));
  }
    
  long2 _cl_overloadable
  convert_long2_sat(uchar2 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long2(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long2(a > (uchar)DST_MAX) ? (long2)DST_MAX :
            convert_long2(a));
  }
    
  long4 _cl_overloadable
  convert_long4_sat(uchar4 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long4(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long4(a > (uchar)DST_MAX) ? (long4)DST_MAX :
            convert_long4(a));
  }
    
  long8 _cl_overloadable
  convert_long8_sat(uchar8 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long8(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long8(a > (uchar)DST_MAX) ? (long8)DST_MAX :
            convert_long8(a));
  }
    
  long16 _cl_overloadable
  convert_long16_sat(uchar16 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long16(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long16(a > (uchar)DST_MAX) ? (long16)DST_MAX :
            convert_long16(a));
  }
    
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat(uchar a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong(a > (uchar)DST_MAX) ? (ulong)DST_MAX :
            convert_ulong(a));
  }
    
  ulong2 _cl_overloadable
  convert_ulong2_sat(uchar2 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong2(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong2(a > (uchar)DST_MAX) ? (ulong2)DST_MAX :
            convert_ulong2(a));
  }
    
  ulong4 _cl_overloadable
  convert_ulong4_sat(uchar4 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong4(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong4(a > (uchar)DST_MAX) ? (ulong4)DST_MAX :
            convert_ulong4(a));
  }
    
  ulong8 _cl_overloadable
  convert_ulong8_sat(uchar8 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong8(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong8(a > (uchar)DST_MAX) ? (ulong8)DST_MAX :
            convert_ulong8(a));
  }
    
  ulong16 _cl_overloadable
  convert_ulong16_sat(uchar16 a)
  {
    int const src_size = sizeof(uchar);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong16(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong16(a > (uchar)DST_MAX) ? (ulong16)DST_MAX :
            convert_ulong16(a));
  }
    
)
  char _cl_overloadable
  convert_char_sat(short a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char(a < (short)DST_MIN) ? (char)DST_MIN :
            convert_char(a > (short)DST_MAX) ? (char)DST_MAX :
            convert_char(a));
  }
    
  char2 _cl_overloadable
  convert_char2_sat(short2 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char2(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char2(a < (short)DST_MIN) ? (char2)DST_MIN :
            convert_char2(a > (short)DST_MAX) ? (char2)DST_MAX :
            convert_char2(a));
  }
    
  char4 _cl_overloadable
  convert_char4_sat(short4 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char4(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char4(a < (short)DST_MIN) ? (char4)DST_MIN :
            convert_char4(a > (short)DST_MAX) ? (char4)DST_MAX :
            convert_char4(a));
  }
    
  char8 _cl_overloadable
  convert_char8_sat(short8 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char8(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char8(a < (short)DST_MIN) ? (char8)DST_MIN :
            convert_char8(a > (short)DST_MAX) ? (char8)DST_MAX :
            convert_char8(a));
  }
    
  char16 _cl_overloadable
  convert_char16_sat(short16 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char16(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char16(a < (short)DST_MIN) ? (char16)DST_MIN :
            convert_char16(a > (short)DST_MAX) ? (char16)DST_MAX :
            convert_char16(a));
  }
    
  uchar _cl_overloadable
  convert_uchar_sat(short a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar(a < (short)0) ? (uchar)0 :
              convert_uchar(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar(a < (short)0      ) ? (uchar)0 :
            convert_uchar(a > (short)DST_MAX) ? (uchar)DST_MAX :
            convert_uchar(a));
  }
    
  uchar2 _cl_overloadable
  convert_uchar2_sat(short2 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar2(a < (short)0) ? (uchar2)0 :
              convert_uchar2(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar2(a < (short)0      ) ? (uchar2)0 :
            convert_uchar2(a > (short)DST_MAX) ? (uchar2)DST_MAX :
            convert_uchar2(a));
  }
    
  uchar4 _cl_overloadable
  convert_uchar4_sat(short4 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar4(a < (short)0) ? (uchar4)0 :
              convert_uchar4(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar4(a < (short)0      ) ? (uchar4)0 :
            convert_uchar4(a > (short)DST_MAX) ? (uchar4)DST_MAX :
            convert_uchar4(a));
  }
    
  uchar8 _cl_overloadable
  convert_uchar8_sat(short8 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar8(a < (short)0) ? (uchar8)0 :
              convert_uchar8(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar8(a < (short)0      ) ? (uchar8)0 :
            convert_uchar8(a > (short)DST_MAX) ? (uchar8)DST_MAX :
            convert_uchar8(a));
  }
    
  uchar16 _cl_overloadable
  convert_uchar16_sat(short16 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar16(a < (short)0) ? (uchar16)0 :
              convert_uchar16(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar16(a < (short)0      ) ? (uchar16)0 :
            convert_uchar16(a > (short)DST_MAX) ? (uchar16)DST_MAX :
            convert_uchar16(a));
  }
    
  short _cl_overloadable
  convert_short_sat(short a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short(a < (short)DST_MIN) ? (short)DST_MIN :
            convert_short(a > (short)DST_MAX) ? (short)DST_MAX :
            convert_short(a));
  }
    
  short2 _cl_overloadable
  convert_short2_sat(short2 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short2(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short2(a < (short)DST_MIN) ? (short2)DST_MIN :
            convert_short2(a > (short)DST_MAX) ? (short2)DST_MAX :
            convert_short2(a));
  }
    
  short4 _cl_overloadable
  convert_short4_sat(short4 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short4(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short4(a < (short)DST_MIN) ? (short4)DST_MIN :
            convert_short4(a > (short)DST_MAX) ? (short4)DST_MAX :
            convert_short4(a));
  }
    
  short8 _cl_overloadable
  convert_short8_sat(short8 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short8(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short8(a < (short)DST_MIN) ? (short8)DST_MIN :
            convert_short8(a > (short)DST_MAX) ? (short8)DST_MAX :
            convert_short8(a));
  }
    
  short16 _cl_overloadable
  convert_short16_sat(short16 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short16(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short16(a < (short)DST_MIN) ? (short16)DST_MIN :
            convert_short16(a > (short)DST_MAX) ? (short16)DST_MAX :
            convert_short16(a));
  }
    
  ushort _cl_overloadable
  convert_ushort_sat(short a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort(a < (short)0) ? (ushort)0 :
              convert_ushort(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort(a < (short)0      ) ? (ushort)0 :
            convert_ushort(a > (short)DST_MAX) ? (ushort)DST_MAX :
            convert_ushort(a));
  }
    
  ushort2 _cl_overloadable
  convert_ushort2_sat(short2 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort2(a < (short)0) ? (ushort2)0 :
              convert_ushort2(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort2(a < (short)0      ) ? (ushort2)0 :
            convert_ushort2(a > (short)DST_MAX) ? (ushort2)DST_MAX :
            convert_ushort2(a));
  }
    
  ushort4 _cl_overloadable
  convert_ushort4_sat(short4 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort4(a < (short)0) ? (ushort4)0 :
              convert_ushort4(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort4(a < (short)0      ) ? (ushort4)0 :
            convert_ushort4(a > (short)DST_MAX) ? (ushort4)DST_MAX :
            convert_ushort4(a));
  }
    
  ushort8 _cl_overloadable
  convert_ushort8_sat(short8 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort8(a < (short)0) ? (ushort8)0 :
              convert_ushort8(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort8(a < (short)0      ) ? (ushort8)0 :
            convert_ushort8(a > (short)DST_MAX) ? (ushort8)DST_MAX :
            convert_ushort8(a));
  }
    
  ushort16 _cl_overloadable
  convert_ushort16_sat(short16 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort16(a < (short)0) ? (ushort16)0 :
              convert_ushort16(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort16(a < (short)0      ) ? (ushort16)0 :
            convert_ushort16(a > (short)DST_MAX) ? (ushort16)DST_MAX :
            convert_ushort16(a));
  }
    
  int _cl_overloadable
  convert_int_sat(short a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int(a < (short)DST_MIN) ? (int)DST_MIN :
            convert_int(a > (short)DST_MAX) ? (int)DST_MAX :
            convert_int(a));
  }
    
  int2 _cl_overloadable
  convert_int2_sat(short2 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int2(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int2(a < (short)DST_MIN) ? (int2)DST_MIN :
            convert_int2(a > (short)DST_MAX) ? (int2)DST_MAX :
            convert_int2(a));
  }
    
  int4 _cl_overloadable
  convert_int4_sat(short4 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int4(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int4(a < (short)DST_MIN) ? (int4)DST_MIN :
            convert_int4(a > (short)DST_MAX) ? (int4)DST_MAX :
            convert_int4(a));
  }
    
  int8 _cl_overloadable
  convert_int8_sat(short8 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int8(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int8(a < (short)DST_MIN) ? (int8)DST_MIN :
            convert_int8(a > (short)DST_MAX) ? (int8)DST_MAX :
            convert_int8(a));
  }
    
  int16 _cl_overloadable
  convert_int16_sat(short16 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int16(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int16(a < (short)DST_MIN) ? (int16)DST_MIN :
            convert_int16(a > (short)DST_MAX) ? (int16)DST_MAX :
            convert_int16(a));
  }
    
  uint _cl_overloadable
  convert_uint_sat(short a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint(a < (short)0) ? (uint)0 :
              convert_uint(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint(a < (short)0      ) ? (uint)0 :
            convert_uint(a > (short)DST_MAX) ? (uint)DST_MAX :
            convert_uint(a));
  }
    
  uint2 _cl_overloadable
  convert_uint2_sat(short2 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint2(a < (short)0) ? (uint2)0 :
              convert_uint2(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint2(a < (short)0      ) ? (uint2)0 :
            convert_uint2(a > (short)DST_MAX) ? (uint2)DST_MAX :
            convert_uint2(a));
  }
    
  uint4 _cl_overloadable
  convert_uint4_sat(short4 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint4(a < (short)0) ? (uint4)0 :
              convert_uint4(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint4(a < (short)0      ) ? (uint4)0 :
            convert_uint4(a > (short)DST_MAX) ? (uint4)DST_MAX :
            convert_uint4(a));
  }
    
  uint8 _cl_overloadable
  convert_uint8_sat(short8 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint8(a < (short)0) ? (uint8)0 :
              convert_uint8(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint8(a < (short)0      ) ? (uint8)0 :
            convert_uint8(a > (short)DST_MAX) ? (uint8)DST_MAX :
            convert_uint8(a));
  }
    
  uint16 _cl_overloadable
  convert_uint16_sat(short16 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint16(a < (short)0) ? (uint16)0 :
              convert_uint16(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint16(a < (short)0      ) ? (uint16)0 :
            convert_uint16(a > (short)DST_MAX) ? (uint16)DST_MAX :
            convert_uint16(a));
  }
    
__IF_INT64(
  long _cl_overloadable
  convert_long_sat(short a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long(a < (short)DST_MIN) ? (long)DST_MIN :
            convert_long(a > (short)DST_MAX) ? (long)DST_MAX :
            convert_long(a));
  }
    
  long2 _cl_overloadable
  convert_long2_sat(short2 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long2(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long2(a < (short)DST_MIN) ? (long2)DST_MIN :
            convert_long2(a > (short)DST_MAX) ? (long2)DST_MAX :
            convert_long2(a));
  }
    
  long4 _cl_overloadable
  convert_long4_sat(short4 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long4(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long4(a < (short)DST_MIN) ? (long4)DST_MIN :
            convert_long4(a > (short)DST_MAX) ? (long4)DST_MAX :
            convert_long4(a));
  }
    
  long8 _cl_overloadable
  convert_long8_sat(short8 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long8(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long8(a < (short)DST_MIN) ? (long8)DST_MIN :
            convert_long8(a > (short)DST_MAX) ? (long8)DST_MAX :
            convert_long8(a));
  }
    
  long16 _cl_overloadable
  convert_long16_sat(short16 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long16(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long16(a < (short)DST_MIN) ? (long16)DST_MIN :
            convert_long16(a > (short)DST_MAX) ? (long16)DST_MAX :
            convert_long16(a));
  }
    
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat(short a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong(a < (short)0) ? (ulong)0 :
              convert_ulong(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong(a < (short)0      ) ? (ulong)0 :
            convert_ulong(a > (short)DST_MAX) ? (ulong)DST_MAX :
            convert_ulong(a));
  }
    
  ulong2 _cl_overloadable
  convert_ulong2_sat(short2 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong2(a < (short)0) ? (ulong2)0 :
              convert_ulong2(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong2(a < (short)0      ) ? (ulong2)0 :
            convert_ulong2(a > (short)DST_MAX) ? (ulong2)DST_MAX :
            convert_ulong2(a));
  }
    
  ulong4 _cl_overloadable
  convert_ulong4_sat(short4 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong4(a < (short)0) ? (ulong4)0 :
              convert_ulong4(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong4(a < (short)0      ) ? (ulong4)0 :
            convert_ulong4(a > (short)DST_MAX) ? (ulong4)DST_MAX :
            convert_ulong4(a));
  }
    
  ulong8 _cl_overloadable
  convert_ulong8_sat(short8 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong8(a < (short)0) ? (ulong8)0 :
              convert_ulong8(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong8(a < (short)0      ) ? (ulong8)0 :
            convert_ulong8(a > (short)DST_MAX) ? (ulong8)DST_MAX :
            convert_ulong8(a));
  }
    
  ulong16 _cl_overloadable
  convert_ulong16_sat(short16 a)
  {
    int const src_size = sizeof(short);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong16(a < (short)0) ? (ulong16)0 :
              convert_ulong16(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong16(a < (short)0      ) ? (ulong16)0 :
            convert_ulong16(a > (short)DST_MAX) ? (ulong16)DST_MAX :
            convert_ulong16(a));
  }
    
)
  char _cl_overloadable
  convert_char_sat(ushort a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char(a > (ushort)DST_MAX) ? (char)DST_MAX :
            convert_char(a));
  }
    
  char2 _cl_overloadable
  convert_char2_sat(ushort2 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char2(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char2(a > (ushort)DST_MAX) ? (char2)DST_MAX :
            convert_char2(a));
  }
    
  char4 _cl_overloadable
  convert_char4_sat(ushort4 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char4(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char4(a > (ushort)DST_MAX) ? (char4)DST_MAX :
            convert_char4(a));
  }
    
  char8 _cl_overloadable
  convert_char8_sat(ushort8 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char8(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char8(a > (ushort)DST_MAX) ? (char8)DST_MAX :
            convert_char8(a));
  }
    
  char16 _cl_overloadable
  convert_char16_sat(ushort16 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char16(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char16(a > (ushort)DST_MAX) ? (char16)DST_MAX :
            convert_char16(a));
  }
    
  uchar _cl_overloadable
  convert_uchar_sat(ushort a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar(a > (ushort)DST_MAX) ? (uchar)DST_MAX :
            convert_uchar(a));
  }
    
  uchar2 _cl_overloadable
  convert_uchar2_sat(ushort2 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar2(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar2(a > (ushort)DST_MAX) ? (uchar2)DST_MAX :
            convert_uchar2(a));
  }
    
  uchar4 _cl_overloadable
  convert_uchar4_sat(ushort4 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar4(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar4(a > (ushort)DST_MAX) ? (uchar4)DST_MAX :
            convert_uchar4(a));
  }
    
  uchar8 _cl_overloadable
  convert_uchar8_sat(ushort8 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar8(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar8(a > (ushort)DST_MAX) ? (uchar8)DST_MAX :
            convert_uchar8(a));
  }
    
  uchar16 _cl_overloadable
  convert_uchar16_sat(ushort16 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar16(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar16(a > (ushort)DST_MAX) ? (uchar16)DST_MAX :
            convert_uchar16(a));
  }
    
  short _cl_overloadable
  convert_short_sat(ushort a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short(a > (ushort)DST_MAX) ? (short)DST_MAX :
            convert_short(a));
  }
    
  short2 _cl_overloadable
  convert_short2_sat(ushort2 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short2(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short2(a > (ushort)DST_MAX) ? (short2)DST_MAX :
            convert_short2(a));
  }
    
  short4 _cl_overloadable
  convert_short4_sat(ushort4 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short4(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short4(a > (ushort)DST_MAX) ? (short4)DST_MAX :
            convert_short4(a));
  }
    
  short8 _cl_overloadable
  convert_short8_sat(ushort8 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short8(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short8(a > (ushort)DST_MAX) ? (short8)DST_MAX :
            convert_short8(a));
  }
    
  short16 _cl_overloadable
  convert_short16_sat(ushort16 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short16(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short16(a > (ushort)DST_MAX) ? (short16)DST_MAX :
            convert_short16(a));
  }
    
  ushort _cl_overloadable
  convert_ushort_sat(ushort a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort(a > (ushort)DST_MAX) ? (ushort)DST_MAX :
            convert_ushort(a));
  }
    
  ushort2 _cl_overloadable
  convert_ushort2_sat(ushort2 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort2(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort2(a > (ushort)DST_MAX) ? (ushort2)DST_MAX :
            convert_ushort2(a));
  }
    
  ushort4 _cl_overloadable
  convert_ushort4_sat(ushort4 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort4(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort4(a > (ushort)DST_MAX) ? (ushort4)DST_MAX :
            convert_ushort4(a));
  }
    
  ushort8 _cl_overloadable
  convert_ushort8_sat(ushort8 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort8(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort8(a > (ushort)DST_MAX) ? (ushort8)DST_MAX :
            convert_ushort8(a));
  }
    
  ushort16 _cl_overloadable
  convert_ushort16_sat(ushort16 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort16(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort16(a > (ushort)DST_MAX) ? (ushort16)DST_MAX :
            convert_ushort16(a));
  }
    
  int _cl_overloadable
  convert_int_sat(ushort a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int(a > (ushort)DST_MAX) ? (int)DST_MAX :
            convert_int(a));
  }
    
  int2 _cl_overloadable
  convert_int2_sat(ushort2 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int2(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int2(a > (ushort)DST_MAX) ? (int2)DST_MAX :
            convert_int2(a));
  }
    
  int4 _cl_overloadable
  convert_int4_sat(ushort4 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int4(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int4(a > (ushort)DST_MAX) ? (int4)DST_MAX :
            convert_int4(a));
  }
    
  int8 _cl_overloadable
  convert_int8_sat(ushort8 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int8(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int8(a > (ushort)DST_MAX) ? (int8)DST_MAX :
            convert_int8(a));
  }
    
  int16 _cl_overloadable
  convert_int16_sat(ushort16 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int16(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int16(a > (ushort)DST_MAX) ? (int16)DST_MAX :
            convert_int16(a));
  }
    
  uint _cl_overloadable
  convert_uint_sat(ushort a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint(a > (ushort)DST_MAX) ? (uint)DST_MAX :
            convert_uint(a));
  }
    
  uint2 _cl_overloadable
  convert_uint2_sat(ushort2 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint2(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint2(a > (ushort)DST_MAX) ? (uint2)DST_MAX :
            convert_uint2(a));
  }
    
  uint4 _cl_overloadable
  convert_uint4_sat(ushort4 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint4(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint4(a > (ushort)DST_MAX) ? (uint4)DST_MAX :
            convert_uint4(a));
  }
    
  uint8 _cl_overloadable
  convert_uint8_sat(ushort8 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint8(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint8(a > (ushort)DST_MAX) ? (uint8)DST_MAX :
            convert_uint8(a));
  }
    
  uint16 _cl_overloadable
  convert_uint16_sat(ushort16 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint16(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint16(a > (ushort)DST_MAX) ? (uint16)DST_MAX :
            convert_uint16(a));
  }
    
__IF_INT64(
  long _cl_overloadable
  convert_long_sat(ushort a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long(a > (ushort)DST_MAX) ? (long)DST_MAX :
            convert_long(a));
  }
    
  long2 _cl_overloadable
  convert_long2_sat(ushort2 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long2(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long2(a > (ushort)DST_MAX) ? (long2)DST_MAX :
            convert_long2(a));
  }
    
  long4 _cl_overloadable
  convert_long4_sat(ushort4 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long4(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long4(a > (ushort)DST_MAX) ? (long4)DST_MAX :
            convert_long4(a));
  }
    
  long8 _cl_overloadable
  convert_long8_sat(ushort8 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long8(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long8(a > (ushort)DST_MAX) ? (long8)DST_MAX :
            convert_long8(a));
  }
    
  long16 _cl_overloadable
  convert_long16_sat(ushort16 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long16(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long16(a > (ushort)DST_MAX) ? (long16)DST_MAX :
            convert_long16(a));
  }
    
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat(ushort a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong(a > (ushort)DST_MAX) ? (ulong)DST_MAX :
            convert_ulong(a));
  }
    
  ulong2 _cl_overloadable
  convert_ulong2_sat(ushort2 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong2(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong2(a > (ushort)DST_MAX) ? (ulong2)DST_MAX :
            convert_ulong2(a));
  }
    
  ulong4 _cl_overloadable
  convert_ulong4_sat(ushort4 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong4(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong4(a > (ushort)DST_MAX) ? (ulong4)DST_MAX :
            convert_ulong4(a));
  }
    
  ulong8 _cl_overloadable
  convert_ulong8_sat(ushort8 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong8(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong8(a > (ushort)DST_MAX) ? (ulong8)DST_MAX :
            convert_ulong8(a));
  }
    
  ulong16 _cl_overloadable
  convert_ulong16_sat(ushort16 a)
  {
    int const src_size = sizeof(ushort);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong16(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong16(a > (ushort)DST_MAX) ? (ulong16)DST_MAX :
            convert_ulong16(a));
  }
    
)
  char _cl_overloadable
  convert_char_sat(int a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char(a < (int)DST_MIN) ? (char)DST_MIN :
            convert_char(a > (int)DST_MAX) ? (char)DST_MAX :
            convert_char(a));
  }
    
  char2 _cl_overloadable
  convert_char2_sat(int2 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char2(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char2(a < (int)DST_MIN) ? (char2)DST_MIN :
            convert_char2(a > (int)DST_MAX) ? (char2)DST_MAX :
            convert_char2(a));
  }
    
  char4 _cl_overloadable
  convert_char4_sat(int4 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char4(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char4(a < (int)DST_MIN) ? (char4)DST_MIN :
            convert_char4(a > (int)DST_MAX) ? (char4)DST_MAX :
            convert_char4(a));
  }
    
  char8 _cl_overloadable
  convert_char8_sat(int8 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char8(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char8(a < (int)DST_MIN) ? (char8)DST_MIN :
            convert_char8(a > (int)DST_MAX) ? (char8)DST_MAX :
            convert_char8(a));
  }
    
  char16 _cl_overloadable
  convert_char16_sat(int16 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char16(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char16(a < (int)DST_MIN) ? (char16)DST_MIN :
            convert_char16(a > (int)DST_MAX) ? (char16)DST_MAX :
            convert_char16(a));
  }
    
  uchar _cl_overloadable
  convert_uchar_sat(int a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar(a < (int)0) ? (uchar)0 :
              convert_uchar(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar(a < (int)0      ) ? (uchar)0 :
            convert_uchar(a > (int)DST_MAX) ? (uchar)DST_MAX :
            convert_uchar(a));
  }
    
  uchar2 _cl_overloadable
  convert_uchar2_sat(int2 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar2(a < (int)0) ? (uchar2)0 :
              convert_uchar2(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar2(a < (int)0      ) ? (uchar2)0 :
            convert_uchar2(a > (int)DST_MAX) ? (uchar2)DST_MAX :
            convert_uchar2(a));
  }
    
  uchar4 _cl_overloadable
  convert_uchar4_sat(int4 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar4(a < (int)0) ? (uchar4)0 :
              convert_uchar4(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar4(a < (int)0      ) ? (uchar4)0 :
            convert_uchar4(a > (int)DST_MAX) ? (uchar4)DST_MAX :
            convert_uchar4(a));
  }
    
  uchar8 _cl_overloadable
  convert_uchar8_sat(int8 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar8(a < (int)0) ? (uchar8)0 :
              convert_uchar8(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar8(a < (int)0      ) ? (uchar8)0 :
            convert_uchar8(a > (int)DST_MAX) ? (uchar8)DST_MAX :
            convert_uchar8(a));
  }
    
  uchar16 _cl_overloadable
  convert_uchar16_sat(int16 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar16(a < (int)0) ? (uchar16)0 :
              convert_uchar16(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar16(a < (int)0      ) ? (uchar16)0 :
            convert_uchar16(a > (int)DST_MAX) ? (uchar16)DST_MAX :
            convert_uchar16(a));
  }
    
  short _cl_overloadable
  convert_short_sat(int a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short(a < (int)DST_MIN) ? (short)DST_MIN :
            convert_short(a > (int)DST_MAX) ? (short)DST_MAX :
            convert_short(a));
  }
    
  short2 _cl_overloadable
  convert_short2_sat(int2 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short2(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short2(a < (int)DST_MIN) ? (short2)DST_MIN :
            convert_short2(a > (int)DST_MAX) ? (short2)DST_MAX :
            convert_short2(a));
  }
    
  short4 _cl_overloadable
  convert_short4_sat(int4 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short4(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short4(a < (int)DST_MIN) ? (short4)DST_MIN :
            convert_short4(a > (int)DST_MAX) ? (short4)DST_MAX :
            convert_short4(a));
  }
    
  short8 _cl_overloadable
  convert_short8_sat(int8 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short8(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short8(a < (int)DST_MIN) ? (short8)DST_MIN :
            convert_short8(a > (int)DST_MAX) ? (short8)DST_MAX :
            convert_short8(a));
  }
    
  short16 _cl_overloadable
  convert_short16_sat(int16 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short16(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short16(a < (int)DST_MIN) ? (short16)DST_MIN :
            convert_short16(a > (int)DST_MAX) ? (short16)DST_MAX :
            convert_short16(a));
  }
    
  ushort _cl_overloadable
  convert_ushort_sat(int a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort(a < (int)0) ? (ushort)0 :
              convert_ushort(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort(a < (int)0      ) ? (ushort)0 :
            convert_ushort(a > (int)DST_MAX) ? (ushort)DST_MAX :
            convert_ushort(a));
  }
    
  ushort2 _cl_overloadable
  convert_ushort2_sat(int2 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort2(a < (int)0) ? (ushort2)0 :
              convert_ushort2(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort2(a < (int)0      ) ? (ushort2)0 :
            convert_ushort2(a > (int)DST_MAX) ? (ushort2)DST_MAX :
            convert_ushort2(a));
  }
    
  ushort4 _cl_overloadable
  convert_ushort4_sat(int4 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort4(a < (int)0) ? (ushort4)0 :
              convert_ushort4(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort4(a < (int)0      ) ? (ushort4)0 :
            convert_ushort4(a > (int)DST_MAX) ? (ushort4)DST_MAX :
            convert_ushort4(a));
  }
    
  ushort8 _cl_overloadable
  convert_ushort8_sat(int8 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort8(a < (int)0) ? (ushort8)0 :
              convert_ushort8(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort8(a < (int)0      ) ? (ushort8)0 :
            convert_ushort8(a > (int)DST_MAX) ? (ushort8)DST_MAX :
            convert_ushort8(a));
  }
    
  ushort16 _cl_overloadable
  convert_ushort16_sat(int16 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort16(a < (int)0) ? (ushort16)0 :
              convert_ushort16(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort16(a < (int)0      ) ? (ushort16)0 :
            convert_ushort16(a > (int)DST_MAX) ? (ushort16)DST_MAX :
            convert_ushort16(a));
  }
    
  int _cl_overloadable
  convert_int_sat(int a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int(a < (int)DST_MIN) ? (int)DST_MIN :
            convert_int(a > (int)DST_MAX) ? (int)DST_MAX :
            convert_int(a));
  }
    
  int2 _cl_overloadable
  convert_int2_sat(int2 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int2(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int2(a < (int)DST_MIN) ? (int2)DST_MIN :
            convert_int2(a > (int)DST_MAX) ? (int2)DST_MAX :
            convert_int2(a));
  }
    
  int4 _cl_overloadable
  convert_int4_sat(int4 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int4(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int4(a < (int)DST_MIN) ? (int4)DST_MIN :
            convert_int4(a > (int)DST_MAX) ? (int4)DST_MAX :
            convert_int4(a));
  }
    
  int8 _cl_overloadable
  convert_int8_sat(int8 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int8(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int8(a < (int)DST_MIN) ? (int8)DST_MIN :
            convert_int8(a > (int)DST_MAX) ? (int8)DST_MAX :
            convert_int8(a));
  }
    
  int16 _cl_overloadable
  convert_int16_sat(int16 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int16(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int16(a < (int)DST_MIN) ? (int16)DST_MIN :
            convert_int16(a > (int)DST_MAX) ? (int16)DST_MAX :
            convert_int16(a));
  }
    
  uint _cl_overloadable
  convert_uint_sat(int a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint(a < (int)0) ? (uint)0 :
              convert_uint(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint(a < (int)0      ) ? (uint)0 :
            convert_uint(a > (int)DST_MAX) ? (uint)DST_MAX :
            convert_uint(a));
  }
    
  uint2 _cl_overloadable
  convert_uint2_sat(int2 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint2(a < (int)0) ? (uint2)0 :
              convert_uint2(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint2(a < (int)0      ) ? (uint2)0 :
            convert_uint2(a > (int)DST_MAX) ? (uint2)DST_MAX :
            convert_uint2(a));
  }
    
  uint4 _cl_overloadable
  convert_uint4_sat(int4 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint4(a < (int)0) ? (uint4)0 :
              convert_uint4(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint4(a < (int)0      ) ? (uint4)0 :
            convert_uint4(a > (int)DST_MAX) ? (uint4)DST_MAX :
            convert_uint4(a));
  }
    
  uint8 _cl_overloadable
  convert_uint8_sat(int8 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint8(a < (int)0) ? (uint8)0 :
              convert_uint8(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint8(a < (int)0      ) ? (uint8)0 :
            convert_uint8(a > (int)DST_MAX) ? (uint8)DST_MAX :
            convert_uint8(a));
  }
    
  uint16 _cl_overloadable
  convert_uint16_sat(int16 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint16(a < (int)0) ? (uint16)0 :
              convert_uint16(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint16(a < (int)0      ) ? (uint16)0 :
            convert_uint16(a > (int)DST_MAX) ? (uint16)DST_MAX :
            convert_uint16(a));
  }
    
__IF_INT64(
  long _cl_overloadable
  convert_long_sat(int a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long(a < (int)DST_MIN) ? (long)DST_MIN :
            convert_long(a > (int)DST_MAX) ? (long)DST_MAX :
            convert_long(a));
  }
    
  long2 _cl_overloadable
  convert_long2_sat(int2 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long2(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long2(a < (int)DST_MIN) ? (long2)DST_MIN :
            convert_long2(a > (int)DST_MAX) ? (long2)DST_MAX :
            convert_long2(a));
  }
    
  long4 _cl_overloadable
  convert_long4_sat(int4 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long4(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long4(a < (int)DST_MIN) ? (long4)DST_MIN :
            convert_long4(a > (int)DST_MAX) ? (long4)DST_MAX :
            convert_long4(a));
  }
    
  long8 _cl_overloadable
  convert_long8_sat(int8 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long8(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long8(a < (int)DST_MIN) ? (long8)DST_MIN :
            convert_long8(a > (int)DST_MAX) ? (long8)DST_MAX :
            convert_long8(a));
  }
    
  long16 _cl_overloadable
  convert_long16_sat(int16 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long16(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long16(a < (int)DST_MIN) ? (long16)DST_MIN :
            convert_long16(a > (int)DST_MAX) ? (long16)DST_MAX :
            convert_long16(a));
  }
    
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat(int a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong(a < (int)0) ? (ulong)0 :
              convert_ulong(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong(a < (int)0      ) ? (ulong)0 :
            convert_ulong(a > (int)DST_MAX) ? (ulong)DST_MAX :
            convert_ulong(a));
  }
    
  ulong2 _cl_overloadable
  convert_ulong2_sat(int2 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong2(a < (int)0) ? (ulong2)0 :
              convert_ulong2(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong2(a < (int)0      ) ? (ulong2)0 :
            convert_ulong2(a > (int)DST_MAX) ? (ulong2)DST_MAX :
            convert_ulong2(a));
  }
    
  ulong4 _cl_overloadable
  convert_ulong4_sat(int4 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong4(a < (int)0) ? (ulong4)0 :
              convert_ulong4(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong4(a < (int)0      ) ? (ulong4)0 :
            convert_ulong4(a > (int)DST_MAX) ? (ulong4)DST_MAX :
            convert_ulong4(a));
  }
    
  ulong8 _cl_overloadable
  convert_ulong8_sat(int8 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong8(a < (int)0) ? (ulong8)0 :
              convert_ulong8(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong8(a < (int)0      ) ? (ulong8)0 :
            convert_ulong8(a > (int)DST_MAX) ? (ulong8)DST_MAX :
            convert_ulong8(a));
  }
    
  ulong16 _cl_overloadable
  convert_ulong16_sat(int16 a)
  {
    int const src_size = sizeof(int);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong16(a < (int)0) ? (ulong16)0 :
              convert_ulong16(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong16(a < (int)0      ) ? (ulong16)0 :
            convert_ulong16(a > (int)DST_MAX) ? (ulong16)DST_MAX :
            convert_ulong16(a));
  }
    
)
  char _cl_overloadable
  convert_char_sat(uint a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char(a > (uint)DST_MAX) ? (char)DST_MAX :
            convert_char(a));
  }
    
  char2 _cl_overloadable
  convert_char2_sat(uint2 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char2(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char2(a > (uint)DST_MAX) ? (char2)DST_MAX :
            convert_char2(a));
  }
    
  char4 _cl_overloadable
  convert_char4_sat(uint4 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char4(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char4(a > (uint)DST_MAX) ? (char4)DST_MAX :
            convert_char4(a));
  }
    
  char8 _cl_overloadable
  convert_char8_sat(uint8 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char8(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char8(a > (uint)DST_MAX) ? (char8)DST_MAX :
            convert_char8(a));
  }
    
  char16 _cl_overloadable
  convert_char16_sat(uint16 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char16(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char16(a > (uint)DST_MAX) ? (char16)DST_MAX :
            convert_char16(a));
  }
    
  uchar _cl_overloadable
  convert_uchar_sat(uint a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar(a > (uint)DST_MAX) ? (uchar)DST_MAX :
            convert_uchar(a));
  }
    
  uchar2 _cl_overloadable
  convert_uchar2_sat(uint2 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar2(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar2(a > (uint)DST_MAX) ? (uchar2)DST_MAX :
            convert_uchar2(a));
  }
    
  uchar4 _cl_overloadable
  convert_uchar4_sat(uint4 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar4(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar4(a > (uint)DST_MAX) ? (uchar4)DST_MAX :
            convert_uchar4(a));
  }
    
  uchar8 _cl_overloadable
  convert_uchar8_sat(uint8 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar8(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar8(a > (uint)DST_MAX) ? (uchar8)DST_MAX :
            convert_uchar8(a));
  }
    
  uchar16 _cl_overloadable
  convert_uchar16_sat(uint16 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar16(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar16(a > (uint)DST_MAX) ? (uchar16)DST_MAX :
            convert_uchar16(a));
  }
    
  short _cl_overloadable
  convert_short_sat(uint a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short(a > (uint)DST_MAX) ? (short)DST_MAX :
            convert_short(a));
  }
    
  short2 _cl_overloadable
  convert_short2_sat(uint2 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short2(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short2(a > (uint)DST_MAX) ? (short2)DST_MAX :
            convert_short2(a));
  }
    
  short4 _cl_overloadable
  convert_short4_sat(uint4 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short4(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short4(a > (uint)DST_MAX) ? (short4)DST_MAX :
            convert_short4(a));
  }
    
  short8 _cl_overloadable
  convert_short8_sat(uint8 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short8(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short8(a > (uint)DST_MAX) ? (short8)DST_MAX :
            convert_short8(a));
  }
    
  short16 _cl_overloadable
  convert_short16_sat(uint16 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short16(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short16(a > (uint)DST_MAX) ? (short16)DST_MAX :
            convert_short16(a));
  }
    
  ushort _cl_overloadable
  convert_ushort_sat(uint a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort(a > (uint)DST_MAX) ? (ushort)DST_MAX :
            convert_ushort(a));
  }
    
  ushort2 _cl_overloadable
  convert_ushort2_sat(uint2 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort2(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort2(a > (uint)DST_MAX) ? (ushort2)DST_MAX :
            convert_ushort2(a));
  }
    
  ushort4 _cl_overloadable
  convert_ushort4_sat(uint4 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort4(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort4(a > (uint)DST_MAX) ? (ushort4)DST_MAX :
            convert_ushort4(a));
  }
    
  ushort8 _cl_overloadable
  convert_ushort8_sat(uint8 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort8(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort8(a > (uint)DST_MAX) ? (ushort8)DST_MAX :
            convert_ushort8(a));
  }
    
  ushort16 _cl_overloadable
  convert_ushort16_sat(uint16 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort16(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort16(a > (uint)DST_MAX) ? (ushort16)DST_MAX :
            convert_ushort16(a));
  }
    
  int _cl_overloadable
  convert_int_sat(uint a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int(a > (uint)DST_MAX) ? (int)DST_MAX :
            convert_int(a));
  }
    
  int2 _cl_overloadable
  convert_int2_sat(uint2 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int2(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int2(a > (uint)DST_MAX) ? (int2)DST_MAX :
            convert_int2(a));
  }
    
  int4 _cl_overloadable
  convert_int4_sat(uint4 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int4(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int4(a > (uint)DST_MAX) ? (int4)DST_MAX :
            convert_int4(a));
  }
    
  int8 _cl_overloadable
  convert_int8_sat(uint8 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int8(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int8(a > (uint)DST_MAX) ? (int8)DST_MAX :
            convert_int8(a));
  }
    
  int16 _cl_overloadable
  convert_int16_sat(uint16 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int16(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int16(a > (uint)DST_MAX) ? (int16)DST_MAX :
            convert_int16(a));
  }
    
  uint _cl_overloadable
  convert_uint_sat(uint a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint(a > (uint)DST_MAX) ? (uint)DST_MAX :
            convert_uint(a));
  }
    
  uint2 _cl_overloadable
  convert_uint2_sat(uint2 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint2(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint2(a > (uint)DST_MAX) ? (uint2)DST_MAX :
            convert_uint2(a));
  }
    
  uint4 _cl_overloadable
  convert_uint4_sat(uint4 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint4(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint4(a > (uint)DST_MAX) ? (uint4)DST_MAX :
            convert_uint4(a));
  }
    
  uint8 _cl_overloadable
  convert_uint8_sat(uint8 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint8(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint8(a > (uint)DST_MAX) ? (uint8)DST_MAX :
            convert_uint8(a));
  }
    
  uint16 _cl_overloadable
  convert_uint16_sat(uint16 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint16(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint16(a > (uint)DST_MAX) ? (uint16)DST_MAX :
            convert_uint16(a));
  }
    
__IF_INT64(
  long _cl_overloadable
  convert_long_sat(uint a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long(a > (uint)DST_MAX) ? (long)DST_MAX :
            convert_long(a));
  }
    
  long2 _cl_overloadable
  convert_long2_sat(uint2 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long2(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long2(a > (uint)DST_MAX) ? (long2)DST_MAX :
            convert_long2(a));
  }
    
  long4 _cl_overloadable
  convert_long4_sat(uint4 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long4(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long4(a > (uint)DST_MAX) ? (long4)DST_MAX :
            convert_long4(a));
  }
    
  long8 _cl_overloadable
  convert_long8_sat(uint8 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long8(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long8(a > (uint)DST_MAX) ? (long8)DST_MAX :
            convert_long8(a));
  }
    
  long16 _cl_overloadable
  convert_long16_sat(uint16 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long16(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long16(a > (uint)DST_MAX) ? (long16)DST_MAX :
            convert_long16(a));
  }
    
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat(uint a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong(a > (uint)DST_MAX) ? (ulong)DST_MAX :
            convert_ulong(a));
  }
    
  ulong2 _cl_overloadable
  convert_ulong2_sat(uint2 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong2(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong2(a > (uint)DST_MAX) ? (ulong2)DST_MAX :
            convert_ulong2(a));
  }
    
  ulong4 _cl_overloadable
  convert_ulong4_sat(uint4 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong4(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong4(a > (uint)DST_MAX) ? (ulong4)DST_MAX :
            convert_ulong4(a));
  }
    
  ulong8 _cl_overloadable
  convert_ulong8_sat(uint8 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong8(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong8(a > (uint)DST_MAX) ? (ulong8)DST_MAX :
            convert_ulong8(a));
  }
    
  ulong16 _cl_overloadable
  convert_ulong16_sat(uint16 a)
  {
    int const src_size = sizeof(uint);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong16(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong16(a > (uint)DST_MAX) ? (ulong16)DST_MAX :
            convert_ulong16(a));
  }
    
)
__IF_INT64(
  char _cl_overloadable
  convert_char_sat(long a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char(a < (long)DST_MIN) ? (char)DST_MIN :
            convert_char(a > (long)DST_MAX) ? (char)DST_MAX :
            convert_char(a));
  }
    
  char2 _cl_overloadable
  convert_char2_sat(long2 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char2(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char2(a < (long)DST_MIN) ? (char2)DST_MIN :
            convert_char2(a > (long)DST_MAX) ? (char2)DST_MAX :
            convert_char2(a));
  }
    
  char4 _cl_overloadable
  convert_char4_sat(long4 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char4(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char4(a < (long)DST_MIN) ? (char4)DST_MIN :
            convert_char4(a > (long)DST_MAX) ? (char4)DST_MAX :
            convert_char4(a));
  }
    
  char8 _cl_overloadable
  convert_char8_sat(long8 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char8(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char8(a < (long)DST_MIN) ? (char8)DST_MIN :
            convert_char8(a > (long)DST_MAX) ? (char8)DST_MAX :
            convert_char8(a));
  }
    
  char16 _cl_overloadable
  convert_char16_sat(long16 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char16(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char16(a < (long)DST_MIN) ? (char16)DST_MIN :
            convert_char16(a > (long)DST_MAX) ? (char16)DST_MAX :
            convert_char16(a));
  }
    
)
__IF_INT64(
  uchar _cl_overloadable
  convert_uchar_sat(long a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar(a < (long)0) ? (uchar)0 :
              convert_uchar(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar(a < (long)0      ) ? (uchar)0 :
            convert_uchar(a > (long)DST_MAX) ? (uchar)DST_MAX :
            convert_uchar(a));
  }
    
  uchar2 _cl_overloadable
  convert_uchar2_sat(long2 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar2(a < (long)0) ? (uchar2)0 :
              convert_uchar2(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar2(a < (long)0      ) ? (uchar2)0 :
            convert_uchar2(a > (long)DST_MAX) ? (uchar2)DST_MAX :
            convert_uchar2(a));
  }
    
  uchar4 _cl_overloadable
  convert_uchar4_sat(long4 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar4(a < (long)0) ? (uchar4)0 :
              convert_uchar4(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar4(a < (long)0      ) ? (uchar4)0 :
            convert_uchar4(a > (long)DST_MAX) ? (uchar4)DST_MAX :
            convert_uchar4(a));
  }
    
  uchar8 _cl_overloadable
  convert_uchar8_sat(long8 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar8(a < (long)0) ? (uchar8)0 :
              convert_uchar8(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar8(a < (long)0      ) ? (uchar8)0 :
            convert_uchar8(a > (long)DST_MAX) ? (uchar8)DST_MAX :
            convert_uchar8(a));
  }
    
  uchar16 _cl_overloadable
  convert_uchar16_sat(long16 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar16(a < (long)0) ? (uchar16)0 :
              convert_uchar16(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar16(a < (long)0      ) ? (uchar16)0 :
            convert_uchar16(a > (long)DST_MAX) ? (uchar16)DST_MAX :
            convert_uchar16(a));
  }
    
)
__IF_INT64(
  short _cl_overloadable
  convert_short_sat(long a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short(a < (long)DST_MIN) ? (short)DST_MIN :
            convert_short(a > (long)DST_MAX) ? (short)DST_MAX :
            convert_short(a));
  }
    
  short2 _cl_overloadable
  convert_short2_sat(long2 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short2(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short2(a < (long)DST_MIN) ? (short2)DST_MIN :
            convert_short2(a > (long)DST_MAX) ? (short2)DST_MAX :
            convert_short2(a));
  }
    
  short4 _cl_overloadable
  convert_short4_sat(long4 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short4(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short4(a < (long)DST_MIN) ? (short4)DST_MIN :
            convert_short4(a > (long)DST_MAX) ? (short4)DST_MAX :
            convert_short4(a));
  }
    
  short8 _cl_overloadable
  convert_short8_sat(long8 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short8(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short8(a < (long)DST_MIN) ? (short8)DST_MIN :
            convert_short8(a > (long)DST_MAX) ? (short8)DST_MAX :
            convert_short8(a));
  }
    
  short16 _cl_overloadable
  convert_short16_sat(long16 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short16(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short16(a < (long)DST_MIN) ? (short16)DST_MIN :
            convert_short16(a > (long)DST_MAX) ? (short16)DST_MAX :
            convert_short16(a));
  }
    
)
__IF_INT64(
  ushort _cl_overloadable
  convert_ushort_sat(long a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort(a < (long)0) ? (ushort)0 :
              convert_ushort(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort(a < (long)0      ) ? (ushort)0 :
            convert_ushort(a > (long)DST_MAX) ? (ushort)DST_MAX :
            convert_ushort(a));
  }
    
  ushort2 _cl_overloadable
  convert_ushort2_sat(long2 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort2(a < (long)0) ? (ushort2)0 :
              convert_ushort2(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort2(a < (long)0      ) ? (ushort2)0 :
            convert_ushort2(a > (long)DST_MAX) ? (ushort2)DST_MAX :
            convert_ushort2(a));
  }
    
  ushort4 _cl_overloadable
  convert_ushort4_sat(long4 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort4(a < (long)0) ? (ushort4)0 :
              convert_ushort4(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort4(a < (long)0      ) ? (ushort4)0 :
            convert_ushort4(a > (long)DST_MAX) ? (ushort4)DST_MAX :
            convert_ushort4(a));
  }
    
  ushort8 _cl_overloadable
  convert_ushort8_sat(long8 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort8(a < (long)0) ? (ushort8)0 :
              convert_ushort8(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort8(a < (long)0      ) ? (ushort8)0 :
            convert_ushort8(a > (long)DST_MAX) ? (ushort8)DST_MAX :
            convert_ushort8(a));
  }
    
  ushort16 _cl_overloadable
  convert_ushort16_sat(long16 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort16(a < (long)0) ? (ushort16)0 :
              convert_ushort16(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort16(a < (long)0      ) ? (ushort16)0 :
            convert_ushort16(a > (long)DST_MAX) ? (ushort16)DST_MAX :
            convert_ushort16(a));
  }
    
)
__IF_INT64(
  int _cl_overloadable
  convert_int_sat(long a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int(a < (long)DST_MIN) ? (int)DST_MIN :
            convert_int(a > (long)DST_MAX) ? (int)DST_MAX :
            convert_int(a));
  }
    
  int2 _cl_overloadable
  convert_int2_sat(long2 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int2(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int2(a < (long)DST_MIN) ? (int2)DST_MIN :
            convert_int2(a > (long)DST_MAX) ? (int2)DST_MAX :
            convert_int2(a));
  }
    
  int4 _cl_overloadable
  convert_int4_sat(long4 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int4(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int4(a < (long)DST_MIN) ? (int4)DST_MIN :
            convert_int4(a > (long)DST_MAX) ? (int4)DST_MAX :
            convert_int4(a));
  }
    
  int8 _cl_overloadable
  convert_int8_sat(long8 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int8(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int8(a < (long)DST_MIN) ? (int8)DST_MIN :
            convert_int8(a > (long)DST_MAX) ? (int8)DST_MAX :
            convert_int8(a));
  }
    
  int16 _cl_overloadable
  convert_int16_sat(long16 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int16(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int16(a < (long)DST_MIN) ? (int16)DST_MIN :
            convert_int16(a > (long)DST_MAX) ? (int16)DST_MAX :
            convert_int16(a));
  }
    
)
__IF_INT64(
  uint _cl_overloadable
  convert_uint_sat(long a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint(a < (long)0) ? (uint)0 :
              convert_uint(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint(a < (long)0      ) ? (uint)0 :
            convert_uint(a > (long)DST_MAX) ? (uint)DST_MAX :
            convert_uint(a));
  }
    
  uint2 _cl_overloadable
  convert_uint2_sat(long2 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint2(a < (long)0) ? (uint2)0 :
              convert_uint2(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint2(a < (long)0      ) ? (uint2)0 :
            convert_uint2(a > (long)DST_MAX) ? (uint2)DST_MAX :
            convert_uint2(a));
  }
    
  uint4 _cl_overloadable
  convert_uint4_sat(long4 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint4(a < (long)0) ? (uint4)0 :
              convert_uint4(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint4(a < (long)0      ) ? (uint4)0 :
            convert_uint4(a > (long)DST_MAX) ? (uint4)DST_MAX :
            convert_uint4(a));
  }
    
  uint8 _cl_overloadable
  convert_uint8_sat(long8 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint8(a < (long)0) ? (uint8)0 :
              convert_uint8(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint8(a < (long)0      ) ? (uint8)0 :
            convert_uint8(a > (long)DST_MAX) ? (uint8)DST_MAX :
            convert_uint8(a));
  }
    
  uint16 _cl_overloadable
  convert_uint16_sat(long16 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint16(a < (long)0) ? (uint16)0 :
              convert_uint16(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint16(a < (long)0      ) ? (uint16)0 :
            convert_uint16(a > (long)DST_MAX) ? (uint16)DST_MAX :
            convert_uint16(a));
  }
    
)
__IF_INT64(
  long _cl_overloadable
  convert_long_sat(long a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long(a < (long)DST_MIN) ? (long)DST_MIN :
            convert_long(a > (long)DST_MAX) ? (long)DST_MAX :
            convert_long(a));
  }
    
  long2 _cl_overloadable
  convert_long2_sat(long2 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long2(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long2(a < (long)DST_MIN) ? (long2)DST_MIN :
            convert_long2(a > (long)DST_MAX) ? (long2)DST_MAX :
            convert_long2(a));
  }
    
  long4 _cl_overloadable
  convert_long4_sat(long4 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long4(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long4(a < (long)DST_MIN) ? (long4)DST_MIN :
            convert_long4(a > (long)DST_MAX) ? (long4)DST_MAX :
            convert_long4(a));
  }
    
  long8 _cl_overloadable
  convert_long8_sat(long8 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long8(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long8(a < (long)DST_MIN) ? (long8)DST_MIN :
            convert_long8(a > (long)DST_MAX) ? (long8)DST_MAX :
            convert_long8(a));
  }
    
  long16 _cl_overloadable
  convert_long16_sat(long16 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long16(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long16(a < (long)DST_MIN) ? (long16)DST_MIN :
            convert_long16(a > (long)DST_MAX) ? (long16)DST_MAX :
            convert_long16(a));
  }
    
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat(long a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong(a < (long)0) ? (ulong)0 :
              convert_ulong(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong(a < (long)0      ) ? (ulong)0 :
            convert_ulong(a > (long)DST_MAX) ? (ulong)DST_MAX :
            convert_ulong(a));
  }
    
  ulong2 _cl_overloadable
  convert_ulong2_sat(long2 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong2(a < (long)0) ? (ulong2)0 :
              convert_ulong2(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong2(a < (long)0      ) ? (ulong2)0 :
            convert_ulong2(a > (long)DST_MAX) ? (ulong2)DST_MAX :
            convert_ulong2(a));
  }
    
  ulong4 _cl_overloadable
  convert_ulong4_sat(long4 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong4(a < (long)0) ? (ulong4)0 :
              convert_ulong4(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong4(a < (long)0      ) ? (ulong4)0 :
            convert_ulong4(a > (long)DST_MAX) ? (ulong4)DST_MAX :
            convert_ulong4(a));
  }
    
  ulong8 _cl_overloadable
  convert_ulong8_sat(long8 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong8(a < (long)0) ? (ulong8)0 :
              convert_ulong8(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong8(a < (long)0      ) ? (ulong8)0 :
            convert_ulong8(a > (long)DST_MAX) ? (ulong8)DST_MAX :
            convert_ulong8(a));
  }
    
  ulong16 _cl_overloadable
  convert_ulong16_sat(long16 a)
  {
    int const src_size = sizeof(long);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong16(a < (long)0) ? (ulong16)0 :
              convert_ulong16(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong16(a < (long)0      ) ? (ulong16)0 :
            convert_ulong16(a > (long)DST_MAX) ? (ulong16)DST_MAX :
            convert_ulong16(a));
  }
    
)
__IF_INT64(
  char _cl_overloadable
  convert_char_sat(ulong a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char(a > (ulong)DST_MAX) ? (char)DST_MAX :
            convert_char(a));
  }
    
  char2 _cl_overloadable
  convert_char2_sat(ulong2 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char2(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char2(a > (ulong)DST_MAX) ? (char2)DST_MAX :
            convert_char2(a));
  }
    
  char4 _cl_overloadable
  convert_char4_sat(ulong4 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char4(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char4(a > (ulong)DST_MAX) ? (char4)DST_MAX :
            convert_char4(a));
  }
    
  char8 _cl_overloadable
  convert_char8_sat(ulong8 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char8(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char8(a > (ulong)DST_MAX) ? (char8)DST_MAX :
            convert_char8(a));
  }
    
  char16 _cl_overloadable
  convert_char16_sat(ulong16 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(char);
    if (dst_size > src_size) return convert_char16(a);
    char const DST_MAX = (char)1 << (char)(CHAR_BIT * dst_size);
    return (convert_char16(a > (ulong)DST_MAX) ? (char16)DST_MAX :
            convert_char16(a));
  }
    
)
__IF_INT64(
  uchar _cl_overloadable
  convert_uchar_sat(ulong a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar(a > (ulong)DST_MAX) ? (uchar)DST_MAX :
            convert_uchar(a));
  }
    
  uchar2 _cl_overloadable
  convert_uchar2_sat(ulong2 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar2(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar2(a > (ulong)DST_MAX) ? (uchar2)DST_MAX :
            convert_uchar2(a));
  }
    
  uchar4 _cl_overloadable
  convert_uchar4_sat(ulong4 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar4(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar4(a > (ulong)DST_MAX) ? (uchar4)DST_MAX :
            convert_uchar4(a));
  }
    
  uchar8 _cl_overloadable
  convert_uchar8_sat(ulong8 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar8(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar8(a > (ulong)DST_MAX) ? (uchar8)DST_MAX :
            convert_uchar8(a));
  }
    
  uchar16 _cl_overloadable
  convert_uchar16_sat(ulong16 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) return convert_uchar16(a);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar16(a > (ulong)DST_MAX) ? (uchar16)DST_MAX :
            convert_uchar16(a));
  }
    
)
__IF_INT64(
  short _cl_overloadable
  convert_short_sat(ulong a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short(a > (ulong)DST_MAX) ? (short)DST_MAX :
            convert_short(a));
  }
    
  short2 _cl_overloadable
  convert_short2_sat(ulong2 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short2(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short2(a > (ulong)DST_MAX) ? (short2)DST_MAX :
            convert_short2(a));
  }
    
  short4 _cl_overloadable
  convert_short4_sat(ulong4 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short4(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short4(a > (ulong)DST_MAX) ? (short4)DST_MAX :
            convert_short4(a));
  }
    
  short8 _cl_overloadable
  convert_short8_sat(ulong8 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short8(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short8(a > (ulong)DST_MAX) ? (short8)DST_MAX :
            convert_short8(a));
  }
    
  short16 _cl_overloadable
  convert_short16_sat(ulong16 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(short);
    if (dst_size > src_size) return convert_short16(a);
    short const DST_MAX = (short)1 << (short)(CHAR_BIT * dst_size);
    return (convert_short16(a > (ulong)DST_MAX) ? (short16)DST_MAX :
            convert_short16(a));
  }
    
)
__IF_INT64(
  ushort _cl_overloadable
  convert_ushort_sat(ulong a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort(a > (ulong)DST_MAX) ? (ushort)DST_MAX :
            convert_ushort(a));
  }
    
  ushort2 _cl_overloadable
  convert_ushort2_sat(ulong2 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort2(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort2(a > (ulong)DST_MAX) ? (ushort2)DST_MAX :
            convert_ushort2(a));
  }
    
  ushort4 _cl_overloadable
  convert_ushort4_sat(ulong4 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort4(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort4(a > (ulong)DST_MAX) ? (ushort4)DST_MAX :
            convert_ushort4(a));
  }
    
  ushort8 _cl_overloadable
  convert_ushort8_sat(ulong8 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort8(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort8(a > (ulong)DST_MAX) ? (ushort8)DST_MAX :
            convert_ushort8(a));
  }
    
  ushort16 _cl_overloadable
  convert_ushort16_sat(ulong16 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) return convert_ushort16(a);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort16(a > (ulong)DST_MAX) ? (ushort16)DST_MAX :
            convert_ushort16(a));
  }
    
)
__IF_INT64(
  int _cl_overloadable
  convert_int_sat(ulong a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int(a > (ulong)DST_MAX) ? (int)DST_MAX :
            convert_int(a));
  }
    
  int2 _cl_overloadable
  convert_int2_sat(ulong2 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int2(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int2(a > (ulong)DST_MAX) ? (int2)DST_MAX :
            convert_int2(a));
  }
    
  int4 _cl_overloadable
  convert_int4_sat(ulong4 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int4(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int4(a > (ulong)DST_MAX) ? (int4)DST_MAX :
            convert_int4(a));
  }
    
  int8 _cl_overloadable
  convert_int8_sat(ulong8 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int8(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int8(a > (ulong)DST_MAX) ? (int8)DST_MAX :
            convert_int8(a));
  }
    
  int16 _cl_overloadable
  convert_int16_sat(ulong16 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(int);
    if (dst_size > src_size) return convert_int16(a);
    int const DST_MAX = (int)1 << (int)(CHAR_BIT * dst_size);
    return (convert_int16(a > (ulong)DST_MAX) ? (int16)DST_MAX :
            convert_int16(a));
  }
    
)
__IF_INT64(
  uint _cl_overloadable
  convert_uint_sat(ulong a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint(a > (ulong)DST_MAX) ? (uint)DST_MAX :
            convert_uint(a));
  }
    
  uint2 _cl_overloadable
  convert_uint2_sat(ulong2 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint2(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint2(a > (ulong)DST_MAX) ? (uint2)DST_MAX :
            convert_uint2(a));
  }
    
  uint4 _cl_overloadable
  convert_uint4_sat(ulong4 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint4(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint4(a > (ulong)DST_MAX) ? (uint4)DST_MAX :
            convert_uint4(a));
  }
    
  uint8 _cl_overloadable
  convert_uint8_sat(ulong8 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint8(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint8(a > (ulong)DST_MAX) ? (uint8)DST_MAX :
            convert_uint8(a));
  }
    
  uint16 _cl_overloadable
  convert_uint16_sat(ulong16 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) return convert_uint16(a);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint16(a > (ulong)DST_MAX) ? (uint16)DST_MAX :
            convert_uint16(a));
  }
    
)
__IF_INT64(
  long _cl_overloadable
  convert_long_sat(ulong a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long(a > (ulong)DST_MAX) ? (long)DST_MAX :
            convert_long(a));
  }
    
  long2 _cl_overloadable
  convert_long2_sat(ulong2 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long2(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long2(a > (ulong)DST_MAX) ? (long2)DST_MAX :
            convert_long2(a));
  }
    
  long4 _cl_overloadable
  convert_long4_sat(ulong4 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long4(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long4(a > (ulong)DST_MAX) ? (long4)DST_MAX :
            convert_long4(a));
  }
    
  long8 _cl_overloadable
  convert_long8_sat(ulong8 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long8(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long8(a > (ulong)DST_MAX) ? (long8)DST_MAX :
            convert_long8(a));
  }
    
  long16 _cl_overloadable
  convert_long16_sat(ulong16 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(long);
    if (dst_size > src_size) return convert_long16(a);
    long const DST_MAX = (long)1 << (long)(CHAR_BIT * dst_size);
    return (convert_long16(a > (ulong)DST_MAX) ? (long16)DST_MAX :
            convert_long16(a));
  }
    
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat(ulong a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong(a > (ulong)DST_MAX) ? (ulong)DST_MAX :
            convert_ulong(a));
  }
    
  ulong2 _cl_overloadable
  convert_ulong2_sat(ulong2 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong2(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong2(a > (ulong)DST_MAX) ? (ulong2)DST_MAX :
            convert_ulong2(a));
  }
    
  ulong4 _cl_overloadable
  convert_ulong4_sat(ulong4 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong4(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong4(a > (ulong)DST_MAX) ? (ulong4)DST_MAX :
            convert_ulong4(a));
  }
    
  ulong8 _cl_overloadable
  convert_ulong8_sat(ulong8 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong8(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong8(a > (ulong)DST_MAX) ? (ulong8)DST_MAX :
            convert_ulong8(a));
  }
    
  ulong16 _cl_overloadable
  convert_ulong16_sat(ulong16 a)
  {
    int const src_size = sizeof(ulong);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) return convert_ulong16(a);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong16(a > (ulong)DST_MAX) ? (ulong16)DST_MAX :
            convert_ulong16(a));
  }
    
)
  char _cl_overloadable
  convert_char_sat(float a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(char);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char(a < (float)DST_MIN) ? (char)DST_MIN :
            convert_char(a > (float)DST_MAX) ? (char)DST_MAX :
            convert_char(a));
  }
    
  char2 _cl_overloadable
  convert_char2_sat(float2 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(char);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char2(a < (float)DST_MIN) ? (char2)DST_MIN :
            convert_char2(a > (float)DST_MAX) ? (char2)DST_MAX :
            convert_char2(a));
  }
    
  char4 _cl_overloadable
  convert_char4_sat(float4 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(char);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char4(a < (float)DST_MIN) ? (char4)DST_MIN :
            convert_char4(a > (float)DST_MAX) ? (char4)DST_MAX :
            convert_char4(a));
  }
    
  char8 _cl_overloadable
  convert_char8_sat(float8 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(char);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char8(a < (float)DST_MIN) ? (char8)DST_MIN :
            convert_char8(a > (float)DST_MAX) ? (char8)DST_MAX :
            convert_char8(a));
  }
    
  char16 _cl_overloadable
  convert_char16_sat(float16 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(char);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char16(a < (float)DST_MIN) ? (char16)DST_MIN :
            convert_char16(a > (float)DST_MAX) ? (char16)DST_MAX :
            convert_char16(a));
  }
    
  uchar _cl_overloadable
  convert_uchar_sat(float a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(uchar);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar(a > (float)DST_MAX) ? (uchar)DST_MAX :
            convert_uchar(a));
  }
    
  uchar2 _cl_overloadable
  convert_uchar2_sat(float2 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(uchar);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar2(a > (float)DST_MAX) ? (uchar2)DST_MAX :
            convert_uchar2(a));
  }
    
  uchar4 _cl_overloadable
  convert_uchar4_sat(float4 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(uchar);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar4(a > (float)DST_MAX) ? (uchar4)DST_MAX :
            convert_uchar4(a));
  }
    
  uchar8 _cl_overloadable
  convert_uchar8_sat(float8 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(uchar);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar8(a > (float)DST_MAX) ? (uchar8)DST_MAX :
            convert_uchar8(a));
  }
    
  uchar16 _cl_overloadable
  convert_uchar16_sat(float16 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(uchar);
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar16(a > (float)DST_MAX) ? (uchar16)DST_MAX :
            convert_uchar16(a));
  }
    
  short _cl_overloadable
  convert_short_sat(float a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(short);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short(a < (float)DST_MIN) ? (short)DST_MIN :
            convert_short(a > (float)DST_MAX) ? (short)DST_MAX :
            convert_short(a));
  }
    
  short2 _cl_overloadable
  convert_short2_sat(float2 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(short);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short2(a < (float)DST_MIN) ? (short2)DST_MIN :
            convert_short2(a > (float)DST_MAX) ? (short2)DST_MAX :
            convert_short2(a));
  }
    
  short4 _cl_overloadable
  convert_short4_sat(float4 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(short);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short4(a < (float)DST_MIN) ? (short4)DST_MIN :
            convert_short4(a > (float)DST_MAX) ? (short4)DST_MAX :
            convert_short4(a));
  }
    
  short8 _cl_overloadable
  convert_short8_sat(float8 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(short);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short8(a < (float)DST_MIN) ? (short8)DST_MIN :
            convert_short8(a > (float)DST_MAX) ? (short8)DST_MAX :
            convert_short8(a));
  }
    
  short16 _cl_overloadable
  convert_short16_sat(float16 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(short);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short16(a < (float)DST_MIN) ? (short16)DST_MIN :
            convert_short16(a > (float)DST_MAX) ? (short16)DST_MAX :
            convert_short16(a));
  }
    
  ushort _cl_overloadable
  convert_ushort_sat(float a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(ushort);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort(a > (float)DST_MAX) ? (ushort)DST_MAX :
            convert_ushort(a));
  }
    
  ushort2 _cl_overloadable
  convert_ushort2_sat(float2 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(ushort);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort2(a > (float)DST_MAX) ? (ushort2)DST_MAX :
            convert_ushort2(a));
  }
    
  ushort4 _cl_overloadable
  convert_ushort4_sat(float4 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(ushort);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort4(a > (float)DST_MAX) ? (ushort4)DST_MAX :
            convert_ushort4(a));
  }
    
  ushort8 _cl_overloadable
  convert_ushort8_sat(float8 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(ushort);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort8(a > (float)DST_MAX) ? (ushort8)DST_MAX :
            convert_ushort8(a));
  }
    
  ushort16 _cl_overloadable
  convert_ushort16_sat(float16 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(ushort);
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort16(a > (float)DST_MAX) ? (ushort16)DST_MAX :
            convert_ushort16(a));
  }
    
  int _cl_overloadable
  convert_int_sat(float a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(int);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int(a < (float)DST_MIN) ? (int)DST_MIN :
            convert_int(a > (float)DST_MAX) ? (int)DST_MAX :
            convert_int(a));
  }
    
  int2 _cl_overloadable
  convert_int2_sat(float2 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(int);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int2(a < (float)DST_MIN) ? (int2)DST_MIN :
            convert_int2(a > (float)DST_MAX) ? (int2)DST_MAX :
            convert_int2(a));
  }
    
  int4 _cl_overloadable
  convert_int4_sat(float4 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(int);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int4(a < (float)DST_MIN) ? (int4)DST_MIN :
            convert_int4(a > (float)DST_MAX) ? (int4)DST_MAX :
            convert_int4(a));
  }
    
  int8 _cl_overloadable
  convert_int8_sat(float8 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(int);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int8(a < (float)DST_MIN) ? (int8)DST_MIN :
            convert_int8(a > (float)DST_MAX) ? (int8)DST_MAX :
            convert_int8(a));
  }
    
  int16 _cl_overloadable
  convert_int16_sat(float16 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(int);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int16(a < (float)DST_MIN) ? (int16)DST_MIN :
            convert_int16(a > (float)DST_MAX) ? (int16)DST_MAX :
            convert_int16(a));
  }
    
  uint _cl_overloadable
  convert_uint_sat(float a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(uint);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint(a > (float)DST_MAX) ? (uint)DST_MAX :
            convert_uint(a));
  }
    
  uint2 _cl_overloadable
  convert_uint2_sat(float2 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(uint);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint2(a > (float)DST_MAX) ? (uint2)DST_MAX :
            convert_uint2(a));
  }
    
  uint4 _cl_overloadable
  convert_uint4_sat(float4 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(uint);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint4(a > (float)DST_MAX) ? (uint4)DST_MAX :
            convert_uint4(a));
  }
    
  uint8 _cl_overloadable
  convert_uint8_sat(float8 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(uint);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint8(a > (float)DST_MAX) ? (uint8)DST_MAX :
            convert_uint8(a));
  }
    
  uint16 _cl_overloadable
  convert_uint16_sat(float16 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(uint);
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint16(a > (float)DST_MAX) ? (uint16)DST_MAX :
            convert_uint16(a));
  }
    
__IF_INT64(
  long _cl_overloadable
  convert_long_sat(float a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(long);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long(a < (float)DST_MIN) ? (long)DST_MIN :
            convert_long(a > (float)DST_MAX) ? (long)DST_MAX :
            convert_long(a));
  }
    
  long2 _cl_overloadable
  convert_long2_sat(float2 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(long);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long2(a < (float)DST_MIN) ? (long2)DST_MIN :
            convert_long2(a > (float)DST_MAX) ? (long2)DST_MAX :
            convert_long2(a));
  }
    
  long4 _cl_overloadable
  convert_long4_sat(float4 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(long);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long4(a < (float)DST_MIN) ? (long4)DST_MIN :
            convert_long4(a > (float)DST_MAX) ? (long4)DST_MAX :
            convert_long4(a));
  }
    
  long8 _cl_overloadable
  convert_long8_sat(float8 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(long);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long8(a < (float)DST_MIN) ? (long8)DST_MIN :
            convert_long8(a > (float)DST_MAX) ? (long8)DST_MAX :
            convert_long8(a));
  }
    
  long16 _cl_overloadable
  convert_long16_sat(float16 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(long);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long16(a < (float)DST_MIN) ? (long16)DST_MIN :
            convert_long16(a > (float)DST_MAX) ? (long16)DST_MAX :
            convert_long16(a));
  }
    
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat(float a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(ulong);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong(a > (float)DST_MAX) ? (ulong)DST_MAX :
            convert_ulong(a));
  }
    
  ulong2 _cl_overloadable
  convert_ulong2_sat(float2 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(ulong);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong2(a > (float)DST_MAX) ? (ulong2)DST_MAX :
            convert_ulong2(a));
  }
    
  ulong4 _cl_overloadable
  convert_ulong4_sat(float4 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(ulong);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong4(a > (float)DST_MAX) ? (ulong4)DST_MAX :
            convert_ulong4(a));
  }
    
  ulong8 _cl_overloadable
  convert_ulong8_sat(float8 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(ulong);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong8(a > (float)DST_MAX) ? (ulong8)DST_MAX :
            convert_ulong8(a));
  }
    
  ulong16 _cl_overloadable
  convert_ulong16_sat(float16 a)
  {
    int const src_size = sizeof(float);
    int const dst_size = sizeof(ulong);
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong16(a > (float)DST_MAX) ? (ulong16)DST_MAX :
            convert_ulong16(a));
  }
    
)
__IF_FP64(
  char _cl_overloadable
  convert_char_sat(double a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char(a < (double)DST_MIN) ? (char)DST_MIN :
            convert_char(a > (double)DST_MAX) ? (char)DST_MAX :
            convert_char(a));
  }
    
  char2 _cl_overloadable
  convert_char2_sat(double2 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char2(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char2(a < (double)DST_MIN) ? (char2)DST_MIN :
            convert_char2(a > (double)DST_MAX) ? (char2)DST_MAX :
            convert_char2(a));
  }
    
  char4 _cl_overloadable
  convert_char4_sat(double4 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char4(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char4(a < (double)DST_MIN) ? (char4)DST_MIN :
            convert_char4(a > (double)DST_MAX) ? (char4)DST_MAX :
            convert_char4(a));
  }
    
  char8 _cl_overloadable
  convert_char8_sat(double8 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char8(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char8(a < (double)DST_MIN) ? (char8)DST_MIN :
            convert_char8(a > (double)DST_MAX) ? (char8)DST_MAX :
            convert_char8(a));
  }
    
  char16 _cl_overloadable
  convert_char16_sat(double16 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(char);
    if (dst_size >= src_size) return convert_char16(a);
    char const DST_MIN = (char)1 << (char)(CHAR_BIT * dst_size - 1);
    char const DST_MAX = DST_MIN - (char)1;
    return (convert_char16(a < (double)DST_MIN) ? (char16)DST_MIN :
            convert_char16(a > (double)DST_MAX) ? (char16)DST_MAX :
            convert_char16(a));
  }
    
)
__IF_FP64(
  uchar _cl_overloadable
  convert_uchar_sat(double a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar(a < (double)0) ? (uchar)0 :
              convert_uchar(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar(a < (double)0      ) ? (uchar)0 :
            convert_uchar(a > (double)DST_MAX) ? (uchar)DST_MAX :
            convert_uchar(a));
  }
    
  uchar2 _cl_overloadable
  convert_uchar2_sat(double2 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar2(a < (double)0) ? (uchar2)0 :
              convert_uchar2(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar2(a < (double)0      ) ? (uchar2)0 :
            convert_uchar2(a > (double)DST_MAX) ? (uchar2)DST_MAX :
            convert_uchar2(a));
  }
    
  uchar4 _cl_overloadable
  convert_uchar4_sat(double4 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar4(a < (double)0) ? (uchar4)0 :
              convert_uchar4(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar4(a < (double)0      ) ? (uchar4)0 :
            convert_uchar4(a > (double)DST_MAX) ? (uchar4)DST_MAX :
            convert_uchar4(a));
  }
    
  uchar8 _cl_overloadable
  convert_uchar8_sat(double8 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar8(a < (double)0) ? (uchar8)0 :
              convert_uchar8(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar8(a < (double)0      ) ? (uchar8)0 :
            convert_uchar8(a > (double)DST_MAX) ? (uchar8)DST_MAX :
            convert_uchar8(a));
  }
    
  uchar16 _cl_overloadable
  convert_uchar16_sat(double16 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(uchar);
    if (dst_size >= src_size) {
      return (convert_uchar16(a < (double)0) ? (uchar16)0 :
              convert_uchar16(a));
    }
    uchar const DST_MAX = (uchar)0 - (uchar)1;
    return (convert_uchar16(a < (double)0      ) ? (uchar16)0 :
            convert_uchar16(a > (double)DST_MAX) ? (uchar16)DST_MAX :
            convert_uchar16(a));
  }
    
)
__IF_FP64(
  short _cl_overloadable
  convert_short_sat(double a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short(a < (double)DST_MIN) ? (short)DST_MIN :
            convert_short(a > (double)DST_MAX) ? (short)DST_MAX :
            convert_short(a));
  }
    
  short2 _cl_overloadable
  convert_short2_sat(double2 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short2(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short2(a < (double)DST_MIN) ? (short2)DST_MIN :
            convert_short2(a > (double)DST_MAX) ? (short2)DST_MAX :
            convert_short2(a));
  }
    
  short4 _cl_overloadable
  convert_short4_sat(double4 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short4(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short4(a < (double)DST_MIN) ? (short4)DST_MIN :
            convert_short4(a > (double)DST_MAX) ? (short4)DST_MAX :
            convert_short4(a));
  }
    
  short8 _cl_overloadable
  convert_short8_sat(double8 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short8(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short8(a < (double)DST_MIN) ? (short8)DST_MIN :
            convert_short8(a > (double)DST_MAX) ? (short8)DST_MAX :
            convert_short8(a));
  }
    
  short16 _cl_overloadable
  convert_short16_sat(double16 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(short);
    if (dst_size >= src_size) return convert_short16(a);
    short const DST_MIN = (short)1 << (short)(CHAR_BIT * dst_size - 1);
    short const DST_MAX = DST_MIN - (short)1;
    return (convert_short16(a < (double)DST_MIN) ? (short16)DST_MIN :
            convert_short16(a > (double)DST_MAX) ? (short16)DST_MAX :
            convert_short16(a));
  }
    
)
__IF_FP64(
  ushort _cl_overloadable
  convert_ushort_sat(double a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort(a < (double)0) ? (ushort)0 :
              convert_ushort(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort(a < (double)0      ) ? (ushort)0 :
            convert_ushort(a > (double)DST_MAX) ? (ushort)DST_MAX :
            convert_ushort(a));
  }
    
  ushort2 _cl_overloadable
  convert_ushort2_sat(double2 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort2(a < (double)0) ? (ushort2)0 :
              convert_ushort2(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort2(a < (double)0      ) ? (ushort2)0 :
            convert_ushort2(a > (double)DST_MAX) ? (ushort2)DST_MAX :
            convert_ushort2(a));
  }
    
  ushort4 _cl_overloadable
  convert_ushort4_sat(double4 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort4(a < (double)0) ? (ushort4)0 :
              convert_ushort4(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort4(a < (double)0      ) ? (ushort4)0 :
            convert_ushort4(a > (double)DST_MAX) ? (ushort4)DST_MAX :
            convert_ushort4(a));
  }
    
  ushort8 _cl_overloadable
  convert_ushort8_sat(double8 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort8(a < (double)0) ? (ushort8)0 :
              convert_ushort8(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort8(a < (double)0      ) ? (ushort8)0 :
            convert_ushort8(a > (double)DST_MAX) ? (ushort8)DST_MAX :
            convert_ushort8(a));
  }
    
  ushort16 _cl_overloadable
  convert_ushort16_sat(double16 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(ushort);
    if (dst_size >= src_size) {
      return (convert_ushort16(a < (double)0) ? (ushort16)0 :
              convert_ushort16(a));
    }
    ushort const DST_MAX = (ushort)0 - (ushort)1;
    return (convert_ushort16(a < (double)0      ) ? (ushort16)0 :
            convert_ushort16(a > (double)DST_MAX) ? (ushort16)DST_MAX :
            convert_ushort16(a));
  }
    
)
__IF_FP64(
  int _cl_overloadable
  convert_int_sat(double a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int(a < (double)DST_MIN) ? (int)DST_MIN :
            convert_int(a > (double)DST_MAX) ? (int)DST_MAX :
            convert_int(a));
  }
    
  int2 _cl_overloadable
  convert_int2_sat(double2 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int2(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int2(a < (double)DST_MIN) ? (int2)DST_MIN :
            convert_int2(a > (double)DST_MAX) ? (int2)DST_MAX :
            convert_int2(a));
  }
    
  int4 _cl_overloadable
  convert_int4_sat(double4 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int4(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int4(a < (double)DST_MIN) ? (int4)DST_MIN :
            convert_int4(a > (double)DST_MAX) ? (int4)DST_MAX :
            convert_int4(a));
  }
    
  int8 _cl_overloadable
  convert_int8_sat(double8 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int8(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int8(a < (double)DST_MIN) ? (int8)DST_MIN :
            convert_int8(a > (double)DST_MAX) ? (int8)DST_MAX :
            convert_int8(a));
  }
    
  int16 _cl_overloadable
  convert_int16_sat(double16 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(int);
    if (dst_size >= src_size) return convert_int16(a);
    int const DST_MIN = (int)1 << (int)(CHAR_BIT * dst_size - 1);
    int const DST_MAX = DST_MIN - (int)1;
    return (convert_int16(a < (double)DST_MIN) ? (int16)DST_MIN :
            convert_int16(a > (double)DST_MAX) ? (int16)DST_MAX :
            convert_int16(a));
  }
    
)
__IF_FP64(
  uint _cl_overloadable
  convert_uint_sat(double a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint(a < (double)0) ? (uint)0 :
              convert_uint(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint(a < (double)0      ) ? (uint)0 :
            convert_uint(a > (double)DST_MAX) ? (uint)DST_MAX :
            convert_uint(a));
  }
    
  uint2 _cl_overloadable
  convert_uint2_sat(double2 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint2(a < (double)0) ? (uint2)0 :
              convert_uint2(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint2(a < (double)0      ) ? (uint2)0 :
            convert_uint2(a > (double)DST_MAX) ? (uint2)DST_MAX :
            convert_uint2(a));
  }
    
  uint4 _cl_overloadable
  convert_uint4_sat(double4 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint4(a < (double)0) ? (uint4)0 :
              convert_uint4(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint4(a < (double)0      ) ? (uint4)0 :
            convert_uint4(a > (double)DST_MAX) ? (uint4)DST_MAX :
            convert_uint4(a));
  }
    
  uint8 _cl_overloadable
  convert_uint8_sat(double8 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint8(a < (double)0) ? (uint8)0 :
              convert_uint8(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint8(a < (double)0      ) ? (uint8)0 :
            convert_uint8(a > (double)DST_MAX) ? (uint8)DST_MAX :
            convert_uint8(a));
  }
    
  uint16 _cl_overloadable
  convert_uint16_sat(double16 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(uint);
    if (dst_size >= src_size) {
      return (convert_uint16(a < (double)0) ? (uint16)0 :
              convert_uint16(a));
    }
    uint const DST_MAX = (uint)0 - (uint)1;
    return (convert_uint16(a < (double)0      ) ? (uint16)0 :
            convert_uint16(a > (double)DST_MAX) ? (uint16)DST_MAX :
            convert_uint16(a));
  }
    
)
__IF_INT64(
  long _cl_overloadable
  convert_long_sat(double a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long(a < (double)DST_MIN) ? (long)DST_MIN :
            convert_long(a > (double)DST_MAX) ? (long)DST_MAX :
            convert_long(a));
  }
    
  long2 _cl_overloadable
  convert_long2_sat(double2 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long2(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long2(a < (double)DST_MIN) ? (long2)DST_MIN :
            convert_long2(a > (double)DST_MAX) ? (long2)DST_MAX :
            convert_long2(a));
  }
    
  long4 _cl_overloadable
  convert_long4_sat(double4 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long4(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long4(a < (double)DST_MIN) ? (long4)DST_MIN :
            convert_long4(a > (double)DST_MAX) ? (long4)DST_MAX :
            convert_long4(a));
  }
    
  long8 _cl_overloadable
  convert_long8_sat(double8 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long8(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long8(a < (double)DST_MIN) ? (long8)DST_MIN :
            convert_long8(a > (double)DST_MAX) ? (long8)DST_MAX :
            convert_long8(a));
  }
    
  long16 _cl_overloadable
  convert_long16_sat(double16 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(long);
    if (dst_size >= src_size) return convert_long16(a);
    long const DST_MIN = (long)1 << (long)(CHAR_BIT * dst_size - 1);
    long const DST_MAX = DST_MIN - (long)1;
    return (convert_long16(a < (double)DST_MIN) ? (long16)DST_MIN :
            convert_long16(a > (double)DST_MAX) ? (long16)DST_MAX :
            convert_long16(a));
  }
    
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat(double a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong(a < (double)0) ? (ulong)0 :
              convert_ulong(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong(a < (double)0      ) ? (ulong)0 :
            convert_ulong(a > (double)DST_MAX) ? (ulong)DST_MAX :
            convert_ulong(a));
  }
    
  ulong2 _cl_overloadable
  convert_ulong2_sat(double2 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong2(a < (double)0) ? (ulong2)0 :
              convert_ulong2(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong2(a < (double)0      ) ? (ulong2)0 :
            convert_ulong2(a > (double)DST_MAX) ? (ulong2)DST_MAX :
            convert_ulong2(a));
  }
    
  ulong4 _cl_overloadable
  convert_ulong4_sat(double4 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong4(a < (double)0) ? (ulong4)0 :
              convert_ulong4(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong4(a < (double)0      ) ? (ulong4)0 :
            convert_ulong4(a > (double)DST_MAX) ? (ulong4)DST_MAX :
            convert_ulong4(a));
  }
    
  ulong8 _cl_overloadable
  convert_ulong8_sat(double8 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong8(a < (double)0) ? (ulong8)0 :
              convert_ulong8(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong8(a < (double)0      ) ? (ulong8)0 :
            convert_ulong8(a > (double)DST_MAX) ? (ulong8)DST_MAX :
            convert_ulong8(a));
  }
    
  ulong16 _cl_overloadable
  convert_ulong16_sat(double16 a)
  {
    int const src_size = sizeof(double);
    int const dst_size = sizeof(ulong);
    if (dst_size >= src_size) {
      return (convert_ulong16(a < (double)0) ? (ulong16)0 :
              convert_ulong16(a));
    }
    ulong const DST_MAX = (ulong)0 - (ulong)1;
    return (convert_ulong16(a < (double)0      ) ? (ulong16)0 :
            convert_ulong16(a > (double)DST_MAX) ? (ulong16)DST_MAX :
            convert_ulong16(a));
  }
    
)
  char _cl_overloadable
  convert_char_rte(float a)
  {
    cl_set_rounding_mode(1);
    char result = convert_char(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char _cl_overloadable
  convert_char_sat_rte(float a)
  {
    cl_set_rounding_mode(1);
    char result = convert_char_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char _cl_overloadable
  convert_char_rtz(float a)
  {
    cl_set_rounding_mode(0);
    char result = convert_char(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char _cl_overloadable
  convert_char_sat_rtz(float a)
  {
    cl_set_rounding_mode(0);
    char result = convert_char_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char _cl_overloadable
  convert_char_rtp(float a)
  {
    cl_set_rounding_mode(2);
    char result = convert_char(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char _cl_overloadable
  convert_char_sat_rtp(float a)
  {
    cl_set_rounding_mode(2);
    char result = convert_char_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char _cl_overloadable
  convert_char_rtn(float a)
  {
    cl_set_rounding_mode(3);
    char result = convert_char(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char _cl_overloadable
  convert_char_sat_rtn(float a)
  {
    cl_set_rounding_mode(3);
    char result = convert_char_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char2 _cl_overloadable
  convert_char2_rte(float2 a)
  {
    cl_set_rounding_mode(1);
    char2 result = convert_char2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char2 _cl_overloadable
  convert_char2_sat_rte(float2 a)
  {
    cl_set_rounding_mode(1);
    char2 result = convert_char2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char2 _cl_overloadable
  convert_char2_rtz(float2 a)
  {
    cl_set_rounding_mode(0);
    char2 result = convert_char2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char2 _cl_overloadable
  convert_char2_sat_rtz(float2 a)
  {
    cl_set_rounding_mode(0);
    char2 result = convert_char2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char2 _cl_overloadable
  convert_char2_rtp(float2 a)
  {
    cl_set_rounding_mode(2);
    char2 result = convert_char2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char2 _cl_overloadable
  convert_char2_sat_rtp(float2 a)
  {
    cl_set_rounding_mode(2);
    char2 result = convert_char2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char2 _cl_overloadable
  convert_char2_rtn(float2 a)
  {
    cl_set_rounding_mode(3);
    char2 result = convert_char2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char2 _cl_overloadable
  convert_char2_sat_rtn(float2 a)
  {
    cl_set_rounding_mode(3);
    char2 result = convert_char2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char4 _cl_overloadable
  convert_char4_rte(float4 a)
  {
    cl_set_rounding_mode(1);
    char4 result = convert_char4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char4 _cl_overloadable
  convert_char4_sat_rte(float4 a)
  {
    cl_set_rounding_mode(1);
    char4 result = convert_char4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char4 _cl_overloadable
  convert_char4_rtz(float4 a)
  {
    cl_set_rounding_mode(0);
    char4 result = convert_char4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char4 _cl_overloadable
  convert_char4_sat_rtz(float4 a)
  {
    cl_set_rounding_mode(0);
    char4 result = convert_char4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char4 _cl_overloadable
  convert_char4_rtp(float4 a)
  {
    cl_set_rounding_mode(2);
    char4 result = convert_char4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char4 _cl_overloadable
  convert_char4_sat_rtp(float4 a)
  {
    cl_set_rounding_mode(2);
    char4 result = convert_char4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char4 _cl_overloadable
  convert_char4_rtn(float4 a)
  {
    cl_set_rounding_mode(3);
    char4 result = convert_char4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char4 _cl_overloadable
  convert_char4_sat_rtn(float4 a)
  {
    cl_set_rounding_mode(3);
    char4 result = convert_char4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char8 _cl_overloadable
  convert_char8_rte(float8 a)
  {
    cl_set_rounding_mode(1);
    char8 result = convert_char8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char8 _cl_overloadable
  convert_char8_sat_rte(float8 a)
  {
    cl_set_rounding_mode(1);
    char8 result = convert_char8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char8 _cl_overloadable
  convert_char8_rtz(float8 a)
  {
    cl_set_rounding_mode(0);
    char8 result = convert_char8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char8 _cl_overloadable
  convert_char8_sat_rtz(float8 a)
  {
    cl_set_rounding_mode(0);
    char8 result = convert_char8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char8 _cl_overloadable
  convert_char8_rtp(float8 a)
  {
    cl_set_rounding_mode(2);
    char8 result = convert_char8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char8 _cl_overloadable
  convert_char8_sat_rtp(float8 a)
  {
    cl_set_rounding_mode(2);
    char8 result = convert_char8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char8 _cl_overloadable
  convert_char8_rtn(float8 a)
  {
    cl_set_rounding_mode(3);
    char8 result = convert_char8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char8 _cl_overloadable
  convert_char8_sat_rtn(float8 a)
  {
    cl_set_rounding_mode(3);
    char8 result = convert_char8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char16 _cl_overloadable
  convert_char16_rte(float16 a)
  {
    cl_set_rounding_mode(1);
    char16 result = convert_char16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char16 _cl_overloadable
  convert_char16_sat_rte(float16 a)
  {
    cl_set_rounding_mode(1);
    char16 result = convert_char16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char16 _cl_overloadable
  convert_char16_rtz(float16 a)
  {
    cl_set_rounding_mode(0);
    char16 result = convert_char16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char16 _cl_overloadable
  convert_char16_sat_rtz(float16 a)
  {
    cl_set_rounding_mode(0);
    char16 result = convert_char16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char16 _cl_overloadable
  convert_char16_rtp(float16 a)
  {
    cl_set_rounding_mode(2);
    char16 result = convert_char16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char16 _cl_overloadable
  convert_char16_sat_rtp(float16 a)
  {
    cl_set_rounding_mode(2);
    char16 result = convert_char16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char16 _cl_overloadable
  convert_char16_rtn(float16 a)
  {
    cl_set_rounding_mode(3);
    char16 result = convert_char16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  char16 _cl_overloadable
  convert_char16_sat_rtn(float16 a)
  {
    cl_set_rounding_mode(3);
    char16 result = convert_char16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar _cl_overloadable
  convert_uchar_rte(float a)
  {
    cl_set_rounding_mode(1);
    uchar result = convert_uchar(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar _cl_overloadable
  convert_uchar_sat_rte(float a)
  {
    cl_set_rounding_mode(1);
    uchar result = convert_uchar_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar _cl_overloadable
  convert_uchar_rtz(float a)
  {
    cl_set_rounding_mode(0);
    uchar result = convert_uchar(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar _cl_overloadable
  convert_uchar_sat_rtz(float a)
  {
    cl_set_rounding_mode(0);
    uchar result = convert_uchar_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar _cl_overloadable
  convert_uchar_rtp(float a)
  {
    cl_set_rounding_mode(2);
    uchar result = convert_uchar(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar _cl_overloadable
  convert_uchar_sat_rtp(float a)
  {
    cl_set_rounding_mode(2);
    uchar result = convert_uchar_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar _cl_overloadable
  convert_uchar_rtn(float a)
  {
    cl_set_rounding_mode(3);
    uchar result = convert_uchar(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar _cl_overloadable
  convert_uchar_sat_rtn(float a)
  {
    cl_set_rounding_mode(3);
    uchar result = convert_uchar_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar2 _cl_overloadable
  convert_uchar2_rte(float2 a)
  {
    cl_set_rounding_mode(1);
    uchar2 result = convert_uchar2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar2 _cl_overloadable
  convert_uchar2_sat_rte(float2 a)
  {
    cl_set_rounding_mode(1);
    uchar2 result = convert_uchar2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar2 _cl_overloadable
  convert_uchar2_rtz(float2 a)
  {
    cl_set_rounding_mode(0);
    uchar2 result = convert_uchar2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar2 _cl_overloadable
  convert_uchar2_sat_rtz(float2 a)
  {
    cl_set_rounding_mode(0);
    uchar2 result = convert_uchar2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar2 _cl_overloadable
  convert_uchar2_rtp(float2 a)
  {
    cl_set_rounding_mode(2);
    uchar2 result = convert_uchar2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar2 _cl_overloadable
  convert_uchar2_sat_rtp(float2 a)
  {
    cl_set_rounding_mode(2);
    uchar2 result = convert_uchar2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar2 _cl_overloadable
  convert_uchar2_rtn(float2 a)
  {
    cl_set_rounding_mode(3);
    uchar2 result = convert_uchar2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar2 _cl_overloadable
  convert_uchar2_sat_rtn(float2 a)
  {
    cl_set_rounding_mode(3);
    uchar2 result = convert_uchar2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar4 _cl_overloadable
  convert_uchar4_rte(float4 a)
  {
    cl_set_rounding_mode(1);
    uchar4 result = convert_uchar4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar4 _cl_overloadable
  convert_uchar4_sat_rte(float4 a)
  {
    cl_set_rounding_mode(1);
    uchar4 result = convert_uchar4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar4 _cl_overloadable
  convert_uchar4_rtz(float4 a)
  {
    cl_set_rounding_mode(0);
    uchar4 result = convert_uchar4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar4 _cl_overloadable
  convert_uchar4_sat_rtz(float4 a)
  {
    cl_set_rounding_mode(0);
    uchar4 result = convert_uchar4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar4 _cl_overloadable
  convert_uchar4_rtp(float4 a)
  {
    cl_set_rounding_mode(2);
    uchar4 result = convert_uchar4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar4 _cl_overloadable
  convert_uchar4_sat_rtp(float4 a)
  {
    cl_set_rounding_mode(2);
    uchar4 result = convert_uchar4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar4 _cl_overloadable
  convert_uchar4_rtn(float4 a)
  {
    cl_set_rounding_mode(3);
    uchar4 result = convert_uchar4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar4 _cl_overloadable
  convert_uchar4_sat_rtn(float4 a)
  {
    cl_set_rounding_mode(3);
    uchar4 result = convert_uchar4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar8 _cl_overloadable
  convert_uchar8_rte(float8 a)
  {
    cl_set_rounding_mode(1);
    uchar8 result = convert_uchar8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar8 _cl_overloadable
  convert_uchar8_sat_rte(float8 a)
  {
    cl_set_rounding_mode(1);
    uchar8 result = convert_uchar8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar8 _cl_overloadable
  convert_uchar8_rtz(float8 a)
  {
    cl_set_rounding_mode(0);
    uchar8 result = convert_uchar8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar8 _cl_overloadable
  convert_uchar8_sat_rtz(float8 a)
  {
    cl_set_rounding_mode(0);
    uchar8 result = convert_uchar8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar8 _cl_overloadable
  convert_uchar8_rtp(float8 a)
  {
    cl_set_rounding_mode(2);
    uchar8 result = convert_uchar8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar8 _cl_overloadable
  convert_uchar8_sat_rtp(float8 a)
  {
    cl_set_rounding_mode(2);
    uchar8 result = convert_uchar8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar8 _cl_overloadable
  convert_uchar8_rtn(float8 a)
  {
    cl_set_rounding_mode(3);
    uchar8 result = convert_uchar8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar8 _cl_overloadable
  convert_uchar8_sat_rtn(float8 a)
  {
    cl_set_rounding_mode(3);
    uchar8 result = convert_uchar8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar16 _cl_overloadable
  convert_uchar16_rte(float16 a)
  {
    cl_set_rounding_mode(1);
    uchar16 result = convert_uchar16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar16 _cl_overloadable
  convert_uchar16_sat_rte(float16 a)
  {
    cl_set_rounding_mode(1);
    uchar16 result = convert_uchar16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar16 _cl_overloadable
  convert_uchar16_rtz(float16 a)
  {
    cl_set_rounding_mode(0);
    uchar16 result = convert_uchar16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar16 _cl_overloadable
  convert_uchar16_sat_rtz(float16 a)
  {
    cl_set_rounding_mode(0);
    uchar16 result = convert_uchar16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar16 _cl_overloadable
  convert_uchar16_rtp(float16 a)
  {
    cl_set_rounding_mode(2);
    uchar16 result = convert_uchar16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar16 _cl_overloadable
  convert_uchar16_sat_rtp(float16 a)
  {
    cl_set_rounding_mode(2);
    uchar16 result = convert_uchar16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar16 _cl_overloadable
  convert_uchar16_rtn(float16 a)
  {
    cl_set_rounding_mode(3);
    uchar16 result = convert_uchar16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uchar16 _cl_overloadable
  convert_uchar16_sat_rtn(float16 a)
  {
    cl_set_rounding_mode(3);
    uchar16 result = convert_uchar16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short _cl_overloadable
  convert_short_rte(float a)
  {
    cl_set_rounding_mode(1);
    short result = convert_short(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short _cl_overloadable
  convert_short_sat_rte(float a)
  {
    cl_set_rounding_mode(1);
    short result = convert_short_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short _cl_overloadable
  convert_short_rtz(float a)
  {
    cl_set_rounding_mode(0);
    short result = convert_short(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short _cl_overloadable
  convert_short_sat_rtz(float a)
  {
    cl_set_rounding_mode(0);
    short result = convert_short_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short _cl_overloadable
  convert_short_rtp(float a)
  {
    cl_set_rounding_mode(2);
    short result = convert_short(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short _cl_overloadable
  convert_short_sat_rtp(float a)
  {
    cl_set_rounding_mode(2);
    short result = convert_short_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short _cl_overloadable
  convert_short_rtn(float a)
  {
    cl_set_rounding_mode(3);
    short result = convert_short(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short _cl_overloadable
  convert_short_sat_rtn(float a)
  {
    cl_set_rounding_mode(3);
    short result = convert_short_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short2 _cl_overloadable
  convert_short2_rte(float2 a)
  {
    cl_set_rounding_mode(1);
    short2 result = convert_short2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short2 _cl_overloadable
  convert_short2_sat_rte(float2 a)
  {
    cl_set_rounding_mode(1);
    short2 result = convert_short2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short2 _cl_overloadable
  convert_short2_rtz(float2 a)
  {
    cl_set_rounding_mode(0);
    short2 result = convert_short2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short2 _cl_overloadable
  convert_short2_sat_rtz(float2 a)
  {
    cl_set_rounding_mode(0);
    short2 result = convert_short2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short2 _cl_overloadable
  convert_short2_rtp(float2 a)
  {
    cl_set_rounding_mode(2);
    short2 result = convert_short2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short2 _cl_overloadable
  convert_short2_sat_rtp(float2 a)
  {
    cl_set_rounding_mode(2);
    short2 result = convert_short2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short2 _cl_overloadable
  convert_short2_rtn(float2 a)
  {
    cl_set_rounding_mode(3);
    short2 result = convert_short2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short2 _cl_overloadable
  convert_short2_sat_rtn(float2 a)
  {
    cl_set_rounding_mode(3);
    short2 result = convert_short2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short4 _cl_overloadable
  convert_short4_rte(float4 a)
  {
    cl_set_rounding_mode(1);
    short4 result = convert_short4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short4 _cl_overloadable
  convert_short4_sat_rte(float4 a)
  {
    cl_set_rounding_mode(1);
    short4 result = convert_short4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short4 _cl_overloadable
  convert_short4_rtz(float4 a)
  {
    cl_set_rounding_mode(0);
    short4 result = convert_short4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short4 _cl_overloadable
  convert_short4_sat_rtz(float4 a)
  {
    cl_set_rounding_mode(0);
    short4 result = convert_short4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short4 _cl_overloadable
  convert_short4_rtp(float4 a)
  {
    cl_set_rounding_mode(2);
    short4 result = convert_short4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short4 _cl_overloadable
  convert_short4_sat_rtp(float4 a)
  {
    cl_set_rounding_mode(2);
    short4 result = convert_short4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short4 _cl_overloadable
  convert_short4_rtn(float4 a)
  {
    cl_set_rounding_mode(3);
    short4 result = convert_short4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short4 _cl_overloadable
  convert_short4_sat_rtn(float4 a)
  {
    cl_set_rounding_mode(3);
    short4 result = convert_short4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short8 _cl_overloadable
  convert_short8_rte(float8 a)
  {
    cl_set_rounding_mode(1);
    short8 result = convert_short8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short8 _cl_overloadable
  convert_short8_sat_rte(float8 a)
  {
    cl_set_rounding_mode(1);
    short8 result = convert_short8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short8 _cl_overloadable
  convert_short8_rtz(float8 a)
  {
    cl_set_rounding_mode(0);
    short8 result = convert_short8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short8 _cl_overloadable
  convert_short8_sat_rtz(float8 a)
  {
    cl_set_rounding_mode(0);
    short8 result = convert_short8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short8 _cl_overloadable
  convert_short8_rtp(float8 a)
  {
    cl_set_rounding_mode(2);
    short8 result = convert_short8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short8 _cl_overloadable
  convert_short8_sat_rtp(float8 a)
  {
    cl_set_rounding_mode(2);
    short8 result = convert_short8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short8 _cl_overloadable
  convert_short8_rtn(float8 a)
  {
    cl_set_rounding_mode(3);
    short8 result = convert_short8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short8 _cl_overloadable
  convert_short8_sat_rtn(float8 a)
  {
    cl_set_rounding_mode(3);
    short8 result = convert_short8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short16 _cl_overloadable
  convert_short16_rte(float16 a)
  {
    cl_set_rounding_mode(1);
    short16 result = convert_short16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short16 _cl_overloadable
  convert_short16_sat_rte(float16 a)
  {
    cl_set_rounding_mode(1);
    short16 result = convert_short16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short16 _cl_overloadable
  convert_short16_rtz(float16 a)
  {
    cl_set_rounding_mode(0);
    short16 result = convert_short16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short16 _cl_overloadable
  convert_short16_sat_rtz(float16 a)
  {
    cl_set_rounding_mode(0);
    short16 result = convert_short16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short16 _cl_overloadable
  convert_short16_rtp(float16 a)
  {
    cl_set_rounding_mode(2);
    short16 result = convert_short16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short16 _cl_overloadable
  convert_short16_sat_rtp(float16 a)
  {
    cl_set_rounding_mode(2);
    short16 result = convert_short16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short16 _cl_overloadable
  convert_short16_rtn(float16 a)
  {
    cl_set_rounding_mode(3);
    short16 result = convert_short16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  short16 _cl_overloadable
  convert_short16_sat_rtn(float16 a)
  {
    cl_set_rounding_mode(3);
    short16 result = convert_short16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort _cl_overloadable
  convert_ushort_rte(float a)
  {
    cl_set_rounding_mode(1);
    ushort result = convert_ushort(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort _cl_overloadable
  convert_ushort_sat_rte(float a)
  {
    cl_set_rounding_mode(1);
    ushort result = convert_ushort_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort _cl_overloadable
  convert_ushort_rtz(float a)
  {
    cl_set_rounding_mode(0);
    ushort result = convert_ushort(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort _cl_overloadable
  convert_ushort_sat_rtz(float a)
  {
    cl_set_rounding_mode(0);
    ushort result = convert_ushort_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort _cl_overloadable
  convert_ushort_rtp(float a)
  {
    cl_set_rounding_mode(2);
    ushort result = convert_ushort(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort _cl_overloadable
  convert_ushort_sat_rtp(float a)
  {
    cl_set_rounding_mode(2);
    ushort result = convert_ushort_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort _cl_overloadable
  convert_ushort_rtn(float a)
  {
    cl_set_rounding_mode(3);
    ushort result = convert_ushort(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort _cl_overloadable
  convert_ushort_sat_rtn(float a)
  {
    cl_set_rounding_mode(3);
    ushort result = convert_ushort_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort2 _cl_overloadable
  convert_ushort2_rte(float2 a)
  {
    cl_set_rounding_mode(1);
    ushort2 result = convert_ushort2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort2 _cl_overloadable
  convert_ushort2_sat_rte(float2 a)
  {
    cl_set_rounding_mode(1);
    ushort2 result = convert_ushort2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort2 _cl_overloadable
  convert_ushort2_rtz(float2 a)
  {
    cl_set_rounding_mode(0);
    ushort2 result = convert_ushort2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort2 _cl_overloadable
  convert_ushort2_sat_rtz(float2 a)
  {
    cl_set_rounding_mode(0);
    ushort2 result = convert_ushort2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort2 _cl_overloadable
  convert_ushort2_rtp(float2 a)
  {
    cl_set_rounding_mode(2);
    ushort2 result = convert_ushort2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort2 _cl_overloadable
  convert_ushort2_sat_rtp(float2 a)
  {
    cl_set_rounding_mode(2);
    ushort2 result = convert_ushort2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort2 _cl_overloadable
  convert_ushort2_rtn(float2 a)
  {
    cl_set_rounding_mode(3);
    ushort2 result = convert_ushort2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort2 _cl_overloadable
  convert_ushort2_sat_rtn(float2 a)
  {
    cl_set_rounding_mode(3);
    ushort2 result = convert_ushort2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort4 _cl_overloadable
  convert_ushort4_rte(float4 a)
  {
    cl_set_rounding_mode(1);
    ushort4 result = convert_ushort4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort4 _cl_overloadable
  convert_ushort4_sat_rte(float4 a)
  {
    cl_set_rounding_mode(1);
    ushort4 result = convert_ushort4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort4 _cl_overloadable
  convert_ushort4_rtz(float4 a)
  {
    cl_set_rounding_mode(0);
    ushort4 result = convert_ushort4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort4 _cl_overloadable
  convert_ushort4_sat_rtz(float4 a)
  {
    cl_set_rounding_mode(0);
    ushort4 result = convert_ushort4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort4 _cl_overloadable
  convert_ushort4_rtp(float4 a)
  {
    cl_set_rounding_mode(2);
    ushort4 result = convert_ushort4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort4 _cl_overloadable
  convert_ushort4_sat_rtp(float4 a)
  {
    cl_set_rounding_mode(2);
    ushort4 result = convert_ushort4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort4 _cl_overloadable
  convert_ushort4_rtn(float4 a)
  {
    cl_set_rounding_mode(3);
    ushort4 result = convert_ushort4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort4 _cl_overloadable
  convert_ushort4_sat_rtn(float4 a)
  {
    cl_set_rounding_mode(3);
    ushort4 result = convert_ushort4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort8 _cl_overloadable
  convert_ushort8_rte(float8 a)
  {
    cl_set_rounding_mode(1);
    ushort8 result = convert_ushort8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort8 _cl_overloadable
  convert_ushort8_sat_rte(float8 a)
  {
    cl_set_rounding_mode(1);
    ushort8 result = convert_ushort8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort8 _cl_overloadable
  convert_ushort8_rtz(float8 a)
  {
    cl_set_rounding_mode(0);
    ushort8 result = convert_ushort8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort8 _cl_overloadable
  convert_ushort8_sat_rtz(float8 a)
  {
    cl_set_rounding_mode(0);
    ushort8 result = convert_ushort8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort8 _cl_overloadable
  convert_ushort8_rtp(float8 a)
  {
    cl_set_rounding_mode(2);
    ushort8 result = convert_ushort8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort8 _cl_overloadable
  convert_ushort8_sat_rtp(float8 a)
  {
    cl_set_rounding_mode(2);
    ushort8 result = convert_ushort8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort8 _cl_overloadable
  convert_ushort8_rtn(float8 a)
  {
    cl_set_rounding_mode(3);
    ushort8 result = convert_ushort8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort8 _cl_overloadable
  convert_ushort8_sat_rtn(float8 a)
  {
    cl_set_rounding_mode(3);
    ushort8 result = convert_ushort8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort16 _cl_overloadable
  convert_ushort16_rte(float16 a)
  {
    cl_set_rounding_mode(1);
    ushort16 result = convert_ushort16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort16 _cl_overloadable
  convert_ushort16_sat_rte(float16 a)
  {
    cl_set_rounding_mode(1);
    ushort16 result = convert_ushort16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort16 _cl_overloadable
  convert_ushort16_rtz(float16 a)
  {
    cl_set_rounding_mode(0);
    ushort16 result = convert_ushort16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort16 _cl_overloadable
  convert_ushort16_sat_rtz(float16 a)
  {
    cl_set_rounding_mode(0);
    ushort16 result = convert_ushort16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort16 _cl_overloadable
  convert_ushort16_rtp(float16 a)
  {
    cl_set_rounding_mode(2);
    ushort16 result = convert_ushort16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort16 _cl_overloadable
  convert_ushort16_sat_rtp(float16 a)
  {
    cl_set_rounding_mode(2);
    ushort16 result = convert_ushort16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort16 _cl_overloadable
  convert_ushort16_rtn(float16 a)
  {
    cl_set_rounding_mode(3);
    ushort16 result = convert_ushort16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  ushort16 _cl_overloadable
  convert_ushort16_sat_rtn(float16 a)
  {
    cl_set_rounding_mode(3);
    ushort16 result = convert_ushort16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int _cl_overloadable
  convert_int_rte(float a)
  {
    cl_set_rounding_mode(1);
    int result = convert_int(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int _cl_overloadable
  convert_int_sat_rte(float a)
  {
    cl_set_rounding_mode(1);
    int result = convert_int_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int _cl_overloadable
  convert_int_rtz(float a)
  {
    cl_set_rounding_mode(0);
    int result = convert_int(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int _cl_overloadable
  convert_int_sat_rtz(float a)
  {
    cl_set_rounding_mode(0);
    int result = convert_int_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int _cl_overloadable
  convert_int_rtp(float a)
  {
    cl_set_rounding_mode(2);
    int result = convert_int(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int _cl_overloadable
  convert_int_sat_rtp(float a)
  {
    cl_set_rounding_mode(2);
    int result = convert_int_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int _cl_overloadable
  convert_int_rtn(float a)
  {
    cl_set_rounding_mode(3);
    int result = convert_int(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int _cl_overloadable
  convert_int_sat_rtn(float a)
  {
    cl_set_rounding_mode(3);
    int result = convert_int_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int2 _cl_overloadable
  convert_int2_rte(float2 a)
  {
    cl_set_rounding_mode(1);
    int2 result = convert_int2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int2 _cl_overloadable
  convert_int2_sat_rte(float2 a)
  {
    cl_set_rounding_mode(1);
    int2 result = convert_int2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int2 _cl_overloadable
  convert_int2_rtz(float2 a)
  {
    cl_set_rounding_mode(0);
    int2 result = convert_int2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int2 _cl_overloadable
  convert_int2_sat_rtz(float2 a)
  {
    cl_set_rounding_mode(0);
    int2 result = convert_int2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int2 _cl_overloadable
  convert_int2_rtp(float2 a)
  {
    cl_set_rounding_mode(2);
    int2 result = convert_int2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int2 _cl_overloadable
  convert_int2_sat_rtp(float2 a)
  {
    cl_set_rounding_mode(2);
    int2 result = convert_int2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int2 _cl_overloadable
  convert_int2_rtn(float2 a)
  {
    cl_set_rounding_mode(3);
    int2 result = convert_int2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int2 _cl_overloadable
  convert_int2_sat_rtn(float2 a)
  {
    cl_set_rounding_mode(3);
    int2 result = convert_int2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int4 _cl_overloadable
  convert_int4_rte(float4 a)
  {
    cl_set_rounding_mode(1);
    int4 result = convert_int4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int4 _cl_overloadable
  convert_int4_sat_rte(float4 a)
  {
    cl_set_rounding_mode(1);
    int4 result = convert_int4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int4 _cl_overloadable
  convert_int4_rtz(float4 a)
  {
    cl_set_rounding_mode(0);
    int4 result = convert_int4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int4 _cl_overloadable
  convert_int4_sat_rtz(float4 a)
  {
    cl_set_rounding_mode(0);
    int4 result = convert_int4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int4 _cl_overloadable
  convert_int4_rtp(float4 a)
  {
    cl_set_rounding_mode(2);
    int4 result = convert_int4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int4 _cl_overloadable
  convert_int4_sat_rtp(float4 a)
  {
    cl_set_rounding_mode(2);
    int4 result = convert_int4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int4 _cl_overloadable
  convert_int4_rtn(float4 a)
  {
    cl_set_rounding_mode(3);
    int4 result = convert_int4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int4 _cl_overloadable
  convert_int4_sat_rtn(float4 a)
  {
    cl_set_rounding_mode(3);
    int4 result = convert_int4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int8 _cl_overloadable
  convert_int8_rte(float8 a)
  {
    cl_set_rounding_mode(1);
    int8 result = convert_int8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int8 _cl_overloadable
  convert_int8_sat_rte(float8 a)
  {
    cl_set_rounding_mode(1);
    int8 result = convert_int8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int8 _cl_overloadable
  convert_int8_rtz(float8 a)
  {
    cl_set_rounding_mode(0);
    int8 result = convert_int8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int8 _cl_overloadable
  convert_int8_sat_rtz(float8 a)
  {
    cl_set_rounding_mode(0);
    int8 result = convert_int8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int8 _cl_overloadable
  convert_int8_rtp(float8 a)
  {
    cl_set_rounding_mode(2);
    int8 result = convert_int8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int8 _cl_overloadable
  convert_int8_sat_rtp(float8 a)
  {
    cl_set_rounding_mode(2);
    int8 result = convert_int8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int8 _cl_overloadable
  convert_int8_rtn(float8 a)
  {
    cl_set_rounding_mode(3);
    int8 result = convert_int8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int8 _cl_overloadable
  convert_int8_sat_rtn(float8 a)
  {
    cl_set_rounding_mode(3);
    int8 result = convert_int8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int16 _cl_overloadable
  convert_int16_rte(float16 a)
  {
    cl_set_rounding_mode(1);
    int16 result = convert_int16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int16 _cl_overloadable
  convert_int16_sat_rte(float16 a)
  {
    cl_set_rounding_mode(1);
    int16 result = convert_int16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int16 _cl_overloadable
  convert_int16_rtz(float16 a)
  {
    cl_set_rounding_mode(0);
    int16 result = convert_int16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int16 _cl_overloadable
  convert_int16_sat_rtz(float16 a)
  {
    cl_set_rounding_mode(0);
    int16 result = convert_int16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int16 _cl_overloadable
  convert_int16_rtp(float16 a)
  {
    cl_set_rounding_mode(2);
    int16 result = convert_int16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int16 _cl_overloadable
  convert_int16_sat_rtp(float16 a)
  {
    cl_set_rounding_mode(2);
    int16 result = convert_int16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int16 _cl_overloadable
  convert_int16_rtn(float16 a)
  {
    cl_set_rounding_mode(3);
    int16 result = convert_int16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  int16 _cl_overloadable
  convert_int16_sat_rtn(float16 a)
  {
    cl_set_rounding_mode(3);
    int16 result = convert_int16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint _cl_overloadable
  convert_uint_rte(float a)
  {
    cl_set_rounding_mode(1);
    uint result = convert_uint(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint _cl_overloadable
  convert_uint_sat_rte(float a)
  {
    cl_set_rounding_mode(1);
    uint result = convert_uint_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint _cl_overloadable
  convert_uint_rtz(float a)
  {
    cl_set_rounding_mode(0);
    uint result = convert_uint(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint _cl_overloadable
  convert_uint_sat_rtz(float a)
  {
    cl_set_rounding_mode(0);
    uint result = convert_uint_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint _cl_overloadable
  convert_uint_rtp(float a)
  {
    cl_set_rounding_mode(2);
    uint result = convert_uint(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint _cl_overloadable
  convert_uint_sat_rtp(float a)
  {
    cl_set_rounding_mode(2);
    uint result = convert_uint_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint _cl_overloadable
  convert_uint_rtn(float a)
  {
    cl_set_rounding_mode(3);
    uint result = convert_uint(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint _cl_overloadable
  convert_uint_sat_rtn(float a)
  {
    cl_set_rounding_mode(3);
    uint result = convert_uint_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint2 _cl_overloadable
  convert_uint2_rte(float2 a)
  {
    cl_set_rounding_mode(1);
    uint2 result = convert_uint2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint2 _cl_overloadable
  convert_uint2_sat_rte(float2 a)
  {
    cl_set_rounding_mode(1);
    uint2 result = convert_uint2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint2 _cl_overloadable
  convert_uint2_rtz(float2 a)
  {
    cl_set_rounding_mode(0);
    uint2 result = convert_uint2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint2 _cl_overloadable
  convert_uint2_sat_rtz(float2 a)
  {
    cl_set_rounding_mode(0);
    uint2 result = convert_uint2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint2 _cl_overloadable
  convert_uint2_rtp(float2 a)
  {
    cl_set_rounding_mode(2);
    uint2 result = convert_uint2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint2 _cl_overloadable
  convert_uint2_sat_rtp(float2 a)
  {
    cl_set_rounding_mode(2);
    uint2 result = convert_uint2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint2 _cl_overloadable
  convert_uint2_rtn(float2 a)
  {
    cl_set_rounding_mode(3);
    uint2 result = convert_uint2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint2 _cl_overloadable
  convert_uint2_sat_rtn(float2 a)
  {
    cl_set_rounding_mode(3);
    uint2 result = convert_uint2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint4 _cl_overloadable
  convert_uint4_rte(float4 a)
  {
    cl_set_rounding_mode(1);
    uint4 result = convert_uint4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint4 _cl_overloadable
  convert_uint4_sat_rte(float4 a)
  {
    cl_set_rounding_mode(1);
    uint4 result = convert_uint4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint4 _cl_overloadable
  convert_uint4_rtz(float4 a)
  {
    cl_set_rounding_mode(0);
    uint4 result = convert_uint4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint4 _cl_overloadable
  convert_uint4_sat_rtz(float4 a)
  {
    cl_set_rounding_mode(0);
    uint4 result = convert_uint4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint4 _cl_overloadable
  convert_uint4_rtp(float4 a)
  {
    cl_set_rounding_mode(2);
    uint4 result = convert_uint4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint4 _cl_overloadable
  convert_uint4_sat_rtp(float4 a)
  {
    cl_set_rounding_mode(2);
    uint4 result = convert_uint4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint4 _cl_overloadable
  convert_uint4_rtn(float4 a)
  {
    cl_set_rounding_mode(3);
    uint4 result = convert_uint4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint4 _cl_overloadable
  convert_uint4_sat_rtn(float4 a)
  {
    cl_set_rounding_mode(3);
    uint4 result = convert_uint4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint8 _cl_overloadable
  convert_uint8_rte(float8 a)
  {
    cl_set_rounding_mode(1);
    uint8 result = convert_uint8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint8 _cl_overloadable
  convert_uint8_sat_rte(float8 a)
  {
    cl_set_rounding_mode(1);
    uint8 result = convert_uint8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint8 _cl_overloadable
  convert_uint8_rtz(float8 a)
  {
    cl_set_rounding_mode(0);
    uint8 result = convert_uint8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint8 _cl_overloadable
  convert_uint8_sat_rtz(float8 a)
  {
    cl_set_rounding_mode(0);
    uint8 result = convert_uint8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint8 _cl_overloadable
  convert_uint8_rtp(float8 a)
  {
    cl_set_rounding_mode(2);
    uint8 result = convert_uint8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint8 _cl_overloadable
  convert_uint8_sat_rtp(float8 a)
  {
    cl_set_rounding_mode(2);
    uint8 result = convert_uint8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint8 _cl_overloadable
  convert_uint8_rtn(float8 a)
  {
    cl_set_rounding_mode(3);
    uint8 result = convert_uint8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint8 _cl_overloadable
  convert_uint8_sat_rtn(float8 a)
  {
    cl_set_rounding_mode(3);
    uint8 result = convert_uint8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint16 _cl_overloadable
  convert_uint16_rte(float16 a)
  {
    cl_set_rounding_mode(1);
    uint16 result = convert_uint16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint16 _cl_overloadable
  convert_uint16_sat_rte(float16 a)
  {
    cl_set_rounding_mode(1);
    uint16 result = convert_uint16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint16 _cl_overloadable
  convert_uint16_rtz(float16 a)
  {
    cl_set_rounding_mode(0);
    uint16 result = convert_uint16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint16 _cl_overloadable
  convert_uint16_sat_rtz(float16 a)
  {
    cl_set_rounding_mode(0);
    uint16 result = convert_uint16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint16 _cl_overloadable
  convert_uint16_rtp(float16 a)
  {
    cl_set_rounding_mode(2);
    uint16 result = convert_uint16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint16 _cl_overloadable
  convert_uint16_sat_rtp(float16 a)
  {
    cl_set_rounding_mode(2);
    uint16 result = convert_uint16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint16 _cl_overloadable
  convert_uint16_rtn(float16 a)
  {
    cl_set_rounding_mode(3);
    uint16 result = convert_uint16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  uint16 _cl_overloadable
  convert_uint16_sat_rtn(float16 a)
  {
    cl_set_rounding_mode(3);
    uint16 result = convert_uint16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
__IF_INT64(
  long _cl_overloadable
  convert_long_rte(float a)
  {
    cl_set_rounding_mode(1);
    long result = convert_long(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long _cl_overloadable
  convert_long_sat_rte(float a)
  {
    cl_set_rounding_mode(1);
    long result = convert_long_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long _cl_overloadable
  convert_long_rtz(float a)
  {
    cl_set_rounding_mode(0);
    long result = convert_long(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long _cl_overloadable
  convert_long_sat_rtz(float a)
  {
    cl_set_rounding_mode(0);
    long result = convert_long_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long _cl_overloadable
  convert_long_rtp(float a)
  {
    cl_set_rounding_mode(2);
    long result = convert_long(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long _cl_overloadable
  convert_long_sat_rtp(float a)
  {
    cl_set_rounding_mode(2);
    long result = convert_long_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long _cl_overloadable
  convert_long_rtn(float a)
  {
    cl_set_rounding_mode(3);
    long result = convert_long(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long _cl_overloadable
  convert_long_sat_rtn(float a)
  {
    cl_set_rounding_mode(3);
    long result = convert_long_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long2 _cl_overloadable
  convert_long2_rte(float2 a)
  {
    cl_set_rounding_mode(1);
    long2 result = convert_long2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long2 _cl_overloadable
  convert_long2_sat_rte(float2 a)
  {
    cl_set_rounding_mode(1);
    long2 result = convert_long2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long2 _cl_overloadable
  convert_long2_rtz(float2 a)
  {
    cl_set_rounding_mode(0);
    long2 result = convert_long2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long2 _cl_overloadable
  convert_long2_sat_rtz(float2 a)
  {
    cl_set_rounding_mode(0);
    long2 result = convert_long2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long2 _cl_overloadable
  convert_long2_rtp(float2 a)
  {
    cl_set_rounding_mode(2);
    long2 result = convert_long2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long2 _cl_overloadable
  convert_long2_sat_rtp(float2 a)
  {
    cl_set_rounding_mode(2);
    long2 result = convert_long2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long2 _cl_overloadable
  convert_long2_rtn(float2 a)
  {
    cl_set_rounding_mode(3);
    long2 result = convert_long2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long2 _cl_overloadable
  convert_long2_sat_rtn(float2 a)
  {
    cl_set_rounding_mode(3);
    long2 result = convert_long2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long4 _cl_overloadable
  convert_long4_rte(float4 a)
  {
    cl_set_rounding_mode(1);
    long4 result = convert_long4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long4 _cl_overloadable
  convert_long4_sat_rte(float4 a)
  {
    cl_set_rounding_mode(1);
    long4 result = convert_long4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long4 _cl_overloadable
  convert_long4_rtz(float4 a)
  {
    cl_set_rounding_mode(0);
    long4 result = convert_long4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long4 _cl_overloadable
  convert_long4_sat_rtz(float4 a)
  {
    cl_set_rounding_mode(0);
    long4 result = convert_long4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long4 _cl_overloadable
  convert_long4_rtp(float4 a)
  {
    cl_set_rounding_mode(2);
    long4 result = convert_long4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long4 _cl_overloadable
  convert_long4_sat_rtp(float4 a)
  {
    cl_set_rounding_mode(2);
    long4 result = convert_long4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long4 _cl_overloadable
  convert_long4_rtn(float4 a)
  {
    cl_set_rounding_mode(3);
    long4 result = convert_long4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long4 _cl_overloadable
  convert_long4_sat_rtn(float4 a)
  {
    cl_set_rounding_mode(3);
    long4 result = convert_long4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long8 _cl_overloadable
  convert_long8_rte(float8 a)
  {
    cl_set_rounding_mode(1);
    long8 result = convert_long8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long8 _cl_overloadable
  convert_long8_sat_rte(float8 a)
  {
    cl_set_rounding_mode(1);
    long8 result = convert_long8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long8 _cl_overloadable
  convert_long8_rtz(float8 a)
  {
    cl_set_rounding_mode(0);
    long8 result = convert_long8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long8 _cl_overloadable
  convert_long8_sat_rtz(float8 a)
  {
    cl_set_rounding_mode(0);
    long8 result = convert_long8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long8 _cl_overloadable
  convert_long8_rtp(float8 a)
  {
    cl_set_rounding_mode(2);
    long8 result = convert_long8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long8 _cl_overloadable
  convert_long8_sat_rtp(float8 a)
  {
    cl_set_rounding_mode(2);
    long8 result = convert_long8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long8 _cl_overloadable
  convert_long8_rtn(float8 a)
  {
    cl_set_rounding_mode(3);
    long8 result = convert_long8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long8 _cl_overloadable
  convert_long8_sat_rtn(float8 a)
  {
    cl_set_rounding_mode(3);
    long8 result = convert_long8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long16 _cl_overloadable
  convert_long16_rte(float16 a)
  {
    cl_set_rounding_mode(1);
    long16 result = convert_long16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long16 _cl_overloadable
  convert_long16_sat_rte(float16 a)
  {
    cl_set_rounding_mode(1);
    long16 result = convert_long16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long16 _cl_overloadable
  convert_long16_rtz(float16 a)
  {
    cl_set_rounding_mode(0);
    long16 result = convert_long16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long16 _cl_overloadable
  convert_long16_sat_rtz(float16 a)
  {
    cl_set_rounding_mode(0);
    long16 result = convert_long16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long16 _cl_overloadable
  convert_long16_rtp(float16 a)
  {
    cl_set_rounding_mode(2);
    long16 result = convert_long16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long16 _cl_overloadable
  convert_long16_sat_rtp(float16 a)
  {
    cl_set_rounding_mode(2);
    long16 result = convert_long16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long16 _cl_overloadable
  convert_long16_rtn(float16 a)
  {
    cl_set_rounding_mode(3);
    long16 result = convert_long16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long16 _cl_overloadable
  convert_long16_sat_rtn(float16 a)
  {
    cl_set_rounding_mode(3);
    long16 result = convert_long16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_rte(float a)
  {
    cl_set_rounding_mode(1);
    ulong result = convert_ulong(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat_rte(float a)
  {
    cl_set_rounding_mode(1);
    ulong result = convert_ulong_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_rtz(float a)
  {
    cl_set_rounding_mode(0);
    ulong result = convert_ulong(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat_rtz(float a)
  {
    cl_set_rounding_mode(0);
    ulong result = convert_ulong_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_rtp(float a)
  {
    cl_set_rounding_mode(2);
    ulong result = convert_ulong(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat_rtp(float a)
  {
    cl_set_rounding_mode(2);
    ulong result = convert_ulong_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_rtn(float a)
  {
    cl_set_rounding_mode(3);
    ulong result = convert_ulong(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat_rtn(float a)
  {
    cl_set_rounding_mode(3);
    ulong result = convert_ulong_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong2 _cl_overloadable
  convert_ulong2_rte(float2 a)
  {
    cl_set_rounding_mode(1);
    ulong2 result = convert_ulong2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong2 _cl_overloadable
  convert_ulong2_sat_rte(float2 a)
  {
    cl_set_rounding_mode(1);
    ulong2 result = convert_ulong2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong2 _cl_overloadable
  convert_ulong2_rtz(float2 a)
  {
    cl_set_rounding_mode(0);
    ulong2 result = convert_ulong2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong2 _cl_overloadable
  convert_ulong2_sat_rtz(float2 a)
  {
    cl_set_rounding_mode(0);
    ulong2 result = convert_ulong2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong2 _cl_overloadable
  convert_ulong2_rtp(float2 a)
  {
    cl_set_rounding_mode(2);
    ulong2 result = convert_ulong2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong2 _cl_overloadable
  convert_ulong2_sat_rtp(float2 a)
  {
    cl_set_rounding_mode(2);
    ulong2 result = convert_ulong2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong2 _cl_overloadable
  convert_ulong2_rtn(float2 a)
  {
    cl_set_rounding_mode(3);
    ulong2 result = convert_ulong2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong2 _cl_overloadable
  convert_ulong2_sat_rtn(float2 a)
  {
    cl_set_rounding_mode(3);
    ulong2 result = convert_ulong2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong4 _cl_overloadable
  convert_ulong4_rte(float4 a)
  {
    cl_set_rounding_mode(1);
    ulong4 result = convert_ulong4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong4 _cl_overloadable
  convert_ulong4_sat_rte(float4 a)
  {
    cl_set_rounding_mode(1);
    ulong4 result = convert_ulong4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong4 _cl_overloadable
  convert_ulong4_rtz(float4 a)
  {
    cl_set_rounding_mode(0);
    ulong4 result = convert_ulong4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong4 _cl_overloadable
  convert_ulong4_sat_rtz(float4 a)
  {
    cl_set_rounding_mode(0);
    ulong4 result = convert_ulong4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong4 _cl_overloadable
  convert_ulong4_rtp(float4 a)
  {
    cl_set_rounding_mode(2);
    ulong4 result = convert_ulong4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong4 _cl_overloadable
  convert_ulong4_sat_rtp(float4 a)
  {
    cl_set_rounding_mode(2);
    ulong4 result = convert_ulong4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong4 _cl_overloadable
  convert_ulong4_rtn(float4 a)
  {
    cl_set_rounding_mode(3);
    ulong4 result = convert_ulong4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong4 _cl_overloadable
  convert_ulong4_sat_rtn(float4 a)
  {
    cl_set_rounding_mode(3);
    ulong4 result = convert_ulong4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong8 _cl_overloadable
  convert_ulong8_rte(float8 a)
  {
    cl_set_rounding_mode(1);
    ulong8 result = convert_ulong8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong8 _cl_overloadable
  convert_ulong8_sat_rte(float8 a)
  {
    cl_set_rounding_mode(1);
    ulong8 result = convert_ulong8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong8 _cl_overloadable
  convert_ulong8_rtz(float8 a)
  {
    cl_set_rounding_mode(0);
    ulong8 result = convert_ulong8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong8 _cl_overloadable
  convert_ulong8_sat_rtz(float8 a)
  {
    cl_set_rounding_mode(0);
    ulong8 result = convert_ulong8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong8 _cl_overloadable
  convert_ulong8_rtp(float8 a)
  {
    cl_set_rounding_mode(2);
    ulong8 result = convert_ulong8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong8 _cl_overloadable
  convert_ulong8_sat_rtp(float8 a)
  {
    cl_set_rounding_mode(2);
    ulong8 result = convert_ulong8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong8 _cl_overloadable
  convert_ulong8_rtn(float8 a)
  {
    cl_set_rounding_mode(3);
    ulong8 result = convert_ulong8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong8 _cl_overloadable
  convert_ulong8_sat_rtn(float8 a)
  {
    cl_set_rounding_mode(3);
    ulong8 result = convert_ulong8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong16 _cl_overloadable
  convert_ulong16_rte(float16 a)
  {
    cl_set_rounding_mode(1);
    ulong16 result = convert_ulong16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong16 _cl_overloadable
  convert_ulong16_sat_rte(float16 a)
  {
    cl_set_rounding_mode(1);
    ulong16 result = convert_ulong16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong16 _cl_overloadable
  convert_ulong16_rtz(float16 a)
  {
    cl_set_rounding_mode(0);
    ulong16 result = convert_ulong16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong16 _cl_overloadable
  convert_ulong16_sat_rtz(float16 a)
  {
    cl_set_rounding_mode(0);
    ulong16 result = convert_ulong16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong16 _cl_overloadable
  convert_ulong16_rtp(float16 a)
  {
    cl_set_rounding_mode(2);
    ulong16 result = convert_ulong16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong16 _cl_overloadable
  convert_ulong16_sat_rtp(float16 a)
  {
    cl_set_rounding_mode(2);
    ulong16 result = convert_ulong16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong16 _cl_overloadable
  convert_ulong16_rtn(float16 a)
  {
    cl_set_rounding_mode(3);
    ulong16 result = convert_ulong16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong16 _cl_overloadable
  convert_ulong16_sat_rtn(float16 a)
  {
    cl_set_rounding_mode(3);
    ulong16 result = convert_ulong16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char _cl_overloadable
  convert_char_rte(double a)
  {
    cl_set_rounding_mode(1);
    char result = convert_char(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char _cl_overloadable
  convert_char_sat_rte(double a)
  {
    cl_set_rounding_mode(1);
    char result = convert_char_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char _cl_overloadable
  convert_char_rtz(double a)
  {
    cl_set_rounding_mode(0);
    char result = convert_char(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char _cl_overloadable
  convert_char_sat_rtz(double a)
  {
    cl_set_rounding_mode(0);
    char result = convert_char_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char _cl_overloadable
  convert_char_rtp(double a)
  {
    cl_set_rounding_mode(2);
    char result = convert_char(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char _cl_overloadable
  convert_char_sat_rtp(double a)
  {
    cl_set_rounding_mode(2);
    char result = convert_char_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char _cl_overloadable
  convert_char_rtn(double a)
  {
    cl_set_rounding_mode(3);
    char result = convert_char(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char _cl_overloadable
  convert_char_sat_rtn(double a)
  {
    cl_set_rounding_mode(3);
    char result = convert_char_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char2 _cl_overloadable
  convert_char2_rte(double2 a)
  {
    cl_set_rounding_mode(1);
    char2 result = convert_char2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char2 _cl_overloadable
  convert_char2_sat_rte(double2 a)
  {
    cl_set_rounding_mode(1);
    char2 result = convert_char2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char2 _cl_overloadable
  convert_char2_rtz(double2 a)
  {
    cl_set_rounding_mode(0);
    char2 result = convert_char2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char2 _cl_overloadable
  convert_char2_sat_rtz(double2 a)
  {
    cl_set_rounding_mode(0);
    char2 result = convert_char2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char2 _cl_overloadable
  convert_char2_rtp(double2 a)
  {
    cl_set_rounding_mode(2);
    char2 result = convert_char2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char2 _cl_overloadable
  convert_char2_sat_rtp(double2 a)
  {
    cl_set_rounding_mode(2);
    char2 result = convert_char2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char2 _cl_overloadable
  convert_char2_rtn(double2 a)
  {
    cl_set_rounding_mode(3);
    char2 result = convert_char2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char2 _cl_overloadable
  convert_char2_sat_rtn(double2 a)
  {
    cl_set_rounding_mode(3);
    char2 result = convert_char2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char4 _cl_overloadable
  convert_char4_rte(double4 a)
  {
    cl_set_rounding_mode(1);
    char4 result = convert_char4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char4 _cl_overloadable
  convert_char4_sat_rte(double4 a)
  {
    cl_set_rounding_mode(1);
    char4 result = convert_char4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char4 _cl_overloadable
  convert_char4_rtz(double4 a)
  {
    cl_set_rounding_mode(0);
    char4 result = convert_char4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char4 _cl_overloadable
  convert_char4_sat_rtz(double4 a)
  {
    cl_set_rounding_mode(0);
    char4 result = convert_char4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char4 _cl_overloadable
  convert_char4_rtp(double4 a)
  {
    cl_set_rounding_mode(2);
    char4 result = convert_char4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char4 _cl_overloadable
  convert_char4_sat_rtp(double4 a)
  {
    cl_set_rounding_mode(2);
    char4 result = convert_char4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char4 _cl_overloadable
  convert_char4_rtn(double4 a)
  {
    cl_set_rounding_mode(3);
    char4 result = convert_char4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char4 _cl_overloadable
  convert_char4_sat_rtn(double4 a)
  {
    cl_set_rounding_mode(3);
    char4 result = convert_char4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char8 _cl_overloadable
  convert_char8_rte(double8 a)
  {
    cl_set_rounding_mode(1);
    char8 result = convert_char8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char8 _cl_overloadable
  convert_char8_sat_rte(double8 a)
  {
    cl_set_rounding_mode(1);
    char8 result = convert_char8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char8 _cl_overloadable
  convert_char8_rtz(double8 a)
  {
    cl_set_rounding_mode(0);
    char8 result = convert_char8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char8 _cl_overloadable
  convert_char8_sat_rtz(double8 a)
  {
    cl_set_rounding_mode(0);
    char8 result = convert_char8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char8 _cl_overloadable
  convert_char8_rtp(double8 a)
  {
    cl_set_rounding_mode(2);
    char8 result = convert_char8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char8 _cl_overloadable
  convert_char8_sat_rtp(double8 a)
  {
    cl_set_rounding_mode(2);
    char8 result = convert_char8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char8 _cl_overloadable
  convert_char8_rtn(double8 a)
  {
    cl_set_rounding_mode(3);
    char8 result = convert_char8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char8 _cl_overloadable
  convert_char8_sat_rtn(double8 a)
  {
    cl_set_rounding_mode(3);
    char8 result = convert_char8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char16 _cl_overloadable
  convert_char16_rte(double16 a)
  {
    cl_set_rounding_mode(1);
    char16 result = convert_char16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char16 _cl_overloadable
  convert_char16_sat_rte(double16 a)
  {
    cl_set_rounding_mode(1);
    char16 result = convert_char16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char16 _cl_overloadable
  convert_char16_rtz(double16 a)
  {
    cl_set_rounding_mode(0);
    char16 result = convert_char16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char16 _cl_overloadable
  convert_char16_sat_rtz(double16 a)
  {
    cl_set_rounding_mode(0);
    char16 result = convert_char16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char16 _cl_overloadable
  convert_char16_rtp(double16 a)
  {
    cl_set_rounding_mode(2);
    char16 result = convert_char16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char16 _cl_overloadable
  convert_char16_sat_rtp(double16 a)
  {
    cl_set_rounding_mode(2);
    char16 result = convert_char16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char16 _cl_overloadable
  convert_char16_rtn(double16 a)
  {
    cl_set_rounding_mode(3);
    char16 result = convert_char16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  char16 _cl_overloadable
  convert_char16_sat_rtn(double16 a)
  {
    cl_set_rounding_mode(3);
    char16 result = convert_char16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar _cl_overloadable
  convert_uchar_rte(double a)
  {
    cl_set_rounding_mode(1);
    uchar result = convert_uchar(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar _cl_overloadable
  convert_uchar_sat_rte(double a)
  {
    cl_set_rounding_mode(1);
    uchar result = convert_uchar_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar _cl_overloadable
  convert_uchar_rtz(double a)
  {
    cl_set_rounding_mode(0);
    uchar result = convert_uchar(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar _cl_overloadable
  convert_uchar_sat_rtz(double a)
  {
    cl_set_rounding_mode(0);
    uchar result = convert_uchar_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar _cl_overloadable
  convert_uchar_rtp(double a)
  {
    cl_set_rounding_mode(2);
    uchar result = convert_uchar(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar _cl_overloadable
  convert_uchar_sat_rtp(double a)
  {
    cl_set_rounding_mode(2);
    uchar result = convert_uchar_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar _cl_overloadable
  convert_uchar_rtn(double a)
  {
    cl_set_rounding_mode(3);
    uchar result = convert_uchar(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar _cl_overloadable
  convert_uchar_sat_rtn(double a)
  {
    cl_set_rounding_mode(3);
    uchar result = convert_uchar_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar2 _cl_overloadable
  convert_uchar2_rte(double2 a)
  {
    cl_set_rounding_mode(1);
    uchar2 result = convert_uchar2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar2 _cl_overloadable
  convert_uchar2_sat_rte(double2 a)
  {
    cl_set_rounding_mode(1);
    uchar2 result = convert_uchar2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar2 _cl_overloadable
  convert_uchar2_rtz(double2 a)
  {
    cl_set_rounding_mode(0);
    uchar2 result = convert_uchar2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar2 _cl_overloadable
  convert_uchar2_sat_rtz(double2 a)
  {
    cl_set_rounding_mode(0);
    uchar2 result = convert_uchar2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar2 _cl_overloadable
  convert_uchar2_rtp(double2 a)
  {
    cl_set_rounding_mode(2);
    uchar2 result = convert_uchar2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar2 _cl_overloadable
  convert_uchar2_sat_rtp(double2 a)
  {
    cl_set_rounding_mode(2);
    uchar2 result = convert_uchar2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar2 _cl_overloadable
  convert_uchar2_rtn(double2 a)
  {
    cl_set_rounding_mode(3);
    uchar2 result = convert_uchar2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar2 _cl_overloadable
  convert_uchar2_sat_rtn(double2 a)
  {
    cl_set_rounding_mode(3);
    uchar2 result = convert_uchar2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar4 _cl_overloadable
  convert_uchar4_rte(double4 a)
  {
    cl_set_rounding_mode(1);
    uchar4 result = convert_uchar4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar4 _cl_overloadable
  convert_uchar4_sat_rte(double4 a)
  {
    cl_set_rounding_mode(1);
    uchar4 result = convert_uchar4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar4 _cl_overloadable
  convert_uchar4_rtz(double4 a)
  {
    cl_set_rounding_mode(0);
    uchar4 result = convert_uchar4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar4 _cl_overloadable
  convert_uchar4_sat_rtz(double4 a)
  {
    cl_set_rounding_mode(0);
    uchar4 result = convert_uchar4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar4 _cl_overloadable
  convert_uchar4_rtp(double4 a)
  {
    cl_set_rounding_mode(2);
    uchar4 result = convert_uchar4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar4 _cl_overloadable
  convert_uchar4_sat_rtp(double4 a)
  {
    cl_set_rounding_mode(2);
    uchar4 result = convert_uchar4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar4 _cl_overloadable
  convert_uchar4_rtn(double4 a)
  {
    cl_set_rounding_mode(3);
    uchar4 result = convert_uchar4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar4 _cl_overloadable
  convert_uchar4_sat_rtn(double4 a)
  {
    cl_set_rounding_mode(3);
    uchar4 result = convert_uchar4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar8 _cl_overloadable
  convert_uchar8_rte(double8 a)
  {
    cl_set_rounding_mode(1);
    uchar8 result = convert_uchar8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar8 _cl_overloadable
  convert_uchar8_sat_rte(double8 a)
  {
    cl_set_rounding_mode(1);
    uchar8 result = convert_uchar8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar8 _cl_overloadable
  convert_uchar8_rtz(double8 a)
  {
    cl_set_rounding_mode(0);
    uchar8 result = convert_uchar8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar8 _cl_overloadable
  convert_uchar8_sat_rtz(double8 a)
  {
    cl_set_rounding_mode(0);
    uchar8 result = convert_uchar8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar8 _cl_overloadable
  convert_uchar8_rtp(double8 a)
  {
    cl_set_rounding_mode(2);
    uchar8 result = convert_uchar8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar8 _cl_overloadable
  convert_uchar8_sat_rtp(double8 a)
  {
    cl_set_rounding_mode(2);
    uchar8 result = convert_uchar8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar8 _cl_overloadable
  convert_uchar8_rtn(double8 a)
  {
    cl_set_rounding_mode(3);
    uchar8 result = convert_uchar8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar8 _cl_overloadable
  convert_uchar8_sat_rtn(double8 a)
  {
    cl_set_rounding_mode(3);
    uchar8 result = convert_uchar8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar16 _cl_overloadable
  convert_uchar16_rte(double16 a)
  {
    cl_set_rounding_mode(1);
    uchar16 result = convert_uchar16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar16 _cl_overloadable
  convert_uchar16_sat_rte(double16 a)
  {
    cl_set_rounding_mode(1);
    uchar16 result = convert_uchar16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar16 _cl_overloadable
  convert_uchar16_rtz(double16 a)
  {
    cl_set_rounding_mode(0);
    uchar16 result = convert_uchar16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar16 _cl_overloadable
  convert_uchar16_sat_rtz(double16 a)
  {
    cl_set_rounding_mode(0);
    uchar16 result = convert_uchar16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar16 _cl_overloadable
  convert_uchar16_rtp(double16 a)
  {
    cl_set_rounding_mode(2);
    uchar16 result = convert_uchar16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar16 _cl_overloadable
  convert_uchar16_sat_rtp(double16 a)
  {
    cl_set_rounding_mode(2);
    uchar16 result = convert_uchar16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar16 _cl_overloadable
  convert_uchar16_rtn(double16 a)
  {
    cl_set_rounding_mode(3);
    uchar16 result = convert_uchar16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uchar16 _cl_overloadable
  convert_uchar16_sat_rtn(double16 a)
  {
    cl_set_rounding_mode(3);
    uchar16 result = convert_uchar16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short _cl_overloadable
  convert_short_rte(double a)
  {
    cl_set_rounding_mode(1);
    short result = convert_short(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short _cl_overloadable
  convert_short_sat_rte(double a)
  {
    cl_set_rounding_mode(1);
    short result = convert_short_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short _cl_overloadable
  convert_short_rtz(double a)
  {
    cl_set_rounding_mode(0);
    short result = convert_short(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short _cl_overloadable
  convert_short_sat_rtz(double a)
  {
    cl_set_rounding_mode(0);
    short result = convert_short_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short _cl_overloadable
  convert_short_rtp(double a)
  {
    cl_set_rounding_mode(2);
    short result = convert_short(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short _cl_overloadable
  convert_short_sat_rtp(double a)
  {
    cl_set_rounding_mode(2);
    short result = convert_short_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short _cl_overloadable
  convert_short_rtn(double a)
  {
    cl_set_rounding_mode(3);
    short result = convert_short(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short _cl_overloadable
  convert_short_sat_rtn(double a)
  {
    cl_set_rounding_mode(3);
    short result = convert_short_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short2 _cl_overloadable
  convert_short2_rte(double2 a)
  {
    cl_set_rounding_mode(1);
    short2 result = convert_short2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short2 _cl_overloadable
  convert_short2_sat_rte(double2 a)
  {
    cl_set_rounding_mode(1);
    short2 result = convert_short2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short2 _cl_overloadable
  convert_short2_rtz(double2 a)
  {
    cl_set_rounding_mode(0);
    short2 result = convert_short2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short2 _cl_overloadable
  convert_short2_sat_rtz(double2 a)
  {
    cl_set_rounding_mode(0);
    short2 result = convert_short2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short2 _cl_overloadable
  convert_short2_rtp(double2 a)
  {
    cl_set_rounding_mode(2);
    short2 result = convert_short2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short2 _cl_overloadable
  convert_short2_sat_rtp(double2 a)
  {
    cl_set_rounding_mode(2);
    short2 result = convert_short2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short2 _cl_overloadable
  convert_short2_rtn(double2 a)
  {
    cl_set_rounding_mode(3);
    short2 result = convert_short2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short2 _cl_overloadable
  convert_short2_sat_rtn(double2 a)
  {
    cl_set_rounding_mode(3);
    short2 result = convert_short2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short4 _cl_overloadable
  convert_short4_rte(double4 a)
  {
    cl_set_rounding_mode(1);
    short4 result = convert_short4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short4 _cl_overloadable
  convert_short4_sat_rte(double4 a)
  {
    cl_set_rounding_mode(1);
    short4 result = convert_short4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short4 _cl_overloadable
  convert_short4_rtz(double4 a)
  {
    cl_set_rounding_mode(0);
    short4 result = convert_short4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short4 _cl_overloadable
  convert_short4_sat_rtz(double4 a)
  {
    cl_set_rounding_mode(0);
    short4 result = convert_short4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short4 _cl_overloadable
  convert_short4_rtp(double4 a)
  {
    cl_set_rounding_mode(2);
    short4 result = convert_short4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short4 _cl_overloadable
  convert_short4_sat_rtp(double4 a)
  {
    cl_set_rounding_mode(2);
    short4 result = convert_short4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short4 _cl_overloadable
  convert_short4_rtn(double4 a)
  {
    cl_set_rounding_mode(3);
    short4 result = convert_short4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short4 _cl_overloadable
  convert_short4_sat_rtn(double4 a)
  {
    cl_set_rounding_mode(3);
    short4 result = convert_short4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short8 _cl_overloadable
  convert_short8_rte(double8 a)
  {
    cl_set_rounding_mode(1);
    short8 result = convert_short8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short8 _cl_overloadable
  convert_short8_sat_rte(double8 a)
  {
    cl_set_rounding_mode(1);
    short8 result = convert_short8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short8 _cl_overloadable
  convert_short8_rtz(double8 a)
  {
    cl_set_rounding_mode(0);
    short8 result = convert_short8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short8 _cl_overloadable
  convert_short8_sat_rtz(double8 a)
  {
    cl_set_rounding_mode(0);
    short8 result = convert_short8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short8 _cl_overloadable
  convert_short8_rtp(double8 a)
  {
    cl_set_rounding_mode(2);
    short8 result = convert_short8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short8 _cl_overloadable
  convert_short8_sat_rtp(double8 a)
  {
    cl_set_rounding_mode(2);
    short8 result = convert_short8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short8 _cl_overloadable
  convert_short8_rtn(double8 a)
  {
    cl_set_rounding_mode(3);
    short8 result = convert_short8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short8 _cl_overloadable
  convert_short8_sat_rtn(double8 a)
  {
    cl_set_rounding_mode(3);
    short8 result = convert_short8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short16 _cl_overloadable
  convert_short16_rte(double16 a)
  {
    cl_set_rounding_mode(1);
    short16 result = convert_short16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short16 _cl_overloadable
  convert_short16_sat_rte(double16 a)
  {
    cl_set_rounding_mode(1);
    short16 result = convert_short16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short16 _cl_overloadable
  convert_short16_rtz(double16 a)
  {
    cl_set_rounding_mode(0);
    short16 result = convert_short16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short16 _cl_overloadable
  convert_short16_sat_rtz(double16 a)
  {
    cl_set_rounding_mode(0);
    short16 result = convert_short16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short16 _cl_overloadable
  convert_short16_rtp(double16 a)
  {
    cl_set_rounding_mode(2);
    short16 result = convert_short16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short16 _cl_overloadable
  convert_short16_sat_rtp(double16 a)
  {
    cl_set_rounding_mode(2);
    short16 result = convert_short16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short16 _cl_overloadable
  convert_short16_rtn(double16 a)
  {
    cl_set_rounding_mode(3);
    short16 result = convert_short16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  short16 _cl_overloadable
  convert_short16_sat_rtn(double16 a)
  {
    cl_set_rounding_mode(3);
    short16 result = convert_short16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort _cl_overloadable
  convert_ushort_rte(double a)
  {
    cl_set_rounding_mode(1);
    ushort result = convert_ushort(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort _cl_overloadable
  convert_ushort_sat_rte(double a)
  {
    cl_set_rounding_mode(1);
    ushort result = convert_ushort_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort _cl_overloadable
  convert_ushort_rtz(double a)
  {
    cl_set_rounding_mode(0);
    ushort result = convert_ushort(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort _cl_overloadable
  convert_ushort_sat_rtz(double a)
  {
    cl_set_rounding_mode(0);
    ushort result = convert_ushort_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort _cl_overloadable
  convert_ushort_rtp(double a)
  {
    cl_set_rounding_mode(2);
    ushort result = convert_ushort(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort _cl_overloadable
  convert_ushort_sat_rtp(double a)
  {
    cl_set_rounding_mode(2);
    ushort result = convert_ushort_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort _cl_overloadable
  convert_ushort_rtn(double a)
  {
    cl_set_rounding_mode(3);
    ushort result = convert_ushort(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort _cl_overloadable
  convert_ushort_sat_rtn(double a)
  {
    cl_set_rounding_mode(3);
    ushort result = convert_ushort_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort2 _cl_overloadable
  convert_ushort2_rte(double2 a)
  {
    cl_set_rounding_mode(1);
    ushort2 result = convert_ushort2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort2 _cl_overloadable
  convert_ushort2_sat_rte(double2 a)
  {
    cl_set_rounding_mode(1);
    ushort2 result = convert_ushort2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort2 _cl_overloadable
  convert_ushort2_rtz(double2 a)
  {
    cl_set_rounding_mode(0);
    ushort2 result = convert_ushort2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort2 _cl_overloadable
  convert_ushort2_sat_rtz(double2 a)
  {
    cl_set_rounding_mode(0);
    ushort2 result = convert_ushort2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort2 _cl_overloadable
  convert_ushort2_rtp(double2 a)
  {
    cl_set_rounding_mode(2);
    ushort2 result = convert_ushort2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort2 _cl_overloadable
  convert_ushort2_sat_rtp(double2 a)
  {
    cl_set_rounding_mode(2);
    ushort2 result = convert_ushort2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort2 _cl_overloadable
  convert_ushort2_rtn(double2 a)
  {
    cl_set_rounding_mode(3);
    ushort2 result = convert_ushort2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort2 _cl_overloadable
  convert_ushort2_sat_rtn(double2 a)
  {
    cl_set_rounding_mode(3);
    ushort2 result = convert_ushort2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort4 _cl_overloadable
  convert_ushort4_rte(double4 a)
  {
    cl_set_rounding_mode(1);
    ushort4 result = convert_ushort4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort4 _cl_overloadable
  convert_ushort4_sat_rte(double4 a)
  {
    cl_set_rounding_mode(1);
    ushort4 result = convert_ushort4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort4 _cl_overloadable
  convert_ushort4_rtz(double4 a)
  {
    cl_set_rounding_mode(0);
    ushort4 result = convert_ushort4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort4 _cl_overloadable
  convert_ushort4_sat_rtz(double4 a)
  {
    cl_set_rounding_mode(0);
    ushort4 result = convert_ushort4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort4 _cl_overloadable
  convert_ushort4_rtp(double4 a)
  {
    cl_set_rounding_mode(2);
    ushort4 result = convert_ushort4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort4 _cl_overloadable
  convert_ushort4_sat_rtp(double4 a)
  {
    cl_set_rounding_mode(2);
    ushort4 result = convert_ushort4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort4 _cl_overloadable
  convert_ushort4_rtn(double4 a)
  {
    cl_set_rounding_mode(3);
    ushort4 result = convert_ushort4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort4 _cl_overloadable
  convert_ushort4_sat_rtn(double4 a)
  {
    cl_set_rounding_mode(3);
    ushort4 result = convert_ushort4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort8 _cl_overloadable
  convert_ushort8_rte(double8 a)
  {
    cl_set_rounding_mode(1);
    ushort8 result = convert_ushort8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort8 _cl_overloadable
  convert_ushort8_sat_rte(double8 a)
  {
    cl_set_rounding_mode(1);
    ushort8 result = convert_ushort8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort8 _cl_overloadable
  convert_ushort8_rtz(double8 a)
  {
    cl_set_rounding_mode(0);
    ushort8 result = convert_ushort8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort8 _cl_overloadable
  convert_ushort8_sat_rtz(double8 a)
  {
    cl_set_rounding_mode(0);
    ushort8 result = convert_ushort8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort8 _cl_overloadable
  convert_ushort8_rtp(double8 a)
  {
    cl_set_rounding_mode(2);
    ushort8 result = convert_ushort8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort8 _cl_overloadable
  convert_ushort8_sat_rtp(double8 a)
  {
    cl_set_rounding_mode(2);
    ushort8 result = convert_ushort8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort8 _cl_overloadable
  convert_ushort8_rtn(double8 a)
  {
    cl_set_rounding_mode(3);
    ushort8 result = convert_ushort8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort8 _cl_overloadable
  convert_ushort8_sat_rtn(double8 a)
  {
    cl_set_rounding_mode(3);
    ushort8 result = convert_ushort8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort16 _cl_overloadable
  convert_ushort16_rte(double16 a)
  {
    cl_set_rounding_mode(1);
    ushort16 result = convert_ushort16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort16 _cl_overloadable
  convert_ushort16_sat_rte(double16 a)
  {
    cl_set_rounding_mode(1);
    ushort16 result = convert_ushort16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort16 _cl_overloadable
  convert_ushort16_rtz(double16 a)
  {
    cl_set_rounding_mode(0);
    ushort16 result = convert_ushort16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort16 _cl_overloadable
  convert_ushort16_sat_rtz(double16 a)
  {
    cl_set_rounding_mode(0);
    ushort16 result = convert_ushort16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort16 _cl_overloadable
  convert_ushort16_rtp(double16 a)
  {
    cl_set_rounding_mode(2);
    ushort16 result = convert_ushort16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort16 _cl_overloadable
  convert_ushort16_sat_rtp(double16 a)
  {
    cl_set_rounding_mode(2);
    ushort16 result = convert_ushort16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort16 _cl_overloadable
  convert_ushort16_rtn(double16 a)
  {
    cl_set_rounding_mode(3);
    ushort16 result = convert_ushort16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  ushort16 _cl_overloadable
  convert_ushort16_sat_rtn(double16 a)
  {
    cl_set_rounding_mode(3);
    ushort16 result = convert_ushort16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int _cl_overloadable
  convert_int_rte(double a)
  {
    cl_set_rounding_mode(1);
    int result = convert_int(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int _cl_overloadable
  convert_int_sat_rte(double a)
  {
    cl_set_rounding_mode(1);
    int result = convert_int_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int _cl_overloadable
  convert_int_rtz(double a)
  {
    cl_set_rounding_mode(0);
    int result = convert_int(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int _cl_overloadable
  convert_int_sat_rtz(double a)
  {
    cl_set_rounding_mode(0);
    int result = convert_int_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int _cl_overloadable
  convert_int_rtp(double a)
  {
    cl_set_rounding_mode(2);
    int result = convert_int(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int _cl_overloadable
  convert_int_sat_rtp(double a)
  {
    cl_set_rounding_mode(2);
    int result = convert_int_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int _cl_overloadable
  convert_int_rtn(double a)
  {
    cl_set_rounding_mode(3);
    int result = convert_int(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int _cl_overloadable
  convert_int_sat_rtn(double a)
  {
    cl_set_rounding_mode(3);
    int result = convert_int_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int2 _cl_overloadable
  convert_int2_rte(double2 a)
  {
    cl_set_rounding_mode(1);
    int2 result = convert_int2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int2 _cl_overloadable
  convert_int2_sat_rte(double2 a)
  {
    cl_set_rounding_mode(1);
    int2 result = convert_int2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int2 _cl_overloadable
  convert_int2_rtz(double2 a)
  {
    cl_set_rounding_mode(0);
    int2 result = convert_int2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int2 _cl_overloadable
  convert_int2_sat_rtz(double2 a)
  {
    cl_set_rounding_mode(0);
    int2 result = convert_int2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int2 _cl_overloadable
  convert_int2_rtp(double2 a)
  {
    cl_set_rounding_mode(2);
    int2 result = convert_int2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int2 _cl_overloadable
  convert_int2_sat_rtp(double2 a)
  {
    cl_set_rounding_mode(2);
    int2 result = convert_int2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int2 _cl_overloadable
  convert_int2_rtn(double2 a)
  {
    cl_set_rounding_mode(3);
    int2 result = convert_int2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int2 _cl_overloadable
  convert_int2_sat_rtn(double2 a)
  {
    cl_set_rounding_mode(3);
    int2 result = convert_int2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int4 _cl_overloadable
  convert_int4_rte(double4 a)
  {
    cl_set_rounding_mode(1);
    int4 result = convert_int4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int4 _cl_overloadable
  convert_int4_sat_rte(double4 a)
  {
    cl_set_rounding_mode(1);
    int4 result = convert_int4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int4 _cl_overloadable
  convert_int4_rtz(double4 a)
  {
    cl_set_rounding_mode(0);
    int4 result = convert_int4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int4 _cl_overloadable
  convert_int4_sat_rtz(double4 a)
  {
    cl_set_rounding_mode(0);
    int4 result = convert_int4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int4 _cl_overloadable
  convert_int4_rtp(double4 a)
  {
    cl_set_rounding_mode(2);
    int4 result = convert_int4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int4 _cl_overloadable
  convert_int4_sat_rtp(double4 a)
  {
    cl_set_rounding_mode(2);
    int4 result = convert_int4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int4 _cl_overloadable
  convert_int4_rtn(double4 a)
  {
    cl_set_rounding_mode(3);
    int4 result = convert_int4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int4 _cl_overloadable
  convert_int4_sat_rtn(double4 a)
  {
    cl_set_rounding_mode(3);
    int4 result = convert_int4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int8 _cl_overloadable
  convert_int8_rte(double8 a)
  {
    cl_set_rounding_mode(1);
    int8 result = convert_int8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int8 _cl_overloadable
  convert_int8_sat_rte(double8 a)
  {
    cl_set_rounding_mode(1);
    int8 result = convert_int8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int8 _cl_overloadable
  convert_int8_rtz(double8 a)
  {
    cl_set_rounding_mode(0);
    int8 result = convert_int8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int8 _cl_overloadable
  convert_int8_sat_rtz(double8 a)
  {
    cl_set_rounding_mode(0);
    int8 result = convert_int8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int8 _cl_overloadable
  convert_int8_rtp(double8 a)
  {
    cl_set_rounding_mode(2);
    int8 result = convert_int8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int8 _cl_overloadable
  convert_int8_sat_rtp(double8 a)
  {
    cl_set_rounding_mode(2);
    int8 result = convert_int8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int8 _cl_overloadable
  convert_int8_rtn(double8 a)
  {
    cl_set_rounding_mode(3);
    int8 result = convert_int8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int8 _cl_overloadable
  convert_int8_sat_rtn(double8 a)
  {
    cl_set_rounding_mode(3);
    int8 result = convert_int8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int16 _cl_overloadable
  convert_int16_rte(double16 a)
  {
    cl_set_rounding_mode(1);
    int16 result = convert_int16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int16 _cl_overloadable
  convert_int16_sat_rte(double16 a)
  {
    cl_set_rounding_mode(1);
    int16 result = convert_int16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int16 _cl_overloadable
  convert_int16_rtz(double16 a)
  {
    cl_set_rounding_mode(0);
    int16 result = convert_int16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int16 _cl_overloadable
  convert_int16_sat_rtz(double16 a)
  {
    cl_set_rounding_mode(0);
    int16 result = convert_int16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int16 _cl_overloadable
  convert_int16_rtp(double16 a)
  {
    cl_set_rounding_mode(2);
    int16 result = convert_int16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int16 _cl_overloadable
  convert_int16_sat_rtp(double16 a)
  {
    cl_set_rounding_mode(2);
    int16 result = convert_int16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int16 _cl_overloadable
  convert_int16_rtn(double16 a)
  {
    cl_set_rounding_mode(3);
    int16 result = convert_int16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  int16 _cl_overloadable
  convert_int16_sat_rtn(double16 a)
  {
    cl_set_rounding_mode(3);
    int16 result = convert_int16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint _cl_overloadable
  convert_uint_rte(double a)
  {
    cl_set_rounding_mode(1);
    uint result = convert_uint(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint _cl_overloadable
  convert_uint_sat_rte(double a)
  {
    cl_set_rounding_mode(1);
    uint result = convert_uint_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint _cl_overloadable
  convert_uint_rtz(double a)
  {
    cl_set_rounding_mode(0);
    uint result = convert_uint(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint _cl_overloadable
  convert_uint_sat_rtz(double a)
  {
    cl_set_rounding_mode(0);
    uint result = convert_uint_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint _cl_overloadable
  convert_uint_rtp(double a)
  {
    cl_set_rounding_mode(2);
    uint result = convert_uint(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint _cl_overloadable
  convert_uint_sat_rtp(double a)
  {
    cl_set_rounding_mode(2);
    uint result = convert_uint_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint _cl_overloadable
  convert_uint_rtn(double a)
  {
    cl_set_rounding_mode(3);
    uint result = convert_uint(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint _cl_overloadable
  convert_uint_sat_rtn(double a)
  {
    cl_set_rounding_mode(3);
    uint result = convert_uint_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint2 _cl_overloadable
  convert_uint2_rte(double2 a)
  {
    cl_set_rounding_mode(1);
    uint2 result = convert_uint2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint2 _cl_overloadable
  convert_uint2_sat_rte(double2 a)
  {
    cl_set_rounding_mode(1);
    uint2 result = convert_uint2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint2 _cl_overloadable
  convert_uint2_rtz(double2 a)
  {
    cl_set_rounding_mode(0);
    uint2 result = convert_uint2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint2 _cl_overloadable
  convert_uint2_sat_rtz(double2 a)
  {
    cl_set_rounding_mode(0);
    uint2 result = convert_uint2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint2 _cl_overloadable
  convert_uint2_rtp(double2 a)
  {
    cl_set_rounding_mode(2);
    uint2 result = convert_uint2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint2 _cl_overloadable
  convert_uint2_sat_rtp(double2 a)
  {
    cl_set_rounding_mode(2);
    uint2 result = convert_uint2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint2 _cl_overloadable
  convert_uint2_rtn(double2 a)
  {
    cl_set_rounding_mode(3);
    uint2 result = convert_uint2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint2 _cl_overloadable
  convert_uint2_sat_rtn(double2 a)
  {
    cl_set_rounding_mode(3);
    uint2 result = convert_uint2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint4 _cl_overloadable
  convert_uint4_rte(double4 a)
  {
    cl_set_rounding_mode(1);
    uint4 result = convert_uint4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint4 _cl_overloadable
  convert_uint4_sat_rte(double4 a)
  {
    cl_set_rounding_mode(1);
    uint4 result = convert_uint4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint4 _cl_overloadable
  convert_uint4_rtz(double4 a)
  {
    cl_set_rounding_mode(0);
    uint4 result = convert_uint4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint4 _cl_overloadable
  convert_uint4_sat_rtz(double4 a)
  {
    cl_set_rounding_mode(0);
    uint4 result = convert_uint4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint4 _cl_overloadable
  convert_uint4_rtp(double4 a)
  {
    cl_set_rounding_mode(2);
    uint4 result = convert_uint4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint4 _cl_overloadable
  convert_uint4_sat_rtp(double4 a)
  {
    cl_set_rounding_mode(2);
    uint4 result = convert_uint4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint4 _cl_overloadable
  convert_uint4_rtn(double4 a)
  {
    cl_set_rounding_mode(3);
    uint4 result = convert_uint4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint4 _cl_overloadable
  convert_uint4_sat_rtn(double4 a)
  {
    cl_set_rounding_mode(3);
    uint4 result = convert_uint4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint8 _cl_overloadable
  convert_uint8_rte(double8 a)
  {
    cl_set_rounding_mode(1);
    uint8 result = convert_uint8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint8 _cl_overloadable
  convert_uint8_sat_rte(double8 a)
  {
    cl_set_rounding_mode(1);
    uint8 result = convert_uint8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint8 _cl_overloadable
  convert_uint8_rtz(double8 a)
  {
    cl_set_rounding_mode(0);
    uint8 result = convert_uint8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint8 _cl_overloadable
  convert_uint8_sat_rtz(double8 a)
  {
    cl_set_rounding_mode(0);
    uint8 result = convert_uint8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint8 _cl_overloadable
  convert_uint8_rtp(double8 a)
  {
    cl_set_rounding_mode(2);
    uint8 result = convert_uint8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint8 _cl_overloadable
  convert_uint8_sat_rtp(double8 a)
  {
    cl_set_rounding_mode(2);
    uint8 result = convert_uint8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint8 _cl_overloadable
  convert_uint8_rtn(double8 a)
  {
    cl_set_rounding_mode(3);
    uint8 result = convert_uint8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint8 _cl_overloadable
  convert_uint8_sat_rtn(double8 a)
  {
    cl_set_rounding_mode(3);
    uint8 result = convert_uint8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint16 _cl_overloadable
  convert_uint16_rte(double16 a)
  {
    cl_set_rounding_mode(1);
    uint16 result = convert_uint16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint16 _cl_overloadable
  convert_uint16_sat_rte(double16 a)
  {
    cl_set_rounding_mode(1);
    uint16 result = convert_uint16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint16 _cl_overloadable
  convert_uint16_rtz(double16 a)
  {
    cl_set_rounding_mode(0);
    uint16 result = convert_uint16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint16 _cl_overloadable
  convert_uint16_sat_rtz(double16 a)
  {
    cl_set_rounding_mode(0);
    uint16 result = convert_uint16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint16 _cl_overloadable
  convert_uint16_rtp(double16 a)
  {
    cl_set_rounding_mode(2);
    uint16 result = convert_uint16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint16 _cl_overloadable
  convert_uint16_sat_rtp(double16 a)
  {
    cl_set_rounding_mode(2);
    uint16 result = convert_uint16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint16 _cl_overloadable
  convert_uint16_rtn(double16 a)
  {
    cl_set_rounding_mode(3);
    uint16 result = convert_uint16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  uint16 _cl_overloadable
  convert_uint16_sat_rtn(double16 a)
  {
    cl_set_rounding_mode(3);
    uint16 result = convert_uint16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long _cl_overloadable
  convert_long_rte(double a)
  {
    cl_set_rounding_mode(1);
    long result = convert_long(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long _cl_overloadable
  convert_long_sat_rte(double a)
  {
    cl_set_rounding_mode(1);
    long result = convert_long_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long _cl_overloadable
  convert_long_rtz(double a)
  {
    cl_set_rounding_mode(0);
    long result = convert_long(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long _cl_overloadable
  convert_long_sat_rtz(double a)
  {
    cl_set_rounding_mode(0);
    long result = convert_long_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long _cl_overloadable
  convert_long_rtp(double a)
  {
    cl_set_rounding_mode(2);
    long result = convert_long(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long _cl_overloadable
  convert_long_sat_rtp(double a)
  {
    cl_set_rounding_mode(2);
    long result = convert_long_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long _cl_overloadable
  convert_long_rtn(double a)
  {
    cl_set_rounding_mode(3);
    long result = convert_long(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long _cl_overloadable
  convert_long_sat_rtn(double a)
  {
    cl_set_rounding_mode(3);
    long result = convert_long_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long2 _cl_overloadable
  convert_long2_rte(double2 a)
  {
    cl_set_rounding_mode(1);
    long2 result = convert_long2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long2 _cl_overloadable
  convert_long2_sat_rte(double2 a)
  {
    cl_set_rounding_mode(1);
    long2 result = convert_long2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long2 _cl_overloadable
  convert_long2_rtz(double2 a)
  {
    cl_set_rounding_mode(0);
    long2 result = convert_long2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long2 _cl_overloadable
  convert_long2_sat_rtz(double2 a)
  {
    cl_set_rounding_mode(0);
    long2 result = convert_long2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long2 _cl_overloadable
  convert_long2_rtp(double2 a)
  {
    cl_set_rounding_mode(2);
    long2 result = convert_long2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long2 _cl_overloadable
  convert_long2_sat_rtp(double2 a)
  {
    cl_set_rounding_mode(2);
    long2 result = convert_long2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long2 _cl_overloadable
  convert_long2_rtn(double2 a)
  {
    cl_set_rounding_mode(3);
    long2 result = convert_long2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long2 _cl_overloadable
  convert_long2_sat_rtn(double2 a)
  {
    cl_set_rounding_mode(3);
    long2 result = convert_long2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long4 _cl_overloadable
  convert_long4_rte(double4 a)
  {
    cl_set_rounding_mode(1);
    long4 result = convert_long4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long4 _cl_overloadable
  convert_long4_sat_rte(double4 a)
  {
    cl_set_rounding_mode(1);
    long4 result = convert_long4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long4 _cl_overloadable
  convert_long4_rtz(double4 a)
  {
    cl_set_rounding_mode(0);
    long4 result = convert_long4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long4 _cl_overloadable
  convert_long4_sat_rtz(double4 a)
  {
    cl_set_rounding_mode(0);
    long4 result = convert_long4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long4 _cl_overloadable
  convert_long4_rtp(double4 a)
  {
    cl_set_rounding_mode(2);
    long4 result = convert_long4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long4 _cl_overloadable
  convert_long4_sat_rtp(double4 a)
  {
    cl_set_rounding_mode(2);
    long4 result = convert_long4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long4 _cl_overloadable
  convert_long4_rtn(double4 a)
  {
    cl_set_rounding_mode(3);
    long4 result = convert_long4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long4 _cl_overloadable
  convert_long4_sat_rtn(double4 a)
  {
    cl_set_rounding_mode(3);
    long4 result = convert_long4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long8 _cl_overloadable
  convert_long8_rte(double8 a)
  {
    cl_set_rounding_mode(1);
    long8 result = convert_long8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long8 _cl_overloadable
  convert_long8_sat_rte(double8 a)
  {
    cl_set_rounding_mode(1);
    long8 result = convert_long8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long8 _cl_overloadable
  convert_long8_rtz(double8 a)
  {
    cl_set_rounding_mode(0);
    long8 result = convert_long8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long8 _cl_overloadable
  convert_long8_sat_rtz(double8 a)
  {
    cl_set_rounding_mode(0);
    long8 result = convert_long8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long8 _cl_overloadable
  convert_long8_rtp(double8 a)
  {
    cl_set_rounding_mode(2);
    long8 result = convert_long8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long8 _cl_overloadable
  convert_long8_sat_rtp(double8 a)
  {
    cl_set_rounding_mode(2);
    long8 result = convert_long8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long8 _cl_overloadable
  convert_long8_rtn(double8 a)
  {
    cl_set_rounding_mode(3);
    long8 result = convert_long8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long8 _cl_overloadable
  convert_long8_sat_rtn(double8 a)
  {
    cl_set_rounding_mode(3);
    long8 result = convert_long8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long16 _cl_overloadable
  convert_long16_rte(double16 a)
  {
    cl_set_rounding_mode(1);
    long16 result = convert_long16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long16 _cl_overloadable
  convert_long16_sat_rte(double16 a)
  {
    cl_set_rounding_mode(1);
    long16 result = convert_long16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long16 _cl_overloadable
  convert_long16_rtz(double16 a)
  {
    cl_set_rounding_mode(0);
    long16 result = convert_long16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long16 _cl_overloadable
  convert_long16_sat_rtz(double16 a)
  {
    cl_set_rounding_mode(0);
    long16 result = convert_long16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long16 _cl_overloadable
  convert_long16_rtp(double16 a)
  {
    cl_set_rounding_mode(2);
    long16 result = convert_long16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long16 _cl_overloadable
  convert_long16_sat_rtp(double16 a)
  {
    cl_set_rounding_mode(2);
    long16 result = convert_long16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long16 _cl_overloadable
  convert_long16_rtn(double16 a)
  {
    cl_set_rounding_mode(3);
    long16 result = convert_long16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  long16 _cl_overloadable
  convert_long16_sat_rtn(double16 a)
  {
    cl_set_rounding_mode(3);
    long16 result = convert_long16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_rte(double a)
  {
    cl_set_rounding_mode(1);
    ulong result = convert_ulong(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat_rte(double a)
  {
    cl_set_rounding_mode(1);
    ulong result = convert_ulong_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_rtz(double a)
  {
    cl_set_rounding_mode(0);
    ulong result = convert_ulong(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat_rtz(double a)
  {
    cl_set_rounding_mode(0);
    ulong result = convert_ulong_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_rtp(double a)
  {
    cl_set_rounding_mode(2);
    ulong result = convert_ulong(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat_rtp(double a)
  {
    cl_set_rounding_mode(2);
    ulong result = convert_ulong_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_rtn(double a)
  {
    cl_set_rounding_mode(3);
    ulong result = convert_ulong(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong _cl_overloadable
  convert_ulong_sat_rtn(double a)
  {
    cl_set_rounding_mode(3);
    ulong result = convert_ulong_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong2 _cl_overloadable
  convert_ulong2_rte(double2 a)
  {
    cl_set_rounding_mode(1);
    ulong2 result = convert_ulong2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong2 _cl_overloadable
  convert_ulong2_sat_rte(double2 a)
  {
    cl_set_rounding_mode(1);
    ulong2 result = convert_ulong2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong2 _cl_overloadable
  convert_ulong2_rtz(double2 a)
  {
    cl_set_rounding_mode(0);
    ulong2 result = convert_ulong2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong2 _cl_overloadable
  convert_ulong2_sat_rtz(double2 a)
  {
    cl_set_rounding_mode(0);
    ulong2 result = convert_ulong2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong2 _cl_overloadable
  convert_ulong2_rtp(double2 a)
  {
    cl_set_rounding_mode(2);
    ulong2 result = convert_ulong2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong2 _cl_overloadable
  convert_ulong2_sat_rtp(double2 a)
  {
    cl_set_rounding_mode(2);
    ulong2 result = convert_ulong2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong2 _cl_overloadable
  convert_ulong2_rtn(double2 a)
  {
    cl_set_rounding_mode(3);
    ulong2 result = convert_ulong2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong2 _cl_overloadable
  convert_ulong2_sat_rtn(double2 a)
  {
    cl_set_rounding_mode(3);
    ulong2 result = convert_ulong2_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong4 _cl_overloadable
  convert_ulong4_rte(double4 a)
  {
    cl_set_rounding_mode(1);
    ulong4 result = convert_ulong4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong4 _cl_overloadable
  convert_ulong4_sat_rte(double4 a)
  {
    cl_set_rounding_mode(1);
    ulong4 result = convert_ulong4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong4 _cl_overloadable
  convert_ulong4_rtz(double4 a)
  {
    cl_set_rounding_mode(0);
    ulong4 result = convert_ulong4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong4 _cl_overloadable
  convert_ulong4_sat_rtz(double4 a)
  {
    cl_set_rounding_mode(0);
    ulong4 result = convert_ulong4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong4 _cl_overloadable
  convert_ulong4_rtp(double4 a)
  {
    cl_set_rounding_mode(2);
    ulong4 result = convert_ulong4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong4 _cl_overloadable
  convert_ulong4_sat_rtp(double4 a)
  {
    cl_set_rounding_mode(2);
    ulong4 result = convert_ulong4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong4 _cl_overloadable
  convert_ulong4_rtn(double4 a)
  {
    cl_set_rounding_mode(3);
    ulong4 result = convert_ulong4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong4 _cl_overloadable
  convert_ulong4_sat_rtn(double4 a)
  {
    cl_set_rounding_mode(3);
    ulong4 result = convert_ulong4_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong8 _cl_overloadable
  convert_ulong8_rte(double8 a)
  {
    cl_set_rounding_mode(1);
    ulong8 result = convert_ulong8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong8 _cl_overloadable
  convert_ulong8_sat_rte(double8 a)
  {
    cl_set_rounding_mode(1);
    ulong8 result = convert_ulong8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong8 _cl_overloadable
  convert_ulong8_rtz(double8 a)
  {
    cl_set_rounding_mode(0);
    ulong8 result = convert_ulong8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong8 _cl_overloadable
  convert_ulong8_sat_rtz(double8 a)
  {
    cl_set_rounding_mode(0);
    ulong8 result = convert_ulong8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong8 _cl_overloadable
  convert_ulong8_rtp(double8 a)
  {
    cl_set_rounding_mode(2);
    ulong8 result = convert_ulong8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong8 _cl_overloadable
  convert_ulong8_sat_rtp(double8 a)
  {
    cl_set_rounding_mode(2);
    ulong8 result = convert_ulong8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong8 _cl_overloadable
  convert_ulong8_rtn(double8 a)
  {
    cl_set_rounding_mode(3);
    ulong8 result = convert_ulong8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong8 _cl_overloadable
  convert_ulong8_sat_rtn(double8 a)
  {
    cl_set_rounding_mode(3);
    ulong8 result = convert_ulong8_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong16 _cl_overloadable
  convert_ulong16_rte(double16 a)
  {
    cl_set_rounding_mode(1);
    ulong16 result = convert_ulong16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong16 _cl_overloadable
  convert_ulong16_sat_rte(double16 a)
  {
    cl_set_rounding_mode(1);
    ulong16 result = convert_ulong16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong16 _cl_overloadable
  convert_ulong16_rtz(double16 a)
  {
    cl_set_rounding_mode(0);
    ulong16 result = convert_ulong16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong16 _cl_overloadable
  convert_ulong16_sat_rtz(double16 a)
  {
    cl_set_rounding_mode(0);
    ulong16 result = convert_ulong16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong16 _cl_overloadable
  convert_ulong16_rtp(double16 a)
  {
    cl_set_rounding_mode(2);
    ulong16 result = convert_ulong16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong16 _cl_overloadable
  convert_ulong16_sat_rtp(double16 a)
  {
    cl_set_rounding_mode(2);
    ulong16 result = convert_ulong16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong16 _cl_overloadable
  convert_ulong16_rtn(double16 a)
  {
    cl_set_rounding_mode(3);
    ulong16 result = convert_ulong16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  ulong16 _cl_overloadable
  convert_ulong16_sat_rtn(double16 a)
  {
    cl_set_rounding_mode(3);
    ulong16 result = convert_ulong16_sat(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
  float _cl_overloadable
  convert_float_rte(char a)
  {
    cl_set_rounding_mode(1);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtz(char a)
  {
    cl_set_rounding_mode(0);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtp(char a)
  {
    cl_set_rounding_mode(2);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtn(char a)
  {
    cl_set_rounding_mode(3);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rte(char2 a)
  {
    cl_set_rounding_mode(1);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtz(char2 a)
  {
    cl_set_rounding_mode(0);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtp(char2 a)
  {
    cl_set_rounding_mode(2);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtn(char2 a)
  {
    cl_set_rounding_mode(3);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rte(char4 a)
  {
    cl_set_rounding_mode(1);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtz(char4 a)
  {
    cl_set_rounding_mode(0);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtp(char4 a)
  {
    cl_set_rounding_mode(2);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtn(char4 a)
  {
    cl_set_rounding_mode(3);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rte(char8 a)
  {
    cl_set_rounding_mode(1);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtz(char8 a)
  {
    cl_set_rounding_mode(0);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtp(char8 a)
  {
    cl_set_rounding_mode(2);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtn(char8 a)
  {
    cl_set_rounding_mode(3);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rte(char16 a)
  {
    cl_set_rounding_mode(1);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtz(char16 a)
  {
    cl_set_rounding_mode(0);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtp(char16 a)
  {
    cl_set_rounding_mode(2);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtn(char16 a)
  {
    cl_set_rounding_mode(3);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
__IF_FP64(
  double _cl_overloadable
  convert_double_rte(char a)
  {
    cl_set_rounding_mode(1);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtz(char a)
  {
    cl_set_rounding_mode(0);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtp(char a)
  {
    cl_set_rounding_mode(2);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtn(char a)
  {
    cl_set_rounding_mode(3);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rte(char2 a)
  {
    cl_set_rounding_mode(1);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtz(char2 a)
  {
    cl_set_rounding_mode(0);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtp(char2 a)
  {
    cl_set_rounding_mode(2);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtn(char2 a)
  {
    cl_set_rounding_mode(3);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rte(char4 a)
  {
    cl_set_rounding_mode(1);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtz(char4 a)
  {
    cl_set_rounding_mode(0);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtp(char4 a)
  {
    cl_set_rounding_mode(2);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtn(char4 a)
  {
    cl_set_rounding_mode(3);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rte(char8 a)
  {
    cl_set_rounding_mode(1);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtz(char8 a)
  {
    cl_set_rounding_mode(0);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtp(char8 a)
  {
    cl_set_rounding_mode(2);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtn(char8 a)
  {
    cl_set_rounding_mode(3);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rte(char16 a)
  {
    cl_set_rounding_mode(1);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtz(char16 a)
  {
    cl_set_rounding_mode(0);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtp(char16 a)
  {
    cl_set_rounding_mode(2);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtn(char16 a)
  {
    cl_set_rounding_mode(3);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
  float _cl_overloadable
  convert_float_rte(uchar a)
  {
    cl_set_rounding_mode(1);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtz(uchar a)
  {
    cl_set_rounding_mode(0);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtp(uchar a)
  {
    cl_set_rounding_mode(2);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtn(uchar a)
  {
    cl_set_rounding_mode(3);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rte(uchar2 a)
  {
    cl_set_rounding_mode(1);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtz(uchar2 a)
  {
    cl_set_rounding_mode(0);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtp(uchar2 a)
  {
    cl_set_rounding_mode(2);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtn(uchar2 a)
  {
    cl_set_rounding_mode(3);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rte(uchar4 a)
  {
    cl_set_rounding_mode(1);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtz(uchar4 a)
  {
    cl_set_rounding_mode(0);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtp(uchar4 a)
  {
    cl_set_rounding_mode(2);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtn(uchar4 a)
  {
    cl_set_rounding_mode(3);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rte(uchar8 a)
  {
    cl_set_rounding_mode(1);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtz(uchar8 a)
  {
    cl_set_rounding_mode(0);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtp(uchar8 a)
  {
    cl_set_rounding_mode(2);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtn(uchar8 a)
  {
    cl_set_rounding_mode(3);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rte(uchar16 a)
  {
    cl_set_rounding_mode(1);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtz(uchar16 a)
  {
    cl_set_rounding_mode(0);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtp(uchar16 a)
  {
    cl_set_rounding_mode(2);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtn(uchar16 a)
  {
    cl_set_rounding_mode(3);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
__IF_FP64(
  double _cl_overloadable
  convert_double_rte(uchar a)
  {
    cl_set_rounding_mode(1);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtz(uchar a)
  {
    cl_set_rounding_mode(0);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtp(uchar a)
  {
    cl_set_rounding_mode(2);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtn(uchar a)
  {
    cl_set_rounding_mode(3);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rte(uchar2 a)
  {
    cl_set_rounding_mode(1);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtz(uchar2 a)
  {
    cl_set_rounding_mode(0);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtp(uchar2 a)
  {
    cl_set_rounding_mode(2);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtn(uchar2 a)
  {
    cl_set_rounding_mode(3);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rte(uchar4 a)
  {
    cl_set_rounding_mode(1);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtz(uchar4 a)
  {
    cl_set_rounding_mode(0);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtp(uchar4 a)
  {
    cl_set_rounding_mode(2);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtn(uchar4 a)
  {
    cl_set_rounding_mode(3);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rte(uchar8 a)
  {
    cl_set_rounding_mode(1);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtz(uchar8 a)
  {
    cl_set_rounding_mode(0);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtp(uchar8 a)
  {
    cl_set_rounding_mode(2);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtn(uchar8 a)
  {
    cl_set_rounding_mode(3);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rte(uchar16 a)
  {
    cl_set_rounding_mode(1);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtz(uchar16 a)
  {
    cl_set_rounding_mode(0);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtp(uchar16 a)
  {
    cl_set_rounding_mode(2);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtn(uchar16 a)
  {
    cl_set_rounding_mode(3);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
  float _cl_overloadable
  convert_float_rte(short a)
  {
    cl_set_rounding_mode(1);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtz(short a)
  {
    cl_set_rounding_mode(0);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtp(short a)
  {
    cl_set_rounding_mode(2);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtn(short a)
  {
    cl_set_rounding_mode(3);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rte(short2 a)
  {
    cl_set_rounding_mode(1);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtz(short2 a)
  {
    cl_set_rounding_mode(0);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtp(short2 a)
  {
    cl_set_rounding_mode(2);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtn(short2 a)
  {
    cl_set_rounding_mode(3);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rte(short4 a)
  {
    cl_set_rounding_mode(1);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtz(short4 a)
  {
    cl_set_rounding_mode(0);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtp(short4 a)
  {
    cl_set_rounding_mode(2);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtn(short4 a)
  {
    cl_set_rounding_mode(3);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rte(short8 a)
  {
    cl_set_rounding_mode(1);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtz(short8 a)
  {
    cl_set_rounding_mode(0);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtp(short8 a)
  {
    cl_set_rounding_mode(2);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtn(short8 a)
  {
    cl_set_rounding_mode(3);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rte(short16 a)
  {
    cl_set_rounding_mode(1);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtz(short16 a)
  {
    cl_set_rounding_mode(0);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtp(short16 a)
  {
    cl_set_rounding_mode(2);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtn(short16 a)
  {
    cl_set_rounding_mode(3);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
__IF_FP64(
  double _cl_overloadable
  convert_double_rte(short a)
  {
    cl_set_rounding_mode(1);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtz(short a)
  {
    cl_set_rounding_mode(0);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtp(short a)
  {
    cl_set_rounding_mode(2);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtn(short a)
  {
    cl_set_rounding_mode(3);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rte(short2 a)
  {
    cl_set_rounding_mode(1);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtz(short2 a)
  {
    cl_set_rounding_mode(0);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtp(short2 a)
  {
    cl_set_rounding_mode(2);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtn(short2 a)
  {
    cl_set_rounding_mode(3);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rte(short4 a)
  {
    cl_set_rounding_mode(1);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtz(short4 a)
  {
    cl_set_rounding_mode(0);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtp(short4 a)
  {
    cl_set_rounding_mode(2);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtn(short4 a)
  {
    cl_set_rounding_mode(3);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rte(short8 a)
  {
    cl_set_rounding_mode(1);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtz(short8 a)
  {
    cl_set_rounding_mode(0);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtp(short8 a)
  {
    cl_set_rounding_mode(2);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtn(short8 a)
  {
    cl_set_rounding_mode(3);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rte(short16 a)
  {
    cl_set_rounding_mode(1);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtz(short16 a)
  {
    cl_set_rounding_mode(0);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtp(short16 a)
  {
    cl_set_rounding_mode(2);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtn(short16 a)
  {
    cl_set_rounding_mode(3);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
  float _cl_overloadable
  convert_float_rte(ushort a)
  {
    cl_set_rounding_mode(1);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtz(ushort a)
  {
    cl_set_rounding_mode(0);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtp(ushort a)
  {
    cl_set_rounding_mode(2);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtn(ushort a)
  {
    cl_set_rounding_mode(3);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rte(ushort2 a)
  {
    cl_set_rounding_mode(1);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtz(ushort2 a)
  {
    cl_set_rounding_mode(0);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtp(ushort2 a)
  {
    cl_set_rounding_mode(2);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtn(ushort2 a)
  {
    cl_set_rounding_mode(3);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rte(ushort4 a)
  {
    cl_set_rounding_mode(1);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtz(ushort4 a)
  {
    cl_set_rounding_mode(0);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtp(ushort4 a)
  {
    cl_set_rounding_mode(2);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtn(ushort4 a)
  {
    cl_set_rounding_mode(3);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rte(ushort8 a)
  {
    cl_set_rounding_mode(1);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtz(ushort8 a)
  {
    cl_set_rounding_mode(0);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtp(ushort8 a)
  {
    cl_set_rounding_mode(2);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtn(ushort8 a)
  {
    cl_set_rounding_mode(3);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rte(ushort16 a)
  {
    cl_set_rounding_mode(1);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtz(ushort16 a)
  {
    cl_set_rounding_mode(0);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtp(ushort16 a)
  {
    cl_set_rounding_mode(2);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtn(ushort16 a)
  {
    cl_set_rounding_mode(3);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
__IF_FP64(
  double _cl_overloadable
  convert_double_rte(ushort a)
  {
    cl_set_rounding_mode(1);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtz(ushort a)
  {
    cl_set_rounding_mode(0);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtp(ushort a)
  {
    cl_set_rounding_mode(2);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtn(ushort a)
  {
    cl_set_rounding_mode(3);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rte(ushort2 a)
  {
    cl_set_rounding_mode(1);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtz(ushort2 a)
  {
    cl_set_rounding_mode(0);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtp(ushort2 a)
  {
    cl_set_rounding_mode(2);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtn(ushort2 a)
  {
    cl_set_rounding_mode(3);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rte(ushort4 a)
  {
    cl_set_rounding_mode(1);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtz(ushort4 a)
  {
    cl_set_rounding_mode(0);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtp(ushort4 a)
  {
    cl_set_rounding_mode(2);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtn(ushort4 a)
  {
    cl_set_rounding_mode(3);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rte(ushort8 a)
  {
    cl_set_rounding_mode(1);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtz(ushort8 a)
  {
    cl_set_rounding_mode(0);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtp(ushort8 a)
  {
    cl_set_rounding_mode(2);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtn(ushort8 a)
  {
    cl_set_rounding_mode(3);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rte(ushort16 a)
  {
    cl_set_rounding_mode(1);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtz(ushort16 a)
  {
    cl_set_rounding_mode(0);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtp(ushort16 a)
  {
    cl_set_rounding_mode(2);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtn(ushort16 a)
  {
    cl_set_rounding_mode(3);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
  float _cl_overloadable
  convert_float_rte(int a)
  {
    cl_set_rounding_mode(1);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtz(int a)
  {
    cl_set_rounding_mode(0);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtp(int a)
  {
    cl_set_rounding_mode(2);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtn(int a)
  {
    cl_set_rounding_mode(3);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rte(int2 a)
  {
    cl_set_rounding_mode(1);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtz(int2 a)
  {
    cl_set_rounding_mode(0);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtp(int2 a)
  {
    cl_set_rounding_mode(2);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtn(int2 a)
  {
    cl_set_rounding_mode(3);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rte(int4 a)
  {
    cl_set_rounding_mode(1);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtz(int4 a)
  {
    cl_set_rounding_mode(0);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtp(int4 a)
  {
    cl_set_rounding_mode(2);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtn(int4 a)
  {
    cl_set_rounding_mode(3);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rte(int8 a)
  {
    cl_set_rounding_mode(1);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtz(int8 a)
  {
    cl_set_rounding_mode(0);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtp(int8 a)
  {
    cl_set_rounding_mode(2);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtn(int8 a)
  {
    cl_set_rounding_mode(3);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rte(int16 a)
  {
    cl_set_rounding_mode(1);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtz(int16 a)
  {
    cl_set_rounding_mode(0);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtp(int16 a)
  {
    cl_set_rounding_mode(2);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtn(int16 a)
  {
    cl_set_rounding_mode(3);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
__IF_FP64(
  double _cl_overloadable
  convert_double_rte(int a)
  {
    cl_set_rounding_mode(1);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtz(int a)
  {
    cl_set_rounding_mode(0);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtp(int a)
  {
    cl_set_rounding_mode(2);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtn(int a)
  {
    cl_set_rounding_mode(3);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rte(int2 a)
  {
    cl_set_rounding_mode(1);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtz(int2 a)
  {
    cl_set_rounding_mode(0);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtp(int2 a)
  {
    cl_set_rounding_mode(2);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtn(int2 a)
  {
    cl_set_rounding_mode(3);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rte(int4 a)
  {
    cl_set_rounding_mode(1);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtz(int4 a)
  {
    cl_set_rounding_mode(0);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtp(int4 a)
  {
    cl_set_rounding_mode(2);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtn(int4 a)
  {
    cl_set_rounding_mode(3);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rte(int8 a)
  {
    cl_set_rounding_mode(1);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtz(int8 a)
  {
    cl_set_rounding_mode(0);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtp(int8 a)
  {
    cl_set_rounding_mode(2);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtn(int8 a)
  {
    cl_set_rounding_mode(3);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rte(int16 a)
  {
    cl_set_rounding_mode(1);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtz(int16 a)
  {
    cl_set_rounding_mode(0);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtp(int16 a)
  {
    cl_set_rounding_mode(2);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtn(int16 a)
  {
    cl_set_rounding_mode(3);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
  float _cl_overloadable
  convert_float_rte(uint a)
  {
    cl_set_rounding_mode(1);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtz(uint a)
  {
    cl_set_rounding_mode(0);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtp(uint a)
  {
    cl_set_rounding_mode(2);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float _cl_overloadable
  convert_float_rtn(uint a)
  {
    cl_set_rounding_mode(3);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rte(uint2 a)
  {
    cl_set_rounding_mode(1);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtz(uint2 a)
  {
    cl_set_rounding_mode(0);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtp(uint2 a)
  {
    cl_set_rounding_mode(2);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float2 _cl_overloadable
  convert_float2_rtn(uint2 a)
  {
    cl_set_rounding_mode(3);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rte(uint4 a)
  {
    cl_set_rounding_mode(1);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtz(uint4 a)
  {
    cl_set_rounding_mode(0);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtp(uint4 a)
  {
    cl_set_rounding_mode(2);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float4 _cl_overloadable
  convert_float4_rtn(uint4 a)
  {
    cl_set_rounding_mode(3);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rte(uint8 a)
  {
    cl_set_rounding_mode(1);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtz(uint8 a)
  {
    cl_set_rounding_mode(0);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtp(uint8 a)
  {
    cl_set_rounding_mode(2);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float8 _cl_overloadable
  convert_float8_rtn(uint8 a)
  {
    cl_set_rounding_mode(3);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rte(uint16 a)
  {
    cl_set_rounding_mode(1);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtz(uint16 a)
  {
    cl_set_rounding_mode(0);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtp(uint16 a)
  {
    cl_set_rounding_mode(2);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
  float16 _cl_overloadable
  convert_float16_rtn(uint16 a)
  {
    cl_set_rounding_mode(3);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
__IF_FP64(
  double _cl_overloadable
  convert_double_rte(uint a)
  {
    cl_set_rounding_mode(1);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtz(uint a)
  {
    cl_set_rounding_mode(0);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtp(uint a)
  {
    cl_set_rounding_mode(2);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double _cl_overloadable
  convert_double_rtn(uint a)
  {
    cl_set_rounding_mode(3);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rte(uint2 a)
  {
    cl_set_rounding_mode(1);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtz(uint2 a)
  {
    cl_set_rounding_mode(0);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtp(uint2 a)
  {
    cl_set_rounding_mode(2);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double2 _cl_overloadable
  convert_double2_rtn(uint2 a)
  {
    cl_set_rounding_mode(3);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rte(uint4 a)
  {
    cl_set_rounding_mode(1);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtz(uint4 a)
  {
    cl_set_rounding_mode(0);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtp(uint4 a)
  {
    cl_set_rounding_mode(2);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double4 _cl_overloadable
  convert_double4_rtn(uint4 a)
  {
    cl_set_rounding_mode(3);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rte(uint8 a)
  {
    cl_set_rounding_mode(1);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtz(uint8 a)
  {
    cl_set_rounding_mode(0);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtp(uint8 a)
  {
    cl_set_rounding_mode(2);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double8 _cl_overloadable
  convert_double8_rtn(uint8 a)
  {
    cl_set_rounding_mode(3);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rte(uint16 a)
  {
    cl_set_rounding_mode(1);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtz(uint16 a)
  {
    cl_set_rounding_mode(0);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtp(uint16 a)
  {
    cl_set_rounding_mode(2);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_FP64(
  double16 _cl_overloadable
  convert_double16_rtn(uint16 a)
  {
    cl_set_rounding_mode(3);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float _cl_overloadable
  convert_float_rte(long a)
  {
    cl_set_rounding_mode(1);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float _cl_overloadable
  convert_float_rtz(long a)
  {
    cl_set_rounding_mode(0);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float _cl_overloadable
  convert_float_rtp(long a)
  {
    cl_set_rounding_mode(2);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float _cl_overloadable
  convert_float_rtn(long a)
  {
    cl_set_rounding_mode(3);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float2 _cl_overloadable
  convert_float2_rte(long2 a)
  {
    cl_set_rounding_mode(1);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float2 _cl_overloadable
  convert_float2_rtz(long2 a)
  {
    cl_set_rounding_mode(0);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float2 _cl_overloadable
  convert_float2_rtp(long2 a)
  {
    cl_set_rounding_mode(2);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float2 _cl_overloadable
  convert_float2_rtn(long2 a)
  {
    cl_set_rounding_mode(3);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float4 _cl_overloadable
  convert_float4_rte(long4 a)
  {
    cl_set_rounding_mode(1);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float4 _cl_overloadable
  convert_float4_rtz(long4 a)
  {
    cl_set_rounding_mode(0);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float4 _cl_overloadable
  convert_float4_rtp(long4 a)
  {
    cl_set_rounding_mode(2);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float4 _cl_overloadable
  convert_float4_rtn(long4 a)
  {
    cl_set_rounding_mode(3);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float8 _cl_overloadable
  convert_float8_rte(long8 a)
  {
    cl_set_rounding_mode(1);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float8 _cl_overloadable
  convert_float8_rtz(long8 a)
  {
    cl_set_rounding_mode(0);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float8 _cl_overloadable
  convert_float8_rtp(long8 a)
  {
    cl_set_rounding_mode(2);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float8 _cl_overloadable
  convert_float8_rtn(long8 a)
  {
    cl_set_rounding_mode(3);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float16 _cl_overloadable
  convert_float16_rte(long16 a)
  {
    cl_set_rounding_mode(1);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float16 _cl_overloadable
  convert_float16_rtz(long16 a)
  {
    cl_set_rounding_mode(0);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float16 _cl_overloadable
  convert_float16_rtp(long16 a)
  {
    cl_set_rounding_mode(2);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float16 _cl_overloadable
  convert_float16_rtn(long16 a)
  {
    cl_set_rounding_mode(3);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double _cl_overloadable
  convert_double_rte(long a)
  {
    cl_set_rounding_mode(1);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double _cl_overloadable
  convert_double_rtz(long a)
  {
    cl_set_rounding_mode(0);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double _cl_overloadable
  convert_double_rtp(long a)
  {
    cl_set_rounding_mode(2);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double _cl_overloadable
  convert_double_rtn(long a)
  {
    cl_set_rounding_mode(3);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double2 _cl_overloadable
  convert_double2_rte(long2 a)
  {
    cl_set_rounding_mode(1);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double2 _cl_overloadable
  convert_double2_rtz(long2 a)
  {
    cl_set_rounding_mode(0);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double2 _cl_overloadable
  convert_double2_rtp(long2 a)
  {
    cl_set_rounding_mode(2);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double2 _cl_overloadable
  convert_double2_rtn(long2 a)
  {
    cl_set_rounding_mode(3);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double4 _cl_overloadable
  convert_double4_rte(long4 a)
  {
    cl_set_rounding_mode(1);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double4 _cl_overloadable
  convert_double4_rtz(long4 a)
  {
    cl_set_rounding_mode(0);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double4 _cl_overloadable
  convert_double4_rtp(long4 a)
  {
    cl_set_rounding_mode(2);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double4 _cl_overloadable
  convert_double4_rtn(long4 a)
  {
    cl_set_rounding_mode(3);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double8 _cl_overloadable
  convert_double8_rte(long8 a)
  {
    cl_set_rounding_mode(1);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double8 _cl_overloadable
  convert_double8_rtz(long8 a)
  {
    cl_set_rounding_mode(0);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double8 _cl_overloadable
  convert_double8_rtp(long8 a)
  {
    cl_set_rounding_mode(2);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double8 _cl_overloadable
  convert_double8_rtn(long8 a)
  {
    cl_set_rounding_mode(3);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double16 _cl_overloadable
  convert_double16_rte(long16 a)
  {
    cl_set_rounding_mode(1);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double16 _cl_overloadable
  convert_double16_rtz(long16 a)
  {
    cl_set_rounding_mode(0);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double16 _cl_overloadable
  convert_double16_rtp(long16 a)
  {
    cl_set_rounding_mode(2);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double16 _cl_overloadable
  convert_double16_rtn(long16 a)
  {
    cl_set_rounding_mode(3);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float _cl_overloadable
  convert_float_rte(ulong a)
  {
    cl_set_rounding_mode(1);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float _cl_overloadable
  convert_float_rtz(ulong a)
  {
    cl_set_rounding_mode(0);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float _cl_overloadable
  convert_float_rtp(ulong a)
  {
    cl_set_rounding_mode(2);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float _cl_overloadable
  convert_float_rtn(ulong a)
  {
    cl_set_rounding_mode(3);
    float result = convert_float(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float2 _cl_overloadable
  convert_float2_rte(ulong2 a)
  {
    cl_set_rounding_mode(1);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float2 _cl_overloadable
  convert_float2_rtz(ulong2 a)
  {
    cl_set_rounding_mode(0);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float2 _cl_overloadable
  convert_float2_rtp(ulong2 a)
  {
    cl_set_rounding_mode(2);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float2 _cl_overloadable
  convert_float2_rtn(ulong2 a)
  {
    cl_set_rounding_mode(3);
    float2 result = convert_float2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float4 _cl_overloadable
  convert_float4_rte(ulong4 a)
  {
    cl_set_rounding_mode(1);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float4 _cl_overloadable
  convert_float4_rtz(ulong4 a)
  {
    cl_set_rounding_mode(0);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float4 _cl_overloadable
  convert_float4_rtp(ulong4 a)
  {
    cl_set_rounding_mode(2);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float4 _cl_overloadable
  convert_float4_rtn(ulong4 a)
  {
    cl_set_rounding_mode(3);
    float4 result = convert_float4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float8 _cl_overloadable
  convert_float8_rte(ulong8 a)
  {
    cl_set_rounding_mode(1);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float8 _cl_overloadable
  convert_float8_rtz(ulong8 a)
  {
    cl_set_rounding_mode(0);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float8 _cl_overloadable
  convert_float8_rtp(ulong8 a)
  {
    cl_set_rounding_mode(2);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float8 _cl_overloadable
  convert_float8_rtn(ulong8 a)
  {
    cl_set_rounding_mode(3);
    float8 result = convert_float8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float16 _cl_overloadable
  convert_float16_rte(ulong16 a)
  {
    cl_set_rounding_mode(1);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float16 _cl_overloadable
  convert_float16_rtz(ulong16 a)
  {
    cl_set_rounding_mode(0);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float16 _cl_overloadable
  convert_float16_rtp(ulong16 a)
  {
    cl_set_rounding_mode(2);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  float16 _cl_overloadable
  convert_float16_rtn(ulong16 a)
  {
    cl_set_rounding_mode(3);
    float16 result = convert_float16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double _cl_overloadable
  convert_double_rte(ulong a)
  {
    cl_set_rounding_mode(1);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double _cl_overloadable
  convert_double_rtz(ulong a)
  {
    cl_set_rounding_mode(0);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double _cl_overloadable
  convert_double_rtp(ulong a)
  {
    cl_set_rounding_mode(2);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double _cl_overloadable
  convert_double_rtn(ulong a)
  {
    cl_set_rounding_mode(3);
    double result = convert_double(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double2 _cl_overloadable
  convert_double2_rte(ulong2 a)
  {
    cl_set_rounding_mode(1);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double2 _cl_overloadable
  convert_double2_rtz(ulong2 a)
  {
    cl_set_rounding_mode(0);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double2 _cl_overloadable
  convert_double2_rtp(ulong2 a)
  {
    cl_set_rounding_mode(2);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double2 _cl_overloadable
  convert_double2_rtn(ulong2 a)
  {
    cl_set_rounding_mode(3);
    double2 result = convert_double2(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double4 _cl_overloadable
  convert_double4_rte(ulong4 a)
  {
    cl_set_rounding_mode(1);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double4 _cl_overloadable
  convert_double4_rtz(ulong4 a)
  {
    cl_set_rounding_mode(0);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double4 _cl_overloadable
  convert_double4_rtp(ulong4 a)
  {
    cl_set_rounding_mode(2);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double4 _cl_overloadable
  convert_double4_rtn(ulong4 a)
  {
    cl_set_rounding_mode(3);
    double4 result = convert_double4(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double8 _cl_overloadable
  convert_double8_rte(ulong8 a)
  {
    cl_set_rounding_mode(1);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double8 _cl_overloadable
  convert_double8_rtz(ulong8 a)
  {
    cl_set_rounding_mode(0);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double8 _cl_overloadable
  convert_double8_rtp(ulong8 a)
  {
    cl_set_rounding_mode(2);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double8 _cl_overloadable
  convert_double8_rtn(ulong8 a)
  {
    cl_set_rounding_mode(3);
    double8 result = convert_double8(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double16 _cl_overloadable
  convert_double16_rte(ulong16 a)
  {
    cl_set_rounding_mode(1);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double16 _cl_overloadable
  convert_double16_rtz(ulong16 a)
  {
    cl_set_rounding_mode(0);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double16 _cl_overloadable
  convert_double16_rtp(ulong16 a)
  {
    cl_set_rounding_mode(2);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
__IF_INT64(
  double16 _cl_overloadable
  convert_double16_rtn(ulong16 a)
  {
    cl_set_rounding_mode(3);
    double16 result = convert_double16(a);
    cl_set_default_rounding_mode();
    return result;
  }
  
)
