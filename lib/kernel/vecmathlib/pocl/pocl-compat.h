// -*-C++-*- Compatibility layer to help instantiante functions to
// create a library that can be called from elsewhere



// Make things go fast (and debugging difficult...)
#define VML_NODEBUG
#include "../vecmathlib.h"

#include <algorithm>
#include <cstring>

#define pocl_static_assert(b) typedef char _static_assert[(b)?+1:-1]

// If double precision is not supported, then define single-precision
// (dummy) values to avoid compiler warnings for double precision
// values
#ifndef cl_khr_fp64
#  undef M_PI
#  define M_PI M_PI_F
#endif



// Define vector types

#define int std::int32_t
typedef int int2  __attribute__((__ext_vector_type__( 2)));
typedef int int3  __attribute__((__ext_vector_type__( 3)));
typedef int int4  __attribute__((__ext_vector_type__( 4)));
typedef int int8  __attribute__((__ext_vector_type__( 8)));
typedef int int16 __attribute__((__ext_vector_type__(16)));

#define uint std::uint32_t
typedef uint uint2  __attribute__((__ext_vector_type__( 2)));
typedef uint uint3  __attribute__((__ext_vector_type__( 3)));
typedef uint uint4  __attribute__((__ext_vector_type__( 4)));
typedef uint uint8  __attribute__((__ext_vector_type__( 8)));
typedef uint uint16 __attribute__((__ext_vector_type__(16)));

#ifdef cles_khr_int64
#define long std::int64_t
typedef long long2  __attribute__((__ext_vector_type__( 2)));
typedef long long3  __attribute__((__ext_vector_type__( 3)));
typedef long long4  __attribute__((__ext_vector_type__( 4)));
typedef long long8  __attribute__((__ext_vector_type__( 8)));
typedef long long16 __attribute__((__ext_vector_type__(16)));

#define ulong std::uint64_t
typedef ulong ulong2  __attribute__((__ext_vector_type__( 2)));
typedef ulong ulong3  __attribute__((__ext_vector_type__( 3)));
typedef ulong ulong4  __attribute__((__ext_vector_type__( 4)));
typedef ulong ulong8  __attribute__((__ext_vector_type__( 8)));
typedef ulong ulong16 __attribute__((__ext_vector_type__(16)));
#endif

typedef float float2  __attribute__((__ext_vector_type__( 2)));
typedef float float3  __attribute__((__ext_vector_type__( 3)));
typedef float float4  __attribute__((__ext_vector_type__( 4)));
typedef float float8  __attribute__((__ext_vector_type__( 8)));
typedef float float16 __attribute__((__ext_vector_type__(16)));

#ifdef cl_khr_fp64
typedef double double2  __attribute__((__ext_vector_type__( 2)));
typedef double double3  __attribute__((__ext_vector_type__( 3)));
typedef double double4  __attribute__((__ext_vector_type__( 4)));
typedef double double8  __attribute__((__ext_vector_type__( 8)));
typedef double double16 __attribute__((__ext_vector_type__(16)));
#endif



// Declare pair types for assembling/disassembling vectors
struct pair_int   { int   lo, hi; };
struct pair_int2  { int2  lo, hi; };
struct pair_int3  { int3  lo, hi; };
struct pair_int4  { int4  lo, hi; };
struct pair_int8  { int8  lo, hi; };
struct pair_int16 { int16 lo, hi; };

#ifdef cles_khr_int64
struct pair_long   { long   lo, hi; };
struct pair_long2  { long2  lo, hi; };
struct pair_long3  { long3  lo, hi; };
struct pair_long4  { long4  lo, hi; };
struct pair_long8  { long8  lo, hi; };
struct pair_long16 { long16 lo, hi; };
#endif

struct pair_float   { float   lo, hi; };
struct pair_float2  { float2  lo, hi; };
struct pair_float3  { float3  lo, hi; };
struct pair_float4  { float4  lo, hi; };
struct pair_float8  { float8  lo, hi; };
struct pair_float16 { float16 lo, hi; };

#ifdef cl_khr_fp64
struct pair_double   { double   lo, hi; };
struct pair_double2  { double2  lo, hi; };
struct pair_double3  { double3  lo, hi; };
struct pair_double4  { double4  lo, hi; };
struct pair_double8  { double8  lo, hi; };
struct pair_double16 { double16 lo, hi; };
#endif



// Generic conversion function
template<typename A, typename B>
static B bitcast(A a)
{
  B b;
  std::memcpy(&b, &a, std::min(sizeof a, sizeof b));
  if (sizeof b > sizeof a) {
    std::memset((char*)&b + sizeof a, 0, sizeof b - sizeof a);
  }
  return b;
}
