// TESTING: as_TYPEn

#include "common.cl"


_CL_NOINLINE
void clear_bytes(uchar* p, uchar c, size_t n)
{
  for (size_t i = 0; i < n; ++i) {
    p[i] = c;
  }
}

_CL_NOINLINE
void compare_bytes(
    constant char* name,
    const uchar* dst, size_t dst_size, size_t dst_elsize,
    const uchar* src, size_t src_size, size_t src_elsize)
{
  const size_t n = dst_elsize < src_elsize ? dst_elsize : src_elsize;
  if (dst_size != src_size) {
    printf("FAIL: %s - size mismatch! dst_size: %u src_size: %u\n",
           name, (uint)dst_size, (uint)src_size);
    return;
  }
  for (size_t i = 0; i < n; ++i) {
    if (dst[i] != src[i]) {
      printf("FAIL: %s - byte #: %u expected: %#.2x actual: %#.2x\n",
             name, (uint)i, src[i], dst[i]);
    }
  }
}

kernel void test_as_type()
{
  __attribute__((aligned(128)))
  char data[128] =
  {
    0xe4, 0xf9, 0xb4, 0x88, 0x19, 0x65, 0xa2, 0xb6,
    0xa0, 0xfd, 0xa4, 0xbc, 0x11, 0x9e, 0x8c, 0xb2,
    0x7c, 0x15, 0xae, 0x6b, 0x7d, 0xe5, 0xba, 0x8e,
    0xaa, 0x5a, 0x8b, 0x62, 0xd0, 0xcf, 0x17, 0x92,
    0x6d, 0xa4, 0x80, 0x63, 0x05, 0x89, 0x11, 0x4f,
    0xb3, 0x22, 0x8f, 0x3b, 0x5b, 0x2f, 0x26, 0x6b,
    0xb6, 0x97, 0xea, 0x55, 0xf5, 0xb2, 0x37, 0xdb,
    0x41, 0xbc, 0x03, 0x8e, 0x0f, 0xdb, 0x90, 0x88,
    0x53, 0x54, 0x20, 0x25, 0x8a, 0x77, 0x8d, 0x0b,
    0xcc, 0xf6, 0x79, 0x0e, 0xe4, 0x7d, 0xfb, 0xe9,
    0xb2, 0x08, 0x88, 0x91, 0xb5, 0xa3, 0xc0, 0x58,
    0x54, 0x29, 0x5c, 0xef, 0x74, 0xcd, 0x6e, 0xf5,
    0x1c, 0xd7, 0xe4, 0xf2, 0xc3, 0xb8, 0xee, 0x51,
    0x8b, 0x9e, 0x4e, 0xc4, 0xc8, 0xc9, 0xf4, 0x82,
    0x2d, 0x45, 0xbc, 0xa2, 0xfc, 0x16, 0x09, 0xa0,
    0x45, 0x84, 0xdf, 0xe7, 0xd7, 0x1a, 0x32, 0x25,
  };

#define TEST_AS_TYPE(DST, N, SRC, M)                            \
  {                                                             \
    union { SRC value; uchar raw[sizeof(SRC)]; } src;           \
    union { DST value; uchar raw[sizeof(DST)]; } dst;           \
    clear_bytes(src.raw, 0x44, sizeof(SRC));                    \
    clear_bytes(dst.raw, 0x99, sizeof(DST));                    \
    src.value = *((private SRC*)data);                          \
    dst.value = as_##DST(src.value);                            \
    compare_bytes("as_" #DST "((" #SRC "))",                    \
        dst.raw, sizeof(DST), N, src.raw, sizeof(SRC), M);      \
  }

/* 1 byte */
#define TEST_AS_TYPE_1(DST)              \
  TEST_AS_TYPE(DST, 1, char, 1)          \
  TEST_AS_TYPE(DST, 1, uchar, 1)

  TEST_AS_TYPE_1(char)
  TEST_AS_TYPE_1(uchar)

/* 2 bytes */
#define TEST_AS_TYPE_2(DST)              \
  TEST_AS_TYPE(DST, 2, char2, 2)         \
  TEST_AS_TYPE(DST, 2, uchar2, 2)        \
  TEST_AS_TYPE(DST, 2, short, 2)         \
  TEST_AS_TYPE(DST, 2, ushort, 2)

  TEST_AS_TYPE_2(char2)
  TEST_AS_TYPE_2(uchar2)
  TEST_AS_TYPE_2(short)
  TEST_AS_TYPE_2(ushort)

/* 4 bytes */
#define TEST_AS_TYPE_4(DST, N)           \
  TEST_AS_TYPE(DST, N, char4, 4)         \
  TEST_AS_TYPE(DST, N, uchar4, 4)        \
  TEST_AS_TYPE(DST, N, char3, 3)         \
  TEST_AS_TYPE(DST, N, uchar3, 3)        \
  TEST_AS_TYPE(DST, N, short2, 4)        \
  TEST_AS_TYPE(DST, N, ushort2, 4)       \
  TEST_AS_TYPE(DST, N, int, 4)           \
  TEST_AS_TYPE(DST, N, uint, 4)          \
  TEST_AS_TYPE(DST, N, float, 4)

  TEST_AS_TYPE_4(char4, 4)
  TEST_AS_TYPE_4(uchar4, 4)
  TEST_AS_TYPE_4(char3, 3)
  TEST_AS_TYPE_4(uchar3, 3)
  TEST_AS_TYPE_4(short2, 4)
  TEST_AS_TYPE_4(ushort2, 4)
  TEST_AS_TYPE_4(int, 4)
  TEST_AS_TYPE_4(uint, 4)
  TEST_AS_TYPE_4(float, 4)

/* 8 bytes */
#define TEST_AS_TYPE_8(DST, N)           \
  TEST_AS_TYPE(DST, N, char8, 8)         \
  TEST_AS_TYPE(DST, N, uchar8, 8)        \
  TEST_AS_TYPE(DST, N, short4, 8)        \
  TEST_AS_TYPE(DST, N, ushort4, 8)       \
  TEST_AS_TYPE(DST, N, short3, 6)        \
  TEST_AS_TYPE(DST, N, ushort3, 6)       \
  TEST_AS_TYPE(DST, N, int2, 8)          \
  TEST_AS_TYPE(DST, N, uint2, 8)         \
  __IF_INT64(                            \
  TEST_AS_TYPE(DST, N, long, 8)          \
  TEST_AS_TYPE(DST, N, ulong, 8))        \
  TEST_AS_TYPE(DST, N, float2, 8)        \
  __IF_FP64(                             \
  TEST_AS_TYPE(DST, N, double, 8))

  TEST_AS_TYPE_8(char8, 8)
  TEST_AS_TYPE_8(uchar8, 8)
  TEST_AS_TYPE_8(short4, 8)
  TEST_AS_TYPE_8(ushort4, 8)
  TEST_AS_TYPE_8(short3, 6)
  TEST_AS_TYPE_8(ushort3, 6)
  TEST_AS_TYPE_8(int2, 8)
  TEST_AS_TYPE_8(uint2, 8)
  __IF_INT64(
  TEST_AS_TYPE_8(long, 8)
  TEST_AS_TYPE_8(ulong, 8))
  TEST_AS_TYPE_8(float2, 8)
  __IF_FP64(
  TEST_AS_TYPE_8(double, 8))

/* 16 bytes */
#define TEST_AS_TYPE_16(DST, N)          \
  TEST_AS_TYPE(DST, N, char16, 16)       \
  TEST_AS_TYPE(DST, N, uchar16, 16)      \
  TEST_AS_TYPE(DST, N, short8, 16)       \
  TEST_AS_TYPE(DST, N, ushort8, 16)      \
  TEST_AS_TYPE(DST, N, int4, 16)         \
  TEST_AS_TYPE(DST, N, uint4, 16)        \
  TEST_AS_TYPE(DST, N, int3, 12)         \
  TEST_AS_TYPE(DST, N, uint3, 12)        \
  __IF_INT64(                            \
  TEST_AS_TYPE(DST, N, long2, 16)        \
  TEST_AS_TYPE(DST, N, ulong2, 16))      \
  TEST_AS_TYPE(DST, N, float4, 16)       \
  TEST_AS_TYPE(DST, N, float3, 12)       \
  __IF_FP64(                             \
  TEST_AS_TYPE(DST, N, double2, 16))

  TEST_AS_TYPE_16(char16, 16)
  TEST_AS_TYPE_16(uchar16, 16)
  TEST_AS_TYPE_16(short8, 16)
  TEST_AS_TYPE_16(ushort8, 16)
  TEST_AS_TYPE_16(int4, 16)
  TEST_AS_TYPE_16(uint4, 16)
  TEST_AS_TYPE_16(int3, 12)
  TEST_AS_TYPE_16(uint3, 12)
  __IF_INT64(
  TEST_AS_TYPE_16(long2, 16)
  TEST_AS_TYPE_16(ulong2, 16))
  TEST_AS_TYPE_16(float4, 16)
  TEST_AS_TYPE_16(float3, 12)
  __IF_FP64(
  TEST_AS_TYPE_16(double2, 16))

/* 32 bytes */
#define TEST_AS_TYPE_32(DST, N)           \
  TEST_AS_TYPE(DST, N, short16, 32)       \
  TEST_AS_TYPE(DST, N, ushort16, 32)      \
  TEST_AS_TYPE(DST, N, int8, 32)          \
  TEST_AS_TYPE(DST, N, uint8, 32)         \
  __IF_INT64(                             \
  TEST_AS_TYPE(DST, N, long4, 32)         \
  TEST_AS_TYPE(DST, N, ulong4, 32)        \
  TEST_AS_TYPE(DST, N, long3, 24)         \
  TEST_AS_TYPE(DST, N, ulong3, 24))       \
  TEST_AS_TYPE(DST, N, float8, 32)        \
  __IF_FP64(                              \
  TEST_AS_TYPE(DST, N, double4, 32)       \
  TEST_AS_TYPE(DST, N, double3, 24))

  TEST_AS_TYPE_32(short16, 32)
  TEST_AS_TYPE_32(ushort16, 32)
  TEST_AS_TYPE_32(int8, 32)
  TEST_AS_TYPE_32(uint8, 32)
  __IF_INT64(
  TEST_AS_TYPE_32(long4, 32)
  TEST_AS_TYPE_32(ulong4, 32)
  TEST_AS_TYPE_32(long3, 24)
  TEST_AS_TYPE_32(ulong3, 24))
  TEST_AS_TYPE_32(float8, 32)
  __IF_FP64(
  TEST_AS_TYPE_32(double4, 32)
  TEST_AS_TYPE_32(double3, 24))

/* 64 bytes */
#define TEST_AS_TYPE_64(DST)              \
  TEST_AS_TYPE(DST, 64, int16, 64)        \
  TEST_AS_TYPE(DST, 64, uint16, 64)       \
  __IF_INT64(                             \
  TEST_AS_TYPE(DST, 64, long8, 64)        \
  TEST_AS_TYPE(DST, 64, ulong8, 64))      \
  TEST_AS_TYPE(DST, 64, float16, 64)      \
  __IF_FP64(                              \
  TEST_AS_TYPE(DST, 64, double8, 64))

  TEST_AS_TYPE_64(int16)
  TEST_AS_TYPE_64(uint16)
  __IF_INT64(
  TEST_AS_TYPE_64(long8)
  TEST_AS_TYPE_64(ulong8))
  TEST_AS_TYPE_64(float16)
  __IF_FP64(
  TEST_AS_TYPE_64(double8))

/* 128 bytes */
#define TEST_AS_TYPE_128(DST, N)          \
  __IF_INT64(                             \
  TEST_AS_TYPE(DST, N, long16, 128)       \
  TEST_AS_TYPE(DST, N, ulong16, 128))     \
  __IF_FP64(                              \
  TEST_AS_TYPE(DST, N, double16, 128))

  __IF_INT64(
  TEST_AS_TYPE_128(long16, 128)
  TEST_AS_TYPE_128(ulong16, 128))
  __IF_FP64(
  TEST_AS_TYPE_128(double16, 128))             
}

