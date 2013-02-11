// TESTING: as_TYPEn

__attribute__((__aligned__(256)))
constant char data[256] =
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
  0xb6, 0x66, 0x3f, 0xf1, 0x1f, 0xb5, 0xd4, 0xa4,
  0x1b, 0xca, 0x91, 0x76, 0x84, 0x9c, 0x13, 0xdf,
  0x78, 0xc0, 0x5d, 0x2e, 0x2d, 0xd9, 0x68, 0x76,
  0xb8, 0x05, 0xfd, 0x8f, 0xc5, 0xd4, 0x06, 0xd4,
  0xa3, 0x54, 0x47, 0x76, 0xf6, 0x74, 0x16, 0xd5,
  0xfb, 0x1d, 0xf3, 0xed, 0xd1, 0x57, 0xb4, 0xeb,
  0x7b, 0xb9, 0x98, 0x60, 0x08, 0x2e, 0x41, 0xe5,
  0x7e, 0xdf, 0xf2, 0xaa, 0x4a, 0x6b, 0x08, 0xff,
  0x15, 0x82, 0xe4, 0x5e, 0x03, 0xc8, 0x1b, 0x21,
  0xbb, 0x80, 0x79, 0xa3, 0x1d, 0x7a, 0x3d, 0x18,
  0x4b, 0xa7, 0xd8, 0x6a, 0xc0, 0x67, 0x5d, 0xb0,
  0x87, 0xde, 0x19, 0xa8, 0xa3, 0x35, 0x4e, 0x9b,
  0x96, 0x59, 0x00, 0x2c, 0x95, 0x47, 0x68, 0xce,
  0xa8, 0x50, 0xcb, 0xed, 0x34, 0xad, 0x77, 0x38,
  0xab, 0x58, 0x35, 0x43, 0x5a, 0xe9, 0xa1, 0x5d,
  0x8b, 0x14, 0x48, 0x49, 0xf1, 0x1e, 0x0d, 0xb4
};

void compare_bytes(
    char* name,
    uchar* dst, size_t dst_size,
    uchar* src, size_t src_size
) {
  if (dst_size != src_size) {
    printf("FAIL: %s - size mismatch! dst_size: %u src_size: %u",
           name, (uint)dst_size, (uint)src_size);
    return;
  }
  for (size_t i = 0; i < dst_size; ++i) {
    if (dst[i] != src[i]) {
      printf("FAIL: %s - byte #: %u expected: %#.2x actual: %#.2x",
             name, (uint)i, src[i], dst[i]);
    }
  }
}

kernel void test_as_type()
{
#define TEST_AS_TYPE(SRC, DST)                                      \
  {                                                                 \
    union { SRC value; uchar raw[sizeof(SRC)]; } src;               \
    union { DST value; uchar raw[sizeof(DST)]; } dst;               \
    src.value = *((constant SRC*)data);                             \
    dst.value = as_##DST(src.value);                                \
    compare_bytes("as_" #DST "(" #SRC ")",                          \
                  dst.raw, sizeof(DST), src.raw, sizeof(SRC));      \
  }

/* 1 byte */
#define TEST_AS_TYPE_1(SRC)              \
  TEST_AS_TYPE(SRC, char)                \
  TEST_AS_TYPE(SRC, uchar)

  TEST_AS_TYPE_1(char)
  TEST_AS_TYPE_1(uchar)

/* 2 bytes */
#define TEST_AS_TYPE_2(SRC)              \
  TEST_AS_TYPE(SRC, char2)               \
  TEST_AS_TYPE(SRC, uchar2)              \
  TEST_AS_TYPE(SRC, short)               \
  TEST_AS_TYPE(SRC, ushort)

  TEST_AS_TYPE_2(char2)
  TEST_AS_TYPE_2(uchar2)
  TEST_AS_TYPE_2(short)
  TEST_AS_TYPE_2(ushort)

/* 4 bytes */
#define TEST_AS_TYPE_4(SRC)              \
  TEST_AS_TYPE(SRC, char4)               \
  TEST_AS_TYPE(SRC, uchar4)              \
  TEST_AS_TYPE(SRC, char3)               \
  TEST_AS_TYPE(SRC, uchar3)              \
  TEST_AS_TYPE(SRC, short2)              \
  TEST_AS_TYPE(SRC, ushort2)             \
  TEST_AS_TYPE(SRC, int)                 \
  TEST_AS_TYPE(SRC, uint)                \
  TEST_AS_TYPE(SRC, float)

  TEST_AS_TYPE_4(char4)
  TEST_AS_TYPE_4(uchar4)
  TEST_AS_TYPE_4(char3)
  TEST_AS_TYPE_4(uchar3)
  TEST_AS_TYPE_4(short2)
  TEST_AS_TYPE_4(ushort2)
  TEST_AS_TYPE_4(int)
  TEST_AS_TYPE_4(uint)
  TEST_AS_TYPE_4(float)

/* 8 bytes */
#define TEST_AS_TYPE_8(DST)                   \
  TEST_AS_TYPE(DST, char8)                    \
  TEST_AS_TYPE(DST, uchar8)                   \
  TEST_AS_TYPE(DST, short4)                   \
  TEST_AS_TYPE(DST, ushort4)                  \
  TEST_AS_TYPE(DST, short3)                   \
  TEST_AS_TYPE(DST, ushort3)                  \
  TEST_AS_TYPE(DST, int2)                     \
  TEST_AS_TYPE(DST, uint2)                    \
  __IF_INT64(                                 \
  TEST_AS_TYPE(DST, long)                     \
  TEST_AS_TYPE(DST, ulong))                   \
  TEST_AS_TYPE(DST, float2)                   \
  __IF_FP64(                                  \
  TEST_AS_TYPE(DST, double))

  TEST_AS_TYPE_8(char8)
  TEST_AS_TYPE_8(uchar8)
  TEST_AS_TYPE_8(short4)
  TEST_AS_TYPE_8(ushort4)
  TEST_AS_TYPE_8(short3)
  TEST_AS_TYPE_8(ushort3)
  TEST_AS_TYPE_8(int2)
  TEST_AS_TYPE_8(uint2)
  __IF_INT64(
  TEST_AS_TYPE_8(long)
  TEST_AS_TYPE_8(ulong))
  TEST_AS_TYPE_8(float2)
  __IF_FP64(
  TEST_AS_TYPE_8(double))

/* 16 bytes */
#define TEST_AS_TYPE_16(DST)                  \
  TEST_AS_TYPE(DST, char16)                   \
  TEST_AS_TYPE(DST, uchar16)                  \
  TEST_AS_TYPE(DST, short8)                   \
  TEST_AS_TYPE(DST, ushort8)                  \
  TEST_AS_TYPE(DST, int4)                     \
  TEST_AS_TYPE(DST, uint4)                    \
  TEST_AS_TYPE(DST, int3)                     \
  TEST_AS_TYPE(DST, uint3)                    \
  __IF_INT64(                                 \
  TEST_AS_TYPE(DST, long2)                    \
  TEST_AS_TYPE(DST, ulong2))                  \
  TEST_AS_TYPE(DST, float4)                   \
  TEST_AS_TYPE(DST, float3)                   \
  __IF_FP64(                                  \
  TEST_AS_TYPE(DST, double2))

  TEST_AS_TYPE_16(char16)
  TEST_AS_TYPE_16(uchar16)
  TEST_AS_TYPE_16(short8)
  TEST_AS_TYPE_16(ushort8)
  TEST_AS_TYPE_16(int4)
  TEST_AS_TYPE_16(uint4)
  TEST_AS_TYPE_16(int3)
  TEST_AS_TYPE_16(uint3)
  __IF_INT64(
  TEST_AS_TYPE_16(long2)
  TEST_AS_TYPE_16(ulong2))
  TEST_AS_TYPE_16(float4)
  TEST_AS_TYPE_16(float3)
  __IF_FP64(
  TEST_AS_TYPE_16(double2))

/* 32 bytes */
#define TEST_AS_TYPE_32(DST)                  \
  TEST_AS_TYPE(DST, short16)                  \
  TEST_AS_TYPE(DST, ushort16)                 \
  TEST_AS_TYPE(DST, int8)                     \
  TEST_AS_TYPE(DST, uint8)                    \
  __IF_INT64(                                 \
  TEST_AS_TYPE(DST, long4)                    \
  TEST_AS_TYPE(DST, ulong4)                   \
  TEST_AS_TYPE(DST, long3)                    \
  TEST_AS_TYPE(DST, ulong3))                  \
  TEST_AS_TYPE(DST, float8)                   \
  __IF_FP64(                                  \
  TEST_AS_TYPE(DST, double4)                  \
  TEST_AS_TYPE(DST, double3))

  TEST_AS_TYPE_32(short16)
  TEST_AS_TYPE_32(ushort16)
  TEST_AS_TYPE_32(int8)
  TEST_AS_TYPE_32(uint8)
  __IF_INT64(
  TEST_AS_TYPE_32(long4)
  TEST_AS_TYPE_32(ulong4)
  TEST_AS_TYPE_32(long3)
  TEST_AS_TYPE_32(ulong3))
  TEST_AS_TYPE_32(float8)
  __IF_FP64(
  TEST_AS_TYPE_32(double4)
  TEST_AS_TYPE_32(double3))

/* 64 bytes */
#define TEST_AS_TYPE_64(DST)                  \
  TEST_AS_TYPE(DST, int16)                    \
  TEST_AS_TYPE(DST, uint16)                   \
  __IF_INT64(                                 \
  TEST_AS_TYPE(DST, long8)                    \
  TEST_AS_TYPE(DST, ulong8))                  \
  TEST_AS_TYPE(DST, float16)                  \
  __IF_FP64(                                  \
  TEST_AS_TYPE(DST, double8))

  TEST_AS_TYPE_64(int16)
  TEST_AS_TYPE_64(uint16)
  __IF_INT64(
  TEST_AS_TYPE_64(long8)
  TEST_AS_TYPE_64(ulong8))
  TEST_AS_TYPE_64(float16)
  __IF_FP64(
  TEST_AS_TYPE_64(double8))

/* 128 bytes */
#define TEST_AS_TYPE_128(DST)                 \
  __IF_INT64(                                 \
  TEST_AS_TYPE(DST, long16)                   \
  TEST_AS_TYPE(DST, ulong16))                 \
  __IF_FP64(                                  \
  TEST_AS_TYPE(DST, double16))

  __IF_INT64(
  TEST_AS_TYPE_128(long16)
  TEST_AS_TYPE_128(ulong16))
  __IF_FP64(
  TEST_AS_TYPE_128(double16))             
}

