kernel void test_sizeof() {
#ifdef __EMBEDDED_PROFILE__
// 64-bit longs might not be available
    printf("sizeof(uint) == %d\n", (int)sizeof(uint));
    printf("sizeof(uint2) == %d\n", (int)sizeof(uint2));
    printf("sizeof(uint3) == %d\n", (int)sizeof(uint3));
    printf("sizeof(uint4) == %d\n", (int)sizeof(uint4));
    printf("sizeof(uint8) == %d\n", (int)sizeof(uint8));
    printf("sizeof(uint16) == %d\n", (int)sizeof(uint16));
#else
    printf("sizeof(uint) == %lu\n", sizeof(uint));
    printf("sizeof(uint2) == %lu\n", sizeof(uint2));
    printf("sizeof(uint3) == %lu\n", sizeof(uint3));
    printf("sizeof(uint4) == %lu\n", sizeof(uint4));
    printf("sizeof(uint8) == %lu\n", sizeof(uint8));
    printf("sizeof(uint16) == %lu\n", sizeof(uint16));
#endif
}
