kernel void test_sizeof() {
    // cast to ints as ulong might not be available
    printf("sizeof(uint) == %d\n", (int)sizeof(uint));
    printf("sizeof(uint2) == %d\n", (int)sizeof(uint2));
    printf("sizeof(uint3) == %d\n", (int)sizeof(uint3));
    printf("sizeof(uint4) == %d\n", (int)sizeof(uint4));
    printf("sizeof(uint8) == %d\n", (int)sizeof(uint8));
    printf("sizeof(uint16) == %d\n", (int)sizeof(uint16));
}
