kernel 
void test_convert_sat_regression() {
#ifdef cl_khr_int64
    volatile long2 input = (long2)(0, 1);
    uint2 res = convert_uint2_sat(input);
    if (res[0] != 0 || res[1] != 1) {
        printf("expected (0, 1), got (%x, %x)\n", res[0], res[1]);
    }
#endif
}

