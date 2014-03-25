kernel 
void test_min_max() {
/*    +FAIL: max(a,b)[0] type=uint2 a=0x15b348c9 b=0xf88e7d07 want=0xf88e7d07 got=0x15b348c9
      +FAIL: min(a,b)[0] type=uint2 a=0x15b348c9 b=0xf88e7d07 want=0x15b348c9 got=0xf88e7d07 */
    volatile uint2 a = (uint2)(0x15b348c9, 0x15b348c9);
    volatile uint2 b = (uint2)(0xf88e7d07, 0xf88e7d07);
    uint2 max_ = max(a, b);
    uint2 min_ = min(a, b);
    if (max_[0] != 0xf88e7d07 || min_[0] != 0x15b348c9) {
        printf("max(a,b)[0] type=uint2 a=0x15b348c9 b=0xf88e7d07 want=0xf88e7d07 got=%x\n", max_[0]);
        printf("min(a,b)[0] type=uint2 a=0x15b348c9 b=0xf88e7d07 want=0x15b348c9 got=%x\n", min_[0]);
    }

    volatile float4 va = (float4)(3.0f, 5.0f, -2.0f, -9.0f);
    volatile float4 vb = (float4)(2.0f, -4.4f, -1.0f, -20.0f); 
    float4 vmax = max(va, vb);
    float4 vmin = min(va, vb);

    if (any(vmax != (float4)(3.0f, 5.0f, -1.0f, -9.0f)) ||
        any(vmin != (float4)(2.0f, -4.4f, -2.0f, -20.0f))) {
        printf("min or max on float4 failed.\n");
    }
}
