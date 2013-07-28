kernel 
void test_fmin_fmax_fma() {
    volatile float2 a = (float2)(-1.0f, -2.2f);
    volatile float2 b = (float2)(-3.1f, fma(a[0], -4.2f, a[1]));
    float2 max_ = fmax(a, b);
    float2 min_ = fmin(a, b);
    if ((int)max_[0] == -3 && (int)max_[1] == 8 &&
        (int)min_[0] == 1 && (int)min_[1] == 2)
        printf("OK\n");
}
