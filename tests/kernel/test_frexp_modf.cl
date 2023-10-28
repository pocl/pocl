kernel
void test_frexp_modf() {
    volatile float a = 8e2f, b = 1.5f, frac, frac2;
    int exp;
    frac = frexp(a, &exp);
    float x;
    frac2 = modf(b, &x);
    printf("frexp(8e2f): %f %d\n", frac, exp);
    printf("modf(1.5f): %f %f\n", frac2, x);
}
