kernel 
void test_fmin_fmax_fma()
{
    volatile float2 a = (float2)(-1.0f, -2.2f);
    volatile float c = fma(a.x, -4.2f, a.y);
    volatile float2 b = (float2)(-3.1f, c);

    float2 max_ = fmax(a, b);

    float2 min_ = fmin(a, b);

    if (((int)max_.x == -3) && ((int)max_.y == 8) &&
        ((int)min_.x == 1) && ((int)min_.y == 2))
    {

        printf("OK\n");
    }
}
