#pragma OPENCL EXTENSION cl_khr_fp16 : enable

volatile global half x = 1.5h;
volatile global half y = 2.0h;
volatile global int n = 3;
volatile global half8 vx = (half8)(1.5h);
volatile global half8 vy = (half8)(2.0h);
volatile global int8 vn = (int8)(3);

volatile global half hsink;
volatile global int isink;
volatile global half8 vsink;
volatile global int8 visink;

kernel void test_fp16_math_builtins ()
{
  half hx = x;
  half hy = y;
  int hn = n;
  half hip;
  int iexp;
  int quo;
  half hcos;
  int isign;

  hsink = ceil (hx);
  hsink = floor (hx);
  hsink = trunc (hx);
  hsink = rint (hx);
  hsink = round (hx);
  hsink = fabs (hx);
  hsink = fmax (hx, hy);
  hsink = fmin (hx, hy);
  hsink = fma (hx, hy, hx);
  hsink = fdim (hy, hx);
  hsink = maxmag (hx, hy);
  hsink = minmag (hx, hy);
  hsink = fmod (hy, hx);
  hsink = frexp (hx, &iexp);
  hsink = logb (hx);
  isink = ilogb (hx);
  hsink = ldexp (hx, hn);
  hsink = rootn (hy, hn);
  hsink = pown (hx, hn);
  hsink = remainder (hy, hx);
  hsink = modf (hx, &hip);
  hsink = remquo (hy, hx, &quo);
  isink = iexp + quo;
  hsink = powr (hy, hx);
  hsink = sincos (hx, &hcos);
  hsink = lgamma_r (hx, &isign);
  hsink = nextafter (hx, hy);

  half8 hvx = vx;
  half8 hvy = vy;
  int8 hvn = vn;
  half8 vip;
  int8 vquo;
  int8 viexp;
  half8 vcos;
  int8 visign;

  vsink = ceil (hvx);
  vsink = floor (hvx);
  vsink = trunc (hvx);
  vsink = rint (hvx);
  vsink = round (hvx);
  vsink = fabs (hvx);
  vsink = fmax (hvx, hvy);
  vsink = fmin (hvx, hvy);
  vsink = fma (hvx, hvy, hvx);
  vsink = fdim (hvy, hvx);
  vsink = maxmag (hvx, hvy);
  vsink = minmag (hvx, hvy);
  vsink = fmod (hvy, hvx);
  vsink = frexp (hvx, &viexp);
  vsink = logb (hvx);
  visink = ilogb (hvx);
  vsink = ldexp (hvx, hvn);
  vsink = rootn (hvy, hvn);
  vsink = pown (hvx, hvn);
  vsink = remainder (hvy, hvx);
  vsink = modf (hvx, &vip);
  vsink = remquo (hvy, hvx, &vquo);
  visink = vquo;
  vsink = powr (hvy, hvx);
  vsink = sincos (hvx, &vcos);
  vsink = lgamma_r (hvx, &visign);
  vsink = nextafter (hvx, hvy);
}
