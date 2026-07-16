#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_khr_fp16
volatile global half a = INFINITY;
volatile global half b = 1.0h;
volatile global half8 va = (half8)(INFINITY);
volatile global half8 vb = (half8)(1.0h);
#endif

kernel
void test_halfs() {
#ifdef cl_khr_fp16
  if (!isinf(a))
    printf("FAIL at line %d\n", __LINE__ - 1);
  if (isinf(b))
    printf("FAIL at line %d\n", __LINE__ - 1);
  if (!all(isinf(va) == (short8)(-1)))
    printf("FAIL at line %d\n", __LINE__ - 1);
  if (!all(isinf(vb) == (short8)(0)))
    printf("FAIL at line %d\n", __LINE__ - 1);

  // Test conversions involving half
  {
    float f = 2.5f;
    half h = convert_half(f);
    if ((float)h != 2.5f) {
      printf("FAIL: convert_half(float) got %f\n", (float)h);
    }
  }

  {
    float4 f4 = (float4)(1.0f, 2.0f, 3.0f, 4.0f);
    half4 h4 = convert_half4(f4);
    if ((float)h4.x != 1.0f || (float)h4.y != 2.0f || (float)h4.z != 3.0f || (float)h4.w != 4.0f) {
      printf("FAIL: convert_half4(float4) got %f %f %f %f\n", (float)h4.x, (float)h4.y, (float)h4.z, (float)h4.w);
    }
  }

  {
    half h = 1.6h;
    int i = convert_int_rtz(h);
    if (i != 1) {
      printf("FAIL: convert_int_rtz(half) got %d\n", i);
    }

    int i_rte = convert_int_rte(h);
    if (i_rte != 2) {
      printf("FAIL: convert_int_rte(half) got %d\n", i_rte);
    }
  }

  {
    half4 h4 = (half4)(-129.5h, 127.5h, 128.5h, 0.0h);
    char4 c4 = convert_char4_sat_rte(h4);
    // sat char range is [-128, 127]
    // -129.5 -> rounds to even -130 -> clamps to -128
    // 127.5 -> rounds to even 128 -> clamps to 127
    // 128.5 -> rounds to even 128 -> clamps to 127
    if (c4.x != -128 || c4.y != 127 || c4.z != 127 || c4.w != 0) {
      printf("FAIL: convert_char4_sat_rte(half4) got %d %d %d %d\n", c4.x, c4.y, c4.z, c4.w);
    }
  }
#endif
}

