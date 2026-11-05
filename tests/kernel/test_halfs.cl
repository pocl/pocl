#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_khr_fp16
volatile global half a = INFINITY;
volatile global half b = 1.0h;
volatile global half8 va = (half8)(INFINITY);
volatile global half8 vb = (half8)(1.0h);
/* runtime operands for the half <-> float/double conversions, which lower to
   the __truncsfhf2 & co. runtime routines on CPUs without F16C */
volatile global float cf = -2.75f;
volatile global ushort ch = 0xc180; /* -2.75h */
#endif
#if defined(cl_khr_fp16) && defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
volatile global double cd = 0.1;
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

  // Conversions of runtime values; compare bit patterns where possible, so a
  // broken extension cannot mask a broken truncation
  {
    ushort bits = as_ushort((half)cf);
    if (bits != 0xc180)
      printf("FAIL: (half)%f got 0x%04x, expected 0xc180\n", cf, bits);
    float f = (float)as_half(ch);
    if (f != -2.75f)
      printf("FAIL: (float)as_half(0x%04x) got %f\n", ch, f);
  }
#ifdef cl_khr_fp64
  {
    // 0.1 is inexact in half: rounds to 0x2e66
    ushort bits = as_ushort((half)cd);
    if (bits != 0x2e66)
      printf("FAIL: (half)%f got 0x%04x, expected 0x2e66\n", cd, bits);
    double d = (double)as_half(ch);
    if (d != -2.75)
      printf("FAIL: (double)as_half(0x%04x) got %f\n", ch, d);
  }
#endif
#endif
}

