#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// Helper macros to check float/half values
#define CHECK_VAL(typename, actual, expected, tol) \
  do { \
    if (fabs((float)(actual) - (float)(expected)) > (float)(tol)) { \
      printf("FAIL: dot_%s actual %f expected %f\n", #typename, (float)(actual), (float)(expected)); \
      return; \
    } \
  } while (0)

kernel void test_dot_fp32()
{
  // Test float (scalar)
  {
    float a = 1.5f;
    float b = 2.0f;
    float res = dot(a, b);
    CHECK_VAL(float, res, 3.0f, 1e-5f);
  }

  // Test float2
  {
    float2 a = (float2)(1.0f, 2.0f);
    float2 b = (float2)(3.0f, 4.0f);
    float res = dot(a, b);
    CHECK_VAL(float2, res, 11.0f, 1e-5f);
  }

  // Test float3
  {
    float3 a = (float3)(1.0f, 2.0f, 3.0f);
    float3 b = (float3)(4.0f, 5.0f, 6.0f);
    float res = dot(a, b);
    CHECK_VAL(float3, res, 32.0f, 1e-5f);
  }

  // Test float4
  {
    float4 a = (float4)(1.0f, 2.0f, 3.0f, 4.0f);
    float4 b = (float4)(5.0f, 6.0f, 7.0f, 8.0f);
    float res = dot(a, b);
    CHECK_VAL(float4, res, 70.0f, 1e-5f);
  }

  // Test float8
  {
    float8 a = (float8)(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    float8 b = (float8)(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
    float res = dot(a, b);
    CHECK_VAL(float8, res, 36.0f, 1e-5f);
  }

  // Test float16
  {
    float16 a = (float16)(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
    float16 b = (float16)(2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                          2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f);
    float res = dot(a, b);
    CHECK_VAL(float16, res, 32.0f, 1e-5f);
  }
}

#ifdef cl_khr_fp16
kernel void test_dot_fp16()
{
  // Test half (scalar)
  {
    half a = (half)1.5f;
    half b = (half)2.0f;
    half res = dot(a, b);
    CHECK_VAL(half, res, 3.0f, 1e-3f);
  }

  // Test half2
  {
    half2 a = (half2)(1.0f, 2.0f);
    half2 b = (half2)(3.0f, 4.0f);
    half res = dot(a, b);
    CHECK_VAL(half2, res, 11.0f, 1e-3f);
  }

  // Test half3
  {
    half3 a = (half3)(1.0f, 2.0f, 3.0f);
    half3 b = (half3)(4.0f, 5.0f, 6.0f);
    half res = dot(a, b);
    CHECK_VAL(half3, res, 32.0f, 1e-3f);
  }

  // Test half4
  {
    half4 a = (half4)(1.0f, 2.0f, 3.0f, 4.0f);
    half4 b = (half4)(5.0f, 6.0f, 7.0f, 8.0f);
    half res = dot(a, b);
    CHECK_VAL(half4, res, 70.0f, 1e-3f);
  }

  // Test half8
  {
    half8 a = (half8)(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    half8 b = (half8)(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
    half res = dot(a, b);
    CHECK_VAL(half8, res, 36.0f, 1e-3f);
  }

  // Test half16
  {
    half16 a = (half16)(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
    half16 b = (half16)(2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f);
    half res = dot(a, b);
    CHECK_VAL(half16, res, 32.0f, 1e-3f);
  }
}
#endif
