kernel 
void test_length_distance() {
  volatile float a = 18446744073709551616.0f; /* 2^64 */
  volatile float e = 1.0f;
  volatile float l = length(a);
  if (l != 18446744073709551616.0f) {
    printf("length(float) failed.\n");
  }
  volatile float d = distance(a, e);
  if (d != 18446744073709551616.0f) {
    printf("distance(float,float) failed.\n");
  }
  volatile float n = normalize(a);
  if (n != 1.0f) {
    printf("normalize(float) failed.\n");
  }

  volatile float2 a2 = (float2)(18446744073709551616.0f,
                                18446744073709551616.0f); /* 2^64 */
  volatile float2 e2 = (float2)(1.0f, 1.0f);
  volatile float l2 = length(a2);
  if (l2 != 2.6087635e19f) {
    printf("length(float2) failed.\n");
  }
  volatile float d2 = distance(a2, e2);
  if (d2 != 2.6087635e19f) {
    printf("distance(float2,float2) failed.\n");
  }
  volatile float2 n2 = normalize(a2);
  if (any(n2 != (float2)(0.70710677f, 0.70710677f))) {
    printf("normalize(float2) failed.\n");
  }

  volatile float3 a3 = (float3)(18446744073709551616.0f,
                                18446744073709551616.0f,
                                18446744073709551616.0f); /* 2^64 */
  volatile float3 e3 = (float3)(1.0f, 1.0f, 1.0f);
  float l3 = length(a3);
  if (l3 != 3.1950697e19f) {
    printf("length(float3) failed.\n");
  }
  float d3 = distance(a3, e3);
  if (d3 != 3.1950697e19f) {
    printf("distance(float3,float3) failed.\n");
  }
  float3 n3 = normalize(a3);
  if (any(n3 != (float3)(0.57735026f, 0.57735026f, 0.57735026f))) {
    printf("normalize(float3) failed.\n");
  }

  volatile float4 a4 = (float4)(18446744073709551616.0f,
                                18446744073709551616.0f,
                                18446744073709551616.0f,
                                18446744073709551616.0f); /* 2^64 */
  volatile float4 e4 = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
  float l4 = length(a4);
  if (l4 != 3.689349e19f) {
    printf("length(float4) failed.\n");
  }
  float d4 = distance(a4, e4);
  if (d4 != 3.689349e19f) {
    printf("distance(float4,float4) failed.\n");
  }
  float4 n4 = normalize(a4);
  if (any(n4 != (float4)(0.5f, 0.5f, 0.5f, 0.5f))) {
    printf("normalize(float4) failed.\n");
  }
}
