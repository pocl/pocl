// Note: Chapter 6.12.5 of the OpenCL standard says to use half_rsqrt,
// not fast_rsqrt

__attribute__((__overloadable__))
float fast_normalize(float p)
{
  return p * half_rsqrt(dot(p, p));
}

__attribute__((__overloadable__))
float2 fast_normalize(float2 p)
{
  return p * half_rsqrt(dot(p, p));
}

__attribute__((__overloadable__))
float3 fast_normalize(float3 p)
{
  return p * half_rsqrt(dot(p, p));
}

__attribute__((__overloadable__))
float4 fast_normalize(float4 p)
{
  return p * half_rsqrt(dot(p, p));
}
