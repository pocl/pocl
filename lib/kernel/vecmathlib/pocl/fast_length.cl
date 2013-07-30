// Note: Chapter 6.12.5 of the OpenCL standard says to use half_sqrt,
// not fast_sqrt

__attribute__((__overloadable__))
float fast_length(float p)
{
  return half_sqrt(dot(p, p));
}

__attribute__((__overloadable__))
float fast_length(float2 p)
{
  return half_sqrt(dot(p, p));
}

__attribute__((__overloadable__))
float fast_length(float3 p)
{
  return half_sqrt(dot(p, p));
}

__attribute__((__overloadable__))
float fast_length(float4 p)
{
  return half_sqrt(dot(p, p));
}
