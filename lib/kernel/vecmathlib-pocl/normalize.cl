__attribute__((__overloadable__))
float normalize(float p)
{
  return p * rsqrt(dot(p, p));
}

__attribute__((__overloadable__))
float2 normalize(float2 p)
{
  return p * rsqrt(dot(p, p));
}

__attribute__((__overloadable__))
float3 normalize(float3 p)
{
  return p * rsqrt(dot(p, p));
}

__attribute__((__overloadable__))
float4 normalize(float4 p)
{
  return p * rsqrt(dot(p, p));
}

#ifdef cl_khr_fp64
__attribute__((__overloadable__))
double normalize(double p)
{
  return p * rsqrt(dot(p, p));
}

__attribute__((__overloadable__))
double2 normalize(double2 p)
{
  return p * rsqrt(dot(p, p));
}

__attribute__((__overloadable__))
double3 normalize(double3 p)
{
  return p * rsqrt(dot(p, p));
}

__attribute__((__overloadable__))
double4 normalize(double4 p)
{
  return p * rsqrt(dot(p, p));
}
#endif
