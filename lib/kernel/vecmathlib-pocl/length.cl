#ifdef cl_khr_fp16
__attribute__((__overloadable__))
half length(half p)
{
  return fabs(p);
}

__attribute__((__overloadable__))
half length(half2 p)
{
  p = fabs(p);
  half maxp = max(p.s0, p.s1);
  if (maxp == (half)0.0f) return (half)0.0f;
  if (isinf(maxp)) return maxp;
  p /= maxp;
  return sqrt(dot(p, p)) * maxp;
}

__attribute__((__overloadable__))
half length(half3 p)
{
  p = fabs(p);
  half maxp = max(max(p.s0, p.s1), p.s2);
  if (maxp == (half)0.0f) return (half)0.0f;
  if (isinf(maxp)) return maxp;
  p /= maxp;
  return sqrt(dot(p, p)) * maxp;
}

__attribute__((__overloadable__))
half length(half4 p)
{
  p = fabs(p);
  half maxp = max(max(p.s0, p.s1), max(p.s2, p.s3));
  if (maxp == (half)0.0f) return (half)0.0f;
  if (isinf(maxp)) return maxp;
  p /= maxp;
  return sqrt(dot(p, p)) * maxp;
}
#endif

__attribute__((__overloadable__))
float length(float p)
{
  return fabs(p);
}

__attribute__((__overloadable__))
float length(float2 p)
{
  p = fabs(p);
  float maxp = max(p.s0, p.s1);
  if (maxp == 0.0f) return 0.0f;
  if (isinf(maxp)) return maxp;
  p /= maxp;
  return sqrt(dot(p, p)) * maxp;
}

__attribute__((__overloadable__))
float length(float3 p)
{
  p = fabs(p);
  float maxp = max(max(p.s0, p.s1), p.s2);
  if (maxp == 0.0f) return 0.0f;
  if (isinf(maxp)) return maxp;
  p /= maxp;
  return sqrt(dot(p, p)) * maxp;
}

__attribute__((__overloadable__))
float length(float4 p)
{
  p = fabs(p);
  float maxp = max(max(p.s0, p.s1), max(p.s2, p.s3));
  if (maxp == 0.0f) return 0.0f;
  if (isinf(maxp)) return maxp;
  p /= maxp;
  return sqrt(dot(p, p)) * maxp;
}

#ifdef cl_khr_fp64
__attribute__((__overloadable__))
double length(double p)
{
  return fabs(p);
}

__attribute__((__overloadable__))
double length(double2 p)
{
  p = fabs(p);
  double maxp = max(p.s0, p.s1);
  if (maxp == 0.0) return 0.0;
  if (isinf(maxp)) return maxp;
  p /= maxp;
  return sqrt(dot(p, p)) * maxp;
}

__attribute__((__overloadable__))
double length(double3 p)
{
  p = fabs(p);
  double maxp = max(max(p.s0, p.s1), p.s2);
  if (maxp == 0.0) return 0.0;
  if (isinf(maxp)) return maxp;
  p /= maxp;
  return sqrt(dot(p, p)) * maxp;
}

__attribute__((__overloadable__))
double length(double4 p)
{
  p = fabs(p);
  double maxp = max(max(p.s0, p.s1), max(p.s2, p.s3));
  if (maxp == 0.0) return 0.0;
  if (isinf(maxp)) return maxp;
  p /= maxp;
  return sqrt(dot(p, p)) * maxp;
}
#endif
