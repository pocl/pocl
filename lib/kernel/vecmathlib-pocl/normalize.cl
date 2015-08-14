#ifdef cl_khr_fp16
__attribute__((__overloadable__))
half normalize(half p)
{
  if (p == (half)0.0f) return p;
  return copysign((half)1.0f, p);
}

__attribute__((__overloadable__))
half2 normalize(half2 p)
{
  if (all(p == (half2)(half)0.0f)) return p;
  half maxp = max(fabs(p.s0), fabs(p.s1));
  p /= maxp;
  return p * rsqrt(dot(p, p));
}

__attribute__((__overloadable__))
half3 normalize(half3 p)
{
  if (all(p == (half3)(half)0.0f)) return p;
  half maxp = max(max(fabs(p.s0), fabs(p.s1)), fabs(p.s2));
  p /= maxp;
  return p * rsqrt(dot(p, p));
}

__attribute__((__overloadable__))
half4 normalize(half4 p)
{
  if (all(p == (half4)(half)0.0f)) return p;
  half maxp = max(max(fabs(p.s0), fabs(p.s1)), max(fabs(p.s2), fabs(p.s3)));
  p /= maxp;
  return p * rsqrt(dot(p, p));
}
#endif

__attribute__((__overloadable__))
float normalize(float p)
{
  if (p == 0.0f) return p;
  return copysign(1.0f, p);
}

__attribute__((__overloadable__))
float2 normalize(float2 p)
{
  if (all(p == (float2)0.0f)) return p;
  float maxp = max(fabs(p.s0), fabs(p.s1));
  p /= maxp;
  return p * rsqrt(dot(p, p));
}

__attribute__((__overloadable__))
float3 normalize(float3 p)
{
  if (all(p == (float3)0.0f)) return p;
  float maxp = max(max(fabs(p.s0), fabs(p.s1)), fabs(p.s2));
  p /= maxp;
  return p * rsqrt(dot(p, p));
}

__attribute__((__overloadable__))
float4 normalize(float4 p)
{
  if (all(p == (float4)0.0f)) return p;
  float maxp = max(max(fabs(p.s0), fabs(p.s1)), max(fabs(p.s2), fabs(p.s3)));
  p /= maxp;
  return p * rsqrt(dot(p, p));
}

#ifdef cl_khr_fp64
__attribute__((__overloadable__))
double normalize(double p)
{
  if (p == 0.0) return p;
  return copysign(1.0, p);
}

__attribute__((__overloadable__))
double2 normalize(double2 p)
{
  if (all(p == (double2)0.0)) return p;
  double maxp = max(fabs(p.s0), fabs(p.s1));
  p /= maxp;
  return p * rsqrt(dot(p, p));
}

__attribute__((__overloadable__))
double3 normalize(double3 p)
{
  if (all(p == (double3)0.0)) return p;
  double maxp = max(max(fabs(p.s0), fabs(p.s1)), fabs(p.s2));
  p /= maxp;
  return p * rsqrt(dot(p, p));
}

__attribute__((__overloadable__))
double4 normalize(double4 p)
{
  if (all(p == (double4)0.0)) return p;
  double maxp = max(max(fabs(p.s0), fabs(p.s1)), max(fabs(p.s2), fabs(p.s3)));
  p /= maxp;
  return p * rsqrt(dot(p, p));
}
#endif
