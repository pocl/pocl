#ifdef cl_khr_fp16
__attribute__((__overloadable__))
half distance(half p0, half p1)
{
  return length(p0-p1);
}

__attribute__((__overloadable__))
half distance(half2 p0, half2 p1)
{
  return length(p0-p1);
}

__attribute__((__overloadable__))
half distance(half3 p0, half3 p1)
{
  return length(p0-p1);
}

__attribute__((__overloadable__))
half distance(half4 p0, half4 p1)
{
  return length(p0-p1);
}
#endif

__attribute__((__overloadable__))
float distance(float p0, float p1)
{
  return length(p0-p1);
}

__attribute__((__overloadable__))
float distance(float2 p0, float2 p1)
{
  return length(p0-p1);
}

__attribute__((__overloadable__))
float distance(float3 p0, float3 p1)
{
  return length(p0-p1);
}

__attribute__((__overloadable__))
float distance(float4 p0, float4 p1)
{
  return length(p0-p1);
}

#ifdef cl_khr_fp64
__attribute__((__overloadable__))
double distance(double p0, double p1)
{
  return length(p0-p1);
}

__attribute__((__overloadable__))
double distance(double2 p0, double2 p1)
{
  return length(p0-p1);
}

__attribute__((__overloadable__))
double distance(double3 p0, double3 p1)
{
  return length(p0-p1);
}

__attribute__((__overloadable__))
double distance(double4 p0, double4 p1)
{
  return length(p0-p1);
}
#endif
