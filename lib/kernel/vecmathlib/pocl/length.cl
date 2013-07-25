__attribute__((__overloadable__))
float length(float p)
{
  return sqrt(dot(p, p));
}

__attribute__((__overloadable__))
float length(float2 p)
{
  return sqrt(dot(p, p));
}

__attribute__((__overloadable__))
float length(float3 p)
{
  return sqrt(dot(p, p));
}

__attribute__((__overloadable__))
float length(float4 p)
{
  return sqrt(dot(p, p));
}

#ifdef cl_khr_fp64
__attribute__((__overloadable__))
double length(double p)
{
  return sqrt(dot(p, p));
}

__attribute__((__overloadable__))
double length(double2 p)
{
  return sqrt(dot(p, p));
}

__attribute__((__overloadable__))
double length(double3 p)
{
  return sqrt(dot(p, p));
}

__attribute__((__overloadable__))
double length(double4 p)
{
  return sqrt(dot(p, p));
}
#endif
